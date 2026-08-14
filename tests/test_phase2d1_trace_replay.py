from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from eval.controlled_intervention import validate_intervention
from eval.inference_trace import TraceConfig, raw_seg_positions, tensor_sha256, trace_full_forward
from eval.phase2d_scope_validation import assert_exact_identity
from eval.representation_metrics import recovery

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2d1_trace_replay"


class _PromptEncoder(nn.Module):
    def forward(self, *, points, boxes, masks, text_embeds):
        return text_embeds, torch.zeros((1, 1, 2, 2), dtype=text_embeds.dtype)

    def get_dense_pe(self):
        return torch.zeros((1, 1, 2, 2))


class _MaskDecoder(nn.Module):
    def forward(self, *, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output):
        value = sparse_prompt_embeddings.sum().reshape(1, 1, 1, 1)
        return value.expand(1, 1, 2, 2).clone(), None


class _Grounding(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = _PromptEncoder()
        self.mask_decoder = _MaskDecoder()

    @staticmethod
    def postprocess_masks(value, *, input_size, original_size):
        return value


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.seg_token_idx = 9
        self.model = SimpleNamespace(
            text_hidden_fcs=nn.ModuleList([nn.Sequential(nn.Identity())]),
            grounding_encoder=_Grounding(),
        )

    @staticmethod
    def _get_last_hidden_state(value):
        return value

    def _inference_path(self, input_ids, global_images, attention, offset, bboxes):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=12).float()
        return SimpleNamespace(logits=hidden), hidden

    @staticmethod
    def get_grounding_encoder_embs(images):
        return images


def _batch():
    return {"global_enc_images": torch.ones(1, 1, 2, 2),
            "grounding_enc_images": torch.ones(1, 1, 2, 2), "offset": torch.tensor([0, 1]),
            "bboxes": [None], "resize_list": [(2, 2)]}


def test_trace_disabled_is_explicit_default():
    assert TraceConfig().enabled is False


def test_trace_is_read_only_and_bitwise_invariant():
    model = _FakeModel().eval()
    ids = torch.tensor([[1, 2, 9]])
    parameter_hash_before = tensor_sha256(next(model.model.grounding_encoder.parameters(), torch.zeros(1)))
    first = trace_full_forward(model, _batch(), ids, original_size=(2, 2))
    second = trace_full_forward(model, _batch(), ids, original_size=(2, 2))
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert first["seg_token_position_raw"] == second["seg_token_position_raw"] == 2
    assert first["predictor_position_raw"] == second["predictor_position_raw"] == 1
    assert torch.equal(first["postprocessed_mask_logits"], second["postprocessed_mask_logits"])
    assert torch.equal(first["binary_mask"], second["binary_mask"])
    assert tensor_sha256(next(model.model.grounding_encoder.parameters(), torch.zeros(1))) == parameter_hash_before


def test_replay_sequence_requires_exactly_one_seg():
    assert raw_seg_positions(torch.tensor([[1, 2, 9]]), 9) == (2, 1)
    with pytest.raises(ValueError, match="exactly one"):
        raw_seg_positions(torch.tensor([[1, 2, 3]]), 9)


def test_hidden_swap_integrity_allows_only_same_image_scope():
    tensor = torch.ones(1)
    trace = {"image_embedding": tensor}
    valid = validate_intervention(trace, trace, base_sample_id="official:a", donor_sample_id="official:a",
                                  intervention="H2a_predictor_hidden_swap", checkpoint_sha256="x",
                                  donor_checkpoint_sha256="x", threshold=0.0)
    assert valid["status"] == "VALID"
    invalid = validate_intervention(trace, trace, base_sample_id="official:a", donor_sample_id="internal:a",
                                    intervention="H2a_predictor_hidden_swap", checkpoint_sha256="x",
                                    donor_checkpoint_sha256="x", threshold=0.0)
    assert invalid["status"] == "INTERVENTION_SCOPE_VIOLATION"
    assert "sample_identity" in invalid["violations"]


def test_projection_swap_rejects_threshold_change():
    trace = {"image_embedding": torch.ones(1)}
    result = validate_intervention(trace, trace, base_sample_id="a", donor_sample_id="a",
                                   intervention="H2b_projection_swap", checkpoint_sha256="x",
                                   donor_checkpoint_sha256="x", threshold=.5)
    assert result["status"] == "INTERVENTION_SCOPE_VIOLATION"
    assert "mask_threshold" in result["violations"]


def test_scope_rejects_internal_official_pairing():
    with pytest.raises(ValueError, match="identity mismatch"):
        assert_exact_identity([{"sample_id": "official:a"}], [{"sample_id": "internal:a"}],
                              reference_name="official1000", candidate_name="internal1104")


def test_recovery_ratio_is_guarded_for_negative_or_tiny_gap():
    assert recovery(.5, .2, .8)["fraction_of_TF_gap_recovered"] == pytest.approx(.5)
    assert recovery(.5, .6, .4)["fraction_of_TF_gap_recovered"] is None
    assert recovery(.5, .5, .505)["fraction_of_TF_gap_recovered"] is None


def test_real_trace_artifact_schema_and_scope():
    rows = [json.loads(line) for line in (OUT / "controlled_modes/per_sample_results.jsonl").read_text().splitlines()]
    assert len(rows) == 64
    assert len({row["sample_id"] for row in rows}) == 64
    assert sum(row["status"] == "COMPLETE" for row in rows) == 60
    assert all(row["historical_G0_batch_size"] == 8 for row in rows)
    assert all(row["historical_iou_tolerance"] == 5e-4 for row in rows)
    assert all(not row.get("interventions_executed", False) for row in rows if row["status"] != "COMPLETE")


def test_real_mode_specs_declare_every_intervention():
    specs = json.loads((OUT / "controlled_modes/mode_specs.json").read_text())
    by_mode = {spec["mode"]: spec for spec in specs}
    for mode in ("H2a_predictor_hidden_swap", "H2b_projection_swap", "H2c_sam_prompt_swap"):
        assert by_mode[mode]["status"] == "executed"
        assert by_mode[mode]["changed_variables"]
        assert by_mode[mode]["threshold"] == 0
        assert by_mode[mode]["checkpoint_sha256"] == "07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072"


def test_real_invariance_and_no_weight_mutation_artifacts():
    invariance = [json.loads(line) for line in (OUT / "replay/trace_invariance_samples.jsonl").read_text().splitlines()]
    assert len(invariance) == 3
    assert all(row["status"] == "PASS" for row in invariance)
    mutation = json.loads((OUT / "audit/no_weight_mutation.json").read_text())
    assert mutation["checkpoint_unchanged"]
    assert mutation["runtime_modules_unchanged"]
    assert not mutation["training_started"]
    assert not mutation["model_weights_modified"]


def test_real_intervention_integrity_and_manifest_discipline():
    integrity = json.loads((OUT / "controlled_modes/intervention_integrity.json").read_text())
    assert integrity == {"eligible_n": 60, "all_valid": True,
                         "excluded_before_intervention": {"reproducibility_mismatch": 2, "no_G0_predictor": 2}}
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["trace_invariance"] == "PASS"
    assert manifest["discipline"] == {
        "training_started": False, "model_weights_modified": False,
        "checkpoint_selection_performed": False, "test_set_model_selection_performed": False,
        "threshold_selection_performed": False, "proxy_category_labels_created": False,
    }
