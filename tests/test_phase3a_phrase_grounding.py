import json
from pathlib import Path

import pytest
import yaml

from dataset.forensics.unified import (
    PHRASE_FIELD_PREFIX,
    TARGET_PROTOCOL_HISTORICAL,
    TARGET_PROTOCOL_PHRASE_ALIGNED,
    UnifiedForensicsDataset,
)
from eval.phase3a_metrics import parse_phrase_aligned_generation, validate_prediction_record
from scripts.phase3a_evaluate import summarize_localization


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a_phrase_grounding"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def test_phrase_audit_is_authoritative_and_complete():
    audit = load_json(OUT / "audit/phrase_annotation_audit.json")
    assert audit["source"] == "frozen_manifest_refs.phrase"
    assert audit["authoritative_only"] is True
    assert audit["proxy_phrase_labels_created"] is False
    for split in ("train", "val", "test"):
        item = audit["splits"][split]
        assert item["fake_with_authoritative_phrase"] == item["fake_images"]
        assert item["empty_phrase_count"] == 0
        assert item["refs_without_polygon_target"] == 0
        assert item["real_with_phrase"] == 0


def test_phrase_construction_preserves_order_and_only_exact_deduplicates():
    row = {"sample_id": "x", "forensics_domain": "fake", "refs": [
        {"phrase": " left hand  "}, {"phrase": "right hand"}, {"phrase": "left hand"},
    ]}
    value = UnifiedForensicsDataset.authoritative_localization_field(row)
    assert value["raw_phrases"] == ["left hand", "right hand", "left hand"]
    assert value["normalized_training_phrase"] == "left hand; right hand"


def test_c0_and_p1_change_only_fake_language_target():
    fake = {"sample_id": "f", "forensics_domain": "fake", "explanation": "Full evidence.",
            "refs": [{"phrase": "left hand"}]}
    real = {"sample_id": "r", "forensics_domain": "real"}
    c0 = UnifiedForensicsDataset._target(fake, TARGET_PROTOCOL_HISTORICAL)
    p1 = UnifiedForensicsDataset._target(fake, TARGET_PROTOCOL_PHRASE_ALIGNED)
    assert c0 == "[FAKE] Full evidence. [SEG]"
    assert p1 == f"[FAKE] Full evidence.\n{PHRASE_FIELD_PREFIX} left hand [SEG]"
    assert UnifiedForensicsDataset._target(real, TARGET_PROTOCOL_HISTORICAL) == \
           UnifiedForensicsDataset._target(real, TARGET_PROTOCOL_PHRASE_ALIGNED)


def test_template_examples_preserve_mask_and_seg_protocol():
    rows = load_jsonl(OUT / "audit/template_examples.jsonl")
    assert len(rows) == 80
    for row in rows:
        assert row["mask_target_identical"] is True
        assert row["mask_target_sha256_c0"] == row["mask_target_sha256_p1"]
        if row["domain"] == "fake":
            assert len(row["c0"]["seg_positions"]) == 1
            assert len(row["p1"]["seg_positions"]) == 1
            assert row["p1"]["localization_field_token_span"][1] <= row["p1"]["seg_positions"][0]
            assert row["p1"]["localization_field_included_in_lm_loss"] is True
            assert row["original_explanation"] in row["p1"]["target"]
        else:
            assert row["c0"]["target"] == row["p1"]["target"]
            assert not row["p1"]["seg_positions"]


def test_training_protocol_gate_passes():
    audit = load_json(OUT / "audit/training_protocol_audit.json")
    assert audit["status"] == "PASS"
    assert audit["p1_fake_phrase_before_seg"] is True
    assert audit["p1_fake_phrase_in_lm_loss"] is True
    assert audit["mask_target_identity"] is True
    assert audit["official_test_used"] is False


def test_config_diff_contains_only_preregistered_and_admin_fields():
    diff = load_json(OUT / "configs/config_diff.json")
    assert diff["status"] == "PASS"
    assert diff["unexpected_differences"] == []
    primary = {item["path"] for item in diff["primary_training_differences"]}
    assert primary == {
        "forensics.target_protocol", "forensics.target_template",
        "forensics.localization_phrase_insertion", "forensics.localization_phrase_lm_target",
    }


def test_c0_matches_phase2a_training_recipe():
    c0 = yaml.safe_load((ROOT / "configs/phase3a_c0.yaml").read_text())
    old = yaml.safe_load((ROOT / "configs/phase2a_unified_baseline_full.yaml").read_text())
    for section in ("architecture", "model", "data", "trainable", "optimizer", "training",
                    "loss", "text_loss", "mask_loss", "classification_loss", "features", "validation"):
        old_section = old[section]
        c0_section = c0[section]
        for key, value in old_section.items():
            assert c0_section[key] == value, (section, key)
    assert c0["forensics"]["target_protocol"] == "historical"
    assert c0["checkpoint"]["primary_best_metric"] == old["checkpoint"]["primary_best_metric"]


def test_scope_threshold_and_no_forensic_fusion():
    for name in ("c0", "p1"):
        config = yaml.safe_load((ROOT / f"configs/phase3a_{name}.yaml").read_text())
        assert config["data"]["hyperparameter_selection_split"] == "val"
        assert "external_synthscars_test" in config["data"]["prohibited_selection_splits"]
        assert config["evaluation"]["mask_logit_threshold"] == 0.0
        assert config["features"]["npr"] is False
        assert config["features"]["srm"] is False
        assert config["features"]["forensic_fusion"] == "none"


def test_deterministic_p1_parser_does_not_guess_missing_target():
    parsed = parse_phrase_aligned_generation(
        "[FAKE] Evidence. Target regions: left hand; face [SEG]"
    )
    assert parsed["parse_success"] is True
    assert parsed["target_region"] == "left hand; face"
    missing = parse_phrase_aligned_generation("[FAKE] Evidence only [SEG]")
    assert missing["parse_success"] is False
    assert missing["target_region"] is None
    assert "MISSING_TARGET_FIELD" in missing["parse_status"]


def test_prediction_artifact_contract_requires_raw_spatial_outputs():
    record = {key: 0 for key in (
        "sample_id", "generated_token_ids", "decoded_text", "generated_localization_phrase",
        "seg_position", "mask_logits_path", "binary_mask_path", "tp", "fp", "fn", "tn",
        "foreground_iou", "foreground_f1", "fg_bg_miou", "mask_logit_threshold",
    )}
    validate_prediction_record(record)
    record.pop("mask_logits_path")
    with pytest.raises(ValueError):
        validate_prediction_record(record)


def test_manifest_experiment_discipline():
    manifest = load_json(OUT / "manifest.json")
    assert isinstance(manifest["training_started"], bool)
    assert manifest["official_test_used_for_training"] is False
    assert manifest["official_test_used_for_checkpoint_selection"] is False
    assert manifest["threshold_sweep_performed"] is False
    assert manifest["new_segmentation_labels_created"] is False
    assert manifest["multi_seg_training_used"] is False


def test_phase3a_fg_bg_aggregation_is_explicit():
    records = [{
        "tp": 2, "fp": 1, "fn": 1, "tn": 6,
        "foreground_iou": 0.5, "foreground_f1": 2 / 3,
        "background_iou": 0.75, "fg_bg_miou": 0.625,
        "seg_triggered": True, "has_pred_mask": True,
        "phrase_parse": {"parse_success": True},
    }]
    metrics = summarize_localization(records, autoregressive=True)
    assert metrics["per_image_mean"]["foreground_iou"] == pytest.approx(0.5)
    assert metrics["global_pixel"]["background_iou"] == pytest.approx(0.75)
    assert metrics["global_pixel"]["fg_bg_miou"] == pytest.approx(0.625)
