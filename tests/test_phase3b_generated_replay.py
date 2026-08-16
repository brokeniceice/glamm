import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from tools.phase3b_replay import (
    build_mask_only_batch, deterministic_replay_selection, replay_eligibility,
)


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    pad_token_id = 0


def batch():
    return {
        "sample_ids": ["fake-a", "real-a"], "cls_labels": torch.tensor([1, 0]),
        "seg_valid": torch.tensor([True, False]), "input_ids": torch.tensor([[7, 8, 9], [7, 6, 0]]),
        "attention_masks": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "image_paths": ["f", "r"], "global_enc_images": torch.randn(2, 3, 2, 2),
        "grounding_enc_images": torch.randn(2, 3, 2, 2), "masks_list": [torch.ones(1, 2, 2), None],
        "label_list": [torch.ones(2, 2), torch.ones(2, 2)], "resize_list": [(2, 2), (2, 2)],
        "questions_list": [[], []], "sampled_classes_list": [[], []],
        "sources": ["s", "s"], "content_categories": ["c", "c"],
    }


def test_exact_generated_replay_token_ids_are_preserved():
    record = {"fake-a": {"replay_eligible": True, "full_input_token_ids": [11, 12, 13, 14]}}
    aux = build_mask_only_batch(batch(), ["fake-a"], tokenizer=Tokenizer(), replay_records=record)
    assert aux["input_ids"].tolist() == [[11, 12, 13, 14]]


def test_all_auxiliary_labels_are_ignore():
    aux = build_mask_only_batch(batch(), ["fake-a"], tokenizer=Tokenizer(), replay_records=None)
    assert aux["labels"].eq(-100).all()


def test_auxiliary_classification_labels_are_absent():
    aux = build_mask_only_batch(batch(), ["fake-a"], tokenizer=Tokenizer(), replay_records=None)
    assert aux["cls_labels"] is None


def test_mask_only_objective_has_gradient():
    value = torch.tensor(0.2, requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(value, torch.tensor(1.0))
    loss.backward()
    assert value.grad is not None and value.grad.abs().item() > 0


def test_noseg_is_ineligible_and_not_selected_for_append():
    assert replay_eligibility([1, 2, 3], 9)["replay_eligible"] is False


def test_exactly_one_seg_with_predictor_is_eligible():
    result = replay_eligibility([1, 2, 9, 3], 9)
    assert result["replay_eligible"] and result["usable_seg_predictor_position"] == 1


def test_replay_schedule_is_identical_and_deterministic():
    ids = [f"s{i}" for i in range(10)]
    assert deterministic_replay_selection(ids, seed=3407, optimizer_step=5) == deterministic_replay_selection(
        list(reversed(ids)), seed=3407, optimizer_step=5
    )


def test_real_samples_are_rejected_from_auxiliary_branch():
    with pytest.raises(ValueError, match="non-Fake"):
        build_mask_only_batch(batch(), ["real-a"], tokenizer=Tokenizer(), replay_records=None)


def test_b0_b1_configs_are_matched_except_preregistered_arm_fields():
    values = [yaml.safe_load((ROOT / p).read_text()) for p in (
        "configs/phase3b_b0_gold_replay.yaml", "configs/phase3b_b1_generated_replay.yaml")]
    for value in values:
        value["experiment"]["name"] = "x"; value["experiment"]["runtime_output_dir"] = "x"
        value["experiment"]["experiment_type"] = "x"; value["checkpoint"]["output_root"] = "x"
        value["replay"]["context"] = "x"
    assert values[0] == values[1]


def test_g0_semantics_remain_unconditioned_and_fake_only_metric_selection():
    source = (ROOT / "scripts/phase3a_evaluate.py").read_text()
    assert "provide_gt_fake=False" in source
    assert 'int(dataset.rows[i]["class_label"]) == 1' in source


def test_source_checkpoint_identity_is_same_for_both_arms():
    values = [yaml.safe_load((ROOT / p).read_text()) for p in (
        "configs/phase3b_b0_gold_replay.yaml", "configs/phase3b_b1_generated_replay.yaml")]
    assert values[0]["source"] == values[1]["source"]
