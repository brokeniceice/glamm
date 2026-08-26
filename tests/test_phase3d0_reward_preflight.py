import json
from pathlib import Path

import pytest

from tools.phase3d0 import (
    deterministic_stratified_subset, group_diversity, pareto_dominates,
    pareto_statistics, parse_structure, reward_candidates, reward_components,
)


def parsed(verdict="FAKE", phrase="eyes", target=True):
    return {
        "verdict": verdict, "target_region": phrase,
        "target_field_present": target, "parse_status": ["PASS"],
    }


def test_deterministic_stratified_subset_exact_and_reproducible():
    rows = []
    for index in range(40):
        rows.append({
            "sample_id": f"s{index}", "class_label": index % 2,
            "content_category": ("human", "object")[index % 2],
            "source": f"source{index % 4}",
        })
    a = deterministic_stratified_subset(rows, count=12, seed=3407, label=1)
    b = deterministic_stratified_subset(rows, count=12, seed=3407, label=1)
    assert [row["sample_id"] for row in a] == [row["sample_id"] for row in b]
    assert len(a) == 12 and all(row["class_label"] == 1 for row in a)


def test_structure_uses_emitted_verdict_not_gt():
    fake = parse_structure(parsed("FAKE", "eyes"), [10, 99], 99)
    real = parse_structure(parsed("REAL", None, False), [10, 11], 99)
    contradictory = parse_structure(parsed("REAL", "eyes", True), [10, 99], 99)
    assert fake["structural_validity"] and fake["usable_seg"]
    assert real["structural_validity"] and not real["usable_seg"]
    assert not contradictory["structural_validity"]


def test_missing_seg_is_not_repaired_and_zeroes_fake_evidence():
    values = reward_components(
        gt_label=1, parsed=parsed("FAKE", "eyes"), generated_ids=[10, 11],
        seg_token_id=99, authoritative_phrase="eyes", foreground_iou=0.9,
    )
    assert values["seg_status"] == "SEG_UNAVAILABLE"
    assert values["R_phrase_lex"] == 0 and values["R_mask"] == 0


def test_reward_formulas_exact():
    c = {"R_cls": 1.0, "R_struct": 1.0, "R_phrase_lex": .5, "R_mask": .25, "R_align": 1/3}
    r = reward_candidates(1, c)
    assert r["R0"] == .25 and r["R1"] == .5
    assert r["R2"] == pytest.approx(.20 + .10 + .15 + .10)
    assert r["R3"] == pytest.approx(.20 + .10 + .10 + .075 + .20/3)
    real = reward_candidates(0, {**c, "R_cls": 1.0, "R_struct": 0.0})
    assert real == {"R0": 1.0, "R1": 1.0, "R2": .7, "R3": .7}


def test_components_and_rewards_remain_in_unit_interval():
    c = reward_components(
        gt_label=1, parsed=parsed(), generated_ids=[10, 99], seg_token_id=99,
        authoritative_phrase="eyes", foreground_iou=.8,
    )
    assert all(0 <= c[key] <= 1 for key in ("R_cls", "R_struct", "R_phrase_lex", "R_mask", "R_align"))
    assert all(0 <= value <= 1 for value in reward_candidates(1, c).values())


def test_pareto_contract_counts_ties_as_not_correct():
    weak = {"R_cls": 1, "R_struct": 1, "R_phrase_lex": 0, "R_mask": 0,
            "R0": 0, "rollout_index": 1}
    strong = {"R_cls": 1, "R_struct": 1, "R_phrase_lex": 1, "R_mask": 1,
              "R0": 0, "rollout_index": 0}
    assert pareto_dominates(strong, weak)
    stats = pareto_statistics([[strong, weak]], "R0")
    assert stats["dominance_pair_count"] == 1 and stats["tie_rate"] == 1


def test_group_diversity_requires_exact_record_rewards():
    rows = []
    for index, text in enumerate(("a", "b")):
        rows.append({
            "decoded_text": text, "normalized_phrase": text,
            "structural_validity": True, "R_mask": float(index),
            "R_phrase_lex": float(index), "rollout_index": index,
            "R0": float(index), "R1": float(index), "R2": float(index), "R3": float(index),
        })
    stats = group_diversity(rows)
    assert stats["number_unique_normalized_outputs"] == 2
    assert stats["R3_std"] == .5 and stats["mask_iou_range"] == 1


def test_frozen_manifests_exclude_test_populations_if_prepared():
    root = Path("outputs/phase3d0_reward_preflight/manifests")
    if not root.exists(): pytest.skip("Phase 3D.0 manifests not prepared")
    dev = json.loads((root / "reward_dev_manifest.json").read_text())["records"]
    val = json.loads((root / "reward_val_manifest.json").read_text())["records"]
    assert len(dev) == 1024 and len(val) == 2212
    assert not ({row["sample_id"] for row in dev} & {row["sample_id"] for row in val})


def test_preregistered_threshold_and_no_training_contract_in_config():
    text = Path("configs/phase3d0_reward_preflight.yaml").read_text()
    assert "mask_logit_threshold: 0.0" in text
    assert "K: 8" in text
    assert "structural_validity_tolerance_absolute: 0.05" in text
    assert "max_length_hit_rate_max: 0.05" in text
    source = Path("scripts/phase3d0_generate.py").read_text()
    assert "model.requires_grad_(False)" in source
    assert "optimizer" not in source.lower().replace('"optimizer_created"', '')
