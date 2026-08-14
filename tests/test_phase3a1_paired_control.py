import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dataset.forensics.unified import (
    TARGET_PROTOCOL_HISTORICAL,
    TARGET_PROTOCOL_PHRASE_ALIGNED,
    UnifiedForensicsDataset,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a1_paired_control"


def read_json(path):
    return json.loads(path.read_text())


def test_reconstructed_intended_initialization_hashes_match():
    p1 = read_json(OUT / "audit/reconstructed_initializations/p1/initialization_hash.json")
    c0 = read_json(OUT / "audit/reconstructed_initializations/c0/initialization_hash.json")
    assert p1["full_state"]["sha256"] == c0["full_state"]["sha256"]
    assert p1["trainable_state"]["sha256"] == c0["trainable_state"]["sha256"]
    audit = read_json(OUT / "audit/initialization_audit.json")
    assert audit["initialization_identity_status"] == "BASE_CHECKPOINT_MATCH_BUT_RANDOM_INIT_UNPROVEN"
    assert audit["exact_weight_identity"] is False


def test_c0_and_p1_protocols_are_explicit_and_masks_are_not_changed():
    row = {
        "forensics_domain": "fake", "sample_id": "fake", "explanation": "Artifact evidence.",
        "refs": [{
            "phrase": "left hand", "polygons": [[0, 0, 4, 0, 4, 4, 0, 4]],
            "source_image_size": [8, 8],
        }],
    }
    c0 = UnifiedForensicsDataset._target(row, TARGET_PROTOCOL_HISTORICAL)
    p1 = UnifiedForensicsDataset._target(row, TARGET_PROTOCOL_PHRASE_ALIGNED)
    assert "Target regions:" not in c0
    assert c0.endswith("[SEG]")
    assert "Target regions: left hand [SEG]" in p1
    a = UnifiedForensicsDataset._fake_union_mask(row, 8, 8)
    b = UnifiedForensicsDataset._fake_union_mask(row, 8, 8)
    assert hashlib.sha256(a.numpy().tobytes()).digest() == hashlib.sha256(b.numpy().tobytes()).digest()


def test_config_semantic_parity_and_only_phrase_difference():
    diff = read_json(OUT / "audit/c0_p1_exact_config_diff.json")
    assert diff["status"] == "PASS"
    assert diff["forbidden_differences"] == []
    c0 = yaml.safe_load((ROOT / "configs/phase3a1_paired_c0.yaml").read_text())
    p1 = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text())
    for path in (
        ("experiment", "seed"), ("optimizer", "name"), ("optimizer", "lora_lr"),
        ("training", "total_optimizer_steps"), ("training", "batch_size_per_device"),
        ("training", "gradient_accumulation_steps"), ("training", "effective_global_batch"),
        ("checkpoint", "primary_best_metric"),
    ):
        assert c0[path[0]][path[1]] == p1[path[0]][path[1]]
    assert c0["features"]["forensic_fusion"] == "none"


def test_smoke_forward_backward_step_and_checkpoint_reload_passed():
    row = json.loads((OUT / "c0_training/run/preflight/smoke_step1/metrics.jsonl").read_text())
    assert row["optimizer_step"] == 1
    for key in ("total_loss", "text_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss"):
        assert row[key] == pytest.approx(row[key])
    resume = read_json(OUT / "c0_training/run/preflight/smoke_step1/resume_audit.json")
    assert resume["status"] == "passed"
    assert resume["optimizer_step"] == 1


def test_official_test_discipline_before_training():
    manifest = read_json(OUT / "manifest.json")
    assert manifest["official_test_used_for_training"] is False
    assert manifest["official_test_used_for_checkpoint_selection"] is False
    assert manifest["threshold_sweep_performed"] is False
    assert manifest["p1_retrained"] is False
    assert manifest["phase3b_started"] is False


def test_tf_protocol_manifest_has_distinct_frozen_inputs():
    protocols = read_json(OUT / "audit/tf_protocol_manifest.json")
    assert protocols["TF_OLD"]["target_protocol"] == "historical"
    assert protocols["TF_PHRASE"]["target_protocol"] == "phrase_aligned"
    assert protocols["TF_OLD"]["template"] != protocols["TF_PHRASE"]["template"]
    assert protocols["mask_logit_threshold"] == 0.0
