import json
from pathlib import Path

import pytest

from eval.phase2d_paired_metrics import aggregate_global, per_image_mask_metrics
from eval.phase2d_scope_validation import assert_exact_identity, reject_proxy_category


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2d_grounding_gap"


def test_scope_rejects_different_split_or_identity():
    official = [{"sample_id": "synthscars:test:a"}]
    internal = [{"sample_id": "synthscars:internal:a"}]
    with pytest.raises(ValueError, match="identity mismatch"):
        assert_exact_identity(official, internal, reference_name="official", candidate_name="internal")


def test_proxy_category_cannot_be_official():
    with pytest.raises(ValueError, match="Proxy category"):
        reject_proxy_category({"label_provenance": "filename_heuristic_proxy"})


def test_per_image_and_global_metric_semantics_are_distinct():
    first = {"tp": 1, "fp": 0, "fn": 0}
    second = {"tp": 1, "fp": 9, "fn": 0}
    per_image = [per_image_mask_metrics(first, 10), per_image_mask_metrics(second, 10)]
    global_metric = aggregate_global([first, second])
    assert sum(row["foreground_iou"] for row in per_image) / 2 == pytest.approx(.55)
    assert global_metric["global_foreground_iou"] == pytest.approx(.1818181818)
    assert "per_image_fg_bg_miou" in per_image[0]
    assert "per_image_fg_bg_miou" not in global_metric


def test_paired_delta_uses_same_per_image_metric():
    g0 = per_image_mask_metrics({"tp": 1, "fp": 1, "fn": 2}, 10)
    tf = per_image_mask_metrics({"tp": 3, "fp": 0, "fn": 0}, 10)
    assert tf["foreground_iou"] - g0["foreground_iou"] == pytest.approx(.75)


def test_frozen_reproducibility_artifact_matches():
    artifact = json.loads((OUT / "metrics/reproducibility.json").read_text())
    assert artifact["status"] == "MATCH"
    assert all(check["n_match"] for check in artifact["checks"].values())
    assert all(check["global_iou_abs_error"] < 1e-12 for check in artifact["checks"].values())


def test_every_controlled_mode_has_machine_readable_spec():
    specs = json.loads((OUT / "controlled_modes/mode_specs.json").read_text())
    assert {spec["mode"] for spec in specs} == {"H1", "H2", "H3", "H4"}
    for spec in specs:
        assert spec["status"]
        assert spec["reason"]
        assert spec["fixed_variables"]


def test_phase2d_is_diagnostic_only_and_does_not_claim_masks_exist():
    manifest = json.loads((OUT / "manifest.json").read_text())
    discipline = manifest["discipline"]
    assert discipline == {
        "training_started": False,
        "model_weights_modified": False,
        "checkpoint_selection_performed": False,
        "test_set_model_selection_performed": False,
        "proxy_category_labels_created": False,
    }
    assert manifest["artifacts"]["predicted_mask_visualization"] == "NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS"


def test_official1000_pairing_is_complete_and_unique():
    rows = [json.loads(line) for line in (OUT / "per_image/official1000_grounding_gap.jsonl").read_text().splitlines()]
    assert len(rows) == 1000
    assert len({row["sample_id"] for row in rows}) == 1000
    assert all(row["dataset"] == "SynthScars" and row["split"] == "official_test" for row in rows)
    assert all(row["mask_threshold"] == 0 for row in rows)
