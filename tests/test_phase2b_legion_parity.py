import json
from pathlib import Path

from scripts.phase2b_legion_parity_audit import safe_ratio, spearman, suite


def test_unified_metric_suite_distinguishes_foreground_and_two_class_miou():
    metrics = suite([{"tp": 1, "fp": 1, "fn": 1, "pixels": 10}])
    assert metrics["IoU_fg_per_image_mean"] == 1 / 3
    assert metrics["IoU_fg_global"] == 1 / 3
    assert metrics["IoU_bg_per_image_mean"] == 7 / 9
    assert metrics["mIoU_fg_bg_per_image"] == (1 / 3 + 7 / 9) / 2
    assert metrics["PixelF1_fg_global"] == 0.5


def test_spearman_uses_tied_average_ranks():
    assert spearman([1, 2, 3], [3, 2, 1])["rho"] == -1.0
    assert spearman([1, 1, 2], [1, 1, 2])["rho"] == 1.0


def test_phase2b_fixed_threshold_and_no_exact_table2_claim():
    path = Path("outputs/phase2b_legion_parity/metric_parity/metric_definition.json")
    definition = json.loads(path.read_text(encoding="utf-8"))
    assert definition["fixed_threshold"] == "prediction mask logit > 0 (equivalent sigmoid > 0.5)"
    assert definition["exact_parity_possible"] is False
    assert definition["LEGION_public_repo_validation_metric"]["class_index"] == 1
    assert definition["LEGION_public_repo_validation_metric"]["foreground_background_mean"] is False


def test_official_test_is_not_contaminated():
    path = Path("outputs/phase2b_legion_parity/split_audit/official_vs_frozen_overlap.json")
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["official_test_in_our_train"] == 0
    assert audit["official_test_in_our_val"] == 0
    assert audit["official_test_in_our_test"] == 0
    assert audit["official_test_absent"] == 1000


def test_same_source_annotations_are_pixel_exact():
    path = Path("outputs/phase2b_legion_parity/split_audit/annotation_parity.json")
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["compared_same_image_count"] == 11046
    assert audit["union_mask_pixel_exact_count"] == audit["compared_same_image_count"]
    assert audit["max_union_area_ratio_abs_diff"] == 0.0
