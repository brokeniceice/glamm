import json
from pathlib import Path

import pytest

from eval.comparison_scope import (
    ComparisonScope,
    ComparisonScopeError,
    assert_comparable_scope,
    derived_count_weighted_diagnostic,
    scoped_delta,
    validate_authoritative_mapping,
)


def scope(*, category="object", split="official test", aggregation="global pixels"):
    return ComparisonScope(
        dataset="SynthScars",
        split=split,
        category=category,
        n=162,
        target_representation="single union mask",
        metric_family="fg/bg mIoU",
        aggregation=aggregation,
        threshold="mask logit > 0",
    )


def mapping_status():
    return json.loads(Path("outputs/phase2b1_legion_category_correction/official_content_mapping_status.json").read_text())


def authoritative_rows_or_skip():
    status = mapping_status()
    if not status["authoritative_mapping_available"]:
        pytest.skip(status["mapping_tests"])
    mapping = Path("outputs/phase2b1_legion_category_correction/official1000_content_mapping.jsonl")
    return [json.loads(line) for line in mapping.read_text().splitlines() if line]


def test_reject_overall_vs_object_comparison():
    with pytest.raises(ComparisonScopeError, match="category"):
        assert_comparable_scope(scope(category="overall"), scope(category="object"))


def test_allow_object_vs_object_comparison():
    assert scoped_delta(0.55, 0.54, scope(), scope()) == pytest.approx(0.01)


def test_different_split_comparison_is_rejected():
    with pytest.raises(ComparisonScopeError, match="split"):
        assert_comparable_scope(scope(split="internal test"), scope(split="official test"))


def test_different_aggregation_is_rejected():
    with pytest.raises(ComparisonScopeError, match="aggregation"):
        assert_comparable_scope(scope(aggregation="global pixels"), scope(aggregation="per-image macro"))


def test_proxy_labels_cannot_be_marked_official():
    row = {"image_id": "a", "image_sha256": "0" * 64, "content_type": "human", "mapping_method": "proxy_clip"}
    with pytest.raises(ComparisonScopeError, match="Proxy"):
        validate_authoritative_mapping([row], {"human": 1})


def test_official_category_counts_sum_to_1000():
    status = mapping_status()
    assert status["counts_sum"] == 1000
    assert status["official_aggregate_category_counts"] == {"human": 587, "object": 162, "animal": 134, "scene": 117}


def test_weighted_aggregate_math_and_derived_flag():
    counts = {"human": 587, "object": 162, "animal": 134, "scene": 117}
    miou = {"object": 54.62, "animal": 54.52, "human": 60.82, "scene": 53.67}
    result = derived_count_weighted_diagnostic(counts, miou)
    assert result["value_percent"] == pytest.approx(58.13485)
    assert result["derived"] is True
    assert result["paper_reported_overall"] is False


def test_authoritative_mapping_unique_image_count_or_skip():
    rows = authoritative_rows_or_skip()
    assert len({row["image_id"] for row in rows}) == 1000


def test_authoritative_mapping_sha256_has_no_duplicates_or_skip():
    rows = authoritative_rows_or_skip()
    assert len({row["image_sha256"] for row in rows}) == 1000


def test_authoritative_mapping_category_counts_or_skip():
    rows = authoritative_rows_or_skip()
    result = validate_authoritative_mapping(rows, {"human": 587, "object": 162, "animal": 134, "scene": 117})
    assert result["category_counts"] == {"human": 587, "object": 162, "animal": 134, "scene": 117}


def test_unavailable_mapping_does_not_create_proxy_official_artifact():
    status = mapping_status()
    if not status["authoritative_mapping_available"]:
        assert not Path("outputs/phase2b1_legion_category_correction/official1000_content_mapping.jsonl").exists()
        assert not Path("outputs/phase2b1_legion_category_correction/phase2a_category_metrics.json").exists()


def test_comparison_scope_audit_marks_all_discovered_errors_invalid():
    rows = json.loads(Path("outputs/phase2b1_legion_category_correction/comparison_scope_audit.json").read_text())
    assert len(rows) == 6
    assert all(row["valid_comparison"] is False and row["correction_required"] is True for row in rows)


def test_correction_does_not_modify_phase2a_or_phase2c_selection():
    summary = json.loads(Path("outputs/phase2b1_legion_category_correction/correction_summary.json").read_text())
    assert summary["phase2a_checkpoint_changed"] is False
    assert summary["phase2c_training_or_selection_changed"] is False
    assert summary["corrected_status"] == "CATEGORY_PARITY_UNRESOLVED"
