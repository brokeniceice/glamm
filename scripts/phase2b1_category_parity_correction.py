#!/usr/bin/env python3
"""Generate the read-only Phase 2B.1 category-scope correction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.comparison_scope import derived_count_weighted_diagnostic


DEFAULT_OUTPUT = ROOT / "outputs/phase2b1_legion_category_correction"
EXPECTED_COUNTS = {"human": 587, "object": 162, "animal": 134, "scene": 117}
LEGION_MIOU = {"object": 54.62, "animal": 54.52, "human": 60.82, "scene": 53.67}
LEGION_F1 = {"object": 29.90, "animal": 27.43, "human": 39.44, "scene": 24.51}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_scope_audit() -> list[dict]:
    overall = {
        "dataset": "SynthScars",
        "split": "official test",
        "category": "overall",
        "n": 1000,
        "target_representation": "single union mask",
        "metric": "fg/bg mIoU candidate and foreground F1",
        "aggregation": "global pixels",
        "threshold": "mask logit > 0",
    }
    legion_object = {
        "dataset": "SynthScars",
        "split": "official test",
        "category": "object",
        "n": 162,
        "target_representation": "LEGION Table 2 localization output",
        "metric": "paper mIoU and F1",
        "aggregation": "unpublished exact aggregation",
        "threshold": "unpublished",
    }
    rows = [
        ("docs/phase2b_legion_parity_and_localization_gap.md", "Conclusion, paragraph 1",
         "Internal overall G0 fg/bg mIoU 0.544256 was described as only about 0.19 points from LEGION 0.5462."),
        ("docs/phase2b_legion_parity_and_localization_gap.md", "Conclusion, paragraph 2",
         "Official-1000 overall G0 0.545930/0.295168 was subtracted from LEGION Object 0.5462/0.2990."),
        ("docs/phase2b_legion_parity_and_localization_gap.md", "Section 10.C Real model-performance gap",
         "The overall-vs-Object residual was called the closest observable candidate and approximately zero."),
        ("docs/phase2c_npr_srm_forensic_enhancement.md", "Section 2 Phase 2B conclusion and motivation",
         "Official-1000 overall 0.545930/0.295168 was described as close to LEGION 0.5462/0.2990."),
        ("outputs/phase2b_legion_parity/future_hypotheses.md", "Primary recommendation: Case A",
         "Official-1000 overall 0.545930/0.295168 was placed versus LEGION Object 0.5462/0.2990."),
        ("outputs/phase2b_legion_parity/metric_parity/optional_official1000.json", "paper_table2_reference",
         "Object-only Table 2 numbers were stored beside overall official-1000 metrics without category scope."),
    ]
    return [
        {
            "file": file,
            "section": section,
            "original_statement": statement,
            "phase2a_scope": overall,
            "legion_scope": legion_object,
            "valid_comparison": False,
            "correction_required": True,
            "scope_mismatches": ["category", "n", "target_representation", "aggregation", "threshold"],
        }
        for file, section, statement in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-test-json",
        type=Path,
        default=Path("/data/yz/myLISA_storage/AIGC/SynthScars/test/annotations/test.json"),
    )
    args = parser.parse_args()
    output = args.output
    for directory in ("source_reaudit", "category_mapping", "comparison_scope_audit", "rescoring", "derived_aggregate", "tests"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    annotations = json.loads(args.official_test_json.read_text(encoding="utf-8"))
    record_fields, ref_fields = set(), set()
    for outer in annotations:
        for record in outer.values():
            record_fields.update(record)
            for ref in record.get("refs", []):
                ref_fields.update(ref)

    source_audit = {
        "audit_date": "2026-08-11",
        "official_legion_repository": "https://github.com/opendatalab/LEGION",
        "repository_commit_audited": "d21535dd45f6fea509337a83095966f0b86ac924",
        "repository_history_commits_audited": 27,
        "repository_history_mapping_found": False,
        "official_huggingface_repository": "https://huggingface.co/datasets/khr0516/SynthScars",
        "huggingface_files_all_history": [".gitattributes", "README.md", "SynthScars.zip"],
        "huggingface_zip_commit": "ad75f2fd47594f484c5d63fbfe412b179ff2526f",
        "huggingface_zip_lfs_sha256": "2ed817007d3d8483df72fabf35b2553eceab63ed4b5a8bcab8fae8979b6e8d87",
        "official_test_json": str(args.official_test_json),
        "official_test_json_sha256": sha256(args.official_test_json),
        "official_test_rows": len(annotations),
        "record_fields": sorted(record_fields),
        "ref_fields": sorted(ref_fields),
        "content_field_present": False,
        "archive_category_directories_present": False,
        "paper_supplies_aggregate_counts_only": EXPECTED_COUNTS,
        "authoritative_per_image_mapping_found": False,
        "prohibited_proxies_not_used": ["LLM/image inspection", "CLIP zero-shot", "filename guessing", "internal-1104 labels"],
    }
    dump(output / "source_reaudit/official_sources.json", source_audit)

    mapping_status = {
        "status": "AUTHORITATIVE_PER_IMAGE_CONTENT_MAPPING_UNAVAILABLE",
        "corrected_status": "CATEGORY_PARITY_UNRESOLVED",
        "authoritative_mapping_available": False,
        "official_test_unique_images": 1000,
        "official_aggregate_category_counts": EXPECTED_COUNTS,
        "counts_sum": sum(EXPECTED_COUNTS.values()),
        "official1000_content_mapping_jsonl_created": False,
        "phase2a_category_metrics_created": False,
        "mapping_tests": "SKIP: no authoritative per-image mapping is publicly available",
        "reason": "Official release annotations, archive layout, Hugging Face history, LEGION repository/history, paper, and supplement provide no per-image content field or mapping.",
    }
    dump(output / "official_content_mapping_status.json", mapping_status)
    dump(output / "category_mapping/official_content_mapping_status.json", mapping_status)
    dump(output / "rescoring/status.json", {
        "performed": False,
        "reason": "authoritative per-image content mapping unavailable",
        "phase2a_category_metrics_created": False,
    })

    legion_reference = {
        "dataset": "SynthScars",
        "split": "official test",
        "source": "LEGION ICCV 2025 paper Table 2 and supplementary category counts",
        "categories": {
            category: {"n": EXPECTED_COUNTS[category], "mIoU_percent": LEGION_MIOU[category], "F1_percent": LEGION_F1[category]}
            for category in ("object", "animal", "human", "scene")
        },
        "reported_overall": False,
        "exact_aggregation_public": False,
    }
    dump(output / "legion_table2_reference.json", legion_reference)
    dump(output / "source_reaudit/legion_table2_reference.json", legion_reference)

    weighted = {
        "name": "LEGION_COUNT_WEIGHTED_CATEGORY_DIAGNOSTIC",
        "mIoU": derived_count_weighted_diagnostic(EXPECTED_COUNTS, LEGION_MIOU),
        "F1": derived_count_weighted_diagnostic(EXPECTED_COUNTS, LEGION_F1),
        "derived": True,
        "assumption_required": True,
        "warning": "This is not a LEGION paper-reported overall metric; exact Table 2 aggregation is unknown.",
    }
    dump(output / "derived_weighted_diagnostic.json", weighted)
    dump(output / "derived_aggregate/derived_weighted_diagnostic.json", weighted)

    scope_rows = comparison_scope_audit()
    dump(output / "comparison_scope_audit.json", scope_rows)
    dump(output / "comparison_scope_audit/comparison_scope_audit.json", scope_rows)

    correction = {
        "corrected_status": "CATEGORY_PARITY_UNRESOLVED",
        "phase2a_official1000_scope": "overall, n=1000",
        "phase2a_values": {"global_fg_bg_mIoU": 0.5459300410760968, "global_FG_F1": 0.2951678961075328},
        "legion_05462_02990_scope": "Object category, n=162",
        "original_subtraction_valid": False,
        "withdrawn_claim": "closest observable candidate approximately zero",
        "certain_metric_conclusion": "0.1396 foreground IoU versus LEGION fg/bg mIoU is invalid and severely exaggerates the apparent gap.",
        "certain_scale_conclusion": "Phase2A overall fg/bg-style value is in the numerical range of LEGION category values, but this is not parity.",
        "unresolved": "same-category, exact-aggregation model gap",
        "phase2a_checkpoint_changed": False,
        "phase2c_training_or_selection_changed": False,
        "phase2c_forensic_conclusion_independent": True,
        "category_rescoring_performed": False,
        "category_rescoring_reason": "authoritative per-image content mapping unavailable",
    }
    dump(output / "correction_summary.json", correction)
    dump(output / "tests/test_contract.json", {
        "expected_existing_regression": "Phase1A-Phase2C",
        "mapping_tests_expected": "skipped with explicit reason",
        "scope_guard_required": True,
        "training_or_inference_required": False,
    })
    print(json.dumps({"status": correction["corrected_status"], "output": str(output)}))


if __name__ == "__main__":
    main()
