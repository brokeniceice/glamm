#!/usr/bin/env python3
"""Validate completed Phase 2B inference artifacts and write comparison JSON."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2b_legion_parity_audit import dump, load_jsonl, suite


OUT = ROOT / "outputs/phase2b_legion_parity"
PHASE2A = ROOT / "outputs/phase2a_unified_baseline"
MANIFESTS = ROOT / "outputs/data_audits/unified_forensics_split_v1"


def enrich(predictions: list[dict], manifest: dict[str, dict]) -> list[dict]:
    output = []
    for row in predictions:
        source = manifest[row["sample_id"]]
        with Image.open(source["image_path"]) as image:
            pixels = image.width * image.height
        output.append({
            "sample_id": row["sample_id"], "content": source.get("content_category"),
            "tp": int(row["tp"]), "fp": int(row["fp"]), "fn": int(row["fn"]), "pixels": pixels,
        })
    return output


def rescore(raw_root: Path, manifest_path: Path, expected_detection: int, expected_fake: int,
            split_label: str) -> dict:
    manifest_rows = load_jsonl(manifest_path)
    manifest = {r["sample_id"]: r for r in manifest_rows}
    detection = load_jsonl(raw_root / "detection/predictions.jsonl")
    if len(detection) != expected_detection:
        raise ValueError(f"{raw_root}: detection incomplete {len(detection)}/{expected_detection}")
    detection_ids = [row["sample_id"] for row in detection]
    if len(set(detection_ids)) != expected_detection:
        raise ValueError(
            f"{raw_root}: detection has duplicate sample IDs "
            f"({len(set(detection_ids))}/{expected_detection} unique)"
        )
    expected_detection_ids = set(manifest)
    if set(detection_ids) != expected_detection_ids:
        missing = sorted(expected_detection_ids - set(detection_ids))[:5]
        extra = sorted(set(detection_ids) - expected_detection_ids)[:5]
        raise ValueError(f"{raw_root}: detection ID mismatch; missing={missing}, extra={extra}")
    result = {"split": split_label, "threshold": "mask_logit > 0", "modes": {}}
    for mode in ("G0", "G1", "tf_full_context"):
        rows = load_jsonl(raw_root / mode / "predictions.jsonl")
        if len(rows) != expected_fake:
            raise ValueError(f"{raw_root}: {mode} incomplete {len(rows)}/{expected_fake}")
        row_ids = [row["sample_id"] for row in rows]
        if len(set(row_ids)) != expected_fake:
            raise ValueError(
                f"{raw_root}: {mode} has duplicate sample IDs "
                f"({len(set(row_ids))}/{expected_fake} unique)"
            )
        expected_fake_ids = {
            sample_id for sample_id, source in manifest.items()
            if source.get("forensics_domain") == "fake" or source.get("class_label") == 1
        }
        if set(row_ids) != expected_fake_ids:
            missing = sorted(expected_fake_ids - set(row_ids))[:5]
            extra = sorted(set(row_ids) - expected_fake_ids)[:5]
            raise ValueError(f"{raw_root}: {mode} ID mismatch; missing={missing}, extra={extra}")
        values = enrich(rows, manifest)
        result["modes"]["TF" if mode == "tf_full_context" else mode] = {
            "overall": suite(values),
            "by_content": {
                content: suite([r for r in values if r["content"] == content])
                for content in ("human", "animal", "object", "scene")
                if any(r["content"] == content for r in values)
            },
        }
    summaries = [json.loads(path.read_text()) for path in raw_root.glob("summary_*.json")]
    summary = next((value for value in summaries if "detection" in value.get("modes", {})), None)
    if summary is None:
        raise ValueError(f"{raw_root}: no completed summary containing detection")
    result["checkpoint"] = summary["checkpoint"]
    result["detection"] = json.loads((raw_root / "detection/metrics.json").read_text())
    return result


def delta(a, b):
    return float(b) - float(a)


def main():
    best = json.loads((OUT / "metric_parity/phase2a_best_internal.json").read_text())
    last = rescore(
        OUT / "last_internal_raw", MANIFESTS / "test_combined.jsonl", 2208, 1104,
        "Phase2A frozen internal test",
    )
    official = rescore(
        OUT / "official1000_raw", OUT / "split_audit/official1000_manifest/test_combined.jsonl",
        1000, 1000, "Official SynthScars test 1000 (zero overlap with Phase2A train)",
    )
    official["comparison_status"] = "ELIGIBLE_SPLIT_PARITY"
    official["content_breakdown_status"] = "unresolved: release JSON has no per-image content labels"
    official["paper_table2_reference"] = {"LEGION_mIoU": 0.5462, "LEGION_F1": 0.2990}
    official["warning"] = "Paper Table 2 exact aggregation/F1 implementation is unavailable; do not subtract these as an exact model-only gap."
    dump(OUT / "metric_parity/phase2a_last_internal.json", last)
    dump(OUT / "metric_parity/optional_official1000.json", official)

    val_best = json.loads((PHASE2A / "validation/epoch_05/metrics.json").read_text())
    val_last = json.loads((PHASE2A / "validation/epoch_10/metrics.json").read_text())
    best_det = json.loads((PHASE2A / "test/detection/metrics.json").read_text())
    tradeoff = {
        "policy": "best remains step2500 selected by min validation total loss; last is diagnostic only",
        "validation": {
            "best_step2500": {
                "total_loss": val_best["val_total_loss"], "cls_accuracy": val_best["classification"]["accuracy"],
                "lm_accuracy": val_best["lm_verdict"]["accuracy"],
                "TF_mean_fg_iou": val_best["tf_full_context"]["mean_iou"],
                "TF_global_fg_iou": val_best["tf_full_context"]["global_iou"],
            },
            "last_step5000": {
                "total_loss": val_last["val_total_loss"], "cls_accuracy": val_last["classification"]["accuracy"],
                "lm_accuracy": val_last["lm_verdict"]["accuracy"],
                "TF_mean_fg_iou": val_last["tf_full_context"]["mean_iou"],
                "TF_global_fg_iou": val_last["tf_full_context"]["global_iou"],
            },
            "last_minus_best": {
                "total_loss": delta(val_best["val_total_loss"], val_last["val_total_loss"]),
                "cls_accuracy": delta(val_best["classification"]["accuracy"], val_last["classification"]["accuracy"]),
                "lm_accuracy": delta(val_best["lm_verdict"]["accuracy"], val_last["lm_verdict"]["accuracy"]),
                "TF_mean_fg_iou": delta(val_best["tf_full_context"]["mean_iou"], val_last["tf_full_context"]["mean_iou"]),
                "TF_global_fg_iou": delta(val_best["tf_full_context"]["global_iou"], val_last["tf_full_context"]["global_iou"]),
            },
        },
        "internal_test": {
            "best_step2500": {
                "cls_accuracy": best_det["classification_head"]["accuracy"],
                "lm_accuracy": best_det["lm_verdict"]["accuracy"],
                **{mode: best["overall"][mode] for mode in ("TF", "G0", "G1")},
            },
            "last_step5000": {
                "cls_accuracy": last["detection"]["classification_head"]["accuracy"],
                "lm_accuracy": last["detection"]["lm_verdict"]["accuracy"],
                **{mode: last["modes"][mode]["overall"] for mode in ("TF", "G0", "G1")},
            },
            "last_minus_best": {
                "cls_accuracy": delta(best_det["classification_head"]["accuracy"], last["detection"]["classification_head"]["accuracy"]),
                "lm_accuracy": delta(best_det["lm_verdict"]["accuracy"], last["detection"]["lm_verdict"]["accuracy"]),
                **{
                    f"{mode}_{metric}": delta(best["overall"][mode][metric], last["modes"][mode]["overall"][metric])
                    for mode in ("TF", "G0", "G1")
                    for metric in ("IoU_fg_per_image_mean", "IoU_fg_global", "mIoU_fg_bg_per_image", "PixelF1_fg_per_image_mean")
                },
            },
        },
    }
    dump(OUT / "metric_parity/best_vs_last_diagnostic.json", tradeoff)
    panels = load_jsonl(OUT / "visualizations/bounded_inference.jsonl")
    if len(panels) != 80:
        raise ValueError(f"Visualization inference incomplete {len(panels)}/80")
    quadrants = {}
    for row in panels:
        quadrants[row["quadrant"]] = quadrants.get(row["quadrant"], 0) + 1
    dump(OUT / "visualizations/summary.json", {
        "cases": len(panels), "per_quadrant": quadrants,
        "required_panels": ["Original", "GT", "TF", "G0", "G1"],
        "checkpoint_sha256": panels[0]["checkpoint"]["checkpoint_sha256"],
    })
    print(json.dumps({"status": "complete", "last": 1104, "official": 1000, "panels": 80}))


if __name__ == "__main__":
    main()
