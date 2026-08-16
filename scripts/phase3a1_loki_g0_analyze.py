#!/usr/bin/env python3
"""Finalize paired new-C0/P1 LOKI G0/G1 metrics without model inference."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3a1_analyze import paired_summary
from scripts.phase3a_evaluate import dump_json, summarize_localization


OUT = ROOT / "outputs/phase3a1_paired_control/evaluation/external_g0/loki"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    fields = {
        "foreground_iou": "foreground_iou",
        "foreground_f1": "foreground_f1",
        "fg_bg_miou": "fg_bg_miou",
    }
    predictions = {
        mode: {
            model: rows(OUT / model / f"{mode}/predictions.jsonl")
            for model in ("new_c0", "p1")
        }
        for mode in ("G0", "G1")
    }
    all_values = [values for by_model in predictions.values() for values in by_model.values()]
    ordered_ids = [[row["sample_id"] for row in values] for values in all_values]
    if any(len(values) != 229 for values in all_values) or any(ids != ordered_ids[0] for ids in ordered_ids[1:]):
        raise RuntimeError("new-C0/P1 LOKI G0/G1 paired scope or ordering mismatch")

    metrics = {mode: {} for mode in predictions}
    for mode, by_model in predictions.items():
        for model, values in by_model.items():
            metric = summarize_localization(values, autoregressive=True)
            metric["legion_table2_metric_aligned"] = {
                "mIoU_percent": 100.0 * metric["global_pixel"]["fg_bg_miou"],
                "foreground_F1_percent": 100.0 * metric["global_pixel"]["foreground_f1"],
                "aggregation": "dataset-global pixel confusion, then foreground/background IoU mean",
                "gt_semantics": "union of LOKI regional bounding boxes rasterized as filled rectangles",
                "scope": "229 fully synthetic LOKI images with valid regional annotations",
                "protocol_boundary": (
                    "G1 supplies a structural GT [FAKE] continuation prefix; LEGION uses a direct "
                    "artifact-localization instruction, so inference prompts are not identical."
                    if mode == "G1" else
                    "G0 freely generates the authenticity verdict; LEGION uses a direct "
                    "artifact-localization instruction, so inference prompts are not aligned."
                ),
            }
            metric["stop_reason_counts"] = dict(Counter(row["stop_reason"] for row in values))
            metrics[mode][model] = metric
            dump_json(OUT / model / f"{mode}/metrics.json", metric)
            summary_path = OUT / model / ("summary.json" if mode == "G0" else f"{mode}/summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics"] = metric
            dump_json(summary_path, summary)

    def paired(left: list[dict], right: list[dict]) -> dict:
        return {
            name: paired_summary(np.asarray([
                float(r[field]) - float(l[field]) for l, r in zip(left, right)
            ]))
            for name, field in fields.items()
        }

    p1_minus_c0 = {
        mode: paired(predictions[mode]["new_c0"], predictions[mode]["p1"])
        for mode in predictions
    }
    g1_minus_g0 = {
        model: paired(predictions["G0"][model], predictions["G1"][model])
        for model in ("new_c0", "p1")
    }

    per_sample = []
    for index, sample_id in enumerate(ordered_ids[0]):
        record = {"sample_id": sample_id}
        for mode in ("G0", "G1"):
            for model in ("new_c0", "p1"):
                row = predictions[mode][model][index]
                for name, field in fields.items():
                    record[f"{model}_{mode}_{name}"] = float(row[field])
                record[f"{model}_{mode}_seg_triggered"] = bool(row["seg_triggered"])
        per_sample.append(record)
    per_sample_path = OUT / "g1_per_sample_comparison.jsonl"
    per_sample_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_sample),
        encoding="utf-8",
    )

    comparison = {
        "schema_version": "phase3a1_loki_g0_g1_comparison_v1",
        "scope": "LOKI 229 fully synthetic images with valid regional bbox annotations",
        "gt_semantics": "LEGION-compatible filled rectangle union derived from LOKI regional xywh boxes",
        "pixel_level_human_masks_available": False,
        "selection_used_loki": False,
        "models": metrics,
        "p1_minus_new_c0_paired": p1_minus_c0,
        "g1_minus_g0_paired": g1_minus_g0,
        "legion_table2_reference_percent": {"mIoU": 48.66, "foreground_F1": 16.71},
        "comparison_boundary": (
            "Same public 229-image scope, bbox-rasterization semantics, threshold, and metric aggregation. "
            "G1 conditions this model with a structural GT [FAKE] prefix, whereas LEGION directly requests "
            "artifact localization; the prompts are not identical."
        ),
    }
    dump_json(OUT / "g1_comparison.json", comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
