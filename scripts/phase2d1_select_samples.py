#!/usr/bin/env python3
"""Deterministically select the Phase 2D.1 official diagnostic subset."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2d1_trace_replay/selection"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    rows = read_jsonl(ROOT / "outputs/phase2d_grounding_gap/per_image/official1000_grounding_gap.jsonl")
    old_g0 = {row["sample_id"]: row for row in read_jsonl(
        ROOT / "outputs/phase2b_legion_parity/official1000_raw/G0/predictions.jsonl")}
    selected: dict[str, dict] = {}

    def add(row, label):
        item = selected.setdefault(row["sample_id"], {
            "sample_id": row["sample_id"], "subsets": [],
            "historical_g0_iou": row["G0"]["foreground_iou"],
            "historical_tf_iou": row["TF"]["foreground_iou"],
            "historical_gap": row["paired"]["tf_minus_g0_foreground_iou"],
            "taxonomy": row["text_analysis"]["taxonomy"],
            "B3_fixed_B0": row["classification"]["B3_fixed_B0"],
        })
        if label not in item["subsets"]:
            item["subsets"].append(label)

    severe = [row for row in rows if row["TF"]["foreground_iou"] >= .7 and row["G0"]["foreground_iou"] <= .3]
    f_candidates = [row for row in severe if row["text_analysis"]["taxonomy"] == "F_SEG_STATE_ANOMALY_CANDIDATE"]
    for row in f_candidates:
        add(row, "S2_all_F_candidates")
    for taxonomy in sorted({row["text_analysis"]["taxonomy"] for row in severe}):
        category = sorted((row for row in severe if row["text_analysis"]["taxonomy"] == taxonomy),
                          key=lambda row: row["paired"]["tf_minus_g0_foreground_iou"], reverse=True)
        for row in category[:4]:
            add(row, "S1_severe_taxonomy_coverage")
    ordered_severe = sorted(severe, key=lambda row: row["paired"]["tf_minus_g0_foreground_iou"], reverse=True)
    for row in ordered_severe[:10]:
        add(row, "S1_largest_gap")
    median_severe = sorted(severe, key=lambda row: row["paired"]["tf_minus_g0_foreground_iou"])
    for row in median_severe[len(median_severe)//2-2:len(median_severe)//2+3]:
        add(row, "S1_median_severe_gap")
    for row in severe:
        if row["classification"]["B3_fixed_B0"]:
            add(row, "S1_B3_corrected")

    negatives = sorted((row for row in rows if row["paired"]["tf_minus_g0_foreground_iou"] < 0),
                       key=lambda row: row["paired"]["tf_minus_g0_foreground_iou"])
    for row in negatives[:12]:
        add(row, "S3_largest_negative_gap")
    for row in negatives[len(negatives)//2-3:len(negatives)//2+3]:
        add(row, "S3_moderate_negative_gap")

    anchors = ordered_severe[:16]
    controls = [row for row in rows if row not in severe and row["G0"]["foreground_iou"] > .3]
    used = set(selected)
    for anchor in anchors:
        anchor_old = old_g0[anchor["sample_id"]]
        gt_area = int(anchor_old["tp"]) + int(anchor_old["fn"])
        seq_len = len(anchor["G0"]["generated_token_ids"] or [])
        candidates = [row for row in controls if row["sample_id"] not in used]
        if not candidates:
            break
        def cost(row):
            old = old_g0[row["sample_id"]]
            area = int(old["tp"]) + int(old["fn"])
            length = len(row["G0"]["generated_token_ids"] or [])
            return (abs(row["TF"]["foreground_iou"] - anchor["TF"]["foreground_iou"]) * 4
                    + abs(area - gt_area) / max(gt_area, 1)
                    + abs(length - seq_len) / max(seq_len, 1))
        match = min(candidates, key=cost)
        add(match, "S4_matched_control")
        selected[match["sample_id"]]["matched_to"] = anchor["sample_id"]
        used.add(match["sample_id"])

    for row in ordered_severe:
        if len(selected) >= 64:
            break
        add(row, "S1_severe_fill")
    output = list(selected.values())[:64]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    counts = {}
    for row in output:
        for subset in row["subsets"]:
            counts[subset] = counts.get(subset, 0) + 1
    summary = {"n": len(output), "selection_seed": None, "deterministic": True,
               "all_13_F_candidates_included": len(f_candidates) == sum(
                   "S2_all_F_candidates" in row["subsets"] for row in output), "subset_memberships": counts}
    (OUT / "subset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
