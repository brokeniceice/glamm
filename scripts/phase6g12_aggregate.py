#!/usr/bin/env python3
"""Aggregate Phase 6G.12 A0/A2 and the Phase6G.11 A1 reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3c1 import paired_statistics

OUT = ROOT / "outputs/phase6g12_full_rectifier_side_correction"
G11 = ROOT / "outputs/phase6g11_zero_init_correction_side_path"
G10 = ROOT / "outputs/phase6g10_post_attention_projection_bypass"
A0_VAL = G10 / "arms/A0/validation/selected.jsonl"
A0_SUMMARY = G10 / "arms/A0/summary.json"
A1_VAL = G11 / "training/validation/selected.jsonl"
A1_RESULTS = G11 / "results.json"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def iou(path):
    return np.asarray([row["foreground_iou"] for row in rows(path)], dtype=np.float64)


def stable_positive(paired):
    return (
        paired["mean_difference"] > 0
        and paired["bootstrap_95_ci"][0] > 0
        and paired["wilcoxon_pvalue"] < 0.05
    )


def stable_negative(paired):
    return (
        paired["mean_difference"] < 0
        and paired["bootstrap_95_ci"][1] < 0
        and paired["wilcoxon_pvalue"] < 0.05
    )


def compact(paired):
    return {
        "mean": paired["mean_difference"],
        "median": paired["median_difference"],
        "ci": paired["bootstrap_95_ci"],
        "wins": paired["wins"],
        "ties": paired["ties"],
        "losses": paired["losses"],
        "wilcoxon_p": paired["wilcoxon_pvalue"],
    }


def main():
    a2_path = OUT / "arms/A2/validation/selected.jsonl"
    a2_summary = OUT / "arms/A2/summary.json"
    for path in (A0_VAL, A0_SUMMARY, a2_path, a2_summary, A1_VAL, A1_RESULTS):
        if not path.exists():
            print(json.dumps({"status": "INCOMPLETE", "missing": str(path)}))
            return

    a0, a2, a1 = iou(A0_VAL), iou(a2_path), iou(A1_VAL)
    if not (len(a0) == len(a2) == len(a1)):
        raise RuntimeError("paired population drift")
    p_a2_a0 = paired_statistics(a2, a0)
    p_a2_a1 = paired_statistics(a2, a1)
    a1_a0_raw = json.loads(A1_RESULTS.read_text())["paired_A1_minus_A0"]
    result = {
        "schema": "phase6g12_results_v1",
        "status": "COMPLETE_STOP",
        "A0": json.loads(A0_SUMMARY.read_text())["rectifier_selector"]["selected_metrics"],
        "A2": json.loads(a2_summary.read_text())["selected_metrics"],
        "A1_reference": json.loads(A1_RESULTS.read_text())["A1"],
        "paired": {
            "A2_minus_A0": compact(p_a2_a0),
            "A2_minus_A1": compact(p_a2_a1),
            "A1_minus_A0_reference": a1_a0_raw,
        },
        "mechanism": {
            "A1_reference": json.loads(A1_RESULTS.read_text())["mechanism_diagnostics"],
            "A2": json.loads(a2_summary.read_text()).get("mechanism_diagnostics"),
            "A2_main_composition": json.loads(a2_summary.read_text())["matrix_spectrum"]["main_composition"],
            "A2_side_spectrum": json.loads(a2_summary.read_text())["matrix_spectrum"]["side"],
            "A2_side_subspace": json.loads(a2_summary.read_text())["matrix_spectrum"]["side_subspace"],
        },
        "firewall": {"utility": False, "joint_r1": False, "fusion": False, "adapter": False,
                     "internal_test": False, "official1000": False, "ood": False},
    }
    a2_a0_pos = stable_positive(p_a2_a0)
    a2_a1_pos = stable_positive(p_a2_a1)
    a2_a1_neg = stable_negative(p_a2_a1)
    a1_a0_pos = stable_positive(a1_a0_raw)
    if a2_a0_pos and a2_a1_pos:
        decision = "JOINT_MAIN_SIDE_RECTIFIER_SUPPORTED"
        additional = "JOINT_ADAPTATION_IMPROVES_COMPLEMENTARY_CORRECTION"
    elif a2_a0_pos:
        decision = "FROZEN_MAIN_SIDE_CORRECTION_PREFERRED"
        additional = None
    elif a1_a0_pos:
        decision = "MAIN_TRANSLATOR_SHOULD_REMAIN_FROZEN"
        additional = None
    elif a2_a1_neg or stable_negative(p_a2_a0):
        decision = "JOINT_RECTIFIER_COADAPTATION_HARMS_COMPLEMENTARITY"
        additional = None
    else:
        decision = "FROZEN_MAIN_SIDE_CORRECTION_PREFERRED"
        additional = None
    result["decision"] = decision
    result["additional_record"] = additional
    dump(OUT / "results.json", result)
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                "additional_record": additional, "firewall": result["firewall"]})
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)


def render(result):
    lines = [
        "# Phase 6G.12 — Full Rectifier Training with Complementary Side Correction",
        "",
        "Status: **COMPLETE STOP**.",
        "",
        "| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |",
        "|---|---:|---:|---:|",
        f"| A0 main-only | {result['A0']['mean_foreground_iou']:.6f} | {result['A0']['median_foreground_iou']:.6f} | {result['A0']['mean_foreground_f1']:.6f} |",
        f"| A1 frozen main + side (ref) | {result['A1_reference']['mean_foreground_iou']:.6f} | {result['A1_reference']['median_foreground_iou']:.6f} | {result['A1_reference']['mean_foreground_f1']:.6f} |",
        f"| A2 joint main + side | {result['A2']['mean_foreground_iou']:.6f} | {result['A2']['median_foreground_iou']:.6f} | {result['A2']['mean_foreground_f1']:.6f} |",
        "",
        f"- A2 - A0: `{result['paired']['A2_minus_A0']}`",
        f"- A2 - A1: `{result['paired']['A2_minus_A1']}`",
        f"- A1 - A0 reference: `{result['paired']['A1_minus_A0_reference']}`",
        "",
        f"```text\n{result['decision']}\n```",
    ]
    if result.get("additional_record"):
        lines += ["", f"Additional record: `{result['additional_record']}`"]
    lines += ["", "No Utility, joint R1, internal test, Official1000, or OOD was accessed."]
    (ROOT / "docs/phase6g12_full_rectifier_side_correction.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
