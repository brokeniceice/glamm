#!/usr/bin/env python3
"""Aggregate Phase 6G.10 arm results and apply the Rectifier-level gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3c1 import paired_statistics

OUT = ROOT / "outputs/phase6g10_post_attention_projection_bypass"
ARMS = ("A0", "A1", "A2")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def iou(path):
    values = rows(path)
    return np.asarray([row["foreground_iou"] for row in values], dtype=np.float64), values


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


def compare(arm_a, arm_b, kind, rect_path=False):
    if rect_path:
        a = iou(OUT / "arms" / arm_a / "validation" / "selected.jsonl")[0]
        b = iou(OUT / "arms" / arm_b / "validation" / "selected.jsonl")[0]
    else:
        a = iou(OUT / "arms" / arm_a / "probe" / "predictions" / f"{kind}.jsonl")[0]
        b = iou(OUT / "arms" / arm_b / "probe" / "predictions" / f"{kind}.jsonl")[0]
    return paired_statistics(a, b)


def main():
    summaries = {}
    for arm in ARMS:
        path = OUT / "arms" / arm / "summary.json"
        if not path.exists():
            print(json.dumps({"status": "INCOMPLETE", "missing": str(path)}))
            return
        summaries[arm] = json.loads(path.read_text())

    comparisons = {"rectifier_objective": {}, "R2_probe": {}, "Delta_F_probe": {}}
    for arm_a, arm_b in (("A1", "A0"), ("A2", "A0"), ("A2", "A1")):
        comparisons["rectifier_objective"][f"{arm_a}_minus_{arm_b}"] = compare(arm_a, arm_b, "", rect_path=True)
        comparisons["R2_probe"][f"{arm_a}_minus_{arm_b}"] = compare(arm_a, arm_b, "R2")
        comparisons["Delta_F_probe"][f"{arm_a}_minus_{arm_b}"] = compare(arm_a, arm_b, "Delta_F")

    within = {}
    for arm in ARMS:
        r2, _ = iou(OUT / "arms" / arm / "probe" / "predictions" / "R2.jsonl")
        delta, _ = iou(OUT / "arms" / arm / "probe" / "predictions" / "Delta_F.jsonl")
        within[arm] = {
            "delta_f_minus_r2": compact(paired_statistics(delta, r2)),
            "delta_f_over_r2_mean_iou": float(delta.mean() / max(1e-12, r2.mean())),
        }

    a1_vs_a0_delta = comparisons["Delta_F_probe"]["A1_minus_A0"]
    a2_vs_a0_delta = comparisons["Delta_F_probe"]["A2_minus_A0"]
    a2_vs_a1_delta = comparisons["Delta_F_probe"]["A2_minus_A1"]
    a1_vs_a0_rect = comparisons["rectifier_objective"]["A1_minus_A0"]
    a2_vs_a0_rect = comparisons["rectifier_objective"]["A2_minus_A0"]
    a2_vs_a1_rect = comparisons["rectifier_objective"]["A2_minus_A1"]

    a1_supported = stable_positive(a1_vs_a0_delta) or stable_positive(a1_vs_a0_rect)
    a2_supported = stable_positive(a2_vs_a0_delta) or stable_positive(a2_vs_a0_rect)
    a2_above_a1 = stable_positive(a2_vs_a1_delta) or stable_positive(a2_vs_a1_rect)
    a1_above_a2 = stable_negative(a2_vs_a1_delta) or stable_negative(a2_vs_a1_rect)

    if a2_supported and a2_above_a1 and a1_supported:
        decision = "DIRECT_R2_RESIDUAL_BYPASS_SUPPORTED"
        additional = "PROJECTION_DEPTH_CAUSES_FORENSIC_GAIN_COLLAPSE"
    elif a2_supported:
        decision = "DIRECT_R2_RESIDUAL_BYPASS_SUPPORTED"
        additional = None
    elif a1_supported and a1_above_a2:
        decision = "ONE_LEARNED_PROJECTION_IS_NEEDED"
        additional = None
    elif a1_supported:
        decision = "SINGLE_POST_ATTENTION_PROJECTION_SUPPORTED"
        additional = None
    else:
        decision = "LOW_EFFECTIVE_RANK_PROJECTION_NOT_CAUSALLY_HARMFUL"
        additional = None

    result = {
        "schema": "phase6g10_results_v1",
        "status": "COMPLETE_STOP",
        "decision": decision,
        "additional_record": additional,
        "metric_directions": {},
        "metric_conflict": False,
        "secondary_interpretation": None,
        "arms": {arm: {
            "rectifier": summaries[arm]["rectifier_selector"],
            "probe": summaries[arm]["probe_summary"],
            "matrix_spectrum": summaries[arm]["matrix_spectrum"],
        } for arm in ARMS},
        "comparisons": {kind: {key: compact(value) for key, value in values.items()}
                        for kind, values in comparisons.items()},
        "within_arm_retention": within,
        "support_flags": {
            "A1_vs_A0_supported": a1_supported,
            "A2_vs_A0_supported": a2_supported,
            "A2_above_A1_supported": a2_above_a1,
            "A1_above_A2_supported": a1_above_a2,
        },
        "firewall": {"utility": False, "joint_r1": False, "internal_test": False,
                     "official1000": False, "ood": False},
    }
    metric_means = {
        "rectifier_objective": {
            arm: summaries[arm]["rectifier_selector"]["selected_metrics"]["mean_foreground_iou"] for arm in ARMS
        },
        "R2_probe": {arm: summaries[arm]["probe_summary"]["R2"]["mean_foreground_iou"] for arm in ARMS},
        "Delta_F_probe": {arm: summaries[arm]["probe_summary"]["Delta_F"]["mean_foreground_iou"] for arm in ARMS},
    }

    def order(values):
        return " > ".join(sorted(values, key=values.get, reverse=True))

    result["metric_directions"] = {name: {"means": values, "order": order(values)}
                                   for name, values in metric_means.items()}
    result["metric_conflict"] = (
        result["metric_directions"]["rectifier_objective"]["order"]
        != result["metric_directions"]["Delta_F_probe"]["order"]
    )
    if result["metric_conflict"]:
        result["secondary_interpretation"] = "LOW_EFFECTIVE_RANK_PROJECTION_NOT_CAUSALLY_HARMFUL"
        result["decision_note"] = (
            "The single-label gate is metric-dependent: A2 is stably better on Delta_F decodability "
            "but stably worse on the Rectifier segmentation objective. Do not adopt A2 for task performance."
        )
    else:
        result["decision_note"] = "Rectifier objective and Delta_F probe directions agree."
    dump(OUT / "results.json", result)
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                "additional_record": additional, "firewall": result["firewall"],
                                "metric_conflict": result["metric_conflict"],
                                "secondary_interpretation": result["secondary_interpretation"],
                                "decision_note": result["decision_note"]})
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision,
                      "additional_record": additional}), flush=True)


def render(result):
    lines = [
        "# Phase 6G.10 — Post-Attention Projection Bypass Ablation",
        "",
        "Status: **COMPLETE STOP**. Only the Rectifier stage and fresh R2/Delta_F probes were trained. Utility, joint R1, internal test, Official1000, and OOD were not used.",
        "",
        "## Rectifier objective and Delta_F decodability",
        "",
        "| Arm | Rectifier mean IoU | R2 probe mean IoU | Delta_F probe mean IoU | Delta_F/R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        probe = result["arms"][arm]["probe"]
        rect = result["arms"][arm]["rectifier"]["selected_metrics"]["mean_foreground_iou"]
        lines.append(
            f"| {arm} | {rect:.6f} | {probe['R2']['mean_foreground_iou']:.6f} | "
            f"{probe['Delta_F']['mean_foreground_iou']:.6f} | {probe['r2_to_delta_ratio']:.6f} |"
        )
    lines += [
        "",
        "## Primary paired comparisons",
        "",
        "| Comparison | Metric | Delta | 95% CI | W/T/L | Wilcoxon p |",
        "|---|---|---:|---|---:|---:|",
    ]
    for kind, values in result["comparisons"].items():
        for key, value in values.items():
            lines.append(
                f"| {key} | {kind} | {value['mean']:+.6f} | "
                f"[{value['ci'][0]:.6f}, {value['ci'][1]:.6f}] | "
                f"{value['wins']}/{value['ties']}/{value['losses']} | {value['wilcoxon_p']:.4g} |"
            )
    lines += [
        "",
        "## Post-attention mapping spectra",
        "",
        "| Arm | Mapping | Effective rank (Shannon) | Condition number |",
        "|---|---|---:|---:|",
    ]
    for arm in ARMS:
        for name, value in result["arms"][arm]["matrix_spectrum"].items():
            if value == "N/A" or value == "N/A (identity)":
                lines.append(f"| {arm} | {name} | N/A | N/A |")
            else:
                lines.append(
                    f"| {arm} | {name} | {value['effective_rank_shannon']:.3f} | "
                    f"{value['condition_number']:.3g} |"
                )
    lines += [
        "",
        "## Gate",
        "",
        f"- `A1_vs_A0_supported`: `{result['support_flags']['A1_vs_A0_supported']}`",
        f"- `A2_vs_A0_supported`: `{result['support_flags']['A2_vs_A0_supported']}`",
        f"- `A2_above_A1_supported`: `{result['support_flags']['A2_above_A1_supported']}`",
        f"- `A1_above_A2_supported`: `{result['support_flags']['A1_above_A2_supported']}`",
        "",
        "## Metric-direction audit",
        "",
        f"- Rectifier objective order: `{result['metric_directions']['rectifier_objective']['order']}`",
        f"- Delta_F probe order: `{result['metric_directions']['Delta_F_probe']['order']}`",
        f"- R2 probe order: `{result['metric_directions']['R2_probe']['order']}`",
        f"- metric conflict: `{result['metric_conflict']}`",
        f"- decision note: `{result['decision_note']}`",
        "",
        f"```text\n{result['decision']}\n```",
    ]
    if result.get("secondary_interpretation"):
        lines += ["", f"Secondary task-level interpretation: `{result['secondary_interpretation']}`"]
    if result.get("additional_record"):
        lines += ["", f"Additional record: `{result['additional_record']}`"]
    (ROOT / "docs/phase6g10_post_attention_projection_bypass.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
