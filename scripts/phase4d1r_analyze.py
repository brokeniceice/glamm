#!/usr/bin/env python3
"""Finalize paired statistics and the Phase 4D-1R mechanism decision."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4c_b import summarize
from tools.phase4d1 import dump, paired_with_wilcoxon, read_jsonl, write_csv

ARMS = ("pos_clip", "pos_forensic")
CONDITIONS = ("matched", "cross_image", "spatial_shuffle", "zero")


def status(stat):
    low, high = stat["foreground_iou"]["bootstrap_95_ci"]
    if low > 0:
        return "TRUE"
    if high < 0:
        return "FALSE"
    return "INCONCLUSIVE"


def stat_line(stat):
    x = stat["foreground_iou"]
    return (f"mean `{x['mean_difference']:+.6f}`, median `{x['median_difference']:+.6f}`, "
            f"95% CI `[{x['bootstrap_95_ci'][0]:+.6f}, {x['bootstrap_95_ci'][1]:+.6f}]`, "
            f"W/T/L `{x['wins']}/{x['ties']}/{x['losses']}`, Wilcoxon p `{x['wilcoxon_p_value']:.6g}`")


def records(out, split, arm, condition):
    return read_jsonl(out / split / arm / f"{condition}.jsonl")


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase4d1r_corrected_position_aware.yaml").read_text())
    parent_cfg = yaml.safe_load((ROOT / cfg["parent_phase4d1"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    bout = ROOT / parent_cfg["phase4c_b"]["output_root"]
    for arm in ARMS:
        state = json.loads((out / "arm_completion" / f"{arm}.json").read_text())
        if state["status"] != "COMPLETE" or state["formal_step"] != 512:
            raise RuntimeError(f"incomplete arm: {arm}")

    p1 = read_jsonl(bout / "evaluation/G0/P1/predictions.jsonl")
    if len(p1) != 1106:
        raise RuntimeError("P1 validation baseline mismatch")
    val = {arm: {condition: records(out, "validation", arm, condition)
                 for condition in CONDITIONS} for arm in ARMS}
    ids = [x["sample_id"] for x in p1]
    for arm in ARMS:
        for condition in CONDITIONS:
            if [x["sample_id"] for x in val[arm][condition]] != ids:
                raise RuntimeError(f"identity/order mismatch: {arm}/{condition}")
    train = {arm: {condition: records(out, "train_subset_evaluation", arm, condition)
                   for condition in CONDITIONS} for arm in ARMS}

    stats = {"protocol": {"formal_endpoint": 512, "selector": "none",
             "bootstrap_repeats": 10000, "bootstrap_seed": 3407,
             "threshold_logit": 0.0, "validation_n": 1106},
             "validation": {}, "train_subset": {}, "train_signal": {}}
    comparisons = {
        "pos_forensic_matched_minus_cross_image": (val["pos_forensic"]["matched"], val["pos_forensic"]["cross_image"]),
        "pos_forensic_matched_minus_spatial_shuffle": (val["pos_forensic"]["matched"], val["pos_forensic"]["spatial_shuffle"]),
        "pos_forensic_matched_minus_zero": (val["pos_forensic"]["matched"], val["pos_forensic"]["zero"]),
        "pos_forensic_minus_pos_clip_matched": (val["pos_forensic"]["matched"], val["pos_clip"]["matched"]),
        "pos_forensic_matched_minus_p1": (val["pos_forensic"]["matched"], p1),
        "pos_clip_matched_minus_cross_image": (val["pos_clip"]["matched"], val["pos_clip"]["cross_image"]),
        "pos_clip_matched_minus_spatial_shuffle": (val["pos_clip"]["matched"], val["pos_clip"]["spatial_shuffle"]),
    }
    for name, pair in comparisons.items():
        stats["validation"][name] = paired_with_wilcoxon(*pair)
    for arm in ARMS:
        for condition in ("cross_image", "spatial_shuffle", "zero"):
            name = f"{arm}_matched_minus_{condition}"
            stats["train_subset"][name] = paired_with_wilcoxon(train[arm]["matched"], train[arm][condition])
        stats["train_signal"][arm] = json.loads((out / "train_subset_evaluation" / arm / "signal_summary.json").read_text())

    result_rows = [{"arm": "p1", "condition": "no_reader", **summarize(p1)}]
    for arm in ARMS:
        for condition in CONDITIONS:
            result_rows.append({"arm": arm, "condition": condition, **summarize(val[arm][condition])})
    write_csv(out / "phase4d1r_results.csv", list(result_rows[0]), result_rows)

    indices = {arm: {c: {x["sample_id"]: x for x in val[arm][c]} for c in CONDITIONS} for arm in ARMS}
    p1i = {x["sample_id"]: x for x in p1}; per_rows = []
    for sid in ids:
        row = {"sample_id": sid, "p1_iou": p1i[sid]["foreground_iou"], "p1_f1": p1i[sid]["foreground_f1"]}
        for arm in ARMS:
            for condition in CONDITIONS:
                item = indices[arm][condition][sid]
                row[f"{arm}_{condition}_iou"] = item["foreground_iou"]
                row[f"{arm}_{condition}_f1"] = item["foreground_f1"]
        row["pos_forensic_matched_minus_cross_iou"] = row["pos_forensic_matched_iou"] - row["pos_forensic_cross_image_iou"]
        row["pos_forensic_matched_minus_shuffle_iou"] = row["pos_forensic_matched_iou"] - row["pos_forensic_spatial_shuffle_iou"]
        row["forensic_minus_clip_matched_iou"] = row["pos_forensic_matched_iou"] - row["pos_clip_matched_iou"]
        row["pos_forensic_matched_minus_p1_iou"] = row["pos_forensic_matched_iou"] - row["p1_iou"]
        per_rows.append(row)
    write_csv(out / "phase4d1r_per_sample.csv", list(per_rows[0]), per_rows)

    training_rows = []
    for arm in ARMS:
        training_rows.extend(json.loads((out / "training" / arm / "diagnostics.json").read_text()))
    training_rows.sort(key=lambda x: (x["arm"], x["step"]))
    fields = []
    for row in training_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(out / "phase4d1r_training_log.csv", fields, training_rows)

    signal_rows = []
    signal_fields = ["record_type", "arm", "step", "comparison", "beta",
                     "matched_shuffle_fp32_q_difference", "matched_shuffle_bf16_survival",
                     "mean_sam_logit_absolute_difference", "effective_residual_norm",
                     "effective_residual_over_query_norm"]
    for row in training_rows:
        signal_rows.append({"record_type": "fixed_minibatch_training", "arm": row["arm"],
            "step": row["step"], "comparison": "matched_vs_spatial_shuffle", "beta": row["beta"],
            "matched_shuffle_fp32_q_difference": row["matched_shuffle_fp32_q_difference"],
            "matched_shuffle_bf16_survival": row["matched_shuffle_bf16_survival"],
            "mean_sam_logit_absolute_difference": "", "effective_residual_norm": row["effective_residual_norm"],
            "effective_residual_over_query_norm": row["effective_residual_over_query_norm"]})
    for arm in ARMS:
        beta = json.loads((out / "training" / arm / "completion.json").read_text())["beta_final"]
        for comparison, value in stats["train_signal"][arm].items():
            signal_rows.append({"record_type": "full_train_endpoint", "arm": arm, "step": 512,
                "comparison": comparison, "beta": beta, "matched_shuffle_fp32_q_difference": "",
                "matched_shuffle_bf16_survival": value["bf16_survival"],
                "mean_sam_logit_absolute_difference": value["mean_sam_logit_absolute_difference"],
                "effective_residual_norm": "", "effective_residual_over_query_norm": ""})
    write_csv(out / "phase4d1r_signal_diagnostics.csv", signal_fields, signal_rows)

    image_stat = stats["validation"]["pos_forensic_matched_minus_cross_image"]
    spatial_stat = stats["validation"]["pos_forensic_matched_minus_spatial_shuffle"]
    forensic_stat = stats["validation"]["pos_forensic_minus_pos_clip_matched"]
    nonreg_stat = stats["validation"]["pos_forensic_matched_minus_p1"]
    train_image = stats["train_subset"]["pos_forensic_matched_minus_cross_image"]
    train_spatial = stats["train_subset"]["pos_forensic_matched_minus_spatial_shuffle"]
    image_status, spatial_status, forensic_status = status(image_stat), status(spatial_stat), status(forensic_stat)
    grad_visible = all(float(x["reader_body_grad_norm"]) > 1e-8 for x in training_rows if int(x["step"]) > 0)
    bf16_visible = all(any(float(x["matched_shuffle_bf16_survival"]) > 0 for x in training_rows if x["arm"] == arm) for arm in ARMS)
    optimization = "HEALTHY" if grad_visible and bf16_visible else ("PARTIAL" if grad_visible else "FAILED")
    train_sensitive = status(train_image) == "TRUE" and status(train_spatial) == "TRUE"
    validation_sensitive = image_status == "TRUE" and spatial_status == "TRUE"
    if validation_sensitive:
        position_transfer = "SUPPORTED"
    elif optimization == "HEALTHY":
        position_transfer = "NOT_SUPPORTED"
    else:
        position_transfer = "INCONCLUSIVE"
    forensic_transfer = "SUPPORTED" if forensic_status == "TRUE" else ("NOT_SUPPORTED" if forensic_status == "FALSE" else "INCONCLUSIVE")
    nonreg_delta = nonreg_stat["foreground_iou"]["mean_difference"]
    harmful = nonreg_delta < float(cfg["evaluation"]["nonregression_floor"])
    proceed = validation_sensitive and forensic_status == "TRUE" and not harmful
    if validation_sensitive and train_sensitive:
        attribution = "POSITION_AWARE_INTERACTION_LEARNED_AND_GENERALIZED"
    elif train_sensitive:
        attribution = "LEARNED_BUT_FAILED_TO_GENERALIZE"
    elif optimization == "HEALTHY":
        attribution = "CORRECTED_READER_DID_NOT_LEARN_POSITION_SENSITIVE_USE"
    else:
        attribution = "OPTIMIZATION_REMAINS_UNRESOLVED"
    decision = {
        "IMAGE_SPECIFIC_UTILIZATION": image_status,
        "SPATIAL_SPECIFIC_UTILIZATION": spatial_status,
        "FORENSIC_SPECIFIC_UTILIZATION": forensic_status,
        "READER_OPTIMIZATION": optimization,
        "POSITION_AWARE_TRANSFER": position_transfer,
        "FORENSIC_TRANSFER": forensic_transfer,
        "TRAIN_TO_VALIDATION_ATTRIBUTION": attribution,
        "POS_FORENSIC_MINUS_P1_MEAN_IOU": nonreg_delta,
        "MECHANISM_EXISTS_BUT_NOT_USEFUL": bool(harmful and validation_sensitive),
        "PROCEED_TO_PHASE_4D_2": "YES" if proceed else "NO",
        "internal_test_access": False, "official1000_access": False, "phase4d2_started": False,
    }
    stats["decision"] = decision
    dump(out / "phase4d1r_statistics.json", stats); dump(out / "decision.json", decision)

    figures = out / "figures/phase4d1r"; figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(4); width = .34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, arm in ((-width/2, "pos_clip"), (width/2, "pos_forensic")):
        means = [summarize(val[arm][c])["mean_foreground_iou"] for c in CONDITIONS]
        ax.bar(x + offset, means, width, label=arm)
    ax.axhline(summarize(p1)["mean_foreground_iou"], color="black", linestyle="--", label="P1")
    ax.set_xticks(x, CONDITIONS); ax.set_ylabel("Validation mean FG IoU"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "intervention_iou.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    chosen = [image_stat, spatial_stat, forensic_stat]; names = ["matched-cross", "matched-shuffle", "forensic-clip"]
    means = [s["foreground_iou"]["mean_difference"] for s in chosen]
    lows = [m-s["foreground_iou"]["bootstrap_95_ci"][0] for m,s in zip(means,chosen)]
    highs = [s["foreground_iou"]["bootstrap_95_ci"][1]-m for m,s in zip(means,chosen)]
    ax.errorbar(np.arange(3), means, yerr=[lows, highs], fmt="o", capsize=5); ax.axhline(0,color="black")
    ax.set_xticks(np.arange(3), names); ax.set_ylabel("Paired FG IoU delta (95% CI)"); fig.tight_layout()
    fig.savefig(figures / "mechanism_deltas.png", dpi=180); plt.close(fig)

    metrics = {f"{x['arm']}/{x['condition']}": x for x in result_rows}
    preflight = json.loads((out / "preflight_manifest.json").read_text())
    report = f"""# Phase 4D-1R — Corrected Minimal Position-Aware Evidence Rerun

## 1. Executive Summary

- Image-specific utilization: **{image_status}**.
- Spatial-specific utilization: **{spatial_status}**.
- Forensic-specific utilization: **{forensic_status}**.
- Reader optimization: **{optimization}**.
- Position-aware transfer: **{position_transfer}**.
- Phase 4D-2 proposal gate: **{'YES' if proceed else 'NO'}**; Phase 4D-2 was not started.

## 2. Single-Variable Protocol Verification

Only beta initialization changed from `0.0` to `0.03`. The value came from the preregistered Phase 4D-1B numerical viability rule, not performance selection. Both arms share corrected init hash `{preflight['corrected_reader_init_hash']}`, frozen train order `{preflight['train_subset_hash']}`, optimizer, scheduler, loss, batch size, 512-step endpoint, evaluator, threshold 0, mappings, and positional encoding.

## 3. Beta / Gradient Diagnostics

Both arms recorded steps 0/64/128/256/384/512 in `phase4d1r_training_log.csv`, including Reader-body and Q/K/V/output projection gradients. Optimization status is `{optimization}`. Final beta values are POS-CLIP `{json.loads((out/'training/pos_clip/completion.json').read_text())['beta_final']:.8f}` and POS-FORENSIC `{json.loads((out/'training/pos_forensic/completion.json').read_text())['beta_final']:.8f}`.

## 4. Train Mechanism Diagnostics

- POS-FORENSIC train matched-cross: {stat_line(train_image)}.
- POS-FORENSIC train matched-shuffle: {stat_line(train_spatial)}.
- Attribution: `{attribution}`.
- Full-subset BF16 survival and SAM continuous-logit differences are in `phase4d1r_signal_diagnostics.csv`.

## 5. Validation Main Results

| Arm / condition | N | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|---:|
| P1 / no Reader | {metrics['p1/no_reader']['n']} | {metrics['p1/no_reader']['mean_foreground_iou']:.6f} | {metrics['p1/no_reader']['median_foreground_iou']:.6f} | {metrics['p1/no_reader']['mean_foreground_f1']:.6f} |
"""
    for arm in ARMS:
        for condition in CONDITIONS:
            x = metrics[f"{arm}/{condition}"]
            report += f"| {arm} / {condition} | {x['n']} | {x['mean_foreground_iou']:.6f} | {x['median_foreground_iou']:.6f} | {x['mean_foreground_f1']:.6f} |\n"
    report += f"""

## 6. Matched vs Cross

POS-FORENSIC-R matched-cross: {stat_line(image_stat)}.

## 7. Matched vs Shuffle

POS-FORENSIC-R matched-spatial-shuffle: {stat_line(spatial_stat)}. The intervention permutes `F` content before adding the fixed positional lattice; `(F+P)` is never permuted jointly.

## 8. Forensic vs CLIP

POS-FORENSIC-R matched minus POS-CLIP-R matched: {stat_line(forensic_stat)}.

## 9. P1 Non-regression

POS-FORENSIC-R matched minus P1 G0: {stat_line(nonreg_stat)}. Mean delta `{nonreg_delta:+.6f}` against the `-0.005` floor.

## 10. Train-to-Validation Generalization

`{attribution}`. A failure here is limited to this corrected minimal Reader and cannot establish that forensic information is impossible to transfer.

## 11. Final Decision

```text
IMAGE_SPECIFIC_UTILIZATION: {image_status}
SPATIAL_SPECIFIC_UTILIZATION: {spatial_status}
FORENSIC_SPECIFIC_UTILIZATION: {forensic_status}
READER_OPTIMIZATION: {optimization}
POSITION_AWARE_TRANSFER: {position_transfer}
FORENSIC_TRANSFER: {forensic_transfer}
PROCEED_TO_PHASE_4D_2: {'YES' if proceed else 'NO'}
```

The formal endpoint is step512 with no checkpoint selection. Internal test and official1000 remained sealed. Phase 4D-2 was not started.
"""
    (out / "phase4d1r_mechanism_report.md").write_text(report, encoding="utf-8")
    required = ("phase4d1r_preflight.md", "phase4d1r_training_log.csv", "phase4d1r_results.csv",
                "phase4d1r_per_sample.csv", "phase4d1r_statistics.json",
                "phase4d1r_signal_diagnostics.csv", "phase4d1r_mechanism_report.md")
    dump(out / "completion_manifest.json", {"status": "COMPLETE", "decision": decision,
         "required_outputs": {x: (out/x).exists() for x in required}, "formal_endpoint": 512,
         "selector_used": False, "internal_test_access": False, "official1000_access": False,
         "phase4d2_started": False})
    print(json.dumps({"status": "COMPLETE", "decision": decision}, indent=2))


if __name__ == "__main__":
    main()

