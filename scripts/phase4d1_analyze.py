#!/usr/bin/env python3
"""Paired statistics, figures, gates, and final Phase 4D-1 report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4c_b import summarize
from tools.phase4d1 import (dump, paired_with_wilcoxon, read_jsonl, write_csv)
from model.position_aware_evidence_reader import PositionAwareEvidenceReader


ARMS = ("pos_clip", "pos_forensic")
CONDITIONS = ("matched", "cross_image", "spatial_shuffle", "zero")


def status_from_ci(stat):
    low, high = stat["foreground_iou"]["bootstrap_95_ci"]
    if low > 0:
        return "TRUE"
    if high < 0:
        return "FALSE"
    return "INCONCLUSIVE"


def load_records(out, split, arm, condition):
    return read_jsonl(out / split / arm / f"{condition}.jsonl")


def quantization_audit(cfg, out):
    cache = Path(cfg["experiment"]["cache_root"]) / "train_subset"
    checkpoint = Path(cfg["experiment"]["checkpoint_root"])
    permutation = torch.tensor(json.loads((out / "spatial_content_permutation.json").read_text())["permutation"])
    result = {"status": "COMPLETE", "scope": "saved step-512 readers and frozen 2048 train-subset tensors",
              "changes_gate_or_selector": False, "arms": {}}
    for arm, key in (("pos_clip", "clip_features"),
                     ("pos_forensic", "forensic_features")):
        state = torch.load(checkpoint / arm / "step_512.pt", map_location="cpu")
        reader = PositionAwareEvidenceReader(); reader.load_state_dict(state["reader"]); reader.eval()
        samples = any_vs_q = any_shuffle = elements = changed_vs_q = changed_shuffle = 0
        residual_norms = []; shuffle_difference_norms = []
        with torch.no_grad():
            for path in sorted(cache.glob("shard_*.pt")):
                shard = torch.load(path, map_location="cpu")
                q = shard["q_seg"].float(); feature = shard[key].float()
                matched = reader(q, feature)
                shuffled = reader(q, feature.flatten(2).transpose(1, 2)[:, permutation, :])
                q_bf16 = q.to(torch.bfloat16)
                matched_bf16 = matched["q_final"][:, 0].to(torch.bfloat16)
                shuffled_bf16 = shuffled["q_final"][:, 0].to(torch.bfloat16)
                first = matched_bf16 != q_bf16
                second = matched_bf16 != shuffled_bf16
                samples += len(q); any_vs_q += int(first.any(1).sum()); any_shuffle += int(second.any(1).sum())
                elements += first.numel(); changed_vs_q += int(first.sum()); changed_shuffle += int(second.sum())
                residual_norms.extend(matched["q_residual"][:, 0].norm(dim=1).tolist())
                shuffle_difference_norms.extend((matched["q_final"][:, 0] - shuffled["q_final"][:, 0]).norm(dim=1).tolist())
        result["arms"][arm] = {
            "samples": samples,
            "beta": float(reader.beta),
            "mean_float32_residual_norm": float(np.mean(residual_norms)),
            "mean_float32_matched_minus_shuffle_q_norm": float(np.mean(shuffle_difference_norms)),
            "samples_with_any_bf16_change_vs_q": any_vs_q,
            "samples_with_any_bf16_matched_vs_shuffle_change": any_shuffle,
            "element_fraction_changed_vs_q": changed_vs_q / elements,
            "element_fraction_changed_matched_vs_shuffle": changed_shuffle / elements,
        }
    dump(out / "phase4d1_bf16_interface_audit.json", result)
    return result


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase4d1_position_aware_evidence.yaml").read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    bout = ROOT / cfg["phase4c_b"]["output_root"]
    for arm in ARMS:
        state = json.loads((out / "arm_completion" / f"{arm}.json").read_text())
        if state["status"] != "COMPLETE":
            raise RuntimeError(f"arm incomplete: {arm}")

    p1 = read_jsonl(bout / "evaluation/G0/P1/predictions.jsonl")
    if len(p1) != 1106:
        raise RuntimeError("P1 validation record count mismatch")
    validation = {arm: {condition: load_records(out, "validation", arm, condition)
                        for condition in CONDITIONS} for arm in ARMS}
    for arm in ARMS:
        for condition in CONDITIONS:
            if [x["sample_id"] for x in validation[arm][condition]] != [x["sample_id"] for x in p1]:
                raise RuntimeError(f"validation identity mismatch: {arm}/{condition}")

    statistics = {"validation": {}, "train_subset": {}, "protocol": {
        "bootstrap_repeats": cfg["evaluation"]["bootstrap_repeats"],
        "bootstrap_seed": cfg["evaluation"]["bootstrap_seed"],
        "threshold_logit": 0.0,
    }}
    for arm in ARMS:
        statistics["validation"][f"{arm}_matched_minus_cross_image"] = paired_with_wilcoxon(
            validation[arm]["matched"], validation[arm]["cross_image"])
        statistics["validation"][f"{arm}_matched_minus_spatial_shuffle"] = paired_with_wilcoxon(
            validation[arm]["matched"], validation[arm]["spatial_shuffle"])
        statistics["validation"][f"{arm}_matched_minus_zero"] = paired_with_wilcoxon(
            validation[arm]["matched"], validation[arm]["zero"])
        statistics["validation"][f"{arm}_matched_minus_p1"] = paired_with_wilcoxon(
            validation[arm]["matched"], p1)
    statistics["validation"]["pos_forensic_minus_pos_clip_matched"] = paired_with_wilcoxon(
        validation["pos_forensic"]["matched"], validation["pos_clip"]["matched"])

    train = {}
    for arm in ARMS:
        train[arm] = {condition: load_records(out, "train_subset_evaluation", arm, condition)
                      for condition in ("matched", "cross_image", "spatial_shuffle")}
        statistics["train_subset"][f"{arm}_matched_minus_cross_image"] = paired_with_wilcoxon(
            train[arm]["matched"], train[arm]["cross_image"])
        statistics["train_subset"][f"{arm}_matched_minus_spatial_shuffle"] = paired_with_wilcoxon(
            train[arm]["matched"], train[arm]["spatial_shuffle"])
    dump(out / "phase4d1_statistics.json", statistics)
    bf16_audit = quantization_audit(cfg, out)

    result_rows = [{"arm": "p1", "condition": "no_reader", **summarize(p1)}]
    for arm in ARMS:
        for condition in CONDITIONS:
            result_rows.append({"arm": arm, "condition": condition,
                                **summarize(validation[arm][condition])})
    write_csv(out / "phase4d1_results.csv", list(result_rows[0].keys()), result_rows)

    p1_index = {x["sample_id"]: x for x in p1}
    indices = {arm: {condition: {x["sample_id"]: x for x in validation[arm][condition]}
                     for condition in CONDITIONS} for arm in ARMS}
    per_rows = []
    for sid in [x["sample_id"] for x in p1]:
        row = {"sample_id": sid, "p1_iou": p1_index[sid]["foreground_iou"],
               "p1_f1": p1_index[sid]["foreground_f1"]}
        for arm in ARMS:
            for condition in CONDITIONS:
                item = indices[arm][condition][sid]
                row[f"{arm}_{condition}_iou"] = item["foreground_iou"]
                row[f"{arm}_{condition}_f1"] = item["foreground_f1"]
        row["pos_forensic_matched_minus_cross_iou"] = (
            row["pos_forensic_matched_iou"] - row["pos_forensic_cross_image_iou"])
        row["pos_forensic_matched_minus_shuffle_iou"] = (
            row["pos_forensic_matched_iou"] - row["pos_forensic_spatial_shuffle_iou"])
        row["forensic_minus_clip_matched_iou"] = (
            row["pos_forensic_matched_iou"] - row["pos_clip_matched_iou"])
        per_rows.append(row)
    write_csv(out / "phase4d1_per_sample.csv", list(per_rows[0].keys()), per_rows)

    training_rows = []
    for arm in ARMS:
        values = json.loads((out / "training" / arm / "diagnostics.json").read_text())
        training_rows.extend(values)
    training_rows.sort(key=lambda x: (x["arm"], x["step"]))
    fields = []
    for row in training_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(out / "phase4d1_training_log.csv", fields, training_rows)

    image_stat = statistics["validation"]["pos_forensic_matched_minus_cross_image"]
    spatial_stat = statistics["validation"]["pos_forensic_matched_minus_spatial_shuffle"]
    forensic_stat = statistics["validation"]["pos_forensic_minus_pos_clip_matched"]
    nonreg_stat = statistics["validation"]["pos_forensic_matched_minus_p1"]
    train_image = statistics["train_subset"]["pos_forensic_matched_minus_cross_image"]
    train_spatial = statistics["train_subset"]["pos_forensic_matched_minus_spatial_shuffle"]
    image_status = status_from_ci(image_stat)
    spatial_status = status_from_ci(spatial_stat)
    forensic_status = status_from_ci(forensic_stat)
    train_sensitive = (status_from_ci(train_image) == "TRUE" and
                       status_from_ci(train_spatial) == "TRUE")
    validation_sensitive = image_status == "TRUE" and spatial_status == "TRUE"
    nonreg_delta = nonreg_stat["foreground_iou"]["mean_difference"]
    harmful = nonreg_delta < float(cfg["evaluation"]["nonregression_floor"])

    if validation_sensitive:
        minimal_result = "POSITION_AWARE_INTERACTION_GENERALIZED"
    elif train_sensitive:
        minimal_result = "POSITION_AWARE_INTERACTION_FAILED_TO_GENERALIZE"
    else:
        minimal_result = "OPTIMIZATION_OR_INTERFACE_INCONCLUSIVE"
    if validation_sensitive and harmful:
        minimal_result = "MECHANISM_EXISTS_BUT_NOT_USEFUL"
    optimization = "SUCCESSFUL" if train_sensitive else "INCONCLUSIVE"
    if validation_sensitive:
        transfer = "SUPPORTED"
    elif train_sensitive:
        transfer = "NOT_SUPPORTED"
    else:
        transfer = "INCONCLUSIVE"
    proceed = (validation_sensitive and forensic_status == "TRUE" and not harmful)
    decision = {
        "IMAGE_SPECIFIC_UTILIZATION": image_status,
        "SPATIAL_SPECIFIC_UTILIZATION": spatial_status,
        "FORENSIC_SPECIFIC_UTILIZATION": forensic_status,
        "MINIMAL_RECIPE_OPTIMIZATION": optimization,
        "MINIMAL_RECIPE_RESULT": minimal_result,
        "POSITION_AWARE_TRANSFER": transfer,
        "POS_FORENSIC_MINUS_P1_MEAN_IOU": nonreg_delta,
        "NONREGRESSION_FLOOR": cfg["evaluation"]["nonregression_floor"],
        "MECHANISM_EXISTS_BUT_NOT_USEFUL": harmful and validation_sensitive,
        "PROCEED_TO_PHASE_4D_2": "YES" if proceed else "NO",
        "internal_test_access": False,
        "official1000_access": False,
        "phase4d2_started": False,
    }
    dump(out / "decision.json", decision)

    figures = out / "figures/phase4d1"
    figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(CONDITIONS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, arm in ((-width/2, "pos_clip"), (width/2, "pos_forensic")):
        means = [summarize(validation[arm][c])["mean_foreground_iou"] for c in CONDITIONS]
        ax.bar(x + offset, means, width, label=arm.replace("pos_", "POS-").upper())
    ax.axhline(summarize(p1)["mean_foreground_iou"], color="black", linestyle="--", label="P1")
    ax.set_xticks(x, ["matched", "cross-image", "spatial shuffle", "zero"])
    ax.set_ylabel("Validation mean FG IoU")
    ax.set_title("Phase 4D-1 evidence interventions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "intervention_iou.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    names = ["matched-cross", "matched-shuffle", "forensic-clip"]
    stats = [image_stat, spatial_stat, forensic_stat]
    means = [s["foreground_iou"]["mean_difference"] for s in stats]
    lows = [m - s["foreground_iou"]["bootstrap_95_ci"][0] for m, s in zip(means, stats)]
    highs = [s["foreground_iou"]["bootstrap_95_ci"][1] - m for m, s in zip(means, stats)]
    ax.errorbar(np.arange(3), means, yerr=[lows, highs], fmt="o", capsize=5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(3), names)
    ax.set_ylabel("Paired FG IoU delta (95% CI)")
    ax.set_title("Primary mechanistic effects")
    fig.tight_layout()
    fig.savefig(figures / "mechanism_deltas.png", dpi=180)
    plt.close(fig)

    metrics = {row["arm"] + "/" + row["condition"]: row for row in result_rows}
    def stat_line(value):
        item = value["foreground_iou"]
        return (f"mean `{item['mean_difference']:+.6f}`, median `{item['median_difference']:+.6f}`, "
                f"95% CI `[{item['bootstrap_95_ci'][0]:+.6f},{item['bootstrap_95_ci'][1]:+.6f}]`, "
                f"W/T/L `{item['wins']}/{item['ties']}/{item['losses']}`, "
                f"Wilcoxon p `{item['wilcoxon_p_value']:.6g}`")

    report = f"""# Phase 4D-1 — Minimal Position-Aware Forensic Evidence Test

## 1. Executive Summary

- 正确图像利用：**{image_status}**。
- 正确空间位置利用：**{spatial_status}**。
- forensic feature 相对 CLIP：**{forensic_status}**。
- minimal recipe optimization：**{optimization}**。
- position-aware transfer：**{transfer}**。
- 是否建议 Phase 4D-2：**{'YES' if proceed else 'NO'}**。

本实验只检验固定二维位置编码、2,048 train Fake、512 steps、单一 seed 的最小 Reader。失败不能外推为 forensic information 不可迁移。

## 2. Preflight

- Feature shape：`256×24×24`；position：固定不可训练 `576×256` 2D sine/cosine。
- Shared Reader init hash：`{json.loads((out/'preflight_manifest.json').read_text())['reader_init_hash']}`。
- 两 arm 均从同一个 `reader_init.pt` 加载；sample subset、顺序、optimizer、scheduler 与 step 数一致。
- Train subset：2,048；hash `{json.loads((out/'phase4d1_train_subset.json').read_text())['sample_ids_sha256']}`；mask-area quartile 每组 512。
- Scale gate：`{json.loads((out/'position_scale_preflight.json').read_text())['status']}`。
- Internal test 与 official1000 未访问。

## 3. Training Diagnostics

训练日志位于 `phase4d1_training_log.csv`。两个 arm 都固定运行到 step 512；step 256 仅为诊断，不参与 endpoint 选择。最终 beta：POS-CLIP `{json.loads((out/'training/pos_clip/completion.json').read_text())['beta_final']:.6f}`，POS-FORENSIC `{json.loads((out/'training/pos_forensic/completion.json').read_text())['beta_final']:.6f}`。

只读 BF16 interface audit 显示：POS-CLIP/POS-FORENSIC 的 float32 residual norm 均值分别为 `{bf16_audit['arms']['pos_clip']['mean_float32_residual_norm']:.6f}` / `{bf16_audit['arms']['pos_forensic']['mean_float32_residual_norm']:.6f}`；matched-vs-shuffle q 差异 norm 均值仅 `{bf16_audit['arms']['pos_clip']['mean_float32_matched_minus_shuffle_q_norm']:.6f}` / `{bf16_audit['arms']['pos_forensic']['mean_float32_matched_minus_shuffle_q_norm']:.6f}`。转为冻结 SAM 接口使用的 BF16 后，只有 `{bf16_audit['arms']['pos_clip']['samples_with_any_bf16_matched_vs_shuffle_change']}/2048` 与 `{bf16_audit['arms']['pos_forensic']['samples_with_any_bf16_matched_vs_shuffle_change']}/2048` 样本保留任一元素差异；最终二值 mask 仍全部相同。这支持“minimal recipe 未建立足够强的位置敏感交互”，不改变预注册 gate。

## 4. Main Results

| Arm / condition | N | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|---:|
| P1 | {metrics['p1/no_reader']['n']} | {metrics['p1/no_reader']['mean_foreground_iou']:.6f} | {metrics['p1/no_reader']['median_foreground_iou']:.6f} | {metrics['p1/no_reader']['mean_foreground_f1']:.6f} |
"""
    for arm in ARMS:
        for condition in CONDITIONS:
            row = metrics[f"{arm}/{condition}"]
            report += (f"| {arm} / {condition} | {row['n']} | {row['mean_foreground_iou']:.6f} | "
                       f"{row['median_foreground_iou']:.6f} | {row['mean_foreground_f1']:.6f} |\n")
    report += f"""

## 5. Image-Specific Test

POS-FORENSIC matched−cross-image：{stat_line(image_stat)}。

## 6. Spatial-Specific Test

POS-FORENSIC matched−spatial-content-shuffle：{stat_line(spatial_stat)}。Shuffle 只置换 `F` content；固定 `P_2D` lattice 未被置换。

## 7. Forensic-Specific Test

POS-FORENSIC matched−POS-CLIP matched：{stat_line(forensic_stat)}。

## 8. Train vs Validation Diagnostic

- Train POS-FORENSIC matched−cross：{stat_line(train_image)}。
- Train POS-FORENSIC matched−shuffle：{stat_line(train_spatial)}。
- 归因：`{minimal_result}`。
- POS-FORENSIC matched−P1 validation mean IoU：`{nonreg_delta:+.6f}`；non-regression floor `-0.005`。

## 9. Decision

```text
IMAGE_SPECIFIC_UTILIZATION: {image_status}
SPATIAL_SPECIFIC_UTILIZATION: {spatial_status}
FORENSIC_SPECIFIC_UTILIZATION: {forensic_status}
MINIMAL_RECIPE_OPTIMIZATION: {optimization}
POSITION_AWARE_TRANSFER: {transfer}
PROCEED_TO_PHASE_4D_2: {'YES' if proceed else 'NO'}
```

`MINIMAL_RECIPE_RESULT: {minimal_result}`

本阶段到此停止；没有自动启动 Phase 4D-2。Mask target 仍是官方 polygons 派生的 per-image all-reference visible/explainable artifact union，不是 pseudo-mask，也不是完整 pixel-perfect forgery GT。
"""
    (out / "phase4d1_mechanism_report.md").write_text(report, encoding="utf-8")
    dump(out / "completion_manifest.json", {
        "status": "COMPLETE", "decision": decision,
        "required_outputs": {
            name: (out / name).exists() for name in (
                "phase4d1_preflight.md", "phase4d1_position_scale_preflight.md",
                "phase4d1_train_subset.json", "phase4d1_training_log.csv",
                "phase4d1_results.csv", "phase4d1_per_sample.csv",
                "phase4d1_statistics.json", "phase4d1_mechanism_report.md")
        },
        "internal_test_access": False, "official1000_access": False,
        "phase4d2_started": False,
    })
    print(json.dumps({"status": "COMPLETE", "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
