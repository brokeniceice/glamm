#!/usr/bin/env python3
"""Finalize Phase 3D.2 statistics, audits, failure analysis, gates, and report."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256
from scripts.phase3d0_analyze import paired
from scripts.phase3d1_finalize import bootstrap_summary
from scripts.phase3d1_validate_select import semantic_summary
from tools.phase3d0r import FrozenSentenceEncoder


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def index(values): return {row["sample_id"]: row for row in values}


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase3d2_direct_spatial_path.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    selector = load(root / "evaluation/selector/selector.json")
    selected_step = int(selector["optimizer_step"])
    p1_tf = rows(root / "evaluation/tf_phrase/step_0000/tf_full_context/predictions.jsonl")
    spatial_tf = rows(root / f"evaluation/tf_phrase/step_{selected_step:04d}/tf_full_context/predictions.jsonl")
    p1_g0 = rows(root / "evaluation/g0/P1_FROZEN/G0/predictions.jsonl")
    spatial_g0 = rows(root / "evaluation/g0/P3D2_SPATIAL/G0/predictions.jsonl")
    p1_det = rows(root / "evaluation/g0/P1_FROZEN/detection/predictions.jsonl")
    spatial_det = rows(root / "evaluation/g0/P3D2_SPATIAL/detection/predictions.jsonl")
    pt, st, pg, sg = map(index, (p1_tf, spatial_tf, p1_g0, spatial_g0))
    fake_ids = sorted(set(pt) & set(st) & set(pg) & set(sg))
    if len(fake_ids) != len(p1_tf) or not fake_ids:
        raise RuntimeError("paired Fake population mismatch")

    contrasts = {
        "TF_PHRASE": {
            "foreground_iou": paired([st[x]["foreground_iou"] for x in fake_ids], [pt[x]["foreground_iou"] for x in fake_ids]),
            "foreground_f1": paired([st[x]["foreground_f1"] for x in fake_ids], [pt[x]["foreground_f1"] for x in fake_ids]),
        },
        "G0": {
            "foreground_iou": paired([sg[x]["foreground_iou"] for x in fake_ids], [pg[x]["foreground_iou"] for x in fake_ids]),
            "foreground_f1": paired([sg[x]["foreground_f1"] for x in fake_ids], [pg[x]["foreground_f1"] for x in fake_ids]),
        },
    }
    summaries = {
        "P1-FROZEN": {
            "TF_PHRASE": {k: bootstrap_summary([row[k] for row in p1_tf]) for k in ("foreground_iou", "foreground_f1")},
            "G0": {k: bootstrap_summary([row[k] for row in p1_g0]) for k in ("foreground_iou", "foreground_f1")},
        },
        "P3D2-SPATIAL": {
            "TF_PHRASE": {k: bootstrap_summary([row[k] for row in spatial_tf]) for k in ("foreground_iou", "foreground_f1")},
            "G0": {k: bootstrap_summary([row[k] for row in spatial_g0]) for k in ("foreground_iou", "foreground_f1")},
        },
    }
    p1_detection = load(root / "evaluation/g0/P1_FROZEN/summary.json")["modes"]["detection"]["classification_head"]
    spatial_detection = load(root / "evaluation/g0/P3D2_SPATIAL/summary.json")["modes"]["detection"]["classification_head"]
    summaries["P1-FROZEN"]["classification"] = p1_detection
    summaries["P3D2-SPATIAL"]["classification"] = spatial_detection

    val_rows = rows(ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl")
    qcfg = yaml.safe_load((ROOT / "configs/phase3d0r_reward_reformulation.yaml").read_text())
    enc = qcfg["semantic_encoder"]
    encoder = FrozenSentenceEncoder(enc["model_id"], enc["revision"], enc["cache_dir"], "cpu")
    idf = load(ROOT / "outputs/phase3d0r_reward_reformulation/token_idf.json")["values"]
    p1_sem, p1_detail = semantic_summary(p1_g0, val_rows, encoder, idf)
    spatial_sem, spatial_detail = semantic_summary(spatial_g0, val_rows, encoder, idf)
    summaries["P1-FROZEN"]["G0_behavior"] = p1_sem
    summaries["P3D2-SPATIAL"]["G0_behavior"] = spatial_sem
    dump(root / "evaluation/g0/P1_FROZEN/semantic_structure.json", {"summary": p1_sem, "records": p1_detail})
    dump(root / "evaluation/g0/P3D2_SPATIAL/semantic_structure.json", {"summary": spatial_sem, "records": spatial_detail})

    # Full-population language behavior must also be exact; the 16-sample audit separately checks raw cls logits.
    exact_fields = ("generated_token_ids", "decoded_text", "generated_localization_phrase", "seg_position")
    full_exact = {
        field: all(pg[sid].get(field) == sg[sid].get(field) for sid in fake_ids) for field in exact_fields
    }
    pd, sd = index(p1_det), index(spatial_det); detection_ids = sorted(set(pd) & set(sd))
    full_exact["classification_prediction"] = all(pd[sid].get("cls_pred") == sd[sid].get("cls_pred") for sid in detection_ids)
    full_exact["classification_probability"] = all(pd[sid].get("cls_prob_fake") == sd[sid].get("cls_prob_fake") for sid in detection_ids)
    behavioral = load(root / "behavioral_invariance_audit.json")
    behavioral["full_validation_exact"] = full_exact
    behavioral["full_validation_fake_count"] = len(fake_ids)
    behavioral["full_validation_detection_count"] = len(detection_ids)
    if not all(full_exact.values()):
        behavioral["status"] = "FAIL"; dump(root / "behavioral_invariance_audit.json", behavioral)
        raise RuntimeError("full-validation behavioral invariance failed")
    dump(root / "behavioral_invariance_audit.json", behavioral)

    shutil.copy2(root / "training/parameter_invariance_audit.json", root / "parameter_invariance_audit.json")
    shutil.copy2(root / "training/gradient_audit.json", root / "gradient_audit.json")
    parameter_audit = load(root / "parameter_invariance_audit.json")
    gradient_audit = load(root / "gradient_audit.json")
    if parameter_audit["status"] != "PASS" or gradient_audit["status"] != "PASS" or behavioral["status"] != "PASS":
        raise RuntimeError("mandatory invariance/gradient audit did not pass")

    tf_iou = contrasts["TF_PHRASE"]["foreground_iou"]
    g0_iou = contrasts["G0"]["foreground_iou"]
    gate_a = tf_iou["mean_difference"] > 0 and tf_iou["bootstrap_95ci"][0] > 0
    gate_b = gate_a and g0_iou["mean_difference"] > 0 and g0_iou["bootstrap_95ci"][0] > 0
    if gate_a and gate_b:
        primary = "GATE_SPATIAL_GAIN_TRANSFERS_TO_G0"
    elif gate_a:
        primary = "GATE_SPATIAL_PATH_TRAINABLE_BUT_GAIN_NOT_TRANSFERRED_TO_G0"
    elif g0_iou["mean_difference"] > 0 and g0_iou["bootstrap_95ci"][0] > 0:
        primary = "GATE_UNEXPECTED_G0_ONLY_RESULT_REQUIRES_AUDIT"
    else:
        primary = "GATE_EXISTING_SPATIAL_PATH_OPTIMIZATION_INSUFFICIENT"
    route = {
        "primary_gate": primary,
        "gate_A": "GATE_SPATIAL_PATH_TRAINABLE" if gate_a else "GATE_SPATIAL_PATH_TRAINABILITY_NOT_SUPPORTED",
        "gate_B": "GATE_SPATIAL_GAIN_TRANSFERS_TO_G0" if gate_b else "GATE_SPATIAL_GAIN_DOES_NOT_SIGNIFICANTLY_TRANSFER_TO_G0",
        "conditions": {"tf_iou_positive": tf_iou["mean_difference"] > 0, "tf_iou_ci_lower_gt_zero": tf_iou["bootstrap_95ci"][0] > 0,
                       "g0_iou_positive": g0_iou["mean_difference"] > 0, "g0_iou_ci_lower_gt_zero": g0_iou["bootstrap_95ci"][0] > 0},
        "automatic_next_phase_started": False, "internal_test_used": False, "official1000_used": False,
    }
    dump(root / "route_gate.json", route)
    dump(root / "paired_bootstrap.json", contrasts)
    final_metrics = {"selected": selector, "arms": summaries, "P3D2_minus_P1": contrasts}
    dump(root / "final_metrics.json", final_metrics); dump(root / "statistics/final_metrics.json", final_metrics)

    cases = {"TF_and_G0_both_improve": [], "TF_improves_G0_does_not": [], "both_remain_poor": []}
    for sid in fake_ids:
        row = {"sample_id": sid, "P1_TF_IoU": pt[sid]["foreground_iou"], "SPATIAL_TF_IoU": st[sid]["foreground_iou"],
               "TF_delta": st[sid]["foreground_iou"] - pt[sid]["foreground_iou"],
               "P1_G0_IoU": pg[sid]["foreground_iou"], "SPATIAL_G0_IoU": sg[sid]["foreground_iou"],
               "G0_delta": sg[sid]["foreground_iou"] - pg[sid]["foreground_iou"],
               "generated_text": pg[sid].get("decoded_text")}
        if row["TF_delta"] > 0 and row["G0_delta"] > 0: cases["TF_and_G0_both_improve"].append(row)
        if row["TF_delta"] > 0 and row["G0_delta"] <= 0: cases["TF_improves_G0_does_not"].append(row)
        if row["SPATIAL_TF_IoU"] <= .20 and row["SPATIAL_G0_IoU"] <= .20: cases["both_remain_poor"].append(row)
    failure = {key: {"count": len(value), "examples": sorted(value, key=lambda x: (-abs(x["TF_delta"]), x["sample_id"]))[:100]}
               for key, value in cases.items()}
    failure["interpretation_boundary"] = "Case categories are descriptive; they do not prove an architecture bottleneck."
    dump(root / "failure_analysis.json", failure)

    checkpoint_rows = [{"optimizer_step": 0, "checkpoint": cfg["source"]["checkpoint"], "source_P1": True}]
    checkpoint_rows.extend(rows(root / "checkpoint_metadata.jsonl"))
    dump(root / "checkpoint_metadata.json", checkpoint_rows)
    f = lambda value: f"{value:.6f}"
    candidate_base = selector["candidates"][0]
    candidate_table = "\n".join(
        f"| {row['optimizer_step']} | {row['mean_foreground_iou']:.6f} | "
        f"{row['mean_foreground_iou'] - candidate_base['mean_foreground_iou']:+.6f} | "
        f"{row['mean_foreground_f1']:.6f} | "
        f"{row['mean_foreground_f1'] - candidate_base['mean_foreground_f1']:+.6f} |"
        for row in selector["candidates"]
    )
    trained_candidates = [row for row in selector["candidates"] if int(row["optimizer_step"]) > 0]
    best_trained = max(
        trained_candidates,
        key=lambda row: (row["mean_foreground_iou"], row["mean_foreground_f1"], -row["optimizer_step"]),
    )
    initialization = load(root / "training/initialization_audit.json")
    training_summary = load(root / "training/run_summary.json")
    mask_audit = next(row for row in parameter_audit["modules"] if row["module_name"] == "mask_decoder")
    text_fc_audit = next(row for row in parameter_audit["modules"] if row["module_name"] == "text_hidden_fcs")
    behavior_p1 = summaries["P1-FROZEN"]["G0_behavior"]
    report = f"""# Phase 3D.2 — Direct Spatial-Path Optimization

## 1. 研究问题与冻结边界

Phase 3D.1 已以 `GATE_REWARD_FORMULATION_NOT_PRIMARY_BOTTLENECK` 结束，因此本阶段不继续 R3/Q2 reward tuning，也不混入 reward、语言交叉熵、分类损失或新 forensic branch。Phase 3D.2 只回答：在语言策略、图像表征和模型结构保持不变时，直接训练现有 spatial grounding pathway，能否显著改善 forensic localization？

所有实验从原始 P1 checkpoint 初始化，SHA256 为 `{cfg['source']['checkpoint_sha256']}`。未使用 R3/Q2、Phase 3D.0 rollout 或其他微调 checkpoint。internal test 和 official1000 全程封存，未参与训练、选择或解释。

## 2. 为什么使用 authoritative context

训练采用 teacher-forced authoritative forensic target：`[FAKE] explanation` 加 `Target regions: <authoritative phrase> [SEG]`。冻结 P1 LLM 产生 `[SEG]` hidden state，随后仅由 `text_hidden_fcs` 和 `mask_decoder` 预测 mask。这样可以隔离“给定正确语言证据时空间路径是否可改善”。本阶段明确不使用 P1 自主生成文本再回放监督的 generated replay，避免把语言错误和空间优化混在同一干预中。

## 3. 数据、目标与训练配置

- 训练 split：统一 forensic train，共 17,672 张图，其中 Fake 8,836 张；primary spatial loss 仅使用带非空定位目标的 Fake。
- 实际暴露：1,000 个按冻结 schedule 选定的 Fake image exposures；250 optimizer steps，batch size 4，gradient accumulation 1。
- target：冻结的 SynthScars annotation polygons，经既有 per-image all-ref union、rasterization 和 preprocessing 构造；未重新生成 mask，也未使用 pseudo mask。
- optimizer：AdamW，学习率 `3e-4`，betas `(0.9, 0.95)`，weight decay 0；10-step warmup 后线性衰减；bf16；gradient clip 1.0。
- 唯一目标：direct mask supervision，BCE 权重 2.0、Dice 权重 0.5；reward、language CE、classification loss 均为 0 或 null。
- 训练状态：`{training_summary['status']}`，耗时 {training_summary['elapsed_seconds']:.1f} 秒，完成 {training_summary['total_optimizer_steps']} steps。

## 4. 可训练参数与冻结参数

总参数量 `{initialization['total_parameters']:,}`，实际可训练参数 `{initialization['trainable_parameters']:,}`：

- `text_hidden_fcs`：17,830,144 参数；
- `mask_decoder`：4,058,340 参数。

LLM base、全部 LoRA、token embedding、LM head、classification head、vision tower、grounding image encoder、mm projector、region encoder 及其他 grounding encoder 均设置为 `requires_grad=False`，且未注册进 optimizer。

## 5. 冻结的 checkpoint selector

Primary selector 为 internal-validation Fake TF-PHRASE mean foreground IoU，tie-breaker 为 mean foreground F1。候选为 step {'/'.join(map(str, cfg['selector']['candidate_steps']))}；G0、training loss、internal test、official1000 和 qualitative case 均未参与选择。

| Step | TF FG IoU | IoU vs P1 | TF FG F1 | F1 vs P1 |
|---:|---:|---:|---:|---:|
{candidate_table}

所有实际训练 checkpoint 都低于 step 0。训练 checkpoint 中最好的是 step {best_trained['optimizer_step']}，TF FG IoU 为 {best_trained['mean_foreground_iou']:.6f}，相对 P1 为 {best_trained['mean_foreground_iou'] - candidate_base['mean_foreground_iou']:+.6f}。因此 selector 按预注册规则选择 **step {selected_step}**，即原始 P1，而不是训练后的 checkpoint。这是负结果，不是训练后模型与 P1 达到相同结果。

## 6. TF-PHRASE mechanistic result

| Path | P1 FG IoU | Spatial FG IoU | Delta | 95% CI | P1 FG F1 | Spatial FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-PHRASE | {f(summaries['P1-FROZEN']['TF_PHRASE']['foreground_iou']['mean'])} | {f(summaries['P3D2-SPATIAL']['TF_PHRASE']['foreground_iou']['mean'])} | {tf_iou['mean_difference']:+.6f} | [{tf_iou['bootstrap_95ci'][0]:+.6f}, {tf_iou['bootstrap_95ci'][1]:+.6f}] | {f(summaries['P1-FROZEN']['TF_PHRASE']['foreground_f1']['mean'])} | {f(summaries['P3D2-SPATIAL']['TF_PHRASE']['foreground_f1']['mean'])} |

这里 P3D2-SPATIAL 指 selector 选中的正式 arm；由于 selector 回退到 step 0，配对的 1,106 个 Fake 全部为 tie，差值和 10,000 次 paired bootstrap 置信区间均严格为 0。P1 自身 TF FG IoU 的均值 95% bootstrap CI 为 [{summaries['P1-FROZEN']['TF_PHRASE']['foreground_iou']['bootstrap_95ci'][0]:.6f}, {summaries['P1-FROZEN']['TF_PHRASE']['foreground_iou']['bootstrap_95ci'][1]:.6f}]。结合候选表，直接空间训练没有产生可选择的 validation gain，Gate A 不通过。

## 7. G0 greedy deployment result

| Metric | P1 | Selected P3D2 | Delta / relation |
|---|---:|---:|---:|
| Fake FG IoU | {f(summaries['P1-FROZEN']['G0']['foreground_iou']['mean'])} | {f(summaries['P3D2-SPATIAL']['G0']['foreground_iou']['mean'])} | {g0_iou['mean_difference']:+.6f} |
| Fake FG F1 | {f(summaries['P1-FROZEN']['G0']['foreground_f1']['mean'])} | {f(summaries['P3D2-SPATIAL']['G0']['foreground_f1']['mean'])} | {contrasts['G0']['foreground_f1']['mean_difference']:+.6f} |
| Classification accuracy | {p1_detection['accuracy']:.6f} | {spatial_detection['accuracy']:.6f} | {spatial_detection['accuracy'] - p1_detection['accuracy']:+.6f} |
| Classification F1 | {p1_detection['f1']:.6f} | {spatial_detection['f1']:.6f} | {spatial_detection['f1'] - p1_detection['f1']:+.6f} |
| Phrase semantic score | {behavior_p1['R_phrase_sem']:.6f} | {summaries['P3D2-SPATIAL']['G0_behavior']['R_phrase_sem']:.6f} | identical |
| Structure validity | {behavior_p1['structure_validity']:.6f} | {summaries['P3D2-SPATIAL']['G0_behavior']['structure_validity']:.6f} | identical |
| Malformed rate | {behavior_p1['malformed_output_rate']:.6f} | {summaries['P3D2-SPATIAL']['G0_behavior']['malformed_output_rate']:.6f} | identical |

G0 的 FG IoU paired-bootstrap 95% CI 为 [{g0_iou['bootstrap_95ci'][0]:+.6f}, {g0_iou['bootstrap_95ci'][1]:+.6f}]。因为 Gate A 未通过且 selector 为 P1，G0 结果只确认正式部署 arm 没有变化，不能解释成训练后的空间模型保持了完全相同的部署性能。

## 8. Invariance 与 gradient audit

- Parameter audit：**{parameter_audit['status']}**。`text_hidden_fcs` 和 `mask_decoder` 分别发生参数变化，delta norm 为 {text_fc_audit['parameter_delta_norm']:.6f} 和 {mask_audit['parameter_delta_norm']:.6f}；所有冻结模块 SHA256 前后一致、delta norm 为 0。
- Gradient audit：**{gradient_audit['status']}**。在 step 1、150、250，只有 `text_hidden_fcs` 与 `mask_decoder` 存在梯度和更新；其余组 grad absent、update norm 0。
- Behavioral audit：**{behavioral['status']}**。16 个 deterministic samples 的 token IDs、文本、verdict、target phrase、SEG 次数与位置、classification logits/prediction、structure validity 均逐项一致。
- 全 validation exact audit：1,106 个 Fake 的 greedy token/text/phrase/SEG，以及 2,212 个样本的 classification prediction/probability 均完全一致。

这些审计证明训练边界正确且允许模块确实被优化；它们不等价于证明 validation localization 得到改善。

## 9. Failure analysis

正式 selected-arm 对比中，`TF and G0 both improve` 为 {failure['TF_and_G0_both_improve']['count']}，`TF improves but G0 does not` 为 {failure['TF_improves_G0_does_not']['count']}，`both remain poor` 为 {failure['both_remain_poor']['count']}。由于 selected arm 就是 P1，前两类为 0 是 selector 回退的必然结果；386 个双路径低 IoU 样例仅用于描述残余困难，不能单凭这些样例证明 architecture bottleneck。

## 10. Final route gate

Gate A：**{route['gate_A']}**。Gate B：**{route['gate_B']}**。主门：**{primary}**。

这里“Gate A 不支持 trainability”专指未检测到满足预注册显著性条件的 validation localization gain；参数和梯度审计已经证明优化过程在工程意义上实际发生。科学结论是：**在本阶段固定的数据、teacher-forced context、BCE+Dice 目标和 1,000 Fake exposures 下，仅训练现有 `text_hidden_fcs + mask_decoder` 不足以改善 validation forensic localization。** 限制可能来自 frozen representation、image features、language-to-spatial interface、target complexity 或现有 architecture capacity，本阶段无法区分这些解释。

## 11. 结论边界与停止条件

不得将本结果表述为“mask decoder 是唯一或首要瓶颈”，也不得推广为所有直接空间监督都无效。未查看 internal test，未使用 official1000 调参，未修改历史 Phase 3B、3C 或 3D.1 结论。Phase 3D.2 到此停止；未自动启动 FC-only/decoder-only ablation、扩大训练预算、joint language-spatial training、reward+mask、forensic branch 或架构修改。
"""
    (root / "final_comparison.md").write_text(report, encoding="utf-8")
    shutil.copy2(root / "final_comparison.md", ROOT / "docs/phase3d2_direct_spatial_path_optimization.md")
    required = ["experiment_manifest.json", "initial_checkpoint_manifest.json", "train_manifest.json", "training_config.json",
                "checkpoint_metadata.json", "checkpoint_selection_protocol.json", "parameter_invariance_audit.json",
                "behavioral_invariance_audit.json", "gradient_audit.json", "final_metrics.json", "paired_bootstrap.json",
                "failure_analysis.json", "route_gate.json", "final_comparison.md"]
    dump(root / "completion_manifest.json", {
        "status": "COMPLETE", "selected_checkpoint": selector["selected_checkpoint"], "selected_step": selected_step,
        "primary_gate": primary, "internal_test_used": False, "official1000_used": False,
        "artifacts": [{"path": str((root / name).resolve()), "sha256": file_sha256(root / name)} for name in required],
    })
    print(json.dumps(route, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
