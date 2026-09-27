#!/usr/bin/env python3
"""Read-only Phase6H.1 paired finalization and Chinese report generation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6h1_c_spatial_supervision as h

OUT = ROOT / "outputs/phase6h"
DOC = ROOT / "docs/phase6h1_c_spatial_supervision.md"
HIST = ROOT / "outputs/phase6e2_c1_specific_r1/training_curve.csv"


def fmt(x):
    return f"{x:.6f}"


def main():
    a0 = h.read(OUT / h.ARMS["A0"] / "summary.json")
    a1 = h.read(OUT / h.ARMS["A1"] / "summary.json")
    current = h.read(OUT / h.ARMS["A0"] / "per_epoch_metrics.json")
    with HIST.open(newline="") as handle:
        old = list(csv.DictReader(handle))
    if len(current) != 10 or len(old) != 10:
        raise RuntimeError("A0/historical trajectory is not 10 complete epochs")
    rows = []
    for a, b in zip(current, old):
        if int(a["epoch"]) != int(b["epoch"]):
            raise RuntimeError("epoch identity mismatch")
        rows.append({"epoch": int(a["epoch"]), "historical_iou": float(b["dev_g0_mean_iou"]),
                     "A0_iou": float(a["dev_g0_mean_iou"]),
                     "historical_loss": float(b["total_loss"]), "A0_loss": float(a["original_loss"]),
                     "order_equal": a["sample_order_sha256"] == b["sample_order_sha256"],
                     "updates_equal": int(a["optimizer_updates"]) == int(b["optimizer_updates"]),
                     "eligible_equal": int(a["optimization_eligible_exposures"]) == int(b["optimization_eligible_exposures"])})
    max_iou_drift = max(abs(x["A0_iou"] - x["historical_iou"]) for x in rows)
    max_loss_drift = max(abs(x["A0_loss"] - x["historical_loss"]) for x in rows)
    exact_contract = all(x["order_equal"] and x["updates_equal"] and x["eligible_equal"] for x in rows)
    # Coarse drift gate: historical paper trail, not a hyperparameter selector.
    reproduction_ok = (exact_contract and .195 <= a0["selected_final_metrics"]["mean_foreground_iou"] < .210
                       and max_iou_drift < .020 and max_loss_drift < .10)
    h.dump(OUT / "A0_reproduction_audit.json", {"status": "PASS" if reproduction_ok else "STOP",
        "selected_epoch": a0["selected_epoch"], "historical_epoch": 8,
        "selected_mean_iou": a0["selected_final_metrics"]["mean_foreground_iou"],
        "historical_mean_iou": 0.20273210126852226,
        "selected_difference": a0["difference_from_historical"],
        "max_abs_per_epoch_iou_drift": max_iou_drift,
        "max_abs_per_epoch_loss_drift": max_loss_drift,
        "exact_order_update_population": exact_contract, "per_epoch": rows})
    if not reproduction_ok:
        DOC.write_text("# Phase 6H.1 — Correction Spatial Supervision\n\n"
                       "**STOP：A0 复现关口未通过。** 不解释 A1 与 A0 的性能差异。"
                       "详见 `outputs/phase6h/A0_reproduction_audit.json`。\n")
        print(json.dumps({"decision": "STOP", "reason": "A0 reproduction mismatch"}), flush=True)
        return
    h.finalize()
    result = h.read(OUT / "paired_A0_A1/summary.json")
    scale = h.read(OUT / h.ARMS["A1"] / "loss_scale_check.json")
    init = h.read(OUT / "initialization_hashes.json")
    m0, m1 = result["A0"], result["A1"]
    aux = result["A1_aux"]
    iou = result["paired"]["foreground_iou"]
    f1 = result["paired"]["foreground_f1"]
    ratio = [x["aux_to_original_ratio"] for x in scale["batches"]]
    table = ["| epoch | 6E.2 IoU | A0 IoU | A0−6E.2 | 6E.2 loss | A0 loss |",
             "|---:|---:|---:|---:|---:|---:|"]
    table += [f"| {r['epoch']} | {fmt(r['historical_iou'])} | {fmt(r['A0_iou'])} | "
              f"{fmt(r['A0_iou']-r['historical_iou'])} | {fmt(r['historical_loss'])} | {fmt(r['A0_loss'])} |"
              for r in rows]
    lines = ["# Phase 6H.1 — Correction Spatial Supervision", "",
        f"状态：**{result['decision']}**。Phase6H.1 的 A0/A1 匹配训练、selector 与因果比较只使用 internal validation；后续用户单独授权 A1 Official1000 固定评测，见 F 节。未访问 internal test 或 OOD，也未进入 Phase6H.2。", "",
        "## A. 复现完整性", "",
        "- 首次 A0/A1 作业随交互会话结束而中断在 epoch 2，未用于任何指标判定；其 epoch 1 检查点与日志保存在 `outputs/phase6h/interrupted_20260924T0257Z/`。当前 A0/A1 从相同初始状态完整重跑，由独立的 systemd 用户服务运行。",
        f"- C1 SHA256：`{init['c1_sha256']}`；A0/A1 Utility 初始 state SHA256 相同：`{init['A0']['utility']}`；Rectifier 初始 state SHA256 相同：`{init['A0']['rectifier']}`。",
        f"- A1 相对 A0 只增加 `{', '.join(init['extra_trainable_A1'])}`，合计 **{init['extra_parameter_count']}** 个参数。原 Utility/Rectifier 可训练参数分别为 371,803/329,985；CLIP、Adapter、C1、SAM 及来源 evidence heads 冻结。",
        "- 原始 Phase6E.2 recipe：seed 3407；10 epochs；batch 8；AdamW lr/weight decay 1e-4；clip 1.0；无 scheduler/gradient accumulation；相同 8836 train Fake 和 1106 DEV Fake、相同样本顺序/cross/shuffle；按 final-mask DEV G0 mean IoU 选 epoch。",
        f"- A1 λ=0.1 的前三个 batch `λL_aux/L_original` 为 `{', '.join(fmt(x) for x in ratio)}`，未触发一次缩放修正。",
        "- 数据、初始化和逐 epoch 更新数/样本顺序证据见 `outputs/phase6h/initialization_hashes.json`、`outputs/phase6h/A0_reproduction_audit.json` 与各臂 `config_snapshot/protocol.json`。", "",
        "## B. A0 对 Phase6E.2 的复现", "",
        f"原始 selector 为 epoch **8**、mean FG IoU **0.2027321013**；A0 选 epoch **{a0['selected_epoch']}**，mean FG IoU **{fmt(m0['mean_foreground_iou'])}**，差 **{fmt(a0['difference_from_historical'])}**。最大逐 epoch IoU 差 **{fmt(max_iou_drift)}**，最大逐 epoch original loss 差 **{fmt(max_loss_drift)}**；样本顺序、有效曝光数和 optimizer updates 全部逐 epoch 相等：**{exact_contract}**。", "",
        "`0.202488` 是后续 Phase6F.1/F.3 冻结回放的次级参考值，**不是**原始 6E.2 epoch-8 selector。", "",
        *table, "",
        "## C. A1 的 C 空间辅助指标", "",
        "| 指标 | A1 auxiliary mask |", "|---|---:|",
        f"| mean FG IoU | {fmt(aux['mean_foreground_iou'])} |",
        f"| mean FG F1 | {fmt(aux['mean_foreground_f1'])} |",
        f"| global FG IoU | {fmt(aux['global_foreground_iou'])} |",
        f"| global FG F1 | {fmt(aux['global_foreground_f1'])} |", "",
        "A0 按预注册要求没有辅助头；因此 A1 的 auxiliary mask 仅证明该受监督 C 可被其 1×1 head 解码，**不能单凭此值量化 C 相对 A0 的可解码性增量**。不把 auxiliary 指标用作 checkpoint selector。", "",
        "## D. 最终 Utility + SAM 掩码", "",
        "| 指标 | A0 | A1 | A1−A0 |", "|---|---:|---:|---:|",
        f"| mean FG IoU | {fmt(m0['mean_foreground_iou'])} | {fmt(m1['mean_foreground_iou'])} | {fmt(iou['mean_difference'])} |",
        f"| mean FG F1 | {fmt(m0['mean_foreground_f1'])} | {fmt(m1['mean_foreground_f1'])} | {fmt(f1['mean_difference'])} |",
        f"| global FG IoU | {fmt(m0['global_foreground_iou'])} | {fmt(m1['global_foreground_iou'])} | {fmt(result['delta_global_iou'])} |",
        f"| global FG F1 | {fmt(m0['global_foreground_f1'])} | {fmt(m1['global_foreground_f1'])} | {fmt(result['delta_global_f1'])} |", "",
        f"1106 张严格配对（1090 有效单 `[SEG]`，16 张无 `[SEG]` 按零计；SEG trigger rate **{fmt(result['seg_trigger_rate'])}**）。IoU paired median Δ **{fmt(iou['median_difference'])}**，bootstrap 95% CI **[{fmt(iou['bootstrap_95_ci'][0])}, {fmt(iou['bootstrap_95_ci'][1])}]**，胜/平/负 **{iou['wins']}/{iou['ties']}/{iou['losses']}**；F1 paired mean Δ **{fmt(f1['mean_difference'])}**，CI **[{fmt(f1['bootstrap_95_ci'][0])}, {fmt(f1['bootstrap_95_ci'][1])}]**。global 指标是聚合像素指标，没有样本级 CI。", "",
        "## E. 预注册判定", "",
        f"主条件 `Δ mean FG IoU ≥ +0.010`：**{result['primary_mean_gain_at_least_0_010']}**；paired bootstrap CI 下界 >0：**{result['primary_bootstrap_lower_positive']}**；global 反方向 trade-off：**{result['tradeoff']}**。最终判定：**{result['decision']}**。", "",
        "如果辅助掩码表现较好而最终掩码未达门槛，结果只能说明该监督在当前原始 Utility/SAM 路径下未稳定转化为最终收益；不能单凭 A1 auxiliary 指标把瓶颈唯一定位到 Utility 或 SAM。Phase6H.2 需用户另行决定。", ""]
    official_path = OUT / "A1_official1000/results.json"
    if official_path.exists():
        official = h.read(official_path)
        oi = official["paired_A1_minus_historical_phase6e2"]["foreground_iou"]
        lines += ["## F. 后续单独授权的 A1 Official1000", "",
            "用户在 A1 selected checkpoint 冻结后单独授权了 Official1000 canonical G0 测试。"
            f"A1 mean FG IoU {fmt(official['A1']['mean_foreground_iou'])}、"
            f"global FG IoU {fmt(official['A1']['global_foreground_iou'])}；"
            f"历史 Phase6E.2 参考分别为 {fmt(official['historical_phase6e2_reference']['mean_foreground_iou'])}/"
            f"{fmt(official['historical_phase6e2_reference']['global_foreground_iou'])}。"
            f"严格配对 mean IoU 差 {fmt(oi['mean_difference'])}，"
            f"bootstrap 95% CI [{fmt(oi['bootstrap_95_ci'][0])},{fmt(oi['bootstrap_95_ci'][1])}]。"
            "A0 没有单独跑 Official1000；这项历史参考差值不能替代内部 A1−A0 因果比较，"
            "也不改变本阶段 STOP 判定。详见 `docs/phase6h1_a1_official1000.md`。", ""]
    DOC.write_text("\n".join(lines))
    print(json.dumps({"status": "REPORT_COMPLETE", "decision": result["decision"],
                      "report": str(DOC)}), flush=True)


if __name__ == "__main__": main()
