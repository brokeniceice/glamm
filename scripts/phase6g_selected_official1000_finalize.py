#!/usr/bin/env python3
"""Audit two frozen Phase6G Official1000 passes and render their joint report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase6g_selected_official1000 import OUT, BASELINE, ids_sha, rows, dump
from tools.phase4c_b import file_sha256
from tools.phase4e1 import compare, summarize

DOC = ROOT / "docs/phase6g_selected_official1000.md"


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    if selection["status"] != "FROZEN_BEFORE_CANDIDATE_INFERENCE" or selection["candidate_order"] != ["g7", "g1"]:
        raise RuntimeError("preselected candidate registry drift")
    baseline = rows(BASELINE)
    ids = [r["sample_id"] for r in baseline]
    if len(ids) != len(set(ids)) != 1000:
        raise RuntimeError("baseline identity drift")
    candidate_rows, results = {}, {}
    for arm in selection["candidate_order"]:
        root = OUT / arm
        status = json.loads((root / "status.json").read_text())
        result = json.loads((root / "results.json").read_text())
        current = rows(root / "predictions.jsonl")
        if (status["status"] != "COMPLETE" or result["status"] != "COMPLETE" or
                len(current) != 1000 or [r["sample_id"] for r in current] != ids or
                file_sha256(root / "predictions.jsonl") != result["records_sha256"] or
                result["candidate"] != selection["candidates"][arm] or
                result["sample_ids_sha256"] != ids_sha(ids)):
            raise RuntimeError(f"{arm} result identity/status/provenance drift")
        metrics = summarize(current)
        metrics["seg_trigger_rate"] = sum(bool(r["valid_q_seg"]) for r in current) / 1000
        if any(abs(metrics[k] - result["A1"][k]) > 1e-12 for k in metrics):
            raise RuntimeError(f"{arm} independent metrics recomputation drift")
        candidate_rows[arm] = current
        results[arm] = result
    historical = summarize(baseline)
    historical["seg_trigger_rate"] = sum(bool(r["valid_q_seg"]) for r in baseline) / 1000
    cross = compare(candidate_rows["g7"], candidate_rows["g1"], seed=3407)
    audit = {
        "schema": "phase6g_selected_official1000_joint_v1", "status": "COMPLETE",
        "selection": selection, "population": 1000, "sample_ids_sha256": ids_sha(ids),
        "historical_6e2": historical,
        "g7": results["g7"]["A1"], "g1": results["g1"]["A1"],
        "g7_minus_historical": results["g7"]["paired_A1_minus_historical"],
        "g1_minus_historical": results["g1"]["paired_A1_minus_historical"],
        "g7_minus_g1": cross,
    }
    dump(OUT / "joint_results.json", audit)
    lines = ["# Phase6G 历史 R1 候选 Official1000 固定评测", "",
             "本次在访问候选 Official1000 前，依据 internal DEV canonical G0 冻结 G.7 与 G.1 两个完整 R1 检查点；均为第 8 轮。G.7 的 DEV mean IoU 最高，但 global IoU 下降；G.1 的 DEV mean/global IoU 均略高于同口径 6E.2 回放基线。两者 DEV paired mean-IoU CI 都跨 0，故这次是验证性评测，不把 DEV 点估计当成稳健增益。", "",
             "Official1000 为相同的 1000 张 Fake、C1 canonical G0、全引用掩码并集、logit 阈值 0；无 `[SEG]` 计零，多掩码取像素级 logit 最大值。候选与历史 Phase6E.2 逐样本 ID、顺序完全相同。没有用 Official1000 调参数或重选 epoch。", "",
             "| 模型 | mean FG IoU | mean FG F1 | global FG IoU | global FG F1 | SEG trigger |", "|---|---:|---:|---:|---:|---:|",]
    for name, metrics in (("历史 C1+new R1", historical), ("G.7 block11+17 staged", audit["g7"]),
                          ("G.1 block17 staged", audit["g1"])):
        lines.append(f"| {name} | {metrics['mean_foreground_iou']:.6f} | {metrics['mean_foreground_f1']:.6f} | "
                     f"{metrics['global_foreground_iou']:.6f} | {metrics['global_foreground_f1']:.6f} | "
                     f"{metrics['seg_trigger_rate']:.6f} |")
    lines += ["", "| 严格配对差值 | mean IoU Δ | IoU bootstrap 95% CI | 胜/平/负 | mean F1 Δ | F1 bootstrap 95% CI |",
              "|---|---:|---|---:|---:|---|"]
    for label, key in (("G.7 − 历史 6E.2", "g7_minus_historical"),
                       ("G.1 − 历史 6E.2", "g1_minus_historical"),
                       ("G.7 − G.1", "g7_minus_g1")):
        iou, f1 = audit[key]["foreground_iou"], audit[key]["foreground_f1"]
        lines.append(f"| {label} | {iou['mean_difference']:+.6f} | "
                     f"[{iou['bootstrap_95_ci'][0]:+.6f}, {iou['bootstrap_95_ci'][1]:+.6f}] | "
                     f"{iou['wins']}/{iou['ties']}/{iou['losses']} | "
                     f"{f1['mean_difference']:+.6f} | "
                     f"[{f1['bootstrap_95_ci'][0]:+.6f}, {f1['bootstrap_95_ci'][1]:+.6f}] |")
    lines += ["", "## 解释边界", "",
              "这两次评测只检验已冻结的完整 R1 路线。判断总体收益时同时查看 mean、global 与严格配对区间；若方向不同，明确记录 trade-off。G.13 的 side/Utility 阳性结果属于另一匹配对照，本次未推断它在正式 C1+new R1 上的外部增益。", "",
              "证据：`outputs/phase6g_selected_official1000/selection.json`、各臂 `protocol.json`、`predictions.jsonl`、`results.json` 与 `joint_results.json`。没有访问 internal test 或 OOD。", ""]
    DOC.write_text("\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "report": str(DOC)}), flush=True)


if __name__ == "__main__":
    main()
