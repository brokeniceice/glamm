#!/usr/bin/env python3
"""Strict paired Official1000 comparison of C1-native staged and prior C1 R1."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6e2_main_vs_i2_paired as paired
from scripts import phase6e3_c1_native_staged as staged
from tools.phase4c_b import file_sha256
from tools.phase4e1 import compare, summarize

ROOT = staged.ROOT
OUT = staged.OUT
NEW = OUT / "official1000_records.jsonl"
NEW_RESULT = OUT / "official1000_results.json"
OLD = ROOT / "outputs/phase6e2_c1_specific_r1/official1000_new_r1_records.jsonl"
OLD_RESULT = ROOT / "outputs/phase6e2_c1_specific_r1/phase6e2_c1_specific_r1.json"
E1 = ROOT / "outputs/phase6e1_c1_old_r1_transfer/c1_records.jsonl"
E1_RESULT = ROOT / "outputs/phase6e1_c1_old_r1_transfer/phase6e1_c1_old_r1_transfer.json"
JSON_OUT = OUT / "official1000_paired_vs_original_c1_r1.json"
DOC = ROOT / "docs/phase6e3_c1_native_vs_original_c1_r1_official1000.md"


def main():
    source = json.loads(NEW_RESULT.read_text())
    staged.require(source["status"] == "COMPLETE", "C1-native Official1000 incomplete")
    new = paired.rows(NEW)
    old = paired.rows(OLD)
    c1_rows = paired.rows(E1)
    manifest = paired.rows(paired.MANIFEST)
    expected = [row["sample_id"] for row in manifest]
    staged.require(paired.sha(paired.MANIFEST) == "fff3c3839e54d831955f8055501882f27a408c885e0524c2d7bf5d3edbe8c700", "Official1000 manifest drift")
    staged.require(len(expected) == len(set(expected)) == 1000, "Official1000 population drift")
    for name, rows in (("native", new), ("original", old), ("C1", c1_rows)):
        staged.require([r["sample_id"] for r in rows] == expected, f"{name} sample/order drift")
    staged.require(source["records_sha256"] == paired.sha(NEW), "C1-native records SHA drift")
    prior = json.loads(OLD_RESULT.read_text())
    staged.require(prior["provenance"]["official_records_sha256"] == paired.sha(OLD), "original C1 R1 records SHA drift")
    staged.require(source["c1_sha256"] == prior["provenance"]["c1_sha256"] == staged.C1_SHA, "C1 checkpoint mismatch")
    staged.require(all(a["tp"] + a["fn"] == b["tp"] + b["fn"] for a, b in zip(new, old)), "paired GT pixel mismatch")
    staged.require(all(a["valid_q_seg"] == b["valid_q_seg"] for a, b in zip(new, old)), "paired SEG validity mismatch")
    for name, rows in (("native", new), ("original", old)):
        for row in rows:
            tp, fp, fn = (row[k] for k in ("tp", "fp", "fn"))
            staged.require(all(isinstance(v, int) and v >= 0 for v in (tp, fp, fn)), f"{name} invalid pixel counts")
            staged.require(all(math.isfinite(row[k]) for k in ("foreground_iou", "foreground_f1")), f"{name} nonfinite metric")
            staged.require(abs(row["foreground_iou"] - tp / max(1, tp + fp + fn)) < 1e-10, f"{name} IoU/count mismatch")
            staged.require(abs(row["foreground_f1"] - 2 * tp / max(1, 2 * tp + fp + fn)) < 1e-10, f"{name} F1/count mismatch")
    native_metric = summarize(new)
    original_metric = summarize(old)
    c1_old_metric = json.loads(E1_RESULT.read_text())["arms"]["C1+old_R1"]
    staged.require(all(abs(native_metric[k] - source["metrics"][k]) < 1e-12 for k in native_metric if isinstance(native_metric[k], float)), "C1-native summary drift")
    staged.require(all(abs(original_metric[k] - prior["arms"]["C1+new_R1"][k]) < 1e-12 for k in original_metric if isinstance(original_metric[k], float)), "original summary drift")
    statistics = compare(new, old, seed=3407)
    ci = paired.global_iou_ci(new, old)
    delta = {k: native_metric[k] - original_metric[k] for k in
             ("mean_foreground_iou", "mean_foreground_f1", "global_foreground_iou", "global_foreground_f1")}
    ci_mean = statistics["foreground_iou"]["bootstrap_95_ci"]
    if ci_mean[0] > 0 and ci[0] > 0:
        verdict = "C1_NATIVE_STAGED_GAIN_SUPPORTED_ON_OFFICIAL1000"
    elif ci_mean[1] < 0 and ci[1] < 0:
        verdict = "ORIGINAL_C1_R1_GAIN_SUPPORTED_ON_OFFICIAL1000"
    else:
        verdict = "NO_STABLE_UNIDIRECTIONAL_GAIN"
    result = {"status": "PASS", "direction": "C1-native staged R1 minus original C1+new R1",
              "n": 1000, "manifest_sha256": paired.sha(paired.MANIFEST), "ids_order_matched": True,
              "gt_foreground_pixels_matched": True, "seg_validity_matched": True,
              "valid_seg": sum(bool(r["valid_q_seg"]) for r in new),
              "native": {"metrics": native_metric, "records_sha256": paired.sha(NEW),
                         "checkpoint_sha256": source["selected_checkpoint_sha256"]},
              "original": {"metrics": original_metric, "records_sha256": paired.sha(OLD),
                           "checkpoint_sha256": prior["provenance"]["new_r1_sha256"]},
              "historical_c1_old_r1_reference": {"metrics": c1_old_metric,
                  "paired_mean_iou": source["paired_vs_c1_old_r1"]["foreground_iou"]},
              "delta": delta, "paired_statistics": statistics,
              "global_iou_paired_image_bootstrap_95_ci": ci,
              "bootstrap": {"seed": 3407, "repeats": 10000, "unit": "paired image"},
              "verdict": verdict,
              "claim_limit": "Official1000 was previously used for model choice; this is a matched comparison, not an untouched final test"}
    staged.dump(JSON_OUT, result)
    order = ("mean_foreground_iou", "mean_foreground_f1", "global_foreground_iou", "global_foreground_f1")
    labels = ("Mean FG IoU", "Mean FG F1", "Global FG IoU", "Global FG F1")
    lines = ["# Phase6E.3：C1 原生分阶段 R1 vs 原 C1+new R1", "",
             "Official1000 canonical G0，完整 1000 张，逐图 ID、顺序、GT 前景像素数和 `[SEG]` 有效性一致。差值方向为新方案减原方案。", "",
             "| 指标 | C1 原生分阶段 R1 | 原 C1+new R1 | 差值 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {label} | {native_metric[key]:.6f} | {original_metric[key]:.6f} | {delta[key]:+.6f} |" for key, label in zip(order, labels))
    iou = statistics["foreground_iou"]
    lines.extend(["", f"逐图配对 mean FG IoU 差的 95% bootstrap CI：`[{ci_mean[0]:+.6f}, {ci_mean[1]:+.6f}]`；胜/平/负 `{iou['wins']}/{iou['ties']}/{iou['losses']}`，Wilcoxon 双侧 `p={iou['wilcoxon_pvalue']:.6f}`。",
                  f"图像级配对重采样的 global FG IoU 差 95% CI：`[{ci[0]:+.6f}, {ci[1]:+.6f}]`。",
                  f"判定：`{verdict}`。按预定主指标 mean FG IoU，C1 原生分阶段路线没有稳定优于原方案；global FG IoU 的配对区间为正，应单独报告。两指标方向不同，不据此宣称全面改进。", "",
                  "本轮阶段选择只使用 internal DEV；Official1000 已参与历史候选选择，因此这里是匹配对照，不能称为全新独立测试。两方案的预训练来源与总训练量不同，该对照估计的是整条训练路线的效果，不是单个预训练步骤的因果效应。分类保持 C1-center，本对照只检验定位。",
                  "", f"历史参考：C1+old R1（Phase6E.1 冻结迁移）的 mean/global FG IoU 为 `{c1_old_metric['mean_foreground_iou']:.6f}/{c1_old_metric['global_foreground_iou']:.6f}`；C1 原生分阶段方案相对它的配对 mean FG IoU 差为 `+{source['paired_vs_c1_old_r1']['foreground_iou']['mean_difference']:.6f}`，95% CI `{source['paired_vs_c1_old_r1']['foreground_iou']['bootstrap_95_ci']}`。这不是上表“原 C1+new R1”的主比较。",
                  "", f"逐图、哈希和完整配对统计见[JSON](../outputs/phase6e3_c1_native_staged/{JSON_OUT.name})。", ""])
    DOC.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "verdict": verdict, "mean_iou_delta": delta["mean_foreground_iou"],
                      "global_iou_delta": delta["global_foreground_iou"]}), flush=True)


if __name__ == "__main__":
    main()
