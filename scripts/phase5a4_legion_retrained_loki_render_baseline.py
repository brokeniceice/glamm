#!/usr/bin/env python3
"""Idempotently append Phase 5A-4 LOKI results to the reusable baseline report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/phase5a4_legion_retrained_loki/results.json"
REPORT = ROOT / "docs/p1_reusable_evaluation_baselines.md"
START = "<!-- PHASE5A4_LEGION_RETRAINED_LOKI_START -->"
END = "<!-- PHASE5A4_LEGION_RETRAINED_LOKI_END -->"


def f(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE" or result.get("n") != 229:
        raise RuntimeError("Phase 5A-4 result is not complete")
    r1 = result["r1_g1"]["primary"]
    legion = result["legion_retrained_lfree"]["primary"]
    public = result["legion_public_intermediate_lfree"]["primary"]
    delta = result["descriptive_r1_minus_legion_retrained"]["foreground_iou"]
    legion_delta = result["strict_legion_retrained_minus_public_intermediate"]["foreground_iou"]
    valid = result["legion_retrained_lfree"]["valid_seg_and_mask"]
    ci = delta["bootstrap_95_ci"]
    block = "\n".join([
        START,
        "## 10. Phase 5A-4 LEGION-retrained LOKI localization",
        "",
        "Population 为冻结 LOKI 229-image localization scope，GT 与 §4.6 相同：原分辨率 filled bbox union。LEGION 使用官方 image-only L-FREE prompt、free generation、`[SEG]`→SAM、mask logit `>0`、multiple masks union；无 `[SEG]`/空预测在 full-N 主结果中计零。checkpoint 是本项目 internal-train 重训的 Stage-1 merged LE，而不是 public intermediate `legion_LE`。",
        "",
        "| Model / protocol | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| R1 G1 known-Fake | 229 | {f(r1['mean_foreground_iou'])} | {f(r1['median_foreground_iou'])} | {f(r1['mean_foreground_f1'])} | {f(r1['global_foreground_iou'])} | {f(r1['global_foreground_f1'])} | {result['r1_g1']['valid_seg_and_mask']}/229 |",
        f"| LEGION public intermediate L-FREE | 229 | {f(public['mean_foreground_iou'])} | {f(public['median_foreground_iou'])} | {f(public['mean_foreground_f1'])} | {f(public['global_foreground_iou'])} | {f(public['global_foreground_f1'])} | {result['legion_public_intermediate_lfree']['valid_seg_and_mask']}/229 |",
        f"| LEGION-retrained L-FREE | 229 | {f(legion['mean_foreground_iou'])} | {f(legion['median_foreground_iou'])} | {f(legion['mean_foreground_f1'])} | {f(legion['global_foreground_iou'])} | {f(legion['global_foreground_f1'])} | {valid}/229 |",
        "",
        f"R1−LEGION-retrained 的逐图 descriptive mean FG IoU delta=`{delta['mean_difference']:+.6f}`，bootstrap 95% CI=`[{ci[0]:+.6f}, {ci[1]:+.6f}]`，W/T/L=`{delta['wins']}/{delta['ties']}/{delta['losses']}`，Wilcoxon p=`{delta['wilcoxon_pvalue']:.8g}`。",
        "",
        "该行是 **cross-protocol diagnostic**，不是严格同输入条件主 baseline：R1 使用 structural GT `[FAKE]` prefix 的 G1，LEGION 使用不含 GT authenticity/phrase/explanation 的官方 L-FREE。因此 paired statistics 只描述同一图像/GT/evaluator 下的数值差异，不将差异归因为纯模型效应。",
        "",
        f"LEGION-retrained−public intermediate 是严格同 L-FREE 输入与 evaluator 的配对比较：mean FG IoU delta=`{legion_delta['mean_difference']:+.6f}`，bootstrap 95% CI=`[{legion_delta['bootstrap_95_ci'][0]:+.6f}, {legion_delta['bootstrap_95_ci'][1]:+.6f}]`，W/T/L=`{legion_delta['wins']}/{legion_delta['ties']}/{legion_delta['losses']}`，Wilcoxon p=`{legion_delta['wilcoxon_pvalue']:.8g}`。公开权重是 official released intermediate，不是论文 final checkpoint。",
        "",
        f"LEGION source commit=`{result['legion_source_commit']}`；Stage-1 checkpoint canonical SHA256=`{result['legion_checkpoint_canonical_sha256']}`；machine result=`outputs/phase5a4_legion_retrained_loki/results.json`。",
        END,
    ])
    text = REPORT.read_text(encoding="utf-8")
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
            raise RuntimeError("Phase 5A-4 report markers are malformed")
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
