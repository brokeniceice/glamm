#!/usr/bin/env python3
"""Append the controlled LEGION-retrained evaluation to the reusable baseline report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/phase5a4_legion_retrained_controlled/results.json"
REPORT = ROOT / "docs/p1_reusable_evaluation_baselines.md"
START = "<!-- PHASE5A4_LEGION_RETRAINED_CONTROLLED_START -->"
END = "<!-- PHASE5A4_LEGION_RETRAINED_CONTROLLED_END -->"


def f(value: float) -> str: return f"{value:.6f}"


def main() -> None:
    d = json.loads(RESULT.read_text()); loc = d["official1000_localization"]
    r1, public, retrained = (loc[key]["primary"] for key in ("r1_g0", "legion_public_intermediate_lfree", "legion_retrained_lfree"))
    internal = d["classification"]["internal"]; aigi = d["classification"]["aigi_test"]
    im, am = internal["metrics"], aigi["metrics"]
    idelta = im["accuracy"] - 0.9836956521739131; adelta = am["accuracy"] - 0.733102253032929
    rp = loc["r1_minus_legion_retrained"]["foreground_iou"]
    pp = loc["legion_retrained_minus_public_intermediate"]["foreground_iou"]
    block = "\n".join([
        START, "## 11. Phase 5A-4 LEGION-retrained controlled evaluation", "",
        "### 11.1 Official1000 localization", "",
        "三方使用相同的 1,000-image SynthScars population、pixel-union GT、原分辨率 evaluator、logit `>0` 和 full-N failure policy。R1 保留 canonical G0；两种 LEGION 保留官方 image-only L-FREE prompt。", "",
        "| Model / protocol | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| R1 G0 | 1000 | {f(r1['mean_foreground_iou'])} | {f(r1['median_foreground_iou'])} | {f(r1['mean_foreground_f1'])} | {f(r1['global_foreground_iou'])} | {f(r1['global_foreground_f1'])} | historical frozen coverage |",
        f"| LEGION public intermediate L-FREE | 1000 | {f(public['mean_foreground_iou'])} | {f(public['median_foreground_iou'])} | {f(public['mean_foreground_f1'])} | {f(public['global_foreground_iou'])} | {f(public['global_foreground_f1'])} | {loc['legion_public_intermediate_lfree']['valid_seg_and_mask']}/1000 |",
        f"| LEGION-retrained L-FREE | 1000 | {f(retrained['mean_foreground_iou'])} | {f(retrained['median_foreground_iou'])} | {f(retrained['mean_foreground_f1'])} | {f(retrained['global_foreground_iou'])} | {f(retrained['global_foreground_f1'])} | {loc['legion_retrained_lfree']['valid_seg_and_mask']}/1000 |", "",
        f"R1−LEGION-retrained mean FG IoU delta=`{rp['mean_difference']:+.6f}`，95% CI=`[{rp['bootstrap_95_ci'][0]:+.6f}, {rp['bootstrap_95_ci'][1]:+.6f}]`，W/T/L=`{rp['wins']}/{rp['ties']}/{rp['losses']}`，Wilcoxon p=`{rp['wilcoxon_pvalue']:.8g}`。两者属于 condition-level image-only free-generation parity，但 prompt/architecture 不逐 token 相同。", "",
        f"LEGION-retrained−public intermediate 使用严格相同 L-FREE/evaluator：mean FG IoU delta=`{pp['mean_difference']:+.6f}`，95% CI=`[{pp['bootstrap_95_ci'][0]:+.6f}, {pp['bootstrap_95_ci'][1]:+.6f}]`，W/T/L=`{pp['wins']}/{pp['ties']}/{pp['losses']}`，Wilcoxon p=`{pp['wilcoxon_pvalue']:.8g}`。", "",
        "### 11.2 Classification", "",
        "R1 classification 路径对 P1 是结构性 EXACT REUSE。LEGION 使用官方 Stage-2 CLIP CLS→prediction_head，官方标签 Real=1/Fake=0；下表统一按 Fake-positive 口径报告，threshold=0.5。Internal 保留历史 direct batch=1，AIGI-test 保留历史 batch=8。", "",
        "| Dataset / model | N | Accuracy | Precision | Fake recall | Specificity | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Internal R1/P1 exact reuse | 2208 | 0.983696 | 0.988117 | 0.979167 | 0.988225 | 0.983621 | 0.998389 |",
        f"| Internal LEGION-retrained | 2208 | {f(im['accuracy'])} | {f(im['precision'])} | {f(im['recall'])} | {f(im['specificity'])} | {f(im['f1'])} | {f(im['roc_auc'])} |",
        "| AIGI-test R1/P1 exact reuse | 1731 | 0.733102 | 0.739496 | 0.715447 | 0.750575 | 0.727273 | 0.818418 |",
        f"| AIGI-test LEGION-retrained | 1731 | {f(am['accuracy'])} | {f(am['precision'])} | {f(am['recall'])} | {f(am['specificity'])} | {f(am['f1'])} | {f(am['roc_auc'])} |", "",
        f"Accuracy delta（LEGION-retrained−R1/P1）：Internal `{idelta:+.6f}`，McNemar p=`{internal['paired_vs_r1']['mcnemar_exact_two_sided_p']:.8g}`；AIGI-test `{adelta:+.6f}`，McNemar p=`{aigi['paired_vs_r1']['mcnemar_exact_two_sided_p']:.8g}`。", "",
        f"LEGION source commit=`{d['source_commit']}`；Stage-1 SHA256=`{d['stage1_le_canonical_sha256']}`；Stage-2 SHA256=`{d['stage2_cls_canonical_sha256']}`；machine result=`outputs/phase5a4_legion_retrained_controlled/results.json`。",
        END,
    ])
    text = REPORT.read_text()
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1: raise RuntimeError("malformed report markers")
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else: text = text.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(text)
    print(REPORT)


if __name__ == "__main__": main()
