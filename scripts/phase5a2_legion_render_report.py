#!/usr/bin/env python3
"""Render the completed Phase 5A-2 public-checkpoint comparison in Markdown."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a2_legion_public_reference/shared_results.json"
DOC = ROOT / "docs/phase5a2_legion_public_reference_results.md"
CONDITIONS = ("original", "jpeg70", "jpeg80", "gaussian5", "gaussian10")


def f(value: float) -> str:
    return f"{float(value):.6f}"


def paired(value: dict) -> str:
    return f"{f(value['mean_difference'])} [{f(value['bootstrap_95_ci'][0])}, {f(value['bootstrap_95_ci'][1])}], p={value['wilcoxon_pvalue']:.3g}"


def main() -> None:
    result = json.loads(OUT.read_text(encoding="utf-8"))
    original_workers = result["conditions"]["original"]["legion_lfree"]["workers"]
    worker = original_workers[0]
    lines = [
        "# Phase 5A-2 — R1 G0 vs LEGION public intermediate L-FREE",
        "",
        "## Status",
        "",
        "COMPLETED. This is a reference/diagnostic comparison of the released LEGION intermediate LE checkpoint, not a reproduction of the paper-final checkpoint and not a main paper baseline.",
        "",
        "## Frozen protocol",
        "",
        f"- Shared official1000 manifest: `{result['manifest']}`",
        f"- Manifest SHA256: `{result['manifest_sha256']}`",
        f"- Ordered 1000-ID SHA256: `{result['ordered_sample_id_sha256']}`",
        f"- LEGION source commit: `{worker['legion_source_commit']}`",
        "- LEGION mode: official L-FREE prompt, free generation, `[SEG]` to SAM, mask logit `> 0`, all returned masks unioned at original resolution.",
        "- R1 mode: existing frozen P1 G0; no retraining, prompt change, or classification gate.",
        "- Primary metric: all 1,000 images. No `[SEG]` / no mask is an all-zero prediction and therefore receives IoU/F1=0 because every official1000 image has a nonempty Fake target.",
        "- Conditional `both_valid_seg_and_mask` rows are coverage-conditioned diagnostics only, never the primary comparison.",
        "- R1 Phrase and R1 TF-PHRASE: N/A for this LEGION comparison; no artificial LEGION phrase/teacher-forcing condition was constructed.",
        "",
        "## Primary all-sample results",
        "",
        "| Condition | Model | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid `[SEG]`+mask |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        data = result["conditions"][condition]
        for label, key in (("R1 G0", "r1_g0"), ("LEGION L-FREE", "legion_lfree")):
            item = data[key]
            metric, coverage = item["primary"], item["coverage"]
            lines.append(
                f"| {condition} | {label} | {f(metric['mean_foreground_iou'])} | {f(metric['median_foreground_iou'])} | {f(metric['mean_foreground_f1'])} | {f(metric['global_foreground_iou'])} | {f(metric['global_foreground_f1'])} | {coverage['valid_seg_and_mask']}/{coverage['n']} |"
            )
    lines += [
        "",
        "## Paired primary comparison",
        "",
        "Values are R1 minus LEGION; a positive value favors R1. CI is deterministic paired bootstrap 95%; p is two-sided Wilcoxon signed-rank.",
        "",
        "| Condition | Δ mean FG IoU (R1−LEGION) | Δ mean FG F1 (R1−LEGION) |",
        "| --- | --- | --- |",
    ]
    for condition in CONDITIONS:
        comparison = result["conditions"][condition]["primary_r1_minus_legion"]
        lines.append(f"| {condition} | {paired(comparison['foreground_iou'])} | {paired(comparison['foreground_f1'])} |")
    lines += [
        "",
        "## Robustness: original minus corrupted condition",
        "",
        "Positive values denote degradation from original. These are independently paired within each model; they are not a cross-model superiority test.",
        "",
        "| Condition | R1 Δ mean FG IoU | LEGION Δ mean FG IoU | R1 Δ mean FG F1 | LEGION Δ mean FG F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in CONDITIONS[1:]:
        data = result["conditions"][condition]["robustness_degradation_original_minus_condition"]
        lines.append(f"| {condition} | {paired(data['r1_g0']['foreground_iou'])} | {paired(data['legion_lfree']['foreground_iou'])} | {paired(data['r1_g0']['foreground_f1'])} | {paired(data['legion_lfree']['foreground_f1'])} |")
    lines += [
        "",
        "## Both-valid `[SEG]` diagnostic",
        "",
        "This subset excludes any image without a valid localization output from either model. It describes localization quality conditional on both pipelines emitting a segment, and must not replace the all-sample primary result.",
        "",
        "| Condition | N | R1 mean FG IoU | LEGION mean FG IoU | Δ mean FG IoU (R1−LEGION) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for condition in CONDITIONS:
        data = result["conditions"][condition]["both_valid_seg_and_mask_diagnostic"]
        lines.append(f"| {condition} | {data['n']} | {f(data['r1']['mean_foreground_iou'])} | {f(data['legion']['mean_foreground_iou'])} | {paired(data['r1_minus_legion']['foreground_iou'])} |")
    lines += [
        "",
        "## Boundary of interpretation",
        "",
        "The released `legion_LE` checkpoint is an intermediate public checkpoint. These results establish its behavior under the frozen shared protocol only. The formal main baseline route remains retraining official LEGION on the shared training data before making paper-level comparative claims.",
        "",
        "Machine-readable artifact: `outputs/phase5a2_legion_public_reference/shared_results.json`.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOC)


if __name__ == "__main__":
    main()
