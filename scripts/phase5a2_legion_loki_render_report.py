#!/usr/bin/env python3
"""Render the LOKI cross-protocol diagnostic report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/phase5a2_legion_loki_reference/shared_results.json"
REPORT = ROOT / "docs/phase5a2_legion_loki_localization_diagnostic.md"


def f(value: float) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    x = json.loads(RESULT.read_text()); r1 = x["r1_g1"]; legion = x["legion_lfree"]
    pair = x["descriptive_r1_minus_legion"]; diag = x["both_valid_seg_and_mask_diagnostic"]
    lines = [
        "# Phase 5A-2 — LOKI localization supplement",
        "", "## Status", "",
        "COMPLETED — CROSS-PROTOCOL DIAGNOSTIC ONLY.", "",
        "R1 uses the existing known-Fake G1 condition, while LEGION uses official image-only L-FREE. The comparison does not have strict input-condition parity and is not a main baseline or an exact reproduction of the LEGION paper result.",
        "", "## Frozen population and evaluator", "",
        f"- Manifest: `{x['manifest']}`", f"- Manifest SHA256: `{x['manifest_sha256']}`",
        f"- Ordered 229-ID SHA256: `{x['ordered_sample_id_sha256']}`",
        "- GT: per-image union of filled regional xywh bounding boxes at original 512x512 resolution.",
        "- No `[SEG]` / empty / invalid output: all-zero prediction and included in full-N primary metrics.",
        "- LEGION: official prompt, free generation, `[SEG]` to SAM, logit `>0`, multiple masks union; no GT label, phrase, description or bbox is model input.",
        "", "## Full-N=229 descriptive results", "",
        "| Model | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid `[SEG]`+mask |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, value in (("R1 G1", r1), ("LEGION L-FREE", legion)):
        m = value["primary"]
        lines.append(f"| {label} | {f(m['mean_foreground_iou'])} | {f(m['median_foreground_iou'])} | {f(m['mean_foreground_f1'])} | {f(m['global_foreground_iou'])} | {f(m['global_foreground_f1'])} | {value['valid_seg_and_mask']}/229 |")
    stat = pair["foreground_iou"]
    lines += ["", "Descriptive paired R1−LEGION mean FG IoU difference: "
              f"`{f(stat['mean_difference'])}`; bootstrap 95% CI "
              f"`[{f(stat['bootstrap_95_ci'][0])}, {f(stat['bootstrap_95_ci'][1])}]`; "
              f"wins/ties/losses `{stat['wins']}/{stat['ties']}/{stat['losses']}`. Because the conditions differ, this is descriptive rather than a strict model-effect test.",
              "", "## Both-valid diagnostic", "",
              f"N={diag['n']}; R1 G1 mean FG IoU `{f(diag['r1_g1']['mean_foreground_iou'])}`; LEGION L-FREE `{f(diag['legion_lfree']['mean_foreground_iou'])}`.",
              "", "Machine-readable result: `outputs/phase5a2_legion_loki_reference/shared_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
