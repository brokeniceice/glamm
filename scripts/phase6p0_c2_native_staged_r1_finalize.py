#!/usr/bin/env python3
"""Seal the C2-native staged R1 internal-DEV result; no test populations."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase6l0_r2_finalize import global_bootstrap, rows
from scripts.phase6l0_r2_preflight import dump, require
from scripts.phase6p0_c2_native_staged_r1 import OUT, STAGES, C2_SHA
from tools.phase3c1 import paired_statistics
from tools.phase4c_b import file_sha256

DOC = ROOT / "docs/phase6p0_c2_native_staged_r1.md"


def comparison(left, right):
    require(len(left) == len(right) == 1106 and
            [x["sample_id"] for x in left] == [x["sample_id"] for x in right] and
            [x["valid_c2_g0"] for x in left] == [x["valid_c2_g0"] for x in right],
            "paired DEV ID/order/validity drift")
    return {"paired": {key: paired_statistics([x[key] for x in left], [x[key] for x in right],
                                              repeats=2000, seed=3407)
                       for key in ("foreground_iou", "foreground_f1")},
            "global_paired": global_bootstrap(left, right)}


def main():
    gates = json.loads((OUT / "preflight_gates.json").read_text())
    require(gates["status"] == "PASS" and gates["C2_G0_full_DEV_exact"] and
            gates["C2_checkpoint_sha256"] == C2_SHA and
            gates["train_valid"] == 8682 and gates["dev_valid"] == 1084,
            "C2-native staged R1 preflight incomplete")
    selectors, stage_metrics = {}, {}
    for stage in STAGES:
        item = json.loads((OUT / stage / "selector.json").read_text())
        provenance = json.loads((OUT / stage / "provenance.json").read_text())
        require(item["status"] == provenance["status"] == "COMPLETE" and
                item["stage"] == provenance["stage"] == stage and
                item["primary"] == "canonical internal DEV Mean FG IoU only" and
                item["c2_sha256"] == C2_SHA and
                file_sha256(Path(item["selected_checkpoint"])) == item["selected_checkpoint_sha256"],
                f"{stage} selected checkpoint/protocol drift")
        with (OUT / stage / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        require(len(history) == (11 if stage == "rectifier" else 10) and
                [int(x["epoch"]) for x in history] ==
                (list(range(11)) if stage == "rectifier" else list(range(1, 11))),
                f"{stage} epoch history incomplete")
        require(all(int(x["eligible"]) == 8682 and int(x["invalid"]) == 154
                    for x in history if int(x["epoch"]) > 0),
                f"{stage} canonical TRAIN traversal drift")
        best = max(history, key=lambda x: (float(x["dev_mean_fg_iou"]), -int(x["epoch"])))
        require(int(best["epoch"]) == item["selected_epoch"] and
                float(best["dev_mean_fg_iou"]) == item["selected_dev_mean_fg_iou"],
                f"{stage} selector not DEV Mean FG IoU with earlier tie")
        selected_epoch = item["selected_epoch"]
        metric = json.loads((OUT / stage / f"dev_epoch_{selected_epoch:02d}.json").read_text())
        require(metric["n"] == 1106 and metric["valid_c2_g0"] == 1084 and
                metric["mean_foreground_iou"] == item["selected_dev_mean_fg_iou"],
                f"{stage} selected DEV metric drift")
        selectors[stage] = item
        stage_metrics[stage] = metric
    require(selectors["rectifier"]["selected_checkpoint_sha256"] ==
            json.loads((OUT / "utility" / "initialization.json").read_text())["rectifier_input_selector_sha256"] and
            selectors["rectifier"]["selected_checkpoint_sha256"] ==
            json.loads((OUT / "joint" / "initialization.json").read_text())["rectifier_input_selector_sha256"] and
            selectors["utility"]["selected_checkpoint_sha256"] ==
            json.loads((OUT / "joint" / "initialization.json").read_text())["utility_input_selector_sha256"],
            "three-stage selected-checkpoint chain drift")
    base_path = ROOT / "outputs/phase6l0_r2/dev_epoch00_rows.jsonl"
    baseline = rows(base_path)
    selected = rows(OUT / "joint" / f"dev_epoch_{selectors['joint']['selected_epoch']:02d}_rows.jsonl")
    require(len(baseline) == len(selected) == 1106, "C2 DEV rows incomplete")
    paired = comparison(selected, baseline)
    c2_metric = json.loads((ROOT / "outputs/phase6l0_r2/dev_epoch00_summary.json").read_text())
    historical = json.loads((ROOT / "outputs/phase6e3_c1_native_staged/joint/selector.json").read_text())
    require(historical["status"] == "COMPLETE" and historical["stage"] == "joint" and
            file_sha256(Path(historical["selected_checkpoint"])) ==
            historical["selected_checkpoint_sha256"],
            "historical C1-native staged R1 reference drift")
    result = {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV", "protocol": "C2-native E3-style three-stage R1",
              "stages": {stage: {"selector": selectors[stage], "dev_metrics": stage_metrics[stage]}
                         for stage in STAGES},
              "C2_G0": c2_metric, "joint_minus_C2_G0": paired,
              "historical_C1_native_staged_R1": {
                  "selected_epoch": historical["selected_epoch"],
                  "dev_mean_fg_iou": historical["selected_metric"],
                  "matched_causal_comparison": False},
              "firewall": {"internal_test": False, "Official1000": False,
                           "localization_OOD": False, "classification_OOD": False}}
    dump(OUT / "final_internal_dev.json", result)
    f = lambda value: f"{float(value):.6f}"
    lines = [
        "# Phase6P0 — C2-native staged R1", "",
        "Status: **COMPLETE STOP AFTER INTERNAL DEV**.", "",
        "## Protocol", "",
        "Replicated the Phase6E.3 route on frozen C2: C2-native Rectifier training, "
        "then Utility training with the selected Rectifier frozen, then joint training "
        "from both selected checkpoints. C2, C2-native SAM, CLIP forensic adapter "
        "and evidential source heads remained frozen. No P1-trained Rectifier or Utility "
        "state was loaded.", "",
        "Canonical TRAIN: 8,836 images, 8,682 with exactly one C2 [SEG] eligible for optimization. "
        "Canonical DEV: 1,106 images, 1,084 valid; invalid samples count as zero. "
        "Each stage used ten epochs. Only DEV Mean FG IoU selected checkpoints, "
        "with earlier epoch winning ties. The Rectifier stage also considered epoch 0.", "",
        "Preflight included full per-image C2-G0 DEV parity, C2-native SAM state, "
        "frozen forensic adapter parity, C2 TRAIN geometry gamma audit, "
        "trainable parameter counts, gradient routing and frozen-state hashes.", "",
        "## Internal DEV results", "",
        "| Model or stage | Selected epoch | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| C2-G0 | — | {f(c2_metric['mean_foreground_iou'])} | {f(c2_metric['mean_foreground_f1'])} | {f(c2_metric['global_foreground_iou'])} | {f(c2_metric['global_foreground_f1'])} |",
    ]
    for stage in STAGES:
        metric = stage_metrics[stage]
        lines.append(f"| C2-native R1 {stage} | {selectors[stage]['selected_epoch']} | "
                     f"{f(metric['mean_foreground_iou'])} | {f(metric['mean_foreground_f1'])} | "
                     f"{f(metric['global_foreground_iou'])} | {f(metric['global_foreground_f1'])} |")
    p = paired["paired"]["foreground_iou"]
    g = paired["global_paired"]["iou"]
    lines += [
        "", "## Selected joint R1 versus C2-G0", "",
        f"Paired Mean FG IoU difference: {f(p['mean_difference'])}; 2,000-draw bootstrap "
        f"95% CI [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}]; "
        f"W/T/L {p['wins']}/{p['ties']}/{p['losses']}; Wilcoxon p = {p['wilcoxon_pvalue']:.3g}.",
        f"Global FG IoU difference: {f(g['difference'])}; paired bootstrap "
        f"95% CI [{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}].",
        "The checkpoint was selected on this same DEV population; intervals describe "
        "the selected result and are not independent confirmation.", "",
        "## Historical context and boundary", "",
        f"Phase6E.3 C1-native staged R1 selected epoch {historical['selected_epoch']} "
        f"with DEV Mean FG IoU {f(historical['selected_metric'])}. "
        "C1 and C2 differ in backbone and SEG-valid populations, so this is contextual "
        "and not a matched causal comparison.", "",
        "Complete stage histories, selected checkpoint hashes and paired F1/global "
        "statistics are in the output directory. Official1000, internal test and OOD "
        "were not accessed.", ""
    ]
    DOC.write_text("\n".join(lines))
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV",
        "selected_joint_epoch": selectors["joint"]["selected_epoch"],
        "selected_joint_checkpoint_sha256": selectors["joint"]["selected_checkpoint_sha256"],
        "result_sha256": file_sha256(OUT / "final_internal_dev.json"),
        "report": str(DOC), "report_sha256": file_sha256(DOC)})
    print(json.dumps({"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV",
                      "selected_joint_epoch": selectors["joint"]["selected_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
