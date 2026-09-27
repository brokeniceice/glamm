#!/usr/bin/env python3
"""Evaluate the C1-native staged R1 on frozen SynthScars Official1000."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6e2_official_finalize as old_eval
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6e3_c1_native_staged as staged
from scripts.phase6e1_c1_old_r1_transfer import OUT as E1_OUT
from tools.phase4c_b import file_sha256
from tools.phase4e1 import compare, summarize
from tools.phase4f import load_rectifier

OUT = staged.OUT
RECORDS = OUT / "official1000_records.jsonl"
RESULT = OUT / "official1000_results.json"


def fresh_components(device):
    utility, _ = hc.load_utility("a2", device)
    rectifier = load_rectifier(staged.CFG, staged.c1_gamma(), device)
    return utility, rectifier, None


def main():
    selector = json.loads((OUT / "joint/selector.json").read_text())
    staged.require(selector["status"] == "COMPLETE" and not selector["official1000_used_for_selection"], "joint selector not frozen")
    selected = Path(selector["selected_checkpoint"])
    staged.require(file_sha256(selected) == selector["selected_checkpoint_sha256"], "selected checkpoint drift")
    state = torch.load(selected, map_location="cpu", weights_only=False)
    staged.require(state["c1_sha256"] == staged.C1_SHA, "C1 source drift")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # The reused evaluator reads its selector from OUT/selector.json.  The
    # C1-native selector lives under the joint stage; records stay separate.
    old_eval.OUT = OUT / "joint"
    old_eval.SELECTED = selected
    old_eval.RECORDS = RECORDS
    old_eval.hd.load_common = fresh_components
    old_eval.evaluate(device)
    rows = old_eval.rows(RECORDS)
    reference = old_eval.rows(staged.e2.BASE_OUT / "official1000_new_r1_records.jsonl")
    staged.require(len(rows) == len(reference) == 1000, "Official1000 full-N drift")
    staged.require(all(a["sample_id"] == b["sample_id"] and a["tp"] + a["fn"] == b["tp"] + b["fn"]
                       for a, b in zip(rows, reference)), "Official1000 ID/order/GT drift")
    c1_rows = old_eval.rows(E1_OUT / "c1_records.jsonl")
    staged.require([r["sample_id"] for r in rows] == [r["sample_id"] for r in c1_rows], "C1 reference order drift")
    result = {"status": "COMPLETE", "model": "C1-native staged R1", "c1_sha256": staged.C1_SHA,
              "selector": selector, "selected_checkpoint_sha256": file_sha256(selected),
              "records_sha256": file_sha256(RECORDS), "population": 1000,
              "metrics": summarize(rows), "paired_vs_previous_new_r1": compare(rows, reference, seed=3407),
              "paired_vs_c1_old_r1": compare(rows, [r["c1_old_r1"] for r in c1_rows], seed=3407),
              "selection_note": "Official1000 accessed after internal selector freeze; earlier models were already selected with this test set"}
    staged.dump(RESULT, result)
    print(json.dumps({"status": "COMPLETE", "mean_fg_iou": result["metrics"]["mean_foreground_iou"]}), flush=True)


if __name__ == "__main__":
    main()
