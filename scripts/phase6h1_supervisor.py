#!/usr/bin/env python3
"""Quiet completion guard for the two already launched Phase6H.1 training jobs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6h"
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def alive(pid: int) -> bool:
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def save(value):
    p = OUT / "supervisor_status.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def stop_report(reason: str):
    (ROOT / "docs/phase6h1_c_spatial_supervision.md").write_text(
        "# Phase 6H.1 — Correction Spatial Supervision\n\n"
        f"**STOP。** {reason}\n\n"
        "A0/A1 的最终成对结论不成立；未访问 Official1000，也未进入 Phase6H.2。"
        "详见 `outputs/phase6h/supervisor_status.json` 与各臂日志。\n")


def reproduction_ok(a0: dict) -> bool:
    curve = load(OUT / "A0_phase6e2_repro/per_epoch_metrics.json")
    if curve is None or len(curve) != 10:
        return False
    with (ROOT / "outputs/phase6e2_c1_specific_r1/training_curve.csv").open(newline="") as f:
        old = list(csv.DictReader(f))
    if len(old) != 10:
        return False
    for now, before in zip(curve, old):
        if now["epoch"] != int(before["epoch"]): return False
        if now["optimizer_updates"] != int(before["optimizer_updates"]): return False
        if now["optimization_eligible_exposures"] != int(before["optimization_eligible_exposures"]): return False
        if now["sample_order_sha256"] != before["sample_order_sha256"]: return False
        if abs(now["dev_g0_mean_iou"] - float(before["dev_g0_mean_iou"])) >= .020: return False
        if abs(now["original_loss"] - float(before["total_loss"])) >= .10: return False
    return .195 <= a0["selected_final_metrics"]["mean_foreground_iou"] < .210


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-pid", type=int, required=True)
    ap.add_argument("--a1-pid", type=int, required=True)
    args = ap.parse_args()
    a0path = OUT / "A0_phase6e2_repro/summary.json"
    a1path = OUT / "A1_c_spatial_supervision/summary.json"
    save({"status": "WAITING", "A0_pid": args.a0_pid, "A1_pid": args.a1_pid})
    while True:
        a0, a1 = load(a0path), load(a1path)
        if a0 is not None and not reproduction_ok(a0):
            if alive(args.a1_pid): os.kill(args.a1_pid, signal.SIGTERM)
            save({"status": "STOP_A0_REPRODUCTION", "A0": a0,
                  "reason": "historical trajectory, order, updates or selected mean materially mismatched"})
            stop_report("A0 未通过 Phase6E.2 历史轨迹复现关口，A1 不用于得出性能结论。")
            return
        if a0 is not None and a1 is not None:
            result = subprocess.run([PYTHON, str(ROOT / "scripts/phase6h1_finalize.py")],
                                    cwd=ROOT, capture_output=True, text=True)
            save({"status": "COMPLETE" if result.returncode == 0 else "FINALIZE_FAILED",
                  "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                  "A0_selected_epoch": a0.get("selected_epoch"),
                  "A1_selected_epoch": a1.get("selected_epoch")})
            return
        if a0 is None and not alive(args.a0_pid):
            save({"status": "A0_FAILED_BEFORE_SUMMARY"})
            stop_report("A0 在完成 selector/summary 前失败。")
            return
        if a1 is None and not alive(args.a1_pid):
            save({"status": "A1_FAILED_BEFORE_SUMMARY"})
            stop_report("A1 在完成 selector/summary 前失败。")
            return
        time.sleep(30)


if __name__ == "__main__": main()
