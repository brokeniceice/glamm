#!/usr/bin/env python3
"""Run Phase6P0 preflight, three dependent stages, final DEV report, then stop."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase6l0_r2_preflight import dump, require

OUT = ROOT / "outputs/phase6p0_c2_native_staged_r1"
STATUS = OUT / "pipeline_status.json"
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def main():
    require(not STATUS.exists() and not (OUT / "summary.json").exists(),
            "Phase6P0 supervisor already launched or finalized")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    stages = [
        ("PREFLIGHT", "phase6p0_c2_native_staged_r1.py", "preflight"),
        ("RECTIFIER", "phase6p0_c2_native_staged_r1.py", "rectifier"),
        ("UTILITY", "phase6p0_c2_native_staged_r1.py", "utility"),
        ("JOINT", "phase6p0_c2_native_staged_r1.py", "joint"),
        ("FINALIZE", "phase6p0_c2_native_staged_r1_finalize.py", None),
    ]
    for label, script, argument in stages:
        log = OUT / "logs" / f"{label.lower()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        dump(STATUS, {"status": "RUNNING", "stage": label, "physical_gpu": 1,
                      "log": str(log), "started_utc": time.strftime(
                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        cmd = [PYTHON, "-u", str(ROOT / "scripts" / script)]
        if argument is not None:
            cmd.append(argument)
        with log.open("w") as handle:
            outcome = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, check=False)
        if outcome.returncode:
            dump(STATUS, {"status": "FAILED", "stage": label,
                          "exit_code": outcome.returncode, "log": str(log),
                          "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            raise SystemExit(outcome.returncode)
    summary = json.loads((OUT / "summary.json").read_text())
    require(summary["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV",
            "finalizer did not seal Phase6P0")
    dump(STATUS, {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV",
                  "selected_joint_epoch": summary["selected_joint_epoch"],
                  "report": summary["report"], "physical_gpu": 1,
                  "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV"}), flush=True)


if __name__ == "__main__":
    main()
