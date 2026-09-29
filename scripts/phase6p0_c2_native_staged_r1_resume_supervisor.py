#!/usr/bin/env python3
"""Continue geometry-corrected Utility/joint and selected-only Official1000."""
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
from tools.phase4c_b import file_sha256

OUT = ROOT / "outputs/phase6p0_c2_native_staged_r1"
STATUS = OUT / "pipeline_status.json"
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def main():
    gates = json.loads((OUT / "preflight_gates.json").read_text())
    rect = json.loads((OUT / "rectifier/selector.json").read_text())
    require(gates["status"] == "PASS" and gates["C2_G0_full_DEV_exact"] and
            gates["R1_utility_S64_original_geometry_exact"] and
            gates["retained_rectifier_selected_checkpoint_sha256"] ==
            rect["selected_checkpoint_sha256"] and
            file_sha256(Path(rect["selected_checkpoint"])) ==
            rect["selected_checkpoint_sha256"],
            "geometry-corrected preflight or retained Rectifier selector incomplete")
    require(not STATUS.exists() and
            not (OUT / "utility/selector.json").exists() and
            not (OUT / "joint/selector.json").exists() and
            not (OUT / "official1000_results.json").exists(),
            "corrected pipeline already launched/completed")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    stages = [
        ("UTILITY", "phase6p0_c2_native_staged_r1.py", "utility"),
        ("JOINT", "phase6p0_c2_native_staged_r1.py", "joint"),
        ("INTERNAL_FINALIZE", "phase6p0_c2_native_staged_r1_finalize.py", None),
        ("OFFICIAL1000", "phase6p0_c2_native_staged_r1_official.py", None),
    ]
    for label, script, argument in stages:
        log = OUT / "logs" / f"corrected_{label.lower()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        dump(STATUS, {"status": "RUNNING", "stage": label, "physical_gpu": 1,
                      "log": str(log),
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        command = [PYTHON, "-u", str(ROOT / "scripts" / script)]
        if argument is not None:
            command.append(argument)
        with log.open("w") as handle:
            outcome = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT, check=False)
        if outcome.returncode:
            dump(STATUS, {"status": "FAILED", "stage": label,
                          "exit_code": outcome.returncode, "log": str(log),
                          "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            raise SystemExit(outcome.returncode)
    summary = json.loads((OUT / "summary.json").read_text())
    require(summary["status"] == "COMPLETE_STOP_AFTER_OFFICIAL1000",
            "Official1000 finalizer did not seal Phase6P0")
    dump(STATUS, {"status": "COMPLETE_STOP_AFTER_OFFICIAL1000",
                  "selected_joint_epoch": summary["selected_joint_epoch"],
                  "report": summary["report"],
                  "official1000_result": summary["official1000_result"],
                  "physical_gpu": 1,
                  "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"status": "COMPLETE_STOP_AFTER_OFFICIAL1000"}), flush=True)


if __name__ == "__main__":
    main()
