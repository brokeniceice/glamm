#!/usr/bin/env python3
"""Run Phase 4D-1 end to end and stop at the frozen decision gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
OUT = ROOT / "outputs/phase4d1_position_aware_evidence"


def run(command, log_name):
    path = OUT / "logs" / log_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, command))}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run([PYTHON, "scripts/phase4d1_prepare.py"], "prepare.log")
    processes = []
    for arm, device in (("pos_clip", "cuda:0"), ("pos_forensic", "cuda:1")):
        log = (OUT / "logs" / f"{arm}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            [PYTHON, "scripts/phase4d1_run_arm.py", "--arm", arm, "--device", device],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((arm, process, log))
    failures = []
    for arm, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append({"arm": arm, "returncode": code})
    if failures:
        (OUT / "supervisor_status.json").write_text(json.dumps({
            "status": "FAILED", "failures": failures,
            "phase4d2_started": False,
        }, indent=2) + "\n")
        raise RuntimeError(f"Phase 4D-1 arm failure: {failures}")
    run([PYTHON, "scripts/phase4d1_analyze.py"], "analyze.log")
    (OUT / "supervisor_status.json").write_text(json.dumps({
        "status": "COMPLETE", "phase4d2_started": False,
        "internal_test_access": False, "official1000_access": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
