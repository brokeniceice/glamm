#!/usr/bin/env python3
"""Run Phase 4D-1R automatically through the frozen step512 endpoint."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
OUT = ROOT / "outputs/phase4d1r_corrected_position_aware"


def run(command, log_name):
    path = OUT / "logs" / log_name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, command))}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run([PYTHON, "scripts/phase4d1r_prepare.py"], "prepare.log")
    processes = []
    for arm, device in (("pos_clip", "cuda:0"), ("pos_forensic", "cuda:1")):
        log = (OUT / "logs" / f"{arm}.log").open("a", encoding="utf-8")
        process = subprocess.Popen([PYTHON, "scripts/phase4d1r_run_arm.py", "--arm", arm,
                                    "--device", device], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT)
        processes.append((arm, process, log))
    failures = []
    for arm, process, log in processes:
        code = process.wait(); log.close()
        if code:
            failures.append({"arm": arm, "returncode": code})
    if failures:
        dump = {"status": "FAILED", "failures": failures, "phase4d2_started": False}
        (OUT / "supervisor_status.json").write_text(json.dumps(dump, indent=2)+"\n")
        raise RuntimeError(str(failures))
    run([PYTHON, "scripts/phase4d1r_analyze.py"], "analyze.log")
    status = {"status": "COMPLETE", "formal_endpoint": 512, "selector_used": False,
              "internal_test_access": False, "official1000_access": False,
              "phase4d2_started": False}
    (OUT / "supervisor_status.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
