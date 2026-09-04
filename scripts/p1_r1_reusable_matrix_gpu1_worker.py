#!/usr/bin/env python3
"""Disjoint GPU1 worker for Official1000 corruption jobs."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
RUNNER = str(ROOT / "scripts/p1_r1_reusable_matrix.py")
OUT = ROOT / "outputs/p1_r1_reusable_matrix"
STATUS = OUT / "gpu1_worker_status.json"
LOGS = OUT / "logs"
JOBS = ["official_jpeg70", "official_jpeg80", "official_gaussian5", "official_gaussian10"]


def write(value):
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATUS)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"schema": "p1_r1_reusable_matrix_gpu1_worker_v1", "status": "RUNNING",
             "device": "cuda:1", "jobs": JOBS, "completed": [], "current": None,
             "started_unix": time.time()}; write(state)
    for job in JOBS:
        result = OUT / "jobs" / f"{job}.json"
        if result.is_file():
            state["completed"].append(job); write(state); continue
        state["current"] = job; state["current_started_unix"] = time.time(); write(state)
        with (LOGS / f"{job}.log").open("a", buffering=1) as log:
            code = subprocess.run([PYTHON, RUNNER, "run", "--job", job, "--device", "cuda:1"],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0 or not result.is_file():
            state.update({"status": "FAILED_STOP", "failed": job, "returncode": code,
                          "current": None, "stopped_unix": time.time()}); write(state); return
        state["completed"].append(job); state["current"] = None; write(state)
    state.update({"status": "COMPLETE", "completed_unix": time.time()}); write(state)


if __name__ == "__main__":
    main()
