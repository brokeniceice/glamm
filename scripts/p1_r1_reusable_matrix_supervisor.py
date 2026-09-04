#!/usr/bin/env python3
"""Fail-closed sequential supervisor for the frozen R1 reusable matrix."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
RUNNER = ROOT / "scripts/p1_r1_reusable_matrix.py"
OUT = ROOT / "outputs/p1_r1_reusable_matrix"
STATUS = OUT / "supervisor_status.json"
LOGS = OUT / "logs"
JOBS = [
    "official_g0", "internal_g0", "official_phrase", "internal_phrase",
    "official_tf", "internal_tf",
    "official_jpeg70", "official_jpeg80", "official_gaussian5", "official_gaussian10",
    "internal_jpeg70", "internal_jpeg80", "internal_gaussian5", "internal_gaussian10",
    "loki_g1",
]


def write(value):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATUS)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"schema": "p1_r1_reusable_matrix_supervisor_v1", "status": "RUNNING",
             "device": "cuda:2", "jobs": JOBS, "completed": [], "current": None,
             "started_unix": time.time()}
    write(state)
    for job in JOBS:
        result = OUT / "jobs" / f"{job}.json"
        if result.is_file():
            state["completed"].append(job); write(state); continue
        state["current"] = job; state["current_started_unix"] = time.time(); write(state)
        with (LOGS / f"{job}.log").open("a", buffering=1) as log:
            proc = subprocess.run([str(PYTHON), str(RUNNER), "run", "--job", job, "--device", "cuda:2"],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not result.is_file():
            state.update({"status": "FAILED_STOP", "failed": job, "returncode": proc.returncode,
                          "current": None, "stopped_unix": time.time()}); write(state); return
        state["completed"].append(job); state["current"] = None; write(state)
    state.update({"status": "SYNTHETIC_MATRIX_COMPLETE", "current": None, "completed_unix": time.time()})
    write(state)


if __name__ == "__main__":
    main()
