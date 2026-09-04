#!/usr/bin/env python3
"""Automatic LOKI continuation, aggregation, and baseline-document finalization."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
RUNNER = str(ROOT / "scripts/p1_r1_reusable_matrix.py")
OUT = ROOT / "outputs/p1_r1_reusable_matrix"
SUPERVISOR = OUT / "supervisor_status.json"
STATUS = OUT / "finalizer_status.json"


def write(value):
    tmp = STATUS.with_suffix(".json.tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); tmp.replace(STATUS)


def main():
    state = {"schema": "p1_r1_reusable_matrix_finalizer_v1", "status": "WAITING_SYNTHETIC_MATRIX", "started_unix": time.time()}; write(state)
    while True:
        if SUPERVISOR.is_file():
            upstream = json.loads(SUPERVISOR.read_text())
            if upstream["status"] == "FAILED_STOP":
                state.update({"status": "UPSTREAM_FAILED_STOP", "upstream": upstream, "stopped_unix": time.time()}); write(state); return
            if upstream["status"] == "SYNTHETIC_MATRIX_COMPLETE": break
        time.sleep(30)
    state["status"] = "RUNNING_LOKI_G1"; write(state)
    log_path = OUT / "logs/loki_g1.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        code = subprocess.run([PYTHON, RUNNER, "run", "--job", "loki_g1", "--device", "cuda:2"], cwd=ROOT,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        state.update({"status": "LOKI_FAILED_STOP", "returncode": code, "stopped_unix": time.time()}); write(state); return
    state["status"] = "AGGREGATING_AND_UPDATING_DOC"; write(state)
    with (OUT / "logs/aggregate.log").open("a", buffering=1) as log:
        code = subprocess.run([PYTHON, RUNNER, "aggregate"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    state.update({"status": "COMPLETE" if code == 0 else "AGGREGATE_FAILED_STOP", "returncode": code,
                  "results": str(OUT / "results.json"), "document": str(ROOT / "docs/p1_reusable_evaluation_baselines.md"),
                  "completed_unix": time.time()}); write(state)


if __name__ == "__main__": main()
