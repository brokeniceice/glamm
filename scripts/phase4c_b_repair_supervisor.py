#!/usr/bin/env python3
"""Automatic full-validation re-selection and finalization after audit repair."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
OUT = ROOT / "outputs/phase4c_b_evidence_reader"
LOGS = OUT / "logs"
STATUS = OUT / "supervisor_status.json"


def status(stage, value="RUNNING", **extra):
    STATUS.write_text(json.dumps({"status": value, "stage": stage,
                                  "updated_unix": time.time(), **extra}, indent=2) + "\n")


def launch(name, args, gpu):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (LOGS / f"{name}.log").open("a")
    proc = subprocess.Popen([PYTHON, *args], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    return name, proc, log


def wait_all(stage, jobs):
    active = list(jobs)
    while active:
        status(stage, processes={name: proc.pid for name, proc, _ in active})
        for job in list(active):
            name, proc, log = job; code = proc.poll()
            if code is None: continue
            log.close(); active.remove(job)
            if code:
                for _, other, other_log in active:
                    other.terminate(); other_log.close()
                raise RuntimeError(f"{name} failed with exit code {code}")
        if active: time.sleep(15)


def run(name, args, gpu=None):
    env = os.environ.copy()
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    status(name)
    with (LOGS / f"{name}.log").open("a") as log:
        result = subprocess.run([PYTHON, *args], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    wait_all("full_validation_reselection", [
        launch("reselect_clip_reader", ["scripts/phase4c_b_reselect.py", "--arm",
               "clip_reader", "--device", "cuda:0"], 0),
        launch("reselect_forensic_reader", ["scripts/phase4c_b_reselect.py", "--arm",
               "forensic_reader", "--device", "cuda:0"], 1),
    ])
    run("evaluate_G0_repaired", ["scripts/phase4c_b_evaluate.py", "--mode", "G0",
                                  "--device", "cuda:0"], 0)
    wait_all("diagnostic_re_evaluation", [
        launch("evaluate_phrase_only_repaired", ["scripts/phase4c_b_evaluate.py", "--mode",
               "phrase_only", "--device", "cuda:0"], 0),
        launch("evaluate_tf_full_repaired", ["scripts/phase4c_b_evaluate.py", "--mode",
               "tf_full_context", "--device", "cuda:0"], 1),
    ])
    run("finalize_repaired", ["scripts/phase4c_b_finalize.py"])
    status("complete", "COMPLETE")


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        status("failed", "FAILED", error=repr(exc)); raise
