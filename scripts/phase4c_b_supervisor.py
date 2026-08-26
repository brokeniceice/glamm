#!/usr/bin/env python3
"""Resume-safe automatic Phase 4C-B dependency runner after initial cache jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
CFG = yaml.safe_load((ROOT / "configs/phase4c_b_evidence_reader.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
CACHE = Path(CFG["experiment"]["cache_root"])
LOGS = OUT / "logs"
STATUS = OUT / "supervisor_status.json"


def write_status(stage, status="RUNNING", **extra):
    value = {"status": status, "stage": stage, "updated_unix": time.time(), **extra}
    STATUS.write_text(json.dumps(value, indent=2) + "\n")


def complete(path):
    try:
        return json.loads(path.read_text())["status"] == "COMPLETE"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def wait_for(path, label):
    while not complete(path):
        write_status(label)
        time.sleep(15)


def command(name, args, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (LOGS / f"{name}.log").open("a")
    write_status(name)
    result = subprocess.run([str(PYTHON), *args], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def launch(name, args, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (LOGS / f"{name}.log").open("a")
    proc = subprocess.Popen([str(PYTHON), *args], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    return name, proc, log


def wait_group(items, stage):
    active = list(items)
    while active:
        write_status(stage, processes={name: proc.pid for name, proc, _ in active})
        for item in list(active):
            name, proc, log = item
            code = proc.poll()
            if code is None:
                continue
            log.close()
            active.remove(item)
            if code:
                for _, other, other_log in active:
                    other.terminate()
                    other_log.close()
                raise RuntimeError(f"{name} failed with exit code {code}")
        if active:
            time.sleep(15)


def update_feature_manifest():
    path = OUT / "feature_cache_manifest.json"
    value = json.loads(path.read_text())
    value.update({
        "status": "COMPLETE",
        "train_q_seg": json.loads((CACHE / "train_q_seg/complete.json").read_text()),
        "validation_fresh_contexts": {
            mode: json.loads((CACHE / f"validation/{mode}/complete.json").read_text())
            for mode in ("G0", "phrase_only", "tf_full_context")
        },
    })
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    wait_for(CACHE / "train_q_seg/complete.json", "wait_train_q")
    for mode in ("phrase_only", "tf_full_context"):
        marker = CACHE / f"validation/{mode}/complete.json"
        if not complete(marker):
            command(f"cache_{mode}", ["scripts/phase4c_b_cache.py", "--mode", mode,
                                      "--device", "cuda:0"], gpu=0)
    wait_for(CACHE / "validation/G0/complete.json", "wait_G0")
    update_feature_manifest()
    if not complete(OUT / "preflight_two_step_gradient.json"):
        command("preflight", ["scripts/phase4c_b_train.py", "--mode", "preflight",
                              "--device", "cuda:0"], gpu=0)
    jobs = []
    for arm, gpu in (("clip_reader", 0), ("forensic_reader", 1)):
        if not complete(OUT / f"training/{arm}/completion.json"):
            jobs.append(launch(f"train_{arm}", ["scripts/phase4c_b_train.py", "--mode",
                               "train", "--arm", arm, "--device", "cuda:0"], gpu))
    wait_group(jobs, "train_readers")
    command("evaluate_G0", ["scripts/phase4c_b_evaluate.py", "--mode", "G0",
                            "--device", "cuda:0"], gpu=0)
    diagnostics = []
    for mode, gpu in (("phrase_only", 0), ("tf_full_context", 1)):
        diagnostics.append(launch(f"evaluate_{mode}", ["scripts/phase4c_b_evaluate.py",
                                  "--mode", mode, "--device", "cuda:0"], gpu))
    wait_group(diagnostics, "evaluate_diagnostics")
    command("finalize", ["scripts/phase4c_b_finalize.py"])
    write_status("complete", "COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", "FAILED", error=repr(exc))
        raise
