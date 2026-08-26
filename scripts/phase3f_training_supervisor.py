#!/usr/bin/env python3
"""Unattended, bounded Phase 3F teacher-cache and training supervisor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3f_autonomous_oracle_grounding_distillation"
CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase3f_autonomous_oracle_grounding_distillation")
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
LOGS = OUT / "supervisor/logs"


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(name, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART {time.time()} {' '.join(args)}\n"); log.flush()
        completed = subprocess.run(args, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"END {time.time()} rc={completed.returncode}\n"); log.flush()
    return completed.returncode


def latest_checkpoint():
    candidates = []
    for path in CKPT.glob("step_*/checkpoint/mp_rank_00_model_states.pt"):
        try: candidates.append((int(path.parents[1].name.split("_")[-1]), path))
        except ValueError: pass
    return max(candidates, default=(0, None))


def main():
    state = {"status": "RUNNING", "started_at": time.time(), "stages": {}}
    dump(OUT / "supervisor/state.json", state)
    rc = run("teacher_cache", [PYTHON, "scripts/phase3f_train.py", "--mode", "build-teacher-cache", "--physical-gpu", "0"])
    state["stages"]["teacher_cache"] = {"returncode": rc, "finished_at": time.time()}; dump(OUT / "supervisor/state.json", state)
    if rc:
        state["status"] = "FAILED_TEACHER_CACHE"; dump(OUT / "supervisor/state.json", state); return rc
    for attempt in range(1, 4):
        step, checkpoint = latest_checkpoint()
        command = [PYTHON, "scripts/phase3f_train.py", "--mode", "train", "--physical-gpu", "0"]
        if step > 0 and checkpoint is not None: command += ["--resume", str(checkpoint)]
        rc = run(f"training_attempt_{attempt}", command)
        state["stages"][f"training_attempt_{attempt}"] = {"returncode": rc, "resume_step": step, "finished_at": time.time()}
        dump(OUT / "supervisor/state.json", state)
        summary = OUT / "training/run_summary.json"
        if rc == 0 and summary.is_file() and json.loads(summary.read_text())["status"] == "COMPLETE":
            state["status"] = "TRAINING_COMPLETE"; state["finished_at"] = time.time(); dump(OUT / "supervisor/state.json", state); return 0
    state["status"] = "FAILED_AFTER_RETRIES"; state["finished_at"] = time.time(); dump(OUT / "supervisor/state.json", state); return 1


if __name__ == "__main__":
    raise SystemExit(main())
