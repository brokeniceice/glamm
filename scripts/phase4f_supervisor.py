#!/usr/bin/env python3
"""Persistent, resumable Phase 4F supervisor."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
OUT = ROOT / "outputs/phase4f_language_preserving_rectification"
STATE = OUT / "supervisor_state.json"


def dump(value):
    STATE.parent.mkdir(parents=True, exist_ok=True); STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
def load(): return json.loads(STATE.read_text()) if STATE.exists() else {"status": "RUNNING", "stages": {}, "started_at": time.time()}
def run(state, name, args, gpu):
    if state["stages"].get(name, {}).get("status") == "COMPLETE": return
    log = OUT / "logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["PYTHONUNBUFFERED"] = "1"
    state["current_stage"] = name; state["stages"][name] = {"status": "RUNNING", "gpu": gpu, "started_at": time.time(), "log": str(log)}; dump(state)
    with log.open("a") as handle: result = subprocess.run([PYTHON, *args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    state["stages"][name].update({"status": "COMPLETE" if result.returncode == 0 else "FAILED", "returncode": result.returncode, "finished_at": time.time()}); dump(state)
    if result.returncode: raise RuntimeError(f"{name} failed; see {log}")
def parallel(state, jobs):
    pending = [job for job in jobs if state["stages"].get(job[0], {}).get("status") != "COMPLETE"]
    processes = []
    for name, args, gpu in pending:
        log = OUT / "logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("a"); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["PYTHONUNBUFFERED"] = "1"
        state["stages"][name] = {"status": "RUNNING", "gpu": gpu, "started_at": time.time(), "log": str(log)}
        processes.append((name, subprocess.Popen([PYTHON, *args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    state["current_stage"] = [name for name, _, _ in processes]; dump(state); failed = []
    for name, process, handle in processes:
        code = process.wait(); handle.close(); state["stages"][name].update({"status": "COMPLETE" if code == 0 else "FAILED", "returncode": code, "finished_at": time.time()}); failed += [name] if code else []
    dump(state)
    if failed: raise RuntimeError(f"parallel stages failed: {failed}")
def main():
    state = load(); state["status"] = "RUNNING"
    state.pop("error", None); state.pop("failure_stage", None); state.pop("finished_at", None)
    dump(state)
    run(state, "preflight", ["scripts/phase4f_run.py", "--mode", "preflight", "--device", "cuda:0"], 0)
    parallel(state, [("forensic_rect_train", ["scripts/phase4f_run.py", "--mode", "train", "--arm", "forensic_rect", "--device", "cuda:0"], 0), ("clip_rect_train", ["scripts/phase4f_run.py", "--mode", "train", "--arm", "clip_rect", "--device", "cuda:0"], 1)])
    parallel(state, [("forensic_rect_evaluate", ["scripts/phase4f_run.py", "--mode", "evaluate", "--arm", "forensic_rect", "--device", "cuda:0"], 0), ("clip_rect_evaluate", ["scripts/phase4f_run.py", "--mode", "evaluate", "--arm", "clip_rect", "--device", "cuda:0"], 1)])
    run(state, "finalize", ["scripts/phase4f_finalize.py"], 0)
    state["status"] = "COMPLETE"; state["current_stage"] = None; state["finished_at"] = time.time(); dump(state)
if __name__ == "__main__":
    try: main()
    except Exception as error:
        state = load(); state["status"] = "SAFETY_STOP"; state["error"] = repr(error); state["finished_at"] = time.time(); dump(state); raise
