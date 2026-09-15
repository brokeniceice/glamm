#!/usr/bin/env python3
"""Detached Phase 6D.3 matched C0 training supervisor."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6d3_c0"
LOG = OUT / "supervisor_events.jsonl"
DS = Path("/home/yz/miniconda3/envs/glamm_official/bin/deepspeed")

def event(name, **values):
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "event": name, **values}) + "\n")

def main():
    state_path = OUT / "training/training_state.json"
    state = json.load(state_path.open()) if state_path.exists() else {}
    if int(state.get("optimizer_step", 0)) >= 5000:
        event("training_already_complete", optimizer_step=state["optimizer_step"])
        return
    command = [
        str(DS), "--include", "localhost:1",
        "scripts/phase2a_distributed_train.py",
        "--config", "configs/phase6d3_c0_h2_p1.yaml",
        "--mode", "train", "--workers", "4", "--h2-classifier",
    ]
    last = ROOT / "checkpoints/phase6d3_c0/last"
    if state and last.exists():
        command += ["--resume", str(last)]
    event("training_start", command=command)
    with (OUT / "training.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "GLAMM_PRESERVE_CUDA_CACHE": "1"},
        )
    event("training_complete" if result.returncode == 0 else "training_failed",
          returncode=result.returncode)
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()
