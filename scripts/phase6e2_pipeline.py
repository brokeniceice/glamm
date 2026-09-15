#!/usr/bin/env python3
"""Detached fail-fast Phase6E.2 cache -> train/select -> Official1000 pipeline."""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6e2_c1_specific_r1"
PY = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")

def status(stage, state="RUNNING", **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    value = {"status": state, "stage": stage, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    p = OUT / "worker_status.json"; t = p.with_suffix(".json.tmp"); t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); os.replace(t, p)

def run(stage, script, *args, arm="main"):
    status(stage)
    env = {**os.environ, "PHASE6E2_ARM": arm}
    subprocess.run([str(PY), str(ROOT / "scripts" / script), *args], cwd=ROOT, env=env, check=True)

def main():
    try:
        run("CACHE_INTERNAL_C1_G0", "phase6e2_c1_g0_cache.py", "--device", "cuda:0", "--batch-size", "2")
        run("TRAIN_AND_INTERNAL_VAL_SELECT", "phase6e2_c1_specific_r1_train.py")
        run("OFFICIAL1000_SELECTED_ONLY", "phase6e2_official_finalize.py")
        run("I2_RANDOM_TRAIN_AND_INTERNAL_VAL_SELECT", "phase6e2_c1_specific_r1_train.py", arm="i2")
        run("I2_OFFICIAL1000_SELECTED_ONLY", "phase6e2_official_finalize.py", arm="i2")
        run("COMBINE_MAIN_AND_I2", "phase6e2_combine.py")
        status("STOP", "COMPLETE")
    except BaseException as e:
        status("FAILED", "FAILED", exception_type=type(e).__name__, exception=str(e))
        raise

if __name__ == "__main__": main()
