#!/usr/bin/env python3
"""Detached supplemental I2 pipeline; waits for the main Phase6E.2 arm."""
from __future__ import annotations
import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6e2_c1_specific_r1"
PY = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")

def dump(stage, state="RUNNING", **extra):
    value={"status":state,"stage":stage,"pid":os.getpid(),"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra}
    p=OUT/"i2_supervisor_status.json";p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(value,indent=2)+"\n");os.replace(t,p)

def wait_complete(path, predicate):
    while True:
        if path.exists():
            try:
                value=json.loads(path.read_text())
                if predicate(value): return value
            except (json.JSONDecodeError,OSError): pass
        time.sleep(30)

def run(script):
    env={**os.environ,"PHASE6E2_ARM":"i2"}
    subprocess.run([str(PY),str(ROOT/"scripts"/script)],cwd=ROOT,env=env,check=True)

def main():
    try:
        dump("WAIT_SHARED_CACHE")
        wait_complete(OUT/"cache/status.json",lambda x:x.get("status")=="COMPLETE")
        dump("WAIT_MAIN_ARM_COMPLETE")
        wait_complete(OUT/"phase6e2_c1_specific_r1.json",lambda x:x.get("status")=="COMPLETE")
        # Allow the main finalizer process to release CUDA after atomically
        # publishing its result.
        time.sleep(30)
        dump("I2_TRAIN_AND_INTERNAL_VAL_SELECT")
        run("phase6e2_c1_specific_r1_train.py")
        dump("I2_OFFICIAL1000_SELECTED_ONLY")
        run("phase6e2_official_finalize.py")
        dump("COMBINE_MAIN_AND_I2")
        subprocess.run([str(PY),str(ROOT/"scripts/phase6e2_combine.py")],cwd=ROOT,check=True)
        dump("STOP","COMPLETE")
    except BaseException as e:
        dump("FAILED","FAILED",exception_type=type(e).__name__,exception=str(e));raise

if __name__=="__main__":main()
