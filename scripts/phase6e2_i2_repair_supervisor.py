#!/usr/bin/env python3
"""Finish the single-process I2 Official1000 rerun and combine its report."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=Path('/home/yz/miniconda3/envs/glamm_official/bin/python')
OUT=ROOT/'outputs/phase6e2_c1_specific_r1/i2'
STATUS=OUT/'repair_supervisor_status.json'

def dump(v):
    t=STATUS.with_suffix('.json.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n'); os.replace(t,STATUS)

def main():
    state={'status':'RUNNING','stage':'WAIT_OFFICIAL1000','pid':os.getpid()};dump(state)
    try:
        while not (OUT/'phase6e2_i2_random_init.json').exists():
            worker={}
            try: worker=json.loads((OUT/'worker_status.json').read_text())
            except Exception: pass
            if worker.get('status')=='FAILED':
                env={**os.environ,'PHASE6E2_ARM':'i2'}
                subprocess.run([str(PY),str(ROOT/'scripts/phase6e2_official_finalize.py')],cwd=ROOT,env=env,check=True)
            time.sleep(20)
        state['stage']='COMBINE';dump(state)
        subprocess.run([str(PY),str(ROOT/'scripts/phase6e2_combine.py')],cwd=ROOT,check=True)
        state.update(status='COMPLETE',stage='STOP');dump(state)
    except BaseException as e:
        state.update(status='FAILED',exception_type=type(e).__name__,exception=str(e));dump(state);raise

if __name__=='__main__':main()
