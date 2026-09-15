#!/usr/bin/env python3
"""Detached I1 train, validation selection, Official1000, and report merge."""
from __future__ import annotations
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=Path('/home/yz/miniconda3/envs/glamm_official/bin/python')
OUT=ROOT/'outputs/phase6e2_c1_specific_r1/i1'
STATUS=OUT/'pipeline_status.json'

def dump(state):
    OUT.mkdir(parents=True,exist_ok=True);state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    tmp=STATUS.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,STATUS)

def run(script,state,stage):
    state['stage']=stage;dump(state);env={**os.environ,'PHASE6E2_ARM':'i1'}
    subprocess.run([str(PY),str(ROOT/'scripts'/script)],cwd=ROOT,env=env,check=True)

def main():
    state={'status':'RUNNING','stage':'PREFLIGHT','pid':os.getpid()};dump(state)
    try:
        if OUT.exists() and any(OUT.glob('epoch_*.pt')): raise RuntimeError('unexpected I1 output collision')
        run('phase6e2_c1_specific_r1_train.py',state,'TRAIN_10_EPOCHS_AND_INTERNAL_VAL')
        run('phase6e2_official_finalize.py',state,'OFFICIAL1000_SELECTED_ONLY')
        run('phase6e2_combine_i1.py',state,'APPEND_PHASE6E2_REPORT')
        state.update(status='COMPLETE',stage='STOP');dump(state)
    except BaseException as e:
        state.update(status='FAILED',exception_type=type(e).__name__,exception=str(e));dump(state);raise

if __name__=='__main__':main()
