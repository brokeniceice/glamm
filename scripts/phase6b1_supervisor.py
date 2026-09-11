#!/usr/bin/env python3
"""Detached recovery guard for the frozen Phase6B.1 inference."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase6b1_external_confirmation'
DATASETS=('internal','aigi_holmes','genimage','loki','raise998');SEEDS=(3407,3408,3409);ARMS=('C1-L','C1-S')
def complete(ds):return all((OUT/'predictions'/ds/f'{arm}_seed{seed}.pt').is_file() for arm in ARMS for seed in SEEDS)
def stamp(status,**kw):
 p=OUT/'supervisor_status.json';v={'status':status,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw};p.write_text(json.dumps(v,indent=2)+'\n')
def alive(pid):
 try:return 'phase6b1_external_infer.py' in Path(f'/proc/{pid}/cmdline').read_text()
 except (FileNotFoundError,PermissionError):return False
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args();stamp('WAITING_EXISTING_INFERENCE',wait_pid=a.wait_pid)
 while alive(a.wait_pid):time.sleep(30)
 missing=[d for d in DATASETS if not complete(d)]
 if missing:
  stamp('RECOVERING_MISSING',missing=missing)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',HF_HOME='/data/yz/myLISA_storage/checkpoints/.hf_cache',TRANSFORMERS_VERBOSITY='error')
  cmd=[str(Path('/home/yz/miniconda3/envs/glamm_official/bin/python')),str(ROOT/'scripts/phase6b1_external_infer.py'),'--datasets',*missing,'--device','cuda:0','--batch-size','96','--workers','12']
  with (OUT/'supervisor_inference.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 missing=[d for d in DATASETS if not complete(d)]
 if missing:raise RuntimeError(f'incomplete after recovery: {missing}')
 stamp('INFERENCE_COMPLETE',datasets=list(DATASETS),next='await user request for aggregation/report inspection')
if __name__=='__main__':
 try:main()
 except BaseException as e:stamp('FAILED',exception_type=type(e).__name__,exception=str(e));raise
