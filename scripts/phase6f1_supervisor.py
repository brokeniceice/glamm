#!/usr/bin/env python3
"""Wait for the current GPU-0 LEGION worker, then run Phase6F.1 detached."""
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6f1_coverage_attribution'
STATUS=OUT/'supervisor_status.json'
PY='/home/yz/miniconda3/envs/glamm_official/bin/python'

def dump(x):
    OUT.mkdir(parents=True,exist_ok=True); t=STATUS.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,STATUS)

def legion_gpu0_active():
    p=subprocess.run(['pgrep','-af','phase5a2_legion_shared_evaluate.py'],text=True,capture_output=True)
    return any('internal_test_union' in line for line in p.stdout.splitlines())

dump({'status':'WAITING','reason':'GPU0 current LEGION internal-test localization worker','pid':os.getpid()})
while legion_gpu0_active(): time.sleep(30)
dump({'status':'RUNNING','stage':'PHASE6F1_FROZEN_INFERENCE','pid':os.getpid()})
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0'; env['PYTHONUNBUFFERED']='1'
log=OUT/'service.log'
with log.open('a') as h:
    code=subprocess.call([PY,str(ROOT/'scripts/phase6f1_coverage_attribution.py')],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
if code: dump({'status':'FAILED','exit_code':code,'log':str(log)}); raise SystemExit(code)
dump({'status':'COMPLETE','stage':'STOP','summary':str(OUT/'summary.json')})
