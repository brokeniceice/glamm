#!/usr/bin/env python3
"""Wait for the retained GPU-1 Official1000 worker, then replace the stopped old scheduler."""
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/legion_retrained_match_evaluation'
STATUS=OUT/'gpu1_handoff_status.json'
PY='/home/yz/miniconda3/envs/legion/bin/python'

def dump(value):
    t=STATUS.with_suffix('.json.tmp'); t.write_text(json.dumps(value,indent=2)+'\n'); os.replace(t,STATUS)
def official_active():
    p=subprocess.run(['pgrep','-af','phase5a2_legion_shared_evaluate.py'],capture_output=True,text=True)
    return any('official1000' in line and '--end 1000' in line for line in p.stdout.splitlines())

dump({'status':'WAITING_OFFICIAL1000_GPU1','pid':os.getpid(),'old_scheduler_main_stopped':True})
while official_active(): time.sleep(10)
subprocess.run(['systemctl','--user','stop','legion-retrained-match-evaluation.service'],check=False)
dump({'status':'RUNNING_REMAINDER_GPU1','pid':os.getpid()})
env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'
log=OUT/'logs/continue_gpu1.log'
with log.open('a') as h:
    code=subprocess.call([PY,str(ROOT/'scripts/legion_retrained_match_continue_gpu1.py')],cwd=ROOT,env=env,
                         stdout=h,stderr=subprocess.STDOUT)
if code: dump({'status':'FAILED','exit_code':code,'log':str(log)}); raise SystemExit(code)
dump({'status':'COMPLETE','stage':'STOP'})
