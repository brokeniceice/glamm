#!/usr/bin/env python3
"""Wait for the entire Phase6F.4 staged chain, then move LEGION remainder to GPU0."""
from __future__ import annotations
import json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GATE=ROOT/'outputs/phase6f4_full_fov_comparison/phase6f4_full_fov_training_results.json'
OUT=ROOT/'outputs/legion_retrained_match_evaluation'
STATUS=OUT/'after_phase6f4_staged_handoff.json'
PY='/home/yz/miniconda3/envs/legion/bin/python'

def now():return datetime.now(timezone.utc).isoformat()
def dump(x):
    STATUS.parent.mkdir(parents=True,exist_ok=True);t=STATUS.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,STATUS)
def complete(path):
    try:return str(json.loads(Path(path).read_text()).get('status','')).startswith('COMPLETE')
    except Exception:return False
def gpu0_phase6f4_active():
    p=subprocess.run(['pgrep','-af','phase6f4_(full_fov_staged_train|full_fov_official1000|post_i2_supervisor)'],capture_output=True,text=True)
    return bool(p.stdout.strip())

def main():
    dump({'status':'WAITING_ENTIRE_PHASE6F4_STAGED','gate':str(GATE),'pid':os.getpid(),'created_at_utc':now()})
    while not complete(GATE):time.sleep(30)
    # The final result is written at the end of the chain; also require every
    # Phase6F.4 GPU worker/supervisor to have exited before using physical GPU0.
    while gpu0_phase6f4_active():time.sleep(10)
    if complete(OUT/'results.json'):
        dump({'status':'COMPLETE_NO_MOVE_NEEDED','reason':'LEGION evaluation already completed on GPU1','completed_at_utc':now()});return
    dump({'status':'STOPPING_GPU1_HANDOFF_AFTER_STAGED','at_utc':now()})
    subprocess.run(['systemctl','--user','stop','legion-retrained-match-gpu1-handoff.service'],check=False)
    while subprocess.run(['pgrep','-f','legion_retrained_match_continue_gpu1.py|legion_retrained_match_localization_extra.py|phase5a2_legion_shared_evaluate.py'],stdout=subprocess.DEVNULL).returncode==0:time.sleep(5)
    dump({'status':'RUNNING_REMAINDER_GPU0','started_at_utc':now()})
    log=OUT/'logs/continue_gpu0_after_phase6f4_staged.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    with log.open('a') as h:code=subprocess.call([PY,str(ROOT/'scripts/legion_retrained_match_continue_gpu0_after_staged.py')],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
    if code:dump({'status':'FAILED','exit_code':code,'log':str(log),'failed_at_utc':now()});raise SystemExit(code)
    dump({'status':'COMPLETE','stage':'STOP','completed_at_utc':now()})

if __name__=='__main__':main()
