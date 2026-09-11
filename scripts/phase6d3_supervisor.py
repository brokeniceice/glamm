#!/usr/bin/env python3
"""Detached, fail-closed Phase 6D.3 C1 training/evaluation supervisor."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6d3_c1'; LOG=OUT/'supervisor_events.jsonl'
PY=Path('/home/yz/miniconda3/envs/glamm_official/bin/python')
DS=Path('/home/yz/miniconda3/envs/glamm_official/bin/deepspeed')

def event(name, **kw):
 OUT.mkdir(parents=True,exist_ok=True)
 with LOG.open('a') as f:f.write(json.dumps({'time':time.time(),'event':name,**kw})+'\n')

def run(name, command, log):
 event(name+'_start',command=[str(x) for x in command])
 with (OUT/log).open('a') as h:
  result=subprocess.run([str(x) for x in command],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','GLAMM_PRESERVE_CUDA_CACHE':'1'})
 if result.returncode: event(name+'_failed',returncode=result.returncode);raise SystemExit(result.returncode)
 event(name+'_complete')

def main():
 state_path=OUT/'training/training_state.json'
 state=json.load(state_path.open()) if state_path.exists() else {}
 if int(state.get('optimizer_step',0)) < 5000:
  command=[DS,'--include','localhost:1','scripts/phase2a_distributed_train.py','--config','configs/phase6d3_c1_rine_conditioned_p1.yaml','--mode','train','--workers','4','--rine-conditioned-c1','--rine-checkpoint','outputs/phase6b6_rine_training/selected_checkpoint.pt']
  last=ROOT/'checkpoints/phase6d3_c1/last'
  if state and last.exists():command += ['--resume',str(last)]
  run('training',command,'training.log')
 else:event('training_already_complete',optimizer_step=state['optimizer_step'])
 if not (OUT/'results.json').exists():run('posttrain',[PY,'scripts/phase6d3_posttrain.py','--device','cuda:1'],'posttrain.log')
 else:event('posttrain_already_complete')
 event('phase6d3_complete')
if __name__=='__main__':main()
