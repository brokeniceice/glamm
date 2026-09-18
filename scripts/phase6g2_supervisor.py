#!/usr/bin/env python3
"""Gate 6G.2A before any 6G.2B training."""
from __future__ import annotations
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/home/yz/miniconda3/envs/glamm_official/bin/python';OUT=ROOT/'outputs/phase6g2_multilevel_attention'
def run(args):subprocess.run([PY,*map(str,args)],cwd=ROOT,check=True)
def main():
 run([ROOT/'scripts/phase6g2_attention_probe.py'])
 a=json.load(open(OUT/'phase6g2a/summary.json'))
 if a['decision']!='MULTILEVEL_ATTENTION_FUSION_SUPPORTED':
  (ROOT/'docs/phase6g2_multilevel_attention.md').write_text('# Phase 6G.2 — Block11/17 Cross-Layer Attention Fusion\n\nPhase6G.2A did not pass the preregistered paired-IoU gate. Phase6G.2B was not run.\n\n```text\nMULTILEVEL_ATTENTION_FUSION_NOT_SUPPORTED\n```\n')
  return
 # Phase6G.2B is a separate authorization boundary.  A service restart must
 # never turn a completed 2A gate into implicit full-R1 training.
 if os.environ.get('PHASE6G2_ALLOW_B')!='1':
  (OUT/'phase6g2b_paused.json').write_text(json.dumps({
   'status':'PAUSED_AWAITING_EXPLICIT_AUTHORIZATION',
   'phase6g2a_decision':a['decision'],
   'phase6g2b_started_before_pause':(OUT/'phase6g2b/adapter/history.json').exists(),
   'resume_requires_environment':'PHASE6G2_ALLOW_B=1'
  },indent=2)+'\n')
  return
 run([ROOT/'scripts/phase6g2_fusion_adapter.py'])
 for stage in ('rectifier','utility','joint'):run([ROOT/'scripts/phase6g2_staged_r1.py','--stage',stage,'--device','cuda:0'])
if __name__=='__main__':main()
