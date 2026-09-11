#!/usr/bin/env python3
"""Frozen epoch-4 C1 validation-G0 and DEV-OOD diagnostic on GPU 0."""
from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.phase6d3_posttrain import eval_dev

OUT=ROOT/'outputs/phase6d3_c1/diagnostics/epoch_04'
CKPT=ROOT/'checkpoints/phase6d3_c1/diagnostics/epoch_04/mp_rank_00_model_states.pt'
CFG=ROOT/'configs/phase6d3_c1_rine_conditioned_p1.yaml'
PY='/home/yz/miniconda3/envs/glamm_official/bin/python'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 old_predictions=OUT/'internal_validation/G0/predictions.jsonl'
 if old_predictions.exists():
  old_count=sum(1 for line in old_predictions.open() if line.strip())
  (OUT/'invalidated_pre_fix_g0.json').write_text(json.dumps({
   'status':'INVALIDATED','record_count':old_count,
   'reason':'C1 evaluate generation path omitted the RINE-Q2 FRC token and shared RINE hooks retained unrelated CLIP activations until OOM',
   'reuse_allowed':False,'replacement':'full canonical rerun after explicit FRC injection and scoped hooks'
  },indent=2)+'\n')
 provenance={'schema':'phase6d3_epoch4_diagnostic_v1','status':'RUNNING','checkpoint':str(CKPT.resolve()),
  'checkpoint_sha256':sha(CKPT),'epoch':4,'optimizer_step':2000,
  'role':'non_selector_intermediate_diagnostic','selection_effect':False,
  'requested_evaluations':['internal_validation_G0','DEV-OOD-2560_classification'],'started_at':time.time()}
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 subprocess.run([PY,str(ROOT/'scripts/phase3a_evaluate.py'),'--config',str(CFG),'--checkpoint',str(CKPT),
  '--output-dir',str(OUT/'internal_validation'),'--split','val','--device','cuda:0','--modes','G0',
  '--expected-step','2000','--expected-epoch','4','--generation-batch-size','1','--reset'],cwd=ROOT,check=True)
 config=yaml.safe_load(CFG.read_text())
 dev,meta=eval_dev(config,4,2000,'cuda:0',output_root=OUT,checkpoint=CKPT)
 provenance.update(status='COMPLETE',completed_at=time.time(),dev_ood_metrics=dev,
  g0_summary=json.load(open(OUT/'internal_validation/summary.json')),checkpoint_metadata=meta)
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
