#!/usr/bin/env python3
"""Recover the failed GPU2 condition strictly on physical GPU0, then finalize."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/phase3f_autonomous_oracle_grounding_distillation"
PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"
STATE=OUT/"final_supervisor/condition_state.json"
def dump(v):
 p=OUT/"gpu0_recovery/state.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+"\n")
def wait_failed_parent():
 while True:
  if STATE.is_file():
   try:
    x=json.loads(STATE.read_text())
    if x.get("status") in ("FAILED","COMPLETE"): return x
   except json.JSONDecodeError: pass
  time.sleep(20)
def run(name,args):
 p=OUT/f"gpu0_recovery/{name}.log"; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("a") as log: return subprocess.run(args,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},stdout=log,stderr=subprocess.STDOUT).returncode
def main():
 s={"status":"WAITING_FOR_P1_GPU0_CONDITION","started_at":time.time(),"physical_gpu":0,"gpu2_prohibited":True}; dump(s)
 parent=wait_failed_parent()
 if parent.get("status")=="COMPLETE": s["status"]="CONDITIONS_ALREADY_COMPLETE"; dump(s)
 else:
  selector=json.loads((OUT/"evaluation/selector/AOGD_selector.json").read_text()); checkpoint=selector["selected_checkpoint"]
  output=OUT/"evaluation/final/AOGD_SELECTED_STEP_1000"
  command=[PYTHON,"scripts/phase3a_evaluate.py","--config","configs/phase3a_p1.yaml","--checkpoint",checkpoint,
   "--output-dir",str(output),"--manifest-dir",str(ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/validation_manifest"),
   "--device","cuda:0","--modes","phrase_only","tf_full_context","--expected-step","1000","--expected-epoch","10",
   "--generation-batch-size","1","--skip-spatial-save","--tf-user-prompt","canonical","--reset"]
  s["status"]="RUNNING_AOGD_CONDITIONS_GPU0"; dump(s); rc=run("aogd_conditions_gpu0",command)
  s["aogd_conditions_returncode"]=rc; dump(s)
  if rc: s["status"]="FAILED_AOGD_CONDITIONS_GPU0"; dump(s); return 1
  parent["status"]="COMPLETE"; parent["recovery"]={"physical_gpu":0,"returncode":0,"completed_at":time.time(),"gpu2_failed_attempt_excluded":True}
  STATE.write_text(json.dumps(parent,indent=2)+"\n")
 s["status"]="RUNNING_COMPLETION_PIPELINE_GPU0"; dump(s)
 rc=run("completion_pipeline_gpu0",[PYTHON,"scripts/phase3f_completion_supervisor.py","--gpu0-only"])
 s["completion_returncode"]=rc; s["status"]="COMPLETE" if rc==0 else "FAILED_COMPLETION_PIPELINE"; s["finished_at"]=time.time(); dump(s); return rc
if __name__=="__main__": raise SystemExit(main())
