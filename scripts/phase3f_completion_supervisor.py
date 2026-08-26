#!/usr/bin/env python3
"""Unattended remaining Phase 3F conditions, representations, and finalization."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/phase3f_autonomous_oracle_grounding_distillation"
PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"
def dump(v):
 p=OUT/"completion_supervisor/state.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+"\n")
def wait(path):
 while True:
  if path.is_file():
   try:
    x=json.loads(path.read_text())
    if x.get("status")=="COMPLETE": return
    if x.get("status")=="FAILED": raise RuntimeError(str(x))
   except json.JSONDecodeError: pass
  time.sleep(30)
def run(name,args,gpu):
 p=OUT/f"completion_supervisor/{name}.log"; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("a") as log: return subprocess.run(args,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},stdout=log,stderr=subprocess.STDOUT).returncode
def main():
 p=argparse.ArgumentParser(); p.add_argument("--gpu0-only",action="store_true"); cli=p.parse_args()
 s={"status":"WAITING_CONDITIONS","started_at":time.time(),"stages":{}}; dump(s)
 wait(OUT/"final_supervisor/condition_state.json")
 jobs=(("p1_representations","P1_TEACHER_AND_AUTO",0),("aogd_representations","AOGD_AUTO",0 if cli.gpu0_only else 2))
 for name,role,gpu in jobs:
  s["status"]=f"RUNNING_{name.upper()}"; dump(s)
  rc=run(name,[PYTHON,"scripts/phase3f_representation_evaluate.py","--role",role,"--physical-gpu",str(gpu)],gpu)
  s["stages"][name]={"returncode":rc,"finished_at":time.time()}; dump(s)
  if rc: s["status"]=f"FAILED_{name.upper()}"; dump(s); return 1
 s["status"]="FINALIZING"; dump(s); rc=run("finalize",[PYTHON,"scripts/phase3f_finalize.py"],0)
 s["stages"]["finalize"]={"returncode":rc,"finished_at":time.time()}; s["status"]="COMPLETE" if rc==0 else "FAILED_FINALIZE"; s["finished_at"]=time.time(); dump(s); return rc
if __name__=="__main__": raise SystemExit(main())
