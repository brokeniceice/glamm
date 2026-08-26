#!/usr/bin/env python3
"""Resume-safe automatic Phase 4C-C baseline gate, interventions, and analysis."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];PY="/home/yz/miniconda3/envs/glamm_official/bin/python";OUT=ROOT/"outputs/phase4c_c_evidence_attribution";STATUS=OUT/"supervisor_status.json"
def status(stage,value="RUNNING",**extra):STATUS.write_text(json.dumps({"status":value,"stage":stage,"updated_unix":time.time(),**extra},indent=2)+"\n")
def launch(name,args,gpu):
 env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);log=(OUT/f"{name}.log").open("a");p=subprocess.Popen([PY,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);return name,p,log
def wait_jobs(stage,jobs):
 active=list(jobs)
 while active:
  status(stage,processes={n:p.pid for n,p,_ in active})
  for item in list(active):
   n,p,l=item;c=p.poll()
   if c is None:continue
   l.close();active.remove(item)
   if c:
    for _,q,h in active:q.terminate();h.close()
    raise RuntimeError(f"{n} failed: {c}")
  if active:time.sleep(15)
def run(name,args,gpu=None):
 env=os.environ.copy()
 if gpu is not None:env["CUDA_VISIBLE_DEVICES"]=str(gpu)
 status(name)
 with (OUT/f"{name}.log").open("a") as log:r=subprocess.run([PY,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f"{name} failed: {r.returncode}")
def wait_external_baselines():
 targets=[OUT/"baseline_reproduction_clip_reader.json",OUT/"baseline_reproduction_forensic_reader.json"]
 while not all(p.exists() for p in targets):
  status("baseline_reproduction")
  probe=subprocess.run(["pgrep","-f","phase4c_c_evaluate.py.*--mode baseline"],stdout=subprocess.DEVNULL)
  if probe.returncode and not all(p.exists() for p in targets):raise RuntimeError("baseline process ended without manifests")
  time.sleep(15)
def main():
 wait_external_baselines();run("preflight_finalize",["scripts/phase4c_c_preflight_finalize.py"])
 jobs=[]
 for arm,gpu in (("clip_reader",0),("forensic_reader",1)):
  marker=OUT/f"intervention_{arm}.json"
  if not marker.exists():jobs.append(launch(f"interventions_{arm}",["scripts/phase4c_c_evaluate.py","--arm",arm,"--mode","interventions","--device","cuda:0"],gpu))
 wait_jobs("intervention_matrix",jobs);run("analyze",["scripts/phase4c_c_analyze.py"]);status("complete","COMPLETE")
if __name__=="__main__":
 try:main()
 except Exception as e:status("failed","FAILED",error=repr(e));raise
