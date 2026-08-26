#!/usr/bin/env python3
"""Run remaining Phase 3F validation conditions without repeating frozen P1 TF."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/"configs/phase3f_autonomous_oracle_grounding_distillation.yaml").read_text())
OUT=ROOT/CFG["experiment"]["output_root"]
MANIFEST=ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/validation_manifest"

def dump(value):
    path=OUT/"final_supervisor/condition_state.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")

def launch(name,gpu,checkpoint,step,epoch,modes,output):
    command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--config",str(ROOT/CFG["source"]["model_config"]),
      "--checkpoint",str(checkpoint),"--output-dir",str(output),"--manifest-dir",str(MANIFEST),"--device","cuda:0",
      "--modes",*modes,"--expected-step",str(step),"--expected-epoch",str(epoch),"--generation-batch-size","1",
      "--skip-spatial-save","--tf-user-prompt","canonical","--reset"]
    log_path=OUT/f"final_supervisor/{name}.log"; log_path.parent.mkdir(parents=True,exist_ok=True); log=log_path.open("a")
    process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),
      "PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},stdout=log,stderr=subprocess.STDOUT)
    return process,log,command,str(log_path)

def main():
    selector=json.loads((OUT/"evaluation/selector/AOGD_selector.json").read_text())
    if int(selector["optimizer_step"])!=1000: raise RuntimeError("frozen selected AOGD step changed")
    prior_p1_tf=ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/final/P1_FROZEN/summary.json"
    if not prior_p1_tf.is_file(): raise FileNotFoundError(prior_p1_tf)
    prior=json.loads(prior_p1_tf.read_text())
    if prior.get("checkpoint_file_sha256")!=CFG["source"]["checkpoint_sha256"]: raise RuntimeError("prior P1 TF provenance mismatch")
    jobs={
      "P1_PHRASE_ONLY":(0,Path(CFG["source"]["checkpoint"]),3500,7,["phrase_only"],OUT/"evaluation/final/P1_FROZEN_PHRASE_ONLY"),
      "AOGD_SELECTED":(2,Path(selector["selected_checkpoint"]),1000,10,["phrase_only","tf_full_context"],OUT/"evaluation/final/AOGD_SELECTED_STEP_1000"),
    }
    state={"status":"RUNNING","started_at":time.time(),"P1_TF_reused":True,"P1_TF_source":str(prior_p1_tf),"jobs":{}}
    running={}
    for name,(gpu,checkpoint,step,epoch,modes,output) in jobs.items():
        proc,log,cmd,log_path=launch(name,gpu,checkpoint,step,epoch,modes,output); running[name]=(proc,log)
        state["jobs"][name]={"gpu":gpu,"checkpoint":str(checkpoint),"step":step,"epoch":epoch,"modes":modes,
          "output":str(output),"command":cmd,"log":log_path,"status":"RUNNING"}
    dump(state); failed=False
    for name,(proc,log) in running.items():
        rc=proc.wait(); log.close(); state["jobs"][name]["returncode"]=rc
        state["jobs"][name]["status"]="COMPLETE" if rc==0 else "FAILED"; failed|=rc!=0; dump(state)
    state["status"]="FAILED" if failed else "COMPLETE"; state["finished_at"]=time.time(); dump(state)
    return 1 if failed else 0

if __name__=="__main__": raise SystemExit(main())
