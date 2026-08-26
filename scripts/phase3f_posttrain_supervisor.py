#!/usr/bin/env python3
"""Wait for Phase 3F training, select AOGD, and conditionally run matched SFT."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/phase3f_autonomous_oracle_grounding_distillation"
PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"

def dump(value):
    path=OUT/"posttrain_supervisor/state.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")

def run(name,command,gpu):
    log_path=OUT/f"posttrain_supervisor/{name}.log"; log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("a",encoding="utf-8") as log:
        return subprocess.run(command,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},stdout=log,stderr=subprocess.STDOUT).returncode

def wait_json(path,predicate,state,key):
    while True:
        if path.is_file():
            try:
                value=json.loads(path.read_text())
                if predicate(value): return value
                if str(value.get("status","")).startswith("FAILED"): raise RuntimeError(f"upstream failed: {value}")
            except json.JSONDecodeError: pass
        state["last_wait_at"]=time.time(); state["waiting_for"]=key; dump(state); time.sleep(60)

def main():
    state={"status":"WAITING_FOR_AOGD_TRAINING","started_at":time.time()}; dump(state)
    wait_json(OUT/"supervisor/state.json",lambda x:x.get("status")=="TRAINING_COMPLETE",state,"AOGD_training")
    state["status"]="SELECTING_AOGD"; dump(state)
    rc=run("aogd_selector",[PYTHON,"scripts/phase3f_validate_select.py","--arm","AOGD","--physical-gpu","0"],0)
    state["aogd_selector_returncode"]=rc; dump(state)
    if rc: state["status"]="FAILED_AOGD_SELECTOR"; dump(state); return 1
    selector=json.loads((OUT/"evaluation/selector/AOGD_selector.json").read_text())
    trigger=bool(selector["trigger_matched_p3f_sft_control"]); state["matched_sft_trigger"]=trigger; dump(state)
    if not trigger:
        state["status"]="COMPLETE_AOGD_NOT_SUPPORTED_NO_MATCHED_SFT"; state["finished_at"]=time.time(); dump(state); return 0
    state["status"]="RUNNING_CONDITIONALLY_AUTHORIZED_MATCHED_SFT"; dump(state)
    rc=run("matched_sft_supervisor",[PYTHON,"scripts/phase3f_sft_supervisor.py"],2)
    state["matched_sft_returncode"]=rc; dump(state)
    if rc: state["status"]="FAILED_MATCHED_SFT"; dump(state); return 1
    state["status"]="SELECTING_MATCHED_SFT"; dump(state)
    rc=run("matched_sft_selector",[PYTHON,"scripts/phase3f_validate_select.py","--arm","P3F_SFT_CONT","--physical-gpu","2"],2)
    state["matched_sft_selector_returncode"]=rc
    state["status"]="COMPLETE_WITH_MATCHED_SFT" if rc==0 else "FAILED_MATCHED_SFT_SELECTOR"
    state["finished_at"]=time.time(); dump(state); return rc

if __name__=="__main__": raise SystemExit(main())
