#!/usr/bin/env python3
"""Automatically continue Phase 4B-G from training through validation and final report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.phase4b import dump


def load(path): return json.loads(Path(path).read_text())
def log(path,message):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} {message}\n"); handle.flush()


def run(command,gpu,log_path):
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"
    with log_path.open("a",encoding="utf-8") as handle:
        return subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]
    logs=out/"logs"; logs.mkdir(parents=True,exist_ok=True); supervisor_log=logs/"supervisor.log"
    dump(out/"supervisor_state.json",{"status":"WAITING_FOR_TRAINING","pid":os.getpid(),"started":time.time()})
    while True:
        summaries=[]
        for arm in ("PROJ-ONLY","PROJ-LORA"):
            path=out/"experiments"/arm/"run_summary.json"
            summaries.append(load(path) if path.is_file() else None)
        if all(value and value.get("status")=="COMPLETE" and int(value.get("final_step",0))==3000 for value in summaries): break
        time.sleep(60)
    log(supervisor_log,"both training arms complete; starting candidate validation")
    dump(out/"supervisor_state.json",{"status":"VALIDATING_CANDIDATES","pid":os.getpid()})
    jobs=[]
    for arm,gpu in (("PROJ-ONLY",0),("PROJ-LORA",1)):
        jobs.append((arm,run([sys.executable,str(ROOT/"scripts/phase4b_validate_select.py"),"--arm",arm,"--physical-gpu",str(gpu)],gpu,logs/f"validate_{arm}.log")))
    failures=[]
    for arm,process in jobs:
        code=process.wait()
        if code: failures.append({"arm":arm,"exit_code":code})
    if failures:
        dump(out/"supervisor_state.json",{"status":"FAILED_VALIDATION","failures":failures}); raise SystemExit(1)
    log(supervisor_log,"candidate validation complete; running paired analysis")
    subprocess.run([sys.executable,str(ROOT/"scripts/phase4b_analyze.py")],cwd=ROOT,check=True,
                   stdout=(logs/"analysis.log").open("a"),stderr=subprocess.STDOUT)
    route=load(out/"route_gate.json")
    if route.get("matched_sft_triggered"):
        dump(out/"supervisor_state.json",{"status":"MATCHED_SFT_TRIGGERED","reason":"significant PROJ-LORA vs P1 G0 gain"})
        log(supervisor_log,"matched SFT trigger reached; matched continuation is required")
        # The matched branch is deliberately a separate condition. Its launcher
        # is installed before the trigger can be acted upon; never substitute a historical SFT.
        subprocess.run([sys.executable,str(ROOT/"scripts/phase4b_matched_sft.py"),"--physical-gpu","0"],cwd=ROOT,check=True,
                       env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},
                       stdout=(logs/"matched_sft.log").open("a"),stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,str(ROOT/"scripts/phase4b_matched_validate.py"),"--physical-gpu","0"],cwd=ROOT,check=True,
                       env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},
                       stdout=(logs/"matched_validate.log").open("a"),stderr=subprocess.STDOUT)
    # Selected oracle diagnostics: reuse the already canonical P1 validation population,
    # but rerun both diagnostics with the same current evaluator for exact matching.
    lora=load(out/"selector_proj_lora.json"); selected=Path(lora["selected_checkpoint"]); step=int(lora["optimizer_step"])
    cache=load(out/"fepn_global_feature_cache_manifest.json")["splits"]["val"]
    manifest=str((out/"evaluation/validation_manifest").resolve())
    common=["--config",str(ROOT/cfg["source"]["model_config"]),"--manifest-dir",manifest,"--device","cuda:0",
            "--modes","phrase_only","tf_full_context","--generation-batch-size","1","--skip-spatial-save","--reset",
            "--tf-user-prompt","canonical"]
    p1_command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--checkpoint",cfg["source"]["checkpoint"],
      "--output-dir",str(out/"evaluation/final/P1_FROZEN"),"--expected-step","3500","--expected-epoch","7",*common]
    lora_command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--checkpoint",str(selected),
      "--output-dir",str(out/"evaluation/final/PROJ-LORA"),"--expected-step",str(step),"--expected-epoch",str(step//500),*common,
      "--forensic-feature-cache",cache["path"],"--forensic-feature-cache-sha256",cache["sha256"],
      "--forensic-projector-checkpoint",str(selected)]
    diag=[("P1",run(p1_command,0,logs/"diagnostic_P1.log")),("PROJ-LORA",run(lora_command,1,logs/"diagnostic_PROJ-LORA.log"))]
    failed=[{"arm":name,"exit_code":process.wait()} for name,process in diag if process.wait()!=0]
    if failed: dump(out/"supervisor_state.json",{"status":"FAILED_DIAGNOSTICS","failures":failed}); raise SystemExit(1)
    # Consolidate mandatory audit filenames after both arm-specific audits exist.
    gradient={arm:load(out/f"audits/preflight_gradient_{arm}.json") for arm in ("proj-only","proj-lora")}
    functional={arm:load(out/f"audits/one_step_functional_{arm}.json") for arm in ("proj-only","proj-lora")}
    dump(out/"preflight_gradient_audit.json",{"status":"PASS" if all(v["status"]=="PASS" for v in gradient.values()) else "FAIL","arms":gradient})
    dump(out/"one_step_functional_audit.json",{"status":"PASS" if all(v["status"]=="PASS" for v in functional.values()) else "FAIL","arms":functional})
    subprocess.run([sys.executable,str(ROOT/"scripts/phase4b_finalize.py")],cwd=ROOT,check=True,
                   stdout=(logs/"finalize.log").open("a"),stderr=subprocess.STDOUT)
    completion=load(out/"completion_manifest.json")
    dump(out/"supervisor_state.json",{"status":completion["status"],"completed":time.time(),"final_gate":completion.get("final_gate")})
    log(supervisor_log,f"hard stop: {completion['status']} {completion.get('final_gate')}")


if __name__=="__main__": main()
