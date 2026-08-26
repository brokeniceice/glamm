#!/usr/bin/env python3
"""Resumable automatic supervisor for the frozen Phase 4E-1 method study."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
OUT = ROOT / "outputs/phase4e1_tf_fdg_full_method"
STATE = OUT / "supervisor_state.json"


def dump(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"status": "RUNNING", "stages": {}, "started_at": time.time()}


def run(state, name, args, gpu):
    if state["stages"].get(name, {}).get("status") == "COMPLETE": return
    log = OUT / "logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    state["current_stage"] = name; state["stages"][name] = {"status": "RUNNING", "started_at": time.time(), "gpu": gpu, "log": str(log)}; dump(state)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["PYTHONUNBUFFERED"] = "1"
    with log.open("a", encoding="utf-8") as handle:
        value = subprocess.run([PYTHON, *args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    state["stages"][name].update({"status": "COMPLETE" if value.returncode == 0 else "FAILED", "returncode": value.returncode, "finished_at": time.time()}); dump(state)
    if value.returncode:
        state["status"] = "SAFETY_STOP"; state["failure_stage"] = name; dump(state); raise RuntimeError(f"{name} failed; see {log}")


def parallel(state, jobs):
    pending = [(name,args,gpu) for name,args,gpu in jobs if state["stages"].get(name,{}).get("status") != "COMPLETE"]
    if not pending: return
    processes=[]
    for name,args,gpu in pending:
        log=OUT/"logs"/f"{name}.log";log.parent.mkdir(parents=True,exist_ok=True);handle=log.open("a",encoding="utf-8");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);env["PYTHONUNBUFFERED"]="1"
        state["stages"][name]={"status":"RUNNING","started_at":time.time(),"gpu":gpu,"log":str(log)}
        processes.append((name,subprocess.Popen([PYTHON,*args],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT),handle))
    state["current_stage"]=[x[0] for x in processes];dump(state);failed=[]
    for name,process,handle in processes:
        code=process.wait();handle.close();state["stages"][name].update({"status":"COMPLETE" if code==0 else "FAILED","returncode":code,"finished_at":time.time()});failed += [name] if code else []
    dump(state)
    if failed: state["status"]="SAFETY_STOP";state["failure_stage"]=failed;dump(state);raise RuntimeError(f"parallel stages failed: {failed}")


def teacher_then_student(state, arm, gpu):
    run(state,f"{arm}_preflight",["scripts/phase4e1_run.py","--mode","preflight","--arm",arm,"--device","cuda:0"],gpu)
    run(state,f"{arm}_stage_t",["scripts/phase4e1_run.py","--mode","stage_t","--arm",arm,"--device","cuda:0"],gpu)
    run(state,f"{arm}_qualification",["scripts/phase4e1_run.py","--mode","qualify","--arm",arm,"--device","cuda:0"],gpu)
    qualification=json.loads((OUT/"qualification"/arm/"qualification.json").read_text())
    if qualification["TEACHER_MASK_CAPABILITY"] == "FAILED":
        state["stages"][f"{arm}_stage_s"]={"status":"NOT_STARTED_TEACHER_FAILED"};dump(state);return
    run(state,f"{arm}_stage_s",["scripts/phase4e1_run.py","--mode","stage_s","--arm",arm,"--device","cuda:0"],gpu)
    run(state,f"{arm}_evaluation",["scripts/phase4e1_run.py","--mode","evaluate","--arm",arm,"--device","cuda:0"],gpu)


def main():
    state=load();state["status"]="RUNNING";dump(state)
    train_complete=Path("/data/yz/groundingLMM_official/cache/phase4e1_tf_fdg_raw_hidden/train/complete.json")
    while not train_complete.exists():
        state["current_stage"]="waiting_train_hidden_cache";state["heartbeat"]=time.time();dump(state);time.sleep(30)
    run(state,"val_hidden_cache",["scripts/phase4e1_cache_hidden.py","--split","val","--device","cuda:0"],1)
    teacher_then_student(state,"full",0)
    # Mandatory Tier-1 arms. Run the two teacher-dependent structural arms concurrently.
    parallel(state,[
        ("no_forensic_bundle",["scripts/phase4e1_arm_bundle.py","--arm","no_forensic","--device","cuda:0"],0),
        ("k1_bundle",["scripts/phase4e1_arm_bundle.py","--arm","k1","--device","cuda:0"],1),
    ])
    # Matched raw-CLIP and true no-teacher controls can also run concurrently.
    parallel(state,[
        ("full_clip_bundle",["scripts/phase4e1_arm_bundle.py","--arm","full_clip","--device","cuda:0"],0),
        ("no_teacher_bundle",["scripts/phase4e1_arm_bundle.py","--arm","no_teacher","--device","cuda:0"],1),
    ])
    run(state,"finalize",["scripts/phase4e1_finalize.py"],0)
    state["status"]="COMPLETE";state["current_stage"]=None;state["finished_at"]=time.time();dump(state)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        state=load();state["status"]="SAFETY_STOP";state["error"]=repr(error);state["finished_at"]=time.time();dump(state);raise
