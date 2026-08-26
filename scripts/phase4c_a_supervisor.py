#!/usr/bin/env python3
"""Unattended matched-arm execution and finalization for Phase 4C-A."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY="/home/yz/miniconda3/envs/glamm_official/bin/python"; OUT=ROOT/"outputs/phase4c_a_clip_forensic_adapter"; LOG=OUT/"training/supervisor.jsonl"
def record(v):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as h: h.write(json.dumps({"time":time.time(),**v})+"\n")
    print(json.dumps(v),flush=True)
def launch(arm,gpu):
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    (OUT/"training").mkdir(parents=True,exist_ok=True)
    log=(OUT/"training"/f"{arm}.log").open("a")
    cmd=[PY,"scripts/phase4c_a_train.py","--mode","train","--arm",arm,"--device","cuda:0"]
    record({"event":"START","arm":arm,"gpu":gpu,"command":cmd}); return subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log
def main():
    if json.loads((OUT/"preflight_gradient_audit.json").read_text())["status"]!="PASS": raise RuntimeError("PASS preflight required")
    pending={"clip_proj":0,"forensic_adapter":1}
    for attempt in range(1,4):
        running={arm:(*launch(arm,gpu),gpu) for arm,gpu in pending.items()}; failed={}
        for arm,(proc,log,gpu) in running.items():
            code=proc.wait(); log.close(); record({"event":"EXIT","arm":arm,"gpu":gpu,"attempt":attempt,"returncode":code})
            if code: failed[arm]=gpu
        if not failed: break
        if any((OUT/"training"/arm/"safety_stop.json").exists() for arm in failed):
            record({"event":"SAFETY_STOP","arms":list(failed)}); return 4
        pending=failed; time.sleep(3)
    else: return 2
    result=subprocess.run([PY,"scripts/phase4c_a_finalize.py"],cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"})
    record({"event":"FINALIZE_EXIT","returncode":result.returncode})
    if result.returncode: return 3
    record({"event":"COMPLETE","manifest":str(OUT/"completion_manifest.json")}); return 0
if __name__=="__main__": raise SystemExit(main())
