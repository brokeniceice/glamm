#!/usr/bin/env python3
"""Run the authorized Phase6N0 train -> internal-DEV finalizer once."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.phase6l0_r2_preflight import dump, require

OUT=ROOT/"outputs/phase6n0_low_rank_evidence_r2"
STATUS=OUT/"pipeline_status.json"
PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"


def main():
    gates=json.loads((OUT/"preflight_gates.json").read_text())
    require(gates["status"]=="PASS" and all(gates[k]=="PASS" for k in gates if k.startswith("G")),
            "formal Phase6N0 preflight incomplete")
    require(not STATUS.exists() and not (OUT/"selector.json").exists(),
            "existing Phase6N0 supervisor or selector; refusing duplicate launch")
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="1";env["PYTHONUNBUFFERED"]="1"
    stages=(("TRAIN",ROOT/"scripts/phase6n0_low_rank_r2_train.py","train","train.log"),
            ("FINALIZE",ROOT/"scripts/phase6n0_low_rank_r2_finalize.py",None,"finalize.log"))
    for label,path,argument,log_name in stages:
        dump(STATUS,{"status":"RUNNING","stage":label,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                     "physical_gpu":1,"log":str(OUT/"logs"/log_name)})
        command=[PYTHON,"-u",str(path)]
        if argument:command.append(argument)
        log_path=OUT/"logs"/log_name;log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open("w") as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            dump(STATUS,{"status":"FAILED","stage":label,"exit_code":result.returncode,
                         "log":str(log_path),"failed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
            raise SystemExit(result.returncode)
    summary=json.loads((OUT/"summary.json").read_text())
    require(summary["status"]=="COMPLETE_STOP_AFTER_INTERNAL_DEV", "finalizer did not seal Phase6N0")
    dump(STATUS,{"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV","selected_epoch":summary["selected_epoch"],
                 "report":summary["report"],"physical_gpu":1,
                 "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    print(json.dumps({"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV",
                      "selected_epoch":summary["selected_epoch"]}),flush=True)


if __name__=="__main__":main()
