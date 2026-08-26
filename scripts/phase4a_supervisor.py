#!/usr/bin/env python3
"""Unattended Phase 4A training and finalization supervisor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"
OUT=ROOT/"outputs/phase4a_fepn_evidence_learnability"
LOG=OUT/"training/supervisor.jsonl"


def record(value):
 LOG.parent.mkdir(parents=True,exist_ok=True)
 with LOG.open("a",encoding="utf-8") as handle: handle.write(json.dumps({"time":time.time(),**value})+"\n")
 print(json.dumps(value),flush=True)


def run(command,attempt):
 env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]="0"
 record({"event":"START","attempt":attempt,"command":command})
 result=subprocess.run(command,cwd=ROOT,env=env)
 record({"event":"EXIT","attempt":attempt,"returncode":result.returncode})
 return result.returncode


def main():
 preflight=json.loads((OUT/"preflight_gradient_audit.json").read_text())
 if preflight["status"]!="PASS": raise RuntimeError("PASS preflight required")
 train=[PYTHON,"scripts/phase4a_train.py","--mode","train","--physical-gpu","0"]
 for attempt in range(1,4):
  if run(train,attempt)==0: break
  if (OUT/"training/safety_stop.json").is_file():
   record({"event":"SAFETY_STOP","artifact":str(OUT/"training/safety_stop.json")}); return 4
  time.sleep(5)
 else:
  record({"event":"BLOCKED","stage":"training","reason":"three failed resumable attempts"}); return 2
 finalize=[PYTHON,"scripts/phase4a_finalize.py"]
 for attempt in range(1,4):
  if run(finalize,attempt)==0:
   record({"event":"COMPLETE","completion_manifest":str(OUT/"completion_manifest.json")}); return 0
  time.sleep(5)
 record({"event":"BLOCKED","stage":"finalize","reason":"three failed attempts"}); return 3


if __name__=="__main__": raise SystemExit(main())
