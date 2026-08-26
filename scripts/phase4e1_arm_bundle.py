#!/usr/bin/env python3
"""Run one complete matched Tier-1 arm on its assigned visible GPU."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON="/home/yz/miniconda3/envs/glamm_official/bin/python"
def call(mode,arm,device): subprocess.run([PYTHON,"scripts/phase4e1_run.py","--mode",mode,"--arm",arm,"--device",device],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--arm",required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
 call("preflight",a.arm,a.device)
 if a.arm!="no_teacher":
  call("stage_t",a.arm,a.device);call("qualify",a.arm,a.device)
  q=json.loads((ROOT/f"outputs/phase4e1_tf_fdg_full_method/qualification/{a.arm}/qualification.json").read_text())
  if q["TEACHER_MASK_CAPABILITY"]=="FAILED":return
 call("stage_s",a.arm,a.device);call("evaluate",a.arm,a.device)
if __name__=="__main__":main()
