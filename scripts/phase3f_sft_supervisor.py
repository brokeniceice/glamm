#!/usr/bin/env python3
"""Unattended matched 4500-step Phase 3F SFT-control supervisor."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3f_autonomous_oracle_grounding_distillation"
CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase3f_autonomous_oracle_grounding_distillation/P3F_SFT_CONT_MATCHED")
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"

def dump(value):
    path = OUT / "supervisor/sft_state.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def latest():
    rows=[]
    for path in CKPT.glob("step_*/checkpoint/mp_rank_00_model_states.pt"):
        try: rows.append((int(path.parents[1].name.split("_")[-1]), path))
        except ValueError: pass
    return max(rows, default=(0,None))

def main():
    state={"status":"RUNNING","started_at":time.time(),"attempts":[]}; dump(state)
    log_path=OUT/"supervisor/logs/sft_control.log"; log_path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,4):
        step,path=latest(); command=[PYTHON,"scripts/phase3f_train.py","--mode","sft-control","--physical-gpu","2"]
        if step and path: command += ["--resume",str(path)]
        with log_path.open("a",encoding="utf-8") as log:
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"2",
                "PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"},stdout=log,stderr=subprocess.STDOUT)
        state["attempts"].append({"attempt":attempt,"resume_step":step,"returncode":result.returncode}); dump(state)
        summary=OUT/"experiments/P3F_SFT_CONT_MATCHED/run_summary.json"
        if result.returncode==0 and summary.is_file() and json.loads(summary.read_text())["status"]=="COMPLETE":
            state["status"]="COMPLETE"; state["finished_at"]=time.time(); dump(state); return 0
    state["status"]="FAILED_AFTER_RETRIES"; state["finished_at"]=time.time(); dump(state); return 1

if __name__ == "__main__": raise SystemExit(main())
