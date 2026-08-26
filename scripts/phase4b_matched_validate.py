#!/usr/bin/env python3
"""Validate conditionally triggered matched SFT and compare it with selected PROJ-LORA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.phase4b_analyze import paired_bootstrap
from scripts.phase4b_validate_select import derive_metrics
from tools.phase3d0r import FrozenSentenceEncoder
from tools.phase4b import dump,load_jsonl


def load(path): return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--physical-gpu",type=int,default=0); cli=parser.parse_args()
    if cli.physical_gpu!=0 or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"","0"): raise RuntimeError("matched validation uses GPU0")
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]
    baseline=load(ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/selector/P1_FROZEN/selection_metrics.json")
    qcfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text()); enc=qcfg["semantic_encoder"]
    encoder=FrozenSentenceEncoder(enc["model_id"],enc["revision"],enc["cache_dir"],"cpu"); idf=load(ROOT/"outputs/phase3d0r_reward_reformulation/token_idf.json")["values"]
    manifest_rows=load_jsonl(ROOT/cfg["data"]["manifest_dir"]/"val_combined.jsonl"); candidates=[]; margin=.005
    for step in map(int,cfg["selector"]["candidate_steps"]):
        checkpoint=Path(cfg["experiment"]["checkpoint_root"])/"matched_sft"/f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
        output=out/"matched_sft/evaluation"/f"step_{step:04d}"
        if not (output/"summary.json").is_file():
            command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--config",str(ROOT/cfg["source"]["model_config"]),
              "--checkpoint",str(checkpoint),"--output-dir",str(output),"--manifest-dir",str(out/"evaluation/validation_manifest"),
              "--device","cuda:0","--modes","detection","G0","--expected-step",str(step),"--expected-epoch",str(step//500),
              "--generation-batch-size","1","--skip-spatial-save","--reset","--detection-user-prompt","canonical"]
            subprocess.run(command,cwd=ROOT,check=True,env=os.environ.copy())
        value=derive_metrics(output,checkpoint,step,manifest_rows,encoder,idf)
        value["non_regression"]={"classification_accuracy":value["classification_accuracy"]>=baseline["classification_accuracy"]-margin,
          "classification_f1":value["classification_f1"]>=baseline["classification_f1"]-margin,
          "structure_validity":value["structure_validity"]>=baseline["structure_validity"]-margin,
          "valid_seg_rate":value["valid_seg_rate"]>=baseline["valid_seg_rate"]-margin}
        value["eligible"]=all(value["non_regression"].values()); candidates.append(value)
    eligible=[row for row in candidates if row["eligible"]]
    if not eligible: raise RuntimeError("no selectable matched SFT checkpoint")
    selected=max(eligible,key=lambda r:(r["mean_foreground_iou"],r["mean_foreground_f1"],r["R_phrase_sem"],-r["optimizer_step"]))
    selector={"status":"COMPLETE","selected":selected,"candidates":candidates,"same_selector_as_evidence_arms":True,
              "internal_test_used":False,"official1000_used":False}; dump(out/"matched_sft/selector.json",selector)
    lora=load(out/"selector_proj_lora.json"); lora_predictions=out/"evaluation/selector/PROJ-LORA"/f"step_{int(lora['optimizer_step']):04d}/G0/predictions.jsonl"
    sft_predictions=out/"matched_sft/evaluation"/f"step_{int(selected['optimizer_step']):04d}/G0/predictions.jsonl"
    comparison=paired_bootstrap(lora_predictions,sft_predictions,int(cfg["evaluation"]["bootstrap_repeats"]),int(cfg["evaluation"]["bootstrap_seed"])+2,"PROJ-LORA_vs_MATCHED-SFT")
    dump(out/"paired_bootstrap_proj_lora_vs_sft.json",comparison); print(json.dumps(selector,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
