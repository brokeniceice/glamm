#!/usr/bin/env python3
"""Canonical direct-batch1 validation selector and P1 improvement trigger for Phase 3F."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from eval.phase3a_metrics import parse_phrase_aligned_generation
from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d0 import parse_structure


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def evaluate(checkpoint, step, epoch, output, cfg):
    if (output/"summary.json").is_file(): return
    manifest = ROOT / "outputs/phase3e_joint_language_mask_posttraining/evaluation/validation_manifest"
    command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--config",str(ROOT/cfg["source"]["model_config"]),
        "--checkpoint",str(checkpoint),"--output-dir",str(output),"--manifest-dir",str(manifest),"--device","cuda:0",
        "--modes","detection","G0","--expected-step",str(step),"--expected-epoch",str(epoch),
        "--generation-batch-size","1","--skip-spatial-save","--reset","--detection-user-prompt","canonical"]
    subprocess.run(command,cwd=ROOT,check=True,env=os.environ.copy())


def metrics(output, checkpoint, step):
    g0=load(output/"G0/metrics.json"); summary=load(output/"summary.json"); predictions=rows(output/"G0/predictions.jsonl")
    structure=[]
    for prediction in predictions:
        parsed=parse_phrase_aligned_generation(prediction.get("decoded_text") or "")
        structure.append(parse_structure(parsed,prediction.get("generated_token_ids") or [],32004))
    detection=summary["modes"]["detection"]["classification_head"]
    mean=lambda key: float(np.mean([row[key] for row in structure]))
    value={"optimizer_step":step,"checkpoint":str(checkpoint),"checkpoint_sha256":file_sha256(checkpoint),
        "mean_foreground_iou":g0["per_image_mean"]["foreground_iou"],
        "mean_foreground_f1":g0["per_image_mean"]["foreground_f1"],
        "classification_accuracy":detection["accuracy"],"classification_f1":detection["f1"],
        "structure_validity":mean("structural_validity"),"valid_seg_rate":mean("usable_seg"),
        "malformed_output_rate":float(np.mean([not row["structural_validity"] for row in structure]))}
    dump(output/"selection_metrics.json",value); return value


def paired_bootstrap(selected_path, baseline_path, repeats, seed):
    selected={row["sample_id"]:row for row in rows(selected_path)}
    baseline={row["sample_id"]:row for row in rows(baseline_path)}
    ids=sorted(set(selected)&set(baseline))
    if set(selected)!=set(baseline): raise RuntimeError("selector prediction sets differ")
    delta=np.asarray([selected[sid]["foreground_iou"]-baseline[sid]["foreground_iou"] for sid in ids],dtype=float)
    rng=np.random.default_rng(seed); means=[]
    for _ in range(repeats): means.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
    return {"n":len(ids),"mean_difference":float(delta.mean()),"bootstrap_95ci":[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],
            "wins":int((delta>0).sum()),"ties":int((delta==0).sum()),"losses":int((delta<0).sum()),"repeats":repeats,"seed":seed}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",choices=("AOGD","P3F_SFT_CONT"),default="AOGD")
    parser.add_argument("--physical-gpu",type=int,default=0); cli=parser.parse_args()
    cfg=yaml.safe_load((ROOT/"configs/phase3f_autonomous_oracle_grounding_distillation.yaml").read_text())
    expected=int(cfg["runtime"]["physical_gpu"] if cli.arm=="AOGD" else cfg["runtime"]["sft_control_gpu"])
    if cli.physical_gpu!=expected or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,str(expected)): raise RuntimeError("GPU assignment mismatch")
    root=ROOT/cfg["experiment"]["output_root"]; eval_root=root/"evaluation/selector"
    p1=Path(cfg["source"]["checkpoint"]); p1_out=eval_root/"P1_FROZEN"
    evaluate(p1,int(cfg["source"]["optimizer_step"]),int(cfg["source"]["epoch"]),p1_out,cfg)
    baseline=metrics(p1_out,p1,0); baseline["eligible"]=True; baseline["non_regression"]={"initialization_reference":True}
    candidates=[]
    arm_root=Path(cfg["experiment"]["checkpoint_root"])
    if cli.arm=="P3F_SFT_CONT": arm_root=arm_root/"P3F_SFT_CONT_MATCHED"
    for step in map(int,cfg["selector"]["candidate_steps"]):
        if step==0: continue
        checkpoint=arm_root/f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        output=eval_root/cli.arm/f"step_{step:04d}"; evaluate(checkpoint,step,step//100,output,cfg)
        value=metrics(output,checkpoint,step)
        value["non_regression"]={
            "classification_accuracy":value["classification_accuracy"]>=baseline["classification_accuracy"]-float(cfg["selector"]["classification_accuracy_max_drop"]),
            "classification_f1":value["classification_f1"]>=baseline["classification_f1"]-float(cfg["selector"]["classification_f1_max_drop"]),
            "structure_validity":value["structure_validity"]>=baseline["structure_validity"]-float(cfg["selector"]["structure_validity_max_drop"]),
            "no_seg_or_malformed_collapse":value["valid_seg_rate"]>0 and value["malformed_output_rate"]<1}
        value["eligible"]=all(value["non_regression"].values()); dump(output/"selection_metrics.json",value); candidates.append(value)
    eligible=[baseline,*[row for row in candidates if row["eligible"]]]
    selected=max(eligible,key=lambda row:(row["mean_foreground_iou"],row["mean_foreground_f1"],-row["optimizer_step"]))
    selected_out=p1_out if selected["optimizer_step"]==0 else eval_root/cli.arm/f"step_{selected['optimizer_step']:04d}"
    contrast=paired_bootstrap(selected_out/"G0/predictions.jsonl",p1_out/"G0/predictions.jsonl",
                              int(cfg["evaluation"]["bootstrap_repeats"]),int(cfg["evaluation"]["bootstrap_seed"]))
    supported=selected["optimizer_step"]>0 and selected["eligible"] and contrast["mean_difference"]>0 and contrast["bootstrap_95ci"][0]>0
    result={"status":"FROZEN_AFTER_COMPLETE_INTERNAL_VALIDATION","arm":cli.arm,
        "selector":"max_val_fake_canonical_G0_mean_FG_IoU_then_F1_after_nonregression_gates",
        "baseline":baseline,"selected_checkpoint":selected["checkpoint"],"checkpoint_sha256":selected["checkpoint_sha256"],
        "optimizer_step":selected["optimizer_step"],"selected_metrics":selected,"candidates":candidates,
        "selected_vs_p1_paired_bootstrap":contrast,"statistically_supported_positive_vs_p1":supported,
        "trigger_matched_p3f_sft_control":bool(cli.arm=="AOGD" and supported),"fallback_to_P1":selected["optimizer_step"]==0,
        "training_loss_used":False,"TF_used":False,"internal_test_used":False,"official1000_used":False,
        "threshold_tuned":False,"detection_user_prompt":"canonical","g0_user_prompt":"canonical","generation_batch_size":1}
    dump(eval_root/f"{cli.arm}_selector.json",result); print(json.dumps({k:result[k] for k in ("arm","optimizer_step","selected_vs_p1_paired_bootstrap","statistically_supported_positive_vs_p1","trigger_matched_p3f_sft_control")},indent=2))

if __name__=="__main__": main()
