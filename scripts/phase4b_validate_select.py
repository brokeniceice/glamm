#!/usr/bin/env python3
"""Direct-batch1 validation of all frozen Phase 4B-G checkpoint candidates."""

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

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.phase3a_metrics import parse_phrase_aligned_generation
from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d0 import parse_structure
from tools.phase3d0r import FrozenSentenceEncoder, content_phrase, content_tokens, semantic_phrase_score
from tools.phase4b import dump, load_jsonl


def load(path): return json.loads(Path(path).read_text())


def semantic_summary(predictions, manifest_rows, encoder, idf):
    by_id={row["sample_id"]:row for row in manifest_rows}; pairs=[]; parsed_rows=[]
    for prediction in predictions:
        reference=UnifiedForensicsDataset.authoritative_localization_field(by_id[prediction["sample_id"]])["normalized_training_phrase"]
        parsed=parse_phrase_aligned_generation(prediction.get("decoded_text") or "")
        pairs.append((reference,parsed.get("target_region"))); parsed_rows.append((prediction,parsed))
    values=set()
    for left,right in pairs:
        for phrase in (left,right): values.add(content_phrase(phrase)); values.update(content_tokens(phrase))
    embeddings=encoder.encode(values,batch_size=128); semantic=[]; structure=[]; detail=[]
    for (reference,phrase),(prediction,parsed) in zip(pairs,parsed_rows):
        score=semantic_phrase_score(reference,phrase,idf,embeddings)
        struct=parse_structure(parsed,prediction.get("generated_token_ids") or [],32004)
        semantic.append(score); structure.append(struct); detail.append({"sample_id":prediction["sample_id"],
          "reference_phrase":reference,"generated_phrase":phrase,**score,**struct})
    mean=lambda key,rows:float(np.mean([row[key] for row in rows]))
    return {"R_phrase_sem":mean("R_phrase_sem",semantic),"R_key_soft":mean("R_key_soft",semantic),
      "R_sentence_sem":mean("R_sentence_sem",semantic),"structure_validity":mean("structural_validity",structure),
      "malformed_output_rate":float(np.mean([not row["structural_validity"] for row in structure])),
      "valid_seg_rate":mean("usable_seg",structure)},detail


def evaluate(checkpoint, step, output, cfg, cache):
    if (output/"summary.json").is_file(): return
    command=[sys.executable,str(ROOT/"scripts/phase3a_evaluate.py"),"--config",str(ROOT/cfg["source"]["model_config"]),
      "--checkpoint",str(checkpoint),"--output-dir",str(output),"--manifest-dir",str((ROOT/cfg["experiment"]["output_root"]/"evaluation/validation_manifest").resolve()),
      "--device","cuda:0","--modes","detection","G0","--expected-step",str(step),"--expected-epoch",str(step//500),
      "--generation-batch-size","1","--skip-spatial-save","--reset","--detection-user-prompt","canonical",
      "--forensic-feature-cache",cache["path"],"--forensic-feature-cache-sha256",cache["sha256"],
      "--forensic-projector-checkpoint",str(checkpoint)]
    subprocess.run(command,cwd=ROOT,check=True,env=os.environ.copy())


def derive_metrics(output,checkpoint,step,manifest_rows,encoder,idf):
    summary=load(output/"summary.json"); g0=summary["modes"]["G0"]; predictions=load_jsonl(output/"G0/predictions.jsonl")
    semantic,detail=semantic_summary(predictions,manifest_rows,encoder,idf)
    with (output/"semantic_structure.jsonl").open("w",encoding="utf-8") as handle:
        for row in detail: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    detection=summary["modes"]["detection"]["classification_head"]
    value={"optimizer_step":step,"checkpoint":str(checkpoint),"checkpoint_sha256":file_sha256(checkpoint),
      "mean_foreground_iou":g0["per_image_mean"]["foreground_iou"],"mean_foreground_f1":g0["per_image_mean"]["foreground_f1"],
      "global_foreground_iou":g0["global_pixel"]["foreground_iou"],"global_foreground_f1":g0["global_pixel"]["foreground_f1"],
      "median_foreground_iou":float(np.median([r["foreground_iou"] for r in predictions])),
      "median_foreground_f1":float(np.median([r["foreground_f1"] for r in predictions])),
      "classification_accuracy":detection["accuracy"],"classification_f1":detection["f1"],
      "classification_precision":detection["precision"],"classification_recall":detection["recall"],
      "classification_confusion":{k:detection[k] for k in ("tp","tn","fp","fn")},**semantic}
    dump(output/"selection_metrics.json",value); return value


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",choices=("PROJ-ONLY","PROJ-LORA"),required=True)
    parser.add_argument("--physical-gpu",type=int,required=True); cli=parser.parse_args()
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); expected=int(cfg["runtime"][f"{cli.arm}_gpu"])
    if cli.physical_gpu!=expected or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"",str(expected)): raise RuntimeError("GPU assignment mismatch")
    root=ROOT/cfg["experiment"]["output_root"]; eval_root=root/"evaluation/selector"; manifest_rows=load_jsonl(ROOT/cfg["data"]["manifest_dir"]/"val_combined.jsonl")
    qcfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text()); enc=qcfg["semantic_encoder"]
    encoder=FrozenSentenceEncoder(enc["model_id"],enc["revision"],enc["cache_dir"],"cpu")
    idf=load(ROOT/"outputs/phase3d0r_reward_reformulation/token_idf.json")["values"]
    baseline_path=ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/selector/P1_FROZEN/selection_metrics.json"
    baseline=load(baseline_path)
    if baseline["checkpoint_sha256"]!=cfg["source"]["checkpoint_sha256"]: raise RuntimeError("P1 baseline hash drift")
    cache=load(root/"fepn_global_feature_cache_manifest.json")["splits"]["val"]
    candidates=[]; margin=float(cfg["selector"]["hard_nonregression_margin"])
    for step in map(int,cfg["selector"]["candidate_steps"]):
        checkpoint=Path(cfg["experiment"]["checkpoint_root"])/cli.arm/f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        output=eval_root/cli.arm/f"step_{step:04d}"; evaluate(checkpoint,step,output,cfg,cache)
        value=derive_metrics(output,checkpoint,step,manifest_rows,encoder,idf)
        value["non_regression"]={
          "classification_accuracy":value["classification_accuracy"]>=baseline["classification_accuracy"]-margin,
          "classification_f1":value["classification_f1"]>=baseline["classification_f1"]-margin,
          "structure_validity":value["structure_validity"]>=baseline["structure_validity"]-margin,
          "valid_seg_rate":value["valid_seg_rate"]>=baseline["valid_seg_rate"]-margin}
        value["eligible"]=all(value["non_regression"].values()); dump(output/"selection_metrics.json",value); candidates.append(value)
    eligible=[row for row in candidates if row["eligible"]]
    selected=max(eligible,key=lambda row:(row["mean_foreground_iou"],row["mean_foreground_f1"],row["R_phrase_sem"],-row["optimizer_step"])) if eligible else baseline
    result={"status":"FROZEN_AFTER_COMPLETE_INTERNAL_VALIDATION","arm":cli.arm,
      "selector":"hard_nonregression_then_max_G0_mean_FG_IoU_then_F1_then_phrase_semantic",
      "baseline":baseline,"selected_checkpoint":selected["checkpoint"],"checkpoint_sha256":selected["checkpoint_sha256"],
      "optimizer_step":selected["optimizer_step"],"selected_metrics":selected,"candidates":candidates,
      "fallback_to_P1":selected is baseline,"training_loss_used":False,"internal_test_used":False,
      "official1000_used":False,"threshold_tuned":False,"direct_batch_size":1,"canonical_prompt":True}
    result["step_0_definition"]="original P1; existing canonical validation reused without rerun"
    result["excluded_candidates"]=["random-projector evidence initialization",2500]
    dump(root/f"selector_{cli.arm.lower().replace('-','_')}.json",result)
    dump(root/f"checkpoint_metrics_{cli.arm.lower().replace('-','_')}.json",{"arm":cli.arm,"baseline":baseline,"candidates":candidates})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
