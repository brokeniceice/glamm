#!/usr/bin/env python3
"""Paired statistics, route gate, qualitative table, and final Phase 4B-G report."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.phase4b import dump, file_sha256, load_jsonl


def load(path): return json.loads(Path(path).read_text())


def paired_bootstrap(left_path,right_path,repeats,seed,label):
    left={row["sample_id"]:row for row in load_jsonl(left_path)}; right={row["sample_id"]:row for row in load_jsonl(right_path)}
    if set(left)!=set(right) or not left: raise RuntimeError(f"paired population mismatch: {label}")
    ids=sorted(left); rng=np.random.default_rng(seed); result={"comparison":label,"n":len(ids),"repeats":repeats,"seed":seed,"metrics":{}}
    for key in ("foreground_iou","foreground_f1"):
        delta=np.asarray([left[sid][key]-right[sid][key] for sid in ids],dtype=np.float64)
        draws=np.empty(repeats,dtype=np.float64)
        for start in range(0,repeats,256):
            count=min(256,repeats-start); indices=rng.integers(0,len(delta),size=(count,len(delta)))
            draws[start:start+count]=delta[indices].mean(axis=1)
        result["metrics"][key]={"mean_left":float(np.mean([left[s][key] for s in ids])),
          "mean_right":float(np.mean([right[s][key] for s in ids])),"delta":float(delta.mean()),
          "ci95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
          "p_bootstrap_two_sided":float(2*min(np.mean(draws<=0),np.mean(draws>=0)))}
    return result


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]
    only=load(out/"selector_proj_only.json"); lora=load(out/"selector_proj_lora.json")
    if only["fallback_to_P1"] or lora["fallback_to_P1"]: raise RuntimeError("an evidence arm has no selectable checkpoint")
    p1_root=ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/selector/P1_FROZEN"
    only_root=out/"evaluation/selector/PROJ-ONLY"/f"step_{int(only['optimizer_step']):04d}"
    lora_root=out/"evaluation/selector/PROJ-LORA"/f"step_{int(lora['optimizer_step']):04d}"
    repeats=int(cfg["evaluation"]["bootstrap_repeats"]); seed=int(cfg["evaluation"]["bootstrap_seed"])
    vs_p1=paired_bootstrap(lora_root/"G0/predictions.jsonl",p1_root/"G0/predictions.jsonl",repeats,seed,"PROJ-LORA_vs_P1")
    vs_only=paired_bootstrap(lora_root/"G0/predictions.jsonl",only_root/"G0/predictions.jsonl",repeats,seed+1,"PROJ-LORA_vs_PROJ-ONLY")
    dump(out/"paired_bootstrap_proj_lora_vs_p1.json",vs_p1); dump(out/"paired_bootstrap_proj_lora_vs_proj_only.json",vs_only)
    baseline=only["baseline"]; om=only["selected_metrics"]; lm=lora["selected_metrics"]
    detection={"P1":{k:baseline[k] for k in ("classification_accuracy","classification_f1","classification_confusion")},
      "PROJ-ONLY":{k:om[k] for k in ("classification_accuracy","classification_f1","classification_precision","classification_recall","classification_confusion")},
      "PROJ-LORA":{k:lm[k] for k in ("classification_accuracy","classification_f1","classification_precision","classification_recall","classification_confusion")}}
    g0={name:{k:value[k] for k in ("mean_foreground_iou","median_foreground_iou","mean_foreground_f1","median_foreground_f1")}
        for name,value in (("P1",baseline),("PROJ-ONLY",om),("PROJ-LORA",lm))}
    phrase={name:{k:value[k] for k in ("R_phrase_sem","R_key_soft","R_sentence_sem","structure_validity","valid_seg_rate","malformed_output_rate")}
        for name,value in (("P1",baseline),("PROJ-ONLY",om),("PROJ-LORA",lm))}
    dump(out/"detection_metrics.json",detection); dump(out/"g0_metrics.json",g0); dump(out/"phrase_metrics.json",phrase)
    delta_iou=vs_p1["metrics"]["foreground_iou"]; significant=delta_iou["delta"]>0 and delta_iou["ci95"][0]>0
    nonreg=all(lm["non_regression"].values()); phrase_ok=lm["R_phrase_sem"]>=baseline["R_phrase_sem"]-.005
    if significant and nonreg and phrase_ok: gate="GATE_GLOBAL_FORENSIC_EVIDENCE_IMPROVES_AUTONOMOUS_GROUNDING"
    elif lm["classification_accuracy"]>baseline["classification_accuracy"] and lm["R_phrase_sem"]>baseline["R_phrase_sem"] and not significant:
        gate="GATE_GLOBAL_FEPN_IMPROVES_REASONING_NOT_GROUNDING"
    elif lm["classification_accuracy"]>baseline["classification_accuracy"] and not significant:
        gate="GATE_GLOBAL_FEPN_CLASSIFICATION_ONLY"
    else: gate="GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1"
    route={"status":"INTERIM_AWAITING_ORACLE_DIAGNOSTICS","gate":gate,"success":significant and nonreg and phrase_ok,
      "matched_sft_triggered":significant,"conditions":{"g0_delta_positive_significant":significant,
      "g0_iou_delta":delta_iou["delta"],"g0_iou_ci95":delta_iou["ci95"],"nonregression":nonreg,
      "phrase_not_materially_worse":phrase_ok},"selected":{"PROJ-ONLY":only["optimizer_step"],"PROJ-LORA":lora["optimizer_step"]},
      "internal_test_used":False,"official1000_used":False,"threshold_tuned":False}
    dump(out/"route_gate.json",route)
    # Frozen random subset plus paired extremes; no positive-only cherry picking.
    p1={r["sample_id"]:r for r in load_jsonl(p1_root/"G0/predictions.jsonl")}; lr={r["sample_id"]:r for r in load_jsonl(lora_root/"G0/predictions.jsonl")}
    ids=sorted(p1); rng=random.Random(3407); fixed=rng.sample(ids,10); ranked=sorted(ids,key=lambda sid:lr[sid]["foreground_iou"]-p1[sid]["foreground_iou"])
    selected=[]
    for category,values in (("fixed_random",fixed),("largest_worsening",ranked[:5]),("largest_improvement",ranked[-5:])):
        for sid in values:
            selected.append({"category":category,"sample_id":sid,"image_path":p1[sid].get("image_path"),
             "p1_text":p1[sid].get("decoded_text"),"fepn_text":lr[sid].get("decoded_text"),
             "p1_phrase":p1[sid].get("generated_localization_phrase"),"fepn_phrase":lr[sid].get("generated_localization_phrase"),
             "p1_iou":p1[sid]["foreground_iou"],"fepn_iou":lr[sid]["foreground_iou"],
             "delta_iou":lr[sid]["foreground_iou"]-p1[sid]["foreground_iou"]})
    lines=["# Phase 4B-G qualitative analysis","","Fixed seed random cases plus symmetric paired extremes; selection is not positive-only.","",
      "|category|sample|P1 IoU|FEPN IoU|delta|P1 phrase|FEPN phrase|","|---|---|---:|---:|---:|---|---|"]
    for r in selected: lines.append(f"|{r['category']}|{r['sample_id']}|{r['p1_iou']:.4f}|{r['fepn_iou']:.4f}|{r['delta_iou']:+.4f}|{str(r['p1_phrase']).replace('|','/')}|{str(r['fepn_phrase']).replace('|','/')}|")
    (out/"qualitative_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); dump(out/"qualitative_cases.json",selected)
    print(json.dumps(route,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
