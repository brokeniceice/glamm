#!/usr/bin/env python3
"""Finalize matched Phase 4A validation statistics, route gate and reports."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from tools.phase3c1 import inverse_logits, paired_statistics, summarize
from tools.phase4a import Phase4ADataset, load_rows


def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def replace_section(text,header,body):
 start=text.index(header); content_start=start+len(header)
 next_header=text.find("\n# ",content_start)
 if next_header<0: next_header=len(text)
 return text[:content_start]+"\n\n"+body.strip()+"\n"+text[next_header:]


def qualitative(config,out,selected,rows,clip_rows):
 dataset=Phase4ADataset(config,"val",return_original=True); by_id={str(row["sample_id"]):i for i,row in enumerate(dataset.rows)}
 selected_logits=torch.load(selected["low_res_logits"],map_location="cpu"); clip_logits=torch.load(ROOT/config["baseline"]["low_res_logits"],map_location="cpu")
 fepn_map={sid:logit for sid,logit in zip(selected_logits["sample_ids"],selected_logits["low_res_logits"])}
 clip_map={sid:logit for sid,logit in zip(clip_logits["sample_ids"],clip_logits["low_res_logits"])}
 ordered=[r["sample_id"] for r in rows]; rng=random.Random(3407); fixed=rng.sample(ordered,8)
 ranked=sorted(rows,key=lambda r:(r["foreground_iou"],r["sample_id"])); chosen=[]
 for role,ids in (("fixed_random",fixed),("worst",[r["sample_id"] for r in ranked[:4]]),("best",[r["sample_id"] for r in ranked[-4:]])):
  for sid in ids:
   if sid not in [x["sample_id"] for x in chosen]: chosen.append({"sample_id":sid,"role":role})
 qdir=out/"qualitative"; qdir.mkdir(parents=True,exist_ok=True)
 metrics={r["sample_id"]:r for r in rows}; clip_metrics={r["sample_id"]:r for r in clip_rows}
 for ordinal,item in enumerate(chosen):
  sid=item["sample_id"]; sample=dataset[by_id[sid]]; image=np.asarray(Image.open(sample["image_path"]).convert("RGB")); gt=sample["original_mask"].numpy()
  fepn=torch.sigmoid(inverse_logits(fepn_map[sid].float(),sample["geometry"])).numpy(); clip=torch.sigmoid(inverse_logits(clip_map[sid].float(),sample["geometry"])).numpy()
  fig,axes=plt.subplots(1,4,figsize=(16,4)); axes[0].imshow(image); axes[0].set_title("Input")
  axes[1].imshow(gt,cmap="gray",vmin=0,vmax=1); axes[1].set_title("Union evidence target")
  axes[2].imshow(fepn,cmap="magma",vmin=0,vmax=1); axes[2].set_title(f"FEPN p, IoU={metrics[sid]['foreground_iou']:.3f}")
  axes[3].imshow(clip,cmap="magma",vmin=0,vmax=1); axes[3].set_title(f"CLIP p, IoU={clip_metrics[sid]['foreground_iou']:.3f}")
  for ax in axes: ax.axis("off")
  fig.suptitle(f"{item['role']} | {sid}"); fig.tight_layout(); fig.savefig(qdir/f"{ordinal:02d}_{item['role']}.png",dpi=120); plt.close(fig)
 dump(qdir/"selection_manifest.json",{"seed":3407,"rules":{"fixed_random":"8 without replacement from ordered validation Fake IDs","worst":"lowest 4 selected FEPN FG IoU","best":"highest 4 selected FEPN FG IoU"},"samples":chosen})


def main():
 config=yaml.safe_load((ROOT/"configs/phase4a_fepn_evidence_learnability.yaml").read_text()); out=ROOT/config["experiment"]["output_root"]
 selector=json.loads((out/"selector_result.json").read_text()); metrics=json.loads((out/"checkpoint_metrics.json").read_text())
 selected=next(row for row in metrics["checkpoints"] if int(row["epoch"])==int(selector["selected_epoch"]))
 fepn_rows=load_rows(selected["predictions"]); clip_rows=load_rows(ROOT/config["baseline"]["predictions"])
 if [r["sample_id"] for r in fepn_rows]!=[r["sample_id"] for r in clip_rows]: raise RuntimeError("FEPN/CLIP paired population mismatch")
 comparisons={}
 for key in ("foreground_iou","foreground_f1"):
  comparisons[key]=paired_statistics([r[key] for r in fepn_rows],[r[key] for r in clip_rows],seed=int(config["evaluation"]["bootstrap_seed"]),repeats=int(config["evaluation"]["bootstrap_repeats"]))
 iou=comparisons["foreground_iou"]; dense_supported=iou["mean_difference"]>0 and iou["bootstrap_95_ci"][0]>0
 global_supported=bool(selected["global_safety_pass"])
 if dense_supported and global_supported: gate="GATE_TASK_ALIGNED_FORENSIC_EVIDENCE_LEARNABLE"
 elif dense_supported: gate="GATE_DENSE_ONLY_EVIDENCE_LEARNED"
 elif global_supported: gate="GATE_GLOBAL_ONLY_EVIDENCE_LEARNED"
 else: gate="GATE_TASK_ALIGNED_EVIDENCE_NOT_YET_SUPPORTED"
 clip_summary=summarize(clip_rows); clip_totals={k:sum(r[k] for r in clip_rows) for k in ("tp","fp","fn","tn")}
 clip_summary["global_foreground_iou"]=clip_totals["tp"]/max(1,clip_totals["tp"]+clip_totals["fp"]+clip_totals["fn"])
 clip_summary["global_foreground_f1"]=2*clip_totals["tp"]/max(1,2*clip_totals["tp"]+clip_totals["fp"]+clip_totals["fn"])
 bootstrap={"status":"COMPLETE","paired":True,"n":len(fepn_rows),"left":"selected_FEPN-v0","right":"matched_frozen_CLIP_spatial_probe","repeats":config["evaluation"]["bootstrap_repeats"],"seed":config["evaluation"]["bootstrap_seed"],"metrics":comparisons}
 route={"status":"FINAL","gate":gate,"conditions":{"dense_mean_delta_positive":iou["mean_difference"]>0,"dense_iou_ci_lower_gt_zero":iou["bootstrap_95_ci"][0]>0,"global_accuracy_ge_0_60":selected["classification"]["accuracy"]>=0.60,"global_roc_auc_ge_0_60":selected["classification"]["roc_auc"]>=0.60},
        "phase4b_authorized":False,"decision":"REPORT_AND_STOP","internal_test_opened":False,"official1000_opened":False}
 mask_areas=np.asarray([(r["tp"]+r["fn"])/(r["tp"]+r["fp"]+r["fn"]+r["tn"]) for r in fepn_rows]); values=np.asarray([r["foreground_iou"] for r in fepn_rows]); quartiles=np.quantile(mask_areas,[.25,.5,.75])
 bins=np.digitize(mask_areas,quartiles); failure={"status":"COMPLETE","mask_area_quartiles":quartiles.tolist(),"mean_iou_by_mask_area_quartile":[float(values[bins==i].mean()) for i in range(4)],"worst_20":[{"sample_id":r["sample_id"],"foreground_iou":r["foreground_iou"]} for r in sorted(fepn_rows,key=lambda x:x["foreground_iou"])[:20]],"threshold_tuned":False}
 dump(out/"fepn_validation_metrics.json",{"status":"COMPLETE","selected_epoch":selected["epoch"],"dense":selected["dense"],"classification":selected["classification"]})
 dump(out/"clip_baseline_metrics.json",{"status":"REUSED_EXACT_MATCH","dense":clip_summary})
 dump(out/"paired_bootstrap_fepn_vs_clip.json",bootstrap); dump(out/"statistics/paired_bootstrap_fepn_vs_clip.json",bootstrap)
 dump(out/"feature_statistics.json",selected["feature_statistics"]); dump(out/"failure_analysis.json",failure); dump(out/"route_gate.json",route)
 qualitative(config,out,selected,fepn_rows,clip_rows)
 completion={"status":"COMPLETE","phase":"Phase 4A","selected_epoch":selected["epoch"],"terminal_gate":gate,"formal_epochs":list(range(int(config["training"]["max_epochs"])+1)),"optimizer_steps":int(config["training"]["total_steps"]),"image_exposures":int(config["training"]["image_exposures"]),"early_stopping":False,"matched_sft":False,"phase4b_started":False,"internal_test_opened":False,"official1000_opened":False}
 dump(out/"completion_manifest.json",completion)
 report=f"""# Phase 4A — Standalone Task-Aligned Forensic Evidence Learnability

## Outcome

Selected FEPN-v0 checkpoint: epoch **{selected['epoch']}**. Terminal gate: `{gate}`. Phase 4B was not started; internal test and official1000 remained sealed.

## Architecture and protocol

FEPN-v0 has 2,107,458 trainable parameters and receives matched normalized RGB plus a deterministic fixed Laplacian residual view at 336×336. Independent learnable stems, channel concatenation, a 1×1 projection and a lightweight convolutional FPN produce `F_dense` at 128×84×84. P1, LLM, SAM and the SEG predictor are absent.

Training used the frozen 17,672-image train population for {config['training']['max_epochs']} complete passes ({config['training']['image_exposures']:,} exposures; {config['training']['total_steps']:,} optimizer steps), with each full batch balanced between Real and Fake. Formal candidates were epoch 0–{config['training']['max_epochs']}; poor validation performance did not trigger early stopping. Real images received only global BCE; Fake images received global BCE plus `2×BCE+0.5×Dice` dense loss on the official-annotation-derived per-image all-reference union evidence target. Dense threshold remained zero.

## Results

| Model | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| Matched frozen CLIP probe | {clip_summary['mean_foreground_iou']:.6f} | {clip_summary['median_foreground_iou']:.6f} | {clip_summary['mean_foreground_f1']:.6f} | {clip_summary['global_foreground_iou']:.6f} | {clip_summary['global_foreground_f1']:.6f} |
| FEPN-v0 epoch {selected['epoch']} | {selected['dense']['mean_foreground_iou']:.6f} | {selected['dense']['median_foreground_iou']:.6f} | {selected['dense']['mean_foreground_f1']:.6f} | {selected['dense']['global_foreground_iou']:.6f} | {selected['dense']['global_foreground_f1']:.6f} |

FEPN−CLIP paired FG IoU delta is `{iou['mean_difference']:+.6f}`, 95% bootstrap CI `[{iou['bootstrap_95_ci'][0]:+.6f}, {iou['bootstrap_95_ci'][1]:+.6f}]`, win/tie/loss `{iou['wins']}/{iou['ties']}/{iou['losses']}`. FG F1 delta is `{comparisons['foreground_f1']['mean_difference']:+.6f}`, CI `[{comparisons['foreground_f1']['bootstrap_95_ci'][0]:+.6f}, {comparisons['foreground_f1']['bootstrap_95_ci'][1]:+.6f}]`.

Global validation classification: Accuracy `{selected['classification']['accuracy']:.6f}`, Precision `{selected['classification']['precision']:.6f}`, Recall `{selected['classification']['recall']:.6f}`, F1 `{selected['classification']['f1']:.6f}`, ROC-AUC `{selected['classification']['roc_auc']:.6f}`. The preregistered safety condition was `{'PASS' if global_supported else 'FAIL'}`.

`F_dense` non-collapse diagnostic: channel std mean `{selected['feature_statistics']['channel_std_mean']:.6f}`, spatial variance mean `{selected['feature_statistics']['spatial_variance_mean']:.6f}`, status `{'PASS' if selected['feature_statistics']['noncollapsed'] else 'FAIL'}`.

## Final questions and interpretation

1. Non-collapsed feature learning: `{'supported' if selected['feature_statistics']['noncollapsed'] else 'not supported'}` by finite channel and spatial variance.
2. Dense localization is reported above under the same original-space target and zero-threshold evaluator as CLIP.
3. Significant superiority to CLIP: `{'yes' if dense_supported else 'no'}` under the preregistered paired-bootstrap rule.
4. Global authenticity information: `{'meaningfully above chance' if global_supported else 'did not pass the frozen safety definition'}`.
5. RGB+Residual FEPN continuation: governed by `{gate}`; this phase alone does not prove that residual input is causally necessary because RGB-only was not run.
6. Evidence supported: `{'global and dense' if dense_supported and global_supported else 'dense only' if dense_supported else 'global only' if global_supported else 'neither branch sufficiently supported'}`.
7. Phase 4A gate: `{gate}`.
8. The FEPN living route is updated without rewriting its initial hypothesis.
9. Phase 4B remains not authorized and requires an explicit new instruction even if Phase 4A passed.

The result concerns visible/explainable forensic evidence localization, not all forged pixels. It does not establish end-to-end autonomous G0 improvement, because no MLLM integration occurred.
"""
 (out/"final_report.md").write_text(report,encoding="utf-8"); (ROOT/"docs/phase4a_fepn_evidence_learnability.md").write_text(report,encoding="utf-8")
 route_path=ROOT/"docs/fepn_design_route.md"; text=route_path.read_text(encoding="utf-8")
 status=f"Phase 4A is complete under `{gate}`. P1 remains frozen and Phase 4B has not been started."
 hypothesis=("Standalone task-aligned RGB+Residual FEPN evidence is supported for both dense localization and global authenticity prediction; the next unresolved question is whether and where the frozen P1 system can consume it." if dense_supported and global_supported else
             "Phase 4A supports dense but not preregistered global evidence; the global branch requirement and representation design need an explicit route decision before any integration." if dense_supported else
             "Phase 4A supports global authenticity evidence but does not support dense evidence superiority over matched CLIP; full FEPN integration is not justified." if global_supported else
             "FEPN-v0 did not meet the standalone evidence gate; a bounded representation-design decision is required before any integration.")
 text=replace_section(text,"# Current Status",status); text=replace_section(text,"# Current Working Hypothesis",hypothesis)
 text=replace_section(text,"# Current Authorized Phase","None. Phase 4A is complete; Phase 4B requires separate explicit authorization.")
 text=replace_section(text,"# Next Decision Gate",("Await explicit Phase 4B interface-study preregistration and authorization." if dense_supported and global_supported else "Bounded FEPN representation-design decision; P1 integration remains prohibited."))
 text += f"\n\n## Version 0.2 — 2026-08-24 / Phase 4A outcome\n\n- Previous assumption: FEPN-v0 RGB+Residual evidence may exceed matched frozen CLIP while retaining global authenticity information.\n- New evidence: selected epoch {selected['epoch']}; FEPN−CLIP IoU delta {iou['mean_difference']:+.6f}, CI [{iou['bootstrap_95_ci'][0]:+.6f}, {iou['bootstrap_95_ci'][1]:+.6f}]; global Accuracy {selected['classification']['accuracy']:.6f}, ROC-AUC {selected['classification']['roc_auc']:.6f}.\n- Route change: terminal gate `{gate}`; Phase 4B remains separately authorized only.\n- Reason: application of the frozen Phase 4A dense and global gates.\n- Status: confirmatory with respect to the frozen Phase 4A protocol; any later interface design remains a new phase.\n"
 route_path.write_text(text,encoding="utf-8")
 paper=ROOT/"docs/paper_convergence_route.md"; paper_text=paper.read_text(encoding="utf-8")
 paper_text += f"\n\n### Phase 4A — COMPLETE\n\nStandalone FEPN-v0 selected epoch {selected['epoch']}. Matched validation Fake FEPN−CLIP mean FG IoU delta was `{iou['mean_difference']:+.6f}`, paired-bootstrap 95% CI `[{iou['bootstrap_95_ci'][0]:+.6f}, {iou['bootstrap_95_ci'][1]:+.6f}]`; global Accuracy/ROC-AUC were `{selected['classification']['accuracy']:.6f}/{selected['classification']['roc_auc']:.6f}`. Terminal gate: `{gate}`. P1/LLM/SAM integration, Phase 4B, internal test and official1000 were not run. See [Phase 4A report](phase4a_fepn_evidence_learnability.md).\n"
 paper.write_text(paper_text,encoding="utf-8")
 print(json.dumps({"status":"COMPLETE","gate":gate,"selected_epoch":selected["epoch"],"iou":iou,"classification":selected["classification"]},indent=2))


if __name__=="__main__": main()
