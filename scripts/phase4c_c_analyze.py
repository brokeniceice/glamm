#!/usr/bin/env python3
"""Statistics, figures, qualitative panels, gates, and report for Phase 4C-C."""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image,ImageDraw
from scipy.stats import pearsonr,spearmanr,wilcoxon

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4c_b import diagnostic,dump,inverse_sam_logits,load_spatial_shard,paired,rows,spatial_cache_paths,summarize


ARMS=("clip_reader","forensic_reader"); QUERIES=("G0","phrase_repair","phrase_only","tf_full_context"); CONDITIONS=("matched","cross_image","spatial_shuffle","global_repeat","zero")


def p1_records(cfg,bout,query,common):
    if query!="phrase_repair":
        mode=query; path=bout/"evaluation"/mode/"P1/predictions.jsonl"; values=rows(path)
    else:
        source=rows(ROOT/cfg["phase3c0"]["paired_conditions"]); values=[]
        for r in source:
            if r.get("B") is not None:
                b=r["B"]; values.append({"sample_id":r["sample_id"],**{k:b[k] for k in ("foreground_iou","foreground_f1","tp","fp","fn")}})
    index={r["sample_id"]:r for r in values}; return [index[sid] for sid in common]


def wilcox(delta):
    a=np.asarray(delta,np.float64)
    if not np.any(a):return {"statistic":0.0,"p_raw":1.0,"zero_method":"wilcox"}
    x=wilcoxon(a,zero_method="wilcox",alternative="two-sided",method="auto")
    return {"statistic":float(x.statistic),"p_raw":float(x.pvalue),"zero_method":"wilcox"}


def holm(items):
    order=sorted(range(len(items)),key=lambda i:items[i]["p_raw"]); running=0.
    for rank,index in enumerate(order):
        adjusted=min(1.,(len(items)-rank)*items[index]["p_raw"]); running=max(running,adjusted); items[index]["p_holm"]=running


def corr_boot(x,y,seed,repeats=10000):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64); rng=np.random.default_rng(seed); pr=[];sr=[]
    for start in range(0,repeats,200):
        for idx in rng.integers(0,len(x),size=(min(200,repeats-start),len(x))):
            if np.std(x[idx])==0 or np.std(y[idx])==0:continue
            pr.append(pearsonr(x[idx],y[idx]).statistic);sr.append(spearmanr(x[idx],y[idx]).statistic)
    return {"n":len(x),"pearson_r":float(pearsonr(x,y).statistic),"pearson_p":float(pearsonr(x,y).pvalue),"pearson_bootstrap_95_ci":[float(v) for v in np.quantile(pr,[.025,.975])],"spearman_rho":float(spearmanr(x,y).statistic),"spearman_p":float(spearmanr(x,y).pvalue),"spearman_bootstrap_95_ci":[float(v) for v in np.quantile(sr,[.025,.975])],"bootstrap_repeats":repeats}


def positive(s):return s["mean_difference"]>0 and s["bootstrap_95_ci"][0]>0 and s.get("p_holm",1)<.05


def overlay(image,mask,color,alpha=.55):
    a=np.asarray(image.convert("RGB")).astype(np.float32);m=np.asarray(mask.resize(image.size,Image.Resampling.NEAREST)).astype(bool);c=np.asarray(color,np.float32);a[m]=(1-alpha)*a[m]+alpha*c;return Image.fromarray(np.uint8(np.clip(a,0,255)))


def qualitative(cfg,out,bout,common,per_rows):
    bcfg=yaml.safe_load((ROOT/cfg["phase4c_b"]["config"]).read_text()); source=ROOT/bcfg["frozen_spatial_cache"]["phase3c1_root"]
    meta=[];gt=[]
    for path in spatial_cache_paths(source,"sam","val"):
        shard=load_spatial_shard(path,"sam");meta+=shard["records"];gt+=shard["original_masks"]
    full_ids=[r["sample_id"] for r in meta]; full_index={sid:i for i,sid in enumerate(full_ids)}; common_index={sid:i for i,sid in enumerate(common)}
    qdir=out/"qualitative";qdir.mkdir(exist_ok=True);manifest={"selection":"deterministic metric ordering; no cherry-pick","top_k":4,"groups":{}}
    for arm in ARMS:
        table=[r for r in per_rows if r["reader_type"]==arm and r["query_condition"]=="G0"]
        tf={r["sample_id"]:r for r in per_rows if r["reader_type"]==arm and r["query_condition"]=="tf_full_context"}
        groups={"matched_over_cross":sorted(table,key=lambda r:(-r["s_cross"],r["sample_id"]))[:4],"matched_approx_cross":sorted(table,key=lambda r:(abs(r["s_cross"]),r["sample_id"]))[:4],"reader_improves_g0":sorted(table,key=lambda r:(-r["delta_reader"],r["sample_id"]))[:4],"reader_harms_tf":sorted(table,key=lambda r:(tf[r["sample_id"]]["delta_reader"],r["sample_id"]))[:4]}
        p1=torch.load(bout/"evaluation/G0/P1/epoch0_spatial.pt",map_location="cpu"); matched=torch.load(bout/f"evaluation/G0/{arm}/selected_spatial.pt",map_location="cpu"); tfstate=torch.load(bout/f"evaluation/tf_full_context/{arm}/selected_spatial.pt",map_location="cpu")
        cross=torch.load(Path(cfg["experiment"]["cache_root"])/f"spatial/{arm}/G0/cross_image.pt",map_location="cpu");shuffle=torch.load(Path(cfg["experiment"]["cache_root"])/f"spatial/{arm}/G0/spatial_shuffle.pt",map_location="cpu")
        for group,items in groups.items():
            manifest["groups"][f"{arm}:{group}"]=[r["sample_id"] for r in items]
            for rank,r in enumerate(items):
                sid=r["sample_id"];fi=full_index[sid];ci=common_index[sid];im=Image.open(meta[fi]["image_path"]).convert("RGB").resize((220,220)); panels=[im,overlay(im,Image.fromarray(gt[fi].numpy()),(0,255,0))]
                states=((p1,fi),(matched,fi),(cross,ci),(shuffle,ci),(tfstate,fi));
                for state,index in states:
                    mask=inverse_sam_logits(torch.as_tensor(state["low_res_logits"][index]).reshape(1,1,256,256),meta[fi]["geometry"]).gt(0).cpu();panels.append(overlay(im,Image.fromarray(mask.numpy()),(255,0,0)))
                labels=("image","GT","P1 G0","matched","cross","shuffle","TF Reader");canvas=Image.new("RGB",(220*7,246),"white");draw=ImageDraw.Draw(canvas)
                for j,(panel,label) in enumerate(zip(panels,labels)):canvas.paste(panel,(j*220,26));draw.text((j*220+4,5),label,fill="black")
                canvas.save(qdir/f"{arm}_{group}_{rank}.png")
    dump(qdir/"selection_manifest.json",manifest);return manifest


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4c_c_evidence_attribution.yaml").read_text());out=ROOT/cfg["experiment"]["output_root"];bout=ROOT/cfg["phase4c_b"]["output_root"]
    common=json.loads((out/"common_population.json").read_text())["sample_ids"]; predictions={};p1={q:p1_records(cfg,bout,q,common) for q in QUERIES}
    result_rows=[]
    for arm in ARMS:
        for query in QUERIES:
            predictions[arm,query]={c:rows(out/f"evaluation/{arm}/{query}/{c}.jsonl") for c in CONDITIONS}
            for condition,records_ in predictions[arm,query].items():
                s=summarize(records_);result_rows.append({"reader_type":arm,"query_condition":query,"evidence_condition":condition,**s})
    pd.DataFrame(result_rows).to_csv(out/"phase4c_c_results.csv",index=False)
    primary=[];stats={"population_n":len(common),"primary":{},"clip_vs_forensic_sensitivity":{},"correlations":{},"quartiles":{},"optional_controls_not_run":cfg["optional_controls_not_run"]}
    per=[]
    p1idx={q:{r["sample_id"]:r for r in p1[q]} for q in QUERIES}; gap_tf={sid:p1idx["tf_full_context"][sid]["foreground_iou"]-p1idx["G0"][sid]["foreground_iou"] for sid in common};gap_phrase={sid:p1idx["phrase_repair"][sid]["foreground_iou"]-p1idx["G0"][sid]["foreground_iou"] for sid in common}
    for arm in ARMS:
        stats["primary"][arm]={}
        for query in QUERIES:
            vals=predictions[arm,query]; index={c:{r["sample_id"]:r for r in rr} for c,rr in vals.items()};stats["primary"][arm][query]={}
            for intervention in CONDITIONS[1:]:
                comp=paired(vals["matched"],vals[intervention])["foreground_iou"]; comp.update(wilcox([index["matched"][sid]["foreground_iou"]-index[intervention][sid]["foreground_iou"] for sid in common]));comp.update({"arm":arm,"query":query,"comparison":f"matched_minus_{intervention}"});primary.append(comp);stats["primary"][arm][query][intervention]=comp
            for sid in common:
                m=index["matched"][sid]["foreground_iou"];row={"sample_id":sid,"reader_type":arm,"query_condition":query,"iou_p1":p1idx[query][sid]["foreground_iou"],"iou_matched":m,"iou_cross":index["cross_image"][sid]["foreground_iou"],"iou_shuffle":index["spatial_shuffle"][sid]["foreground_iou"],"iou_global":index["global_repeat"][sid]["foreground_iou"],"iou_zero":index["zero"][sid]["foreground_iou"],"delta_reader":m-p1idx[query][sid]["foreground_iou"],"s_cross":m-index["cross_image"][sid]["foreground_iou"],"s_shuffle":m-index["spatial_shuffle"][sid]["foreground_iou"],"s_global":m-index["global_repeat"][sid]["foreground_iou"],"s_zero":m-index["zero"][sid]["foreground_iou"],"gap_tf":gap_tf[sid],"gap_phrase_repair":gap_phrase[sid]};per.append(row)
    holm(primary)
    for x in primary:stats["primary"][x["arm"]][x["query"]][x["comparison"].replace("matched_minus_","")].update({"p_holm":x["p_holm"]})
    df=pd.DataFrame(per);df.to_csv(out/"phase4c_c_per_sample.csv",index=False)
    for arm in ARMS:
        g=df[(df.reader_type==arm)&(df.query_condition=="G0")];stats["correlations"][arm]={"gap_tf":corr_boot(g.gap_tf,g.delta_reader,3407),"gap_phrase_repair":corr_boot(g.gap_phrase_repair,g.delta_reader,3408)}
        bins=pd.qcut(g.gap_tf,4,labels=("Q1","Q2","Q3","Q4"));qrows=[]
        for label in ("Q1","Q2","Q3","Q4"):
            x=g[bins==label];qrows.append({"quartile":label,"n":len(x),"p1_g0_iou":float(x.iou_p1.mean()),"reader_g0_iou":float(x.iou_matched.mean()),"reader_gain":float(x.delta_reader.mean()),"matched_minus_cross":float(x.s_cross.mean())})
        stats["quartiles"][arm]=qrows
    for query in QUERIES:
        stats["clip_vs_forensic_sensitivity"][query]={}
        for key in ("s_cross","s_shuffle","s_global","s_zero"):
            c=df[(df.reader_type=="clip_reader")&(df.query_condition==query)].set_index("sample_id")[key];f=df[(df.reader_type=="forensic_reader")&(df.query_condition==query)].set_index("sample_id")[key]; delta=(f-c).reindex(common)
            base=[{"sample_id":sid,"foreground_iou":float(f[sid]),"foreground_f1":float(f[sid]),"tp":0,"fp":0,"fn":0} for sid in common]; ref=[{"sample_id":sid,"foreground_iou":float(c[sid]),"foreground_f1":float(c[sid]),"tp":0,"fp":0,"fn":0} for sid in common];s=paired(base,ref)["foreground_iou"];s.update(wilcox(delta));stats["clip_vs_forensic_sensitivity"][query][key]=s
    dump(out/"phase4c_c_statistics.json",stats)
    figdir=out/"figures/phase4c_c";figdir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for j,arm in enumerate(ARMS):
        vals=[float(df[(df.reader_type==arm)&(df.query_condition=="G0")]["iou_p1"].mean())]+[float(df[(df.reader_type==arm)&(df.query_condition=="G0")][{"matched":"iou_matched","cross_image":"iou_cross","spatial_shuffle":"iou_shuffle","global_repeat":"iou_global","zero":"iou_zero"}[c]].mean()) for c in CONDITIONS]
        ax[j].bar(("P1","Matched","Cross","Shuffle","Global","Zero"),vals);ax[j].set_title(arm);ax[j].tick_params(axis="x",rotation=35);ax[j].set_ylabel("Mean FG IoU")
    fig.tight_layout();fig.savefig(figdir/"figure1_g0_interventions.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for j,arm in enumerate(ARMS):
        ss=[stats["primary"][arm]["G0"][c] for c in CONDITIONS[1:]];means=[s["mean_difference"] for s in ss];err=[[m-s["bootstrap_95_ci"][0] for m,s in zip(means,ss)],[s["bootstrap_95_ci"][1]-m for m,s in zip(means,ss)]];ax[j].bar(("Cross","Shuffle","Global","Zero"),means,yerr=err,capsize=4);ax[j].axhline(0,color="black",lw=.8);ax[j].set_title(arm);ax[j].set_ylabel("Matched minus intervention IoU")
    fig.tight_layout();fig.savefig(figdir/"figure2_intervention_degradation.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for arm in ARMS:ax.plot([r["quartile"] for r in stats["quartiles"][arm]],[r["reader_gain"] for r in stats["quartiles"][arm]],marker="o",label=arm)
    ax.axhline(0,color="black",lw=.8);ax.set_xlabel("P1 TF minus G0 gap quartile");ax.set_ylabel("Reader G0 delta IoU");ax.legend();fig.tight_layout();fig.savefig(figdir/"figure3_gap_quartile.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
    for j,arm in enumerate(ARMS):
        g=df[(df.reader_type==arm)&(df.query_condition=="G0")];rho=stats["correlations"][arm]["gap_tf"]["spearman_rho"];ax[j].scatter(g.gap_tf,g.delta_reader,s=8,alpha=.35);ax[j].axhline(0,color="black",lw=.8);ax[j].set_title(f"{arm}; Spearman rho={rho:.3f}");ax[j].set_xlabel("P1 TF minus P1 G0 IoU");ax[j].set_ylabel("Reader G0 minus P1 G0 IoU")
    fig.tight_layout();fig.savefig(figdir/"figure4_gap_scatter.png",dpi=180);plt.close(fig)
    qual=qualitative(cfg,out,bout,common,per)
    cross=[positive(stats["primary"][a]["G0"]["cross_image"]) for a in ARMS];shuffle=[positive(stats["primary"][a]["G0"]["spatial_shuffle"]) for a in ARMS];global_=[positive(stats["primary"][a]["G0"]["global_repeat"]) for a in ARMS];zero=[positive(stats["primary"][a]["G0"]["zero"]) for a in ARMS]
    image="TRUE" if all(cross) else ("PARTIAL" if any(cross) else "FALSE"); spatial="TRUE" if all(shuffle) else ("PARTIAL" if any(shuffle) or any(global_) else "FALSE")
    corrpos=[stats["correlations"][a]["gap_tf"]["spearman_bootstrap_95_ci"][0]>0 for a in ARMS];compensator="TRUE" if not any(cross) and not any(zero) and all(corrpos) else ("PARTIAL" if any(corrpos) or not all(zero) else "FALSE")
    sens=stats["clip_vs_forensic_sensitivity"]["G0"]; forensic_adv=any(v["mean_difference"]>0 and v["bootstrap_95_ci"][0]>0 for v in sens.values());forensic_specific="TRUE" if forensic_adv else "INCONCLUSIVE"
    proceed="YES" if image in ("TRUE","PARTIAL") and (spatial in ("TRUE","PARTIAL") or forensic_adv) else "NO"
    decision={"IMAGE_SPECIFIC_EVIDENCE_UTILIZATION":image,"SPATIAL_EVIDENCE_UTILIZATION":spatial,"READER_AS_QUERY_COMPENSATOR":compensator,"FORENSIC_SPECIFIC_DOWNSTREAM_UTILIZATION":forensic_specific,"PROCEED_TO_PHASE_4C_D":proceed,"hard_stop_active":True,"next_phase_started":False};dump(out/"decision.json",decision)
    def line(arm,key):
        s=stats["primary"][arm]["G0"][key];return f"{s['mean_difference']:+.6f} [{s['bootstrap_95_ci'][0]:+.6f}, {s['bootstrap_95_ci'][1]:+.6f}]"
    table="\n".join(f"| {r['reader_type']} | {r['query_condition']} | {r['evidence_condition']} | {r['mean_foreground_iou']:.6f} | {r['median_foreground_iou']:.6f} |" for r in result_rows)
    report=f"""# Phase 4C-C — Image-Specific Evidence Utilization Attribution

## 1. Executive Summary

- Correct-image evidence dependence: **{image}**.
- Spatial-organization dependence: **{spatial}**.
- Query-compensator mechanism: **{compensator}**.
- Mechanistic difference between CLIP and forensic Readers: **{forensic_specific}**.
- Proceed to Phase 4C-D: **{proceed}**. No next phase was started.

## 2. Protocol

Frozen Phase 4C-B selected Readers and Phase 4C-A sources were evaluated with the unchanged Phase 4C-B decoder, inverse resize, logit threshold 0.0, target, metrics, and paired bootstrap. Full baseline reproduction uses 1,106 validation Fake images. The four-query intervention matrix uses the fixed 1,076-image common legal-query population. Cross-image evidence uses the saved seed-3407 bijective derangement. Spatial shuffle uses one saved 576-token permutation. No model was trained and internal test/official1000 remained sealed.

## 3. Baseline Reproduction

Both Readers reproduced Phase 4C-B G0, Phrase-Only, and TF-Full within absolute mean-IoU tolerance 1e-4. Phrase-Repair uses the historical exact token-span replacement; current and historical G0 projected queries were exact for all 1,078 valid cases.

## 4. Main Intervention Results

| Reader | Query | Evidence | Mean FG IoU | Median FG IoU |
|---|---|---|---:|---:|
{table}

## 5. Image-Specific Evidence Test

- CLIP matched-minus-cross G0: {line('clip_reader','cross_image')}.
- Forensic matched-minus-cross G0: {line('forensic_reader','cross_image')}.

## 6. Spatial Evidence Test

- CLIP matched-minus-shuffle/global G0: {line('clip_reader','spatial_shuffle')} / {line('clip_reader','global_repeat')}.
- Forensic matched-minus-shuffle/global G0: {line('forensic_reader','spatial_shuffle')} / {line('forensic_reader','global_repeat')}.
- Because the Reader has no positional embedding, fixed token permutation is expected to be permutation-invariant; the intervention directly audits this architectural property.

## 7. Evidence-Absent Test

- CLIP matched-minus-zero G0: {line('clip_reader','zero')}.
- Forensic matched-minus-zero G0: {line('forensic_reader','zero')}.

## 8. Query-Quality Interaction

- CLIP gap-TF vs G0 gain Spearman rho: {stats['correlations']['clip_reader']['gap_tf']['spearman_rho']:+.4f}, CI {stats['correlations']['clip_reader']['gap_tf']['spearman_bootstrap_95_ci']}.
- Forensic gap-TF vs G0 gain Spearman rho: {stats['correlations']['forensic_reader']['gap_tf']['spearman_rho']:+.4f}, CI {stats['correlations']['forensic_reader']['gap_tf']['spearman_bootstrap_95_ci']}.
- Quartile values and Phrase-Repair gap correlations are recorded in `phase4c_c_statistics.json` and Figures 3–4.

## 9. CLIP vs Forensic Mechanism Comparison

The primary comparison is intervention sensitivity rather than raw matched IoU. Full paired sensitivity results are recorded under `clip_vs_forensic_sensitivity`; forensic-specific downstream use is reported conservatively as **{forensic_specific}**.

## 10. Failure Cases

Cases were selected automatically by fixed metric order for matched-over-cross, matched-approximately-cross, G0 improvement, and TF harm. The selection manifest and panels are under `qualitative/`.

## 11. Decision

```text
IMAGE_SPECIFIC_EVIDENCE_UTILIZATION: {image}
SPATIAL_EVIDENCE_UTILIZATION: {spatial}
READER_AS_QUERY_COMPENSATOR: {compensator}
FORENSIC_SPECIFIC_DOWNSTREAM_UTILIZATION: {forensic_specific}
PROCEED_TO_PHASE_4C_D: {proceed}
```

Evidence:

1. Matched-vs-cross paired effects are reported on the identical 1,076 images with a shared derangement.
2. Matched-vs-shuffle/global/zero isolates token order, dense variation, and evidence absence.
3. Reader gain is related to frozen P1 TF/Phrase-Repair gaps without changing query tokens.
4. CLIP/forensic sensitivity is compared per image, not inferred from raw means alone.
5. Phrase-Repair/TF effects bound whether the Reader merely compensates weak autonomous queries.
6. Optional same-class cross, per-image shuffle, and dataset-mean evidence controls were not run and are not implied.
"""
    (out/"phase4c_c_mechanism_attribution.md").write_text(report);(ROOT/"docs/phase4c_c_mechanism_attribution.md").write_text(report)
    required=["phase4c_c_preflight.md","phase4c_c_results.csv","phase4c_c_per_sample.csv","phase4c_c_statistics.json","phase4c_c_image_permutation.json","phase4c_c_mechanism_attribution.md","decision.json"]
    missing=[x for x in required if not (out/x).exists()];dump(out/"completion_manifest.json",{"status":"COMPLETE" if not missing else "INCOMPLETE","phase":"Phase 4C-C","required":required,"missing":missing,"decision":decision,"training_performed":False,"internal_test_access":False,"official1000_access":False,"next_phase_started":False})
    if missing:raise RuntimeError(missing)
    print(json.dumps({"status":"COMPLETE","decision":decision},indent=2))


if __name__=="__main__":main()
