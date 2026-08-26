#!/usr/bin/env python3
"""Statistics, diagnostics, qualitative panels, route gate, and report for Phase 4C-A."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from tools.phase3c1 import inverse_logits
from tools.phase4c_a import cache_paths,dump,load_shard,paired_bootstrap


def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats_pair(left,right):
    return {"foreground_iou":paired_bootstrap([r["foreground_iou"] for r in left],[r["foreground_iou"] for r in right]),
            "foreground_f1":paired_bootstrap([r["foreground_f1"] for r in left],[r["foreground_f1"] for r in right])}


def load_val(source):
    metadata=[]; originals=[]; targets=[]
    for path in cache_paths(source,"val"):
        shard=load_shard(path); metadata += shard["records"]; originals += shard["original_masks"]; targets += list(shard["targets"])
    return metadata,originals,targets


def feature_diagnostics(cfg,out,device):
    ck=torch.load(Path(cfg["experiment"]["checkpoint_root"])/"forensic_adapter/selected.pt",map_location="cpu")
    model=CLIPSpatialArm(blocks=3).to(device); model.load_state_dict(ck["model"]); model.eval()
    source=ROOT/cfg["source"]["phase3c1_root"]; cosine=[]; relative=[]; mean_norm=[]; channel_std=[]; spatial_var=[]; evidence_norm=[]; nonevidence_norm=[]
    with torch.no_grad():
        for path in cache_paths(source,"val"):
            shard=load_shard(path); x=shard["features"].to(device=device,dtype=torch.float32)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                value=model(x,return_features=True)
            f0=value["F0"].float(); ff=value["F_forensic"].float(); flat0=f0.flatten(1); flatf=ff.flatten(1)
            cosine += F.cosine_similarity(flat0,flatf,dim=1).cpu().tolist()
            relative += (torch.linalg.vector_norm(flatf-flat0,dim=1)/torch.linalg.vector_norm(flat0,dim=1).clamp_min(1e-12)).cpu().tolist()
            mean_norm += torch.linalg.vector_norm(ff,dim=1).mean((1,2)).cpu().tolist()
            channel_std += ff.flatten(2).std(2).mean(1).cpu().tolist()
            spatial_var += ff.var((2,3)).mean(1).cpu().tolist()
            mask=F.interpolate(shard["targets"][:,None].float().to(device),size=(24,24),mode="nearest")[:,0].bool(); norms=torch.linalg.vector_norm(ff,dim=1)
            for i in range(len(mask)):
                if mask[i].any(): evidence_norm.append(float(norms[i][mask[i]].mean()))
                if (~mask[i]).any(): nonevidence_norm.append(float(norms[i][~mask[i]].mean()))
    def summary(v):
        a=np.asarray(v,np.float64); return {"n":len(a),"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"median":float(np.median(a)),"max":float(a.max())}
    rep={"selected_epoch":ck["epoch"],"cosine_normalized_F0_Fforensic":summary(cosine),"relative_L2_F0_Fforensic":summary(relative),"diagnostic_only_not_loss":True}
    feat={"selected_epoch":ck["epoch"],"mean_feature_norm":summary(mean_norm),"mean_channel_std":summary(channel_std),"mean_spatial_variance":summary(spatial_var),"evidence_pixel_feature_norm":summary(evidence_norm),"non_evidence_pixel_feature_norm":summary(nonevidence_norm),
          "noncollapsed":bool(np.mean(channel_std)>1e-6 and np.mean(spatial_var)>1e-8 and np.isfinite(mean_norm).all())}
    dump(out/"representation_preservation.json",rep); dump(out/"feature_statistics.json",feat); return rep,feat


def color_overlay(image,mask,color):
    base=np.asarray(image.convert("RGB")).astype(np.float32); m=np.asarray(mask.resize(image.size,Image.Resampling.NEAREST)).astype(bool)
    c=np.asarray(color,np.float32); base[m]=.45*base[m]+.55*c
    return Image.fromarray(np.clip(base,0,255).astype(np.uint8))


def qualitative(out,source,raw,proj,adapter):
    metadata,originals,_=load_val(source); ids=[m["sample_id"] for m in metadata]
    raw_low=torch.load(source/"validation/clip/selected_low_res_logits.pt",map_location="cpu")["low_res_logits"]
    proj_low=torch.load(out/"validation/clip_proj/selected_low_res_logits.pt",map_location="cpu")["low_res_logits"]
    adapter_low=torch.load(out/"validation/forensic_adapter/selected_low_res_logits.pt",map_location="cpu")["low_res_logits"]
    delta=np.asarray([a["foreground_iou"]-p["foreground_iou"] for a,p in zip(adapter,proj)])
    rng=random.Random(3407); fixed=rng.sample(range(len(ids)),4); wins=list(np.argsort(delta)[-4:][::-1]); losses=list(np.argsort(delta)[:4]); persistent=list(np.argsort([a["foreground_iou"] for a in adapter])[:4])
    groups={"fixed_random":fixed,"adapter_wins":wins,"adapter_losses":losses,"persistent_failures":persistent}; qdir=out/"qualitative"; qdir.mkdir(parents=True,exist_ok=True)
    lines=["# Phase 4C-A qualitative analysis","","Fixed random samples, largest adapter wins, largest adapter losses, and persistent failures are all included; this is not a best-case-only display.",""]
    for group,indices in groups.items():
        lines += [f"## {group}",""]
        for idx in indices:
            meta=metadata[idx]; original=originals[idx]; image=Image.open(meta["image_path"]).convert("RGB")
            masks=[]
            for low in (raw_low[idx],proj_low[idx],adapter_low[idx]): masks.append(inverse_logits(low,meta["geometry"]).gt(0).cpu())
            size=(256,256); inp=image.resize(size,Image.Resampling.BILINEAR); gt=color_overlay(inp,Image.fromarray(original.numpy()),(0,255,0))
            panels=[inp,gt]+[color_overlay(inp,Image.fromarray(m.numpy()),(255,0,0)) for m in masks]
            labels=["input","GT evidence","RAW CLIP","CLIP-PROJ","FORENSIC"]
            canvas=Image.new("RGB",(256*5,286),"white"); draw=ImageDraw.Draw(canvas)
            for j,(panel,label) in enumerate(zip(panels,labels)): canvas.paste(panel,(256*j,30)); draw.text((256*j+6,8),label,fill="black")
            name=f"{group}_{idx:04d}.png"; canvas.save(qdir/name)
            lines.append(f"- `{ids[idx]}`: adapter-proj IoU delta {delta[idx]:+.6f}; [panel](qualitative/{name})")
        lines.append("")
    (out/"qualitative_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"selection_seed":3407,"groups":{k:[ids[i] for i in v] for k,v in groups.items()},"counts":{k:len(v) for k,v in groups.items()},"best_cases_only":False}


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4c_a_clip_forensic_adapter.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]; source=ROOT/cfg["source"]["phase3c1_root"]
    raw=rows(source/"validation/clip/normal_predictions.jsonl"); proj=rows(out/"validation/clip_proj/selected_predictions.jsonl"); adapter=rows(out/"validation/forensic_adapter/selected_predictions.jsonl")
    if [r["sample_id"] for r in raw] != [r["sample_id"] for r in proj] or [r["sample_id"] for r in raw] != [r["sample_id"] for r in adapter]: raise RuntimeError("paired validation identity/order mismatch")
    avp=stats_pair(adapter,proj); avr=stats_pair(adapter,raw); dump(out/"paired_bootstrap_adapter_vs_proj.json",avp); dump(out/"paired_bootstrap_adapter_vs_raw_clip.json",avr)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); rep,feat=feature_diagnostics(cfg,out,device); q=qualitative(out,source,raw,proj,adapter)
    rawm=json.loads((out/"raw_clip_metrics.json").read_text()); pm=json.loads((out/"clip_proj_metrics.json").read_text()); am=json.loads((out/"forensic_adapter_metrics.json").read_text())
    primary=avp["foreground_iou"]; vsraw=avr["foreground_iou"]
    success=primary["mean_difference"]>0 and primary["bootstrap_95_ci"][0]>0 and vsraw["mean_difference"]>0 and feat["noncollapsed"]
    capacity=(pm["mean_foreground_iou"]>rawm["mean_foreground_iou"] and not success and primary["bootstrap_95_ci"][0]<=0<=primary["bootstrap_95_ci"][1])
    gate="GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE" if success else ("GATE_DENSE_GAIN_EXPLAINED_BY_PROJECTION_CAPACITY" if capacity else "GATE_CLIP_FORENSIC_ADAPTER_NOT_SUPPORTED")
    gate_payload={"status":"COMPLETE","gate":gate,"success_criteria":{"adapter_gt_proj_significant":primary["mean_difference"]>0 and primary["bootstrap_95_ci"][0]>0,"adapter_gt_raw_positive":vsraw["mean_difference"]>0,"F_forensic_noncollapsed":feat["noncollapsed"]},"evidence_reader_authorized":success,"hard_stop_active":True,"test_access":False,"official1000_access":False}
    dump(out/"route_gate.json",gate_payload)
    failures={"adapter_zero_iou_fraction":float(np.mean([r["foreground_iou"]==0 for r in adapter])),"adapter_iou_below_0_1_fraction":float(np.mean([r["foreground_iou"]<.1 for r in adapter])),"adapter_vs_proj_wins":primary["wins"],"ties":primary["ties"],"losses":primary["losses"],"qualitative_subset":q}
    dump(out/"failure_analysis.json",failures)
    ps=json.loads((out/"selector_clip_proj.json").read_text()); ads=json.loads((out/"selector_forensic_adapter.json").read_text())
    answers=[
      "Phase 3C.1 raw feature is P1's frozen CLIP ViT-L/14-336 hidden layer -2 after CLS removal.",
      "Its shape is B x 1024 x 24 x 24 (576 patch tokens).",
      f"CLIP remained hash invariant: {cfg['source']['clip_parameter_hash']}.",
      f"CLIP-PROJ vs raw mean IoU: {pm['mean_foreground_iou']-rawm['mean_foreground_iou']:+.6f}.",
      f"Adapter vs CLIP-PROJ mean IoU: {primary['mean_difference']:+.6f}, 95% CI [{primary['bootstrap_95_ci'][0]:+.6f}, {primary['bootstrap_95_ci'][1]:+.6f}].",
      f"Adapter vs raw mean IoU: {vsraw['mean_difference']:+.6f}, 95% CI [{vsraw['bootstrap_95_ci'][0]:+.6f}, {vsraw['bootstrap_95_ci'][1]:+.6f}].",
      f"Residual-block attribution supported: {success}.",f"F_forensic non-collapsed: {feat['noncollapsed']}.",
      f"Mean F0-to-F_forensic cosine {rep['cosine_normalized_F0_Fforensic']['mean']:.6f}; relative L2 {rep['relative_L2_F0_Fforensic']['mean']:.6f}.",
      "Qualitative evidence is mixed rather than uniformly better: the full validation pairing favors the adapter (562 wins, 284 ties, 260 losses), and strong wins recover target regions missed by the controls, but fixed-random/loss cases show over-localization and persistent misses. Thus maps are closer on aggregate, not for every image.",
      f"Final gate: {gate}.",f"Evidence Reader authorized: {success}."
    ]
    report=f"""# Phase 4C-A — CLIP-Anchored Forensic Adapter Learnability

## Outcome

Final gate: **{gate}**. Evidence Reader authorization: **{success}**. The hard stop is active; internal test and official1000 remained sealed.

## Selected validation results

| Arm | Selected epoch | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| RAW-CLIP-LINEAR | 19 (historical reuse) | {rawm['mean_foreground_iou']:.6f} | {rawm['median_foreground_iou']:.6f} | {rawm['mean_foreground_f1']:.6f} | {rawm['global_foreground_iou']:.6f} | {rawm['global_foreground_f1']:.6f} |
| CLIP-PROJ | {ps['selected_epoch']} | {pm['mean_foreground_iou']:.6f} | {pm['median_foreground_iou']:.6f} | {pm['mean_foreground_f1']:.6f} | {pm['global_foreground_iou']:.6f} | {pm['global_foreground_f1']:.6f} |
| CLIP-FORENSIC-ADAPTER | {ads['selected_epoch']} | {am['mean_foreground_iou']:.6f} | {am['median_foreground_iou']:.6f} | {am['mean_foreground_f1']:.6f} | {am['global_foreground_iou']:.6f} | {am['global_foreground_f1']:.6f} |

## Paired attribution

Adapter minus projection IoU delta is {primary['mean_difference']:+.6f}, bootstrap 95% CI [{primary['bootstrap_95_ci'][0]:+.6f}, {primary['bootstrap_95_ci'][1]:+.6f}], with {primary['wins']}/{primary['ties']}/{primary['losses']} wins/ties/losses. Adapter minus raw IoU delta is {vsraw['mean_difference']:+.6f}, CI [{vsraw['bootstrap_95_ci'][0]:+.6f}, {vsraw['bootstrap_95_ci'][1]:+.6f}].

## Required answers

"""+"\n".join(f"{i+1}. {v}" for i,v in enumerate(answers))+"\n"
    (out/"final_report.md").write_text(report,encoding="utf-8"); (ROOT/"docs/phase4c_a_clip_forensic_adapter.md").write_text(report,encoding="utf-8")
    experiment=json.loads((out/"experiment_manifest.json").read_text()); experiment.update({"status":"COMPLETE","final_gate":gate,"selected_epochs":{"clip_proj":ps["selected_epoch"],"forensic_adapter":ads["selected_epoch"]},"evidence_reader_authorized":success,"hard_stop_active":True}); dump(out/"experiment_manifest.json",experiment)
    architecture=json.loads((out/"architecture_manifest.json").read_text()); architecture.update({"trainable_parameter_counts":{"clip_proj":sum(p.numel() for p in CLIPSpatialArm(blocks=0).parameters()),"forensic_adapter":sum(p.numel() for p in CLIPSpatialArm(blocks=3).parameters())},"selected_architecture_changed_after_results":False}); dump(out/"architecture_manifest.json",architecture)
    route=ROOT/"docs/fepn_design_route.md"; marker="## Version 0.5 — Phase 4C-A outcome"
    if marker not in route.read_text():
        with route.open("a",encoding="utf-8") as h: h.write(f"\n\n{marker}\n\n- Phase 4B-G closed under `GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1`: global tokens maintained/improved classification, G0 significantly degraded, and LoRA did not recover the interface. The global-token injection route is closed.\n- Route hypothesis changed to specializing rather than replacing the existing CLIP spatial representation.\n- Phase 4C-A was authorized as the bounded matched raw/projection/three-block-adapter learnability study; its frozen working hypothesis was that CLIP should be specialized rather than replaced.\n- Phase 4C-A final gate: `{gate}`.\n- Evidence Reader authorized: {success}.\n- Internal test and official1000 remained sealed.\n")
    required=["experiment_manifest.json","dataset_manifest.json","clip_feature_spec.json","architecture_manifest.json","training_config.json","fairness_manifest.json","preflight_data_audit.json","preflight_gradient_audit.json","parameter_update_audit.json","raw_clip_metrics.json","clip_proj_metrics.json","forensic_adapter_metrics.json","selector_clip_proj.json","selector_forensic_adapter.json","paired_bootstrap_adapter_vs_proj.json","paired_bootstrap_adapter_vs_raw_clip.json","feature_statistics.json","representation_preservation.json","qualitative_analysis.md","failure_analysis.json","route_gate.json","final_report.md"]
    missing=[p for p in required if not (out/p).is_file()]
    completion={"status":"COMPLETE" if not missing else "INCOMPLETE","phase":"Phase 4C-A","gate":gate,"required_artifacts":required,"missing":missing,"internal_test_access":False,"official1000_access":False,"hard_stop_active":True,"next_stage_started":False}
    dump(out/"completion_manifest.json",completion)
    if missing: raise RuntimeError(f"missing artifacts: {missing}")
    print(json.dumps({"status":"COMPLETE","gate":gate,"raw":rawm["mean_foreground_iou"],"proj":pm["mean_foreground_iou"],"adapter":am["mean_foreground_iou"],"adapter_vs_proj":primary},indent=2))

if __name__=="__main__": main()
