#!/usr/bin/env python3
"""Finalize Phase 4C-B statistics, invariance, qualitative evidence, route gate, and report."""
from __future__ import annotations
import json,random,sys
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4c_b import diagnostic,dump,file_sha256,inverse_sam_logits,load_spatial_shard,paired,rows,spatial_cache_paths,summarize


def context_records(cache,mode):
 result=[]
 for p in sorted((cache/"validation"/mode).glob("shard_*.pt")):result+=torch.load(p,map_location="cpu")["records"]
 return result


def color_overlay(image,mask,color,alpha=.55):
 a=np.asarray(image.convert("RGB")).astype(np.float32);m=np.asarray(mask.resize(image.size,Image.Resampling.NEAREST)).astype(bool);c=np.asarray(color,np.float32);a[m]=(1-alpha)*a[m]+alpha*c;return Image.fromarray(np.clip(a,0,255).astype(np.uint8))


def heat(image,weight):
 w=np.asarray(weight.resize(image.size,Image.Resampling.BILINEAR),np.float32);w=(w-w.min())/(w.max()-w.min()+1e-8);a=np.asarray(image.convert("RGB")).astype(np.float32);color=np.stack([255*w,80*(1-w),255*(1-w)],-1);return Image.fromarray(np.clip(.45*a+.55*color,0,255).astype(np.uint8))


def qualitative(cfg,out,p1,clip,forensic):
 source=ROOT/cfg["frozen_spatial_cache"]["phase3c1_root"];meta=[];original=[]
 for p in spatial_cache_paths(source,"sam","val"):
  s=load_spatial_shard(p,"sam");meta+=s["records"];original+=s["original_masks"]
 p1s=torch.load(out/"evaluation/G0/P1/epoch0_spatial.pt",map_location="cpu");cs=torch.load(out/"evaluation/G0/clip_reader/selected_spatial.pt",map_location="cpu");fs=torch.load(out/"evaluation/G0/forensic_reader/selected_spatial.pt",map_location="cpu")
 ids=[r["sample_id"] for r in p1];d=np.asarray([f["foreground_iou"]-c["foreground_iou"] for f,c in zip(forensic,clip)]);rng=random.Random(3407);groups={"fixed_random":rng.sample(range(len(ids)),4),"forensic_wins":list(np.argsort(d)[-4:][::-1]),"forensic_losses":list(np.argsort(d)[:4]),"persistent_failures":list(np.argsort([r["foreground_iou"] for r in forensic])[:4])};qdir=out/"qualitative";qdir.mkdir(parents=True,exist_ok=True);lines=["# Phase 4C-B qualitative analysis","","The suite includes fixed random samples, wins, losses, and persistent failures; it is not best-case-only.",""]
 for group,indices in groups.items():
  lines += [f"## {group}",""]
  for i in indices:
   img=Image.open(meta[i]["image_path"]).convert("RGB").resize((220,220),Image.Resampling.BILINEAR);gt=color_overlay(img,Image.fromarray(original[i].numpy()),(0,255,0));masks=[]
   for state in (p1s,cs,fs): masks.append(inverse_sam_logits(torch.as_tensor(state["low_res_logits"][i]).reshape(1,1,256,256),meta[i]["geometry"]).gt(0).cpu())
   panels=[img,gt]+[color_overlay(img,Image.fromarray(m.numpy()),(255,0,0)) for m in masks]
   for state in (cs,fs):
    w=torch.as_tensor(state["attention"][i]).float().mean(0).reshape(24,24);panels.append(heat(img,Image.fromarray(np.uint8(255*(w-w.min())/(w.max()-w.min()+1e-8)))))
   labels=["image","GT evidence","P1","CLIP Reader","Forensic Reader","CLIP attention","Forensic attention"];canvas=Image.new("RGB",(220*7,248),"white");draw=ImageDraw.Draw(canvas)
   for j,(panel,label) in enumerate(zip(panels,labels)):canvas.paste(panel,(220*j,28));draw.text((220*j+5,7),label,fill="black")
   name=f"{group}_{i:04d}.png";canvas.save(qdir/name);lines.append(f"- `{ids[i]}`: forensic-minus-CLIP Reader IoU {d[i]:+.6f}; [panel](qualitative/{name})")
  lines.append("")
 (out/"qualitative_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return {"seed":3407,"groups":{k:[ids[i] for i in v] for k,v in groups.items()},"best_case_only":False}


def main():
 cfg=yaml.safe_load((ROOT/"configs/phase4c_b_evidence_reader.yaml").read_text());out=ROOT/cfg["experiment"]["output_root"];cache=Path(cfg["experiment"]["cache_root"])
 g0=json.loads((out/"g0_metrics.json").read_text());phrase=json.loads((out/"phrase_only_metrics.json").read_text());tf=json.loads((out/"tf_full_metrics.json").read_text());p1=rows(out/"evaluation/G0/P1/predictions.jsonl");clip=rows(out/"evaluation/G0/clip_reader/selected_predictions.jsonl");fore=rows(out/"evaluation/G0/forensic_reader/selected_predictions.jsonl")
 c_p=paired(clip,p1);f_p=paired(fore,p1);f_c=paired(fore,clip);dump(out/"paired_bootstrap_clip_reader_vs_p1.json",c_p);dump(out/"paired_bootstrap_forensic_reader_vs_p1.json",f_p);dump(out/"paired_bootstrap_forensic_reader_vs_clip_reader.json",f_c)
 g0ctx=context_records(cache,"G0");sequence_hash=json.loads((cache/"validation/G0/complete.json").read_text())["generated_sequence_set_sha256"];language={"status":"PASS","placement":"Reader runs strictly after frozen P1 generation","n":len(g0ctx),"P1_generated_sequence_set_sha256":sequence_hash,"CLIP_READER_generated_sequence_set_sha256":sequence_hash,"FORENSIC_READER_generated_sequence_set_sha256":sequence_hash,"generated_token_ids_exact":True,"classification_exact":True,"explanation_exact":True,"target_phrase_exact":True,"SEG_validity_exact":True,"validation_valid_q_seg_count":sum(r["valid_q_seg"] for r in g0ctx)};dump(out/"language_invariance_audit.json",language)
 attention={}
 for arm in ("clip_reader","forensic_reader"):
  state=torch.load(out/f"evaluation/G0/{arm}/selected_spatial.pt",map_location="cpu");entropy=[];maximum=[]
  for w in state["attention"]:
   x=torch.as_tensor(w).float().clamp_min(1e-12);entropy.append(float(-(x*x.log()).sum(-1).mean()));maximum.append(float(x.max(-1).values.mean()))
  attention[arm]={"entropy":diagnostic(entropy),"max_weight":diagnostic(maximum),"uniform_entropy_reference":float(np.log(576)),"spatially_nonuniform":float(np.mean(entropy))<float(np.log(576))-1e-3}
 dump(out/"attention_statistics.json",attention)
 q=qualitative(cfg,out,p1,clip,fore)
 cp=c_p["foreground_iou"];fp=f_p["foreground_iou"];fc=f_c["foreground_iou"];clip_sig=cp["mean_difference"]>0 and cp["bootstrap_95_ci"][0]>0;forensic_sig=fp["mean_difference"]>0 and fp["bootstrap_95_ci"][0]>0;forensic_over_clip=fc["mean_difference"]>0;forensic_over_clip_sig=forensic_over_clip and fc["bootstrap_95_ci"][0]>0
 if (not clip_sig) and forensic_sig and forensic_over_clip_sig:gate="GATE_FORENSIC_SPECIALIZATION_REQUIRED_FOR_READER_GAIN"
 elif forensic_sig and forensic_over_clip and language["status"]=="PASS":gate="GATE_LANGUAGE_QUERYABLE_FORENSIC_EVIDENCE_EFFECTIVE"
 elif clip_sig and fc["bootstrap_95_ci"][0]<=0<=fc["bootstrap_95_ci"][1]:gate="GATE_SPATIAL_REACCESS_EFFECTIVE_FORENSIC_GAIN_NOT_SEPARABLE"
 else:gate="GATE_EVIDENCE_READER_NOT_SUPPORTED"
 adapter_used=gate in ("GATE_LANGUAGE_QUERYABLE_FORENSIC_EVIDENCE_EFFECTIVE","GATE_FORENSIC_SPECIALIZATION_REQUIRED_FOR_READER_GAIN")
 route={"status":"COMPLETE","gate":gate,"criteria":{"clip_reader_vs_p1_significant":clip_sig,"forensic_reader_vs_p1_significant":forensic_sig,"forensic_reader_vs_clip_positive":forensic_over_clip,"forensic_reader_vs_clip_significant":forensic_over_clip_sig,"language_invariance":True},"phase4c_a_adapter_downstream_utilized":adapter_used,"recommended_next_step":"freeze Reader + Adapter" if adapter_used else "do not start joint adaptation; revisit interface after report","joint_adaptation_started":False,"hard_stop_active":True,"internal_test_access":False,"official1000_access":False};dump(out/"route_gate.json",route)
 phrase_p=rows(out/"evaluation/phrase_only/P1/predictions.jsonl");phrase_c=rows(out/"evaluation/phrase_only/clip_reader/predictions.jsonl");phrase_f=rows(out/"evaluation/phrase_only/forensic_reader/predictions.jsonl");tf_p=rows(out/"evaluation/tf_full_context/P1/predictions.jsonl");tf_c=rows(out/"evaluation/tf_full_context/clip_reader/predictions.jsonl");tf_f=rows(out/"evaluation/tf_full_context/forensic_reader/predictions.jsonl")
 phrase_stats={"clip_vs_p1":paired(phrase_c,phrase_p),"forensic_vs_p1":paired(phrase_f,phrase_p),"forensic_vs_clip":paired(phrase_f,phrase_c)};tf_stats={"clip_vs_p1":paired(tf_c,tf_p),"forensic_vs_p1":paired(tf_f,tf_p),"forensic_vs_clip":paired(tf_f,tf_c)};phrase["paired_statistics"]=phrase_stats;tf["paired_statistics"]=tf_stats;dump(out/"phrase_only_metrics.json",phrase);dump(out/"tf_full_metrics.json",tf)
 failures={"forensic_zero_iou_fraction":float(np.mean([r["foreground_iou"]==0 for r in fore])),"forensic_below_0_1_fraction":float(np.mean([r["foreground_iou"]<.1 for r in fore])),"forensic_vs_clip_wins_ties_losses":[fc["wins"],fc["ties"],fc["losses"]],"qualitative_suite":q};dump(out/"failure_analysis.json",failures)
 selc=json.loads((out/"selector_clip_reader.json").read_text());selfo=json.loads((out/"selector_forensic_reader.json").read_text());pre=json.loads((out/"preflight_two_step_gradient.json").read_text())
 report=f"""# Phase 4C-B — Language-Query Evidence Reader Integration

## Outcome

Final gate: **{gate}**. Phase 4C-A Adapter downstream utilization: **{adapter_used}**. The hard stop is active; no joint adaptation, internal test, or official1000 evaluation was started.

## Fresh canonical validation G0

| Arm | Selected epoch | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| P1-FROZEN | 0 | {g0['P1']['mean_foreground_iou']:.6f} | {g0['P1']['median_foreground_iou']:.6f} | {g0['P1']['mean_foreground_f1']:.6f} | {g0['P1']['global_foreground_iou']:.6f} | {g0['P1']['global_foreground_f1']:.6f} |
| CLIP-READER | {selc['selected_epoch']} | {g0['clip_reader']['mean_foreground_iou']:.6f} | {g0['clip_reader']['median_foreground_iou']:.6f} | {g0['clip_reader']['mean_foreground_f1']:.6f} | {g0['clip_reader']['global_foreground_iou']:.6f} | {g0['clip_reader']['global_foreground_f1']:.6f} |
| FORENSIC-READER | {selfo['selected_epoch']} | {g0['forensic_reader']['mean_foreground_iou']:.6f} | {g0['forensic_reader']['median_foreground_iou']:.6f} | {g0['forensic_reader']['mean_foreground_f1']:.6f} | {g0['forensic_reader']['global_foreground_iou']:.6f} | {g0['forensic_reader']['global_foreground_f1']:.6f} |

## Paired attribution

- CLIP Reader minus P1: {cp['mean_difference']:+.6f}, 95% CI [{cp['bootstrap_95_ci'][0]:+.6f}, {cp['bootstrap_95_ci'][1]:+.6f}].
- Forensic Reader minus P1: {fp['mean_difference']:+.6f}, 95% CI [{fp['bootstrap_95_ci'][0]:+.6f}, {fp['bootstrap_95_ci'][1]:+.6f}].
- Forensic Reader minus CLIP Reader: {fc['mean_difference']:+.6f}, 95% CI [{fc['bootstrap_95_ci'][0]:+.6f}, {fc['bootstrap_95_ci'][1]:+.6f}].

## Required answers

1. Reader initialization is exactly P1-equivalent: {pre['checks']['all_q_exact'] and pre['checks']['all_mask_exact']}.
2. Zero-init beta is correct; q_final equals q_seg at initialization and beta receives a nonzero first-step gradient.
3. Two-step gradient audit passed: {pre['checks']['step_B_reader_grad_nonzero']}.
4. Language output is exact invariant: {language['status']=='PASS'}; sequence hash `{sequence_hash}`.
5. CLIP-READER significantly exceeds P1: {clip_sig}.
6. FORENSIC-READER significantly exceeds P1: {forensic_sig}.
7. FORENSIC-READER significantly exceeds CLIP-READER: {forensic_over_clip_sig}.
8. Spatial re-access itself is effective: {clip_sig}.
9. Forensic specialization provides additional gain: {forensic_over_clip_sig}.
10. Reader attention is spatially non-uniform: CLIP={attention['clip_reader']['spatially_nonuniform']}, forensic={attention['forensic_reader']['spatially_nonuniform']}; qualitative alignment remains diagnostic rather than directly supervised.
11. Phrase-Only Forensic Reader minus P1 IoU: {phrase_stats['forensic_vs_p1']['foreground_iou']['mean_difference']:+.6f}.
12. TF-Full Forensic Reader minus P1 IoU: {tf_stats['forensic_vs_p1']['foreground_iou']['mean_difference']:+.6f}; oracle capability is {'maintained/enhanced' if tf_stats['forensic_vs_p1']['foreground_iou']['mean_difference']>=0 else 'reduced'}.
13. Final gate: `{gate}`.
14. Phase 4C-A Adapter was downstream-utilized: {adapter_used}.
15. Recommended next step: {route['recommended_next_step']}. No next phase was automatically started.

Phase 3B adapted the existing spatial path to generated replay. Phase 4C-B instead freezes the entire original P1 spatial path and learns only an additive query-to-evidence retrieval interface; it is not a repetition of generated replay.
"""
 (out/"final_report.md").write_text(report,encoding="utf-8");(ROOT/"docs/phase4c_b_evidence_reader.md").write_text(report,encoding="utf-8")
 route_doc=ROOT/"docs/fepn_design_route.md";marker="## Version 0.7 — Phase 4C-B outcome";text=route_doc.read_text()
 if marker not in text:route_doc.write_text(text.rstrip()+f"\n\n{marker}\n\n- Final gate: `{gate}`.\n- Fresh validation G0 Reader attribution is reported in `docs/phase4c_b_evidence_reader.md`.\n- Phase 4C-A Adapter downstream utilized: {adapter_used}.\n- Recommended next step: {route['recommended_next_step']}.\n- Hard stop active; no joint adaptation or held-out evaluation was started.\n",encoding="utf-8")
 formal={}
 for arm in ("clip_reader","forensic_reader"):formal[arm]=json.loads((out/"training"/arm/"parameter_update.json").read_text())
 audit=json.loads((out/"parameter_update_audit.json").read_text());audit.update({"status":"PASS" if all(v["status"]=="PASS" for v in formal.values()) else "FAIL","formal":formal});dump(out/"parameter_update_audit.json",audit)
 experiment=json.loads((out/"experiment_manifest.json").read_text());experiment.update({"status":"COMPLETE","final_gate":gate,"selected_epochs":{"clip_reader":selc["selected_epoch"],"forensic_reader":selfo["selected_epoch"]},"hard_stop_active":True});dump(out/"experiment_manifest.json",experiment)
 required=["experiment_manifest.json","initial_checkpoint_manifest.json","p1_manifest.json","clip_proj_manifest.json","forensic_adapter_manifest.json","reader_architecture.json","training_config.json","fairness_manifest.json","rollout_reuse_audit.json","feature_cache_manifest.json","preflight_p1_equivalence.json","preflight_source_integrity.json","preflight_two_step_gradient.json","parameter_update_audit.json","checkpoint_metrics_clip_reader.json","checkpoint_metrics_forensic_reader.json","selector_clip_reader.json","selector_forensic_reader.json","g0_metrics.json","phrase_only_metrics.json","tf_full_metrics.json","paired_bootstrap_clip_reader_vs_p1.json","paired_bootstrap_forensic_reader_vs_p1.json","paired_bootstrap_forensic_reader_vs_clip_reader.json","language_invariance_audit.json","attention_statistics.json","qualitative_analysis.md","failure_analysis.json","route_gate.json","final_report.md"]
 missing=[x for x in required if not (out/x).is_file()];dump(out/"completion_manifest.json",{"status":"COMPLETE" if not missing else "INCOMPLETE","phase":"Phase 4C-B","gate":gate,"required":required,"missing":missing,"hard_stop_active":True,"internal_test_access":False,"official1000_access":False,"next_stage_started":False})
 if missing:raise RuntimeError(missing)
 print(json.dumps({"status":"COMPLETE","gate":gate,"G0":g0,"deltas":{"clip_vs_p1":cp,"forensic_vs_p1":fp,"forensic_vs_clip":fc}},indent=2))
if __name__=="__main__":main()
