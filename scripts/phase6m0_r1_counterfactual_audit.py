#!/usr/bin/env python3
"""Phase6M0: TRAIN reconstruction and frozen SAM counterfactuals; DEV transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts.phase6m0_r1_intervention_audit import (OUT, EPS, KS, Runtime, Moments, describe, dump,
                                                      file_sha256, require, spectrum, vectorize)
from tools.phase4c_b import inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import evidence_feature, invalid_record
from model.pcerf import sam_lowres_to_original_normalized

SEED = 3407


def paired(x, y):
    d = np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    require(len(d)>0 and np.isfinite(d).all(),"invalid paired values")
    rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(20):
        ix=rng.integers(0,len(d),size=(100,len(d)))
        boot.append(d[ix].mean(1))
    boot=np.concatenate(boot)
    nonzero=d[np.abs(d)>1e-12]
    p=float(stats.wilcoxon(nonzero,zero_method="wilcox").pvalue) if len(nonzero) else 1.0
    return {"n":len(d),"mean":float(d.mean()),"median":float(np.median(d)),
            "bootstrap_95_ci":np.quantile(boot,[.025,.975]).tolist(),"bootstrap_repeats":2000,
            "wins":int((d>1e-12).sum()),"ties":int((np.abs(d)<=1e-12).sum()),
            "losses":int((d< -1e-12).sum()),"wilcoxon_p_two_sided":p}


def correlation(x,y):
    a,b=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    if np.std(a)<1e-12 or np.std(b)<1e-12: return {"pearson":None,"spearman":None,"n":len(a)}
    return {"pearson":float(stats.pearsonr(a,b).statistic),
            "spearman":float(stats.spearmanr(a,b).statistic),"n":len(a)}


def projection_energy(x:torch.Tensor,basis:torch.Tensor,k:int):
    x=x.double();v=basis[:,:k]
    total=float((x*x).sum())
    if total<=EPS: return 1.0
    return float(((x@v)**2).sum())/total


def image_projection_errors(x:torch.Tensor,basis:torch.Tensor):
    x=x.double(); vec=basis
    norms=x.norm(dim=1).clamp_min(EPS)
    energy=x.square().sum(1)
    coordinates=(x@vec[:,:max(KS)]).square()
    errors={}
    for k in KS:
        residual=(energy-coordinates[:,:k].sum(1)).clamp_min(0).sqrt()/norms
        errors[str(k)]=float(residual.mean())
    return errors


@torch.no_grad()
def full_rectifier(rt:Runtime,store,own_sid:str,source_sid:str,permutation=None):
    s64, _, _,sc,cc=e2.phase4f_spatial_batch(store,[own_sid],rt.device)
    _,raw,_,_,_=e2.phase4f_spatial_batch(store,[source_sid],rt.device)
    feature=evidence_feature(rt.source,raw)
    if permutation is not None:
        feature=feature.flatten(2)[:,:,permutation].reshape_as(feature)
    valid=torch.ones(1,576,dtype=torch.bool,device=rt.device)
    with torch.autocast(device_type=rt.device.type,dtype=torch.bfloat16):
        value=rt.rectifier(s64,feature,sc,cc,valid)
    return value["image_embeddings"],value["support"].reshape(1,1,64,64).bool()


@torch.no_grad()
def decode(rt,out,gate,store,sid):
    adapted=hc.gated_embedding(out["s64"],out["rect"],gate)
    with torch.autocast(device_type=rt.device.type,enabled=False):
        low=rt.sam(out["batch"]["q_seg"],adapted.to(torch.bfloat16))
    return sam_lowres_to_original_normalized(low,store.geometries[sid],output_hw=(256,256))[0,0].float().cpu()


def normalized_target(store,sid):
    target=store._values(sid)[2]
    rh,rw=map(int,store.geometries[sid]["resized_hw"])
    return F.interpolate(target[None,None,:rh,:rw].float(),size=(256,256),mode="nearest")[0,0].bool()


def summarize_condition(rows,condition):
    fields=("mean_u","intervention_ratio","delta_iou","delta_f1","logit_change",
            "prob_change","mask_change","iou","f1","projection_energy_K90","projection_energy_K95")
    return {key:describe([row[condition][key] for row in rows]) for key in fields}


def condition_record(logit,base_logit,target,sid,gate,support,delta,s64,basis,k90,k95):
    record=metric_record(sid,logit,target)
    base=metric_record(sid,base_logit,target)
    mask=support[0,0]
    mean_u=float(gate[0,0][mask].float().mean()) if bool(mask.any()) else 0.0
    sn=s64[0].float().permute(1,2,0)[mask]
    dn=delta[0].float().permute(1,2,0)[mask]
    ratio=float((dn.norm(dim=1)/sn.norm(dim=1).clamp_min(EPS)).mean()) if len(dn) else 0.0
    return {"mean_u":mean_u,"intervention_ratio":ratio,
            "delta_iou":record["foreground_iou"]-base["foreground_iou"],
            "delta_f1":record["foreground_f1"]-base["foreground_f1"],
            "iou":record["foreground_iou"],"f1":record["foreground_f1"],
            "logit_change":float((logit-base_logit).abs().mean()),
            "prob_change":float((logit.sigmoid()-base_logit.sigmoid()).abs().mean()),
            "mask_change":float((logit>0).ne(base_logit>0).float().mean()),
            "projection_energy_K90":projection_energy(dn.cpu(),basis,k90) if len(dn) else None,
            "projection_energy_K95":projection_energy(dn.cpu(),basis,k95) if len(dn) else None,
            "correction_norm":float(dn.norm(dim=1).mean()) if len(dn) else 0.0}


def train(rt:Runtime):
    rank=json.loads((OUT/"train_rank.json").read_text())
    require((OUT/"train_magnitude.json").exists() and (OUT/"train_basis.pt").exists(),"TRAIN moments not frozen")
    basis=torch.load(OUT/"train_basis.pt",map_location="cpu",weights_only=False)["delta"]["eigenvectors"]
    k90=rank["delta"]["image_balanced"]["rank"]["90"]
    k95=rank["delta"]["image_balanced"]["rank"]["95"]
    thresholds=rank["active_thresholds_train_image_balanced"]
    ids,store,cache,data=rt.population("train")
    moments={name:{"C":Moments(),"delta":Moments()} for name in thresholds}
    recon={"C":{str(k):[] for k in KS},"delta":{str(k):[] for k in KS}}
    c_basis=torch.load(OUT/"train_basis.pt",map_location="cpu",weights_only=False)["C"]["eigenvectors"]
    rows=[]
    permutation=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(rt.device)
    valid_n=0
    with torch.no_grad():
        for i,sid in enumerate(ids):
            if not bool(cache["valid"][i]): continue
            out=rt.forward(ids,store,cache,data,i)
            s,c,d,u,sn,cn,dn,r=vectorize(out)
            for key,cut in thresholds.items():
                active=r>=cut
                if bool(active.any()):
                    moments[key]["C"].add(c[active])
                    moments[key]["delta"].add(d[active])
            for key,x,v in (("C",c,c_basis),("delta",d,basis)):
                errors=image_projection_errors(x,v)
                for k,value in errors.items(): recon[key][k].append(value)
            batch=out["batch"]
            crossed=dict(batch)
            j=(i+1)%len(ids)
            crossed["F24"]=data["F24"][j:j+1].to(rt.device)
            crossed["z_F24"]=data["z_F24"][j:j+1].to(rt.device)
            u_cross=hc.utility_forward(rt.model,crossed)["U"]
            u_shuffle=hc.utility_forward(rt.model,batch,permutation=permutation)["U"]
            gate_cross=hc.gate_to_sam_grid(u_cross,out["sc"])*out["support"].float()
            gate_shuffle=hc.gate_to_sam_grid(u_shuffle,out["sc"])*out["support"].float()
            # Counterfactual A keeps the matched Rectifier correction fixed.
            interventions={"matched":(out["gate"],out["rect"],out["support"]),
                           "cross_utility":(gate_cross,out["rect"],out["support"]),
                           "shuffle_utility":(gate_shuffle,out["rect"],out["support"])}
            # Counterfactual B changes the forensic source in both branches.
            rc,sc=full_rectifier(rt,store,sid,ids[j])
            rs,ss=full_rectifier(rt,store,sid,sid,permutation)
            interventions["cross_full"]=(hc.gate_to_sam_grid(u_cross,out["sc"])*sc.float(),rc,sc)
            interventions["shuffle_full"]=(hc.gate_to_sam_grid(u_shuffle,out["sc"])*ss.float(),rs,ss)
            base_out=dict(out);base_out["rect"]=out["s64"]
            base_logit=decode(rt,base_out,torch.zeros_like(out["gate"]),store,sid)
            target=normalized_target(store,sid)
            base_record=metric_record(sid,base_logit,target)
            row={"sample_id":sid,"partner_id":ids[j],"baseline_iou":base_record["foreground_iou"],
                 "baseline_f1":base_record["foreground_f1"],"matched_r":float(r.mean()),
                 "matched_C_norm":float(cn.mean()),"matched_delta_norm":float(dn.mean())}
            for condition,(gate,rect,support) in interventions.items():
                changed=dict(out); changed["rect"]=rect
                logit=decode(rt,changed,gate,store,sid)
                adapted=hc.gated_embedding(out["s64"],rect,gate)
                delta=adapted.float()-out["s64"].float()
                row[condition]=condition_record(logit,base_logit,target,sid,gate,support,delta,out["s64"],basis,k90,k95)
            rows.append(row)
            valid_n+=1
            if valid_n%100==0: print(json.dumps({"stage":"TRAIN_COUNTERFACTUAL","done":valid_n,"total":int(cache["valid"].sum())}),flush=True)
    require(valid_n==8741,"TRAIN counterfactual traversal incomplete")
    active={name:{key:m.finish()[0] for key,m in per.items()} for name,per in moments.items()}
    recon_report={key:{k:describe(v,(75,90,95)) for k,v in per.items()} for key,per in recon.items()}
    dump("diagnostics/train_active_rank.json",active)
    dump("diagnostics/train_reconstruction.json",recon_report)
    conditions=("matched","cross_utility","shuffle_utility","cross_full","shuffle_full")
    summary={"n":len(rows),"task_metric_domain":"TRAIN Phase4F SAM-canvas target, padding cropped and nearest-resampled to original-normalized 256 grid; frozen SAM lowres logits follow the historical sam_lowres_to_original_normalized geometry path",
             "A_historical_utility_semantics":{key:summarize_condition(rows,key) for key in conditions[:3]},
             "B_full_evidence_mismatch_new_diagnostic":{key:summarize_condition(rows,key) for key in conditions[3:]},
             "pairwise":{},"correlations":{}}
    for rhs in conditions[1:]:
        summary["pairwise"]["matched_minus_"+rhs]={field:paired([row["matched"][field] for row in rows],
                                                               [row[rhs][field] for row in rows])
                                                   for field in ("mean_u","intervention_ratio","delta_iou","delta_f1","mask_change")}
    summary["correlations"]["mean_U_vs_matched_gain"]=correlation([r["matched"]["mean_u"] for r in rows],[r["matched"]["delta_iou"] for r in rows])
    for rhs in ("cross_utility","shuffle_utility"):
        summary["correlations"]["delta_U_vs_delta_gain_"+rhs]=correlation(
            [r["matched"]["mean_u"]-r[rhs]["mean_u"] for r in rows],
            [r["matched"]["delta_iou"]-r[rhs]["delta_iou"] for r in rows])
    summary["correlations"].update({field+"_vs_gain":correlation([r[field] for r in rows],[r["matched"]["delta_iou"] for r in rows])
                                     for field in ("matched_C_norm","matched_delta_norm","matched_r")})
    order=np.argsort([r["matched_r"] for r in rows]);quartiles={}
    for ix in range(4):
        block=[rows[int(j)] for j in np.array_split(order,4)[ix]]
        quartiles[str(ix+1)]={field:describe([r[field] for r in block]) for field in ("baseline_iou","matched_C_norm","matched_delta_norm")}
        quartiles[str(ix+1)]["gain"]=describe([r["matched"]["delta_iou"] for r in block])
        quartiles[str(ix+1)]["gate"]=describe([r["matched"]["mean_u"] for r in block])
    summary["magnitude_quartiles_train_only"]=quartiles
    require(tensor_state_sha256(rt.model.state_dict())==rt.hashes["utility_state"] and
            tensor_state_sha256(rt.rectifier.state_dict())==rt.hashes["rectifier_state"] and
            tensor_state_sha256(rt.sam.state_dict())==rt.hashes["sam"] and
            tensor_state_sha256(rt.source.state_dict())==rt.hashes["source"],"frozen model changed during counterfactual")
    dump("train_counterfactual.json",summary)
    dump("diagnostics/train_counterfactual_rows.json",{"rows":rows})
    print(json.dumps({"status":"TRAIN_COUNTERFACTUAL_COMPLETE","n":len(rows)}),flush=True)


def dev(rt:Runtime):
    require(all((OUT/name).exists() for name in ("train_rank.json","train_magnitude.json","train_counterfactual.json","train_basis.pt")),"TRAIN audit incomplete")
    train_rank=json.loads((OUT/"train_rank.json").read_text())
    b=torch.load(OUT/"train_basis.pt",map_location="cpu",weights_only=False)["delta"]["eigenvectors"]
    ranks={label:train_rank["delta"]["image_balanced"]["rank"][label] for label in ("90","95","99")}
    ids,store,cache,data=rt.population("val")
    records=[];energy={key:[] for key in ranks};error={key:[] for key in ranks}
    for i,sid in enumerate(ids):
        if not bool(cache["valid"][i]):
            row=invalid_record(sid,data["original_masks"][i]);row["valid_g0"]=False;records.append(row);continue
        with torch.no_grad():
            out=rt.forward(ids,store,cache,data,i)
            with torch.autocast(device_type=rt.device.type,enabled=False):
                low=rt.sam(out["batch"]["q_seg"],out["adapted"].to(torch.bfloat16))
        logit=inverse_sam_logits(low,data["sam_geometries"][i])
        row=metric_record(sid,logit,data["original_masks"][i]);row["valid_g0"]=True;records.append(row)
        _,_,d,_,_,_,_,_=vectorize(out)
        x=d.double()
        for label,k in ranks.items():
            en=projection_energy(x,b,k)
            energy[label].append(en)
            error[label].append(float(((x.square().sum(1)-(x@b[:,:k]).square().sum(1)).clamp_min(0).sqrt()/x.norm(dim=1).clamp_min(EPS)).mean()))
        if (i+1)%100==0:print(json.dumps({"stage":"DEV_TRANSFER","done":i+1,"total":len(ids)}),flush=True)
    metric=summarize(records)
    historical=float(rt.state["validation_g0"]["mean_foreground_iou"])
    drift=abs(metric["mean_foreground_iou"]-historical)
    require(drift<=1e-5,f"historical selected DEV mean IoU replay drift: {drift}")
    require(tensor_state_sha256(rt.model.state_dict())==rt.hashes["utility_state"] and
            tensor_state_sha256(rt.rectifier.state_dict())==rt.hashes["rectifier_state"] and
            tensor_state_sha256(rt.sam.state_dict())==rt.hashes["sam"] and
            tensor_state_sha256(rt.source.state_dict())==rt.hashes["source"],"frozen model changed during DEV transfer")
    dump("dev_basis_transfer.json",{"train_frozen_K":ranks,"projection_energy_image_balanced":{key:describe(v) for key,v in energy.items()},
                                    "relative_error_image_balanced":{key:describe(v) for key,v in error.items()},
                                    "historical_selected_dev_metric":historical,"replayed_dev_metric":metric,
                                    "mean_iou_absolute_drift":drift,"status":"PASS"})
    print(json.dumps({"status":"DEV_TRANSFER_COMPLETE","mean_iou":metric["mean_foreground_iou"]}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("train","dev"));args=p.parse_args()
    torch.set_num_threads(8)
    rt=Runtime(torch.device("cuda:0"))
    if args.stage=="train":train(rt)
    else:dev(rt)


if __name__=="__main__":main()
