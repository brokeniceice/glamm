#!/usr/bin/env python3
"""Phase 6G.16: controlled Type-I versus Type-II correction-aware Utility audit."""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import phase6g10_train_arm as g10
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch, load_c1_cache, phase4f_spatial_batch
from scripts.phase6g13_utility_train import load_head, utility_batch
from scripts.phase6g15_residual_staged_optimization import configure
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid, gated_embedding
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import inverse_sam_logits, metric_record, file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import invalid_record, mask_loss

OUT = ROOT / "results/phase6g16_type2_utility"
DOC = ROOT / "docs/phase6g16_type2_correction_aware_utility.md"
G15 = ROOT / "outputs/phase6g15_residual_staged_optimization/arms/A2/selected_checkpoint.pt"
SEED, EPOCHS, BATCH, LR, WD, CLIP = 3407, 10, 8, 1e-4, 1e-4, 1.0
ARMS = ("A0_source_aware", "A1_type2_correction_aware")


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values))


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


class TypeIIUtility(nn.Module):
    """Minimal spatial G(S64,C,q), capacity-matched to the current CSCU Utility."""
    def __init__(self, width=91):
        super().__init__(); groups = 7
        self.s = nn.Sequential(nn.Conv2d(256,width,1),nn.GroupNorm(groups,width),nn.GELU())
        self.c = nn.Sequential(nn.Conv2d(256,width,1),nn.GroupNorm(groups,width),nn.GELU())
        self.q = nn.Sequential(nn.Linear(256,width),nn.LayerNorm(width),nn.GELU())
        self.net = nn.Sequential(
            nn.Conv2d(3*width,width,3,padding=1),nn.GroupNorm(groups,width),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(groups,width),nn.GELU(),
            nn.Conv2d(width,1,1))

    def forward(self, s64, correction, qseg, support):
        s=self.s(s64.detach().float()); c=self.c(correction.detach().float())
        qq=self.q(qseg.detach().float())[:,:,None,None].expand(-1,-1,64,64)
        logit=self.net(torch.cat((s,c,qq),1)); u=logit.sigmoid()*support.detach().float()
        return {"utility_logit":logit,"U":u,"support":support.detach()}


def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def state_hash(m): return tensor_state_sha256({k:v for k,v in m.state_dict().items()})


def load_collapsed(device):
    original, _, _, _ = configure("A2", device)
    payload=torch.load(G15,map_location="cpu",weights_only=False); original.load_state_dict(payload["model"],strict=True)
    original.eval().requires_grad_(False)
    collapsed=copy.deepcopy(original)
    anchored=collapsed.rectification.projection
    single=nn.Linear(256,256).to(device)
    with torch.no_grad():
        single.weight.copy_(anchored.base.weight+anchored.residual.weight)
        single.bias.copy_(anchored.base.bias+anchored.residual.bias)
    collapsed.rectification.projection=single
    collapsed.eval().requires_grad_(False)
    return original, collapsed, payload


def rect_forward(model,s64,f24,sc,cc,*,fp32=False):
    valid=torch.ones(len(s64),576,dtype=torch.bool,device=s64.device)
    with torch.no_grad(), torch.autocast(device_type=s64.device.type,dtype=torch.bfloat16,enabled=not fp32):
        out=model(s64.float() if fp32 else s64,f24.float() if fp32 else f24,sc,cc,valid)
    adapted=out["image_embeddings"].float(); c=adapted-s64.float()
    support=out["support"].reshape(len(s64),1,64,64).float()
    return out,adapted,c,support


def errors(a,b):
    d=a.float()-b.float(); return {"max_abs_error":float(d.abs().max()),"mean_abs_error":float(d.abs().mean()),
        "relative_L2_error":float(d.norm()/a.float().norm().clamp_min(1e-12))}


def collapse_audit(original,collapsed,sam,fusion,projection,p4,fs,cache,device):
    ids=p4.sample_ids[:2]; pos=torch.arange(2); s64,_,_,sc,cc=phase4f_spatial_batch(p4,ids,device)
    f24=g10.fused_evidence(fusion,projection,fs,ids,device)
    qseg=cache["q_seg"].index_select(0,pos).to(device=device,dtype=torch.bfloat16)
    captures={}
    def hook(name): return lambda mod,inp,out: captures.__setitem__(name,(inp[0].detach().float(),out.detach().float()))
    h0=original.rectification.projection.register_forward_hook(hook("original")); h1=collapsed.rectification.projection.register_forward_hook(hook("collapsed"))
    # The contract explicitly asks for algebraic FP32 collapse equivalence.
    oo,so,co,_=rect_forward(original,s64,f24,sc,cc,fp32=True); oc,scoll,ccorr,_=rect_forward(collapsed,s64,f24,sc,cc,fp32=True); h0.remove();h1.remove()
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
        lo=sam(qseg,so.to(torch.bfloat16)); lc=sam(qseg,scoll.to(torch.bfloat16))
    audit={"R2":errors(captures["original"][0],captures["collapsed"][0]),
           "raw_translator_output":errors(captures["original"][1],captures["collapsed"][1]),
           "actual_injected_correction":errors(co,ccorr),"S_rect":errors(so,scoll),"final_mask_logits":errors(lo,lc)}
    audit["precision"]="FP32 Rectifier; final frozen-SAM input cast identically to BF16"
    core=("R2","raw_translator_output","actual_injected_correction","S_rect")
    audit["tolerance"]={"core_relative_L2":2e-6,"final_mask_logits_relative_L2":5e-3,
                        "task_level":"full DEV mean IoU drift <= 5e-4"}
    audit["passed"]=all(audit[k]["relative_L2_error"]<=2e-6 for k in core) and audit["final_mask_logits"]["relative_L2_error"]<=5e-3
    return audit


def source_forward(model,batch,perm=None): return hc.utility_forward(model,batch,permutation=perm)


def make_context(data,cache,positions,p4,sam,s64,f24,zf):
    b,_=c1_language_batch(data,cache,positions,p4,sam,s64.device); b["S64"]=s64;b["F24"]=f24;b["z_F24"]=zf;return b


def gate_stats(values):
    x=np.asarray(values,dtype=np.float64); eps=1e-8
    return {"n":int(x.size),"mean":float(x.mean()),"median":float(np.median(x)),"std":float(x.std()),
            "min":float(x.min()),"max":float(x.max()),"fraction_lt_0.1":float((x<.1).mean()),
            "fraction_gt_0.9":float((x>.9).mean()),"binary_entropy_mean":float(np.mean(-(x*np.log(x+eps)+(1-x)*np.log(1-x+eps))))}


def step(model,arm,collapsed,sam,fusion,projection,head,p4,fs,data,ids,pos,cache,cross,perm,target_model,device):
    s64,_,targets,sc,cc=phase4f_spatial_batch(p4,ids,device); f=g10.fused_evidence(fusion,projection,fs,ids,device); z=head(f.float())
    batch=make_context(data,cache,pos,p4,sam,s64,f,z); crosspos=cross.index_select(0,pos); crossids=[data["sample_ids"][i] for i in crosspos.tolist()]
    fc=g10.fused_evidence(fusion,projection,fs,crossids,device); zc=head(fc.float()); crossed=dict(batch);crossed["F24"]=fc;crossed["z_F24"]=zc
    _,adapt,c,support=rect_forward(collapsed,s64,f,sc,cc)
    if arm.startswith("A0"):
        out=source_forward(model,batch); negc=source_forward(model,crossed); negs=source_forward(model,batch,perm)
    else:
        out=model(s64,c,batch["q_seg"],support)
        _,_,c_cross,_=rect_forward(collapsed,s64,fc,sc,cc)
        fp=f.flatten(2)[:,:,perm].reshape_as(f); _,_,c_shuffle,_=rect_forward(collapsed,s64,fp,sc,cc)
        negc=model(s64,c_cross,batch["q_seg"],support); negs=model(s64,c_shuffle,batch["q_seg"],support)
    gate=gate_to_sam_grid(out["U"],sc)*support; adapted=gated_embedding(s64,adapt,gate)
    with torch.autocast(device_type=device.type,enabled=False): low=sam(batch["q_seg"],adapted.to(torch.bfloat16))
    seg=mask_loss(low,targets,hd.CFG); target_out=source_forward(model if arm.startswith("A0") else target_model,batch)
    _,soft=q.target_delta(target_out,data["target64"].index_select(0,pos).to(device)); rel=q.image_balanced_loss(out["utility_logit"],soft,out["support"])
    cr=hc.rank_loss(out["U"],negc["U"],out["support"]); sr=hc.rank_loss(out["U"],negs["U"],out["support"]); rank=.5*(cr+sr)
    return seg["total"]+rel+rank,{"seg":seg["total"],"relative":rel,"ranking":rank,"cross":cr,"shuffle":sr}


def evaluate(model,arm,collapsed,sam,fusion,projection,head,p4,fs,dev,cache,device,no_utility=False,collect_gates=True):
    rec,gvals,gmeans=[],[],[]; model.eval() if model is not None else None
    with torch.no_grad():
        for i,sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]): rec.append(invalid_record(sid,dev["original_masks"][i]));gmeans.append(0.);continue
            s64,_,_,sc,cc=phase4f_spatial_batch(p4,[sid],device); f=g10.fused_evidence(fusion,projection,fs,[sid],device);z=head(f.float())
            b,_=c1_language_batch(dev,cache,torch.tensor([i]),p4,sam,device);b["S64"]=s64;b["F24"]=f;b["z_F24"]=z
            _,adapt,c,support=rect_forward(collapsed,s64,f,sc,cc)
            if no_utility: gate=support
            elif arm.startswith("A0"): gate=gate_to_sam_grid(source_forward(model,b)["U"],sc)*support
            else: gate=gate_to_sam_grid(model(s64,c,b["q_seg"],support)["U"],sc)*support
            final=gated_embedding(s64,adapt,gate)
            with torch.autocast(device_type=device.type,enabled=False): low=sam(b["q_seg"],final.to(torch.bfloat16))
            row=metric_record(sid,inverse_sam_logits(low,dev["sam_geometries"][i]),dev["original_masks"][i]);row["valid_g0"]=True;rec.append(row)
            vv=gate[support.bool()].float().cpu().numpy();gmeans.append(float(vv.mean()) if vv.size else 0.);gvals.extend(vv.tolist())
            if (i+1)%200==0: print(json.dumps({"stage":"VAL","arm":arm,"done":i+1,"total":len(dev["sample_ids"])}),flush=True)
    return summarize_extended(rec),rec,{"pixels":gate_stats(gvals) if gvals else None,"per_image":gate_stats(gmeans)}


def init_models(device):
    seed_all(); a0,_=hc.load_utility("a2",device);a0.train()
    for p in list(a0.language_source.parameters())+list(a0.forensic_source.parameters()):p.requires_grad_(False)
    seed_all();a1=TypeIIUtility().to(device)
    seed_all();target,_=hc.load_utility("a2",device);target.eval().requires_grad_(False)
    return a0,a1,target


def train_arm(arm,model,target,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device):
    root=OUT/arm
    if (root/"summary.json").exists(): return
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    ids=p4t.sample_ids;valid=tc["valid"].bool();where={s:i for i,s in enumerate(ids)};cross=torch.tensor([(i+1)%len(ids) for i in range(len(ids))]);perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device)
    hist=[];start=1;updates=0
    ckpts=sorted((root/"checkpoints").glob("epoch_*.pt")) if (root/"checkpoints").exists() else []
    if ckpts:
        last=max(ckpts,key=lambda p:int(p.stem.split('_')[-1]));st=torch.load(last,map_location="cpu",weights_only=False);model.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"]);start=st["epoch"]+1;updates=st["updates"];hist=json.loads((root/"training_curve.json").read_text())
    for epoch in range(start,EPOCHS+1):
        began=time.time();model.train();order=list(ids);random.Random(SEED+1009*epoch).shuffle(order);sums=defaultdict(float);n=0
        for begin in range(0,len(order),BATCH):
            block=order[begin:begin+BATCH];pos=torch.tensor([where[x] for x in block]);pos=pos[valid.index_select(0,pos)]
            if not len(pos):continue
            bid=[ids[i] for i in pos.tolist()];opt.zero_grad(set_to_none=True);loss,parts=step(model,arm,collapsed,sam,fusion,projection,head,p4t,fst,data,bid,pos,tc,cross,perm,target,device);loss.backward();torch.nn.utils.clip_grad_norm_(params,CLIP);opt.step();updates+=1;n+=len(bid)
            for k,v in parts.items():sums[k]+=float(v.detach())*len(bid)
        met,recs,gates=evaluate(model,arm,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device)
        row={"epoch":epoch,"optimizer_updates":updates,"seg_loss":sums["seg"]/n,"relative_loss":sums["relative"]/n,"ranking_loss":sums["ranking"]/n,"total_loss":sum(sums[k] for k in ("seg","relative","ranking"))/n,"dev_g0_mean_iou":met["mean_foreground_iou"],"dev_g0_mean_f1":met["mean_foreground_f1"],"seconds":time.time()-began};hist.append(row);dump(root/"training_curve.json",hist);write_rows(root/f"validation/epoch_{epoch}.jsonl",recs);dump(root/f"gate/epoch_{epoch}.json",gates)
        (root/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"updates":updates,"model":model.state_dict(),"optimizer":opt.state_dict()},root/f"checkpoints/epoch_{epoch}.pt");print(json.dumps({"stage":"G16_EPOCH","arm":arm,**row}),flush=True)
    best=max(hist,key=lambda x:(x["dev_g0_mean_iou"],x["dev_g0_mean_f1"],-x["epoch"]));sel=root/"selected_checkpoint.pt";sel.write_bytes((root/f"checkpoints/epoch_{best['epoch']}.pt").read_bytes());model.load_state_dict(torch.load(sel,map_location="cpu",weights_only=False)["model"])
    met,recs,gates=evaluate(model,arm,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device);write_rows(root/"validation/selected.jsonl",recs);dump(root/"gate/selected.json",gates);dump(root/"summary.json",{"status":"COMPLETE","selected_epoch":best["epoch"],"metrics":met,"gate_stats":gates,"trainable_params":count_trainable(model)})


def bootstrap_mean(x,seed=3407,n=10000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);means=np.empty(n)
    for i in range(n):means[i]=x[rng.integers(0,len(x),len(x))].mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def diagnostics(model,collapsed,sam,fusion,projection,head,p4,fs,dev,cache,device):
    model.eval();permch=torch.randperm(256,generator=torch.Generator().manual_seed(SEED)).to(device);permsp=torch.randperm(4096,generator=torch.Generator().manual_seed(SEED+1)).to(device)
    vals={k:[] for k in ("identity_repeat","cross_image","channel_shuffle","spatial_shuffle","direction_flip")};signed={k:[] for k in vals if k!="identity_repeat"};unorm=[];cnorm=[];image_u=[];image_c=[];per=[]
    validids=[i for i,v in enumerate(cache["valid"]) if bool(v)];cycle={i:validids[(j+1)%len(validids)] for j,i in enumerate(validids)}
    with torch.no_grad():
        for i in validids:
            sid=dev["sample_ids"][i];s64,_,_,sc,cc=phase4f_spatial_batch(p4,[sid],device);f=g10.fused_evidence(fusion,projection,fs,[sid],device);z=head(f.float());b,_=c1_language_batch(dev,cache,torch.tensor([i]),p4,sam,device);_,_,c,support=rect_forward(collapsed,s64,f,sc,cc)
            j=cycle[i];fj=g10.fused_evidence(fusion,projection,fs,[dev["sample_ids"][j]],device);_,_,cx,_=rect_forward(collapsed,s64,fj,sc,cc)
            cch=c[:,permch];flat=c.flatten(2);csp=flat[:,:,permsp].reshape_as(c);csp=csp*support
            variants={"cross_image":cx,"channel_shuffle":cch,"spatial_shuffle":csp,"direction_flip":-c}
            ur=model(s64,c,b["q_seg"],support)["U"];ur2=model(s64,c,b["q_seg"],support)["U"]
            mask=support.bool();base=ur[mask].float();vals["identity_repeat"].extend((base-ur2[mask].float()).abs().cpu().tolist());local={"sample_id":sid,"mean_U":float(base.mean()),"global_C_norm":float(c.norm())}
            localnorm=c.float().square().sum(1,keepdim=True).sqrt()[mask];unorm.extend(base.cpu().tolist());cnorm.extend(localnorm.cpu().tolist());image_u.append(float(base.mean()));image_c.append(float(c.norm()))
            for name,cv in variants.items():
                uv=model(s64,cv,b["q_seg"],support)["U"][mask].float();delta=uv-base;vals[name].extend(delta.abs().cpu().tolist());signed[name].extend(delta.cpu().tolist());local[name+"_mean_abs_dU"]=float(delta.abs().mean());local[name+"_signed_dU"]=float(delta.mean())
            per.append(local)
    out={"identity_sensitivity":{},"norm_shortcut":{},"per_sample":per}
    for k,x in vals.items():out["identity_sensitivity"][k]={"mean_abs_dU":float(np.mean(x)),"median_abs_dU":float(np.median(x)),"bootstrap_95_ci":bootstrap_mean(x),**({} if k=="identity_repeat" else {"signed_dU":float(np.mean(signed[k]))})}
    def corr(a,b):return {"pearson":float(stats.pearsonr(a,b).statistic),"spearman":float(stats.spearmanr(a,b).statistic)}
    out["norm_shortcut"]={"spatial":corr(unorm,cnorm),"per_image":corr(image_u,image_c)}
    return out


def finalize(device):
    a0r=rows(OUT/"A0_source_aware/validation/selected.jsonl");a1r=rows(OUT/"A1_type2_correction_aware/validation/selected.jsonl");b0r=rows(OUT/"B0/validation.jsonl")
    def pair(x,y):
        p=paired_statistics(torch.tensor([r["foreground_iou"] for r in x]),torch.tensor([r["foreground_iou"] for r in y]));d=np.asarray([r["foreground_iou"] for r in x])-np.asarray([r["foreground_iou"] for r in y]);p["wins_ties_losses"]=[int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d<-1e-12).sum())];p["wilcoxon"]=stats.wilcoxon(d).pvalue if np.any(d) else 1.;return p
    paired={"A1_minus_A0":pair(a1r,a0r),"A0_minus_B0":pair(a0r,b0r),"A1_minus_B0":pair(a1r,b0r)};dump(OUT/"paired_stats.json",paired)
    s0=json.loads((OUT/"A0_source_aware/summary.json").read_text());s1=json.loads((OUT/"A1_type2_correction_aware/summary.json").read_text());b0=json.loads((OUT/"B0/summary.json").read_text());sens=json.loads((OUT/"diagnostics/correction_sensitivity.json").read_text())
    primary=paired["A1_minus_A0"]; identity=max(sens["identity_sensitivity"][k]["mean_abs_dU"] for k in ("channel_shuffle","spatial_shuffle","direction_flip"))>10*max(1e-12,sens["identity_sensitivity"]["identity_repeat"]["mean_abs_dU"])
    if primary["mean_difference"]>0 and primary["bootstrap_95_ci"][0]>0 and identity:decision="TYPE2_CORRECTION_AWARE_UTILITY_SUPPORTED"
    elif primary["mean_difference"]>0 and primary["bootstrap_95_ci"][0]>0:decision="TYPE2_TASK_GAIN_BUT_CORRECTION_IDENTITY_NOT_ESTABLISHED"
    elif primary["bootstrap_95_ci"][1]<0:decision="SOURCE_CONTEXT_REMAINS_NECESSARY_UNDER_TYPE2_TEST"
    else:decision="TYPE2_NOT_STABLY_BETTER"
    result={"status":"COMPLETE_STOP","decision":decision,"B0":b0,"A0":s0,"A1":s1,"paired":paired,"correction_sensitivity":sens,"firewall":{"nonlinear_translator":False,"type3":False,"objective_redesign":False,"internal_test":False,"official1000":False,"ood":False}};dump(OUT/"results.json",result)
    lines=["# Phase 6G.16 — Type-II Correction-Aware Utility Audit","","Status: **COMPLETE STOP**.","","## 1. Scientific question","","Does direct observation of `(S64,C,q_seg)` improve Utility under a fixed collapsed G15-A2 Translator?","","## 2. Why G15 permits single-path Translator","",f"Collapse audit: `{json.loads((OUT/'collapse_equivalence.json').read_text())}`","","## 3. Fixed Translator and ownership","",f"`{json.loads((OUT/'ownership.json').read_text())}`","","## 4. Exact definition of C","","`C` is exactly the tensor satisfying `S_adapt = S64 + U*C`, computed as collapsed Rectifier output minus `S64`; it retains existing gamma and hard support semantics.","","## 5. Utility arms","",f"B0: `{b0['metrics']}`",f"A0 Type-I: `{s0['metrics']}`",f"A1 Type-II: `{s1['metrics']}`","","## 6. Capacity matching","",f"A0={s0['trainable_params']}; A1={s1['trainable_params']}; relative difference={(s1['trainable_params']-s0['trainable_params'])/s0['trainable_params']:.6%}.","","## 7. Initialization control","",f"`{json.loads((OUT/'initialization.json').read_text())}`","","## 8. Loss equivalence","","Both arms use the unchanged Phase6G.13 `seg + relative + ranking` objective with unit weights.","","## 9. Ranking-negative propagation","","A1 preserves source negatives but maps `F_positive/F_cross/F_shuffle` through the same frozen Interaction and Translator before Utility sees `C_positive/C_cross/C_shuffle`.","","## 10. DEV results","",f"`{json.dumps({'B0':b0['metrics'],'A0':s0['metrics'],'A1':s1['metrics']},ensure_ascii=False)}`","","## 11. Paired statistics","",f"`{paired}`","","## 12. Gate statistics","",f"A0: `{s0['gate_stats']}`",f"A1: `{s1['gate_stats']}`","","## 13–17. Correction identity and norm-shortcut diagnostics","",f"`{sens}`","","## 18. Representative sample analysis","","See `diagnostics/correction_sensitivity.json` per-sample statistics; selection is statistical, not anecdotal.","","## 19. Supported interpretation","",decision,"","## 20. Unsupported interpretation","","This audit does not show that the current Utility was wrong, that Type-III is warranted, or that Utility objectively understands correctness beyond the reported corruption sensitivities.","","## 21. Decision","",f"```text\n{decision}\n```","","No internal test, Official1000, OOD, Translator redesign, Type-III arm, or objective redesign was used."]
    DOC.write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:1");a=ap.parse_args();device=torch.device(a.device);torch.cuda.set_device(device);seed_all()
    if (OUT/"results.json").exists():print(json.dumps({"status":"ALREADY_COMPLETE"}));return
    original,collapsed,payload=load_collapsed(device);fusion=g10.load_fusion(device);projection=g10.load_projection(device);head=load_head(device);sam=g10.load_sam_runtime(hd.CFG,device)
    p4t,p4v=g10.Phase4FStore(hd.CFG,"train"),g10.Phase4FStore(hd.CFG,"val");fst,fsv=g10.Store("train"),g10.Store("val");dev=g10.load_dev("g0");tc=load_c1_cache("train",p4t.sample_ids);vc=load_c1_cache("val",dev["sample_ids"])
    collapse=collapse_audit(original,collapsed,sam,fusion,projection,p4t,fst,tc,device);dump(OUT/"collapse_equivalence.json",collapse)
    if not collapse["passed"]:raise RuntimeError(f"collapse parity failed: {collapse}")
    a0,a1,target=init_models(device);data=q.load_ids(p4t.sample_ids,("valid_g0","S64","q_seg","z_L","F24","z_F24","target64","clip_geometries"));data["sample_ids"]=p4t.sample_ids
    dump(OUT/"ownership.json",{"frozen":["C1","CLIP","fusion","projection","Interaction","Translator","gamma","SAM","q_seg producer"],"deterministic":"hard geometry support","trainable_only":"Utility","translator_sha256":state_hash(collapsed),"g15_checkpoint":str(G15),"g15_sha256":file_sha256(G15)})
    # Step-0 gates on the same canonical first batch.
    ids=p4t.sample_ids[:BATCH];pos=torch.arange(BATCH);s64,_,_,sc,cc=phase4f_spatial_batch(p4t,ids,device);f=g10.fused_evidence(fusion,projection,fst,ids,device);z=head(f.float());b=make_context(data,tc,pos,p4t,sam,s64,f,z);_,_,c,support=rect_forward(collapsed,s64,f,sc,cc)
    with torch.no_grad():u0=source_forward(a0,b)["U"][support.bool()].float().cpu().numpy();u1=a1(s64,c,b["q_seg"],support)["U"][support.bool()].float().cpu().numpy()
    dump(OUT/"initialization.json",{"seed":SEED,"family":"PyTorch default Linear/Conv with matched final sigmoid head policy","A0":{"params":count_trainable(a0),"state_sha256":state_hash(a0),"step0_gate":gate_stats(u0)},"A1":{"params":count_trainable(a1),"state_sha256":state_hash(a1),"step0_gate":gate_stats(u1)},"relative_param_difference":(count_trainable(a1)-count_trainable(a0))/count_trainable(a0)})
    # B0 no-Utility reference and collapse metric gate.
    if not (OUT/"B0/summary.json").exists():
        met,recs,_=evaluate(None,"B0",collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device,no_utility=True);write_rows(OUT/"B0/validation.jsonl",recs);dump(OUT/"B0/summary.json",{"metrics":met,"expected_G15_A2_mean_iou":.18496412512767835,"absolute_drift":abs(met["mean_foreground_iou"]-.18496412512767835)})
        # BF16 executes one affine instead of two separately rounded affines;
        # metric parity is approximate while the algebraic FP32 gate above is strict.
        if abs(met["mean_foreground_iou"]-.18496412512767835)>5e-4:raise RuntimeError("collapsed DEV metric does not reproduce G15 A2")
    train_arm(ARMS[0],a0,target,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device)
    train_arm(ARMS[1],a1,target,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device)
    a1.load_state_dict(torch.load(OUT/"A1_type2_correction_aware/selected_checkpoint.pt",map_location="cpu",weights_only=False)["model"])
    sens=diagnostics(a1,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device);dump(OUT/"diagnostics/correction_sensitivity.json",sens);dump(OUT/"correction_sensitivity.json",sens)
    dump(OUT/"gate_stats.json",{"A0":json.loads((OUT/"A0_source_aware/summary.json").read_text())["gate_stats"],"A1":json.loads((OUT/"A1_type2_correction_aware/summary.json").read_text())["gate_stats"]})
    finalize(device);print(json.dumps({"status":"COMPLETE_STOP"}),flush=True)

if __name__=="__main__":main()
