#!/usr/bin/env python3
"""Phase 6G.20: explicit local Evidence->Correction attention versus G19 merge."""
from __future__ import annotations

import json, math, os, random, shutil, sys, time
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
from scripts import phase6g19_r1_c_vs_srect as g19
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import BATCH, CLIP, EPOCHS, LR, SEED, WD, load_collapsed, rows, seed_all, write_rows
from tools.phase3c1 import paired_statistics
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256

OUT = ROOT / "outputs/phase6g20_evidence_correction_interaction"
DOC = ROOT / "docs/phase6g20_evidence_correction_interaction.md"
G19 = ROOT / "outputs/phase6g19_r1_c_vs_srect/A1_C"


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); os.replace(tmp, path)


class LocalEvidenceCorrectionAttention(nn.Module):
    """The existing CSCU local-attention family, one-way E(query)->C(key/value)."""
    def __init__(self, width=64, heads=4, window=7):
        super().__init__()
        if width % heads or window % 2 != 1: raise ValueError("invalid local attention geometry")
        self.width, self.heads, self.window, self.head_dim = width, heads, window, width // heads
        self.q = nn.Conv2d(width, width, 1); self.k = nn.Conv2d(width, width, 1); self.v = nn.Conv2d(width, width, 1)
        self.out = nn.Conv2d(width, width, 1); self.norm = nn.GroupNorm(8, width)

    def _attend(self, query, key, value, support):
        batch, _, height, width = query.shape; cells = height * width; neighbors = self.window ** 2
        qq = query.view(batch, self.heads, self.head_dim, cells).permute(0, 1, 3, 2)
        kk = F.unfold(key, self.window, padding=self.window//2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        vv = F.unfold(value, self.window, padding=self.window//2).view(batch, self.heads, self.head_dim, neighbors, cells).permute(0, 1, 4, 3, 2)
        score = (qq.unsqueeze(-2) * kk).sum(-1) / math.sqrt(self.head_dim)
        mask = F.unfold(support.float(), self.window, padding=self.window//2).view(batch, 1, cells, neighbors)
        score = score.masked_fill(mask == 0, -1e4); attention = score.softmax(-1) * mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        output = (attention.unsqueeze(-1) * vv).sum(-2).permute(0, 1, 3, 2).reshape(batch, self.width, height, width)
        return output, attention

    def forward(self, evidence, correction, support):
        update, attention = self._attend(self.q(evidence), self.k(correction), self.v(correction), support)
        return self.norm(evidence + self.out(update)) * support.float(), attention


class CrossAttentionForensicContext64(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct exactly the G19 evidence/X encoders, replacing only concat/Conv3x3 merge.
        base = g19.StructuredForensicContext64()
        self.evidence = base.evidence
        self.x = base.x
        self.interaction = LocalEvidenceCorrectionAttention(width=64, heads=4, window=7)
        self.last_attention = None

    def forward(self, f64, z_f64, support64, correction):
        support = support64.detach().float()
        evidence = self.evidence(f64, z_f64, support)
        c64 = self.x(correction.detach().float()) * support
        output, attention = self.interaction(evidence, c64, support)
        self.last_attention = attention.detach()
        return output


def init_model(device):
    # g19 init consumes the exact common RNG path; replacing only its merge is deterministic.
    model = g19.init_model(device)
    seed_all()
    replacement = CrossAttentionForensicContext64().to(device)
    # Reuse exact shared G19 initialization rather than reinitializing evidence/X.
    replacement.evidence.load_state_dict(model.forensic_context.evidence.state_dict())
    replacement.x.load_state_dict(model.forensic_context.x.state_dict())
    model.forensic_context = replacement
    return model


def shared_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith("forensic_context.merge.") and not k.startswith("forensic_context.interaction.")}


def trainable_state(model):
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    return {n:v for n,v in model.state_dict().items() if n in names}


def train(device):
    root = OUT / "A1_cross_attention"
    if (root / "summary.json").exists(): print(json.dumps({"status":"ALREADY_COMPLETE"})); return
    if not (G19 / "summary.json").exists(): raise RuntimeError("G19 A1_C is not complete")
    model = init_model(device); a0_init = g19.init_model(torch.device("cpu"))
    shared_a1 = shared_state(model); shared_a0 = shared_state(a0_init)
    # Remove device from comparison and verify all common tensors bit-exact.
    shared_exact = set(shared_a1) == set(shared_a0) and all(torch.equal(shared_a1[k].cpu(), shared_a0[k].cpu()) for k in shared_a0)
    if not shared_exact: raise RuntimeError("shared initialization drift from G19 A0")
    _, collapsed, _ = load_collapsed(device); fusion=g10.load_fusion(device); projection=g10.load_projection(device); head=load_head(device)
    sam=g10.load_sam_runtime(hd.CFG,device); p4t,p4v=g10.Phase4FStore(hd.CFG,"train"),g10.Phase4FStore(hd.CFG,"val")
    fst,fsv=g10.Store("train"),g10.Store("val"); dev=g10.load_dev("g0"); ids=p4t.sample_ids
    tc=load_c1_cache("train",ids);vc=load_c1_cache("val",dev["sample_ids"]);valid=tc["valid"].bool();where={s:i for i,s in enumerate(ids)}
    data=q.load_ids(ids,("valid_g0","S64","q_seg","z_L","F24","z_F24","target64","clip_geometries"));data["sample_ids"]=ids
    cross=torch.tensor([(i+1)%len(ids) for i in range(len(ids))]);perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    frozen={"collapsed":tensor_state_sha256(collapsed.state_dict()),"sam":tensor_state_sha256(sam.state_dict()),"fusion":tensor_state_sha256(fusion.state_dict()),
            "projection":tensor_state_sha256(projection.state_dict()),"head":tensor_state_sha256(head.state_dict()),"source_heads":tensor_state_sha256(q.source_state(model))}
    root.mkdir(parents=True,exist_ok=True)
    dump(root/"protocol.json",{"status":"FROZEN_BEFORE_FIRST_STEP","A0":"exact reuse G19 A1_C selected result","A1":"E64 queries C64 with heads4/window7 local attention",
         "shared_initialization_exact":shared_exact,"trainable_parameters":sum(p.numel() for p in params),"A0_trainable_parameters":json.load(open(G19/"summary.json"))["trainable_parameters"],
         "recipe":{"epochs":EPOCHS,"batch":BATCH,"lr":LR,"weight_decay":WD,"seed":SEED,"loss":"seg+relative+ranking","selector":"internal validation G0 mean IoU; tie earlier"},
         "firewall":{"srect":False,"zero_x":False,"film":False,"late_logit":False,"new_loss":False,"test":False,"official1000":False,"ood":False}})
    history=[];updates=0;start=1;checkpoints=sorted((root/"checkpoints").glob("epoch_*.pt")) if (root/"checkpoints").exists() else []
    if checkpoints:
        last=max(checkpoints,key=lambda p:int(p.stem.split("_")[-1]));state=torch.load(last,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);start=state["epoch"]+1;updates=state["updates"];history=json.load(open(root/"training_curve.json"))
    for epoch in range(start,EPOCHS+1):
        began=time.time();model.train();model.language_source.eval();model.forensic_source.eval();order=list(ids);random.Random(SEED+1009*epoch).shuffle(order);sums=defaultdict(float);n=0
        for begin in range(0,len(order),BATCH):
            pos=torch.tensor([where[x] for x in order[begin:begin+BATCH]]);pos=pos[valid.index_select(0,pos)]
            if not len(pos):continue
            bids=[ids[i] for i in pos.tolist()];opt.zero_grad(set_to_none=True)
            # Force X=C; g19's train step then keeps final injection C for every arm.
            loss,parts=g19.train_step(model,"A1_C",collapsed,sam,fusion,projection,head,p4t,fst,data,bids,pos,tc,cross,perm,device)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(params,CLIP);opt.step();updates+=1;n+=len(bids)
            for k,v in parts.items():sums[k]+=float(v.detach())*len(bids)
        met,recs,gates=g19.evaluate(model,"A1_C",collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device)
        row={"epoch":epoch,"updates":updates,"seg_loss":sums["seg"]/n,"relative_loss":sums["relative"]/n,"ranking_loss":sums["ranking"]/n,
             "total_loss":sum(sums.values())/n,"dev_g0_mean_iou":met["mean_foreground_iou"],"dev_g0_mean_f1":met["mean_foreground_f1"],"seconds":time.time()-began}
        history.append(row);dump(root/"training_curve.json",history);write_rows(root/f"validation/epoch_{epoch}.jsonl",recs);dump(root/f"gate/epoch_{epoch}.json",gates)
        (root/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"updates":updates,"model":model.state_dict(),"optimizer":opt.state_dict()},root/f"checkpoints/epoch_{epoch}.pt")
        print(json.dumps({"stage":"G20_EPOCH","arm":"A1_cross_attention",**row}),flush=True)
    best=max(history,key=lambda x:(x["dev_g0_mean_iou"],-x["epoch"]));selected=root/"selected_checkpoint.pt";shutil.copy2(root/f"checkpoints/epoch_{best['epoch']}.pt",selected)
    model.load_state_dict(torch.load(selected,map_location="cpu",weights_only=False)["model"]);met,recs,gates=g19.evaluate(model,"A1_C",collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device)
    write_rows(root/"validation/selected.jsonl",recs)
    after={"collapsed":tensor_state_sha256(collapsed.state_dict()),"sam":tensor_state_sha256(sam.state_dict()),"fusion":tensor_state_sha256(fusion.state_dict()),
           "projection":tensor_state_sha256(projection.state_dict()),"head":tensor_state_sha256(head.state_dict()),"source_heads":tensor_state_sha256(q.source_state(model))}
    if after!=frozen:raise RuntimeError("frozen hash drift")
    dump(root/"summary.json",{"status":"COMPLETE","selected_epoch":best["epoch"],"selected_checkpoint_sha256":file_sha256(selected),"metrics":met,"gate_stats":gates,
         "trainable_parameters":sum(p.numel() for p in params),"shared_initialization_exact":shared_exact})


def pair(a1,a0):
    if [x["sample_id"] for x in a1]!=[x["sample_id"] for x in a0]:raise RuntimeError("sample order drift")
    x=torch.tensor([r["foreground_iou"] for r in a1]);y=torch.tensor([r["foreground_iou"] for r in a0]);out=paired_statistics(x,y);d=(x-y).numpy()
    out["wins_ties_losses"]=[int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d<-1e-12).sum())];out["wilcoxon_pvalue"]=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.
    out["f1"]=paired_statistics(torch.tensor([r["foreground_f1"] for r in a1]),torch.tensor([r["foreground_f1"] for r in a0]));return out


def finalize():
    a0=json.load(open(G19/"summary.json"));a1=json.load(open(OUT/"A1_cross_attention/summary.json"));r0=rows(G19/"validation/selected.jsonl");r1=rows(OUT/"A1_cross_attention/validation/selected.jsonl")
    paired=pair(r1,r0);decision="EXPLICIT_EVIDENCE_CORRECTION_INTERACTION_SUPPORTED" if paired["mean_difference"]>0 and paired["bootstrap_95_ci"][0]>0 else "SIMPLE_STRUCTURED_MERGE_PREFERRED"
    result={"status":"COMPLETE_STOP","decision":decision,"A0_G19_C_merge":a0,"A1_cross_attention":a1,"paired_A1_minus_A0":paired,
            "firewall":{"srect":False,"zero_x":False,"test":False,"official1000":False,"ood":False}}
    dump(OUT/"results.json",result)
    DOC.write_text("\n".join(["# Phase 6G.20 — Evidence–Correction Interaction Audit","","Status: **COMPLETE STOP**.","","## Arms","",
      "A0 is exact reuse of G19 A1_C. A1 replaces only concat/3x3 merge with heads=4, window=7 local Evidence-query/Correction-key-value attention.","",
      f"A0: `{json.dumps(a0,ensure_ascii=False)}`","",f"A1: `{json.dumps(a1,ensure_ascii=False)}`","","## Paired A1-A0","",f"`{json.dumps(paired,ensure_ascii=False)}`","","## Decision","",f"```text\n{decision}\n```",
      "","No Srect, zero-X, FiLM, late-logit residual, third arm, new loss, test, Official1000, or OOD was used."])+"\n")
    print(json.dumps({"status":"COMPLETE_STOP","decision":decision}),flush=True)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:1");p.add_argument("--finalize",action="store_true");a=p.parse_args()
    if a.finalize:finalize();return
    d=torch.device(a.device);torch.cuda.set_device(d);train(d);finalize()


if __name__=="__main__":main()
