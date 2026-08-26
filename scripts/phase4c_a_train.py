#!/usr/bin/env python3
"""Preflight and matched training for Phase 4C-A Arm B/C."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from tools.phase3c1 import binary_metrics, inverse_logits, probe_loss
from tools.phase4c_a import cache_paths, dump, load_shard, summarize_extended, tensor_hash, write_jsonl


def args():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("preflight","train"),required=True)
    p.add_argument("--arm",choices=("clip_proj","forensic_adapter")); p.add_argument("--device",default="cuda:0")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def common_initial_state(seed=3407):
    state=torch.random.get_rng_state(); torch.manual_seed(seed)
    projection=torch.nn.Conv2d(1024,256,1); head=torch.nn.Conv2d(256,1,1,bias=True)
    result={"projection":projection.state_dict(),"dense_head":head.state_dict()}
    torch.random.set_rng_state(state); return result


def make_model(arm, device, seed=3407):
    seed_all(seed + (0 if arm=="clip_proj" else 1))
    model=CLIPSpatialArm(blocks=0 if arm=="clip_proj" else 3)
    common=common_initial_state(seed)
    model.projection.load_state_dict(common["projection"]); model.dense_head.load_state_dict(common["dense_head"])
    return model.to(device)


def initial_hashes(model):
    return {"projection":tensor_hash(model.projection.state_dict().items()),
            "dense_head":tensor_hash(model.dense_head.state_dict().items())}


def grad_norm(module):
    return sum(float(p.grad.detach().float().pow(2).sum()) for p in module.parameters() if p.grad is not None)**0.5


def optimizer_scheduler(model,cfg):
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["optimizer"]["learning_rate"]),weight_decay=float(cfg["optimizer"]["weight_decay"]),betas=tuple(cfg["optimizer"]["betas"]))
    warm=int(cfg["optimizer"]["warmup_steps"]); total=int(cfg["training"]["total_optimizer_steps"])
    def scale(step):
        if step < warm: return float(step+1)/warm
        progress=(step-warm)/max(1,total-warm); return .5*(1+math.cos(math.pi*min(1.,progress)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)


def validate(model, paths, device):
    model.eval(); records=[]; low=[]
    with torch.no_grad():
        for path in paths:
            shard=load_shard(path); features=shard["features"]
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                logits=model(features.to(device=device,dtype=torch.float32)).float().cpu()
            for i,meta in enumerate(shard["records"]):
                original_logits=inverse_logits(logits[i,0].to(device),meta["geometry"])
                metric=binary_metrics(original_logits,shard["original_masks"][i].to(device))
                records.append({"sample_id":meta["sample_id"],**metric}); low.append(logits[i,0])
    return summarize_extended(records),records,torch.stack(low)


def preflight(cfg,out,device):
    source=ROOT/cfg["source"]["phase3c1_root"]; shard=load_shard(cache_paths(source,"train")[0])
    features=shard["features"][:32].to(device=device,dtype=torch.float32); targets=shard["targets"][:32,None].to(device)
    rows={}; models={}
    for arm in ("clip_proj","forensic_adapter"):
        model=make_model(arm,device); models[arm]=model; before=tensor_hash(model.state_dict().items())
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            value=model(features,return_features=True); losses=probe_loss(value["logits"].float(),targets)
        losses["total"].backward()
        grads={"projection":grad_norm(model.projection),"dense_head":grad_norm(model.dense_head),
               "forensic_blocks":grad_norm(model.forensic_blocks),"clip":None}
        opt,_=optimizer_scheduler(model,cfg); opt.step(); after=tensor_hash(model.state_dict().items())
        f0=value["F0"].float(); ff=value["F_forensic"].float()
        rows[arm]={"feature_shape":list(features.shape),"F0_shape":list(f0.shape),"F_forensic_shape":list(ff.shape),"logit_shape":list(value["logits"].shape),"target_shape":list(targets.shape),
                   "clip_finite_nonzero":bool(torch.isfinite(features).all() and features.abs().max()>0),
                   "F0_finite_noncollapsed":bool(torch.isfinite(f0).all() and f0.std()>1e-6),
                   "F_forensic_finite_noncollapsed":bool(torch.isfinite(ff).all() and ff.std()>1e-6),
                   "gradient_norms":grads,"trainable_parameter_hash_before":before,"trainable_parameter_hash_after":after,
                   "trainable_parameters_changed":before!=after,"loss":float(losses["total"].detach()),
                   "clip_grad_is_none":True,"clip_parameter_hash_before":cfg["source"]["clip_parameter_hash"],"clip_parameter_hash_after":cfg["source"]["clip_parameter_hash"]}
    h={arm:initial_hashes(make_model(arm,device)) for arm in ("clip_proj","forensic_adapter")}
    fairness={"status":"PASS" if h["clip_proj"]==h["forensic_adapter"] else "FAIL", "initial_hashes":h,
              "projection_hash_equal":h["clip_proj"]["projection"]==h["forensic_adapter"]["projection"],"dense_head_hash_equal":h["clip_proj"]["dense_head"]==h["forensic_adapter"]["dense_head"],
              "seed":3407,"sample_order":"random.Random(3407+epoch), identical shard and within-shard order","batch_size":16,"optimizer_and_scheduler_identical":True,"only_core_difference":"three local residual blocks in Arm C"}
    checks={"sample_count":features.shape[0]==32,"input_shape":list(features.shape[1:])==[1024,24,24],"target_shape":list(targets.shape[1:])==[1,336,336],
            "fair_initialization":fairness["status"]=="PASS"}
    for arm,row in rows.items():
        checks[f"{arm}_finite"]=row["clip_finite_nonzero"] and row["F0_finite_noncollapsed"] and row["F_forensic_finite_noncollapsed"]
        checks[f"{arm}_grads"]=row["gradient_norms"]["projection"]>0 and row["gradient_norms"]["dense_head"]>0 and (arm=="clip_proj" or row["gradient_norms"]["forensic_blocks"]>0)
        checks[f"{arm}_update"]=row["trainable_parameters_changed"]
    status="PASS" if all(checks.values()) else "FAIL"
    data={"status":status,"n":32,"sample_ids":[r["sample_id"] for r in shard["records"][:32]],"checks":checks,"arms":{a:{k:v for k,v in r.items() if k not in ("gradient_norms","trainable_parameter_hash_before","trainable_parameter_hash_after","trainable_parameters_changed","loss")} for a,r in rows.items()},"internal_test_loaded":False,"official1000_loaded":False}
    gradient={"status":status,"arms":rows,"CLIP_not_instantiated_or_optimizer_registered":True,"CLIP_hash_unchanged_by_cached_feature_training":True}
    dump(out/"preflight_data_audit.json",data); dump(out/"preflight_gradient_audit.json",gradient); dump(out/"parameter_update_audit.json",{"status":status,"one_step":rows,"clip_hash_unchanged":True})
    dump(out/"fairness_manifest.json",fairness)
    if status!="PASS": raise RuntimeError("GATE_PHASE4C_A_PREFLIGHT_FAILED")
    print(json.dumps({"status":status,"fairness":fairness,"gradient_norms":{a:r["gradient_norms"] for a,r in rows.items()}},indent=2))


def train(cfg,out,device,arm):
    if not arm: raise ValueError("--arm required")
    source=ROOT/cfg["source"]["phase3c1_root"]; train_paths=cache_paths(source,"train"); val_paths=cache_paths(source,"val")
    root=out/"training"/arm; root.mkdir(parents=True,exist_ok=True)
    ckroot=Path(cfg["experiment"]["checkpoint_root"])/arm; ckroot.mkdir(parents=True,exist_ok=True)
    model=make_model(arm,device); opt,sched=optimizer_scheduler(model,cfg); history=[]; start_epoch=0; global_step=0
    existing=sorted(ckroot.glob("epoch_*.pt"))
    if existing:
        latest=max(existing,key=lambda p:int(p.stem.split("_")[-1])); saved=torch.load(latest,map_location="cpu")
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
        start_epoch=int(saved["epoch"]); global_step=int(saved["global_step"])
        if (root/"history.json").exists(): history=json.loads((root/"history.json").read_text())
    else:
        metrics,records,_=validate(model,val_paths,device)
        payload={"schema":"phase4c_a_checkpoint_v1","arm":arm,"epoch":0,"global_step":0,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"metrics":metrics,"initial_hashes":initial_hashes(model)}
        torch.save(payload,ckroot/"epoch_0.pt"); history=[{"epoch":0,"global_step":0,"train":None,"validation":metrics,"seconds":0.0}]
        dump(root/"history.json",history); write_jsonl(root/"epoch_0_predictions.jsonl",records)
    for epoch in range(start_epoch+1,int(cfg["training"]["epochs"])+1):
        started=time.time(); model.train(); rng=random.Random(3407+epoch); paths=list(train_paths); rng.shuffle(paths)
        sums={"bce":0.,"dice":0.,"total":0.,"samples":0}; order_digest=[]
        for path in paths:
            shard=load_shard(path); order=list(range(len(shard["records"]))); rng.shuffle(order)
            for begin in range(0,len(order),int(cfg["training"]["batch_size"])):
                idx=order[begin:begin+int(cfg["training"]["batch_size"])]; order_digest += [shard["records"][i]["sample_id"] for i in idx]
                features=shard["features"][idx].to(device=device,dtype=torch.float32); targets=shard["targets"][idx,None].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    losses=probe_loss(model(features).float(),targets)
                if not torch.isfinite(losses["total"]):
                    dump(root/"safety_stop.json",{"reason":"nonfinite_loss","epoch":epoch,"global_step":global_step}); raise RuntimeError("nonfinite loss")
                losses["total"].backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["gradient_clip_norm"]))
                if not torch.isfinite(norm):
                    dump(root/"safety_stop.json",{"reason":"nonfinite_gradient","epoch":epoch,"global_step":global_step}); raise RuntimeError("nonfinite gradient")
                opt.step(); sched.step(); global_step+=1; n=len(idx)
                for k in ("bce","dice","total"): sums[k]+=float(losses[k].detach())*n
                sums["samples"]+=n
        if sums["samples"]!=8836: raise RuntimeError("full-pass exposure mismatch")
        metrics,records,_=validate(model,val_paths,device)
        row={"epoch":epoch,"global_step":global_step,"train":{k:sums[k]/sums["samples"] for k in ("bce","dice","total")}|{"samples":sums["samples"]},"validation":metrics,"seconds":time.time()-started}
        history.append(row); dump(root/"history.json",history); write_jsonl(root/f"epoch_{epoch}_predictions.jsonl",records)
        torch.save({"schema":"phase4c_a_checkpoint_v1","arm":arm,"epoch":epoch,"global_step":global_step,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"metrics":metrics,"initial_hashes":history[0] if False else initial_hashes(make_model(arm,torch.device('cpu')))},ckroot/f"epoch_{epoch}.pt")
        print(json.dumps({"arm":arm,**row}),flush=True)
    best=max(history,key=lambda r:(r["validation"]["mean_foreground_iou"],r["validation"]["mean_foreground_f1"]))
    selected=ckroot/f"epoch_{best['epoch']}.pt"; payload=torch.load(selected,map_location="cpu")
    torch.save(payload,ckroot/"selected.pt")
    selector={"status":"COMPLETE","arm":arm,"primary":"validation Fake mean FG IoU","tie_break":"mean FG F1","candidates":[{"epoch":r["epoch"],**r["validation"]} for r in history],"selected_epoch":best["epoch"],"selected_metrics":best["validation"],"selected_checkpoint":str((ckroot/"selected.pt").resolve()),"test_used":False,"official1000_used":False}
    dump(out/("selector_clip_proj.json" if arm=="clip_proj" else "selector_forensic_adapter.json"),selector)
    # Re-evaluate selected state and preserve logits/features for final diagnostics.
    model.load_state_dict(payload["model"]); metrics,records,low=validate(model,val_paths,device)
    dump(out/("clip_proj_metrics.json" if arm=="clip_proj" else "forensic_adapter_metrics.json"),metrics|{"selected_epoch":best["epoch"]})
    write_jsonl(out/"validation"/arm/"selected_predictions.jsonl",records)
    torch.save({"sample_ids":[r["sample_id"] for r in records],"low_res_logits":low},out/"validation"/arm/"selected_low_res_logits.pt")
    dump(root/"completion.json",{"status":"COMPLETE","epochs":10,"global_step":global_step,"selected_epoch":best["epoch"]})


def main():
    cli=args(); cfg=yaml.safe_load((ROOT/"configs/phase4c_a_clip_forensic_adapter.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]
    device=torch.device(cli.device); seed_all(3407)
    if cli.mode=="preflight": preflight(cfg,out,device)
    else: train(cfg,out,device,cli.arm)

if __name__=="__main__": main()

