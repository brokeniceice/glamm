#!/usr/bin/env python3
"""Conditionally triggered matched 3000-step P1 SFT continuation without FEPN tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from scripts.phase4b_train import (append,boundary,cpu_batch,group_hash,load_stack,parameter_group,seed_all)
from scripts.phase1b_preflight import move_batch
from tools.distributed_loss import batch_supervision_counts
from tools.phase4b import dump,file_sha256


def save(root,step,model,source,optimizer,scheduler,cfg,record):
    destination=Path(root)/"matched_sft"/f"step_{step:04d}"/"checkpoint"; destination.mkdir(parents=True,exist_ok=True)
    module=dict(source["module"]); live=dict(model.named_parameters())
    for name in list(module):
        if name in live and parameter_group(name)=="lora": module[name]=live[name].detach().cpu().clone()
    path=destination/"mp_rank_00_model_states.pt"; torch.save({"module":module,"optimizer":optimizer.state_dict(),
      "lr_scheduler":scheduler.state_dict(),"optimizer_step":step,"epoch":step//500,
      "client_state":{"phase":"Phase4B-G","arm":"MATCHED-SFT","source_checkpoint_sha256":cfg["source"]["checkpoint_sha256"],"training_record":record}},path)
    dump(destination.parent/"metadata.json",{"optimizer_step":step,"checkpoint":str(path),"sha256":file_sha256(path)}); return path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--physical-gpu",type=int,default=0); cli=parser.parse_args()
    if cli.physical_gpu!=0 or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"","0"): raise RuntimeError("matched SFT uses physical GPU0")
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]; route=json.loads((out/"route_gate.json").read_text())
    if not route.get("matched_sft_triggered"): raise RuntimeError("matched SFT trigger is false")
    device=torch.device("cuda:0"); torch.cuda.set_device(device); seed=int(cfg["experiment"]["seed"]); seed_all(seed)
    model,unused_projector,tokenizer,args,source,counts,missing=load_stack(cfg,"PROJ-LORA",device); del unused_projector
    model.forensic_evidence_token_count=0; boundary(model,"PROJ-LORA"); model.train()
    schedule=json.loads((out/"training/schedule.json").read_text()); dataset=UnifiedForensicsDataset(ROOT/cfg["data"]["manifest_dir"],tokenizer,args.vision_tower,split="train",
      datasets_root=cfg["data"]["datasets_root"],synthscars_root=cfg["data"]["synthscars_root"],image_size=args.image_size,target_protocol="phrase_aligned")
    params=[p for n,p in model.named_parameters() if p.requires_grad and parameter_group(n)=="lora"]
    optimizer=torch.optim.AdamW([{"name":"lora","params":params,"lr":float(cfg["optimizer"]["lora_lr"])}],betas=tuple(map(float,cfg["optimizer"]["betas"])),weight_decay=float(cfg["optimizer"]["weight_decay"]))
    total=3000; warmup=150
    def lr(completed): return float(completed+1)/warmup if completed<warmup else max(0.,float(total-completed)/(total-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr); checkpoint_steps=set(map(int,cfg["training"]["checkpoint_steps"])); metrics=out/"matched_sft/training_metrics.jsonl"
    if metrics.exists(): raise RuntimeError("matched SFT metrics already exist; refusing overwrite")
    initial={"lora":group_hash(model,"lora"),"frozen":group_hash(model,"frozen")}; save(cfg["experiment"]["checkpoint_root"],0,model,source,optimizer,scheduler,cfg,{"initial":True})
    started=time.time()
    for zero in range(total):
        optimizer.zero_grad(set_to_none=True); rows=[]; local=[]; step_started=time.time()
        for micro in range(4):
            exposure=zero*4+micro; batch,sample=cpu_batch(dataset,int(schedule["indices"][exposure]),tokenizer,inference=False)
            rows.append((exposure,batch,sample)); local.append(batch_supervision_counts(batch))
        total_text=sum(x["text_tokens"] for x in local); ce=0.; ids=[]
        for (exposure,batch,sample),count in zip(rows,local):
            trajectory_seed=seed*1000003+exposure; torch.manual_seed(trajectory_seed); torch.cuda.manual_seed_all(trajectory_seed)
            batch=move_batch(batch,device,torch.bfloat16); output=model(**batch); language=output["ce_loss"]; scale=count["text_tokens"]/total_text
            (language*scale).backward(); ce+=float(language.detach().float())*scale; ids.append(sample["sample_id"])
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.0)); optimizer.step(); scheduler.step(); step=zero+1
        record={"optimizer_step":step,"language_ce":ce,"gradient_norm_before_clip":grad,"learning_rate":float(optimizer.param_groups[0]["lr"]),
                "sample_ids":ids,"seconds_this_optimizer_step":time.time()-step_started}; append(metrics,record)
        if step<=5 or step%25==0: print(json.dumps(record),flush=True)
        if step in checkpoint_steps: save(cfg["experiment"]["checkpoint_root"],step,model,source,optimizer,scheduler,cfg,record)
    final={"lora":group_hash(model,"lora"),"frozen":group_hash(model,"frozen")}; changed={k:initial[k]!=final[k] for k in initial}
    if changed!={"lora":True,"frozen":False}: raise RuntimeError(f"matched boundary failed {changed}")
    dump(out/"matched_sft/run_summary.json",{"status":"COMPLETE","final_step":3000,"elapsed_seconds":time.time()-started,
      "same_schedule_sha256":schedule["sha256"],"no_fepn_tokens":True,"changed":changed})


if __name__=="__main__": main()
