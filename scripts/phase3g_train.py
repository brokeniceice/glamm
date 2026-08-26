#!/usr/bin/env python3
"""Phase 3G raw-4096 preflight and LoRA-only AOGD trainer."""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from eval.forensics_eval import UNIFIED_FORENSICS_QUESTION
from scripts.phase3f_train import (append, dump, make_language_batch, optimizer_and_scheduler, save_checkpoint, setup, teacher_batch)
from tools.distributed_loss import batch_supervision_counts
from scripts.phase1b_preflight import move_batch
from tools.phase3f_aogd import (capture_representations, core_model, cosine_loss, file_sha256, grad_norm, group_hashes,
 parameter_group, replay_batch)

def args():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/phase3g_raw4096_aogd.yaml"); p.add_argument("--physical-gpu",type=int,default=0)
 p.add_argument("--mode",choices=("preflight","train"),required=True); p.add_argument("--resume"); p.add_argument("--optimizer-steps",type=int); return p.parse_args()
def relative_l2(left,right): return (left.float()-right.float()).norm()/right.float().norm().clamp_min(1e-8)

def preflight(cfg,out,model,backend,dataset,schedule,replay,teacher):
 core=core_model(model); fixed=[(i,s) for i,s,l in zip(schedule["indices"],schedule["sample_ids"],schedule["class_labels"])
  if int(l)==1 and replay[s]["replay_eligible"]][:int(cfg["preflight"]["fake_count"])]
 if len(fixed)!=32: raise RuntimeError("32 eligible Fake unavailable")
 rows=[]; model.eval()
 with torch.no_grad():
  for index,sid in fixed:
   sample=dataset[index]; auto_batch=replay_batch(backend,sample,replay[sid],UNIFIED_FORENSICS_QUESTION); oracle_batch=teacher_batch(backend,sample)
   ar,ap,_,_=capture_representations(core,auto_batch); tr,tp,_,_=capture_representations(core,oracle_batch)
   from tools.phase3f_aogd import mask_metrics_from_projected
   am=mask_metrics_from_projected(core,auto_batch,ap); tm=mask_metrics_from_projected(core,oracle_batch,tp)
   rows.append({"sample_id":sid,"G0_IoU":am["foreground_iou"],"TF_IoU":tm["foreground_iou"],
    "raw_4096_cosine_gap":float(cosine_loss(ar,tr)),"raw_4096_relative_l2":float(relative_l2(ar,tr)),
    "projected_256_cosine_gap":float(cosine_loss(ap,tp))})
 mean=lambda key:float(np.mean([r[key] for r in rows])); checks={"A_oracle_advantage":mean("TF_IoU")>mean("G0_IoU"),
  "B_raw_gap_nonzero":mean("raw_4096_cosine_gap")>0,"B_relative_l2_nonzero":mean("raw_4096_relative_l2")>0}
 index,sid=fixed[0]; sample=dataset[index]; batch=replay_batch(backend,sample,replay[sid],UNIFIED_FORENSICS_QUESTION)
 model.train(); model.zero_grad(set_to_none=True); torch.manual_seed(77101); torch.cuda.manual_seed_all(77101)
 raw,_,_,_=capture_representations(core,batch); loss=cosine_loss(raw,teacher[sid]["raw_4096d"].to(raw.device)); loss.backward()
 frozen_grad=sum(p.grad is not None for n,p in model.named_parameters() if parameter_group(n)!="lora")
 checks.update({"C_LoRA_gradient_positive":grad_norm(model,"lora")>0,"C_all_frozen_gradients_absent":frozen_grad==0}); grad_value=grad_norm(model,"lora"); model.zero_grad(set_to_none=True)
 before_hash=group_hashes(model); saved={n:p.detach().cpu().clone() for n,p in model.named_parameters() if parameter_group(n)=="lora"}
 directional=fixed[:int(cfg["preflight"]["directional_minibatch_count"])]
 def gaps():
  model.eval(); values=[]
  with torch.no_grad():
   for idx,sample_id in directional:
    b=replay_batch(backend,dataset[idx],replay[sample_id],UNIFIED_FORENSICS_QUESTION); h,_,_,_=capture_representations(core,b)
    values.append(float(cosine_loss(h,teacher[sample_id]["raw_4096d"].to(h.device))))
  return values
 gap_before=gaps(); optimizer,_=optimizer_and_scheduler(model,cfg); optimizer.zero_grad(set_to_none=True); model.train()
 for position,(idx,sample_id) in enumerate(directional):
  torch.manual_seed(77200+position); torch.cuda.manual_seed_all(77200+position)
  b=replay_batch(backend,dataset[idx],replay[sample_id],UNIFIED_FORENSICS_QUESTION); h,_,_,_=capture_representations(core,b)
  (cosine_loss(h,teacher[sample_id]["raw_4096d"].to(h.device))/len(directional)).backward()
 optimizer.step(); model.zero_grad(set_to_none=True); gap_after=gaps(); after_hash=group_hashes(model)
 checks.update({"D_minibatch_mean_moved_toward_oracle":float(np.mean(gap_after))<float(np.mean(gap_before)),
  "D_LoRA_changed":before_hash["lora"]!=after_hash["lora"],"D_text_hidden_fcs_unchanged":before_hash["text_hidden_fcs"]==after_hash["text_hidden_fcs"],
  "D_mask_decoder_unchanged":before_hash["mask_decoder"]==after_hash["mask_decoder"],"D_other_frozen_unchanged":before_hash["frozen"]==after_hash["frozen"]})
 with torch.no_grad():
  named=dict(model.named_parameters())
  for n,v in saved.items(): named[n].copy_(v.to(device=named[n].device,dtype=named[n].dtype))
 checks["D_original_P1_restored_exactly"]=group_hashes(model)==before_hash
 result={"status":"PASS" if all(checks.values()) else "FAIL","gate_on_failure":cfg["preflight"]["gate_on_failure"],"checks":checks,
  "sample_count":32,"means":{k:mean(k) for k in ("G0_IoU","TF_IoU","raw_4096_cosine_gap","raw_4096_relative_l2","projected_256_cosine_gap")},
  "Lrepr4096_LoRA_grad_norm":grad_value,"frozen_gradient_tensor_count":frozen_grad,
  "directional_audit":{"sample_ids":[s for _,s in directional],"gaps_before":gap_before,"gaps_after":gap_after,
   "mean_before":float(np.mean(gap_before)),"mean_after":float(np.mean(gap_after)),"delta":float(np.mean(gap_after)-np.mean(gap_before))},"rows":rows}
 dump(out/"preflight_raw4096.json",result); dump(out/"gradient_path_audit.json",{"status":result["status"],"checks":{k:v for k,v in checks.items() if k.startswith("C_")},"LoRA_grad_norm":grad_value})
 dump(out/"one_step_directional_audit.json",{"status":result["status"],**result["directional_audit"],"checks":{k:v for k,v in checks.items() if k.startswith("D_")}})
 if result["status"]!="PASS": raise RuntimeError(cfg["preflight"]["gate_on_failure"])
 print(json.dumps({"status":"PASS","means":result["means"],"directional":result["directional_audit"],"grad":grad_value},indent=2))

def checkpoint_diagnostic(model,core,dataset,tokenizer,backend,replay,teacher,index,cfg):
 sample=dataset[index]; sid=sample["sample_id"]; model.zero_grad(set_to_none=True); model.train(); torch.manual_seed(88101); torch.cuda.manual_seed_all(88101)
 lb,_=make_language_batch(dataset,index,tokenizer,next(model.parameters()).device); output=model(**lb); output["ce_loss"].backward(); lg=grad_norm(model,"lora"); model.zero_grad(set_to_none=True)
 torch.manual_seed(88102); torch.cuda.manual_seed_all(88102); b=replay_batch(backend,sample,replay[sid],UNIFIED_FORENSICS_QUESTION); h,g,_,_=capture_representations(core,b)
 rawloss=cosine_loss(h,teacher[sid]["raw_4096d"].to(h.device)); rawloss.backward(); rg=grad_norm(model,"lora"); model.zero_grad(set_to_none=True); model.eval()
 with torch.no_grad():
  h,g,_,_=capture_representations(core,b); th=teacher[sid]["raw_4096d"].to(h.device); tg=teacher[sid]["projected_256d"].to(g.device)
  result={"fixed_sample_id":sid,"language_lora_gradient_norm":lg,"raw4096_lora_gradient_norm":rg,
   "raw_4096_cosine_gap":float(cosine_loss(h,th)),"raw_4096_relative_l2":float(relative_l2(h,th)),"projected_256_cosine_gap":float(cosine_loss(g,tg))}
 model.train(); return result

def train(cfg,out,model,source,missing,boundary,tokenizer,backend,dataset,schedule,replay,teacher,resume,bounded):
 if json.loads((out/"preflight_raw4096.json").read_text())["status"]!="PASS": raise RuntimeError("preflight PASS required")
 optimizer,scheduler=optimizer_and_scheduler(model,cfg); total=int(cfg["training"]["total_optimizer_steps"]); start=0
 if resume:
  state=torch.load(Path(resume),map_location="cpu"); model.load_state_dict(state["module"],strict=False); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["lr_scheduler"]); start=int(state["optimizer_step"])
 metrics=out/"training/training_metrics.jsonl"
 if metrics.exists() and not resume: raise RuntimeError("existing metrics require resume")
 if resume:
  for path in (metrics,out/"training/checkpoint_metadata.jsonl"):
   if path.exists():
    kept=[line for line in path.read_text().splitlines() if line.strip() and int(json.loads(line)["optimizer_step"])<=start]; path.write_text(("\n".join(kept)+"\n") if kept else "")
 final=total if bounded is None else min(total,start+bounded); ckpt_root=Path(cfg["experiment"]["checkpoint_root"]); initial=group_hashes(model); core=core_model(model)
 dump(out/"training/initialization_audit.json",{"status":"PASS","P1_sha256":cfg["source"]["checkpoint_sha256"],"boundary":boundary,"initial_hashes":initial,"missing_frozen_keys":len(missing),"target":"raw_4096d"})
 dump(ckpt_root/"step_0000/metadata.json",{"role":"original_P1_reference","checkpoint":cfg["source"]["checkpoint"],"sha256":cfg["source"]["checkpoint_sha256"]})
 gas=int(cfg["training"]["gradient_accumulation_steps"]); formal=set(map(int,cfg["training"]["formal_checkpoint_steps"])); diagnostic_index=next(i for i,s,l in zip(schedule["indices"],schedule["sample_ids"],schedule["class_labels"]) if int(l)==1 and replay[s]["replay_eligible"])
 started=time.time(); model.train()
 for zero in range(start,final):
  t=time.time(); optimizer.zero_grad(set_to_none=True); window=[]
  for micro in range(gas):
   exposure=zero*gas+micro; idx=int(schedule["indices"][exposure]); cpu,sample=make_language_batch(dataset,idx,tokenizer,torch.device("cpu")); window.append((exposure,cpu,sample,batch_supervision_counts(cpu),int(sample["cls_label"])==1 and replay[sample["sample_id"]]["replay_eligible"]))
  tokens=sum(x[3]["text_tokens"] for x in window); eligible=sum(x[4] for x in window); ce_sum=raw_sum=l2_sum=proj_sum=0.0; ids=[]; labels=[]
  for exposure,cpu,sample,count,valid in window:
   seed=int(cfg["experiment"]["seed"])*1000003+exposure; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); batch=move_batch(cpu,next(model.parameters()).device,torch.bfloat16)
   output=model(**batch); ce=output["ce_loss"]; scale=count["text_tokens"]/tokens; (ce*scale).backward(); ce_sum+=float(ce.detach().float())*scale; del output,batch,ce
   if valid:
    sid=sample["sample_id"]; b=replay_batch(backend,sample,replay[sid],UNIFIED_FORENSICS_QUESTION); h,g,_,_=capture_representations(core,b); th=teacher[sid]["raw_4096d"].to(h.device); tg=teacher[sid]["projected_256d"].to(g.device)
    loss=cosine_loss(h,th); (loss/eligible).backward(); raw_sum+=float(loss.detach())/eligible; l2_sum+=float(relative_l2(h,th).detach())/eligible; proj_sum+=float(cosine_loss(g,tg).detach())/eligible
   ids.append(sample["sample_id"]); labels.append(int(sample["cls_label"]))
  gn=grad_norm(model,"lora"); before=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg["training"]["gradient_clip_norm"])))
  optimizer.step(); scheduler.step(); step=zero+1; record={"optimizer_step":step,"language_ce":ce_sum,"L_repr_4096":raw_sum,"raw_4096_relative_l2":l2_sum,"projected_256_gap_diagnostic":proj_sum,
   "total_loss":ce_sum+raw_sum,"lora_gradient_norm":gn,"gradient_norm_before_clip":before,"learning_rate":float(optimizer.param_groups[0]["lr"]),"eligible_fake_count":eligible,"sample_ids":ids,"class_labels":labels,
   "rollout_cache_sha256":cfg["rollout"]["cache_sha256"],"seconds_this_optimizer_step":time.time()-t}; append(metrics,record)
  if step<=5 or step%10==0: print(json.dumps(record),flush=True)
  if step in formal and step>0:
   if file_sha256(ROOT/cfg["rollout"]["phase3b_cache"])!=cfg["rollout"]["cache_sha256"]: raise RuntimeError("rollout hash changed")
   diag=checkpoint_diagnostic(model,core,dataset,tokenizer,backend,replay,teacher,diagnostic_index,cfg); metadata={"optimizer_step":step,"epoch":step//100,"allowed_parameter_groups":["lora"],"source_checkpoint_sha256":cfg["source"]["checkpoint_sha256"],"training_record":record,"checkpoint_diagnostics":diag}
   path=save_checkpoint(model,source["module"],optimizer,scheduler,ckpt_root,step,metadata); append(out/"training/checkpoint_metadata.jsonl",{**metadata,"checkpoint":str(path),"sha256":file_sha256(path)})
   if step in (500,1000,4500):
    ip=out/"rollout_integrity_audit.json"; iv=json.loads(ip.read_text()); iv[f"step_{step}"]="PASS"; dump(ip,iv)
 final_hash=group_hashes(model); changed={k:initial[k]!=final_hash[k] for k in initial}; expected={"lora":True,"text_hidden_fcs":False,"mask_decoder":False,"frozen":False}; audit={"status":"PASS" if changed==expected else "FAIL","changed":changed,"expected":expected,"initial":initial,"final":final_hash}; dump(out/"parameter_update_audit.json",audit)
 if audit["status"]!="PASS": raise RuntimeError("boundary audit failed")
 dump(out/"training/run_summary.json",{"status":"COMPLETE" if final==total else "BOUNDED_RUN_COMPLETE","start_step":start,"final_step":final,"elapsed_seconds":time.time()-started,"seconds_per_step":(time.time()-started)/max(1,final-start),"parameter_audit":audit})

def main():
 a=args(); cfg=yaml.safe_load((ROOT/a.config).read_text()); out=ROOT/cfg["experiment"]["output_root"]
 values=setup(cfg,a.physical_gpu,"train"); device,_,tokenizer,model,source,missing,boundary,backend,dataset,schedule,replay=values
 teacher=torch.load(Path(cfg["phase3f_reuse"]["teacher_cache"]),map_location="cpu")["vectors"]
 if a.mode=="preflight": preflight(cfg,out,model,backend,dataset,schedule,replay,teacher)
 else: train(cfg,out,model,source,missing,boundary,tokenizer,backend,dataset,schedule,replay,teacher,a.resume,a.optimizer_steps)
if __name__=="__main__": main()
