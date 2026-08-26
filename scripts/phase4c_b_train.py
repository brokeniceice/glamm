#!/usr/bin/env python3
"""Mandatory preflight and matched Evidence Reader training/selection."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.evidence_reader import EvidenceReader
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.phase3f_aogd import core_model
from tools.phase4c_b import append,decode_low_res,diagnostic,dump,file_sha256,inverse_sam_logits,load_source_model,load_spatial_shard,metric_record,rows,sam_input_loss,source_feature,spatial_cache_paths,summarize,tensor_hash


def cli():
 p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("preflight","train"),required=True); p.add_argument("--arm",choices=("clip_reader","forensic_reader")); p.add_argument("--device",default="cuda:0"); return p.parse_args()


def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_reader(device): seed_all(3407); return EvidenceReader().to(device)


def load_p1(cfg,device):
 p1cfg=yaml.safe_load((ROOT/cfg["p1"]["model_config"]).read_text()); conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
 model,_,_=load_model(p1cfg,Path(cfg["p1"]["checkpoint"]),device,expected_step=3500,expected_epoch=7); model.requires_grad_(False); return model,core_model(model)


def q_index(root,context="train_q_seg"):
 result={}; base=root/context
 for path in sorted(base.glob("shard_*.pt")):
  shard=torch.load(path,map_location="cpu")
  for i,r in enumerate(shard["records"]): result[r["sample_id"]]=(shard["q_seg"][i],bool(shard.get("valid",torch.ones(len(shard["records"]),dtype=torch.bool))[i]))
 return result


def grad_norm(module,exclude_beta=False):
 values=[]
 for name,p in module.named_parameters():
  if exclude_beta and name=="beta": continue
  if p.grad is not None: values.append(p.grad.detach().float().pow(2).sum())
 return math.sqrt(sum(float(v) for v in values)) if values else 0.0


def optimizer_scheduler(reader,cfg):
 opt=torch.optim.AdamW(reader.parameters(),lr=float(cfg["optimizer"]["learning_rate"]),weight_decay=float(cfg["optimizer"]["weight_decay"]),betas=tuple(cfg["optimizer"]["betas"]))
 warm=int(cfg["optimizer"]["warmup_steps"]); total=int(cfg["training"]["total_optimizer_steps"])
 def scale(step):
  if step<warm:return (step+1)/warm
  x=(step-warm)/max(1,total-warm); return .5*(1+math.cos(math.pi*min(1.,x)))
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)


def pair_paths(source,split):
 clip=spatial_cache_paths(source,"clip",split); sam=spatial_cache_paths(source,"sam",split)
 if len(clip)!=len(sam):raise RuntimeError("CLIP/SAM shard mismatch")
 return list(zip(clip,sam))


def frozen_hash(core):
 return {"prompt_encoder":tensor_hash(core.model.grounding_encoder.prompt_encoder.state_dict().items()),"mask_decoder":tensor_hash(core.model.grounding_encoder.mask_decoder.state_dict().items())}


def preflight(cfg,out,device):
 cache=Path(cfg["experiment"]["cache_root"]); q=q_index(cache); source_root=ROOT/cfg["frozen_spatial_cache"]["phase3c1_root"]
 model,core=load_p1(cfg,device); pairs=pair_paths(source_root,"train"); clip=load_spatial_shard(pairs[0][0],"clip"); sam=load_spatial_shard(pairs[0][1],"sam"); ids=[r["sample_id"] for r in clip["records"] if r["sample_id"] in q][:16]
 if len(ids)<16: raise RuntimeError("not enough preflight IDs")
 index={r["sample_id"]:i for i,r in enumerate(clip["records"])}; frozen_before=frozen_hash(core); arm_rows={}; initial={}
 for arm in ("clip_reader","forensic_reader"):
  reader=make_reader(device); source=load_source_model(cfg,arm,device); initial[arm]=tensor_hash(reader.state_dict().items()); saved={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
  equivalence=[]; source_stats=[]
  for sid in ids:
   i=index[sid]; raw=clip["features"][i:i+1].to(device=device,dtype=torch.float32); spatial=source_feature(source,raw,arm).float(); qseg=q[sid][0].to(device=device,dtype=torch.float32).reshape(1,256); image=sam["features"][i].to(device=device,dtype=torch.bfloat16)
   value=reader(qseg,spatial); baseline=decode_low_res(core,qseg.to(torch.bfloat16).reshape(1,1,256),image); active=decode_low_res(core,value["q_final"].to(torch.bfloat16),image)
   equivalence.append({"sample_id":sid,"q_final_q_seg_max_abs":float((value["q_final"][:,0]-qseg).abs().max()),"mask_logits_max_abs":float((active-baseline).abs().max())})
   source_stats.append({"finite":bool(torch.isfinite(spatial).all()),"nonzero":bool(spatial.abs().max()>0),"std":float(spatial.std()),"shape":list(spatial.shape)})
  sid=ids[0]; i=index[sid]; raw=clip["features"][i:i+1].to(device=device,dtype=torch.float32); spatial=source_feature(source,raw,arm).float(); qseg=q[sid][0].to(device=device,dtype=torch.float32).reshape(1,256); image=sam["features"][i].to(device=device,dtype=torch.bfloat16); target=sam["targets"][i].to(device)
  reader.zero_grad(set_to_none=True); value=reader(qseg,spatial); loss=sam_input_loss(decode_low_res(core,value["q_final"].to(torch.bfloat16),image),target)["total"]; loss.backward(); beta_grad=float(reader.beta.grad); step_a_reader_grad=grad_norm(reader,exclude_beta=True)
  opt=torch.optim.SGD(reader.parameters(),lr=.01); opt.step(); beta_after=float(reader.beta.detach()); reader.zero_grad(set_to_none=True); value2=reader(qseg,spatial); loss2=sam_input_loss(decode_low_res(core,value2["q_final"].to(torch.bfloat16),image),target)["total"]; loss2.backward(); step_b_reader_grad=grad_norm(reader,exclude_beta=True); p1_grad=sum(p.grad is not None for p in model.parameters()); source_grad=sum(p.grad is not None for p in source.parameters())
  reader.load_state_dict(saved); restored=tensor_hash(reader.state_dict().items())==initial[arm]
  arm_rows[arm]={"equivalence":equivalence,"source_integrity":source_stats,"step_A":{"beta":0.0,"beta_grad":beta_grad,"reader_grad_norm":step_a_reader_grad,"loss":float(loss)},"temporary_beta_after_update":beta_after,"step_B":{"reader_grad_norm":step_b_reader_grad,"loss":float(loss2)},"frozen_P1_gradient_tensor_count":p1_grad,"frozen_source_gradient_tensor_count":source_grad,"temporary_state_restored_exactly":restored}
 checks={"reader_initial_hash_equal":initial["clip_reader"]==initial["forensic_reader"],"all_q_exact":all(r["q_final_q_seg_max_abs"]==0 for a in arm_rows.values() for r in a["equivalence"]),"all_mask_exact":all(r["mask_logits_max_abs"]==0 for a in arm_rows.values() for r in a["equivalence"]),"all_sources_integral":all(r["finite"] and r["nonzero"] and r["std"]>1e-6 and r["shape"]==[1,256,24,24] for a in arm_rows.values() for r in a["source_integrity"]),"step_A_beta_grad_nonzero":all(abs(a["step_A"]["beta_grad"])>0 for a in arm_rows.values()),"step_A_reader_grad_zero_expected":all(a["step_A"]["reader_grad_norm"]==0 for a in arm_rows.values()),"step_B_reader_grad_nonzero":all(a["step_B"]["reader_grad_norm"]>0 for a in arm_rows.values()),"all_frozen_grads_absent":all(a["frozen_P1_gradient_tensor_count"]==0 and a["frozen_source_gradient_tensor_count"]==0 for a in arm_rows.values()),"temporary_restore_exact":all(a["temporary_state_restored_exactly"] for a in arm_rows.values()),"P1_SAM_hash_unchanged":frozen_hash(core)==frozen_before}
 status="PASS" if all(checks.values()) else "FAIL"
 dump(out/"preflight_p1_equivalence.json",{"status":status,"n":16,"tolerance":0.0,"arms":{a:r["equivalence"] for a,r in arm_rows.items()},"gate_on_failure":"GATE_EVIDENCE_READER_NOT_P1_EQUIVALENT_AT_INIT"})
 dump(out/"preflight_source_integrity.json",{"status":status,"arms":{a:r["source_integrity"] for a,r in arm_rows.items()},"same_shape":True,"checkpoint_hashes":{"clip_proj":cfg["phase4c_a"]["clip_proj_sha256"],"forensic_adapter":cfg["phase4c_a"]["forensic_adapter_sha256"]}})
 dump(out/"preflight_two_step_gradient.json",{"status":status,"checks":checks,"arms":arm_rows})
 dump(out/"parameter_update_audit.json",{"status":"PREFLIGHT_PASS_FORMAL_TRAINING_PENDING","preflight_temporary_updates_restored":checks["temporary_restore_exact"],"formal":{}})
 fair=json.loads((out/"fairness_manifest.json").read_text()); fair.update({"status":"PASS","runtime_initial_hashes":initial,"runtime_initial_hash_equal":checks["reader_initial_hash_equal"]}); dump(out/"fairness_manifest.json",fair)
 if status!="PASS": raise RuntimeError("Phase 4C-B mandatory preflight failed")
 print(json.dumps({"status":status,"checks":checks,"arms":{a:{"beta_grad_A":r["step_A"]["beta_grad"],"reader_grad_B":r["step_B"]["reader_grad_norm"]} for a,r in arm_rows.items()}},indent=2))


def evaluate(reader,source,core,qcache,pairs,device,save_attention=False,
             evidence_by_id=None,evidence_transform=None,include_ids=None):
 reader.eval(); records=[]; low=[]; attn=[]; diag={k:[] for k in ("attention_entropy","max_attention_weight","q_evidence_norm","q_seg_norm","delta_norm")}
 include_ids=None if include_ids is None else set(include_ids)
 with torch.no_grad():
  for cp,sp in pairs:
   clip=load_spatial_shard(cp,"clip"); sam=load_spatial_shard(sp,"sam")
   spatial=None
   if evidence_by_id is None:
    raw=clip["features"].to(device=device,dtype=torch.float32); spatial=source_feature(source,raw,source._phase4c_b_arm).float()
   for i,meta in enumerate(clip["records"]):
    sam_meta=sam["records"][i]
    if sam_meta["sample_id"] != meta["sample_id"]: raise RuntimeError("CLIP/SAM sample identity mismatch")
    sid=meta["sample_id"]; target=sam["original_masks"][i].to(device)
    if include_ids is not None and sid not in include_ids: continue
    if sid not in qcache or not qcache[sid][1]: logits=torch.full(target.shape,-torch.inf,device=device); low.append(torch.full((256,256),-torch.inf)); weight=torch.full((4,1,576),1/576)
    else:
     evidence=(spatial[i:i+1] if evidence_by_id is None else evidence_by_id[sid].to(device=device,dtype=torch.float32).unsqueeze(0))
     if evidence_transform is not None: evidence=evidence_transform(sid,evidence)
     q=qcache[sid][0].to(device=device,dtype=torch.float32).reshape(1,256); value=reader(q,evidence); lowmask=decode_low_res(core,value["q_final"].to(torch.bfloat16),sam["features"][i].to(device=device,dtype=torch.bfloat16)); logits=inverse_sam_logits(lowmask,sam_meta["geometry"]); low.append(lowmask[0,0].detach().cpu().to(torch.bfloat16)); weight=value["attention"][0].detach().cpu()
     w=weight.float().clamp_min(1e-12); diag["attention_entropy"].append(float(-(w*w.log()).sum(-1).mean())); diag["max_attention_weight"].append(float(w.max(-1).values.mean())); diag["q_evidence_norm"].append(float(value["q_evidence"].float().norm())); diag["q_seg_norm"].append(float(q.norm())); diag["delta_norm"].append(float((value["q_final"][:,0]-q).norm()))
    records.append(metric_record(sid,logits,target));
    if save_attention: attn.append(weight.to(torch.bfloat16))
 return summarize(records),records,low,attn,{k:diagnostic(v) if v else None for k,v in diag.items()}


def train(cfg,out,device,arm):
 if not arm:raise ValueError("--arm required")
 pre=json.loads((out/"preflight_two_step_gradient.json").read_text());
 if pre["status"]!="PASS":raise RuntimeError("PASS preflight required")
 cache=Path(cfg["experiment"]["cache_root"]); trainq=q_index(cache); valq=q_index(cache/"validation","G0"); source_root=ROOT/cfg["frozen_spatial_cache"]["phase3c1_root"]; train_pairs=pair_paths(source_root,"train"); val_pairs=pair_paths(source_root,"val")
 model,core=load_p1(cfg,device); frozen_before=frozen_hash(core); source=load_source_model(cfg,arm,device); source._phase4c_b_arm=arm; source_before=tensor_hash(source.state_dict().items()); reader=make_reader(device); initial=tensor_hash(reader.state_dict().items()); opt,sched=optimizer_scheduler(reader,cfg)
 ckroot=Path(cfg["experiment"]["checkpoint_root"])/arm; ckroot.mkdir(parents=True,exist_ok=True); trainroot=out/"training"/arm; trainroot.mkdir(parents=True,exist_ok=True); history=[]; start_epoch=0; global_step=0
 existing=sorted(ckroot.glob("epoch_*.pt"))
 if existing:
  latest=max(existing,key=lambda p:int(p.stem.split("_")[-1])); state=torch.load(latest,map_location="cpu"); reader.load_state_dict(state["reader"]); opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"]); start_epoch=state["epoch"]; global_step=state["global_step"]; history=json.loads((trainroot/"history.json").read_text())
 else:
  metrics,records,low,_,diag=evaluate(reader,source,core,valq,val_pairs,device); state={"schema":"phase4c_b_reader_v1","arm":arm,"epoch":0,"global_step":0,"reader":reader.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"metrics":metrics,"diagnostics":diag,"initial_hash":initial}; torch.save(state,ckroot/"epoch_0.pt"); history=[{"epoch":0,"global_step":0,"train":None,"validation":metrics,"diagnostics":diag,"beta":0.0,"seconds":0.0}]; dump(trainroot/"history.json",history)
  baseline_spatial=out/"evaluation/G0/P1/epoch0_spatial.pt"
  if not baseline_spatial.exists(): baseline_spatial.parent.mkdir(parents=True,exist_ok=True); torch.save({"sample_ids":[r["sample_id"] for r in records],"low_res_logits":low,"source_arm_used_only_for_beta0_equivalence":arm},baseline_spatial)
 for epoch in range(start_epoch+1,11):
  began=time.time(); reader.train(); rng=random.Random(3407+epoch); pairs=list(train_pairs); rng.shuffle(pairs); opt.zero_grad(set_to_none=True); accum=0; sums={k:0. for k in ("bce","dice","total","beta_grad","reader_grad","q_evidence_norm","q_seg_norm","delta_norm","attention_entropy","max_attention_weight")}; n=0; order=[]
  for cp,sp in pairs:
   clip=load_spatial_shard(cp,"clip"); sam=load_spatial_shard(sp,"sam"); indices=[i for i,r in enumerate(clip["records"]) if r["sample_id"] in trainq]; rng.shuffle(indices); raw=clip["features"][indices].to(device=device,dtype=torch.float32); spatial=source_feature(source,raw,arm).float()
   for j,i in enumerate(indices):
    sid=clip["records"][i]["sample_id"]; q=trainq[sid][0].to(device=device,dtype=torch.float32).reshape(1,256); value=reader(q,spatial[j:j+1]); low=decode_low_res(core,value["q_final"].to(torch.bfloat16),sam["features"][i].to(device=device,dtype=torch.bfloat16)); losses=sam_input_loss(low,sam["targets"][i].to(device)); (losses["total"]/4).backward(); accum+=1; n+=1; order.append(sid)
    w=value["attention"].float().clamp_min(1e-12)
    for k in ("bce","dice","total"):sums[k]+=float(losses[k].detach())
    sums["q_evidence_norm"]+=float(value["q_evidence"].detach().float().norm()); sums["q_seg_norm"]+=float(q.norm()); sums["delta_norm"]+=float((value["q_final"][:,0]-q).detach().float().norm()); sums["attention_entropy"]+=float(-(w*w.log()).sum(-1).mean()); sums["max_attention_weight"]+=float(w.max(-1).values.mean())
    if accum==4 or n==8690:
     sums["beta_grad"]+=abs(float(reader.beta.grad)); sums["reader_grad"]+=grad_norm(reader,exclude_beta=True); torch.nn.utils.clip_grad_norm_(reader.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step+=1; accum=0
  if n!=8690:raise RuntimeError(f"epoch exposure mismatch {n}")
  metrics,records,low,_,diag=evaluate(reader,source,core,valq,val_pairs,device); steps=math.ceil(n/4); tr={k:(sums[k]/n if k not in ("beta_grad","reader_grad") else sums[k]/steps) for k in sums}; row={"epoch":epoch,"global_step":global_step,"train":tr,"validation":metrics,"diagnostics":diag,"beta":float(reader.beta.detach()),"sample_order_sha256":hashlib.sha256("\n".join(order).encode()).hexdigest(),"seconds":time.time()-began}; history.append(row); dump(trainroot/"history.json",history)
  state={"schema":"phase4c_b_reader_v1","arm":arm,"epoch":epoch,"global_step":global_step,"reader":reader.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"metrics":metrics,"diagnostics":diag,"initial_hash":initial}; torch.save(state,ckroot/f"epoch_{epoch}.pt"); print(json.dumps({"arm":arm,**row}),flush=True)
 best=max(history,key=lambda r:(r["validation"]["mean_foreground_iou"],r["validation"]["mean_foreground_f1"])); selected=torch.load(ckroot/f"epoch_{best['epoch']}.pt",map_location="cpu"); torch.save(selected,ckroot/"selected.pt"); reader.load_state_dict(selected["reader"]); metrics,records,low,attn,diag=evaluate(reader,source,core,valq,val_pairs,device,save_attention=True)
 predpath=out/"evaluation"/"G0"/arm/"selected_predictions.jsonl"; predpath.parent.mkdir(parents=True,exist_ok=True); predpath.write_text("".join(json.dumps(r)+"\n" for r in records)); torch.save({"sample_ids":[r["sample_id"] for r in records],"low_res_logits":low,"attention":attn},out/"evaluation"/"G0"/arm/"selected_spatial.pt")
 selector={"status":"COMPLETE","arm":arm,"primary":cfg["selector"]["primary"],"tie_break":cfg["selector"]["tie_break"],"candidates":[{"epoch":r["epoch"],**r["validation"],"beta":r["beta"]} for r in history],"selected_epoch":best["epoch"],"selected_metrics":metrics,"selected_beta":float(reader.beta.detach()),"selected_checkpoint":str((ckroot/"selected.pt").resolve()),"fallback_P1_allowed":True,"test_used":False,"official1000_used":False}
 dump(out/("selector_clip_reader.json" if arm=="clip_reader" else "selector_forensic_reader.json"),selector); dump(out/("checkpoint_metrics_clip_reader.json" if arm=="clip_reader" else "checkpoint_metrics_forensic_reader.json"),{"history":history,"selected":selector})
 after=tensor_hash(reader.state_dict().items()); update={"status":"PASS","arm":arm,"reader_changed":after!=initial,"reader_initial_hash":initial,"reader_final_hash":after,"P1_SAM_unchanged":frozen_hash(core)==frozen_before,"source_unchanged":tensor_hash(source.state_dict().items())==source_before,"P1_checkpoint_hash_unchanged":file_sha256(Path(cfg["p1"]["checkpoint"]))==cfg["p1"]["checkpoint_sha256"],"source_checkpoint_hash_unchanged":file_sha256(Path(cfg["phase4c_a"]["clip_proj_checkpoint"] if arm=="clip_reader" else cfg["phase4c_a"]["forensic_adapter_checkpoint"]))==(cfg["phase4c_a"]["clip_proj_sha256"] if arm=="clip_reader" else cfg["phase4c_a"]["forensic_adapter_sha256"])}; dump(trainroot/"parameter_update.json",update)
 dump(trainroot/"completion.json",{"status":"COMPLETE","epochs":10,"exposures":86900,"global_steps":global_step,"selected_epoch":best["epoch"],"elapsed_epoch10_seconds":history[-1]["seconds"]})


def main():
 a=cli(); cfg=yaml.safe_load((ROOT/"configs/phase4c_b_evidence_reader.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]; device=torch.device(a.device); torch.cuda.set_device(device); seed_all(3407)
 if a.mode=="preflight":preflight(cfg,out,device)
 else:train(cfg,out,device,a.arm)
if __name__=="__main__":main()
