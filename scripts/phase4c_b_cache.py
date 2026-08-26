#!/usr/bin/env python3
"""Build frozen autonomous train q_seg and fresh validation trajectory caches."""
from __future__ import annotations
import argparse,gc,json,random,sys,time
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend,UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.phase3f_aogd import capture_representations,core_model,replay_batch
from tools.phase4c_b import canonical_hash,dump,file_sha256,metric_record,rows,tensor_hash


def cli():
 p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("train_q","G0","phrase_only","tf_full_context"),required=True); p.add_argument("--device",default="cuda:0"); return p.parse_args()


def setup(cfg,split,device):
 p1cfg=yaml.safe_load((ROOT/cfg["p1"]["model_config"]).read_text()); conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
 model,tokenizer,meta=load_model(p1cfg,Path(cfg["p1"]["checkpoint"]),device,expected_step=3500,expected_epoch=7)
 backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=400)
 dataset=UnifiedForensicsDataset(ROOT/cfg["data"]["manifest_dir"],tokenizer,p1cfg["model"]["vision_tower"],split=split,datasets_root=p1cfg["data"]["datasets_root"],synthscars_root=p1cfg["data"]["synthscars_root"],image_size=1024,target_protocol="phrase_aligned")
 return model,backend,dataset,meta


def capture_context(core,call):
 holder=[]; original=core._extract_projected_seg_predictor_hidden
 def wrapped(*args,**kwargs):
  result=original(*args,**kwargs); holder.append([x.detach().to(torch.bfloat16).cpu() for x in result[0]]); return result
 core._extract_projected_seg_predictor_hidden=wrapped
 try: output=call()
 finally: core._extract_projected_seg_predictor_hidden=original
 vectors=holder[-1] if holder else []
 q=vectors[0][0] if len(vectors)==1 and vectors[0].shape[0]==1 else None
 return output,q


def scalar(v,kind=float):
 if torch.is_tensor(v): v=v.detach().cpu().item()
 return kind(v)


def save_shard(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); tmp.replace(path)


def train_q(cfg,out,device):
 dest=Path(cfg["experiment"]["cache_root"])/"train_q_seg"; dest.mkdir(parents=True,exist_ok=True); complete=dest/"complete.json"
 if complete.exists() and json.loads(complete.read_text())["status"]=="COMPLETE": print(complete.read_text()); return
 model,backend,dataset,meta=setup(cfg,"train",device); core=core_model(model)
 replay={r["sample_id"]:r for r in rows(ROOT/cfg["rollout"]["cache"])}
 eligible=[i for i,r in enumerate(dataset.rows) if int(r["class_label"])==1 and replay[r["sample_id"]]["replay_eligible"]]
 if len(eligible)!=8690: raise RuntimeError(f"eligible mismatch {len(eligible)}")
 started=time.time(); shard_size=128
 for start in range(0,len(eligible),shard_size):
  end=min(start+shard_size,len(eligible)); path=dest/f"shard_{start:06d}_{end:06d}.pt"
  if path.exists(): continue
  values=[]; records=[]
  with torch.no_grad():
   for pos,index in enumerate(eligible[start:end],start+1):
    sample=dataset[index]; sid=sample["sample_id"]; batch=replay_batch(backend,sample,replay[sid],UNIFIED_FORENSICS_QUESTION)
    _,q,_,_=capture_representations(core,batch); q=q.detach().to(torch.bfloat16).cpu(); values.append(q)
    rr=replay[sid]; records.append({"sample_id":sid,"dataset_index":index,"image_path":rr["image_path"],"rollout_hash":canonical_hash(rr["full_input_token_ids"]),"generated_token_ids_hash":canonical_hash(rr["generated_token_ids"]),"P1_sha256":cfg["p1"]["checkpoint_sha256"],"q_shape":list(q.shape),"q_dtype":str(q.dtype),"q_hash":tensor_hash(((sid,q),))})
    done=pos
    if done<=5 or done%50==0: print(json.dumps({"train_q":done,"total":len(eligible),"seconds_per_sample":(time.time()-started)/done}),flush=True)
  save_shard(path,{"schema":"phase4c_b_train_q_seg_v1","start":start,"end":end,"q_seg":torch.stack(values),"records":records})
 files=sorted(dest.glob("shard_*.pt")); manifest={"status":"COMPLETE","count":len(eligible),"raw_fake_count":8836,"excluded_count":146,"exclusion_reason":"missing valid autonomous SEG predictor","shards":len(files),"shard_size":shard_size,"sample_ids_sha256":canonical_hash([dataset.rows[i]["sample_id"] for i in eligible]),"P1_checkpoint_sha256":cfg["p1"]["checkpoint_sha256"],"rollout_cache_sha256":cfg["rollout"]["cache_sha256"],"feature_shape":[256],"dtype":"torch.bfloat16","elapsed_seconds":time.time()-started,"files":[{"path":str(p),"sha256":file_sha256(p)} for p in files],"P1_checkpoint_hash_unchanged":file_sha256(Path(cfg["p1"]["checkpoint"]))==cfg["p1"]["checkpoint_sha256"]}
 dump(complete,manifest); print(json.dumps({k:manifest[k] for k in ("status","count","elapsed_seconds")},indent=2))


def val_context(cfg,out,device,mode):
 dest=Path(cfg["experiment"]["cache_root"])/"validation"/mode; dest.mkdir(parents=True,exist_ok=True); complete=dest/"complete.json"
 if complete.exists() and json.loads(complete.read_text())["status"]=="COMPLETE": print(complete.read_text()); return
 model,backend,dataset,meta=setup(cfg,"val",device); core=core_model(model); fake=[i for i,r in enumerate(dataset.rows) if int(r["class_label"])==1]
 if len(fake)!=1106: raise RuntimeError("val Fake mismatch")
 started=time.time(); shard_size=16 if mode=="G0" else 64
 for start in range(0,len(fake),shard_size):
  end=min(start+shard_size,len(fake)); path=dest/f"shard_{start:06d}_{end:06d}.pt"
  if path.exists(): continue
  values=[]; valid=[]; records=[]
  for pos,index in enumerate(fake[start:end],start+1):
   sample=dataset[index]; target=torch.as_tensor(sample["masks"]).bool().any(dim=0).cpu()
   if mode=="G0": output,q=capture_context(core,lambda:backend.generate_localization(sample,provide_gt_fake=False,generation_mode="unified_fake_generate")); context_ids=output.get("prompt_token_ids",[])+output.get("generated_token_ids",[])
   elif mode=="phrase_only":
    batch=backend._batch(sample,backend.phrase_only_content(sample),question=UNIFIED_FORENSICS_QUESTION); context_ids=batch["input_ids"][0].detach().cpu().tolist(); output,q=capture_context(core,lambda:backend.phrase_only_localization(sample))
   else:
    batch=backend._batch(sample,backend.tf_phrase_content(sample),question=UNIFIED_FORENSICS_QUESTION); context_ids=batch["input_ids"][0].detach().cpu().tolist(); output,q=capture_context(core,lambda:backend.teacher_forced_localization(sample,context="full",user_prompt="canonical"))
   pred=output.get("pred_mask"); logits=torch.full(target.shape,-torch.inf) if pred is None else torch.as_tensor(pred).detach().float().cpu().amax(0)
   metric=metric_record(sample["sample_id"],logits,target); is_valid=q is not None
   values.append(torch.zeros(256,dtype=torch.bfloat16) if q is None else q); valid.append(is_valid)
   rec={**metric,"mode":mode,"valid_q_seg":is_valid,"context_token_ids_sha256":canonical_hash(context_ids),"context_token_count":len(context_ids),"q_seg_hash":None if q is None else tensor_hash(((sample["sample_id"],q),)),"generated_token_ids":output.get("generated_token_ids",[]) if mode=="G0" else context_ids,"generated_token_ids_sha256":canonical_hash(output.get("generated_token_ids",[]) if mode=="G0" else context_ids),"generated_text":output.get("generated_text") if mode=="G0" else None,"generated_text_sha256":canonical_hash(output.get("generated_text") or ""),"seg_triggered":bool(output.get("seg_triggered",True)),"cls_pred":scalar(output["cls_pred"],int),"cls_prob_fake":scalar(output["cls_prob_fake"]),"lm_verdict_pred":scalar(output["lm_verdict_pred"],int),"lm_verdict_prob_fake":scalar(output["lm_verdict_prob_fake"]),"P1_sha256":cfg["p1"]["checkpoint_sha256"],"canonical_prompt":cfg["prompt"]["text"]}
   records.append(rec); done=pos
   if done<=3 or done%20==0: print(json.dumps({mode:done,"total":len(fake),"valid":sum(valid),"seconds_per_sample":(time.time()-started)/done}),flush=True)
  save_shard(path,{"schema":"phase4c_b_validation_context_v1","mode":mode,"start":start,"end":end,"q_seg":torch.stack(values),"valid":torch.tensor(valid),"records":records})
 files=sorted(dest.glob("shard_*.pt")); all_records=[]
 for p in files: all_records+=torch.load(p,map_location="cpu")["records"]
 manifest={"status":"COMPLETE","mode":mode,"fresh_canonical_P1_execution":True,"generation_batch_size":1 if mode=="G0" else None,"count":len(all_records),"valid_q_seg_count":sum(r["valid_q_seg"] for r in all_records),"sample_ids_sha256":canonical_hash([r["sample_id"] for r in all_records]),"generated_sequence_set_sha256":canonical_hash([r["generated_token_ids_sha256"] for r in all_records]),"P1_checkpoint_sha256":cfg["p1"]["checkpoint_sha256"],"elapsed_seconds":time.time()-started,"files":[{"path":str(p),"sha256":file_sha256(p)} for p in files],"P1_checkpoint_hash_unchanged":file_sha256(Path(cfg["p1"]["checkpoint"]))==cfg["p1"]["checkpoint_sha256"]}
 dump(complete,manifest); print(json.dumps({k:manifest[k] for k in ("status","mode","count","valid_q_seg_count","elapsed_seconds")},indent=2))


def main():
 a=cli(); cfg=yaml.safe_load((ROOT/"configs/phase4c_b_evidence_reader.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]; device=torch.device(a.device); torch.cuda.set_device(device); random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
 if a.mode=="train_q": train_q(cfg,out,device)
 else: val_context(cfg,out,device,a.mode)
if __name__=="__main__": main()
