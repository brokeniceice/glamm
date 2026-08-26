#!/usr/bin/env python3
"""Validation autonomous-oracle representation gaps for P1 and selected AOGD."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.phase3f_aogd import capture_representations, core_model, cosine_loss, file_sha256, replay_batch

CFG=yaml.safe_load((ROOT/"configs/phase3f_autonomous_oracle_grounding_distillation.yaml").read_text())
OUT=ROOT/CFG["experiment"]["output_root"]
TEACHER_CACHE=Path(CFG["experiment"]["checkpoint_root"])/"validation_teacher_cache/p1_oracle_representations.pt"

def dump(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
def append(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("a",encoding="utf-8") as f: f.write(json.dumps(value)+"\n"); f.flush()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--role",choices=("P1_TEACHER_AND_AUTO","AOGD_AUTO"),required=True); p.add_argument("--physical-gpu",type=int,required=True); a=p.parse_args()
 if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,str(a.physical_gpu)): raise RuntimeError("GPU mismatch")
 device=torch.device("cuda:0"); torch.cuda.set_device(device); conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
 model_cfg=yaml.safe_load((ROOT/CFG["source"]["model_config"]).read_text())
 if a.role=="P1_TEACHER_AND_AUTO":
  checkpoint=Path(CFG["source"]["checkpoint"]); step=3500; epoch=7
  pred_path=OUT/"evaluation/selector/P1_FROZEN/G0/predictions.jsonl"; name="P1_FROZEN"
 else:
  selector=json.loads((OUT/"evaluation/selector/AOGD_selector.json").read_text()); checkpoint=Path(selector["selected_checkpoint"]); step=1000; epoch=10
  pred_path=OUT/"evaluation/selector/AOGD/step_1000/G0/predictions.jsonl"; name="AOGD_SELECTED_STEP_1000"
 model,tokenizer,_=load_model(model_cfg,checkpoint,device,expected_step=step,expected_epoch=epoch); model.eval(); model.requires_grad_(False); core=core_model(model)
 backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=400)
 dataset=UnifiedForensicsDataset(ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/validation_manifest",tokenizer,
   model_cfg["model"]["vision_tower"],split="test",datasets_root=CFG["data"]["datasets_root"],synthscars_root=CFG["data"]["synthscars_root"],
   image_size=int(model_cfg["model"]["image_size"]),target_protocol="phrase_aligned")
 predictions={r["sample_id"]:r for r in rows(pred_path)}; fake=[(i,r) for i,r in enumerate(dataset.rows) if int(r["class_label"])==1]
 if set(predictions)!={r["sample_id"] for _,r in fake}: raise RuntimeError("prediction/validation Fake set mismatch")
 output=OUT/f"evaluation/representations/{name}.jsonl"
 if output.exists(): raise RuntimeError(f"refusing to overwrite {output}")
 teacher={}
 if a.role=="AOGD_AUTO":
  state=torch.load(TEACHER_CACHE,map_location="cpu")
  if state["source_checkpoint_sha256"]!=CFG["source"]["checkpoint_sha256"]: raise RuntimeError("teacher cache source mismatch")
  teacher=state["vectors"]
 started=time.time(); eligible=0
 with torch.no_grad():
  for position,(index,row) in enumerate(fake,1):
   sample=dataset[index]; sid=sample["sample_id"]; prediction=predictions[sid]; generated=prediction["generated_token_ids"]
   valid=generated.count(model.seg_token_idx)==1 and generated.index(model.seg_token_idx)>0
   record={"sample_id":sid,"role":name,"autonomous_representation_eligible":valid}
   if a.role=="P1_TEACHER_AND_AUTO":
    oracle_batch=backend._batch(sample,backend.tf_phrase_content(sample),question=UNIFIED_FORENSICS_QUESTION)
    oracle_raw,oracle_projected,_,_=capture_representations(core,oracle_batch)
    teacher[sid]={"raw_4096d":oracle_raw.detach().to(dtype=torch.float16).cpu(),
                  "projected_256d":oracle_projected.detach().to(dtype=torch.float16).cpu()}
   if valid:
    replay={"full_input_token_ids":prediction["prompt_token_ids"]+generated}
    batch=replay_batch(backend,sample,replay,UNIFIED_FORENSICS_QUESTION)
    auto_raw,auto_projected,_,_=capture_representations(core,batch); oracle=teacher[sid]
    record.update({"raw_4096d_cosine_gap":float(cosine_loss(auto_raw,oracle["raw_4096d"].to(auto_raw.device))),
                   "projected_256d_cosine_gap":float(cosine_loss(auto_projected,oracle["projected_256d"].to(auto_projected.device)))})
    eligible+=1
   append(output,record)
   if position<=5 or position%50==0: print(json.dumps({"role":name,"position":position,"total":len(fake),"eligible":eligible,"sec_per_sample":(time.time()-started)/position}),flush=True)
 if a.role=="P1_TEACHER_AND_AUTO":
  TEACHER_CACHE.parent.mkdir(parents=True,exist_ok=True)
  torch.save({"source_checkpoint_sha256":CFG["source"]["checkpoint_sha256"],"population":"internal_validation_fake_1106",
              "trajectory":"authoritative_full_canonical_tf","vectors":teacher},TEACHER_CACHE)
  dump(OUT/"evaluation/representations/teacher_cache_manifest.json",{"path":str(TEACHER_CACHE),"sha256":file_sha256(TEACHER_CACHE),"count":len(teacher),"source_checkpoint_sha256":CFG["source"]["checkpoint_sha256"]})
 dump(OUT/f"evaluation/representations/{name}_run.json",{"status":"COMPLETE","role":name,"checkpoint":str(checkpoint),"checkpoint_sha256":file_sha256(checkpoint),
      "sample_count":len(fake),"eligible_count":eligible,"eligible_rate":eligible/len(fake),"elapsed_seconds":time.time()-started,"output":str(output)})

if __name__=="__main__": main()
