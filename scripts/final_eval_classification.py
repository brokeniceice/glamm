#!/usr/bin/env python3
"""Resumable frozen classification on final official manifests."""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, torch, yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from transformers import CLIPImageProcessor

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from scripts.phase2c_external_evaluate import ExternalClassificationDataset

MANIFESTS={
 "internal":ROOT/"datasets/Internal2208/manifests/eval_manifest.jsonl",
 "aigi_holmes":ROOT/"datasets/AIGI-Holmes/manifests/eval_manifest.jsonl",
 "genimage":ROOT/"datasets/GenImage/manifests/eval_manifest.jsonl",
 "loki":ROOT/"datasets/LOKI/manifests/classification_eval_manifest.jsonl",
 "raise998":ROOT/"datasets/RAISE/manifests/eval_manifest.jsonl",
}
P1=Path("/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt")
LEGION_CLS=Path("/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/merged_cls")
OFFICIAL=ROOT/"external/LEGION_official"
CLIP=Path("/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
SAM=Path("/data/yz/myLISA_storage/checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth")

def now(): return datetime.now(timezone.utc).isoformat()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def append(p,vs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:
  for v in vs:f.write(json.dumps(v,ensure_ascii=False)+'\n')
  f.flush()
def scope(name):
 out=[]
 for r in rows(MANIFESTS[name]):
  out.append({"sample_id":r["sample_id"],"image_path":r["image_path"],"gt":1 if r["label"]=="Fake" else 0,
              "source":r.get("generator/source"),"dataset":name})
 if len(out)!=len({r['sample_id'] for r in out}):raise RuntimeError('duplicate sample ids')
 if name=='genimage':
  expected={g:{0:6000,1:6000} for g in ('adm','biggan','glide','midjourney','sdv4','vqdm','wukong')}
  expected['sdv5']={0:8000,1:8000}
  counts=Counter((r['source'],r['gt']) for r in out)
  observed={g:{label:counts[(g,label)] for label in (0,1)} for g in expected}
  unknown=sorted(set(str(r['source']) for r in out)-set(expected))
  if observed!=expected or unknown:
   raise RuntimeError(f'GenImage generator scope mismatch: observed={observed}, unknown={unknown}')
 return out
def binary_metrics(rs):
 y=np.array([r['gt'] for r in rs]);p=np.array([r['prob_fake'] for r in rs]);pred=(p>=.5).astype(int)
 tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
 pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
 out={"n":len(y),"real":int((y==0).sum()),"fake":int((y==1).sum()),"accuracy":float((y==pred).mean()),"precision":float(pr),"recall":float(re),"specificity_tnr":float(tn/max(1,tn+fp)),"fpr":float(fp/max(1,tn+fp)),"f1":float(f1),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"threshold":.5,"brier":float(np.mean((p-y)**2))}
 if len(set(y))==2: out.update(roc_auc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)))
 return out
def metrics(rs):
 out=binary_metrics(rs)
 per={}
 for source in sorted(set(str(r['source']) for r in rs)):
  sub=[r for r in rs if str(r['source'])==source]
  per[source]=binary_metrics(sub)
 out['per_source']=per;return out
def load_r1(device):
 cfg=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text());conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
 model,tok,_=load_model(cfg,P1,device,expected_step=3500,expected_epoch=7)
 return model,GLaMMForensicsBackend(model,tok,device=device,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=1),CLIPImageProcessor.from_pretrained(cfg['model']['vision_tower'],local_files_only=True)
def load_legion(device):
 repo=str(OFFICIAL.resolve());sys.path.insert(0,repo) if repo not in sys.path else None
 spec=importlib.util.spec_from_file_location('final_legion_cls',OFFICIAL/'scripts/cls/eval.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 saved=sys.argv;sys.argv=[saved[0],'--version',str(LEGION_CLS),'--pretrained','--precision','bf16','--vision_tower',str(CLIP),'--vision_pretrained',str(SAM)]
 try:a=m.parse_args()
 finally:sys.argv=saved
 model,_=m.load_model(a);return model.to(device).eval(),None,CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('r1','legion_retrained'),required=True);ap.add_argument('--datasets',nargs='+',choices=tuple(MANIFESTS),required=True);ap.add_argument('--output-root',type=Path,default=ROOT/'outputs/final_evaluation/classification');ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);device=torch.device(a.device);torch.cuda.set_device(device)
 status=a.output_root/a.model/'worker_status.json';state={"status":"RUNNING","model":a.model,"datasets":a.datasets,"pid":os.getpid(),"started_at_utc":now(),"batch_size":1};atomic(status,state)
 try:
  model,backend,processor=load_r1(device) if a.model=='r1' else load_legion(device)
  for name in a.datasets:
   source=scope(name);dest=a.output_root/a.model/name;predpath=dest/'predictions.jsonl';old=rows(predpath) if predpath.exists() else [];done={r['sample_id'] for r in old}
   if not done.issubset({r['sample_id'] for r in source}):raise RuntimeError('resume scope drift')
   data=ExternalClassificationDataset([{"sample_id":r['sample_id'],"image_path":r['image_path'],"class_label":r['gt'],"source":r['source']} for r in source],processor) if a.model=='r1' else None
   for index,r in enumerate(source):
    if r['sample_id'] in done:continue
    bgr=cv2.imread(r['image_path']);
    if bgr is None:raise OSError(r['image_path'])
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if a.model=='r1':
     sample=data[index];batch=backend._batch_many([sample],'');batch['grounding_enc_images']=None
     with torch.inference_mode():o=backend.model.model_forward(**batch);prob=float(torch.softmax(o['cls_logits'].float(),-1)[0,1].cpu())
    else:
     pix=processor.preprocess(rgb,return_tensors='pt')['pixel_values'].to(device=device,dtype=torch.bfloat16)
     with torch.inference_mode():prob=float(torch.softmax(model(global_enc_images=pix,inference_cls=True)['logits'].float(),-1)[0,0].cpu())
    append(predpath,[{**r,"prob_fake":prob,"pred":"fake" if prob>=.5 else "real"}])
    if (index+1)%100==0:print(json.dumps({"model":a.model,"dataset":name,"done":index+1,"total":len(source)}),flush=True)
   indexed={r['sample_id']:r for r in rows(predpath)};ordered=[indexed[r['sample_id']] for r in source]
   if len(indexed)!=len(source):raise RuntimeError('incomplete output')
   predpath.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered))
   result={"status":"COMPLETE","model":a.model,"dataset":name,"manifest":str(MANIFESTS[name]),"manifest_sha256":sha(MANIFESTS[name]),"metrics":metrics(ordered)}
   if name=='genimage':
    result['reporting_contract']={"overall":True,"per_generator":True,"generator_count":8,"threshold":.5,"required_metrics":["n","real","fake","accuracy","precision","recall","specificity_tnr","fpr","f1","roc_auc","auprc","brier","tp","tn","fp","fn"]}
   atomic(dest/'results.json',result)
  state['status']='COMPLETE'
 except BaseException as e:state.update(status='FAILED',exception_type=type(e).__name__,exception=str(e));raise
 finally:state['updated_at_utc']=now();atomic(status,state)
if __name__=='__main__':main()
