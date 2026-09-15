#!/usr/bin/env python3
"""Two-GPU, resumable full classification-OOD evaluation for selected C1."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, torch, yaml
from sklearn.metrics import roc_auc_score
from transformers import CLIPImageProcessor

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model

OUT=Path(os.environ.get(
 'PHASE6D3_OOD_OUTPUT', str(ROOT/'outputs/phase6d3_c1/full_classification_ood')
)).resolve()
CFG=Path(os.environ.get('PHASE6D3_OOD_CONFIG',str(ROOT/'configs/phase6d3_c1_rine_conditioned_p1.yaml'))).resolve()
CKPT=Path(os.environ.get('PHASE6D3_OOD_CHECKPOINT',str(ROOT/'checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt'))).resolve()
EXPECTED_EPOCH=int(os.environ.get('PHASE6D3_OOD_EXPECTED_EPOCH','5'))
EXPECTED_STEP=int(os.environ.get('PHASE6D3_OOD_EXPECTED_STEP','2500'))
DEVICES=[x.strip() for x in os.environ.get('PHASE6D3_OOD_DEVICES','0,1').split(',') if x.strip()]
WORLD_SIZE=len(DEVICES)
ALL_MANIFESTS={
 'aigi_holmes':ROOT/'datasets/AIGI-Holmes/manifests/eval_manifest.jsonl',
 'genimage':ROOT/'datasets/GenImage/manifests/eval_manifest.jsonl',
 'loki':ROOT/'datasets/LOKI/manifests/classification_eval_manifest.jsonl',
 'raise998':ROOT/'datasets/RAISE/manifests/eval_manifest.jsonl'}
_requested=os.environ.get('PHASE6D3_OOD_DATASETS','').strip()
MANIFESTS=(
 {name:ALL_MANIFESTS[name] for name in _requested.split(',') if name}
 if _requested else ALL_MANIFESTS
)

def rows(path):return [json.loads(x) for x in Path(path).open() if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def metric(records, key='cls'):
 y=np.asarray([r['label'] for r in records],dtype=np.int64);s=np.asarray([r[f'{key}_score_fake'] for r in records]);p=(s>=.5).astype(np.int64)
 tp=int(((p==1)&(y==1)).sum());tn=int(((p==0)&(y==0)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
 precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);tnr=tn/max(1,tn+fp)
 return {'n':len(y),'real':int((y==0).sum()),'fake':int((y==1).sum()),'accuracy':float((p==y).mean()),
  'precision':precision,'fake_recall':recall,'tnr':tnr,'fpr':1-tnr,'f1':2*precision*recall/max(1e-30,precision+recall),
  'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5,'roc_auc':None if len(set(y.tolist()))<2 else float(roc_auc_score(y,s))}

def read_image(row):
 im=cv2.imread(row['image_path'],cv2.IMREAD_COLOR)
 if im is None:raise OSError(f"decode failed: {row['image_path']}")
 return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

def worker(rank, device, batch_size):
 config=yaml.safe_load(CFG.read_text());dev=torch.device(device);torch.cuda.set_device(dev)
 conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
 model,tok,meta=load_model(config,CKPT,dev,expected_step=EXPECTED_STEP,expected_epoch=EXPECTED_EPOCH)
 processor=CLIPImageProcessor.from_pretrained(config['model']['vision_tower'],local_files_only=True)
 backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,max_new_tokens=400)
 for name,manifest in MANIFESTS.items():
  all_rows=rows(manifest);assigned=[r for i,r in enumerate(all_rows) if i%WORLD_SIZE==rank]
  path=OUT/'shards'/name/f'rank{rank}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
  done={r['sample_id'] for r in rows(path)} if path.exists() else set()
  pending=[r for r in assigned if r['sample_id'] not in done]
  with path.open('a') as handle, ThreadPoolExecutor(max_workers=8) as pool:
   for start in range(0,len(pending),batch_size):
    part=pending[start:start+batch_size];images=list(pool.map(read_image,part))
    pixels=processor(images=images,return_tensors='pt')['pixel_values']
    samples=[]
    for row,pixel in zip(part,pixels):
     samples.append({'image_path':row['image_path'],'global_enc_image':pixel,'grounding_enc_image':None,'bboxes':None,
      'conversations':[''],'masks':None,'label':None,'resize':None,'questions':[],'sampled_classes':[],
      'cls_label':1 if row['label']=='Fake' else 0,'seg_valid':False,'sample_id':row['sample_id'],
      'source':row.get('generator/source'),'content_category':row.get('generator/source'),'manifest_row':row})
    batch=backend._batch_many(samples,'',question=UNIFIED_FORENSICS_QUESTION)
    batch['grounding_enc_images']=None
    with torch.inference_mode():out=model.model_forward(**batch)
    cl=out['cls_logits'].float().softmax(-1).cpu();lm=out['lm_verdict_logits'].float().softmax(-1).cpu()
    for row,cp,lp in zip(part,cl[:,1].tolist(),lm[:,1].tolist()):
     rec={'sample_id':row['sample_id'],'dataset':name,'label':1 if row['label']=='Fake' else 0,
      'label_name':row['label'],'generator':row.get('generator/source'),'cls_score_fake':cp,'lm_score_fake':lp}
     handle.write(json.dumps(rec,ensure_ascii=False)+'\n')
    handle.flush()
    if (start//batch_size+1)%25==0:print(json.dumps({'dataset':name,'rank':rank,'done':len(done)+min(start+batch_size,len(pending)),'total':len(assigned)}),flush=True)
  dump(OUT/'shards'/name/f'rank{rank}.complete.json',{'status':'COMPLETE','rank':rank,'world_size':WORLD_SIZE,'count':len(assigned),'manifest_sha256':sha(manifest),'checkpoint_sha256':meta['checkpoint_sha256']})

def finalize():
 result={'schema':'phase6d3_c1_full_classification_ood_v1','status':'COMPLETE','checkpoint':str(CKPT.resolve()),'checkpoint_sha256':sha(CKPT),
  'epoch':EXPECTED_EPOCH,'optimizer_step':EXPECTED_STEP,'prompt':'canonical unified fixed [CLS] query','threshold':.5,'datasets':{},'manifests':{}}
 for name,manifest in MANIFESTS.items():
  source=rows(manifest);byid={}
  for rank in range(WORLD_SIZE):
   complete=OUT/'shards'/name/f'rank{rank}.complete.json'
   if not complete.exists():raise RuntimeError(f'missing {complete}')
   for r in rows(OUT/'shards'/name/f'rank{rank}.jsonl'):
    if r['sample_id'] in byid:raise RuntimeError(f"duplicate {r['sample_id']}")
    byid[r['sample_id']]=r
  ids=[r['sample_id'] for r in source]
  if set(ids)!=set(byid) or len(ids)!=len(byid):raise RuntimeError(f'{name} identity mismatch')
  merged=[byid[x] for x in ids];mp=OUT/'predictions'/f'{name}.jsonl';mp.parent.mkdir(parents=True,exist_ok=True)
  with mp.open('w') as f:
   for r in merged:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  entry={'classification_head':metric(merged,'cls'),'lm_verdict':metric(merged,'lm'),'prediction_file':str(mp.resolve()),'prediction_sha256':sha(mp)}
  if name=='genimage':entry['per_generator']={g:{'classification_head':metric([r for r in merged if r['generator']==g],'cls'),'lm_verdict':metric([r for r in merged if r['generator']==g],'lm')} for g in sorted({r['generator'] for r in merged})}
  result['datasets'][name]=entry;result['manifests'][name]={'path':str(manifest.resolve()),'sha256':sha(manifest),'count':len(source)}
 mixed=[result['datasets'][n]['classification_head'] for n in ('aigi_holmes','genimage','loki') if n in result['datasets']]
 if len(mixed)>1:
  result['mixed_ood_macro']={k:float(np.mean([m[k] for m in mixed])) for k in ('accuracy','roc_auc','fake_recall','tnr','fpr','f1')}
 if 'raise998' in result['datasets']:result['raise998_note']='Real-only; primary quantities are TNR/FPR.'
 dump(OUT/'results.json',result)

def supervisor(batch_size):
 OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol.json',{'schema':'phase6d3_full_ood_protocol_v1','checkpoint_sha256':sha(CKPT),'datasets':{k:{'manifest':str(v.resolve()),'sha256':sha(v),'count':len(rows(v))} for k,v in MANIFESTS.items()},'world_size':WORLD_SIZE,'partition':f'manifest_index_mod_{WORLD_SIZE}','batch_size_per_gpu':batch_size,'threshold':.5,'selection_or_tuning':False})
 procs=[]
 for rank,device_index in enumerate(DEVICES):
  log=(OUT/f'worker_rank{rank}.log').open('a');procs.append((subprocess.Popen([sys.executable,__file__,'--mode','worker','--rank',str(rank),'--device',f'cuda:{device_index}','--batch-size',str(batch_size)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}),log))
 failed=[]
 for p,log in procs:
  code=p.wait();log.close();failed.append(code)
 if any(failed):raise SystemExit(f'workers failed: {failed}')
 finalize()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['supervisor','worker','finalize'],required=True);ap.add_argument('--rank',type=int);ap.add_argument('--device');ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
 if a.mode=='worker':worker(a.rank,a.device,a.batch_size)
 elif a.mode=='finalize':finalize()
 else:supervisor(a.batch_size)
if __name__=='__main__':main()
