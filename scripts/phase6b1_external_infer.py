#!/usr/bin/env python3
"""Frozen shared-CLIP feature extraction and C1-L/C1-S inference."""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path
import cv2, torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6b1_external_confirmation'
P6B=ROOT/'outputs/phase6b_classification_attribution'
MANIFESTS={'internal':ROOT/'datasets/Internal2208/manifests/eval_manifest.jsonl','aigi_holmes':ROOT/'datasets/AIGI-Holmes/manifests/eval_manifest.jsonl','genimage':ROOT/'datasets/GenImage/manifests/eval_manifest.jsonl','loki':ROOT/'datasets/LOKI/manifests/classification_eval_manifest.jsonl','raise998':ROOT/'datasets/RAISE/manifests/eval_manifest.jsonl'}
EXPECTED={'internal':2208,'aigi_holmes':99999,'genimage':100000,'loki':2217,'raise998':998}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def tensor_hash(items):
 h=hashlib.sha256()
 for name,tensor in sorted(items):
  value=tensor.detach().contiguous().cpu();h.update(name.encode()+b'\0');h.update(str(value.dtype).encode()+b'\0');h.update(json.dumps(list(value.shape)).encode()+b'\0');h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]

class Images(Dataset):
 def __init__(self,rs):self.rs=rs
 def __len__(self):return len(self.rs)
 def __getitem__(self,i):
  p=self.rs[i]['image_path'];b=cv2.imread(p,cv2.IMREAD_COLOR)
  if b is None:raise OSError(p)
  return i,cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

def head(payload,device):
 d=payload['in_dim'];h=payload['hidden']
 m=torch.nn.Sequential(torch.nn.Linear(d,h),torch.nn.ReLU(),torch.nn.Linear(h,2))
 m.load_state_dict(payload['state_dict'],strict=True);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',choices=tuple(MANIFESTS),required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
 protocol=json.loads((OUT/'protocol.json').read_text());device=torch.device(a.device)
 if protocol['status']!='FROZEN_BEFORE_C1_EXTERNAL_INFERENCE':raise RuntimeError('protocol not frozen')
 os.environ.setdefault('HF_HOME','/data/yz/myLISA_storage/checkpoints/.hf_cache')
 rev=protocol['feature']['revision'];ident=protocol['feature']['model']
 processor=CLIPImageProcessor.from_pretrained(ident,revision=rev,local_files_only=True)
 clip=CLIPVisionModel.from_pretrained(ident,revision=rev,local_files_only=True).to(device=device,dtype=torch.bfloat16).eval()
 for p in clip.parameters():p.requires_grad_(False)
 clip_hash=tensor_hash([(f'vision_tower.{name}',value) for name,value in clip.state_dict().items()])
 if clip_hash!=protocol['feature']['parameter_sha256']:raise RuntimeError(f'CLIP parameter drift {clip_hash}')
 heads={}
 for arm in ('C1-L','C1-S'):
  for seed in (3407,3408,3409):
   path=P6B/f'checkpoints/{arm}_seed{seed}.pt';payload=torch.load(path,map_location='cpu')
   selected=[x for x in json.loads((P6B/'selected_runs.json').read_text()) if x['arm']==arm and x['seed']==seed][0]
   if sha(path)!=selected['checkpoint_sha256']:raise RuntimeError(f'checkpoint drift {path}')
   heads[(arm,seed)]=head(payload,device)
 feature_manifest={'schema':'phase6b1_external_feature_manifest_v1','status':'RUNNING','protocol_sha256':sha(OUT/'protocol.json'),'clip':protocol['feature'],'datasets':{},'shared_feature_cache':True,'failures':0}
 existing=OUT/'feature_manifest.json'
 if existing.exists():feature_manifest=json.loads(existing.read_text())
 for name in a.datasets:
  manifest=MANIFESTS[name]
  if sha(manifest)!=protocol['datasets'][name]['sha256']:raise RuntimeError(f'manifest drift {name}')
  rs=rows(manifest)
  if len(rs)!=EXPECTED[name] or len({r['sample_id'] for r in rs})!=len(rs):raise RuntimeError(f'population drift {name}')
  ids=[r['sample_id'] for r in rs];labels=torch.tensor([1 if r['label']=='Fake' else 0 for r in rs]);sources=[r.get('generator/source') for r in rs]
  cache=OUT/'features'/name/'clip_cls.pt';cache.parent.mkdir(parents=True,exist_ok=True)
  if cache.exists():
   data=torch.load(cache,map_location='cpu')
   if data['sample_ids']!=ids or not torch.equal(data['labels'],labels) or list(data['clip_cls'].shape)!=[len(rs),1024]:raise RuntimeError(f'cache drift {name}')
  else:
   feats=torch.empty((len(rs),1024),dtype=torch.float16);cursor=0
   def collate(batch):
    idx,ims=zip(*batch);return torch.tensor(idx),processor(images=list(ims),return_tensors='pt')['pixel_values']
   loader=DataLoader(Images(rs),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,collate_fn=collate,persistent_workers=a.workers>0)
   with torch.inference_mode():
    for idx,pix in loader:
     if not torch.equal(idx,torch.arange(cursor,cursor+len(idx))):raise RuntimeError('order drift')
     with torch.autocast('cuda',dtype=torch.bfloat16):f=clip(pix.to(device,non_blocking=True),output_hidden_states=True).hidden_states[-2][:,0]
     feats[cursor:cursor+len(idx)]=f.cpu().half();cursor+=len(idx)
     if cursor%2048<len(idx):print(json.dumps({'dataset':name,'done':cursor,'total':len(rs)}),flush=True)
   if cursor!=len(rs) or not torch.isfinite(feats).all():raise RuntimeError(f'extraction failure {name}')
   torch.save({'schema':'phase6b1_shared_clip_cls_v1','dataset':name,'sample_ids':ids,'labels':labels,'sources':sources,'clip_cls':feats},cache)
   data=torch.load(cache,map_location='cpu')
  feature_manifest['datasets'][name]={'n':len(rs),'manifest':str(manifest.resolve()),'manifest_sha256':sha(manifest),'sample_order_sha256':canon(ids),'label_sha256':canon(labels.tolist()),'cache':str(cache.resolve()),'cache_sha256':sha(cache),'shape':[len(rs),1024],'dtype':'float16','failures':0}
  atomic(OUT/'feature_manifest.json',feature_manifest)
  x=data['clip_cls'].float();pred_dir=OUT/'predictions'/name;pred_dir.mkdir(parents=True,exist_ok=True)
  for (arm,seed),model in heads.items():
   probs=[]
   with torch.inference_mode():
    for begin in range(0,len(x),4096):probs.append(model(x[begin:begin+4096].to(device)).softmax(1)[:,1].cpu())
   prob=torch.cat(probs)
   payload={'schema':'phase6b1_predictions_v1','dataset':name,'arm':arm,'seed':seed,'sample_ids':ids,'labels':labels,'sources':sources,'prob_fake':prob,'threshold':0.5,'feature_cache_sha256':sha(cache)}
   torch.save(payload,pred_dir/f'{arm}_seed{seed}.pt')
  print(json.dumps({'dataset':name,'status':'COMPLETE','n':len(rs)}),flush=True)
 feature_manifest['status']='COMPLETE' if set(feature_manifest['datasets'])==set(MANIFESTS) else 'PARTIAL'
 atomic(OUT/'feature_manifest.json',feature_manifest)
 print(json.dumps({'status':feature_manifest['status'],'datasets':sorted(feature_manifest['datasets'])}),flush=True)

if __name__=='__main__':main()
