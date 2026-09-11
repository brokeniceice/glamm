#!/usr/bin/env python3
"""Phase 6B.2 frozen P1/CLIP feature extraction and internal-only fusion."""

from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset, CANONICAL_PROMPT_SHA256
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model

OUT=ROOT/'outputs/phase6b2_fusion'; MAN=ROOT/'outputs/data_audits/unified_forensics_split_v1'
P1=ROOT/'checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt'
EXACT=ROOT/'outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def seed(s=3407):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def load_head():
 idx=json.loads((EXACT/'pytorch_model.bin.index.json').read_text())['weight_map'];shard=EXACT/idx['prediction_head.0.weight'];full=torch.load(shard,map_location='cpu')
 state={k.removeprefix('prediction_head.'):v for k,v in full.items() if k.startswith('prediction_head.')};del full
 m=torch.nn.Sequential(torch.nn.Linear(1024,2048),torch.nn.ReLU(),torch.nn.Linear(2048,2));m.load_state_dict(state);return m

def prepare():
 cfg=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text()); expected={'train':17672,'val':2212}
 manifests={}
 for s,n in expected.items():
  p=MAN/f'{s}_combined.jsonl';r=rows(p)
  if len(r)!=n or len({x['sample_id'] for x in r})!=n:raise RuntimeError('manifest drift')
  manifests[s]={'path':str(p.resolve()),'n':n,'sha256':sha(p)}
 protocol={'schema':'phase6b2_fusion_protocol_v1','status':'FROZEN_BEFORE_EXTRACTION','seed':3407,
  'baselines':{'F0':'frozen P1/R1 final LLM [CLS] 4096 -> existing Linear(2)','F1':'frozen C1-Exact checkpoint-554 CLIP CLS 1024 -> 2048 -> 2'},
  'features':{'single_online_forward':True,'llm_hidden':'classification_head pre-hook [4096]','clip_cls':'same vision forward hidden_states[-2] token0 [1024]','storage_dtype':'float16','logits_dtype':'float32'},
  'F2_fixed':{'formula':'0.5*z_clip+0.5*z_llm','logit_order':'Real=0,Fake=1'},
  'F2_alpha':{'formula':'alpha*z_clip+(1-alpha)*z_llm','fit':'single sigmoid-constrained scalar; full internal TRAIN cross-entropy; deterministic LBFGS','validation_used_for_fit':False},
  'F3_gate':'run only if F2-fixed or F2-alpha strictly exceeds F0 and F1 on both validation ROC-AUC and Accuracy',
  'F3':{'architecture':'LLM4096->64; CLIP1024->64; concat128->Linear128->ReLU->Linear2','optimizer':'AdamW','lr':1e-3,'weight_decay':0.0,'batch':64,'epochs':3,'scheduler':'cosine','selector':'max val ROC-AUC, then Accuracy, then earlier epoch'},
  'final_selector':'max validation ROC-AUC; then Accuracy; candidates within AUC 1e-4 and Accuracy 1e-3 prefer simpler fusion',
  'threshold':0.5,'positive_class':'Fake=1','manifests':manifests,'checkpoint':{'p1':str(P1.resolve()),'p1_sha256':sha(P1),'c1_exact':str(EXACT.resolve())},
  'firewall':{'internal_test':False,'external':False,'threshold_sweep':False,'F24':False,'SEG_mask_CSCU':False,'dynamic_gating':False}}
 atomic(OUT/'protocol.json',protocol);print(json.dumps(protocol,indent=2))

def extract(split,device,batch_size,chunk):
 protocol=json.loads((OUT/'protocol.json').read_text());cfg=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text());dev=torch.device(device);torch.cuda.set_device(dev)
 model,tok,meta=load_model(cfg,P1,dev,expected_step=3500,expected_epoch=7)
 for parameter in model.parameters():parameter.requires_grad_(False)
 model.eval();conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
 backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=1)
 ds=UnifiedForensicsDataset(MAN,tok,cfg['model']['vision_tower'],split=split,datasets_root=ROOT/cfg['data']['datasets_root'],synthscars_root=ROOT/cfg['data']['synthscars_root'],image_size=1024)
 expected=protocol['manifests'][split]['n']; assert len(ds)==expected
 sharddir=OUT/'features'/split/'shards';sharddir.mkdir(parents=True,exist_ok=True)
 hcap=[];ccap=[]
 hh=model.classification_head.register_forward_pre_hook(lambda m,i:hcap.append(i[0].detach()))
 vision=model.get_vision_tower().vision_tower
 def vh(m,i,o):
  if o.hidden_states is None:raise RuntimeError('vision hidden states absent')
  ccap.append(o.hidden_states[-2][:,0].detach())
 hv=vision.register_forward_hook(vh)
 try:
  for lo in range(0,expected,chunk):
   hi=min(expected,lo+chunk);path=sharddir/f'{lo:06d}_{hi:06d}.pt'
   if path.exists():continue
   hs=[];cs=[];zs=[]
   for st in range(lo,hi,batch_size):
    samples=[ds[i] for i in range(st,min(hi,st+batch_size))];hcap.clear();ccap.clear();batch=backend._batch_many(samples,'');batch['grounding_enc_images']=None
    with torch.inference_mode():o=model.model_forward(**batch)
    if len(hcap)!=1 or len(ccap)!=1 or hcap[0].shape[0]!=len(samples):raise RuntimeError('hook cardinality failure')
    hs.append(hcap[0].cpu().half());cs.append(ccap[0].cpu().half());zs.append(o['cls_logits'].float().cpu())
   source=ds.rows[lo:hi];torch.save({'split':split,'lo':lo,'hi':hi,'sample_ids':[x['sample_id'] for x in source],'labels':torch.tensor([int(x['class_label']) for x in source]),'llm_hidden':torch.cat(hs),'clip_cls':torch.cat(cs),'llm_logits':torch.cat(zs)},path)
   print(json.dumps({'split':split,'done':hi,'total':expected}),flush=True)
 finally:hh.remove();hv.remove()
 parts=[torch.load(sharddir/f'{lo:06d}_{min(expected,lo+chunk):06d}.pt',map_location='cpu') for lo in range(0,expected,chunk)]
 ids=sum((x['sample_ids'] for x in parts),[]);manifest_ids=[x['sample_id'] for x in ds.rows]
 if ids!=manifest_ids:raise RuntimeError('sample order drift')
 payload={'schema':'phase6b2_features_v1','split':split,'sample_ids':ids,'labels':torch.cat([x['labels'] for x in parts]),'llm_hidden':torch.cat([x['llm_hidden'] for x in parts]),'clip_cls':torch.cat([x['clip_cls'] for x in parts]),'llm_logits':torch.cat([x['llm_logits'] for x in parts]),'p1_checkpoint':meta,'prompt_sha256':CANONICAL_PROMPT_SHA256}
 torch.save(payload,OUT/'features'/split/'features.pt');atomic(OUT/'features'/split/'complete.json',{'status':'COMPLETE','n':expected,'feature_file_sha256':sha(OUT/'features'/split/'features.pt')})

def met(y,logits):
 p=logits.float().softmax(1)[:,1].numpy();y=y.numpy();pr=p>=.5;tn=int(((y==0)&~pr).sum());fp=int(((y==0)&pr).sum());tp=int(((y==1)&pr).sum());fn=int(((y==1)&~pr).sum())
 return {'accuracy':accuracy_score(y,pr),'roc_auc':roc_auc_score(y,p),'precision':precision_score(y,pr,zero_division=0),'fake_recall':recall_score(y,pr,zero_division=0),'tnr':tn/(tn+fp),'fpr':fp/(tn+fp),'f1':f1_score(y,pr,zero_division=0),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5}
class Fusion(torch.nn.Module):
 def __init__(self):super().__init__();self.l=torch.nn.Linear(4096,64);self.c=torch.nn.Linear(1024,64);self.m=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,2))
 def forward(self,h,c):return self.m(torch.cat([self.l(h),self.c(c)],1))

def fuse():
 seed();tr=torch.load(OUT/'features/train/features.pt',map_location='cpu');va=torch.load(OUT/'features/val/features.pt',map_location='cpu')
 # C1-Exact preserves official LEGION ordering Fake=0, Real=1; all fusion
 # tensors use the project convention Real=0, Fake=1.
 exact_tr=torch.load(OUT/'features/train/c1_exact_logits.pt',map_location='cpu');exact_va=torch.load(OUT/'features/val/c1_exact_logits.pt',map_location='cpu')
 if exact_tr['sample_ids']!=tr['sample_ids'] or exact_va['sample_ids']!=va['sample_ids']:raise RuntimeError('C1-Exact logit order drift')
 zt=exact_tr['logits_real_fake'].float();zv=exact_va['logits_real_fake'].float()
 z0t=tr['llm_logits'].float();z0v=va['llm_logits'].float();ytr=tr['labels'];yv=va['labels']
 base={'F0':met(yv,z0v),'F1':met(yv,zv)}
 p0=z0v.argmax(1);p1=zv.argmax(1);comp={}
 for name,take in [('all',torch.ones_like(yv,dtype=torch.bool)),('Real',yv==0),('Fake',yv==1)]:
  comp[name]={'n':int(take.sum()),'both_correct':int(((p0==yv)&(p1==yv)&take).sum()),'LLM_only_correct':int(((p0==yv)&(p1!=yv)&take).sum()),'CLIP_only_correct':int(((p0!=yv)&(p1==yv)&take).sum()),'both_wrong':int(((p0!=yv)&(p1!=yv)&take).sum())}
 fixed=.5*zv+.5*z0v;base['F2-fixed']=met(yv,fixed)
 theta=torch.tensor(0.,requires_grad=True);opt=torch.optim.LBFGS([theta],lr=1,max_iter=100,line_search_fn='strong_wolfe')
 def closure():opt.zero_grad();a=theta.sigmoid();loss=torch.nn.functional.cross_entropy(a*zt+(1-a)*z0t,ytr);loss.backward();return loss
 opt.step(closure);alpha=float(theta.sigmoid());za=alpha*zv+(1-alpha)*z0v;base['F2-alpha']=met(yv,za)
 bestbase=max(base['F0']['roc_auc'],base['F1']['roc_auc']);bestacc=max(base['F0']['accuracy'],base['F1']['accuracy']);gate=any(base[k]['roc_auc']>bestbase and base[k]['accuracy']>bestacc for k in ('F2-fixed','F2-alpha'))
 history=[];f3state=None
 if gate:
  dev=torch.device('cuda:0');model=Fusion().to(dev);optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0);steps=3*math.ceil(len(ytr)/64);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps);gen=torch.Generator().manual_seed(3407);cands=[]
  for ep in range(1,4):
   order=torch.randperm(len(ytr),generator=gen);model.train()
   for st in range(0,len(order),64):
    ix=order[st:st+64];optim.zero_grad();loss=torch.nn.functional.cross_entropy(model(tr['llm_hidden'][ix].float().to(dev),tr['clip_cls'][ix].float().to(dev)),ytr[ix].to(dev));loss.backward();optim.step();sched.step()
   model.eval();outs=[]
   with torch.no_grad():
    for st in range(0,len(yv),512):outs.append(model(va['llm_hidden'][st:st+512].float().to(dev),va['clip_cls'][st:st+512].float().to(dev)).cpu())
   mm=met(yv,torch.cat(outs));row={'epoch':ep,'metrics':mm};history.append(row);cands.append((mm['roc_auc'],mm['accuracy'],-ep,{k:v.cpu() for k,v in model.state_dict().items()},mm))
  sel=max(cands);f3state=sel[3];base['F3']=sel[4];torch.save({'schema':'phase6b2_F3_v1','state_dict':f3state,'selected_epoch':-sel[2],'metrics':sel[4]},OUT/'F3.pt')
 candidates=list(base);ranked=sorted(candidates,key=lambda k:(base[k]['roc_auc'],base[k]['accuracy']),reverse=True);chosen=ranked[0]
 complexity={'F0':0,'F1':0,'F2-fixed':1,'F2-alpha':2,'F3':3}
 for k in ranked:
  if base[ranked[0]]['roc_auc']-base[k]['roc_auc']<=1e-4 and base[ranked[0]]['accuracy']-base[k]['accuracy']<=1e-3 and complexity[k]<complexity[chosen]:chosen=k
 result={'schema':'phase6b2_fusion_results_v1','status':'COMPLETE','complementarity':comp,'alpha':alpha,'F3_gate_passed':gate,'F3_history':history,'metrics':base,'selected_candidate':chosen,'selection_order':ranked,'firewall':{'internal_train_val_only':True,'external':False,'threshold_tuning':False}}
 atomic(OUT/'results.json',result);torch.save({'candidate':chosen,'alpha':alpha if chosen=='F2-alpha' else None,'F3_state':f3state if chosen=='F3' else None,'metrics':base[chosen]},OUT/'best_fusion.pt');report(result)

def report(r):
 lines=['# Phase 6B.2 — CLIP CLS + LLM [CLS] Fusion','',f"结论：冻结最佳候选 `{r['selected_candidate']}`。本阶段只使用 internal train/validation，未访问 OOD。",'','## Complementarity','', '| Class | N | Both correct | LLM-only | CLIP-only | Both wrong |','|---|---:|---:|---:|---:|---:|']
 for k,v in r['complementarity'].items():lines.append(f"| {k} | {v['n']} | {v['both_correct']} | {v['LLM_only_correct']} | {v['CLIP_only_correct']} | {v['both_wrong']} |")
 lines+=['',f"Learned alpha: `{r['alpha']:.8f}`; F3 gate: `{'PASS' if r['F3_gate_passed'] else 'NO-GO'}`.",'','## Validation metrics','', '| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |','|---|---:|---:|---:|---:|---:|---:|']
 for k,v in r['metrics'].items():lines.append(f"| {k} | {v['accuracy']:.6f} | {v['roc_auc']:.6f} | {v['fake_recall']:.6f} | {v['tnr']:.6f} | {v['fpr']:.6f} | {v['f1']:.6f} |")
 lines+=['','阶段状态：`COMPLETE_AND_STOPPED_BEFORE_OOD`。'];(ROOT/'docs/phase6b2_classification_fusion_results.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','extract','fuse'));p.add_argument('--split',choices=('train','val'));p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--chunk',type=int,default=256);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='extract':extract(a.split,a.device,a.batch_size,a.chunk)
 else:fuse()
