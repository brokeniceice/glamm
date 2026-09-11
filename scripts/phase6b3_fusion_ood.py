#!/usr/bin/env python3
"""Phase 6B.3 frozen fusion OOD extraction and confirmation."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np,torch,yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.final_eval_classification import MANIFESTS,ExternalClassificationDataset,scope
from scripts.phase2a_final_evaluate import load_model
OUT=ROOT/'outputs/phase6b3_fusion_ood';P1=ROOT/'checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt';F3=ROOT/'outputs/phase6b2_fusion/F3.pt';ALPHA=0.7744729518890381
DATASETS=('aigi_holmes','genimage','loki','raise998')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def atomic(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
class Fusion(torch.nn.Module):
 def __init__(self):super().__init__();self.l=torch.nn.Linear(4096,64);self.c=torch.nn.Linear(1024,64);self.m=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,2))
 def forward(self,h,c):return self.m(torch.cat([self.l(h),self.c(c)],1))
def prepare():
 p6=json.loads((ROOT/'outputs/phase6b2_fusion/results.json').read_text());assert p6['selected_candidate']=='F2-fixed' and abs(p6['alpha']-ALPHA)<1e-12
 d={}
 for n in DATASETS:
  a=rows(ROOT/f'outputs/final_evaluation/classification/r1/{n}/predictions.jsonl');b=rows(ROOT/f'outputs/final_evaluation/classification/legion_retrained/{n}/predictions.jsonl');s=scope(n)
  if [x['sample_id'] for x in a]!=[x['sample_id'] for x in b] or [x['sample_id'] for x in a]!=[x['sample_id'] for x in s]:raise RuntimeError(f'{n} order drift')
  d[n]={'n':len(s),'manifest':str(MANIFESTS[n].resolve()),'manifest_sha256':sha(MANIFESTS[n]),'F0_reuse':str((ROOT/f'outputs/final_evaluation/classification/r1/{n}/predictions.jsonl').resolve()),'F1_LEGION_reuse':str((ROOT/f'outputs/final_evaluation/classification/legion_retrained/{n}/predictions.jsonl').resolve())}
 protocol={'schema':'phase6b3_fusion_ood_protocol_v1','status':'FROZEN_BEFORE_F3_OOD_EXTRACTION','alpha':ALPHA,'F2_fixed':'0.5 each normalized logits','F3_checkpoint':str(F3.resolve()),'F3_sha256':sha(F3),'datasets':d,'positive':'Fake=1','threshold':.5,'paired':['F2-fixed_vs_F1','F2-alpha_vs_F1','F3_vs_F1'],'genimage_per_generator':True,'RAISE_primary':['tnr','fpr'],'decision_rule':'Prefer a simpler fusion that improves F1 on at least 2 of 3 mixed OOD datasets in both Accuracy and ROC-AUC, without >1 percentage point Accuracy loss on another mixed OOD set; use RAISE only as specificity safety evidence.','firewall':{'training':False,'threshold_tuning':False,'calibration':False,'dataset_alpha':False,'checkpoint_selection':False}}
 atomic(OUT/'protocol.json',protocol);print(json.dumps(protocol,indent=2))
def extract(names,device,batch,chunk):
 cfg=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text());dev=torch.device(device);torch.cuda.set_device(dev);model,tok,meta=load_model(cfg,P1,dev,expected_step=3500,expected_epoch=7)
 for p in model.parameters():p.requires_grad_(False)
 model.eval();conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1'];backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=1);capt=[];hook=model.classification_head.register_forward_pre_hook(lambda m,i:capt.append(i[0].detach()))
 try:
  for name in names:
   source=scope(name);processor=__import__('transformers').CLIPImageProcessor.from_pretrained(cfg['model']['vision_tower'],local_files_only=True);ds=ExternalClassificationDataset([{'sample_id':r['sample_id'],'image_path':r['image_path'],'class_label':r['gt'],'source':r['source']} for r in source],processor);sd=OUT/'features'/name/'shards';sd.mkdir(parents=True,exist_ok=True)
   for lo in range(0,len(ds),chunk):
    hi=min(len(ds),lo+chunk);path=sd/f'{lo:06d}_{hi:06d}.pt'
    if path.exists():continue
    hs=[];zs=[]
    for st in range(lo,hi,batch):
     samples=[ds[i] for i in range(st,min(hi,st+batch))];capt.clear();bt=backend._batch_many(samples,'');bt['grounding_enc_images']=None
     with torch.inference_mode():o=model.model_forward(**bt)
     if len(capt)!=1 or capt[0].shape[0]!=len(samples):raise RuntimeError('hidden hook failure')
     hs.append(capt[0].cpu().half());zs.append(o['cls_logits'].float().cpu())
    torch.save({'sample_ids':[r['sample_id'] for r in source[lo:hi]],'labels':torch.tensor([r['gt'] for r in source[lo:hi]]),'llm_hidden':torch.cat(hs),'F0_logits':torch.cat(zs)},path);print(json.dumps({'dataset':name,'done':hi,'total':len(ds)}),flush=True)
   parts=[torch.load(sd/f'{lo:06d}_{min(len(ds),lo+chunk):06d}.pt',map_location='cpu') for lo in range(0,len(ds),chunk)];ids=sum((x['sample_ids'] for x in parts),[])
   if ids!=[r['sample_id'] for r in source]:raise RuntimeError('consolidation order drift')
   dest=OUT/'features'/name/'features.pt';torch.save({'sample_ids':ids,'labels':torch.cat([x['labels'] for x in parts]),'llm_hidden':torch.cat([x['llm_hidden'] for x in parts]),'F0_logits':torch.cat([x['F0_logits'] for x in parts])},dest);atomic(OUT/'features'/name/'complete.json',{'status':'COMPLETE','n':len(ds),'sha256':sha(dest),'p1':meta})
 finally:hook.remove()
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5;tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());tp=int(((y==1)&q).sum());fn=int(((y==1)&~q).sum());o={'n':len(y),'real':int((y==0).sum()),'fake':int((y==1).sum()),'accuracy':float((y==q).mean()),'precision':float(precision_score(y,q,zero_division=0)),'fake_recall':float(recall_score(y,q,zero_division=0)),'tnr':tn/max(1,tn+fp),'fpr':fp/max(1,tn+fp),'f1':float(f1_score(y,q,zero_division=0)),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5}
 if len(set(y))==2:o['roc_auc']=float(roc_auc_score(y,p))
 return o
def pair(y,p,p1):
 a=(p>=.5)==y;b=(p1>=.5)==y;n10=int((a&~b).sum());n01=int((~a&b).sum());n=n10+n01;return {'fusion_correct_F1_wrong':n10,'fusion_wrong_F1_correct':n01,'disagreement':int(((p>=.5)!=(p1>=.5)).sum()),'mcnemar_exact_p':float(binomtest(min(n10,n01),n,.5).pvalue) if n else 1.0}
def finalize():
 saved=torch.load(F3,map_location='cpu');model=Fusion();model.load_state_dict(saved['state_dict']);model.eval();allres={};predroot=OUT/'predictions';predroot.mkdir(parents=True,exist_ok=True)
 for name in DATASETS:
  feat=torch.load(OUT/'features'/name/'features.pt',map_location='cpu');clip=torch.load(ROOT/f'outputs/phase6b1_external_confirmation/features/{name}/clip_cls.pt',map_location='cpu');f0=rows(ROOT/f'outputs/final_evaluation/classification/r1/{name}/predictions.jsonl');f1=rows(ROOT/f'outputs/final_evaluation/classification/legion_retrained/{name}/predictions.jsonl');ids=feat['sample_ids']
  if ids!=clip['sample_ids'] or ids!=[r['sample_id'] for r in f0] or ids!=[r['sample_id'] for r in f1]:raise RuntimeError(f'{name} identity mismatch')
  y=feat['labels'].numpy();p0=np.array([r['prob_fake'] for r in f0]);p1=np.array([r['prob_fake'] for r in f1]);eps=1e-12;l0=np.stack([np.log(np.clip(1-p0,eps,1)),np.log(np.clip(p0,eps,1))],1);l1=np.stack([np.log(np.clip(1-p1,eps,1)),np.log(np.clip(p1,eps,1))],1)
  def prob(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e[:,1]/e.sum(1)
  pf=prob(.5*l1+.5*l0);pa=prob(ALPHA*l1+(1-ALPHA)*l0);outs=[]
  with torch.inference_mode():
   for st in range(0,len(y),1024):outs.append(model(feat['llm_hidden'][st:st+1024].float(),clip['clip_cls'][st:st+1024].float()).softmax(1)[:,1])
  p3=torch.cat(outs).numpy();p0_extract=feat['F0_logits'].float().softmax(1)[:,1].numpy();probs={'F0':p0,'F1':p1,'F2-fixed':pf,'F2-alpha':pa,'F3':p3,'LEGION-retrained':p1};metrics={k:metric(y,v) for k,v in probs.items()};per={}
  if name=='genimage':
   for source in sorted(set(clip['sources'])):take=np.array([x==source for x in clip['sources']]);per[source]={k:metric(y[take],v[take]) for k,v in probs.items()}
  paired={k:pair(y,probs[k],p1) for k in ('F2-fixed','F2-alpha','F3')};allres[name]={'metrics':metrics,'paired_vs_F1':paired,'per_generator':per,'F0_extraction_parity':{'max_abs_probability_delta':float(np.max(np.abs(p0_extract-p0))),'prediction_disagreements':int(((p0_extract>=.5)!=(p0>=.5)).sum())}};torch.save({'sample_ids':ids,'labels':feat['labels'],'probabilities':{k:torch.tensor(v) for k,v in probs.items()}},predroot/f'{name}.pt')
 result={'schema':'phase6b3_fusion_ood_results_v1','status':'COMPLETE','datasets':allres,'frozen_alpha':ALPHA,'frozen_F3_epoch':saved['selected_epoch'],'decision':decision(allres),'firewall':{'training':False,'threshold_tuning':False,'calibration':False,'dataset_specific_alpha':False}}
 atomic(OUT/'results.json',result);report(result)
def decision(r):
 mixed=('aigi_holmes','genimage','loki');summary={}
 for arm in ('F0','F1','F2-fixed','F2-alpha','F3'):
  summary[arm]={'accuracy_mean':float(np.mean([r[d]['metrics'][arm]['accuracy'] for d in mixed])),'roc_auc_mean':float(np.mean([r[d]['metrics'][arm]['roc_auc'] for d in mixed])),'raise_fpr':r['raise998']['metrics'][arm]['fpr']}
 wins={arm:sum(r[d]['metrics'][arm]['accuracy']>r[d]['metrics']['F1']['accuracy'] and r[d]['metrics'][arm]['roc_auc']>r[d]['metrics']['F1']['roc_auc'] for d in mixed) for arm in ('F2-fixed','F2-alpha','F3')};eligible=[a for a,w in wins.items() if w>=2 and all(r[d]['metrics'][a]['accuracy']>=r[d]['metrics']['F1']['accuracy']-.01 for d in mixed)];best=max(eligible,key=lambda a:(summary[a]['roc_auc_mean'],summary[a]['accuracy_mean'])) if eligible else 'F1';return {'mixed_ood_macro':summary,'joint_accuracy_auc_wins_vs_F1':wins,'eligible_fusions':eligible,'recommended_classifier':best}
def report(r):
 lines=['# Phase 6B.3 — Fusion OOD Confirmation','',f"最终建议 classifier：**{r['decision']['recommended_classifier']}**。",'','## Main OOD results','']
 for d,v in r['datasets'].items():
  lines += [f'### {d}','', '| Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |','|---|---:|---:|---:|---:|---:|---:|']
  for a,m in v['metrics'].items():lines.append(f"| {a} | {m['accuracy']:.6f} | {m.get('roc_auc',float('nan')):.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
  lines.append('')
  lines += ['Paired comparison versus F1:','', '| Fusion | Fusion correct/F1 wrong | Fusion wrong/F1 correct | Prediction disagreement | McNemar p |','|---|---:|---:|---:|---:|']
  for a,m in v['paired_vs_F1'].items():lines.append(f"| {a} | {m['fusion_correct_F1_wrong']} | {m['fusion_wrong_F1_correct']} | {m['disagreement']} | {m['mcnemar_exact_p']:.8g} |")
  lines.append('')
  if d=='genimage':
   lines += ['GenImage per-generator:','', '| Generator | Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |','|---|---|---:|---:|---:|---:|---:|---:|']
   for g,arms in v['per_generator'].items():
    for a,m in arms.items():lines.append(f"| {g} | {a} | {m['accuracy']:.6f} | {m.get('roc_auc',float('nan')):.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
   lines.append('')
 lines += ['## Decision','',f"```json\n{json.dumps(r['decision'],indent=2)}\n```",'','No training, tuning, calibration, dataset-specific alpha, seed or checkpoint reselection was performed.','', '阶段状态：`COMPLETE_AND_STOPPED`。'];(ROOT/'docs/phase6b3_fusion_ood_results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','extract','finalize'));p.add_argument('--datasets',nargs='+',choices=DATASETS);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--chunk',type=int,default=256);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='extract':extract(a.datasets,a.device,a.batch_size,a.chunk)
 else:finalize()
