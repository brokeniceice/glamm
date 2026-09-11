#!/usr/bin/env python3
"""Phase 6D.2 frozen LLM [CLS] head-capacity audit."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from pathlib import Path
import cv2, numpy as np, torch, yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.final_eval_classification import ExternalClassificationDataset
from scripts.phase2a_final_evaluate import load_model
from model.rine_on_c1 import RINEOnHFCLIP

OUT=ROOT/'outputs/phase6d2_cls_head_capacity'; P1=ROOT/'checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt'
TRAIN=ROOT/'outputs/phase6b2_fusion/features/train/features.pt'; VAL=ROOT/'outputs/phase6b2_fusion/features/val/features.pt'
GEN=ROOT/'outputs/phase6b3_fusion_ood/features/genimage/features.pt'; GEN_RINE=ROOT/'outputs/phase6b7_rine_ood/predictions/genimage/predictions.pt'
INTERNAL_MAN=ROOT/'datasets/Internal2208/manifests/eval_manifest.jsonl'; GEN_MAN=ROOT/'datasets/GenImage/manifests/eval_manifest.jsonl'
RINE_VAL=ROOT/'outputs/phase6d1_authenticity_audit/rine_raw_logits.pt'
RINE_CKPT=ROOT/'outputs/phase6b6_rine_training/selected_checkpoint.pt'; CLIP=ROOT/'checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
P1_SHA='fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326'
WIDTHS={'H1':128,'H2':512,'H3':2048}; LRS=[1e-4,3e-4,1e-3]; SEEDS=[3407,3408,3409]; EPOCHS=10; BATCH=256
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ch(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def metric(y,z):
 y=np.asarray(y);z=np.asarray(z);q=z>0;tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());tp=int(((y==1)&q).sum());fn=int(((y==1)&~q).sum())
 return {'n':len(y),'accuracy':float((q==y).mean()),'roc_auc':float(roc_auc_score(y,z)),'fake_recall':tp/(tp+fn),'tnr':tn/(tn+fp),'fpr':fp/(tn+fp),'f1':float(f1_score(y,q)),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':0.0}
def prepare():
 if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('refuse overwrite')
 for p in (TRAIN,VAL,GEN,GEN_RINE,RINE_VAL,INTERNAL_MAN,GEN_MAN,P1):
  if not p.exists():raise FileNotFoundError(p)
 if sha(P1)!=P1_SHA:raise RuntimeError('P1 drift')
 gm=rows(GEN_MAN); feat=torch.load(GEN,map_location='cpu',weights_only=False); rp=torch.load(GEN_RINE,map_location='cpu',weights_only=False)
 if [x['sample_id'] for x in gm]!=feat['sample_ids'] or feat['sample_ids']!=rp['sample_ids']:raise RuntimeError('GenImage identity drift')
 rng=random.Random(3407); chosen=[]
 for source in sorted({str(x['generator/source']) for x in gm}):
  for label in ('Real','Fake'):
   pool=[i for i,x in enumerate(gm) if str(x['generator/source'])==source and x['label']==label]
   if len(pool)<160:raise RuntimeError((source,label,len(pool)))
   chosen += rng.sample(pool,160)
 chosen=sorted(chosen); subset=[]
 for i in chosen:
  x=dict(gm[i]);x['source_index']=i;subset.append(x)
 manifest=OUT/'dev_ood_genimage_2560.jsonl';manifest.parent.mkdir(parents=True)
 manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in subset))
 protocol={'schema':'phase6d2_protocol_v1','status':'FROZEN_BEFORE_TRAINING','arms':{'H0':'existing Linear(4096,2)',**{k:f'4096->{v}->ReLU->2' for k,v in WIDTHS.items()}},
  'data':{'train':{'path':str(TRAIN.resolve()),'n':17672,'sha256':sha(TRAIN)},'validation':{'path':str(VAL.resolve()),'n':2212,'sha256':sha(VAL)},'internal_test':{'manifest':str(INTERNAL_MAN.resolve()),'n':2208,'sha256':sha(INTERNAL_MAN)},'dev_ood':{'label':'DEV-OOD DIAGNOSTIC','manifest':str(manifest.resolve()),'n':2560,'sha256':sha(manifest),'seed':3407,'sampling':'160 Real + 160 Fake independently per each of 8 GenImage generators','source_indices_sha256':ch(chosen)}},
  'training':{'hidden_cache_frozen':True,'optimizer':'AdamW','weight_decay':0.0,'lr_candidates':LRS,'epochs':EPOCHS,'batch_size':BATCH,'seeds':SEEDS,'loss':'cross_entropy','trainable':'classification head only','selector':'per arm/seed maximum validation ROC-AUC; tie Accuracy; tie earlier epoch; tie lower LR'},
  'selection':{'best_internal_head':'highest 3-seed mean validation ROC-AUC; tie Accuracy; tie smaller width','best_rine_complement_head':'most validation RINE-error rescues; tie validation Accuracy; tie ROC-AUC; tie smaller width','stable_gain':'one larger head strictly improves both Accuracy and ROC-AUC over H0 on validation, internal test, and DEV-OOD','dev_ood_gain':'BEST_INTERNAL_HEAD strictly improves both Accuracy and ROC-AUC over H0','bottleneck_no':'all H1/H2/H3 within abs Accuracy<=0.001 and abs ROC-AUC<=0.001 of H0 on all three splits; otherwise INCONCLUSIVE','allow_fusion':'YES iff bottleneck=YES and best complement rescues more validation RINE errors than H0'},
  'checkpoints':{'P1':{'path':str(P1.resolve()),'sha256':P1_SHA,'epoch':7,'step':3500},'RINE':{'validation':str(RINE_VAL.resolve()),'genimage':str(GEN_RINE.resolve())}},
  'firewall':{'LLM_update':False,'LoRA_update':False,'vision_update':False,'localization_update':False,'RINE_update':False,'fusion':False,'calibration':False,'OOD_threshold_tuning':False,'DEV_OOD_used_for_selection':False}}
 atomic(OUT/'protocol.json',protocol);atomic(OUT/'status.json',{'status':'PREPARED'});print(json.dumps(protocol,indent=2))
class Head(torch.nn.Module):
 def __init__(self,w):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(4096,w),torch.nn.ReLU(),torch.nn.Linear(w,2))
 def forward(self,x):return self.net(x)
def train(device_name):
 protocol=json.load(open(OUT/'protocol.json'));tr=torch.load(TRAIN,map_location='cpu',weights_only=False);va=torch.load(VAL,map_location='cpu',weights_only=False)
 if len(tr['labels'])!=17672 or len(va['labels'])!=2212:raise RuntimeError('cache drift')
 device=torch.device(device_name);torch.cuda.set_device(device);X=tr['llm_hidden'].float();y=tr['labels'];Xv=va['llm_hidden'].float();yv=va['labels']; hist={}
 croot=OUT/'checkpoints';croot.mkdir(parents=True,exist_ok=True);atomic(OUT/'status_train.json',{'status':'RUNNING','pid':os.getpid()})
 for arm,w in WIDTHS.items():
  hist[arm]={}
  for seed in SEEDS:
   best=None;allrows=[]
   for lr in LRS:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    m=Head(w).to(device);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0);gen=torch.Generator().manual_seed(seed)
    for ep in range(1,EPOCHS+1):
     m.train();order=torch.randperm(len(y),generator=gen)
     for st in range(0,len(y),BATCH):
      ix=order[st:st+BATCH];opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(X[ix].to(device)),y[ix].to(device));loss.backward();opt.step()
     m.eval();zz=[]
     with torch.inference_mode():
      for st in range(0,len(yv),1024):zz.append(m(Xv[st:st+1024].to(device)).float().cpu())
     z=torch.cat(zz);mm=metric(yv.numpy(),(z[:,1]-z[:,0]).numpy());row={'lr':lr,'epoch':ep,'metrics':mm};allrows.append(row)
     key=(mm['roc_auc'],mm['accuracy'],-ep,-lr)
     if best is None or key>best[0]:best=(key,{k:v.detach().cpu() for k,v in m.state_dict().items()},row)
   path=croot/f'{arm}_seed{seed}.pt';torch.save({'schema':'phase6d2_head_v1','arm':arm,'width':w,'seed':seed,'state_dict':best[1],'selected':best[2],'protocol_sha256':sha(OUT/'protocol.json')},path)
   hist[arm][str(seed)]={'selected':best[2],'all_candidates':allrows,'checkpoint':str(path.resolve()),'sha256':sha(path)}
   print(json.dumps({'arm':arm,'seed':seed,'selected':best[2]}),flush=True);atomic(OUT/'status_train.json',{'status':'RUNNING','arm':arm,'seed':seed})
 atomic(OUT/'training_results.json',{'schema':'phase6d2_training_v1','status':'COMPLETE','history':hist});atomic(OUT/'status_train.json',{'status':'COMPLETE'})
def extract_test(device_name,batch):
 cfg=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text());dev=torch.device(device_name);torch.cuda.set_device(dev);model,tok,meta=load_model(cfg,P1,dev,expected_step=3500,expected_epoch=7);model.eval();conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1'];backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=1);source=rows(INTERNAL_MAN);processor=__import__('transformers').CLIPImageProcessor.from_pretrained(cfg['model']['vision_tower'],local_files_only=True);data=ExternalClassificationDataset([{'sample_id':r['sample_id'],'image_path':r['image_path'],'class_label':1 if r['label']=='Fake' else 0,'source':r['generator/source']} for r in source],processor);cap=[];hook=model.classification_head.register_forward_pre_hook(lambda m,i:cap.append(i[0].detach()));hs=[];zs=[]
 atomic(OUT/'status_test_extract.json',{'status':'RUNNING','pid':os.getpid(),'done':0})
 try:
  with torch.inference_mode():
   for st in range(0,len(data),batch):
    ss=[data[i] for i in range(st,min(st+batch,len(data)))];cap.clear();bt=backend._batch_many(ss,'');bt['grounding_enc_images']=None;o=model.model_forward(**bt)
    if len(cap)!=1:
     raise RuntimeError('hook')
    hs.append(cap[0].cpu().half());zs.append(o['cls_logits'].float().cpu());done=min(st+batch,len(data))
    if done%128<batch or done==len(data):print(json.dumps({'test_done':done,'total':len(data)}),flush=True);atomic(OUT/'status_test_extract.json',{'status':'RUNNING','done':done})
 finally:hook.remove()
 out={'sample_ids':[r['sample_id'] for r in source],'labels':torch.tensor([1 if r['label']=='Fake' else 0 for r in source]),'llm_hidden':torch.cat(hs),'H0_logits':torch.cat(zs),'p1':meta};p=OUT/'internal_test_features.pt';torch.save(out,p);atomic(OUT/'status_test_extract.json',{'status':'COMPLETE','n':len(source),'sha256':sha(p)})
def extract_rine_test(device_name,batch):
 source=rows(INTERNAL_MAN);dev=torch.device(device_name);torch.cuda.set_device(dev);proc=CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True);vision=CLIPVisionModel.from_pretrained(CLIP,local_files_only=True).to(device=dev,dtype=torch.bfloat16).eval().requires_grad_(False);model=RINEOnHFCLIP(vision).to(dev).eval();ck=torch.load(RINE_CKPT,map_location='cpu',weights_only=False);model.rine.load_state_dict(ck['rine_state_dict'],strict=True);model.requires_grad_(False)
 class Images(torch.utils.data.Dataset):
  def __len__(self):return len(source)
  def __getitem__(self,i):
   im=cv2.imread(source[i]['image_path'],cv2.IMREAD_COLOR)
   if im is None:raise OSError(source[i]['image_path'])
   return i,proc(images=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),return_tensors='pt')['pixel_values'][0]
 loader=DataLoader(Images(),batch_size=batch,shuffle=False,num_workers=8,pin_memory=True);idx=[];zs=[];atomic(OUT/'status_rine_test.json',{'status':'RUNNING','pid':os.getpid(),'done':0})
 with torch.inference_mode():
  for ii,pix in loader:
   z,_,_=model(pix.to(device=dev,dtype=torch.bfloat16,non_blocking=True));idx+=ii.tolist();zs.append(z[:,0].float().cpu());atomic(OUT/'status_rine_test.json',{'status':'RUNNING','done':len(idx)})
 if idx!=list(range(len(source))):raise RuntimeError('RINE test order')
 out={'sample_ids':[r['sample_id'] for r in source],'labels':torch.tensor([1 if r['label']=='Fake' else 0 for r in source]),'raw_binary_logits':torch.cat(zs)};p=OUT/'internal_test_rine.pt';torch.save(out,p);atomic(OUT/'status_rine_test.json',{'status':'COMPLETE','n':len(source),'sha256':sha(p)})
def eval_heads(feat,checkpoints):
 y=feat['labels'].numpy();res={};raw={}
 z=feat.get('H0_logits',feat.get('F0_logits',feat.get('llm_logits'))).float();raw['H0']=(z[:,1]-z[:,0]).numpy();res['H0']={'metrics':metric(y,raw['H0'])}
 X=feat['llm_hidden'].float()
 for arm,w in WIDTHS.items():
  mets=[];scores=[]
  for seed in SEEDS:
   c=torch.load(checkpoints[arm][str(seed)]['checkpoint'],map_location='cpu',weights_only=False);m=Head(w);m.load_state_dict(c['state_dict']);m.eval();zz=[]
   with torch.inference_mode():
    for st in range(0,len(y),1024):zz.append(m(X[st:st+1024]))
   zz=torch.cat(zz);s=(zz[:,1]-zz[:,0]).numpy();scores.append(s);mets.append(metric(y,s))
  raw[arm]=np.mean(scores,axis=0);res[arm]={'seed_metrics':mets,'ensemble_metrics':metric(y,raw[arm]),'mean_seed_metrics':{k:float(np.mean([m[k] for m in mets])) for k in ('accuracy','roc_auc','fake_recall','tnr','fpr','f1')}}
 return y,res,raw
def comp(y,head,rine):
 hp=head>0;rp=rine>0;hc=hp==y;rc=rp==y;out={}
 for n,t in [('all',np.ones(len(y),bool)),('Real',y==0),('Fake',y==1)]:
  out[n]={'n':int(t.sum()),'both_correct':int((t&hc&rc).sum()),'head_only_correct':int((t&hc&~rc).sum()),'RINE_only_correct':int((t&~hc&rc).sum()),'both_wrong':int((t&~hc&~rc).sum()),'disagreement_rate':float((t&(hp!=rp)).sum()/t.sum())}
 out['correlation']={'pearson':float(pearsonr(head,rine).statistic),'spearman':float(spearmanr(head,rine).statistic)};return out
def finalize():
 protocol=json.load(open(OUT/'protocol.json'));tr=json.load(open(OUT/'training_results.json'));ck=tr['history'];val=torch.load(VAL,map_location='cpu',weights_only=False);test=torch.load(OUT/'internal_test_features.pt',map_location='cpu',weights_only=False);gen=torch.load(GEN,map_location='cpu',weights_only=False);subset=rows(OUT/'dev_ood_genimage_2560.jsonl');ix=[r['source_index'] for r in subset];dev={k:(v[ix] if torch.is_tensor(v) else [v[i] for i in ix]) for k,v in gen.items()}
 split={};raws={};ys={}
 for n,f in [('validation',val),('internal_test',test),('dev_ood',dev)]:ys[n],split[n],raws[n]=eval_heads(f,ck)
 # Selection is validation-only and uses mean across seeds.
 best=min(WIDTHS,key=lambda a:(-split['validation'][a]['mean_seed_metrics']['roc_auc'],-split['validation'][a]['mean_seed_metrics']['accuracy'],WIDTHS[a]))
 rv=torch.load(RINE_VAL,map_location='cpu',weights_only=False);rscore_val=rv['raw_binary_logits'].numpy();rt=torch.load(OUT/'internal_test_rine.pt',map_location='cpu',weights_only=False);rscore_test=rt['raw_binary_logits'].numpy();gr=torch.load(GEN_RINE,map_location='cpu',weights_only=False);rscore_dev=np.log(np.clip(gr['prob_fake'][ix].numpy(),1e-12,1-1e-12)/np.clip(1-gr['prob_fake'][ix].numpy(),1e-12,1))
 if rt['sample_ids']!=test['sample_ids'] or not torch.equal(rt['labels'],test['labels']):raise RuntimeError('RINE internal test alignment')
 comps={}
 for n,rs in [('validation',rscore_val),('internal_test',rscore_test),('dev_ood',rscore_dev)]:comps[n]={a:comp(ys[n],raws[n][a],rs) for a in ('H0',*WIDTHS)}
 rescue={a:comps['validation'][a]['all']['head_only_correct'] for a in ('H0',*WIDTHS)}
 bestcomp=min(('H0',*WIDTHS),key=lambda a:(-rescue[a],-((split['validation'][a].get('mean_seed_metrics') or split['validation'][a]['metrics'])['accuracy']),-((split['validation'][a].get('mean_seed_metrics') or split['validation'][a]['metrics'])['roc_auc']),WIDTHS.get(a,0)))
 def mm(n,a):return split[n][a].get('mean_seed_metrics') or split[n][a]['metrics']
 stable=[a for a in WIDTHS if all(mm(n,a)['accuracy']>mm(n,'H0')['accuracy'] and mm(n,a)['roc_auc']>mm(n,'H0')['roc_auc'] for n in split)]
 close=all(abs(mm(n,a)['accuracy']-mm(n,'H0')['accuracy'])<=.001 and abs(mm(n,a)['roc_auc']-mm(n,'H0')['roc_auc'])<=.001 for n in split for a in WIDTHS)
 bottleneck='YES' if stable else ('NO' if close else 'INCONCLUSIVE');ood='YES' if mm('dev_ood',best)['accuracy']>mm('dev_ood','H0')['accuracy'] and mm('dev_ood',best)['roc_auc']>mm('dev_ood','H0')['roc_auc'] else 'NO';allow='YES' if bottleneck=='YES' and rescue[bestcomp]>rescue['H0'] else 'NO'
 gates={'LLM_CLS_HEAD_BOTTLENECK':bottleneck,'BEST_INTERNAL_HEAD':best,'DEV_OOD_GENERALIZATION_GAIN':ood,'BEST_RINE_COMPLEMENT_HEAD':bestcomp,'ALLOW_RINE_LLM_FUSION_NEXT':allow}
 result={'schema':'phase6d2_results_v1','status':'COMPLETE','metrics':split,'RINE_complementarity':comps,'validation_RINE_error_rescue':rescue,'stable_gain_heads':stable,'gates':gates,'protocol_sha256':sha(OUT/'protocol.json'),'firewall':protocol['firewall']};atomic(OUT/'results.json',result);report(result);atomic(OUT/'status.json',{'status':'COMPLETE','gates':gates})
def report(r):
 protocol=json.load(open(OUT/'protocol.json'));training=json.load(open(OUT/'training_results.json'))
 lines=['# Phase 6D.2 — Frozen LLM [CLS] Head Capacity Audit','','LLM、LoRA、vision、localization 与 RINE 全冻结；仅训练 H1/H2/H3 classification head。DEV-OOD 未参与 LR、epoch、width 或 checkpoint 选择。','',
  '## Frozen contract','',
  f"- hidden cache: Phase 6B.2 train `{protocol['data']['train']['n']}` / validation `{protocol['data']['validation']['n']}`，storage FP16，所有 arms 完全共享。",
  f"- optimizer: AdamW；LR candidates `{protocol['training']['lr_candidates']}`；epoch budget `{protocol['training']['epochs']}`；batch `{protocol['training']['batch_size']}`；seeds `{protocol['training']['seeds']}`。",
  '- selector: 每个 arm/seed 按 validation ROC-AUC、Accuracy、较早 epoch、较小 LR 依次选择。',
  f"- DEV-OOD DIAGNOSTIC: `{protocol['data']['dev_ood']['n']}`，8 generators 各 160 Real + 160 Fake；manifest `{protocol['data']['dev_ood']['manifest']}`，SHA256 `{protocol['data']['dev_ood']['sha256']}`。",'',
  '### Selected LR/epoch per seed','',
  '| Arm | Seed | LR | Epoch | Validation Accuracy | Validation ROC-AUC |','|---|---:|---:|---:|---:|---:|']
 for arm,seeds in training['history'].items():
  for seed,x in seeds.items():
   q=x['selected'];lines.append(f"| {arm} | {seed} | {q['lr']:.4g} | {q['epoch']} | {q['metrics']['accuracy']:.6f} | {q['metrics']['roc_auc']:.6f} |")
 for s,v in r['metrics'].items():
  lines += ['',f'## {s}','','| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |','|---|---:|---:|---:|---:|---:|---:|']
  for a,x in v.items():
   m=x.get('mean_seed_metrics') or x['metrics'];lines.append(f"| {a} | {m['accuracy']:.6f} | {m['roc_auc']:.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
 lines += ['','## RINE complementarity','','Validation RINE-error rescues: '+', '.join(f'`{a}={n}`' for a,n in r['validation_RINE_error_rescue'].items())+'.']
 for s,arms in r['RINE_complementarity'].items():
  lines += ['',f'### {s}','','| Head | Both correct | Head only | RINE only | Both wrong | Head-only Real/Fake | RINE-only Real/Fake | Disagreement | Pearson | Spearman |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for a,x in arms.items():lines.append(f"| {a} | {x['all']['both_correct']} | {x['all']['head_only_correct']} | {x['all']['RINE_only_correct']} | {x['all']['both_wrong']} | {x['Real']['head_only_correct']}/{x['Fake']['head_only_correct']} | {x['Real']['RINE_only_correct']}/{x['Fake']['RINE_only_correct']} | {x['all']['disagreement_rate']:.6f} | {x['correlation']['pearson']:.6f} | {x['correlation']['spearman']:.6f} |")
 lines += ['','## Gates','']+[f'`{k} = {v}`' for k,v in r['gates'].items()]+['','阶段完成并 STOP。'];(ROOT/'docs/phase6d2_llm_cls_head_capacity.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','train','extract-test','extract-rine-test','finalize'));p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='train':train(a.device)
 elif a.command=='extract-test':extract_test(a.device,a.batch_size)
 elif a.command=='extract-rine-test':extract_rine_test(a.device,a.batch_size)
 else:finalize()
