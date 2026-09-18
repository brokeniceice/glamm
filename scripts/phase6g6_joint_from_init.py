#!/usr/bin/env python3
"""Phase 6G.6 adapter + Rectifier + Utility joint training from matched init."""
from __future__ import annotations
import hashlib,json,os,random,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts import phase4g1q_conditional_utility as q,phase4hc_direct_utility_arms as hc,phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4c_a_train import make_model
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import C1_SHA,load_c1_cache,c1_language_batch,phase4f_spatial_batch
from scripts.phase6f3_full_fov_frozen_replay import group_summary
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,invalid_record,load_rectifier,load_sam_runtime,mask_loss

OUT=ROOT/'outputs/phase6g6_joint_from_init';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g6_joint_from_init');SELECTED=OUT/'selected_checkpoint.pt'
FUSION_CKPT=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt';SEED,EPOCHS,BATCH,LR,WD=3407,10,8,1e-4,1e-4
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def write_rows(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def ids_hash(x):return hashlib.sha256('\n'.join(x).encode()).hexdigest()
class FusionStore:
 def __init__(self,split):
  p11,p17=cache_paths('middle',split),cache_paths('late',split);self.values=[];self.loc={};self.ids=[]
  if len(p11)!=len(p17):raise RuntimeError('fusion cache count drift')
  for a,b in zip(p11,p17):
   x,y=load_shard(a),load_shard(b);xa=[r['sample_id'] for r in x['records']];ya=[r['sample_id'] for r in y['records']]
   if xa!=ya:raise RuntimeError('fusion cache pairing drift')
   j=len(self.values);self.values.append((x['features'],y['features']))
   for i,sid in enumerate(xa):self.ids.append(sid);self.loc[sid]=(j,i)
 def batch(self,ids,device):
  a=torch.stack([self.values[self.loc[s][0]][0][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32);b=torch.stack([self.values[self.loc[s][0]][1][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32);return a,b
def load_fusion(device):
 x=torch.load(FUSION_CKPT,map_location='cpu',weights_only=False);m=CrossLayerPatchAttention(1024,8,.01);m.load_state_dict(x['fusion'],strict=True);return m.to(device).eval().requires_grad_(False),x
def evidence(fusion,adapter,store,ids,device):
 a,b=store.batch(ids,device)
 with torch.no_grad():fused=fusion(a,b)
 with torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=adapter(fused,return_features=True)
 return o['F_forensic'],o['logits']
def evaluate(utility,rectifier,sam,fusion,adapter,store,p4store,dev,cache,device):
 rec=[];utility.eval();rectifier.eval();adapter.eval();fusion.eval()
 with torch.no_grad():
  for i,sid in enumerate(dev['sample_ids']):
   if not bool(cache['valid'][i]):rec.append(invalid_record(sid,dev['original_masks'][i]));continue
   idx=torch.tensor([i]);batch,_=c1_language_batch(dev,cache,idx,p4store,sam,device);f,z=evidence(fusion,adapter,store,[sid],device);batch['F24']=f;batch['z_F24']=z;s64,_,_,sc,cc=phase4f_spatial_batch(p4store,[sid],device);valid=torch.ones(1,576,dtype=torch.bool,device=device)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=rectifier(s64,f,sc,cc,valid);u=hc.utility_forward(utility,batch)
   gate=hc.gate_to_sam_grid(u['U'],sc)*rv['support'].reshape(1,1,64,64);emb=hc.gated_embedding(s64,rv['image_embeddings'],gate)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   rec.append(metric_record(sid,inverse_sam_logits(low,dev['sam_geometries'][i]),dev['original_masks'][i]))
   if (i+1)%200==0:print(json.dumps({'stage':'VAL','done':i+1,'total':len(dev['sample_ids'])}),flush=True)
 return summarize_extended(rec),rec
def update_diag(initial,current):
 sq=base=maxabs=0.
 for k,v in current.items():
  d=(v.detach().cpu().float()-initial[k].float());sq+=float(d.square().sum());base+=float(initial[k].float().square().sum());maxabs=max(maxabs,float(d.abs().max()))
 return {'absolute_l2':sq**.5,'relative_l2':(sq/max(base,1e-30))**.5,'max_abs':maxabs}
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);hc.seed_all();OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
 if (OUT/'summary.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 fusion,fx=load_fusion(device);adapter=make_model('forensic_adapter',device);utility,_=hc.load_utility('a2',device);scale=json.load(open(ROOT/'outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json'));rectifier=load_rectifier(hd.CFG,float(scale['selected_gamma']),device);rectifier.requires_grad_(True)
 expected=json.load(open(ROOT/'outputs/phase6e2_c1_specific_r1/i2/initialization_provenance.json'))['new_r1_initialization'];init={'adapter_state_sha256':tensor_state_sha256(adapter.state_dict()),'utility_state_sha256':tensor_state_sha256(utility.state_dict()),'utility_trainable_sha256':tensor_state_sha256(hc.trainable_state(utility)),'rectifier_state_sha256':tensor_state_sha256(rectifier.state_dict())}
 if init['utility_trainable_sha256']!=expected['phase4hc_a2_random_utility_trainable_sha256'] or init['rectifier_state_sha256']!=expected['phase4f_random_rectifier_sha256']:raise RuntimeError('I2 initialization drift')
 initial={'adapter':{k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},'utility':{k:v.detach().cpu().clone() for k,v in utility.state_dict().items()},'rectifier':{k:v.detach().cpu().clone() for k,v in rectifier.state_dict().items()}}
 train,val=Phase4FStore(hd.CFG,'train'),Phase4FStore(hd.CFG,'val');ids=train.sample_ids;dev=load_dev('g0');tc=load_c1_cache('train',ids);vc=load_c1_cache('val',dev['sample_ids']);tl,vl=FusionStore('train'),FusionStore('val')
 if tl.ids!=ids or vl.ids!=dev['sample_ids']:raise RuntimeError('fusion population drift')
 data=q.load_ids(ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries'));targets=data['target64'];valid_mask=tc['valid'].bool();where={s:i for i,s in enumerate(ids)};cross={s:ids[(i+1)%len(ids)] for i,s in enumerate(ids)};perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device);sam=load_sam_runtime(hd.CFG,device)
 frozen={'sam':tensor_state_sha256(sam.state_dict()),'fusion':tensor_state_sha256(fusion.state_dict()),'heads':tensor_state_sha256(q.source_state(utility))};trainable={'adapter':sum(p.numel() for p in adapter.parameters() if p.requires_grad),'rectifier':sum(p.numel() for p in rectifier.parameters() if p.requires_grad),'utility':sum(p.numel() for p in utility.parameters() if p.requires_grad)}
 if trainable!={'adapter':469249,'rectifier':329985,'utility':371803}:raise RuntimeError(f'trainable scope drift: {trainable}')
 protocol={'schema':'phase6g6_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','base':{'C1':'epoch5/step2500','sha256':C1_SHA},'source':{'layers':[11,17],'fusion':str(FUSION_CKPT.resolve()),'fusion_sha256':file_sha256(FUSION_CKPT),'fusion_epoch':fx['epoch'],'frozen':True},'initialization':{**init,'kind':'Phase4C-A matched scratch adapter + Phase6E.2 I2 seed-3407 random Utility/Rectifier','forbidden_selected_weights_loaded':False},'trainable':trainable,'recipe':{'optimizer':'AdamW','lr':LR,'weight_decay':WD,'batch':BATCH,'epochs':EPOCHS,'scheduler':'none','grad_clip':1.0,'seed':SEED,'loss':'L_seg + L_relative + 0.5*(cross_rank+spatial_rank); no standalone adapter dense loss','selector':'internal validation canonical G0 mean IoU; tie earlier epoch','mask_threshold':0.0},'firewall':{'test':False,'official1000':False,'ood':False,'staged_block11_17':False}};dump(OUT/'protocol.json',protocol)
 params=list(adapter.parameters())+[p for p in utility.parameters() if p.requires_grad]+[p for p in rectifier.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD);history=[];updates=start=0;existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);adapter.load_state_dict(x['adapter_state']);utility.load_state_dict(x['utility_state']);rectifier.load_state_dict(x['rectifier_state']);opt.load_state_dict(x['optimizer']);start=x['epoch'];updates=x['optimizer_updates'];history=json.load(open(OUT/'training_curve.json'))
 for epoch in range(start+1,EPOCHS+1):
  began=time.time();order=list(ids);random.Random(SEED+1009*epoch).shuffle(order);s=defaultdict(float);eligible=invalid=0;adapter.train();utility.train();utility.language_source.eval();utility.forensic_source.eval();rectifier.train()
  for begin in range(0,len(order),BATCH):
   block=order[begin:begin+BATCH];pos=torch.tensor([where[x] for x in block]);idx=pos[valid_mask.index_select(0,pos)];invalid+=len(block)-len(idx);eligible+=len(idx)
   if not len(idx):continue
   bids=[ids[i] for i in idx.tolist()];batch,_=c1_language_batch(data,tc,idx,train,sam,device);f,z=evidence(fusion,adapter,tl,bids,device);batch['F24']=f;batch['z_F24']=z;cf,cz=evidence(fusion,adapter,tl,[cross[x] for x in bids],device);other=dict(batch);other['F24']=cf;other['z_F24']=cz;s64,_,mask_targets,sc,cc=phase4f_spatial_batch(train,bids,device);opt.zero_grad(set_to_none=True)
   m=hc.utility_forward(utility,batch);c=hc.utility_forward(utility,other);sh=hc.utility_forward(utility,batch,permutation=perm)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=rectifier(s64,f,sc,cc,torch.ones(len(idx),576,dtype=torch.bool,device=device))
   emb=hc.gated_embedding(s64,rv['image_embeddings'],hc.gate_to_sam_grid(m['U'],sc)*rv['support'].reshape(len(idx),1,64,64))
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   seg=mask_loss(low,mask_targets,hd.CFG);_,soft=q.target_delta({'p_L':m['p_L'],'p_F':m['p_F']},targets.index_select(0,idx).to(device));rel=q.image_balanced_loss(m['utility_logit'],soft,m['support']);cr=hc.rank_loss(m['U'],c['U'],m['support']);sr=hc.rank_loss(m['U'],sh['U'],m['support']);rank=.5*(cr+sr);loss=seg['total']+rel+rank
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   loss.backward();module_grad={'adapter':float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),float('inf'))),'rectifier':float(torch.nn.utils.clip_grad_norm_(rectifier.parameters(),float('inf'))),'utility':float(torch.nn.utils.clip_grad_norm_([p for p in utility.parameters() if p.requires_grad],float('inf')))};norm=torch.nn.utils.clip_grad_norm_(params,1.)
   if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
   opt.step();updates+=1;n=len(idx)
   for k,v in [('seg_loss',seg['total']),('relative_loss',rel),('ranking_loss',rank),('cross_rank_loss',cr),('shuffle_rank_loss',sr),('total_loss',loss)]:s[k]+=float(v.detach())*n
   for k,v in module_grad.items():s[k+'_grad_norm']+=v*n
   if updates<=3 or updates%100==0:print(json.dumps({'stage':'TRAIN','epoch':epoch,'update':updates,'loss':float(loss.detach()),'grad_norm':float(norm),'module_grad':module_grad}),flush=True)
  metric,records=evaluate(utility,rectifier,sam,fusion,adapter,vl,val,dev,vc,device);row={'epoch':epoch,'optimizer_updates':updates,**{k:s[k]/eligible for k in ('seg_loss','relative_loss','ranking_loss','cross_rank_loss','shuffle_rank_loss','total_loss','adapter_grad_norm','rectifier_grad_norm','utility_grad_norm')},'traversal_exposures':len(order),'optimization_eligible_exposures':eligible,'invalid_g0_exposures':invalid,'dev_g0_mean_iou':metric['mean_foreground_iou'],'dev_g0_mean_f1':metric['mean_foreground_f1'],'sample_order_sha256':ids_hash(order),'seconds':time.time()-began};history.append(row);dump(OUT/'training_curve.json',history);write_rows(OUT/f'validation/epoch_{epoch}.jsonl',records)
  payload={'schema':'phase6g6_checkpoint_v1','epoch':epoch,'optimizer_updates':updates,'adapter_state':{k:v.detach().cpu() for k,v in adapter.state_dict().items()},'utility_state':{k:v.detach().cpu() for k,v in utility.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in rectifier.state_dict().items()},'optimizer':opt.state_dict(),'validation_g0':metric,'initialization':init};p=CKPT/f'epoch_{epoch}.pt';t=p.with_suffix('.pt.tmp');torch.save(payload,t);os.replace(t,p);print(json.dumps({'stage':'EPOCH_COMPLETE',**row}),flush=True)
 best=max(history,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']));shutil.copy2(CKPT/f"epoch_{best['epoch']}.pt",SELECTED);chosen=torch.load(SELECTED,map_location='cpu',weights_only=False);adapter.load_state_dict(chosen['adapter_state']);utility.load_state_dict(chosen['utility_state']);rectifier.load_state_dict(chosen['rectifier_state']);selected=rows(OUT/f"validation/epoch_{best['epoch']}.jsonl")
 baseline_raw=rows(ROOT/'outputs/phase6f3_full_fov_frozen_replay/per_sample_results.jsonl');baseline=[{'sample_id':r['sample_id'],'foreground_iou':r['a0_iou'],'foreground_f1':r['a0_f1'],'tp':r['a0_tp'],'fp':r['a0_fp'],'fn':r['a0_fn']} for r in baseline_raw]
 if [r['sample_id'] for r in baseline]!=[r['sample_id'] for r in selected]:raise RuntimeError('baseline pairing drift')
 paired_rows=[{'sample_id':a['sample_id'],'a0_iou':a['foreground_iou'],'a0_f1':a['foreground_f1'],'a0_tp':a['tp'],'a0_fp':a['fp'],'a0_fn':a['fn'],'a1_iou':b['foreground_iou'],'a1_f1':b['foreground_f1'],'a1_tp':b['tp'],'a1_fp':b['fp'],'a1_fn':b['fn'],'delta_iou':b['foreground_iou']-a['foreground_iou'],'delta_f1':b['foreground_f1']-a['foreground_f1']} for a,b in zip(baseline,selected)];comparison=group_summary(paired_rows);d=comparison['paired']['iou'];decision='END_TO_END_FORENSIC_JOINT_TRAINING_SUPPORTED' if d['mean_delta']>0 and d['bootstrap_95_ci'][0]>0 else ('END_TO_END_FORENSIC_JOINT_TRAINING_NOT_STABLY_BETTER' if d['mean_delta']>0 else 'STAGED_INITIALIZATION_STILL_NEEDED')
 after={'sam':tensor_state_sha256(sam.state_dict()),'fusion':tensor_state_sha256(fusion.state_dict()),'heads':tensor_state_sha256(q.source_state(utility))}
 if after!=frozen:raise RuntimeError('frozen state drift')
 updates_diag={'adapter':update_diag(initial['adapter'],chosen['adapter_state']),'rectifier':update_diag(initial['rectifier'],chosen['rectifier_state']),'utility':update_diag(initial['utility'],chosen['utility_state'])};seg={'validation_samples':len(dev['sample_ids']),'valid_seg_samples':int(vc['valid'].sum()),'trigger_rate':float(vc['valid'].float().mean()),'A0_A1_exact':True,'basis':'same frozen C1 canonical G0 cache and sample-valid mask'};refs={'block17_matched_new_r1':json.load(open(ROOT/'outputs/phase6g1_block17_single_layer_replacement/summary.json')),'full_fov_i2':json.load(open(ROOT/'outputs/phase6f4_full_fov_i2/summary.json'))}
 dump(OUT/'selector.json',{'status':'COMPLETE','primary':'internal validation canonical G0 mean IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(SELECTED),'candidates':history,'test_used':False,'official1000_used':False,'ood_used':False});write_rows(OUT/'selected_validation_predictions.jsonl',selected);dump(OUT/'comparison.json',comparison);summary={'schema':'phase6g6_summary_v1','status':'COMPLETE_STOP','decision':decision,'selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(SELECTED),'comparison':comparison,'seg_trigger_invariance':seg,'parameter_update_diagnostics':updates_diag,'references':refs,'firewall':protocol['firewall']};dump(OUT/'summary.json',summary);protocol.update(status='COMPLETE_SELECTED_FROZEN',selected_epoch=best['epoch'],selected_checkpoint_sha256=file_sha256(SELECTED),frozen_hash_before=frozen,frozen_hash_after=after);dump(OUT/'protocol.json',protocol);render(summary)
def render(s):
 c=s['comparison'];a,b=c['A0'],c['A1'];text=f"""# Phase 6G.6 — Adapter + Rectifier + Utility Joint Training

Status: **COMPLETE STOP**. C1, CLIP, selected block11+17 fusion and SAM remained frozen. The original Phase4C-A adapter, random I2 Rectifier and random I2 Utility were optimized jointly from their matched initializations. No standalone adapter dense-head loss was added.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 current block22 new R1 | {a['mean_fg_iou']:.6f} | {a['median_fg_iou']:.6f} | {a['mean_fg_f1']:.6f} | {a['global_fg_iou']:.6f} | {a['global_fg_f1']:.6f} |
| A1 joint-from-init block11+17 | {b['mean_fg_iou']:.6f} | {b['median_fg_iou']:.6f} | {b['mean_fg_f1']:.6f} | {b['global_fg_iou']:.6f} | {b['global_fg_f1']:.6f} |

- paired: `{c['paired']}`
- SEG trigger invariance: `{s['seg_trigger_invariance']}`
- selected parameter updates: `{s['parameter_update_diagnostics']}`
- reference rows are preserved in `summary.json` and were not used for selection.

```text
{s['decision']}
```
""";(ROOT/'docs/phase6g6_joint_from_init.md').write_text(text)
if __name__=='__main__':main()
