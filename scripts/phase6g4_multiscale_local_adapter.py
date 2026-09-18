#!/usr/bin/env python3
"""Phase 6G.4 matched multi-scale local forensic adapter experiment."""
from __future__ import annotations
import json, math, os, random, shutil, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from model.multiscale_local_forensic_adapter import MultiScaleLocalForensicAdapter
from scripts.phase4c_a_train import make_model
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard,dump,rows,write_rows
from tools.phase3c1 import binary_metrics,inverse_logits,paired_statistics,probe_loss
from tools.phase4c_a import summarize_extended,tensor_hash
from tools.phase4c_b import file_sha256

OUT=ROOT/'outputs/phase6g4_multiscale_local_adapter'; CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g4_multiscale_local_adapter')
FUSION=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt'; A0=ROOT/'outputs/phase6g3_forensic_adapter_audit/a0'; SEED=3407

def seed_all(): random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def pairs(split):
 a,b=cache_paths('middle',split),cache_paths('late',split)
 if len(a)!=len(b):raise RuntimeError('cache count drift')
 return list(zip(a,b))
def load(pair):
 a,b=map(load_shard,pair)
 if [r['sample_id'] for r in a['records']] != [r['sample_id'] for r in b['records']] or not torch.equal(a['targets'],b['targets']):raise RuntimeError('pair drift')
 return a,b
def source(device):
 x=torch.load(FUSION,map_location='cpu',weights_only=False);m=CrossLayerPatchAttention(1024,8,.01);m.load_state_dict(x['fusion'],strict=True);return m.to(device).eval().requires_grad_(False),x
def model(device):
 base=make_model('forensic_adapter',torch.device('cpu'))
 seed_all(); m=MultiScaleLocalForensicAdapter(.01);m.projection.load_state_dict(base.projection.state_dict());m.dense_head.load_state_dict(base.dense_head.state_dict());return m.to(device)
def fused(f,a,b,device):
 with torch.no_grad():return f(a['features'].to(device=device,dtype=torch.float32),b['features'].to(device=device,dtype=torch.float32))
def validate(m,f,device,diag=False):
 rec=[];base=residual=tokens=0.;m.eval()
 with torch.no_grad():
  for pair in pairs('val'):
   a,b=load(pair);x=fused(f,a,b,device)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=m(x,return_features=True)
   for i,r in enumerate(b['records']):rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(o['logits'][i,0].float(),r['geometry']),b['original_masks'][i].to(device))})
   if diag:
    q=o['F_base'].float();z=o['residual'].float();base+=float(q.norm(dim=1).sum());residual+=float(z.norm(dim=1).sum());tokens+=q.shape[0]*q.shape[2]*q.shape[3]
 met=summarize_extended(rec)
 if diag:met['diagnostics']={'base_patch_l2_mean':base/tokens,'residual_patch_l2_mean':residual/tokens,'residual_to_base_norm_ratio':residual/max(base,1e-12),'selected_gamma':float(m.gamma.detach())}
 return met,rec
def optimizer(m):
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999));warm,total=277,5530
 def scale(step):return float(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)
def train(m,f,device):
 OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True);opt,sched=optimizer(m);history=[];start=step=0
 existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);m.load_state_dict(x['model']);opt.load_state_dict(x['optimizer']);sched.load_state_dict(x['scheduler']);start=x['epoch'];step=x['global_step'];history=json.load(open(OUT/'history.json'))
 else:
  vm,_=validate(m,f,device);payload={'schema':'phase6g4_checkpoint_v1','epoch':0,'global_step':0,'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':vm};torch.save(payload,CKPT/'epoch_0.pt');history=[{'epoch':0,'global_step':0,'train':None,'validation':vm,'seconds':0.}];dump(OUT/'history.json',history)
 for epoch in range(start+1,11):
  began=time.time();m.train();rng=random.Random(SEED+epoch);order=pairs('train');rng.shuffle(order);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for pair in order:
   a,b=load(pair);ix=list(range(len(b['records'])));rng.shuffle(ix)
   for begin in range(0,len(ix),16):
    ids=ix[begin:begin+16];aa={**a,'features':a['features'][ids]};bb={**b,'features':b['features'][ids]};x=fused(f,aa,bb,device);target=b['targets'][ids,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16):logits=m(x)
    loss=probe_loss(logits.float(),target);loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
    opt.step();sched.step();step+=1;n=len(ids)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  vm,_=validate(m,f,device);row={'epoch':epoch,'global_step':step,'train':{k:s[k]/s['samples'] for k in ('bce','dice','total')}|{'samples':s['samples']},'validation':vm,'seconds':time.time()-began};history.append(row);dump(OUT/'history.json',history);payload={'schema':'phase6g4_checkpoint_v1','epoch':epoch,'global_step':step,'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':vm};tmp=(CKPT/f'epoch_{epoch}.pt.tmp');torch.save(payload,tmp);os.replace(tmp,CKPT/f'epoch_{epoch}.pt');print(json.dumps({'stage':'6G4_EPOCH',**row}),flush=True)
 best=max(history,key=lambda x:(x['validation']['mean_foreground_iou'],x['validation']['mean_foreground_f1']));shutil.copy2(CKPT/f"epoch_{best['epoch']}.pt",OUT/'selected.pt');x=torch.load(OUT/'selected.pt',map_location='cpu',weights_only=False);m.load_state_dict(x['model']);met,rec=validate(m,f,device,True);write_rows(OUT/'selected_predictions.jsonl',rec);dump(OUT/'selector.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(OUT/'selected.pt'),'selected_metrics':met,'candidates':history});return met,rec
def complexity(m):
 p=sum(x.numel() for x in m.parameters());base=1024*256*24*24;dw=3*64*9*24*24;pw=256*256*24*24;head=256*24*24;mac=base+dw+pw+head
 a0=make_model('forensic_adapter',torch.device('cpu'));a0p=sum(x.numel() for x in a0.parameters());a0mac=base+3*(256*9*24*24+256*256*24*24)+head
 return {'A0_parameters':a0p,'A0_MACs_per_image_approx':a0mac,'A0_FLOPs_per_image_approx':2*a0mac,'A1_parameters':p,'A1_MACs_per_image_approx':mac,'A1_FLOPs_per_image_approx':2*mac,'A1_minus_A0_parameters':p-a0p,'A1_minus_A0_MACs_per_image_approx':mac-a0mac,'convention':'one MAC equals two FLOPs; GN, GELU and residual addition excluded'}
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all();f,fx=source(device);m=model(device);before=tensor_hash(f.state_dict().items());base=make_model('forensic_adapter',torch.device('cpu'));shared={'projection_initialization_exact':tensor_hash(m.projection.state_dict().items())==tensor_hash(base.projection.state_dict().items()),'dense_head_initialization_exact':tensor_hash(m.dense_head.state_dict().items())==tensor_hash(base.dense_head.state_dict().items())}
 protocol={'schema':'phase6g4_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','fusion':{'path':str(FUSION.resolve()),'sha256':file_sha256(FUSION),'selected_epoch':fx['epoch'],'frozen':True},'A0':'exact reuse of Phase6G.3 matched original adapter','A1':'projection; 4x64 identity/dilated-DWConv(1,2,3); GN+GELU+PWConv; small-gamma residual; dense head','matched_initialization':shared,'recipe':{'epochs':10,'batch':16,'optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'warmup':277,'total_updates':5530,'loss':'2*BCE+0.5*Dice','seed':SEED,'selector':'val mean FG IoU; tie mean FG F1; epochs0..10','threshold_logit':0.},'complexity':complexity(m),'firewall':{'rectifier':False,'utility':False,'test':False,'official1000':False,'ood':False}};dump(OUT/'protocol.json',protocol);m1,r1=train(m,f,device)
 if tensor_hash(f.state_dict().items())!=before:raise RuntimeError('frozen fusion drift')
 r0=rows(A0/'selected_predictions.jsonl');m0=summarize_extended(r0)
 if [x['sample_id'] for x in r0]!=[x['sample_id'] for x in r1]:raise RuntimeError('pair order drift')
 pi=paired_statistics([x['foreground_iou'] for x in r1],[x['foreground_iou'] for x in r0]);pf=paired_statistics([x['foreground_f1'] for x in r1],[x['foreground_f1'] for x in r0]);up=json.load(open(ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/summary.json'))['comparison']['A1_attention'];supported=pi['mean_difference']>0 and pi['bootstrap_95_ci'][0]>0
 retention='FUSION_REPRESENTATION_FULLY_RETAINED_OR_IMPROVED' if m1['mean_foreground_iou']>=up['mean_foreground_iou'] else ('ADAPTER_IMPROVED_BUT_STILL_LOSES_UPSTREAM_SIGNAL' if m1['mean_foreground_iou']>m0['mean_foreground_iou'] else 'MULTISCALE_FORENSIC_ADAPTER_NOT_SUPPORTED')
 result={'schema':'phase6g4_results_v1','status':'COMPLETE_STOP','decision':'MULTISCALE_FORENSIC_ADAPTER_SUPPORTED' if supported else 'MULTISCALE_FORENSIC_ADAPTER_NOT_SUPPORTED','representation_retention':retention,'A0':m0,'A1':m1,'upstream_attention_linear_probe':up,'paired':{'iou':pi,'f1':pf},'complexity':protocol['complexity'],'fusion_hash_before':before,'fusion_hash_after':tensor_hash(f.state_dict().items()),'firewall':protocol['firewall']};dump(OUT/'results.json',result);render(result)
def render(x):
 lines=['# Phase 6G.4 — Multi-Scale Local Forensic Adapter','','Status: **COMPLETE STOP**. The selected block11+17 fusion was frozen. Rectifier, Utility, test, Official1000 and OOD were not accessed.','','| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |','|---|---:|---:|---:|---:|---:|']
 for n in ('A0','A1'):
  m=x[n];lines.append(f"| {n} | {m['mean_foreground_iou']:.6f} | {m['median_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_iou']:.6f} | {m['global_foreground_f1']:.6f} |")
 u=x['upstream_attention_linear_probe'];lines+=['',f"Upstream attention linear probe mean IoU: **{u['mean_foreground_iou']:.6f}**.",'',f"- IoU paired: `{x['paired']['iou']}`",f"- F1 paired: `{x['paired']['f1']}`",f"- A1 diagnostics: `{x['A1'].get('diagnostics')}`",f"- complexity: `{x['complexity']}`",'',f"```text\n{x['decision']}\n{x['representation_retention']}\n```"]
 (ROOT/'docs/phase6g4_multiscale_local_forensic_adapter.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
