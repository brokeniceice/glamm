#!/usr/bin/env python3
"""Phase 6G.17: exact-architecture real-C versus zero-C Type-III audit."""
from __future__ import annotations
import copy,json,math,os,random,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase4g1q_conditional_utility as q, phase4hc_direct_utility_arms as hc, phase4hd_rectifier_unfreeze_control as hd, phase6g10_train_arm as g10
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import (BATCH,CLIP,EPOCHS,LR,SEED,WD,bootstrap_mean,count_trainable,dump,evaluate as _unused,
    gate_stats,load_collapsed,rect_forward,rows,seed_all,write_rows)
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch,load_c1_cache,phase4f_spatial_batch
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid,gated_embedding
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import inverse_sam_logits,metric_record
from tools.phase4f import invalid_record,mask_loss

OUT=ROOT/'outputs/phase6g17_type3_incremental'
DOC=ROOT/'docs/phase6g17_type3_incremental_correction.md'
ARMS=('H0_zero_C','H1_real_C')

class HybridUtility(nn.Module):
    """Current source-aware CSCU plus a zero-init correction-dependent logit residual."""
    def __init__(self,base):
        super().__init__();self.base=base;w=64
        self.correction=nn.Sequential(nn.Conv2d(256,w,1,bias=False),nn.GroupNorm(8,w),nn.GELU())
        self.delta=nn.Sequential(nn.Conv2d(2*w,w,3,padding=1),nn.GroupNorm(8,w),nn.GELU(),nn.Conv2d(w,1,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)

    def forward(self,batch,correction,support,*,permutation=None):
        src=hc.utility_forward(self.base,batch,permutation=permutation)
        ch=self.correction(correction.detach().float());delta=self.delta(torch.cat((src['utility_hidden'].float(),ch),1))
        logit=src['utility_logit'].float()+delta;u=logit.sigmoid()*support.detach().float()
        return {**src,'utility_logit':logit,'U':u,'correction_hidden':ch,'delta_logit':delta}

def init_model(device):
    seed_all();base,_=hc.load_utility('a2',device);base.train()
    for p in list(base.language_source.parameters())+list(base.forensic_source.parameters()):p.requires_grad_(False)
    return HybridUtility(base).to(device)

def context(data,cache,pos,p4,sam,s64,f,z):
    b,_=c1_language_batch(data,cache,pos,p4,sam,s64.device);b['S64']=s64;b['F24']=f;b['z_F24']=z;return b

def train_step(model,real_c,collapsed,sam,fusion,projection,head,p4,fs,data,ids,pos,cache,cross,perm,device):
    s64,_,targets,sc,cc=phase4f_spatial_batch(p4,ids,device);f=g10.fused_evidence(fusion,projection,fs,ids,device);z=head(f.float());b=context(data,cache,pos,p4,sam,s64,f,z)
    ci=cross.index_select(0,pos);cids=[data['sample_ids'][i] for i in ci.tolist()];fc=g10.fused_evidence(fusion,projection,fs,cids,device);zc=head(fc.float());bc=dict(b);bc['F24']=fc;bc['z_F24']=zc
    _,adapt,c,support=rect_forward(collapsed,s64,f,sc,cc);_,_,cx,_=rect_forward(collapsed,s64,fc,sc,cc);fp=f.flatten(2)[:,:,perm].reshape_as(f);_,_,cs,_=rect_forward(collapsed,s64,fp,sc,cc)
    if not real_c:c=torch.zeros_like(c);cx=torch.zeros_like(cx);cs=torch.zeros_like(cs)
    out=model(b,c,support);oc=model(bc,cx,support);bs=dict(b);bs['F24']=fp;bs['z_F24']=z.flatten(2)[:,:,perm].reshape_as(z);osh=model(bs,cs,support,permutation=None)
    gate=gate_to_sam_grid(out['U'],sc)*support;final=gated_embedding(s64,adapt,gate)
    with torch.autocast(device_type=device.type,enabled=False):low=sam(b['q_seg'],final.to(torch.bfloat16))
    seg=mask_loss(low,targets,hd.CFG);_,soft=q.target_delta(out,data['target64'].index_select(0,pos).to(device));rel=q.image_balanced_loss(out['utility_logit'],soft,out['support']);cr=hc.rank_loss(out['U'],oc['U'],out['support']);sr=hc.rank_loss(out['U'],osh['U'],out['support']);rank=.5*(cr+sr)
    return seg['total']+rel+rank,{'seg':seg['total'],'relative':rel,'ranking':rank}

def evaluate(model,real_c,collapsed,sam,fusion,projection,head,p4,fs,dev,cache,device,corruption=None):
    model.eval();recs=[];pixels=[];means=[];permch=torch.randperm(256,generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.no_grad():
        for i,sid in enumerate(dev['sample_ids']):
            if not bool(cache['valid'][i]):recs.append(invalid_record(sid,dev['original_masks'][i]));means.append(0.);continue
            s64,_,_,sc,cc=phase4f_spatial_batch(p4,[sid],device);f=g10.fused_evidence(fusion,projection,fs,[sid],device);z=head(f.float());b,_=c1_language_batch(dev,cache,torch.tensor([i]),p4,sam,device);b['S64']=s64;b['F24']=f;b['z_F24']=z;_,adapt,c,support=rect_forward(collapsed,s64,f,sc,cc)
            if not real_c or corruption=='zero':c=torch.zeros_like(c)
            elif corruption=='channel':c=c[:,permch]
            elif corruption=='flip':c=-c
            elif corruption=='spatial':
                mask=support[0,0].flatten().bool();idx=mask.nonzero(as_tuple=False).flatten();pm=torch.randperm(len(idx),generator=torch.Generator().manual_seed(SEED+1+i)).to(device);cf=c.flatten(2);cp=c.clone().flatten(2);cp[0,:,idx]=cf[0,:,idx.index_select(0,pm)];c=cp.reshape_as(c)
            out=model(b,c,support);gate=gate_to_sam_grid(out['U'],sc)*support;final=gated_embedding(s64,adapt,gate)
            with torch.autocast(device_type=device.type,enabled=False):low=sam(b['q_seg'],final.to(torch.bfloat16))
            row=metric_record(sid,inverse_sam_logits(low,dev['sam_geometries'][i]),dev['original_masks'][i]);row['valid_g0']=True;recs.append(row);v=gate[support.bool()].float().cpu().numpy();pixels.extend(v.tolist());means.append(float(v.mean()))
            if (i+1)%200==0:print(json.dumps({'stage':'VAL','arm':'H1' if real_c else 'H0','corruption':corruption,'done':i+1}),flush=True)
    return summarize_extended(recs),recs,{'pixels':gate_stats(pixels),'per_image':gate_stats(means)}

def train(arm,model,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device):
    root=OUT/arm
    if (root/'summary.json').exists():return
    real=arm.startswith('H1');params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD);ids=p4t.sample_ids;valid=tc['valid'].bool();where={s:i for i,s in enumerate(ids)};cross=torch.tensor([(i+1)%len(ids) for i in range(len(ids))]);perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device);hist=[];updates=0;start=1
    ck=list((root/'checkpoints').glob('epoch_*.pt')) if (root/'checkpoints').exists() else []
    if ck:
        last=max(ck,key=lambda p:int(p.stem.split('_')[-1]));st=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);start=st['epoch']+1;updates=st['updates'];hist=json.loads((root/'training_curve.json').read_text())
    for ep in range(start,EPOCHS+1):
        began=time.time();model.train();order=list(ids);random.Random(SEED+1009*ep).shuffle(order);sums=defaultdict(float);n=0
        for begin in range(0,len(order),BATCH):
            pos=torch.tensor([where[x] for x in order[begin:begin+BATCH]]);pos=pos[valid.index_select(0,pos)]
            if not len(pos):continue
            bid=[ids[i] for i in pos.tolist()];opt.zero_grad(set_to_none=True);loss,parts=train_step(model,real,collapsed,sam,fusion,projection,head,p4t,fst,data,bid,pos,tc,cross,perm,device);loss.backward();torch.nn.utils.clip_grad_norm_(params,CLIP);opt.step();updates+=1;n+=len(bid)
            for k,v in parts.items():sums[k]+=float(v.detach())*len(bid)
        met,recs,gates=evaluate(model,real,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device);row={'epoch':ep,'updates':updates,'seg_loss':sums['seg']/n,'relative_loss':sums['relative']/n,'ranking_loss':sums['ranking']/n,'total_loss':sum(sums.values())/n,'dev_g0_mean_iou':met['mean_foreground_iou'],'dev_g0_mean_f1':met['mean_foreground_f1'],'seconds':time.time()-began};hist.append(row);dump(root/'training_curve.json',hist);write_rows(root/f'validation/epoch_{ep}.jsonl',recs);dump(root/f'gate/epoch_{ep}.json',gates);(root/'checkpoints').mkdir(parents=True,exist_ok=True);torch.save({'epoch':ep,'updates':updates,'model':model.state_dict(),'optimizer':opt.state_dict()},root/f'checkpoints/epoch_{ep}.pt');print(json.dumps({'stage':'G17_EPOCH','arm':arm,**row}),flush=True)
    best=max(hist,key=lambda x:(x['dev_g0_mean_iou'],x['dev_g0_mean_f1'],-x['epoch']));sel=root/'selected_checkpoint.pt';sel.write_bytes((root/f"checkpoints/epoch_{best['epoch']}.pt").read_bytes());model.load_state_dict(torch.load(sel,map_location='cpu',weights_only=False)['model']);met,recs,gates=evaluate(model,real,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device);write_rows(root/'validation/selected.jsonl',recs);dump(root/'summary.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'metrics':met,'gate_stats':gates,'trainable_params':count_trainable(model)})

def pair(a,b):
    x=torch.tensor([r['foreground_iou'] for r in a]);y=torch.tensor([r['foreground_iou'] for r in b]);p=paired_statistics(x,y);d=(x-y).numpy();p['wins_ties_losses']=[int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d<-1e-12).sum())];p['wilcoxon_pvalue']=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.;return p

def main():
    device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all()
    if (OUT/'results.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
    _,collapsed,_=load_collapsed(device);fusion=g10.load_fusion(device);projection=g10.load_projection(device);head=load_head(device);sam=g10.load_sam_runtime(hd.CFG,device);p4t,p4v=g10.Phase4FStore(hd.CFG,'train'),g10.Phase4FStore(hd.CFG,'val');fst,fsv=g10.Store('train'),g10.Store('val');dev=g10.load_dev('g0');tc=load_c1_cache('train',p4t.sample_ids);vc=load_c1_cache('val',dev['sample_ids']);data=q.load_ids(p4t.sample_ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries'));data['sample_ids']=p4t.sample_ids
    h0=init_model(device);h1=init_model(device);assert count_trainable(h0)==count_trainable(h1);dump(OUT/'protocol.json',{'status':'FROZEN_BEFORE_FIRST_STEP','question':'incremental value of real C given source context','arms':{'H0':'same network with C=0','H1':'same network with real C'},'trainable_params_each':count_trainable(h0),'recipe':{'epochs':EPOCHS,'batch':BATCH,'lr':LR,'wd':WD,'seed':SEED,'selector':'DEV mean FG IoU','loss':'seg+relative+ranking'},'firewall':{'test':False,'official1000':False,'ood':False,'translator_change':False,'objective_change':False}})
    train('H0_zero_C',h0,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device);train('H1_real_C',h1,collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device)
    s0=json.loads((OUT/'H0_zero_C/summary.json').read_text());s1=json.loads((OUT/'H1_real_C/summary.json').read_text());r0=rows(OUT/'H0_zero_C/validation/selected.jsonl');r1=rows(OUT/'H1_real_C/validation/selected.jsonl');paired=pair(r1,r0)
    h1.load_state_dict(torch.load(OUT/'H1_real_C/selected_checkpoint.pt',map_location='cpu',weights_only=False)['model']);diag={}
    for c in ('zero','channel','spatial','flip'):
        m,rr,g=evaluate(h1,True,collapsed,sam,fusion,projection,head,p4v,fsv,dev,vc,device,corruption=c);diag[c]={'metrics':m,'paired_vs_real':pair(rr,r1),'gate_stats':g}
    decision='TYPE3_CORRECTION_INFORMATION_ADDS_VALUE' if paired['mean_difference']>0 and paired['bootstrap_95_ci'][0]>0 else ('CORRECTION_INPUT_HARMS_CURRENT_UTILITY' if paired['bootstrap_95_ci'][1]<0 else 'CORRECTION_INPUT_REDUNDANT_GIVEN_SOURCE_CONTEXT')
    result={'status':'COMPLETE_STOP','decision':decision,'H0':s0,'H1':s1,'paired_H1_minus_H0':paired,'posthoc':diag,'firewall':{'test':False,'official1000':False,'ood':False}};dump(OUT/'results.json',result)
    DOC.write_text('\n'.join(['# Phase 6G.17 — Type-III Incremental Correction Information Audit','','Status: **COMPLETE STOP**.','',f"H0: `{s0['metrics']}`",f"H1: `{s1['metrics']}`",f"Paired H1-H0: `{paired}`",f"Post-hoc: `{diag}`",'',f"```text\n{decision}\n```",'','No test, Official1000, OOD, Translator change, or objective redesign.'])+'\n');print(json.dumps({'status':'COMPLETE_STOP','decision':decision}),flush=True)
if __name__=='__main__':main()
