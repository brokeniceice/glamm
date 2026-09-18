#!/usr/bin/env python3
"""Matched Phase4C-A training of fresh block11/17 fusion plus adapter."""
from __future__ import annotations
import json,math,os,random,shutil,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts.phase4c_a_train import make_model,initial_hashes
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard,dump,write_rows
from tools.phase3c1 import binary_metrics,inverse_logits,probe_loss
from tools.phase4c_a import summarize_extended,tensor_hash
from tools.phase4c_b import file_sha256
OUT=ROOT/'outputs/phase6g2_multilevel_attention';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g2_multilevel_attention/adapter');SEED=3407
def pairs(split):
 a,b=cache_paths('middle',split),cache_paths('late',split)
 if len(a)!=len(b):raise RuntimeError('fusion cache count drift')
 return list(zip(a,b))
def load(pair):
 a,b=map(load_shard,pair)
 if [x['sample_id'] for x in a['records']] != [x['sample_id'] for x in b['records']] or not torch.equal(a['targets'],b['targets']):raise RuntimeError('fusion cache pairing drift')
 return a,b
def forward(fusion,adapter,a,b,device,weights=False):
 f11=a['features'].to(device=device,dtype=torch.float32);f17=b['features'].to(device=device,dtype=torch.float32)
 fused,w=fusion(f11,f17,need_weights=True) if weights else (fusion(f11,f17),None)
 with torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=adapter(fused,return_features=True)
 return o,w
def validate(fusion,adapter,ps,device,weights=False):
 rec=[];att=[];fusion.eval();adapter.eval()
 with torch.no_grad():
  for pair in ps:
   a,b=load(pair);o,w=forward(fusion,adapter,a,b,device,weights);att.append(w.cpu()) if w is not None else None
   for i,r in enumerate(b['records']):rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(o['logits'][i,0].float(),r['geometry']),b['original_masks'][i].to(device))})
 return summarize_extended(rec),rec,att
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);root=OUT/'phase6g2b/adapter';root.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
 if (root/'completion.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 fusion=CrossLayerPatchAttention(1024,8,.01).to(device);adapter=make_model('forensic_adapter',device);old=torch.load('/data/yz/groundingLMM_official/checkpoints/phase4c_a_clip_forensic_adapter/forensic_adapter/epoch_0.pt',map_location='cpu',weights_only=False)
 if initial_hashes(adapter)!=old['initial_hashes']:raise RuntimeError('Phase4C-A adapter init drift')
 init={'fusion':tensor_hash(fusion.state_dict().items()),'adapter':tensor_hash(adapter.state_dict().items()),'adapter_components':initial_hashes(adapter)};protocol={'schema':'phase6g2b_adapter_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','initialization':'fresh seed3407 fusion plus exact Phase4C-A adapter initialization; Phase6G.2A selected weights not loaded','initial_hashes':init,'architecture':{'fusion':'block11/17 standard 8-head per-patch cross-layer attention gamma_init=.01','adapter':'exact Phase4C-A CLIPSpatialArm blocks=3'},'recipe':{'epochs':10,'batch':16,'optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'betas':[.9,.999],'warmup_steps':277,'total_steps':5530,'loss':'2*BCE+0.5*Dice','grad_clip':1.,'seed':SEED,'selector':'val mean FG IoU; tie mean FG F1; epochs0..10'},'firewall':{'test':False,'official1000':False,'ood':False}};dump(root/'protocol.json',protocol)
 params=list(fusion.parameters())+list(adapter.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999));warm,total=277,5530
 def scale(step):return float(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,scale);train,val=pairs('train'),pairs('val');history=[];start=step=0;existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);fusion.load_state_dict(x['fusion']);adapter.load_state_dict(x['adapter']);opt.load_state_dict(x['optimizer']);sched.load_state_dict(x['scheduler']);start=x['epoch'];step=x['global_step'];history=json.load(open(root/'history.json'))
 else:
  m,_,_=validate(fusion,adapter,val,device);payload={'schema':'phase6g2b_adapter_checkpoint_v1','epoch':0,'global_step':0,'fusion':fusion.state_dict(),'adapter':adapter.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m,'initial_hashes':init};torch.save(payload,CKPT/'epoch_0.pt');history=[{'epoch':0,'global_step':0,'train':None,'validation':m,'gamma':float(fusion.gamma.detach()),'seconds':0.}];dump(root/'history.json',history)
 for epoch in range(start+1,11):
  began=time.time();fusion.train();adapter.train();rng=random.Random(SEED+epoch);order=list(train);rng.shuffle(order);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for pair in order:
   a,b=load(pair);idx=list(range(len(b['records'])));rng.shuffle(idx)
   for begin in range(0,len(idx),16):
    ix=idx[begin:begin+16];aa={**a,'features':a['features'][ix]};bb={**b,'features':b['features'][ix]};t=b['targets'][ix,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True);o,_=forward(fusion,adapter,aa,bb,device);loss=probe_loss(o['logits'].float(),t);loss['total'].backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sched.step();step+=1;n=len(ix)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  m,_,_=validate(fusion,adapter,val,device);row={'epoch':epoch,'global_step':step,'train':{k:s[k]/s['samples'] for k in ('bce','dice','total')}|{'samples':s['samples']},'validation':m,'gamma':float(fusion.gamma.detach()),'seconds':time.time()-began};history.append(row);dump(root/'history.json',history);payload={'schema':'phase6g2b_adapter_checkpoint_v1','epoch':epoch,'global_step':step,'fusion':{k:v.detach().cpu() for k,v in fusion.state_dict().items()},'adapter':{k:v.detach().cpu() for k,v in adapter.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m,'initial_hashes':init};p=CKPT/f'epoch_{epoch}.pt';tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,p);print(json.dumps({'stage':'6G2B_ADAPTER_EPOCH',**row}),flush=True)
 best=max(history,key=lambda x:(x['validation']['mean_foreground_iou'],x['validation']['mean_foreground_f1']));selected=root/'selected.pt';shutil.copy2(CKPT/f"epoch_{best['epoch']}.pt",selected);fusion.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['fusion']);adapter.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['adapter']);m,r,att=validate(fusion,adapter,val,device,True);w=torch.cat(att).float();layer=w.mean((0,1,2));write_rows(root/'selected_predictions.jsonl',r);dump(root/'selector.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'selected_metrics':m,'selected_checkpoint_sha256':file_sha256(selected),'candidates':history,'attention':{'block11_mean':float(layer[0]),'block17_mean':float(layer[1]),'gamma':float(fusion.gamma.detach())},'test_used':False,'official1000_used':False,'ood_used':False});dump(root/'completion.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(selected)})
if __name__=='__main__':main()
