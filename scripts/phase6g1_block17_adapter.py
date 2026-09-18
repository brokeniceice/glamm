#!/usr/bin/env python3
"""Matched Phase4C-A forensic-adapter training on frozen CLIP block-17 grids."""
from __future__ import annotations
import csv,json,math,os,random,shutil,sys,time
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.phase4c_a_train import make_model,initial_hashes
from tools.phase3c1 import binary_metrics,inverse_logits,probe_loss
from tools.phase4c_a import summarize_extended,tensor_hash
from tools.phase4c_b import file_sha256

OUT=ROOT/'outputs/phase6g1_block17_single_layer_replacement';CACHE=Path('/data/yz/groundingLMM_official/cache/phase6g0_multilevel_dense_clip_audit/late');CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g1_block17/adapter');SEED=3407
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def paths(split):
 c=json.load(open(CACHE/split/'complete.json'))
 if c['status']!='COMPLETE' or c['block_index']!=17 or not c['frozen_exact']:raise RuntimeError('block17 cache gate failed')
 p=sorted((CACHE/split).glob('shard_*.pt'))
 if len(p)!=c['shards']:raise RuntimeError('block17 shard count drift')
 return p,c
def load(p):
 x=torch.load(p,map_location='cpu',weights_only=False)
 if x.get('schema')!='phase6g0_multilevel_clip_cache_v1' or x.get('block_index')!=17:raise RuntimeError(p)
 return x
def validate(model,ps,device):
 rec=[];low=[];model.eval()
 with torch.no_grad():
  for p in ps:
   x=load(p)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):z=model(x['features'].to(device=device,dtype=torch.float32)).float().cpu()
   for i,r in enumerate(x['records']):
    original=inverse_logits(z[i,0].to(device),r['geometry']);m=binary_metrics(original,x['original_masks'][i].to(device));rec.append({'sample_id':r['sample_id'],**m});low.append(z[i,0])
 return summarize_extended(rec),rec,torch.stack(low)
def scheduler(model):
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999));warm,total=277,5530
 def scale(step):
  if step<warm:return float(step+1)/warm
  return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);trainp,tc=paths('train');valp,vc=paths('val');root=OUT/'adapter';root.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
 if (root/'completion.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 model=make_model('forensic_adapter',device);initial={'full':tensor_hash(model.state_dict().items()),**initial_hashes(model)};old=torch.load('/data/yz/groundingLMM_official/checkpoints/phase4c_a_clip_forensic_adapter/forensic_adapter/epoch_0.pt',map_location='cpu',weights_only=False)
 if initial_hashes(model)!=old['initial_hashes']:raise RuntimeError('Phase4C-A corresponding initialization drift')
 protocol={'schema':'phase6g1_adapter_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','representation_only_change':'CLIP block22 -> block17','architecture':'exact Phase4C-A CLIPSpatialArm blocks=3','initialization':initial,'recipe':{'epochs':10,'batch':16,'optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'betas':[.9,.999],'warmup_steps':277,'scheduler':'linear warmup then cosine','loss':'2*BCE+0.5*Dice','gradient_clip':1.,'selector':'validation mean FG IoU, tie mean FG F1, candidates epoch0..10','seed':SEED},'cache':{'train':tc,'val':vc},'firewall':{'test':False,'official1000':False,'ood':False}}
 dump(root/'protocol.json',protocol);opt,sched=scheduler(model);history=[];start=0;step=0;existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);model.load_state_dict(x['model']);opt.load_state_dict(x['optimizer']);sched.load_state_dict(x['scheduler']);start=x['epoch'];step=x['global_step'];history=json.load(open(root/'history.json'))
 else:
  m,r,_=validate(model,valp,device);x={'schema':'phase6g1_adapter_checkpoint_v1','epoch':0,'global_step':0,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m,'initial_hashes':initial};torch.save(x,CKPT/'epoch_0.pt');history=[{'epoch':0,'global_step':0,'train':None,'validation':m,'seconds':0.}];dump(root/'history.json',history)
 for epoch in range(start+1,11):
  began=time.time();model.train();rng=random.Random(SEED+epoch);ep=list(trainp);rng.shuffle(ep);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for p in ep:
   x=load(p);order=list(range(len(x['records'])));rng.shuffle(order)
   for b in range(0,len(order),16):
    idx=order[b:b+16];f=x['features'][idx].to(device=device,dtype=torch.float32);t=x['targets'][idx,None].to(device);opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16):loss=probe_loss(model(f).float(),t)
    loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();step+=1;n=len(idx)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  if s['samples']!=8836:raise RuntimeError('adapter exposure drift')
  m,r,_=validate(model,valp,device);row={'epoch':epoch,'global_step':step,'train':{k:s[k]/s['samples'] for k in ('bce','dice','total')}|{'samples':s['samples']},'validation':m,'seconds':time.time()-began};history.append(row);dump(root/'history.json',history);payload={'schema':'phase6g1_adapter_checkpoint_v1','epoch':epoch,'global_step':step,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m,'initial_hashes':initial};p=CKPT/f'epoch_{epoch}.pt';tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,p);print(json.dumps({'stage':'ADAPTER_EPOCH',**row}),flush=True)
 best=max(history,key=lambda r:(r['validation']['mean_foreground_iou'],r['validation']['mean_foreground_f1']));shutil.copy2(CKPT/f"epoch_{best['epoch']}.pt",root/'selected.pt');dump(root/'selector.json',{'status':'COMPLETE','primary':'validation Fake mean FG IoU','tie_break':'mean FG F1','candidates':history,'selected_epoch':best['epoch'],'selected_metrics':best['validation'],'selected_checkpoint_sha256':file_sha256(root/'selected.pt'),'test_used':False,'official1000_used':False,'ood_used':False});protocol.update(status='COMPLETE_SELECTED_FROZEN',selected_epoch=best['epoch'],selected_checkpoint_sha256=file_sha256(root/'selected.pt'));dump(root/'protocol.json',protocol);dump(root/'completion.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(root/'selected.pt')})
if __name__=='__main__':main()
