#!/usr/bin/env python3
"""Phase6G.2A block11/17 cross-layer attention plus matched spatial probe."""
from __future__ import annotations
import json,math,os,random,sys,time
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts.phase6g0_multilevel_dense_clip import load_shard,cache_paths,dump,rows,write_rows
from tools.phase3c1 import binary_metrics,inverse_logits,paired_statistics,probe_loss,summarize,tensor_sha256

OUT=ROOT/'outputs/phase6g2_multilevel_attention';BASE=ROOT/'outputs/phase6g0_multilevel_dense_clip_audit/validation/late/predictions.jsonl';SEED=3407
def paired(split):
 a,b=cache_paths('middle',split),cache_paths('late',split)
 if len(a)!=len(b):raise RuntimeError('block11/17 shard count drift')
 return list(zip(a,b))
def batch(pair):
 a,b=map(load_shard,pair)
 ia=[r['sample_id'] for r in a['records']];ib=[r['sample_id'] for r in b['records']]
 if ia!=ib or not torch.equal(a['targets'],b['targets']):raise RuntimeError('block11/17 pairing drift')
 return a,b
def predict(fusion,probe,pairs,device,weights=False):
 rec=[];att=[];fusion.eval();probe.eval()
 with torch.no_grad():
  for pair in pairs:
   a,b=batch(pair);f11=a['features'].to(device=device,dtype=torch.float32);f17=b['features'].to(device=device,dtype=torch.float32)
   if weights:f,w=fusion(f11,f17,need_weights=True);att.append(w.cpu())
   else:f=fusion(f11,f17)
   z=probe(f).float()
   for i,r in enumerate(b['records']):rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(z[i,0],r['geometry']),b['original_masks'][i].to(device))})
 return rec,att
def main():
 root=OUT/'phase6g2a';root.mkdir(parents=True,exist_ok=True)
 if (root/'summary.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 device=torch.device('cuda:0');torch.cuda.set_device(device);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 fusion=CrossLayerPatchAttention(1024,8,.01).to(device);probe=nn.Conv2d(1024,1,1).to(device);params=list(fusion.parameters())+list(probe.parameters());opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.);train,val=paired('train'),paired('val')
 protocol={'schema':'phase6g2a_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','inputs':{'block11':'hidden_state_index12','block17':'hidden_state_index18','patch_alignment':'exact 576 positions'},'fusion':{'architecture':'standard nn.MultiheadAttention; F17 query; stack(F11,F17) key/value; per-patch two-token attention only','dim':1024,'heads':8,'gamma_init':.01,'output':'F17 + gamma*attention'},'probe':'Conv2d(1024,1,1)','recipe':{'epochs':20,'batch':16,'optimizer':'AdamW','lr':1e-3,'weight_decay':0.,'loss':'2*BCE+0.5*Dice','seed':SEED,'selector':'val mean FG IoU; tie mean FG F1'},'baseline':'exact Phase6G.0 selected block17 linear probe','firewall':{'test':False,'official1000':False,'ood':False,'r1':False}}
 dump(root/'protocol.json',protocol);history=[];best=None
 for epoch in range(1,21):
  began=time.time();fusion.train();probe.train();rng=random.Random(SEED+epoch);order=list(train);rng.shuffle(order);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for pair in order:
   a,b=batch(pair);idx=list(range(len(b['records'])));rng.shuffle(idx)
   for start in range(0,len(idx),16):
    ix=idx[start:start+16];f11=a['features'][ix].to(device=device,dtype=torch.float32);f17=b['features'][ix].to(device=device,dtype=torch.float32);t=b['targets'][ix,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True);loss=probe_loss(probe(fusion(f11,f17)),t);loss['total'].backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();n=len(ix)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  vr,_=predict(fusion,probe,val,device);vm=summarize(vr);row={'epoch':epoch,'train_samples':s['samples'],'train_bce':s['bce']/s['samples'],'train_dice':s['dice']/s['samples'],'train_total':s['total']/s['samples'],'val':vm,'gamma':float(fusion.gamma.detach()),'seconds':time.time()-began};history.append(row);dump(root/'history.json',history);score=(vm['mean_foreground_iou'],vm['mean_foreground_f1']);payload={'schema':'phase6g2a_checkpoint_v1','epoch':epoch,'fusion':{k:v.detach().cpu() for k,v in fusion.state_dict().items()},'probe':{k:v.detach().cpu() for k,v in probe.state_dict().items()},'optimizer':opt.state_dict(),'selector':score,'protocol':protocol}
  torch.save(payload,root/'last.pt')
  if best is None or score>best:best=score;torch.save(payload,root/'selected.pt')
  print(json.dumps({'stage':'6G2A_EPOCH',**row}),flush=True)
 selected=torch.load(root/'selected.pt',map_location='cpu',weights_only=False);fusion.load_state_dict(selected['fusion']);probe.load_state_dict(selected['probe']);a1,att=predict(fusion,probe,val,device,True);a0=rows(BASE)
 if [r['sample_id'] for r in a0]!=[r['sample_id'] for r in a1]:raise RuntimeError('A0/A1 sample order drift')
 ai=np.array([r['foreground_iou'] for r in a0]);bi=np.array([r['foreground_iou'] for r in a1]);af=np.array([r['foreground_f1'] for r in a0]);bf=np.array([r['foreground_f1'] for r in a1]);pi=paired_statistics(bi,ai);pf=paired_statistics(bf,af);delta=bi-ai;rescue=[a0[i]['sample_id'] for i in range(len(a0)) if ai[i]<=.1 and delta[i]>=.1];w=torch.cat(att,0).float();layer=w.mean((0,1,2));decision='MULTILEVEL_ATTENTION_FUSION_SUPPORTED' if pi['mean_difference']>0 and pi['bootstrap_95_ci'][0]>0 else 'MULTILEVEL_ATTENTION_FUSION_NOT_SUPPORTED'
 write_rows(root/'predictions.jsonl',a1);comparison={'A0_block17':summarize(a0),'A1_attention':summarize(a1),'paired':{'iou':pi,'f1':pf},'per_sample_gain_spearman_vs_A0_iou':float(stats.spearmanr(delta,ai).statistic),'rescue':{'definition':'A0 IoU<=0.10 and A1-A0>=0.10','n':len(rescue),'sample_ids':rescue},'attention':{'block11_mean':float(layer[0]),'block17_mean':float(layer[1]),'per_head_block11_mean':w[...,0].mean((0,1)).tolist(),'per_head_block17_mean':w[...,1].mean((0,1)).tolist(),'gamma':float(fusion.gamma.detach())}};dump(root/'comparison.json',comparison);summary={'status':'COMPLETE_STOP' if 'NOT_SUPPORTED' in decision else 'COMPLETE_GATE_PASS','decision':decision,'selected_epoch':selected['epoch'],'comparison':comparison,'firewall':protocol['firewall']};dump(root/'summary.json',summary)

if __name__=='__main__':main()
