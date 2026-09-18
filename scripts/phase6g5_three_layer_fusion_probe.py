#!/usr/bin/env python3
"""Phase 6G.5 block11+17+22 patch-aligned fusion probe."""
from __future__ import annotations
import json,random,sys,time
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import ThreeLayerPatchAttention
from scripts.phase6g0_multilevel_dense_clip import load_shard,cache_paths,dump,rows,write_rows
from tools.phase3c1 import binary_metrics,inverse_logits,paired_statistics,probe_loss,summarize
from tools.phase4c_b import file_sha256

OUT=ROOT/'outputs/phase6g5_three_layer_fusion_probe';A0=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a';SEED=3407
LAYERS=('middle','late','current_hidden_minus2')
def triples(split):
 ps=[cache_paths(x,split) for x in LAYERS]
 if len({len(x) for x in ps})!=1:raise RuntimeError('shard count drift')
 return list(zip(*ps))
def batch(paths):
 xs=[load_shard(p) for p in paths];ids=[[r['sample_id'] for r in x['records']] for x in xs]
 if ids[1]!=ids[0] or ids[2]!=ids[0] or not torch.equal(xs[0]['targets'],xs[1]['targets']) or not torch.equal(xs[1]['targets'],xs[2]['targets']):raise RuntimeError('three-layer pairing drift')
 return xs
def predict(fusion,probe,paths,device,weights=False):
 rec=[];att=[];fusion.eval();probe.eval()
 with torch.no_grad():
  for ps in paths:
   a,b,c=batch(ps);fs=[x['features'].to(device=device,dtype=torch.float32) for x in (a,b,c)]
   if weights:f,w=fusion(*fs,need_weights=True);att.append(w.cpu())
   else:f=fusion(*fs)
   z=probe(f).float()
   for i,r in enumerate(b['records']):rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(z[i,0],r['geometry']),b['original_masks'][i].to(device))})
 return rec,att
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'summary.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 device=torch.device('cuda:0');torch.cuda.set_device(device);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 fusion=ThreeLayerPatchAttention(1024,8,.01).to(device);probe=nn.Conv2d(1024,1,1).to(device);params=list(fusion.parameters())+list(probe.parameters());opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.);train,val=triples('train'),triples('val')
 protocol={'schema':'phase6g5_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','inputs':{'block11':'hidden_state_index12','block17':'hidden_state_index18','block22':'hidden_state_index23','patch_alignment':'exact 576 positions'},'fusion':{'architecture':'same standard MHA as Phase6G.2A; block17 query; stack(block11,block17,block22) key/value; per-patch three-token attention only','dim':1024,'heads':8,'gamma_init':.01,'output':'block17 + gamma*attention'},'probe':'Conv2d(1024,1,1)','recipe':{'epochs':20,'batch':16,'optimizer':'AdamW','lr':1e-3,'weight_decay':0.,'loss':'2*BCE+0.5*Dice','seed':SEED,'selector':'val mean FG IoU; tie mean FG F1','threshold_logit':0.},'A0':{'selected_checkpoint':str((A0/'selected.pt').resolve()),'sha256':file_sha256(A0/'selected.pt'),'reuse':True},'firewall':{'adapter':False,'rectifier':False,'utility':False,'test':False,'official1000':False,'ood':False}}
 dump(OUT/'protocol.json',protocol);history=[];best=None
 for epoch in range(1,21):
  began=time.time();fusion.train();probe.train();rng=random.Random(SEED+epoch);order=list(train);rng.shuffle(order);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for ps in order:
   a,b,c=batch(ps);idx=list(range(len(b['records'])));rng.shuffle(idx)
   for start in range(0,len(idx),16):
    ix=idx[start:start+16];fs=[x['features'][ix].to(device=device,dtype=torch.float32) for x in (a,b,c)];t=b['targets'][ix,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True);loss=probe_loss(probe(fusion(*fs)),t);loss['total'].backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();n=len(ix)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  vr,_=predict(fusion,probe,val,device);vm=summarize(vr);row={'epoch':epoch,'train_samples':s['samples'],'train_bce':s['bce']/s['samples'],'train_dice':s['dice']/s['samples'],'train_total':s['total']/s['samples'],'val':vm,'gamma':float(fusion.gamma.detach()),'seconds':time.time()-began};history.append(row);dump(OUT/'history.json',history);score=(vm['mean_foreground_iou'],vm['mean_foreground_f1']);payload={'schema':'phase6g5_checkpoint_v1','epoch':epoch,'fusion':{k:v.detach().cpu() for k,v in fusion.state_dict().items()},'probe':{k:v.detach().cpu() for k,v in probe.state_dict().items()},'optimizer':opt.state_dict(),'selector':score,'protocol':protocol};torch.save(payload,OUT/'last.pt')
  if best is None or score>best:best=score;torch.save(payload,OUT/'selected.pt')
  print(json.dumps({'stage':'6G5_EPOCH',**row}),flush=True)
 selected=torch.load(OUT/'selected.pt',map_location='cpu',weights_only=False);fusion.load_state_dict(selected['fusion']);probe.load_state_dict(selected['probe']);a1,att=predict(fusion,probe,val,device,True);a0=rows(A0/'predictions.jsonl')
 if [r['sample_id'] for r in a0]!=[r['sample_id'] for r in a1]:raise RuntimeError('A0/A1 order drift')
 ai=np.array([r['foreground_iou'] for r in a0]);bi=np.array([r['foreground_iou'] for r in a1]);af=np.array([r['foreground_f1'] for r in a0]);bf=np.array([r['foreground_f1'] for r in a1]);delta=bi-ai;pi=paired_statistics(bi,ai);pf=paired_statistics(bf,af);w=torch.cat(att).float();layer=w.mean((0,1,2));rescue=[a0[i]['sample_id'] for i in range(len(a0)) if ai[i]<=.1 and delta[i]>=.1]
 enriched=[{'sample_id':a0[i]['sample_id'],'A0_iou':float(ai[i]),'A1_iou':float(bi[i]),'iou_gain':float(delta[i]),'A0_f1':float(af[i]),'A1_f1':float(bf[i]),'f1_gain':float(bf[i]-af[i])} for i in range(len(a0))];write_rows(OUT/'predictions.jsonl',a1);write_rows(OUT/'per_sample_gain.jsonl',enriched)
 decision='BLOCK11_17_22_FUSION_SUPPORTED' if pi['mean_difference']>0 and pi['bootstrap_95_ci'][0]>0 else 'BLOCK22_ADDITIONAL_VALUE_NOT_SUPPORTED';attention={'overall':{'block11':float(layer[0]),'block17':float(layer[1]),'block22':float(layer[2])},'per_head':{'block11':w[...,0].mean((0,1)).tolist(),'block17':w[...,1].mean((0,1)).tolist(),'block22':w[...,2].mean((0,1)).tolist()},'gamma':float(fusion.gamma.detach())};comparison={'A0_block11_17':summarize(a0),'A1_block11_17_22':summarize(a1),'paired':{'iou':pi,'f1':pf},'per_sample_gain_spearman_vs_A0_iou':float(stats.spearmanr(delta,ai).statistic),'rescue':{'definition':'A0 IoU<=0.10 and A1-A0>=0.10','n':len(rescue),'sample_ids':rescue},'attention':attention};dump(OUT/'comparison.json',comparison);summary={'status':'COMPLETE_STOP','decision':decision,'selected_epoch':selected['epoch'],'fixed_future_source':'block11 + block17' if decision!='BLOCK11_17_22_FUSION_SUPPORTED' else None,'comparison':comparison,'firewall':protocol['firewall']};dump(OUT/'summary.json',summary);render(summary)
def render(x):
 c=x['comparison'];lines=['# Phase 6G.5 — Block11+17+22 Cross-Layer Fusion Probe','','Status: **COMPLETE STOP**. Only fusion and a linear spatial probe were trained on internal TRAIN/validation.','','| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |','|---|---:|---:|---:|']
 for name,key in [('A0 block11+17','A0_block11_17'),('A1 block11+17+22','A1_block11_17_22')]:m=c[key];lines.append(f"| {name} | {m['mean_foreground_iou']:.6f} | {m['median_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} |")
 lines+=['',f"- IoU paired: `{c['paired']['iou']}`",f"- F1 paired: `{c['paired']['f1']}`",f"- attention: `{c['attention']}`",f"- rescue samples: `{c['rescue']['n']}`",'',f"```text\n{x['decision']}\n```"]
 if x.get('fixed_future_source'):lines+=['',f"Future multi-level dense evidence source is fixed to **{x['fixed_future_source']}**."]
 (ROOT/'docs/phase6g5_block11_17_22_fusion_probe.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
