#!/usr/bin/env python3
"""Phase 6G.7 matched Rectifier -> Utility -> joint training on block11+17 evidence."""
from __future__ import annotations
import argparse,json,math,os,random,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts import phase4g1q_conditional_utility as q,phase4hc_direct_utility_arms as hc,phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache,c1_language_batch,phase4f_spatial_batch
from scripts.phase6f3_full_fov_frozen_replay import group_summary
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,invalid_record,load_rectifier,load_sam_runtime,mask_loss
OUT=ROOT/'outputs/phase6g7_block11_17_staged_r1';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g7_block11_17_staged_r1');ADAPTER=ROOT/'outputs/phase6g3_forensic_adapter_audit/a0/selected.pt';FUSION=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt';SEED,EPOCHS,BATCH=3407,10,8
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def write_rows(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def ids_hash(x):import hashlib;return hashlib.sha256('\n'.join(x).encode()).hexdigest()
class Store:
 def __init__(self,split):
  aa,bb=cache_paths('middle',split),cache_paths('late',split);self.v=[];self.loc={};self.ids=[]
  for pa,pb in zip(aa,bb):
   a,b=load_shard(pa),load_shard(pb);ia=[r['sample_id'] for r in a['records']];ib=[r['sample_id'] for r in b['records']]
   if ia!=ib:raise RuntimeError('cache pairing drift')
   j=len(self.v);self.v.append((a['features'],b['features']))
   for i,s in enumerate(ia):self.ids.append(s);self.loc[s]=(j,i)
 def batch(self,ids,device):return tuple(torch.stack([self.v[self.loc[s][0]][k][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32) for k in (0,1))
def sources(device):
 fx=torch.load(FUSION,map_location='cpu',weights_only=False);f=CrossLayerPatchAttention(1024,8,.01);f.load_state_dict(fx['fusion']);ax=torch.load(ADAPTER,map_location='cpu',weights_only=False);a=CLIPSpatialArm(blocks=3);a.load_state_dict(ax['model']);return f.to(device).eval().requires_grad_(False),a.to(device).eval().requires_grad_(False),fx,ax
def evidence(fusion,adapter,store,ids,device):
 with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=adapter(fusion(*store.batch(ids,device)),return_features=True)
 return o['F_forensic'],o['logits']
def predecessor(stage):
 p=OUT/stage/'selected_checkpoint.pt'
 if not p.exists():raise RuntimeError(f'missing predecessor {p}')
 return torch.load(p,map_location='cpu',weights_only=False),p
def models(stage,device):
 u,_=hc.load_utility('a2',device);scale=json.load(open(ROOT/'outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json'));r=load_rectifier(hd.CFG,float(scale['selected_gamma']),device);line=[]
 if stage=='utility':x,p=predecessor('rectifier');r.load_state_dict(x['rectifier_state']);line=[{'stage':'rectifier','sha256':file_sha256(p)}]
 if stage=='joint':
  x,p=predecessor('rectifier');y,v=predecessor('utility');r.load_state_dict(x['rectifier_state']);u.load_state_dict(y['utility_state']);line=[{'stage':'rectifier','sha256':file_sha256(p)},{'stage':'utility','sha256':file_sha256(v)}]
 u.requires_grad_(stage in ('utility','joint'));u.language_source.eval().requires_grad_(False);u.forensic_source.eval().requires_grad_(False);r.requires_grad_(stage in ('rectifier','joint'));return u,r,line
def optimizer(stage,params,total):
 if stage!='rectifier':return torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4),None
 opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.05,betas=(.9,.999));warm=round(total*.05)
 def scale(step):
  if step<warm:return float(step+1)/max(1,warm)
  x=min(1.,(step-warm)/max(1,total-warm));return .5*(1+math.cos(math.pi*x))
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)
def evaluate(stage,u,r,sam,fusion,adapter,store,p4store,dev,cache,device):
 rec=[];u.eval();r.eval()
 with torch.no_grad():
  for i,sid in enumerate(dev['sample_ids']):
   if not bool(cache['valid'][i]):rec.append(invalid_record(sid,dev['original_masks'][i]));continue
   pos=torch.tensor([i]);batch,_=c1_language_batch(dev,cache,pos,p4store,sam,device);f,z=evidence(fusion,adapter,store,[sid],device);batch['F24']=f;batch['z_F24']=z;s64,_,_,sc,cc=phase4f_spatial_batch(p4store,[sid],device)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=r(s64,f,sc,cc,torch.ones(1,576,dtype=torch.bool,device=device))
   emb=rv['image_embeddings']
   if stage!='rectifier':
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=hc.utility_forward(u,batch)
    emb=hc.gated_embedding(s64,emb,hc.gate_to_sam_grid(o['U'],sc)*rv['support'].reshape(1,1,64,64))
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   rec.append(metric_record(sid,inverse_sam_logits(low,dev['sam_geometries'][i]),dev['original_masks'][i]))
   if (i+1)%200==0:print(json.dumps({'stage':'VAL_'+stage.upper(),'done':i+1,'total':len(dev['sample_ids'])}),flush=True)
 return summarize_extended(rec),rec
def run(stage,device):
 root=OUT/stage
 if (root/'summary.json').exists():print(json.dumps({'stage':stage,'status':'ALREADY_COMPLETE'}));return
 root.mkdir(parents=True,exist_ok=True);(CKPT/stage).mkdir(parents=True,exist_ok=True);fusion,adapter,fx,ax=sources(device);u,r,line=models(stage,device);exp=json.load(open(ROOT/'outputs/phase6e2_c1_specific_r1/i2/initialization_provenance.json'))['new_r1_initialization']
 if stage=='rectifier' and tensor_state_sha256(r.state_dict())!=exp['phase4f_random_rectifier_sha256']:raise RuntimeError('rectifier init drift')
 if stage=='utility' and tensor_state_sha256(hc.trainable_state(u))!=exp['phase4hc_a2_random_utility_trainable_sha256']:raise RuntimeError('utility init drift')
 train,val=Phase4FStore(hd.CFG,'train'),Phase4FStore(hd.CFG,'val');ids=train.sample_ids;dev=load_dev('g0');tc=load_c1_cache('train',ids);vc=load_c1_cache('val',dev['sample_ids']);ts,vs=Store('train'),Store('val')
 if ts.ids!=ids or vs.ids!=dev['sample_ids']:raise RuntimeError('population drift')
 sam=load_sam_runtime(hd.CFG,device);data=q.load_ids(ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries'));targets=data['target64'];valid=tc['valid'].bool();where={s:i for i,s in enumerate(ids)};cross={s:ids[(i+1)%len(ids)] for i,s in enumerate(ids)};perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device);params=[p for p in list(u.parameters())+list(r.parameters()) if p.requires_grad];opt,sched=optimizer(stage,params,math.ceil(len(ids)/BATCH)*EPOCHS);frozen={'sam':tensor_state_sha256(sam.state_dict()),'fusion':tensor_state_sha256(fusion.state_dict()),'adapter':tensor_state_sha256(adapter.state_dict()),'heads':tensor_state_sha256(q.source_state(u))}
 protocol={'schema':'phase6g7_stage_v1','status':'FROZEN_BEFORE_FIRST_STEP','stage':stage,'predecessors':line,'source':'block11+17 selected fusion + exact-reuse Phase6G.3 A0 adapter','fusion':{'path':str(FUSION.resolve()),'sha256':file_sha256(FUSION),'selected_epoch':fx['epoch']},'adapter':{'path':str(ADAPTER.resolve()),'sha256':file_sha256(ADAPTER),'selected_epoch':ax['epoch'],'stage1_exact_reuse':True},'recipe':('Phase4F' if stage=='rectifier' else ('Phase4H-C A2' if stage=='utility' else 'Phase4H-D')),'trainable':{'rectifier':sum(p.numel() for p in r.parameters() if p.requires_grad),'utility':sum(p.numel() for p in u.parameters() if p.requires_grad),'adapter':0},'firewall':{'test':False,'official1000':False,'ood':False}};dump(root/'protocol.json',protocol)
 hist=[];updates=start=0;existing=sorted((CKPT/stage).glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);u.load_state_dict(x['utility_state']);r.load_state_dict(x['rectifier_state']);opt.load_state_dict(x['optimizer']);start=x['epoch'];updates=x['optimizer_updates'];hist=json.load(open(root/'training_curve.json'));sched.load_state_dict(x['scheduler']) if sched is not None else None
 for epoch in range(start+1,EPOCHS+1):
  began=time.time();order=list(ids);random.Random(SEED+1009*epoch).shuffle(order);s=defaultdict(float);eligible=invalid=0;u.train(stage in ('utility','joint'));u.language_source.eval();u.forensic_source.eval();r.train(stage in ('rectifier','joint'))
  for begin in range(0,len(order),BATCH):
   block=order[begin:begin+BATCH];pos=torch.tensor([where[x] for x in block]);idx=pos[valid.index_select(0,pos)];invalid+=len(block)-len(idx);eligible+=len(idx)
   if not len(idx):continue
   bids=[ids[i] for i in idx.tolist()];batch,_=c1_language_batch(data,tc,idx,train,sam,device);f,z=evidence(fusion,adapter,ts,bids,device);batch['F24']=f;batch['z_F24']=z;s64,_,mask_targets,sc,cc=phase4f_spatial_batch(train,bids,device);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=r(s64,f,sc,cc,torch.ones(len(idx),576,dtype=torch.bool,device=device))
   rel=ranking=torch.zeros((),device=device);emb=rv['image_embeddings']
   if stage!='rectifier':
    cf,cz=evidence(fusion,adapter,ts,[cross[x] for x in bids],device);other=dict(batch);other['F24']=cf;other['z_F24']=cz;m=hc.utility_forward(u,batch);c=hc.utility_forward(u,other);sh=hc.utility_forward(u,batch,permutation=perm);emb=hc.gated_embedding(s64,emb,hc.gate_to_sam_grid(m['U'],sc)*rv['support'].reshape(len(idx),1,64,64));_,soft=q.target_delta({'p_L':m['p_L'],'p_F':m['p_F']},targets.index_select(0,idx).to(device));rel=q.image_balanced_loss(m['utility_logit'],soft,m['support']);ranking=.5*(hc.rank_loss(m['U'],c['U'],m['support'])+hc.rank_loss(m['U'],sh['U'],m['support']))
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   seg=mask_loss(low,mask_targets,hd.CFG);loss=seg['total']+rel+ranking
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);opt.step();updates+=1;sched.step() if sched is not None else None
   for k,v in [('seg_loss',seg['total']),('relative_loss',rel),('ranking_loss',ranking),('total_loss',loss)]:s[k]+=float(v.detach())*len(idx)
   if updates<=3 or updates%100==0:print(json.dumps({'stage':stage,'epoch':epoch,'update':updates,'loss':float(loss.detach()),'grad_norm':float(norm)}),flush=True)
  met,rec=evaluate(stage,u,r,sam,fusion,adapter,vs,val,dev,vc,device);row={'epoch':epoch,'optimizer_updates':updates,**{k:s[k]/eligible for k in ('seg_loss','relative_loss','ranking_loss','total_loss')},'traversal_exposures':len(order),'optimization_eligible_exposures':eligible,'invalid_g0_exposures':invalid,'dev_g0_mean_iou':met['mean_foreground_iou'],'dev_g0_mean_f1':met['mean_foreground_f1'],'sample_order_sha256':ids_hash(order),'seconds':time.time()-began};hist.append(row);dump(root/'training_curve.json',hist);write_rows(root/f'validation/epoch_{epoch}.jsonl',rec);payload={'schema':'phase6g7_stage_checkpoint_v1','stage':stage,'epoch':epoch,'optimizer_updates':updates,'utility_state':{k:v.detach().cpu() for k,v in u.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in r.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':None if sched is None else sched.state_dict(),'validation_g0':met};p=CKPT/stage/f'epoch_{epoch}.pt';tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,p);print(json.dumps({'stage':'EPOCH_COMPLETE','arm':stage,**row}),flush=True)
 best=max(hist,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']));selected=root/'selected_checkpoint.pt';shutil.copy2(CKPT/stage/f"epoch_{best['epoch']}.pt",selected);dump(root/'selector.json',{'status':'COMPLETE','primary':'internal validation canonical G0 mean IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(selected),'candidates':hist,'test_used':False,'official1000_used':False,'ood_used':False})
 if frozen!={'sam':tensor_state_sha256(sam.state_dict()),'fusion':tensor_state_sha256(fusion.state_dict()),'adapter':tensor_state_sha256(adapter.state_dict()),'heads':tensor_state_sha256(q.source_state(u))}:raise RuntimeError('frozen hash drift')
 dump(root/'summary.json',{'status':'COMPLETE_SELECTED_FROZEN','stage':stage,'selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(selected)})
 if stage=='joint':finalize(root,best,vc)
def finalize(root,best,vc):
 selected=rows(root/f"validation/epoch_{best['epoch']}.jsonl");raw=rows(ROOT/'outputs/phase6f3_full_fov_frozen_replay/per_sample_results.jsonl');base=[{'sample_id':r['sample_id'],'foreground_iou':r['a0_iou'],'foreground_f1':r['a0_f1'],'tp':r['a0_tp'],'fp':r['a0_fp'],'fn':r['a0_fn']} for r in raw]
 if [x['sample_id'] for x in base]!=[x['sample_id'] for x in selected]:raise RuntimeError('pairing drift')
 paired=[{'sample_id':a['sample_id'],'a0_iou':a['foreground_iou'],'a0_f1':a['foreground_f1'],'a0_tp':a['tp'],'a0_fp':a['fp'],'a0_fn':a['fn'],'a1_iou':b['foreground_iou'],'a1_f1':b['foreground_f1'],'a1_tp':b['tp'],'a1_fp':b['fp'],'a1_fn':b['fn'],'delta_iou':b['foreground_iou']-a['foreground_iou'],'delta_f1':b['foreground_f1']-a['foreground_f1']} for a,b in zip(base,selected)];c=group_summary(paired);d=c['paired']['iou'];decision='BLOCK11_17_STAGED_R1_SUPPORTED' if d['mean_delta']>0 and d['bootstrap_95_ci'][0]>0 else ('BLOCK11_17_STAGED_R1_NOT_STABLY_BETTER' if d['mean_delta']>0 else 'MULTILEVEL_EVIDENCE_NOT_CONVERTED_BY_STAGED_R1');prov={s:json.load(open(OUT/s/'summary.json')) for s in ('rectifier','utility','joint')};prov['adapter']={'status':'EXACT_REUSE','checkpoint':str(ADAPTER.resolve()),'sha256':file_sha256(ADAPTER),'selected_epoch':torch.load(ADAPTER,map_location='cpu',weights_only=False)['epoch']};ref=json.load(open(ROOT/'outputs/phase6g6_joint_from_init/summary.json'))
 summary={'schema':'phase6g7_summary_v1','status':'COMPLETE_STOP','decision':decision,'selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(root/'selected_checkpoint.pt'),'comparison':c,'seg_trigger_invariance':{'n':len(vc['valid']),'valid':int(vc['valid'].sum()),'rate':float(vc['valid'].float().mean()),'A0_A1_exact':True},'stage_provenance':prov,'joint_from_init_reference':{'mean_fg_iou':ref['comparison']['A1']['mean_fg_iou'],'not_used_for_selection':True},'firewall':{'test_accessed':False,'official1000_accessed':False,'ood_accessed':False}};write_rows(OUT/'selected_validation_predictions.jsonl',selected);dump(OUT/'comparison.json',c);dump(OUT/'summary.json',summary);render(summary)
def render(s):
 c=s['comparison'];a,b=c['A0'],c['A1'];lines=['# Phase 6G.7 — Staged New R1 with Block11+17 Evidence','','Status: **COMPLETE STOP**. Stage 1 exactly reused the already completed matched Phase6G.3 A0 adapter pretrain; Rectifier, Utility, and joint stages used the historical matched recipes.','','| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |','|---|---:|---:|---:|---:|---:|',f"| A0 block22 current new R1 | {a['mean_fg_iou']:.6f} | {a['median_fg_iou']:.6f} | {a['mean_fg_f1']:.6f} | {a['global_fg_iou']:.6f} | {a['global_fg_f1']:.6f} |",f"| A1 block11+17 staged new R1 | {b['mean_fg_iou']:.6f} | {b['median_fg_iou']:.6f} | {b['mean_fg_f1']:.6f} | {b['global_fg_iou']:.6f} | {b['global_fg_f1']:.6f} |",'',f"- paired: `{c['paired']}`",f"- stage provenance: `{s['stage_provenance']}`",f"- SEG trigger: `{s['seg_trigger_invariance']}`",f"- Phase6G.6 reference mean IoU: `{s['joint_from_init_reference']['mean_fg_iou']:.6f}` (not used for selection)",'',f"```text\n{s['decision']}\n```"];(ROOT/'docs/phase6g7_block11_17_staged_r1.md').write_text('\n'.join(lines)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('rectifier','utility','joint'),required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();device=torch.device(a.device);torch.cuda.set_device(device);hc.seed_all();run(a.stage,device)
if __name__=='__main__':main()
