#!/usr/bin/env python3
"""Phase 6F.4 staged Full-FOV R1: Rectifier -> Utility -> joint.

Each invocation runs exactly one stage.  Selection is internal-validation G0
only; Official1000 is deliberately handled by a separate post-selection tool.
"""
from __future__ import annotations
import argparse,csv,json,math,os,random,shutil,time,sys
from collections import defaultdict
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase6f4_full_fov_i2_train as base
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,invalid_record,load_rectifier,load_sam_runtime,mask_loss

OUT=ROOT/'outputs/phase6f4_full_fov_staged'
CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6f4_full_fov_staged')
SEED,EPOCHS,BATCH=3407,10,8

def cli():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('rectifier','utility','joint'),required=True);p.add_argument('--device',default='cuda:0');return p.parse_args()

def seed_all():hc.seed_all()
def dump(p,x):base.dump(p,x)
def write_records(p,x):base.write_records(p,x)

def load_selected(stage):
 p=OUT/stage/'selected_checkpoint.pt'
 if not p.exists():raise RuntimeError(f'missing frozen predecessor: {p}')
 return torch.load(p,map_location='cpu',weights_only=False),p

def stage_models(stage,device):
 utility,_=hc.load_utility('a2',device)
 scale=json.load(open(ROOT/'outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json'))
 rectifier=load_rectifier(hd.CFG,float(scale['selected_gamma']),device)
 lineage={'utility':'Phase4H-C A2 seed3407 random initialization','rectifier':'Phase4F seed3407 random initialization','loaded_predecessors':[]}
 if stage=='utility':
  state,p=load_selected('rectifier');rectifier.load_state_dict(state['rectifier_state'],strict=True);lineage['loaded_predecessors'].append({'stage':'rectifier','path':str(p.resolve()),'sha256':file_sha256(p)})
 elif stage=='joint':
  rs,rp=load_selected('rectifier');us,up=load_selected('utility');rectifier.load_state_dict(rs['rectifier_state'],strict=True);utility.load_state_dict(us['utility_state'],strict=True)
  lineage['loaded_predecessors'] += [{'stage':'rectifier','path':str(rp.resolve()),'sha256':file_sha256(rp)},{'stage':'utility','path':str(up.resolve()),'sha256':file_sha256(up)}]
 utility.requires_grad_(stage in ('utility','joint'));utility.language_source.eval().requires_grad_(False);utility.forensic_source.eval().requires_grad_(False)
 rectifier.requires_grad_(stage in ('rectifier','joint'))
 return utility,rectifier,lineage

def rectifier_eval(rectifier,sam,store,cache,full,device):
 rec=[];rectifier.eval()
 with torch.no_grad():
  for i,sid in enumerate(store.sample_ids):
   truth=store.original_masks[sid]
   if not bool(cache['valid'][i]):rec.append(invalid_record(sid,truth));continue
   pos=torch.tensor([i]);batch,_,sc,_=base.language_batch(store,cache,pos,sam,device);rv=base.rectified_batch(rectifier,batch['S64'],sc,[full[sid]],device)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],rv['image_embeddings'].to(torch.bfloat16))
   rec.append(metric_record(sid,inverse_sam_logits(low,store.geometries[sid]),truth))
   if (i+1)%200==0:print(json.dumps({'stage':'VAL_RECTIFIER','done':i+1,'total':len(store.sample_ids)}),flush=True)
 iou=torch.tensor([r['foreground_iou'] for r in rec]);f1=torch.tensor([r['foreground_f1'] for r in rec]);return {'n':len(rec),'mean_foreground_iou':float(iou.mean()),'mean_foreground_f1':float(f1.mean())},rec

def optimizer(stage,params,total_updates):
 if stage=='rectifier':
  opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.05,betas=(.9,.999));warm=round(total_updates*.05)
  def scale(step):
   if step<warm:return float(step+1)/max(1,warm)
   progress=min(1.,(step-warm)/max(1,total_updates-warm));return .5*(1+math.cos(math.pi*progress))
  return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)
 return torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4),None

def save(stage,epoch,updates,utility,rectifier,opt,sched,metric,init):
 p=CKPT/stage/f'epoch_{epoch}.pt';p.parent.mkdir(parents=True,exist_ok=True)
 x={'schema':'phase6f4_full_fov_staged_checkpoint_v1','stage':stage,'epoch':epoch,'optimizer_updates':updates,'utility_state':{k:v.detach().cpu() for k,v in utility.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in rectifier.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':None if sched is None else sched.state_dict(),'validation_g0':metric,'initialization':init}
 t=p.with_suffix('.pt.tmp');torch.save(x,t);os.replace(t,p);return p

def main():
 a=cli();device=torch.device(a.device);torch.cuda.set_device(device);seed_all();root=OUT/a.stage
 if (root/'summary.json').exists():print(json.dumps({'stage':a.stage,'status':'ALREADY_COMPLETE'}));return
 root.mkdir(parents=True,exist_ok=True);utility,rectifier,lineage=stage_models(a.stage,device)
 expected=json.load(open(ROOT/'outputs/phase6e2_c1_specific_r1/i2/initialization_provenance.json'))['new_r1_initialization']
 fresh={'utility':expected['phase4hc_a2_random_utility_trainable_sha256'],'utility_full':expected['utility_state_sha256'],'rectifier':expected['phase4f_random_rectifier_sha256']}
 if a.stage=='rectifier' and tensor_state_sha256(utility.state_dict())!=fresh['utility_full']:raise RuntimeError('historical random utility full-state initialization drift')
 if a.stage=='utility' and tensor_state_sha256(hc.trainable_state(utility))!=fresh['utility']:raise RuntimeError('historical random utility trainable-state initialization drift')
 if a.stage=='rectifier' and tensor_state_sha256(rectifier.state_dict())!=fresh['rectifier']:raise RuntimeError('historical random rectifier initialization drift')
 train=Phase4FStore(hd.CFG,'train');val=Phase4FStore(hd.CFG,'val');train_full,ts=base.load_full('train',train.sample_ids);val_full,vs=base.load_full('val',val.sample_ids);tc=base.load_c1_cache('train',train.sample_ids);vc=base.load_c1_cache('val',val.sample_ids)
 sam=load_sam_runtime(hd.CFG,device);frozen_before={'sam':tensor_state_sha256(sam.state_dict()),'source_heads':tensor_state_sha256(base.q.source_state(utility))}
 valid=tc['valid'].bool();id_to_i={s:i for i,s in enumerate(train.sample_ids)};targets=base.q.load_ids(train.sample_ids,('target64',))['target64'];cross={s:train.sample_ids[(i+1)%len(train.sample_ids)] for i,s in enumerate(train.sample_ids)}
 params=([p for p in rectifier.parameters() if p.requires_grad]+[p for p in utility.parameters() if p.requires_grad]);opt,sched=optimizer(a.stage,params,math.ceil(len(train.sample_ids)/BATCH)*EPOCHS)
 init={'lineage':lineage,'fresh_hashes':fresh,'trainable_counts':{'rectifier':sum(p.numel() for p in rectifier.parameters() if p.requires_grad),'utility':sum(p.numel() for p in utility.parameters() if p.requires_grad)}}
 dump(root/'protocol.json',{'schema':'phase6f4_staged_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','stage':a.stage,'initialization':init,'recipe':('Phase4F exact optimizer/scheduler/mask loss' if a.stage=='rectifier' else ('Phase4H-C A2 exact utility recipe' if a.stage=='utility' else 'Phase4H-D exact joint recipe')),'data':'internal train/validation only; C1 actual valid accounting','full_fov_cache':{'train':ts,'val':vs},'official1000_used_for_selector':False})
 history=[];updates=0;start_epoch=0
 existing=sorted((CKPT/a.stage).glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  resume=torch.load(existing[-1],map_location='cpu',weights_only=False);utility.load_state_dict(resume['utility_state'],strict=True);rectifier.load_state_dict(resume['rectifier_state'],strict=True);opt.load_state_dict(resume['optimizer'])
  if sched is not None:sched.load_state_dict(resume['scheduler'])
  start_epoch=int(resume['epoch']);updates=int(resume['optimizer_updates'])
  curve=root/'training_curve.csv'
  if not curve.exists():raise RuntimeError('checkpoint exists without training curve')
  with curve.open() as f:history=[{k:(int(v) if k in ('epoch','optimizer_updates','traversal_exposures','optimization_eligible_exposures','invalid_g0_exposures') else (float(v) if k not in ('sample_order_sha256',) else v)) for k,v in row.items()} for row in csv.DictReader(f)]
  if len(history)!=start_epoch:raise RuntimeError('resume history/checkpoint epoch drift')
  print(json.dumps({'stage':a.stage,'status':'RESUME','start_epoch':start_epoch,'updates':updates}),flush=True)
 for epoch in range(start_epoch+1,EPOCHS+1):
  began=time.time();order=list(train.sample_ids);random.Random(SEED+1009*epoch).shuffle(order);sums=defaultdict(float);eligible=invalid=0;rectifier.train(a.stage in ('rectifier','joint'));utility.train(a.stage in ('utility','joint'));utility.language_source.eval();utility.forensic_source.eval()
  for begin in range(0,len(order),BATCH):
   block=order[begin:begin+BATCH];pos=torch.tensor([id_to_i[s] for s in block]);pos=pos[valid.index_select(0,pos)];invalid+=len(block)-len(pos);eligible+=len(pos)
   if not len(pos):continue
   batch,mask_targets,sc,bids=base.language_batch(train,tc,pos,sam,device);fr=[train_full[s] for s in bids];opt.zero_grad(set_to_none=True);rv=base.rectified_batch(rectifier,batch['S64'],sc,fr,device)
   if a.stage=='rectifier':emb=rv['image_embeddings'];relative=ranking=torch.zeros((),device=device)
   else:
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16):match=base.utility_aligned(utility,batch,base.aligned(fr,device));crossed=base.utility_aligned(utility,batch,base.aligned([train_full[cross[s]] for s in bids],device));shuffled=base.utility_aligned(utility,batch,base.aligned(fr,device,True))
    gate=hc.gate_to_sam_grid(match['U'],sc)*rv['support'].reshape(len(pos),1,64,64);emb=hc.gated_embedding(batch['S64'],rv['image_embeddings'],gate);_,soft=base.q.target_delta({'p_L':match['p_L'],'p_F':match['p_F']},targets.index_select(0,pos).to(device));relative=base.q.image_balanced_loss(match['utility_logit'],soft,match['support']);ranking=.5*(hc.rank_loss(match['U'],crossed['U'],match['support'])+hc.rank_loss(match['U'],shuffled['U'],match['support']))
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   seg=mask_loss(low,mask_targets,hd.CFG);total=seg['total']+relative+ranking
   if not torch.isfinite(total):raise RuntimeError('nonfinite loss')
   total.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);opt.step();updates+=1
   if sched is not None:sched.step()
   for k,v in [('seg_loss',seg['total']),('relative_loss',relative),('ranking_loss',ranking),('total_loss',total)]:sums[k]+=float(v.detach())*len(pos)
   if updates<=3 or updates%100==0:print(json.dumps({'stage':a.stage,'epoch':epoch,'update':updates,'loss':float(total.detach()),'grad_norm':float(norm)}),flush=True)
  metric,records=(rectifier_eval(rectifier,sam,val,vc,val_full,device) if a.stage=='rectifier' else base.evaluate(utility,rectifier,sam,val,vc,val_full,device))
  row={'epoch':epoch,'optimizer_updates':updates,**{k:sums[k]/eligible for k in ('seg_loss','relative_loss','ranking_loss','total_loss')},'traversal_exposures':len(order),'optimization_eligible_exposures':eligible,'invalid_g0_exposures':invalid,'dev_g0_mean_iou':metric['mean_foreground_iou'],'dev_g0_mean_f1':metric['mean_foreground_f1'],'sample_order_sha256':base.ids_hash(order),'seconds':time.time()-began};history.append(row)
  with (root/'training_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
  write_records(root/f'validation/epoch_{epoch}.jsonl',records);save(a.stage,epoch,updates,utility,rectifier,opt,sched,metric,init);print(json.dumps({'stage':'EPOCH_COMPLETE','arm':a.stage,**row}),flush=True)
 best=max(history,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']));chosen=CKPT/a.stage/f"epoch_{best['epoch']}.pt";shutil.copy2(chosen,root/'selected_checkpoint.pt')
 selector={'status':'COMPLETE','stage':a.stage,'primary':'internal validation canonical G0 mean IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(root/'selected_checkpoint.pt'),'candidates':history,'official1000_used':False,'test_used':False,'ood_used':False};dump(root/'selector.json',selector)
 after={'sam':tensor_state_sha256(sam.state_dict()),'source_heads':tensor_state_sha256(base.q.source_state(utility))}
 if after!=frozen_before:raise RuntimeError('frozen component drift')
 if a.stage=='joint':
  base.OUT=OUT;base.SELECTED=root/'selected_checkpoint.pt';base.DOC=ROOT/'docs/phase6f4_full_fov_staged.md'
  base.finalize(history,base.rows(root/f"validation/epoch_{best['epoch']}.jsonl"),val.sample_ids)
 dump(root/'summary.json',{'status':'COMPLETE_SELECTED_FROZEN','stage':a.stage,'selected_epoch':best['epoch'],'selected_checkpoint':str((root/'selected_checkpoint.pt').resolve()),'selected_checkpoint_sha256':file_sha256(root/'selected_checkpoint.pt'),'official1000_accessed':False})

if __name__=='__main__':main()
