#!/usr/bin/env python3
"""Phase 6F.4 Full-FOV I2 joint Rectifier + Utility training."""
from __future__ import annotations
import csv,hashlib,json,os,random,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.csculf import build_comparison_features
from model.pcerf import ecolaf_discount,evidence_to_dirichlet,sam_lowres_to_original_normalized
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache,phase4f_spatial_batch
from scripts.phase6f3_full_fov_frozen_replay import group_summary
from tools.full_fov_forensic import align_full_grid
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,invalid_record,load_rectifier,load_sam_runtime,mask_loss

OUT=ROOT/'outputs/phase6f4_full_fov_i2';DOC=ROOT/'docs/phase6f4_full_fov_i2.md'
CACHE=Path('/data/yz/groundingLMM_official/cache/phase6f4_full_fov_i2')
CKPTS=Path('/data/yz/groundingLMM_official/checkpoints/phase6f4_full_fov_i2')
SELECTED=OUT/'selected_checkpoint.pt';CURRENT=ROOT/'outputs/phase6f3_full_fov_frozen_replay/per_sample_results.jsonl'
ATTR=ROOT/'outputs/phase6f1_coverage_attribution/per_sample_attribution.jsonl'
C1_SHA='85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff'
SEED,EPOCHS,BATCH,LR,WD,GRAD=3407,10,8,1e-4,1e-4,1.0

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def ids_hash(x):return hashlib.sha256('\n'.join(x).encode()).hexdigest()

def load_full(split,expected):
 out={};status=json.load(open(CACHE/split/'status.json'))
 if status['status']!='COMPLETE' or status['sample_ids_sha256']!=ids_hash(expected):raise RuntimeError(f'{split} full-FOV cache incomplete')
 for p in sorted((CACHE/split).glob('shard_*.pt')):
  x=torch.load(p,map_location='cpu',weights_only=False)
  for r in x['records']:out[r['sample_id']]=r
 if list(out)!=expected:raise RuntimeError(f'{split} cache order drift')
 return out,status

def aligned(records,device,shuffle=False):
 fs=[];zs=[];ms=[];ss=[]
 for r in records:
  f=r['F_full'][None].to(device);z=r['z_full'][None].to(device);m=r['mass_full'][None].to(device);s=r['support_full'][None].to(device)
  if shuffle:
   n=f.shape[-2]*f.shape[-1];perm=torch.randperm(n,generator=torch.Generator().manual_seed(SEED),device='cpu').to(device)
   f=f.flatten(2)[:,:,perm].reshape_as(f);z=z.flatten(2)[:,:,perm].reshape_as(z);m=m.flatten(2)[:,:,perm].reshape_as(m)
  f64,sf=align_full_grid(f,s,(64,64));z64,sz=align_full_grid(z,s,(64,64));m64,sm=align_full_grid(m,s,(64,64),normalize_channels=True,vacuous=True)
  if not torch.equal(sf,sz) or not torch.equal(sf,sm):raise RuntimeError('aligned support drift')
  fs.append(f64);zs.append(z64);ms.append(m64);ss.append(sf)
 return {'F64':torch.cat(fs),'z64':torch.cat(zs),'mass64':torch.cat(ms),'support64':torch.cat(ss)}

def utility_aligned(model,batch,a):
 el=model.language_source(batch['S64'].detach(),batch['q_seg'].detach(),batch['z_L'].detach())/model.temperature_l
 ol=evidence_to_dirichlet(el);ml=ol['masses'];pl=ol['posterior'];mf=a['mass64'];pf=mf[:,:-1]+mf[:,-1:]/2
 l=model.language_context(batch['S64'],batch['q_seg'],batch['z_L']);f=model.forensic_context(a['F64'],a['z64'],a['support64'])
 lr,fr,_=model.rectification(l,f,a['support64']);lr,fr,_=model.exchange(lr,fr,a['support64'])
 conflict=ecolaf_discount(torch.stack((ml,mf),dim=2),classes=2)[1];comp,_=build_comparison_features(lr,fr,pl,pf,conflict,a['support64'])
 logit=model.utility_head.net[-1](model.utility_head.net[:-1](comp));return {'utility_logit':logit,'U':logit.sigmoid()*a['support64'].float(),'support':a['support64'],'p_L':pl,'p_F':pf}

def rectified_batch(rectifier,s64,sc,records,device):
 lengths=[r['F_full'].shape[-2]*r['F_full'].shape[-1] for r in records];maximum=max(lengths);b=len(records)
 evidence=torch.zeros(b,256,1,maximum,dtype=torch.bfloat16,device=device);coords=torch.zeros(b,maximum,2,device=device);valid=torch.zeros(b,maximum,dtype=torch.bool,device=device);support=[]
 for i,(r,n) in enumerate(zip(records,lengths)):
  evidence[i,:,:,0:n]=r['F_full'].to(device).flatten(1)[:,None,:]
  coords[i,:n]=r['coordinates'].to(device);valid[i,:n]=True
  if len(r['tile_boxes_yxyx'])==1:support.append(rectifier.semantic_support(sc[i:i+1],coords[i:i+1,:n],valid[i:i+1,:n])[0])
  else:support.append(((sc[i]>=0)&(sc[i]<=1)).all(-1))
 support=torch.stack(support)
 with torch.autocast(device_type=device.type,dtype=torch.bfloat16):return rectifier(s64,evidence,sc,coords,valid,semantic_support=support)

def language_batch(store,cache,positions,sam,device):
 ids=[cache['sample_ids'][i] for i in positions.tolist()];s64,_,targets,sc,_=phase4f_spatial_batch(store,ids,device)
 qseg=cache['q_seg'].index_select(0,positions).to(device=device,dtype=torch.bfloat16)
 with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):low=sam(qseg,s64)
 zl=torch.cat([sam_lowres_to_original_normalized(low[j:j+1],store.geometries[sid],output_hw=(256,256)) for j,sid in enumerate(ids)]).to(device=device,dtype=torch.bfloat16)
 return {'S64':s64,'q_seg':qseg,'z_L':zl},targets,sc,ids

def evaluate(model,rectifier,sam,store,cache,full,device):
 rec=[];model.eval();model.language_source.eval();model.forensic_source.eval();rectifier.eval()
 with torch.no_grad():
  for i,sid in enumerate(store.sample_ids):
   truth=store.original_masks[sid]
   if not bool(cache['valid'][i]):rec.append(invalid_record(sid,truth));continue
   pos=torch.tensor([i]);batch,_,sc,_=language_batch(store,cache,pos,sam,device);fr=[full[sid]]
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):u=utility_aligned(model,batch,aligned(fr,device))
   rv=rectified_batch(rectifier,batch['S64'],sc,fr,device);gate=hc.gate_to_sam_grid(u['U'],sc)*rv['support'].reshape(1,1,64,64)
   emb=hc.gated_embedding(batch['S64'],rv['image_embeddings'],gate)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   rec.append(metric_record(sid,inverse_sam_logits(low,store.geometries[sid]),truth))
   if (i+1)%200==0:print(json.dumps({'stage':'VAL','done':i+1,'total':len(store.sample_ids)}),flush=True)
 iou=np.array([r['foreground_iou'] for r in rec]);f1=np.array([r['foreground_f1'] for r in rec]);return {'n':len(rec),'mean_foreground_iou':float(iou.mean()),'mean_foreground_f1':float(f1.mean())},rec

def write_records(p,rec):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rec))

def finalize(history,selected_records,ids):
 current=rows(CURRENT);attr={r['sample_id']:r for r in rows(ATTR)}
 if [r['sample_id'] for r in current]!=ids:raise RuntimeError('current baseline order drift')
 paired_rows=[]
 for a,b in zip(current,selected_records):
  sid=a['sample_id'];x={'sample_id':sid,'a0_iou':a['a0_iou'],'a0_f1':a['a0_f1'],'a0_tp':a['a0_tp'],'a0_fp':a['a0_fp'],'a0_fn':a['a0_fn'],
   'a1_iou':b['foreground_iou'],'a1_f1':b['foreground_f1'],'a1_tp':b['tp'],'a1_fp':b['fp'],'a1_fn':b['fn'],
   'delta_iou':b['foreground_iou']-a['a0_iou'],'delta_f1':b['foreground_f1']-a['a0_f1'],'exact_crop_gt_coverage':attr[sid]['exact_crop_gt_coverage'],
   'aspect_ratio':attr[sid]['aspect_ratio'],'tile_count':a['tile_count']};paired_rows.append(x)
 overall=group_summary(paired_rows)
 def cov(v):
  if np.isclose(v,1):return '1.0'
  if v>=.9:return '[0.9,1.0)'
  if v>=.5:return '[0.5,0.9)'
  if v>0:return '(0,0.5)'
  return '0'
 coverage={k:group_summary([r for r in paired_rows if cov(r['exact_crop_gt_coverage'])==k]) for k in ('1.0','[0.9,1.0)','[0.5,0.9)','(0,0.5)','0')}
 aspect={'aspect_ratio_lt_1.2':group_summary([r for r in paired_rows if r['aspect_ratio']<1.2]),'aspect_ratio_ge_1.2':group_summary([r for r in paired_rows if r['aspect_ratio']>=1.2])}
 tiles={k:group_summary([r for r in paired_rows if (r['tile_count']==int(k[2:]) if k!='N>=4' else r['tile_count']>=4)]) for k in ('N=1','N=2','N=3','N>=4')}
 stable=overall['paired']['iou']['bootstrap_95_ci'][0]>0 and overall['paired']['f1']['bootstrap_95_ci'][0]>0
 verdict='FULL_FOV_I2_STABLY_BEATS_CURRENT_NEW_R1' if stable else 'FULL_FOV_I2_NOT_STABLY_BETTER'
 dump(OUT/'comparison.json',{'overall':overall,'coverage':coverage,'aspect':aspect,'tile_count':tiles});write_records(OUT/'selected_validation_predictions.jsonl',selected_records)
 summary={'schema':'phase6f4_summary_v1','status':'COMPLETE_STOP','verdict':verdict,'selected_epoch':max(history,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']))['epoch'],
  'selected_checkpoint_sha256':file_sha256(SELECTED),'overall':overall,'test_accessed':False,'official1000_accessed':False,'ood_accessed':False};dump(OUT/'summary.json',summary)
 DOC.write_text(f"""# Phase 6F.4 — Full-FOV I2 Joint Training\n\nStatus: **COMPLETE STOP**. C1, CLIP, Phase4C-A adapter, SAM and source heads remained frozen. Utility and Rectifier used the exact Phase6E.2 I2 seed-3407 random initialization; no selected new-R1 state was loaded.\n\n- selected epoch: `{summary['selected_epoch']}` by internal-validation canonical-G0 mean FG IoU\n- checkpoint SHA256: `{summary['selected_checkpoint_sha256']}`\n- current new R1 mean IoU: `{overall['A0']['mean_fg_iou']:.6f}`\n- Full-FOV I2 mean IoU: `{overall['A1']['mean_fg_iou']:.6f}`\n- paired IoU: `{overall['paired']['iou']}`\n- current new R1 mean F1: `{overall['A0']['mean_fg_f1']:.6f}`\n- Full-FOV I2 mean F1: `{overall['A1']['mean_fg_f1']:.6f}`\n- paired F1: `{overall['paired']['f1']}`\n\nCoverage/aspect/tile-count results are saved in `outputs/phase6f4_full_fov_i2/comparison.json`.\n\n```text\n{verdict}\n```\n\nNo test, Official1000, or OOD data was accessed. No staged Full-FOV pretraining was started.\n""")

def main():
 if (OUT/'summary.json').exists():raise RuntimeError('Phase6F.4 already complete')
 device=torch.device('cuda:0');torch.cuda.set_device(device);hc.seed_all();OUT.mkdir(parents=True,exist_ok=True);CKPTS.mkdir(parents=True,exist_ok=True)
 utility,_=hc.load_utility('a2',device);scale=json.load(open(ROOT/'outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json'))
 rectifier=load_rectifier(hd.CFG,float(scale['selected_gamma']),device);rectifier.requires_grad_(True)
 init={'utility_state_sha256':tensor_state_sha256(utility.state_dict()),'utility_trainable_sha256':tensor_state_sha256(hc.trainable_state(utility)),
       'rectifier_state_sha256':tensor_state_sha256(rectifier.state_dict())}
 expected=json.load(open(ROOT/'outputs/phase6e2_c1_specific_r1/i2/initialization_provenance.json'))['new_r1_initialization']
 if init['utility_trainable_sha256']!=expected['phase4hc_a2_random_utility_trainable_sha256'] or init['rectifier_state_sha256']!=expected['phase4f_random_rectifier_sha256']:raise RuntimeError('I2 initialization drift')
 if file_sha256(ROOT/'outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt') in (init.values()):raise RuntimeError('forbidden selected new R1 load')
 train_store=Phase4FStore(hd.CFG,'train');val_store=Phase4FStore(hd.CFG,'val');train_ids=train_store.sample_ids;val_ids=val_store.sample_ids
 train_full,train_status=load_full('train',train_ids);val_full,val_status=load_full('val',val_ids)
 train_cache=load_c1_cache('train',train_ids);val_cache=load_c1_cache('val',val_ids);valid=train_cache['valid'].bool();id_to_i={s:i for i,s in enumerate(train_ids)}
 targets=q.load_ids(train_ids,('target64',))['target64'];sam=load_sam_runtime(hd.CFG,device)
 manifest={'schema':'phase6f4_protocol_v1','status':'FROZEN_BEFORE_FIRST_OPTIMIZER_STEP','base':'frozen C1 epoch5/step2500','c1_sha256':C1_SHA,
  'initialization':init,'matched_phase6e2_i2_initialization':True,'selected_new_r1_loaded':False,'trainable':{'utility':sum(p.numel() for p in utility.parameters() if p.requires_grad),'rectifier':sum(p.numel() for p in rectifier.parameters() if p.requires_grad)},
  'recipe':{'optimizer':'AdamW','lr':LR,'weight_decay':WD,'batch':BATCH,'epochs':EPOCHS,'scheduler':'none','grad_clip':GRAD,'seed':SEED,'loss':'L_seg+L_relative+0.5*(cross_rank+spatial_rank)','selector':'internal validation canonical G0 mean IoU, tie earlier'},
  'only_change':'single-crop forensic evidence -> Phase6F.3 invariant Full-FOV evidence','cache':{'train':train_status,'val':val_status},'firewall':{'test':False,'official1000':False,'ood':False}}
 if manifest['trainable']!={'utility':371803,'rectifier':329985}:raise RuntimeError('trainable scope drift')
 dump(OUT/'protocol.json',manifest);frozen_before={'sam':tensor_state_sha256(sam.state_dict()),'source_heads':tensor_state_sha256(q.source_state(utility))}
 params=[p for p in utility.parameters() if p.requires_grad]+[p for p in rectifier.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
 cross={sid:train_ids[(i+1)%len(train_ids)] for i,sid in enumerate(train_ids)};history=[];updates=0
 for epoch in range(1,EPOCHS+1):
  began=time.time();order=list(train_ids);random.Random(SEED+1009*epoch).shuffle(order);sums=defaultdict(float);eligible=invalid=0;utility.train();utility.language_source.eval();utility.forensic_source.eval();rectifier.train()
  for begin in range(0,len(order),BATCH):
   block=order[begin:begin+BATCH];pos=torch.tensor([id_to_i[s] for s in block]);pos=pos[valid.index_select(0,pos)];invalid+=len(block)-len(pos);eligible+=len(pos)
   if not len(pos):continue
   batch,mask_targets,sc,bids=language_batch(train_store,train_cache,pos,sam,device);fr=[train_full[s] for s in bids];cross_fr=[train_full[cross[s]] for s in bids]
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
    match=utility_aligned(utility,batch,aligned(fr,device));crossed=utility_aligned(utility,batch,aligned(cross_fr,device));shuffled=utility_aligned(utility,batch,aligned(fr,device,True))
   rv=rectified_batch(rectifier,batch['S64'],sc,fr,device);gate=hc.gate_to_sam_grid(match['U'],sc)*rv['support'].reshape(len(pos),1,64,64);emb=hc.gated_embedding(batch['S64'],rv['image_embeddings'],gate)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   seg=mask_loss(low,mask_targets,hd.CFG);_,soft=q.target_delta({'p_L':match['p_L'],'p_F':match['p_F']},targets.index_select(0,pos).to(device));rel=q.image_balanced_loss(match['utility_logit'],soft,match['support']);cr=hc.rank_loss(match['U'],crossed['U'],match['support']);sr=hc.rank_loss(match['U'],shuffled['U'],match['support']);rank=.5*(cr+sr);total=seg['total']+rel+rank
   total.backward();norm=torch.nn.utils.clip_grad_norm_(params,GRAD);opt.step();updates+=1;n=len(pos)
   for k,v in [('seg_loss',seg['total']),('relative_loss',rel),('ranking_loss',rank),('cross_rank_loss',cr),('shuffle_rank_loss',sr),('total_loss',total)]:sums[k]+=float(v.detach())*n
   if updates<=3 or updates%100==0:print(json.dumps({'stage':'TRAIN','epoch':epoch,'update':updates,'loss':float(total.detach()),'grad_norm':float(norm)}),flush=True)
  metric,records=evaluate(utility,rectifier,sam,val_store,val_cache,val_full,device);row={'epoch':epoch,'optimizer_updates':updates,**{k:sums[k]/eligible for k in ('seg_loss','relative_loss','ranking_loss','cross_rank_loss','shuffle_rank_loss','total_loss')},'traversal_exposures':len(order),'optimization_eligible_exposures':eligible,'invalid_g0_exposures':invalid,'dev_g0_mean_iou':metric['mean_foreground_iou'],'dev_g0_mean_f1':metric['mean_foreground_f1'],'sample_order_sha256':ids_hash(order),'seconds':time.time()-began};history.append(row)
  with (OUT/'training_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
  write_records(OUT/f'validation/epoch_{epoch}.jsonl',records);payload={'schema':'phase6f4_full_fov_i2_checkpoint_v1','epoch':epoch,'optimizer_updates':updates,'utility_state':{k:v.detach().cpu() for k,v in utility.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in rectifier.state_dict().items()},'optimizer':opt.state_dict(),'validation_g0':metric,'initialization':init};p=CKPTS/f'epoch_{epoch}.pt';t=p.with_suffix('.pt.tmp');torch.save(payload,t);os.replace(t,p);print(json.dumps({'stage':'EPOCH_COMPLETE',**row}),flush=True)
 best=max(history,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']));shutil.copy2(CKPTS/f"epoch_{best['epoch']}.pt",SELECTED);selector={'status':'COMPLETE','primary':'internal validation canonical G0 mean IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(SELECTED),'candidates':history,'test_used':False,'official1000_used':False,'ood_used':False};dump(OUT/'selector.json',selector)
 frozen_after={'sam':tensor_state_sha256(sam.state_dict()),'source_heads':tensor_state_sha256(q.source_state(utility))}
 if frozen_after!=frozen_before:raise RuntimeError('frozen hash drift')
 manifest.update(status='TRAINING_COMPLETE_SELECTED_FROZEN',frozen_hash_before=frozen_before,frozen_hash_after=frozen_after,selected_checkpoint_sha256=file_sha256(SELECTED),selected_epoch=best['epoch'],optimizer_updates=updates);dump(OUT/'protocol.json',manifest)
 finalize(history,rows(OUT/f"validation/epoch_{best['epoch']}.jsonl"),val_ids)

if __name__=='__main__':main()
