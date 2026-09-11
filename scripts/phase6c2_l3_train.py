#!/usr/bin/env python3
"""Dual-GPU Phase 6C.2 L3 training on frozen selected-L2 multiSEG states.

The two R1 modules are shared across T=sum(K). Gradients are manually reduced
to preserve a true cross-rank global per-mask mean for variable K.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts.phase4gf_formal_localization import full_inputs
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid, gated_embedding
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, evidence_feature, load_evidence_source, load_sam_runtime
from model.pcerf import sam_lowres_to_original_normalized

SEED,EPOCHS,GLOBAL_BATCH,LR,WD,CLIP=3407,10,8,1e-4,1e-4,1.0
L2=Path('/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2/best/checkpoint/mp_rank_00_model_states.pt')
L1=ROOT/'outputs/phase4hd/r1/selected_checkpoint.pt'
L2_SHA='dbd7daa8322fe77c8bbfd80223a98ec1e6a4a2c64b4de3b09b09e592ef71d6dd'; L1_SHA='9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5'
CACHE=Path('/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l3_l2_canonical')
CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l3')
OUT=ROOT/'outputs/phase6c2_multiseg_training/l3_training'
CFG=yaml.safe_load((ROOT/'configs/phase4f_language_preserving_rectification.yaml').read_text())

def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def ids_hash(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def seed_all():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class Cache:
 def __init__(self,split):
  root=CACHE/split; done=json.loads((root/'complete.json').read_text()); index=json.loads((root/'index.json').read_text())
  if done['status']!='COMPLETE' or done['L2_checkpoint_sha256']!=L2_SHA: raise RuntimeError('cache provenance drift')
  self.ids=index['sample_ids']; self.offsets=index['offsets']; self.records=index['records']
  self.q=np.memmap(root/'q_seg.float16.bin',mode='r',dtype=np.float16,shape=tuple(index['q_shape']))
  self.target=np.memmap(root/'target1024_packed.uint8.bin',mode='r',dtype=np.uint8,shape=tuple(index['target_shape']))
  self.pos={sid:i for i,sid in enumerate(self.ids)}
 def batch(self,ids):
  qs=[]; targets=[]; ks=[]
  for sid in ids:
   i=self.pos[sid]; a,b=self.offsets[i:i+2]; ks.append(b-a)
   qs.append(torch.from_numpy(np.array(self.q[a:b],copy=True)))
   packed=np.array(self.target[a:b],copy=True); bits=np.unpackbits(packed,axis=1,count=1024*1024)
   targets.append(torch.from_numpy(bits.reshape(b-a,1024,1024).copy()).bool())
  return torch.cat(qs),torch.cat(targets),ks

def load_modules(device):
 if file_sha256(L2)!=L2_SHA or file_sha256(L1)!=L1_SHA: raise RuntimeError('selected source drift')
 utility,rectifier,_=__import__('scripts.phase4hd_rectifier_unfreeze_control',fromlist=['load_common']).load_common(device)
 state=torch.load(L1,map_location='cpu',weights_only=False); utility.load_state_dict(state['utility_state'],strict=True); rectifier.load_state_dict(state['rectifier_state'],strict=True)
 utility.language_source.eval().requires_grad_(False); utility.forensic_source.eval().requires_grad_(False); rectifier.requires_grad_(True)
 sam=load_sam_runtime(CFG,device); l2=torch.load(L2,map_location='cpu',weights_only=False)['module']
 def sub(prefix): return {k[len(prefix):]:v for k,v in l2.items() if k.startswith(prefix)}
 prefixes=['base_model.model.model.grounding_encoder.prompt_encoder.','base_model.model.model.grounding_encoder.mask_decoder.']
 ps,ms=sub(prefixes[0]),sub(prefixes[1])
 # DeepSpeed excluded the frozen prompt-encoder parameters from L2.  They are
 # therefore inherited exactly from the frozen P1 SAM runtime; the complete
 # trainable L2 mask-decoder state must be present and is loaded strictly.
 if not ms: raise RuntimeError('L2 mask-decoder state missing')
 sam.mask_decoder.load_state_dict(ms,strict=True); sam.eval().requires_grad_(False)
 return utility,rectifier,sam

def sync_grads(params,world):
 for p in params:
  if p.grad is None: raise RuntimeError('missing trainable gradient')
  dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad.div_(world)

def global_scale(local_n,device,world):
 count=torch.tensor(float(local_n),device=device); dist.all_reduce(count)
 if count.item()<=0: raise RuntimeError('empty global mask batch')
 return world*local_n/count.item(),int(count.item())

def seg_parts(low,target):
 logits=F.interpolate(low.float(),size=target.shape[-2:],mode='bilinear',align_corners=False); y=target.float()[:,None]
 bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none').flatten(1).mean(1)
 p=logits.sigmoid(); inter=2*(p/1000*y).flatten(1).sum(1); union=(p/1000).flatten(1).sum(1)+(y/1000).flatten(1).sum(1)
 dice=1-(inter+1e-6)/(union+1e-6)
 return bce,dice

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--smoke-steps',type=int,default=0); ap.add_argument('--resume',action='store_true'); ap.add_argument('--local-rank','--local_rank',type=int,default=-1); a=ap.parse_args()
 dist.init_process_group('nccl'); rank,world=dist.get_rank(),dist.get_world_size(); local=int(os.environ.get('LOCAL_RANK',a.local_rank)); torch.cuda.set_device(local); device=torch.device('cuda',local)
 if world!=2: raise RuntimeError('formal L3 contract requires two ranks'); seed_all()
 cache=Cache('train'); store=Phase4FStore(CFG,'train'); ids=train_ids()
 if ids!=cache.ids or ids!=store.sample_ids: raise RuntimeError('canonical train identity drift')
 data=q.load_ids(ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries')); valid=data['valid_g0'].bool(); pos={sid:i for i,sid in enumerate(ids)}
 utility,rectifier,sam=load_modules(device); source=load_evidence_source(CFG,'forensic_rect',device)
 params=[p for p in utility.parameters() if p.requires_grad]+[p for p in rectifier.parameters() if p.requires_grad]
 if sum(p.numel() for p in params)!=701788: raise RuntimeError('trainable parameter scope drift')
 optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=WD); rows=[]; updates=0; start_epoch=1
 if a.resume and not a.smoke_steps:
  existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
  if existing:
   saved=torch.load(existing[-1],map_location='cpu',weights_only=False)
   utility.load_state_dict(saved['utility_state'],strict=True); rectifier.load_state_dict(saved['rectifier_state'],strict=True); optimizer.load_state_dict(saved['optimizer'])
   updates=int(saved['optimizer_updates']); start_epoch=int(saved['epoch'])+1
   history=OUT/'training_history.csv'
   if history.exists(): rows=list(csv.DictReader(history.open()))
 permutation=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device)
 frozen_before={'sam':tensor_state_sha256(sam.state_dict()),'source':tensor_state_sha256(source.state_dict()),'utility_sources':tensor_state_sha256(q.source_state(utility))}
 if rank==0:
  OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
  dump(OUT/'execution_manifest.json',{'schema':'phase6c2_l3_execution_v1','status':'FROZEN_BEFORE_FIRST_STEP','L3_base':{'path':str(L2),'sha256':L2_SHA},'L1_import':{'path':str(L1),'sha256':L1_SHA,'keys':['utility_state','rectifier_state']},'trainable':{'utility':371803,'rectifier':329985,'shared_copies':1},'optimizer':{'name':'AdamW','lr':LR,'weight_decay':WD,'global_image_batch':GLOBAL_BATCH,'epochs':EPOCHS,'scheduler':'none','clip':CLIP},'loss':{'seg':'2*BCE+0.5*Dice','normalization':'cross-rank global per-mask mean'},'firewall':{'internal_test':False,'external':False}})
 dist.barrier(); epochs=1 if a.smoke_steps else EPOCHS
 for epoch in range(start_epoch,epochs+1):
  began=time.time(); order=list(ids); random.Random(SEED+1009*epoch).shuffle(order); sums=defaultdict(float); masks_seen=images_seen=invalid_seen=0
  utility.train(); utility.language_source.eval(); utility.forensic_source.eval(); rectifier.train()
  for step,begin in enumerate(range(0,len(order),GLOBAL_BATCH),1):
   global_ids=order[begin:begin+GLOBAL_BATCH]; local_ids=global_ids[rank::world]; ix=torch.tensor([pos[s] for s in local_ids]); keep=valid.index_select(0,ix); invalid_seen+=int((~keep).sum()); ix=ix[keep]; local_ids=[ids[i] for i in ix.tolist()]
   if not local_ids: raise RuntimeError('rank-local empty eligible batch')
   qseg,target,ks=cache.batch(local_ids); qseg=qseg.to(device=device,dtype=torch.bfloat16); target=target.to(device,non_blocking=True); T=len(qseg)
   batch=full_inputs(data,ix,device); s64,raw,_,sc,cc,_=store.batch(local_ids,device)
   with torch.no_grad():
    evidence=evidence_feature(source,raw); rep=torch.repeat_interleave(torch.arange(len(local_ids),device=device),torch.tensor(ks,device=device))
    base_s=s64.index_select(0,rep)
    with torch.autocast(device_type='cuda',enabled=False): base_low=sam(qseg,base_s.to(torch.bfloat16))
    zl=torch.cat([sam_lowres_to_original_normalized(base_low[j:j+1],store.geometries[local_ids[int(rep[j])]],output_hw=(256,256)) for j in range(T)]).to(torch.bfloat16)
   for key in ('S64','F24','z_F24'):
    batch[key]=batch[key].index_select(0,rep)
   batch['q_seg']=qseg; batch['z_L']=zl; batch['clip_geometries']=[batch['clip_geometries'][int(i)] for i in rep.cpu().tolist()]
   cross_ix=(ix+1)%len(ids); crossed=dict(batch)
   crossed['F24']=data['F24'].index_select(0,cross_ix).to(device).index_select(0,rep)
   crossed['z_F24']=data['z_F24'].index_select(0,cross_ix).to(device).index_select(0,rep)
   valid_e=torch.ones(len(local_ids),576,dtype=torch.bool,device=device)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    rv=rectifier(s64,evidence,sc,cc,valid_e); out=hc.utility_forward(utility,batch)
    cross_out=hc.utility_forward(utility,crossed); shuffle_out=hc.utility_forward(utility,batch,permutation=permutation)
   gate=gate_to_sam_grid(out['U'],sc.index_select(0,rep))*rv['support'].reshape(len(local_ids),1,64,64).index_select(0,rep).float()
   adapted=gated_embedding(base_s,rv['image_embeddings'].index_select(0,rep),gate)
   with torch.autocast(device_type='cuda',enabled=False): low=sam(qseg,adapted.to(torch.bfloat16))
   bce,dice=seg_parts(low,target); pair64=F.interpolate(target[:,None].float(),(64,64),mode='nearest')[:,0]
   _,soft=q.target_delta({'p_L':out['p_L'],'p_F':out['p_F']},pair64)
   relative=q.image_balanced_loss(out['utility_logit'],soft,out['support'])
   cross_rank=hc.rank_loss(out['U'],cross_out['U'],out['support']); shuffle_rank=hc.rank_loss(out['U'],shuffle_out['U'],out['support']); ranking=.5*(cross_rank+shuffle_rank)
   scale,global_t=global_scale(T,device,world); seg=(2*bce.mean()+.5*dice.mean()); total=scale*(seg+relative+ranking)
   optimizer.zero_grad(set_to_none=True); total.backward(); sync_grads(params,world); norm=torch.nn.utils.clip_grad_norm_(params,CLIP); optimizer.step(); updates+=1
   masks_seen+=T; images_seen+=len(local_ids); sums['seg']+=float(seg)*T; sums['relative']+=float(relative)*T; sums['ranking']+=float(ranking)*T
   if rank==0 and (updates<=3 or updates%100==0): print(json.dumps({'stage':'L3_TRAIN','epoch':epoch,'update':updates,'global_masks':global_t,'loss':float(total),'grad_norm':float(norm)}),flush=True)
   if a.smoke_steps and step>=a.smoke_steps: break
  totals=torch.tensor([images_seen,masks_seen,invalid_seen,sums['seg'],sums['relative'],sums['ranking']],dtype=torch.float64,device=device); dist.all_reduce(totals)
  row={'epoch':epoch,'optimizer_updates':updates,'eligible_image_exposures':int(totals[0]),'mask_exposures':int(totals[1]),'invalid_g0_exposures':int(totals[2]),'seg_loss':float(totals[3]/totals[1]),'relative_loss':float(totals[4]/totals[1]),'ranking_loss':float(totals[5]/totals[1]),'sample_order_sha256':ids_hash(order),'seconds':time.time()-began}
  if rank==0:
   combined=dict(row); rows.append(combined)
   with (OUT/('smoke_history.csv' if a.smoke_steps else 'training_history.csv')).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=combined.keys()); w.writeheader(); w.writerows(rows)
   payload={'schema':'phase6c2_l3_checkpoint_v1','epoch':epoch,'optimizer_updates':updates,'utility_state':{k:v.detach().cpu() for k,v in utility.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in rectifier.state_dict().items()},'optimizer':optimizer.state_dict(),'L2_base_sha256':L2_SHA,'L1_import_sha256':L1_SHA}
   path=CKPT/(f'smoke_epoch_{epoch}.pt' if a.smoke_steps else f'epoch_{epoch}.pt'); tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
  dist.barrier()
 frozen_after={'sam':tensor_state_sha256(sam.state_dict()),'source':tensor_state_sha256(source.state_dict()),'utility_sources':tensor_state_sha256(q.source_state(utility))}
 if frozen_before!=frozen_after: raise RuntimeError('frozen source mutation')
 train_hashes={'utility':tensor_state_sha256(utility.state_dict()),'rectifier':tensor_state_sha256(rectifier.state_dict())}; gathered_hashes=[None]*world; dist.all_gather_object(gathered_hashes,train_hashes)
 if len({json.dumps(x,sort_keys=True) for x in gathered_hashes})!=1: raise RuntimeError('rank parameter synchronization drift')
 if rank==0: dump(OUT/('smoke_complete.json' if a.smoke_steps else 'train_complete.json'),{'status':'SMOKE_PASS' if a.smoke_steps else 'COMPLETE','epochs':epochs,'optimizer_updates':updates,'frozen_sources_exact':True,'rank_parameters_exact':True,'trainable_state_sha256':train_hashes,'selector_pending':not bool(a.smoke_steps),'firewall':{'internal_test':False,'external':False}})
 dist.destroy_process_group()

if __name__=='__main__': main()
