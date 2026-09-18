#!/usr/bin/env python3
"""Matched Phase6E.2 main R1 retraining with only block-17 forensic source."""
from __future__ import annotations
import csv,hashlib,json,os,random,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from scripts import phase4g1q_conditional_utility as q,phase4hc_direct_utility_arms as hc,phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import C1_SHA,load_c1_cache,c1_language_batch,phase4f_spatial_batch
from scripts.phase6f3_full_fov_frozen_replay import group_summary
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,invalid_record,load_sam_runtime,mask_loss

OUT=ROOT/'outputs/phase6g1_block17_single_layer_replacement';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g1_block17/r1');CACHE=Path('/data/yz/groundingLMM_official/cache/phase6g0_multilevel_dense_clip_audit/late');ADAPTER=OUT/'adapter/selected.pt';SELECTED=OUT/'r1/selected_checkpoint.pt';SEED,EPOCHS,BATCH,LR,WD=3407,10,8,1e-4,1e-4
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def write_rows(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def ids_hash(x):return hashlib.sha256('\n'.join(x).encode()).hexdigest()
class LayerStore:
 def __init__(self,split):
  self.values=[];self.loc={};self.ids=[]
  for p in sorted((CACHE/split).glob('shard_*.pt')):
   x=torch.load(p,map_location='cpu',weights_only=False)
   if x.get('block_index')!=17:raise RuntimeError('block17 cache drift')
   j=len(self.values);self.values.append(x['features'])
   for i,r in enumerate(x['records']):self.ids.append(r['sample_id']);self.loc[r['sample_id']]=(j,i)
 def batch(self,ids,device):return torch.stack([self.values[self.loc[s][0]][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32)
def adapter(device):
 x=torch.load(ADAPTER,map_location='cpu',weights_only=False);m=CLIPSpatialArm(blocks=3);m.load_state_dict(x['model'],strict=True);return m.to(device).eval().requires_grad_(False)
def forensic(m,layer,ids,device):
 with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=m(layer.batch(ids,device),return_features=True)
 return o['F_forensic'].detach(),o['logits'].detach()
def evaluate(model,rectifier,sam,adapt,layer,store,dev,cache,device):
 rec=[];model.eval();rectifier.eval()
 with torch.no_grad():
  for i,sid in enumerate(dev['sample_ids']):
   if not bool(cache['valid'][i]):rec.append(invalid_record(sid,dev['original_masks'][i]));continue
   idx=torch.tensor([i]);batch,_=c1_language_batch(dev,cache,idx,store,sam,device);f,z=forensic(adapt,layer,[sid],device);batch['F24']=f;batch['z_F24']=z;s64,_,_,sc,cc=phase4f_spatial_batch(store,[sid],device);valid=torch.ones(1,576,dtype=torch.bool,device=device)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=rectifier(s64,f,sc,cc,valid);u=hc.utility_forward(model,batch)
   gate=hc.gate_to_sam_grid(u['U'],sc)*rv['support'].reshape(1,1,64,64);emb=hc.gated_embedding(s64,rv['image_embeddings'],gate)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   rec.append(metric_record(sid,inverse_sam_logits(low,dev['sam_geometries'][i]),dev['original_masks'][i]))
   if (i+1)%200==0:print(json.dumps({'stage':'VAL','done':i+1,'total':len(dev['sample_ids'])}),flush=True)
 return summarize_extended(rec),rec
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);hc.seed_all();root=OUT/'r1';root.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
 if (root/'summary.json').exists():print(json.dumps({'status':'ALREADY_COMPLETE'}));return
 adapt=adapter(device);model,rectifier,rectpath=hd.load_common(device);rectifier.requires_grad_(True);audit=json.load(open(ROOT/'outputs/phase4hd/rectifier_audit.json'))['common_initialization'];init={'utility_state_sha256':tensor_state_sha256(model.state_dict()),'rectifier_state_sha256':tensor_state_sha256(rectifier.state_dict())}
 if init['utility_state_sha256']!=audit['utility_state_sha256'] or init['rectifier_state_sha256']!=audit['rectifier_state_sha256']:raise RuntimeError('Phase6E2 main initialization drift')
 train_store,val_store=Phase4FStore(hd.CFG,'train'),Phase4FStore(hd.CFG,'val');ids=train_store.sample_ids;dev=load_dev('g0');train_cache=load_c1_cache('train',ids);val_cache=load_c1_cache('val',dev['sample_ids']);train_layer,val_layer=LayerStore('train'),LayerStore('val')
 if train_layer.ids!=ids or val_layer.ids!=dev['sample_ids']:raise RuntimeError('block17 population/order drift')
 data=q.load_ids(ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries'));valid_mask=train_cache['valid'].bool();id_to_i={s:i for i,s in enumerate(ids)};sam=load_sam_runtime(hd.CFG,device);frozen={'sam':tensor_state_sha256(sam.state_dict()),'adapter':tensor_state_sha256(adapt.state_dict()),'heads':tensor_state_sha256(q.source_state(model))}
 protocol={'schema':'phase6g1_r1_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','base':'C1 epoch5/step2500 frozen','only_change':'block22 + Phase4C-A adapter -> block17 + matched retrained Phase4C-A adapter','initialization':{**init,'kind':'A2_epoch3_plus_Phase4F_epoch9','utility_source':str(hd.A2_SELECTED.resolve()),'rectifier_source':str(rectpath.resolve()),'current_selected_new_r1_loaded':False},'adapter':{'path':str(ADAPTER.resolve()),'sha256':file_sha256(ADAPTER)},'recipe':hd.config_contract(),'actual_population':{'train':len(ids),'train_valid':int(valid_mask.sum()),'validation':len(dev['sample_ids']),'validation_valid':int(val_cache['valid'].sum())},'firewall':{'test':False,'official1000':False,'ood':False}};dump(root/'protocol.json',protocol)
 params=[p for p in model.parameters() if p.requires_grad]+[p for p in rectifier.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD);cross={s:ids[(i+1)%len(ids)] for i,s in enumerate(ids)};perm=torch.randperm(576,generator=torch.Generator().manual_seed(SEED)).to(device);history=[];updates=0;start=0;existing=sorted(CKPT.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);model.load_state_dict(x['utility_state']);rectifier.load_state_dict(x['rectifier_state']);opt.load_state_dict(x['optimizer']);start=x['epoch'];updates=x['optimizer_updates'];history=json.load(open(root/'training_curve.json'))
 for epoch in range(start+1,EPOCHS+1):
  began=time.time();order=list(ids);random.Random(SEED+1009*epoch).shuffle(order);s=defaultdict(float);eligible=invalid=0;model.train();model.language_source.eval();model.forensic_source.eval();rectifier.train()
  for b in range(0,len(order),BATCH):
   block=order[b:b+BATCH];pos=torch.tensor([id_to_i[x] for x in block]);idx=pos[valid_mask.index_select(0,pos)];invalid+=len(block)-len(idx);eligible+=len(idx)
   if not len(idx):continue
   bids=[ids[i] for i in idx.tolist()];batch,_=c1_language_batch(data,train_cache,idx,train_store,sam,device);f,z=forensic(adapt,train_layer,bids,device);batch['F24']=f;batch['z_F24']=z;cf,cz=forensic(adapt,train_layer,[cross[x] for x in bids],device);crossed=dict(batch);crossed['F24']=cf;crossed['z_F24']=cz;s64,targets,sc,cc,_=phase4f_spatial_batch(train_store,bids,device);valid=torch.ones(len(idx),576,dtype=torch.bool,device=device);opt.zero_grad(set_to_none=True)
   matched=hc.utility_forward(model,batch);crossout=hc.utility_forward(model,crossed);shuffled=hc.utility_forward(model,batch,permutation=perm)
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16):rv=rectifier(s64,f,sc,cc,valid)
   gate=hc.gate_to_sam_grid(matched['U'],sc)*rv['support'].reshape(len(idx),1,64,64);emb=hc.gated_embedding(s64,rv['image_embeddings'],gate)
   with torch.autocast(device_type=device.type,enabled=False):low=sam(batch['q_seg'],emb.to(torch.bfloat16))
   seg=mask_loss(low,targets,hd.CFG);_,soft=q.target_delta({'p_L':matched['p_L'],'p_F':matched['p_F']},data['target64'].index_select(0,idx).to(device));rel=q.image_balanced_loss(matched['utility_logit'],soft,matched['support']);cr=hc.rank_loss(matched['U'],crossout['U'],matched['support']);sr=hc.rank_loss(matched['U'],shuffled['U'],matched['support']);rank=.5*(cr+sr);total=seg['total']+rel+rank;total.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);opt.step();updates+=1;n=len(idx)
   for k,v in [('seg_loss',seg['total']),('relative_loss',rel),('ranking_loss',rank),('cross_rank_loss',cr),('shuffle_rank_loss',sr),('total_loss',total)]:s[k]+=float(v.detach())*n
   if updates<=3 or updates%100==0:print(json.dumps({'stage':'R1_TRAIN','epoch':epoch,'update':updates,'loss':float(total.detach()),'grad_norm':float(norm)}),flush=True)
  metric,records=evaluate(model,rectifier,sam,adapt,val_layer,val_store,dev,val_cache,device);row={'epoch':epoch,'optimizer_updates':updates,**{k:s[k]/eligible for k in ('seg_loss','relative_loss','ranking_loss','cross_rank_loss','shuffle_rank_loss','total_loss')},'traversal_exposures':len(order),'optimization_eligible_exposures':eligible,'invalid_g0_exposures':invalid,'dev_g0_mean_iou':metric['mean_foreground_iou'],'dev_g0_mean_f1':metric['mean_foreground_f1'],'sample_order_sha256':ids_hash(order),'seconds':time.time()-began};history.append(row);dump(root/'training_curve.json',history);write_rows(root/f'validation/epoch_{epoch}.jsonl',records);payload={'schema':'phase6g1_block17_r1_checkpoint_v1','epoch':epoch,'optimizer_updates':updates,'utility_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'rectifier_state':{k:v.detach().cpu() for k,v in rectifier.state_dict().items()},'optimizer':opt.state_dict(),'validation_g0':metric,'adapter_sha256':file_sha256(ADAPTER),'initialization':init};p=CKPT/f'epoch_{epoch}.pt';t=p.with_suffix('.pt.tmp');torch.save(payload,t);os.replace(t,p);print(json.dumps({'stage':'R1_EPOCH_COMPLETE',**row}),flush=True)
 best=max(history,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']));shutil.copy2(CKPT/f"epoch_{best['epoch']}.pt",SELECTED);selected=rows_file(root/f"validation/epoch_{best['epoch']}.jsonl");baseline_raw=[json.loads(x) for x in open(ROOT/'outputs/phase6f3_full_fov_frozen_replay/per_sample_results.jsonl')];baseline=[{'sample_id':r['sample_id'],'foreground_iou':r['a0_iou'],'foreground_f1':r['a0_f1'],'tp':r['a0_tp'],'fp':r['a0_fp'],'fn':r['a0_fn']} for r in baseline_raw]
 if [r['sample_id'] for r in baseline]!=[r['sample_id'] for r in selected]:raise RuntimeError('A0/A1 pairing drift')
 paired_rows=[{'sample_id':a['sample_id'],'a0_iou':a['foreground_iou'],'a0_f1':a['foreground_f1'],'a0_tp':a['tp'],'a0_fp':a['fp'],'a0_fn':a['fn'],'a1_iou':b['foreground_iou'],'a1_f1':b['foreground_f1'],'a1_tp':b['tp'],'a1_fp':b['fp'],'a1_fn':b['fn'],'delta_iou':b['foreground_iou']-a['foreground_iou'],'delta_f1':b['foreground_f1']-a['foreground_f1']} for a,b in zip(baseline,selected)];comparison=group_summary(paired_rows);d=comparison['paired']['iou'];decision='BLOCK17_R1_SUPPORTED' if d['mean_delta']>0 and d['bootstrap_95_ci'][0]>0 else ('BLOCK17_R1_NOT_STABLY_BETTER' if d['mean_delta']>0 else 'BLOCK17_R1_NOT_SUPPORTED');write_rows(root/'selected_validation_predictions.jsonl',selected);dump(root/'selector.json',{'status':'COMPLETE','primary':'internal validation canonical G0 mean IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(SELECTED),'candidates':history,'test_used':False,'official1000_used':False,'ood_used':False});dump(root/'comparison.json',comparison);after={'sam':tensor_state_sha256(sam.state_dict()),'adapter':tensor_state_sha256(adapt.state_dict()),'heads':tensor_state_sha256(q.source_state(model))}
 if after!=frozen:raise RuntimeError('frozen hash drift')
 protocol.update(status='COMPLETE_SELECTED_FROZEN',selected_epoch=best['epoch'],selected_checkpoint_sha256=file_sha256(SELECTED),frozen_hash_before=frozen,frozen_hash_after=after);dump(root/'protocol.json',protocol);summary={'status':'COMPLETE_STOP','decision':decision,'selected_epoch':best['epoch'],'selected_checkpoint_sha256':file_sha256(SELECTED),'comparison':comparison,'firewall':{'test_accessed':False,'official1000_accessed':False,'ood_accessed':False}};dump(root/'summary.json',summary);render(summary)
def rows_file(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def render(s):
 c=s['comparison'];p=c['paired'];text=f"""# Phase 6G.1 — CLIP Block-17 Single-Layer Replacement\n\nStatus: **COMPLETE STOP**. The only representation change is legacy center-crop CLIP block22 to block17. The Phase4C-A adapter was retrained with its matched original protocol; R1 restarted from Phase4H-C A2 epoch3 Utility + Phase4F epoch9 Rectifier, never from current selected new R1.\n\n| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |\n|---|---:|---:|---:|---:|---:|\n| A0 block22 current new R1 | {c['A0']['mean_fg_iou']:.6f} | {c['A0']['median_fg_iou']:.6f} | {c['A0']['mean_fg_f1']:.6f} | {c['A0']['global_fg_iou']:.6f} | {c['A0']['global_fg_f1']:.6f} |\n| A1 block17 matched new R1 | {c['A1']['mean_fg_iou']:.6f} | {c['A1']['median_fg_iou']:.6f} | {c['A1']['mean_fg_f1']:.6f} | {c['A1']['global_fg_iou']:.6f} | {c['A1']['global_fg_f1']:.6f} |\n\n- IoU paired: `{p['iou']}`\n- F1 paired: `{p['f1']}`\n\n```text\n{s['decision']}\n```\n\nNo test, Official1000, OOD, block11 or multi-level fusion was run.\n""";(ROOT/'docs/phase6g1_block17_single_layer_replacement.md').write_text(text)
if __name__=='__main__':main()
