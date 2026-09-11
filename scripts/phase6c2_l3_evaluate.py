#!/usr/bin/env python3
"""Evaluate L3 epoch checkpoints on internal-validation canonical G0 or TF."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts.phase4gf_formal_localization import full_inputs,load_dev
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid,gated_embedding
from scripts.phase6c2_l3_train import Cache,CFG,L2_SHA,load_modules
from tools.phase4c_b import inverse_sam_logits,metric_record
from tools.phase4e1 import summarize
from tools.phase4f import Phase4FStore,evidence_feature,load_evidence_source
from model.pcerf import sam_lowres_to_original_normalized
CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l3'); G0=Path('/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l2_val_g0'); OUT=ROOT/'outputs/phase6c2_multiseg_training/l3_validation'

def load_g0():
 by={}; ranks=[]
 for rank in range(2):
  x=torch.load(G0/f'rank{rank:02d}.pt',map_location='cpu',weights_only=False); off=x['offsets'].tolist(); ids=[]
  for i,r in enumerate(x['records']): by[r['sample_id']]=(x['q_seg'][off[i]:off[i+1]],r); ids.append(r['sample_id'])
  ranks.append(ids)
 order=[]
 for i in range(max(map(len,ranks))):
  for rows in ranks:
   if i<len(rows):order.append(rows[i])
 return by,order
def pair_metric(logit,target):
 pred=F.interpolate(logit[None,None].float(),target.shape[-2:],mode='bilinear',align_corners=False)[0,0].gt(0); gt=target.bool(); tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum()); den=tp+fp+fn; d2=2*tp+fp+fn
 return {'foreground_iou':tp/den if den else 1.,'foreground_f1':2*tp/d2 if d2 else 1.,'tp':tp,'fp':fp,'fn':fn}
def evaluate(epoch,mode,device):
 path=OUT/mode/f'epoch_{epoch:02d}.json'
 if path.exists() and json.loads(path.read_text()).get('status')=='COMPLETE':
  print(json.dumps({'epoch':epoch,'mode':mode,'status':'EXACT_REUSE'}));return
 utility,rectifier,sam=load_modules(device); state=torch.load(CKPT/f'epoch_{epoch}.pt',map_location='cpu',weights_only=False);utility.load_state_dict(state['utility_state']);rectifier.load_state_dict(state['rectifier_state']);utility.eval();rectifier.eval()
 store=Phase4FStore(CFG,'val'); source=load_evidence_source(CFG,'forensic_rect',device); dev=load_dev('g0'); index={s:i for i,s in enumerate(dev['sample_ids'])}; tf=Cache('val') if mode=='tf' else None; g0,order=load_g0() if mode=='g0' else ({},tf.ids)
 if order!=store.sample_ids:raise RuntimeError('validation canonical order drift')
 records=[];pairs=[];counts_ok=0
 with torch.no_grad():
  for n,sid in enumerate(order,1):
   if mode=='g0': qseg,base_record=g0[sid]; targets=None;ks=[len(qseg)]
   else:qseg,targets,ks=tf.batch([sid]);base_record=None
   truth=store.original_masks[sid]
   if not len(qseg): records.append({'sample_id':sid,'foreground_iou':0.,'foreground_f1':0.,'tp':0,'fp':0,'fn':int(truth.sum()),'valid_q_seg':False});continue
   qseg=qseg.to(device=device,dtype=torch.bfloat16); i=torch.tensor([index[sid]]); batch=full_inputs(dev,i,device);s64,raw,_,sc,cc,_=store.batch([sid],device);rep=torch.zeros(len(qseg),dtype=torch.long,device=device)
   evidence=evidence_feature(source,raw);base_s=s64.index_select(0,rep)
   with torch.autocast(device_type='cuda',enabled=False):base_low=sam(qseg,base_s.to(torch.bfloat16))
   zl=torch.cat([sam_lowres_to_original_normalized(base_low[j:j+1],store.geometries[sid],output_hw=(256,256)) for j in range(len(qseg))]).to(torch.bfloat16)
   for key in ('S64','F24','z_F24'):batch[key]=batch[key].repeat_interleave(len(qseg),0)
   batch['q_seg']=qseg;batch['z_L']=zl;batch['clip_geometries']=batch['clip_geometries']*len(qseg)
   valid=torch.ones(1,576,dtype=torch.bool,device=device)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):rv=rectifier(s64,evidence,sc,cc,valid);out=hc.utility_forward(utility,batch)
   gate=gate_to_sam_grid(out['U'],sc.index_select(0,rep))*rv['support'].reshape(1,1,64,64).repeat_interleave(len(qseg),0).float();adapted=gated_embedding(base_s,rv['image_embeddings'].repeat_interleave(len(qseg),0),gate)
   with torch.autocast(device_type='cuda',enabled=False):low=sam(qseg,adapted.to(torch.bfloat16))
   logits=[inverse_sam_logits(low[j:j+1],store.geometries[sid]) for j in range(len(qseg))]; union=torch.stack(logits).max(0).values;row=metric_record(sid,union,truth);row.update({'valid_q_seg':True,'pred_K':len(qseg),'gt_K':int(ks[0]) if mode=='tf' else base_record['gt_K']});records.append(row)
   if mode=='tf' and len(qseg)==len(targets):
    counts_ok+=1
    for j in range(len(qseg)):pairs.append({**pair_metric(low[j,0].cpu(),targets[j]),'sample_id':sid,'pair_index':j,'K':len(qseg)})
   if n%100==0:print(json.dumps({'epoch':epoch,'mode':mode,'done':n,'total':len(order)}),flush=True)
 result={'status':'COMPLETE','epoch':epoch,'mode':mode,'metrics':summarize(records),'records':records,'pair_metrics':summarize(pairs) if pairs else None,'pair_records':pairs,'count_consistent_images':counts_ok,'L2_base_sha256':L2_SHA,'firewall':{'internal_test':False,'external':False}}
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',nargs='+',type=int,required=True);ap.add_argument('--mode',choices=['g0','tf'],required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args();device=torch.device(a.device);torch.cuda.set_device(device)
 for e in a.epochs:evaluate(e,a.mode,device)
if __name__=='__main__':main()
