#!/usr/bin/env python3
"""Evaluate one frozen selected Full-FOV R1 on Official1000, never select it."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import torch,yaml
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4hc_direct_utility_arms as hc,phase4hd_rectifier_unfreeze_control as hd,p1_r1_reusable_matrix as matrix
from scripts.phase2a_final_evaluate import load_model
from scripts.phase6e1_c1_old_r1_transfer import C1_CFG_PATH,C1_CKPT,OUT as E1_OUT
from scripts.phase6f4_full_fov_i2_train import aligned,rectified_batch,utility_aligned
from scripts.phase6f3_full_fov_frozen_replay import group_summary
from scripts.phase6f1_coverage_attribution import region_masks
from tools.full_fov_forensic import acquire_full_fov,full_fov_geometry
from tools.phase3c1 import geometry_for
from tools.phase3f_aogd import core_model
from tools.phase4c_b import file_sha256,inverse_sam_logits,metric_record
from tools.phase4e1 import compare,sam_coordinates,summarize
from tools.phase4f import load_sam_runtime

C1_SHA='85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff';SEED=3407
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def append(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');f.write(json.dumps(x,ensure_ascii=False)+'\n');f.close()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--selector',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--arm-name',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args();device=torch.device(a.device);torch.cuda.set_device(device)
 ckpt=Path(a.checkpoint);selector=Path(a.selector);out=Path(a.output_dir);records=out/'official1000_predictions.jsonl';sel=json.load(open(selector))
 if sel.get('official1000_used') or sel.get('test_used') or sel.get('ood_used'):raise RuntimeError('selector firewall failed')
 if file_sha256(C1_CKPT)!=C1_SHA:raise RuntimeError('C1 checkpoint drift')
 state=torch.load(ckpt,map_location='cpu',weights_only=False);utility,_=hc.load_utility('a2',device);scale=json.load(open(ROOT/'outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json'));from tools.phase4f import load_rectifier
 rectifier=load_rectifier(hd.CFG,float(scale['selected_gamma']),device);utility.load_state_dict(state['utility_state'],strict=True);rectifier.load_state_dict(state['rectifier_state'],strict=True);utility.eval().requires_grad_(False);rectifier.eval().requires_grad_(False)
 sam=load_sam_runtime(hd.CFG,device);cfg=yaml.safe_load(C1_CFG_PATH.read_text());conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1'];model,tokenizer,_=load_model(cfg,C1_CKPT,device,expected_step=2500,expected_epoch=5);model.eval().requires_grad_(False);model.rine_conditioner.eval().requires_grad_(False)
 core=core_model(model);tower=model.get_model().get_vision_tower();adapter=__import__('tools.phase4f',fromlist=['load_evidence_source']).load_evidence_source(hd.CFG,'forensic_rect',device).eval().requires_grad_(False)
 full,indices=matrix.fake_dataset(tokenizer,'official1000');c1_rows=rows(E1_OUT/'c1_records.jsonl');c1={x['sample_id']:x for x in c1_rows};existing=rows(records) if records.exists() else [];expected=[full.rows[i]['sample_id'] for i in indices[:len(existing)]]
 if [x['sample_id'] for x in existing]!=expected:raise RuntimeError('resume prefix drift')
 for ordinal,index in enumerate(indices[len(existing):],start=len(existing)):
  sample=full[index];sid=sample['sample_id'];target=torch.as_tensor(sample['masks']).bool().any(0);old=c1[sid];hidden=torch.load(old['c1_seg_hidden_path'],map_location='cpu',weights_only=True)
  if hidden.shape[0]!=int(old['seg_count']):raise RuntimeError('SEG hidden count drift')
  if not len(hidden):r={'sample_id':sid,'foreground_iou':0.,'foreground_f1':0.,'tp':0,'fp':0,'fn':int(target.sum()),'valid_q_seg':False}
  else:
   with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):
    qvals=core.model.text_hidden_fcs[0](hidden.to(device=device,dtype=torch.bfloat16));raw=model.get_grounding_encoder_embs(sample['grounding_enc_image'][None].to(device=device,dtype=torch.bfloat16))
   with Image.open(full.rows[index]['image_path']) as image:v=acquire_full_fov(image,tower.image_processor,tower.vision_tower,adapter,utility.forensic_source,utility.temperature_f,device)
   fr={'sample_id':sid,'F_full':v['F_full'][0].to(torch.bfloat16).cpu(),'z_full':v['z_full'][0].to(torch.bfloat16).cpu(),'mass_full':v['mass_full'][0].cpu(),'support_full':v['support_full'][0].cpu(),'coordinates':v['coordinates'].cpu(),'tile_boxes_yxyx':v['geometry'].tile_boxes_yxyx}
   h,w=target.shape;sg=geometry_for('sam',(h,w));s64=sam_lowres_to_original_normalized(raw,sg,output_hw=(64,64)).to(torch.bfloat16);sc=sam_coordinates(sg,grid=64)[None].to(device);logits=[]
   for qone in qvals:
    qone=qone[None].to(torch.bfloat16)
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):low0=sam(qone,raw.to(torch.bfloat16))
    zl=sam_lowres_to_original_normalized(low0,sg,output_hw=(256,256)).to(torch.bfloat16);batch={'S64':s64,'q_seg':qone,'z_L':zl}
    with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):u=utility_aligned(utility,batch,aligned([fr],device))
    rv=rectified_batch(rectifier,s64,sc,[fr],device);gate=hc.gate_to_sam_grid(u['U'],sc)*rv['support'].reshape(1,1,64,64);emb=hc.gated_embedding(s64,rv['image_embeddings'],gate)
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):low=sam(qone,emb.to(torch.bfloat16))
    logits.append(inverse_sam_logits(low,sg))
   r=metric_record(sid,torch.stack(logits).amax(0),target);r['valid_q_seg']=True;r['tile_count']=len(fr['tile_boxes_yxyx'])
  cg=geometry_for('clip',tuple(target.shape));crop=region_masks(target,cg)['crop'];r['aspect_ratio']=max(target.shape)/min(target.shape);r['exact_crop_gt_coverage']=float((target&crop).sum()/target.sum().clamp_min(1));r['tile_count']=len(full_fov_geometry(tuple(target.shape)).tile_boxes_yxyx)
  append(records,r)
  if (ordinal+1)%20==0:print(json.dumps({'stage':'OFFICIAL1000_FULL_FOV','arm':a.arm_name,'done':ordinal+1,'total':1000}),flush=True)
 result_rows=rows(records)
 if len(result_rows)!=1000:raise RuntimeError('Official1000 incomplete')
 baseline_rows=rows(ROOT/'outputs/phase6e2_c1_specific_r1/official1000_new_r1_records.jsonl')
 if [r['sample_id'] for r in baseline_rows]!=[r['sample_id'] for r in result_rows]:raise RuntimeError('current-new-R1 baseline pairing drift')
 metrics=summarize(result_rows);metrics['seg_trigger_rate']=sum(r['valid_q_seg'] for r in result_rows)/len(result_rows)
 paired=[]
 for b,n in zip(baseline_rows,result_rows):paired.append({'sample_id':n['sample_id'],'a0_iou':b['foreground_iou'],'a0_f1':b['foreground_f1'],'a0_tp':b['tp'],'a0_fp':b['fp'],'a0_fn':b['fn'],'a1_iou':n['foreground_iou'],'a1_f1':n['foreground_f1'],'a1_tp':n['tp'],'a1_fp':n['fp'],'a1_fn':n['fn'],'delta_iou':n['foreground_iou']-b['foreground_iou'],'delta_f1':n['foreground_f1']-b['foreground_f1'],'exact_crop_gt_coverage':n['exact_crop_gt_coverage'],'aspect_ratio':n['aspect_ratio'],'tile_count':n['tile_count']})
 def cb(v):
  if abs(v-1)<1e-9:return '1.0'
  if v>=.9:return '[0.9,1.0)'
  if v>=.5:return '[0.5,0.9)'
  if v>0:return '(0,0.5)'
  return '0'
 strata={'coverage':{k:group_summary([r for r in paired if cb(r['exact_crop_gt_coverage'])==k]) for k in ('1.0','[0.9,1.0)','[0.5,0.9)','(0,0.5)','0')},'aspect':{'aspect_ratio_lt_1.2':group_summary([r for r in paired if r['aspect_ratio']<1.2]),'aspect_ratio_ge_1.2':group_summary([r for r in paired if r['aspect_ratio']>=1.2])},'tile_count':{k:group_summary([r for r in paired if (r['tile_count']==int(k[2:]) if k!='N>=4' else r['tile_count']>=4)]) for k in ('N=1','N=2','N=3','N>=4')}}
 result={'schema':'phase6f4_full_fov_official1000_v1','status':'COMPLETE','arm':a.arm_name,'metrics':metrics,'paired_vs_current_selected_new_r1':compare(result_rows,baseline_rows,seed=SEED),'paired_strata_vs_current_selected_new_r1':strata,'provenance':{'checkpoint':str(ckpt.resolve()),'checkpoint_sha256':file_sha256(ckpt),'selector':str(selector.resolve()),'selector_sha256':file_sha256(selector),'selector_frozen_before_official':True,'c1_checkpoint_sha256':C1_SHA,'manifest':'SynthScars Official1000 Fake-only canonical G0','full_fov':'raw original RGB -> same frozen CLIP/adapter -> feather stitch','prediction_baseline':'Phase6E2 current selected new R1, exact sample order','predictions':str(records.resolve()),'predictions_sha256':file_sha256(records)}};dump(out/'official1000_results.json',result);dump(out/'status.json',{'status':'COMPLETE','arm':a.arm_name})

if __name__=='__main__':main()
