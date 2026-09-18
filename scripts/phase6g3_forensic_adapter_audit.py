#!/usr/bin/env python3
"""Matched forensic-adapter architecture audit on frozen selected 6G.2 fusion."""
from __future__ import annotations
import json,math,os,random,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy import stats
from transformers import CLIPImageProcessor
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from model.spatial_prior_forensic_adapter import SpatialPriorInteractionAdapter
from scripts.phase4c_a_train import make_model
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard,dump,write_rows
from tools.phase3c1 import binary_metrics,inverse_logits,paired_statistics,probe_loss
from tools.phase4c_a import summarize_extended,tensor_hash
from tools.phase4c_b import file_sha256

OUT=ROOT/'outputs/phase6g3_forensic_adapter_audit';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g3_forensic_adapter_audit');FUSION=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt';CLIP=ROOT/'checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1';SEED=3407
def pairs(split):
 a,b=cache_paths('middle',split),cache_paths('late',split)
 if len(a)!=len(b):raise RuntimeError('cache count drift')
 return list(zip(a,b))
def load(pair):
 a,b=map(load_shard,pair)
 if [r['sample_id'] for r in a['records']] != [r['sample_id'] for r in b['records']] or not torch.equal(a['targets'],b['targets']):raise RuntimeError('cache pairing drift')
 return a,b
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def fusion(device):
 x=torch.load(FUSION,map_location='cpu',weights_only=False);m=CrossLayerPatchAttention(1024,8,.01);m.load_state_dict(x['fusion'],strict=True);return m.to(device).eval().requires_grad_(False),x
def models(device):
 a0=make_model('forensic_adapter',torch.device('cpu'))
 torch.manual_seed(SEED+1);a1=SpatialPriorInteractionAdapter(8,.01)
 # Shared adapter components must begin tensor-exact, independent of constructor order.
 a1.projection.load_state_dict(a0.projection.state_dict());a1.forensic_blocks.load_state_dict(a0.forensic_blocks.state_dict());a1.dense_head.load_state_dict(a0.dense_head.state_dict())
 return a0.to(device),a1.to(device)
def rgb(records,processor,device):
 images=[]
 for r in records:
  x=cv2.imread(r['image_path'],cv2.IMREAD_COLOR)
  if x is None:raise OSError(r['image_path'])
  images.append(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))
 return processor.preprocess(images,return_tensors='pt')['pixel_values'].to(device=device,dtype=torch.float32)
def fused(f,a,b,device):
 with torch.no_grad():return f(a['features'].to(device=device,dtype=torch.float32),b['features'].to(device=device,dtype=torch.float32))
def forward(arm,model,feature,b,processor,device,diag=False):
 if arm=='a0':
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16):return model(feature,return_features=True)
 image=rgb(b['records'],processor,device)
 with torch.autocast(device_type=device.type,dtype=torch.bfloat16):return model(feature,image,return_features=True,need_weights=diag)
def validate(arm,model,f,ps,processor,device,diagnostics=False):
 rec=[];s={'residual_norm':0.,'clip_norm':0.,'tokens':0,'entropy':0.,'max_weight':0.,'diagonal':0.,'attn_count':0};model.eval()
 with torch.no_grad():
  for pair in ps:
   a,b=load(pair);feature=fused(f,a,b,device);o=forward(arm,model,feature,b,processor,device,diagnostics)
   for i,r in enumerate(b['records']):rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(o['logits'][i,0].float(),r['geometry']),b['original_masks'][i].to(device))})
   if diagnostics and arm=='a1':
    q=o['F0'].float().flatten(2).transpose(1,2);res=o['residual_tokens'].float();s['clip_norm']+=float(q.norm(dim=-1).sum());s['residual_norm']+=float(res.norm(dim=-1).sum());s['tokens']+=q.shape[0]*q.shape[1];w=o['attention'].float().clamp_min(1e-12);s['entropy']+=float((-(w*w.log()).sum(-1)).sum());s['max_weight']+=float(w.max(-1).values.sum());s['diagonal']+=float(w.diagonal(dim1=-2,dim2=-1).sum());s['attn_count']+=w.shape[0]*w.shape[1]*w.shape[2]
 m=summarize_extended(rec)
 if diagnostics and arm=='a1':
  cn=s['clip_norm']/s['tokens'];rn=s['residual_norm']/s['tokens'];m['diagnostics']={'clip_token_l2_mean':cn,'spatial_residual_token_l2_mean':rn,'residual_to_clip_norm_ratio':rn/max(cn,1e-12),'attention_entropy_mean':s['entropy']/s['attn_count'],'attention_max_weight_mean':s['max_weight']/s['attn_count'],'attention_diagonal_mass_mean':s['diagonal']/s['attn_count'],'gamma_spatial':float(model.gamma_spatial.detach())}
 return m,rec
def optimizer(model):
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999));warm,total=277,5530
 def scale(step):return float(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)
def train(arm,model,f,processor,device):
 root=OUT/arm;root.mkdir(parents=True,exist_ok=True);ck=CKPT/arm;ck.mkdir(parents=True,exist_ok=True);opt,sched=optimizer(model);trainp,valp=pairs('train'),pairs('val');history=[];start=step=0;existing=sorted(ck.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
 if existing:
  x=torch.load(existing[-1],map_location='cpu',weights_only=False);model.load_state_dict(x['model']);opt.load_state_dict(x['optimizer']);sched.load_state_dict(x['scheduler']);start=x['epoch'];step=x['global_step'];history=json.load(open(root/'history.json'))
 else:
  m,_=validate(arm,model,f,valp,processor,device);payload={'schema':'phase6g3_adapter_checkpoint_v1','arm':arm,'epoch':0,'global_step':0,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m};torch.save(payload,ck/'epoch_0.pt');history=[{'epoch':0,'global_step':0,'train':None,'validation':m,'seconds':0.}];dump(root/'history.json',history)
 for epoch in range(start+1,11):
  began=time.time();model.train();rng=random.Random(SEED+epoch);order=list(trainp);rng.shuffle(order);s={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for pair in order:
   a,b=load(pair);idx=list(range(len(b['records'])));rng.shuffle(idx)
   for begin in range(0,len(idx),16):
    ix=idx[begin:begin+16];aa={**a,'features':a['features'][ix]};bb={**b,'features':b['features'][ix],'records':[b['records'][i] for i in ix]};feature=fused(f,aa,bb,device);target=b['targets'][ix,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True);o=forward(arm,model,feature,bb,processor,device);loss=probe_loss(o['logits'].float(),target);loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
    opt.step();sched.step();step+=1;n=len(ix)
    for k in ('bce','dice','total'):s[k]+=float(loss[k].detach())*n
    s['samples']+=n
  m,_=validate(arm,model,f,valp,processor,device);row={'epoch':epoch,'global_step':step,'train':{k:s[k]/s['samples'] for k in ('bce','dice','total')}|{'samples':s['samples']},'validation':m,'seconds':time.time()-began};history.append(row);dump(root/'history.json',history);payload={'schema':'phase6g3_adapter_checkpoint_v1','arm':arm,'epoch':epoch,'global_step':step,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metrics':m};p=ck/f'epoch_{epoch}.pt';tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,p);print(json.dumps({'stage':'6G3_EPOCH','arm':arm,**row}),flush=True)
 best=max(history,key=lambda x:(x['validation']['mean_foreground_iou'],x['validation']['mean_foreground_f1']));selected=root/'selected.pt';shutil.copy2(ck/f"epoch_{best['epoch']}.pt",selected);x=torch.load(selected,map_location='cpu',weights_only=False);model.load_state_dict(x['model']);m,r=validate(arm,model,f,valp,processor,device,True);write_rows(root/'selected_predictions.jsonl',r);dump(root/'selector.json',{'status':'COMPLETE','selected_epoch':best['epoch'],'selected_metrics':m,'selected_checkpoint_sha256':file_sha256(selected),'candidates':history,'test_used':False,'official1000_used':False,'ood_used':False});return m,r
def complexity(a0,a1):
 p0=sum(x.numel() for x in a0.parameters());p1=sum(x.numel() for x in a1.parameters());n=576;d=256
 stem=168*168*64*3*3*3+84*84*64*3*3+84*84*64*128+42*42*128*3*3+42*42*128*256
 attn=4*n*d*d+2*n*n*d;mac=stem+attn
 return {'A0_parameters':p0,'A1_parameters':p1,'incremental_parameters':p1-p0,'incremental_MACs_per_image_approx':mac,'incremental_FLOPs_per_image_approx':2*mac,'convention':'one MAC equals two FLOPs; normalization, GELU, interpolation and softmax excluded'}
def main():
 device=torch.device('cuda:0');torch.cuda.set_device(device);seed();OUT.mkdir(parents=True,exist_ok=True);f,fx=fusion(device);a0,a1=models(device);processor=CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True)
 shared={'projection':tensor_hash(a0.projection.state_dict().items())==tensor_hash(a1.projection.state_dict().items()),'blocks':tensor_hash(a0.forensic_blocks.state_dict().items())==tensor_hash(a1.forensic_blocks.state_dict().items()),'head':tensor_hash(a0.dense_head.state_dict().items())==tensor_hash(a1.dense_head.state_dict().items())}
 protocol={'schema':'phase6g3_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','evidence_source':{'fusion_checkpoint':str(FUSION.resolve()),'sha256':file_sha256(FUSION),'selected_epoch':fx['epoch'],'frozen':True},'A0':'original Phase4C-A adapter freshly trained on selected fusion','A1':'RGB spatial-prior stem + Q=CLIP/KV=spatial 8-head cross-attention + same local refinement','shared_initialization_exact':shared,'recipe':{'epochs':10,'batch':16,'optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'warmup':277,'total_updates':5530,'loss':'2*BCE+0.5*Dice','seed':SEED,'selector':'val mean FG IoU; tie mean FG F1; epochs0..10'},'rgb_preprocessing':'same CLIPImageProcessor deterministic ResizeShortest336+CenterCrop336 normalized RGB tensor','complexity':complexity(a0,a1),'firewall':{'rectifier':False,'utility':False,'test':False,'official1000':False,'ood':False}};dump(OUT/'protocol.json',protocol);before=tensor_hash(f.state_dict().items());m0,r0=train('a0',a0,f,processor,device);m1,r1=train('a1',a1,f,processor,device)
 if before!=tensor_hash(f.state_dict().items()):raise RuntimeError('frozen fusion drift')
 if [r['sample_id'] for r in r0]!=[r['sample_id'] for r in r1]:raise RuntimeError('paired order drift')
 ai=np.array([r['foreground_iou'] for r in r0]);bi=np.array([r['foreground_iou'] for r in r1]);af=np.array([r['foreground_f1'] for r in r0]);bf=np.array([r['foreground_f1'] for r in r1]);pi=paired_statistics(bi,ai);pf=paired_statistics(bf,af);gain=bi-ai;rescue=[r0[i]['sample_id'] for i in range(len(r0)) if ai[i]<=.1 and gain[i]>=.1];decision='SPATIAL_PRIOR_ADAPTER_SUPPORTED' if pi['mean_difference']>0 and pi['bootstrap_95_ci'][0]>0 else 'SPATIAL_PRIOR_ADAPTER_NOT_SUPPORTED';g2=json.load(open(ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/summary.json'))['comparison']['A1_attention'];retention={'phase6g2a_fusion_linear_probe':g2,'A0_minus_6G2A_mean_iou':m0['mean_foreground_iou']-g2['mean_foreground_iou'],'A1_minus_6G2A_mean_iou':m1['mean_foreground_iou']-g2['mean_foreground_iou']};result={'status':'COMPLETE_STOP','decision':decision,'A0':m0,'A1':m1,'paired':{'iou':pi,'f1':pf},'multilevel_advantage_retention':retention,'rescue':{'definition':'A0 IoU<=0.10 and A1-A0>=0.10','n':len(rescue),'sample_ids':rescue},'gain_correlation_with_A0_iou':{'pearson':float(np.corrcoef(gain,ai)[0,1]),'spearman':float(stats.spearmanr(gain,ai).statistic)},'complexity':protocol['complexity'],'fusion_hash_before':before,'fusion_hash_after':tensor_hash(f.state_dict().items()),'firewall':protocol['firewall']};dump(OUT/'results.json',result);render(result)
def render(x):
 p=x['paired'];text=f"""# Phase 6G.3 — Forensic Adapter Architecture Audit\n\nStatus: **COMPLETE STOP**. Selected Phase6G.2A fusion and CLIP were frozen. No Rectifier, Utility, test, Official1000 or OOD access occurred.\n\n| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |\n|---|---:|---:|---:|---:|---:|\n| A0 original Phase4C-A | {x['A0']['mean_foreground_iou']:.6f} | {x['A0']['median_foreground_iou']:.6f} | {x['A0']['mean_foreground_f1']:.6f} | {x['A0']['global_foreground_iou']:.6f} | {x['A0']['global_foreground_f1']:.6f} |\n| A1 spatial-prior interaction | {x['A1']['mean_foreground_iou']:.6f} | {x['A1']['median_foreground_iou']:.6f} | {x['A1']['mean_foreground_f1']:.6f} | {x['A1']['global_foreground_iou']:.6f} | {x['A1']['global_foreground_f1']:.6f} |\n\n- IoU paired: `{p['iou']}`\n- F1 paired: `{p['f1']}`\n- A1 diagnostics: `{x['A1'].get('diagnostics')}`\n- complexity: `{x['complexity']}`\n- rescue samples: `{x['rescue']['n']}`\n\n```text\n{x['decision']}\n```\n""";(ROOT/'docs/phase6g3_forensic_adapter_audit.md').write_text(text)
if __name__=='__main__':main()
