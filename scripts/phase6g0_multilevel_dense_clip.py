#!/usr/bin/env python3
"""Frozen multi-level CLIP dense-patch probe and complementarity audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch,torch.nn as nn,yaml
from scipy import stats
from transformers import CLIPImageProcessor,CLIPVisionModel

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase3c1 import binary_metrics,inverse_logits,paired_statistics,probe_loss,summarize,tensor_sha256
from tools.phase4c_b import file_sha256

CFG=yaml.safe_load((ROOT/'configs/phase6g0_multilevel_dense_clip.yaml').read_text())
OUT=ROOT/CFG['experiment']['output_root'];CACHE=Path(CFG['experiment']['cache_root']);OLD=ROOT/CFG['data']['phase3c1_clip_cache']
LAYERS=CFG['clip']['layers'];SEED=3407

def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write_rows(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def state_hash(m):return tensor_sha256(list(m.state_dict().items()))
def old_paths(split):return sorted((OLD/split).glob('shard_*.pt'))
def cache_paths(layer,split):return sorted((CACHE/layer/split).glob('shard_*.pt'))

def protocol():
 clip=ROOT/CFG['clip']['checkpoint'];config=json.load(open(clip/'config.json'))['vision_config'];old_train=json.load(open(OLD/'train/complete.json'));old_val=json.load(open(OLD/'val/complete.json'))
 value={'schema':'phase6g0_protocol_v1','status':'FROZEN_BEFORE_EXTRACTION','scope':'internal train Fake 8836 + internal validation Fake 1106 only','clip':{'checkpoint':str(clip.resolve()),'checkpoint_pytorch_sha256':file_sha256(clip/'pytorch_model.bin'),'num_transformer_blocks':config['num_hidden_layers'],'hidden_states_count_expected':config['num_hidden_layers']+1,'hidden_state_semantics':'index0 patch+CLS embeddings; index i is output after transformer block i-1; no final post_layernorm applied to selected intermediate hidden states','current_selector':'GLaMM mm_vision_select_layer=-2 => hidden_states[23] => zero-based block22','hidden_size':config['hidden_size'],'patch_size':config['patch_size'],'image_size':config['image_size'],'patch_tokens':576,'patch_grid':[24,24],'layers':LAYERS,'preprocessing':old_train['preprocessing_version']},'existing_cache_audit':{'only_current_hidden_minus2_exists':True,'train':old_train,'val':old_val},'probe_contract':CFG['probe'],'decision_contract':CFG['decision'],'firewall':CFG['firewall']}
 dump(OUT/'protocol.json',value)

def cache(split,device):
 protocol();clip=ROOT/CFG['clip']['checkpoint'];processor=CLIPImageProcessor.from_pretrained(clip,local_files_only=True);vision=CLIPVisionModel.from_pretrained(clip,local_files_only=True,low_cpu_mem_usage=True).to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False);before=state_hash(vision);paths=old_paths(split);expected=int(CFG['data'][f'{split}_fake']);seen=0;parity={'count':0,'exact':0,'max_abs':0.0}
 for oldp in paths:
  old=torch.load(oldp,map_location='cpu',weights_only=False);start,end=int(old['start']),int(old['end']);outputs={name:CACHE/name/split/oldp.name for name in LAYERS}
  if all(p.exists() for p in outputs.values()):seen+=end-start;continue
  images=[]
  for r in old['records']:
   image=cv2.imread(r['image_path'],cv2.IMREAD_COLOR)
   if image is None:raise OSError(r['image_path'])
   images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
  pixels=processor.preprocess(images,return_tensors='pt')['pixel_values'].to(device=device,dtype=torch.bfloat16)
  with torch.no_grad():h=vision(pixels,output_hidden_states=True).hidden_states
  if len(h)!=25:raise RuntimeError(f'hidden-state count drift: {len(h)}')
  shard_parity=None
  for name,spec in LAYERS.items():
   token=h[int(spec['hidden_state_index'])][:,1:]
   if token.shape[1:]!=(576,1024):raise RuntimeError(f'{name} patch shape drift {token.shape}')
   feature=token.transpose(1,2).reshape(len(images),1024,24,24).to(torch.bfloat16).cpu()
   if name=='current_hidden_minus2':
    d=(feature.float()-old['features'].float()).abs();shard_parity={'count':feature.numel(),'exact':int((feature==old['features']).sum()),'max_abs':float(d.max())};parity['count']+=shard_parity['count'];parity['exact']+=shard_parity['exact'];parity['max_abs']=max(parity['max_abs'],shard_parity['max_abs'])
   payload={'schema':'phase6g0_multilevel_clip_cache_v1','layer':name,'block_index':spec['block_index'],'hidden_state_index':spec['hidden_state_index'],'split':split,'start':start,'end':end,'features':feature,'targets':old['targets'],'records':old['records'],'old_current_cache_parity':shard_parity if name=='current_hidden_minus2' else None}
   if split=='val':payload['original_masks']=old['original_masks']
   p=outputs[name];p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pt.tmp');torch.save(payload,t);os.replace(t,p)
  seen+=end-start;print(json.dumps({'stage':'CACHE','split':split,'done':seen,'total':expected}),flush=True)
 if seen!=expected:raise RuntimeError('cache population drift')
 after=state_hash(vision)
 if before!=after:raise RuntimeError('frozen CLIP parameter hash drift')
 # Aggregate parity from disk so interruption/resume cannot weaken the gate.
 parity={'count':0,'exact':0,'max_abs':0.0}
 for p in cache_paths('current_hidden_minus2',split):
  q=torch.load(p,map_location='cpu',weights_only=False)['old_current_cache_parity'];parity['count']+=q['count'];parity['exact']+=q['exact'];parity['max_abs']=max(parity['max_abs'],q['max_abs'])
 for name,spec in LAYERS.items():
  complete={'status':'COMPLETE','layer':name,'split':split,'samples':expected,'shards':len(paths),'feature_shape':[1024,24,24],'storage_dtype':'torch.bfloat16','block_index':spec['block_index'],'hidden_state_index':spec['hidden_state_index'],'clip_state_sha256_before':before,'clip_state_sha256_after':after,'frozen_exact':before==after,'old_current_cache_parity':parity if name=='current_hidden_minus2' else None}
  if name=='current_hidden_minus2' and parity['exact']!=parity['count']:raise RuntimeError(f'current hidden[-2] cache parity failed: {parity}')
  dump(CACHE/name/split/'complete.json',complete)

def load_shard(p):
 x=torch.load(p,map_location='cpu',weights_only=False)
 if x.get('schema')!='phase6g0_multilevel_clip_cache_v1':raise RuntimeError(p)
 return x

def predictions(probe,paths,device,keep_low=False):
 out=[];low=[];probe.eval()
 with torch.no_grad():
  for p in paths:
   x=load_shard(p)
   for i,r in enumerate(x['records']):
    z=probe(x['features'][i:i+1].to(device=device,dtype=torch.float32))[0,0];original=inverse_logits(z,r['geometry']);m=binary_metrics(original,x['original_masks'][i].to(device));out.append({'sample_id':r['sample_id'],**m});low.append(z.cpu()) if keep_low else None
 return out,low

def train(layer,device):
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);root=OUT/'probes'/layer
 if (root/'training_manifest.json').exists():print(json.dumps({'stage':'PROBE','layer':layer,'status':'ALREADY_COMPLETE'}));return
 trainp,valp=cache_paths(layer,'train'),cache_paths(layer,'val');probe=nn.Conv2d(1024,1,1,bias=True).to(device,dtype=torch.float32);opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0.);history=[];best=None
 for epoch in range(1,21):
  began=time.time();probe.train();rng=random.Random(SEED+epoch);ep=list(trainp);rng.shuffle(ep);sums={'bce':0.,'dice':0.,'total':0.,'samples':0}
  for p in ep:
   x=load_shard(p);order=list(range(len(x['records'])));rng.shuffle(order)
   for start in range(0,len(order),16):
    idx=order[start:start+16];f=x['features'][idx].to(device=device,dtype=torch.float32);t=x['targets'][idx,None].to(device=device,dtype=torch.float32);opt.zero_grad(set_to_none=True);loss=probe_loss(probe(f),t);loss['total'].backward();opt.step();n=len(idx)
    for k in ('bce','dice','total'):sums[k]+=float(loss[k].detach())*n
    sums['samples']+=n
  vr,_=predictions(probe,valp,device);vm=summarize(vr);row={'epoch':epoch,'train_samples':sums['samples'],'train_bce':sums['bce']/sums['samples'],'train_dice':sums['dice']/sums['samples'],'train_total':sums['total']/sums['samples'],'val':vm,'seconds':time.time()-began};history.append(row);dump(root/'history.json',history);candidate=(vm['mean_foreground_iou'],vm['mean_foreground_f1']);payload={'schema':'phase6g0_linear_probe_v1','layer':layer,'epoch':epoch,'channels':1024,'state_dict':probe.state_dict(),'optimizer':opt.state_dict(),'selector':candidate,'hyperparameters':CFG['probe']};root.mkdir(parents=True,exist_ok=True);torch.save(payload,root/'last.pt')
  if best is None or candidate>best:best=candidate;torch.save(payload,root/'selected.pt')
  print(json.dumps({'stage':'PROBE_EPOCH','layer':layer,**row}),flush=True)
 selected=torch.load(root/'selected.pt',map_location='cpu',weights_only=False);probe.load_state_dict(selected['state_dict']);records,low=predictions(probe,valp,device,True);write_rows(OUT/'validation'/layer/'predictions.jsonl',records);torch.save({'sample_ids':[r['sample_id'] for r in records],'low_res_logits':torch.stack(low)},OUT/'validation'/layer/'low_res_logits.pt');dump(OUT/'validation'/layer/'metrics.json',summarize(records));dump(root/'training_manifest.json',{'status':'COMPLETE','layer':layer,'selected_epoch':selected['epoch'],'selected_val':{'mean_foreground_iou':selected['selector'][0],'mean_foreground_f1':selected['selector'][1]},'trainable_parameters':1025,'optimizer':'AdamW','learning_rate':.001,'weight_decay':0.,'epochs':20,'batch_size':16,'loss':'2.0*BCE+0.5*Dice','seed':SEED})

def corr(a,b):
 if np.std(a)<1e-12 or np.std(b)<1e-12:return 1. if np.allclose(a,b) else 0.
 return float(np.corrcoef(a,b)[0,1])
def jaccard(a,b):return len(a&b)/max(1,len(a|b))

def analyze():
 current='current_hidden_minus2';names=list(LAYERS);pred={n:rows(OUT/'validation'/n/'predictions.jsonl') for n in names};ids=[r['sample_id'] for r in pred[current]]
 if any([r['sample_id'] for r in pred[n]]!=ids for n in names):raise RuntimeError('validation pairing drift')
 metrics={n:summarize(pred[n]) for n in names};paired={};rescues={};gains={}
 ci={n:torch.load(OUT/'validation'/n/'low_res_logits.pt',map_location='cpu',weights_only=False)['low_res_logits'].float().numpy() for n in names}
 spatial={};current_iou=np.array([r['foreground_iou'] for r in pred[current]])
 for n in names:
  if n==current:continue
  ni=np.array([r['foreground_iou'] for r in pred[n]]);nf=np.array([r['foreground_f1'] for r in pred[n]]);cf=np.array([r['foreground_f1'] for r in pred[current]]);g=ni-current_iou;gains[n]=g;paired[n]={'iou':paired_statistics(ni,current_iou),'f1':paired_statistics(nf,cf)};rescues[n]={ids[i] for i in range(len(ids)) if current_iou[i]<=.10 and g[i]>=.10};spatial[n]={'per_sample_logit_pearson_mean':float(np.mean([corr(ci[n][i].ravel(),ci[current][i].ravel()) for i in range(len(ids))])),'per_sample_logit_pearson_median':float(np.median([corr(ci[n][i].ravel(),ci[current][i].ravel()) for i in range(len(ids))])),'global_logit_pearson':corr(ci[n].ravel(),ci[current].ravel())}
 gain_corr={};overlap={};alts=[n for n in names if n!=current]
 for i,a in enumerate(alts):
  for b in alts[i+1:]:gain_corr[f'{a}__{b}']={'pearson':corr(gains[a],gains[b]),'spearman':float(stats.spearmanr(gains[a],gains[b]).statistic)};overlap[f'{a}__{b}']={'rescue_intersection':len(rescues[a]&rescues[b]),'rescue_union':len(rescues[a]|rescues[b]),'rescue_jaccard':jaccard(rescues[a],rescues[b])}
 rescue_report={n:{'n':len(rescues[n]),'fraction':len(rescues[n])/len(ids),'sample_ids':sorted(rescues[n])} for n in alts}
 stable=[n for n in alts if paired[n]['iou']['bootstrap_95_ci'][0]>0 and paired[n]['f1']['bootstrap_95_ci'][0]>0]
 complementary=False
 for key,c in gain_corr.items():
  a,b=key.split('__');o=overlap[key]
  if len(rescues[a])/len(ids)>=.02 and len(rescues[b])/len(ids)>=.02 and o['rescue_jaccard']<=.70 and c['spearman']<=.80:complementary=True
 decision='PROCEED_SINGLE_LAYER_REPLACEMENT_TEST' if stable else ('PROCEED_MULTI_LEVEL_FUSION' if complementary else 'MULTI_LEVEL_DENSE_CLIP_NOT_SUPPORTED')
 result={'schema':'phase6g0_results_v1','status':'COMPLETE_STOP','metrics':metrics,'paired_vs_current_hidden_minus2':paired,'per_sample_gain_correlation':gain_corr,'rescue_overlap':overlap,'current_failure_rescues':rescue_report,'spatial_prediction_correlation_vs_current':spatial,'stable_single_layers':stable,'complementarity_gate_pass':complementary,'decision':decision,'firewall':{'internal_test_accessed':False,'official1000_accessed':False,'ood_accessed':False,'r1_trained':False}}
 dump(OUT/'results.json',result);dump(OUT/'complementarity.json',{'gain_correlation':gain_corr,'overlap':overlap,'rescues':rescue_report,'spatial_correlation':spatial});render(result)

def render(r):
 lines=['# Phase 6G.0 — Multi-level Dense CLIP Forensic Evidence Audit','','Status: **COMPLETE STOP**. Frozen CLIP only; four matched linear probes were trained on internal TRAIN and selected on internal validation. No test, Official1000, OOD, C1, SAM, Rectifier or Utility parameters were accessed for training.','','## CLIP audit','','ViT-L/14@336 has 24 transformer blocks and 25 hidden states including embeddings. `hidden[-2] = hidden_states[23]` is the output of zero-based block 22, before CLIP final post-layernorm. Patch tokens are `[B,576,1024]`, reshaped to `[B,1024,24,24]`. Preprocessing is ResizeShortest336 + CenterCrop336 + CLIP rescale/normalization.','','## Matched probes','','| Layer | Block | Mean IoU | Median IoU | Mean F1 | ΔIoU vs current | IoU 95% CI |','|---|---:|---:|---:|---:|---:|---|']
 for n,s in LAYERS.items():
  m=r['metrics'][n];p=r['paired_vs_current_hidden_minus2'].get(n);lines.append(f"| {n} | {s['block_index']} | {m['mean_foreground_iou']:.6f} | {m['median_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {0 if p is None else p['iou']['mean_difference']:.6f} | {'—' if p is None else p['iou']['bootstrap_95_ci']} |")
 lines += ['','## Complementarity','',f"Stable superior single layers: `{r['stable_single_layers']}`.",f"Complementarity gate: `{r['complementarity_gate_pass']}`.",'',f"```text\n{r['decision']}\n```",'','All raw predictions, selected logits, paired statistics, rescue sample IDs and provenance are under `outputs/phase6g0_multilevel_dense_clip_audit/`.']
 (ROOT/'docs/phase6g0_multilevel_dense_clip_audit.md').write_text('\n'.join(lines)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('protocol','cache','train','analyze'));ap.add_argument('--split',choices=('train','val'));ap.add_argument('--layer',choices=tuple(LAYERS));ap.add_argument('--device',default='cuda:0');a=ap.parse_args();device=torch.device(a.device)
 if device.type=='cuda':torch.cuda.set_device(device)
 if a.mode=='protocol':protocol()
 elif a.mode=='cache':cache(a.split,device)
 elif a.mode=='train':train(a.layer,device)
 else:analyze()
if __name__=='__main__':main()
