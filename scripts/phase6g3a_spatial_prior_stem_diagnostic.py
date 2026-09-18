#!/usr/bin/env python3
"""Phase 6G.3A: frozen spatial-prior stem diagnostic (internal train/val only)."""
from __future__ import annotations

import json, os, random, shutil, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from model.clip_forensic_adapter import CLIPSpatialArm
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from model.spatial_prior_forensic_adapter import SpatialPriorInteractionAdapter, SpatialPriorStem
from scripts.phase4c_a_train import make_model
from scripts.phase6g0_multilevel_dense_clip import cache_paths, load_shard, dump, rows, write_rows
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, probe_loss
from tools.phase4c_a import summarize_extended, tensor_hash
from tools.phase4c_b import file_sha256

OUT = ROOT/'outputs/phase6g3a_spatial_prior_stem_diagnostic'
CACHE = Path('/data/yz/groundingLMM_official/cache/phase6g3a_spatial_prior_stem_diagnostic')
CKPT = Path('/data/yz/groundingLMM_official/checkpoints/phase6g3a_spatial_prior_stem_diagnostic')
FUSION = ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt'
A0_ADAPTER = ROOT/'outputs/phase6g3_forensic_adapter_audit/a0/selected.pt'
A1_ADAPTER = ROOT/'outputs/phase6g3_forensic_adapter_audit/a1/selected.pt'
CLIP = ROOT/'checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
SEED = 3407

class StemProbe(nn.Module):
    def __init__(self, head_state):
        super().__init__(); self.head=nn.Conv2d(256,1,1); self.head.load_state_dict(head_state)
    def forward(self, spatial, clip): return self.head(spatial)

class AlignedFusionProbe(nn.Module):
    def __init__(self, head_state):
        super().__init__(); self.project=nn.Conv2d(256,256,1); self.gamma=nn.Parameter(torch.tensor(.01)); self.head=nn.Conv2d(256,1,1); self.head.load_state_dict(head_state)
    def forward(self, spatial, clip): return self.head(clip + self.gamma*self.project(spatial))

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def source_models(device):
    fx=torch.load(FUSION,map_location='cpu',weights_only=False); fusion=CrossLayerPatchAttention(1024,8,.01); fusion.load_state_dict(fx['fusion'],strict=True)
    a0=make_model('forensic_adapter',torch.device('cpu')); a0x=torch.load(A0_ADAPTER,map_location='cpu',weights_only=False); a0.load_state_dict(a0x['model'],strict=True)
    a1=SpatialPriorInteractionAdapter(8,.01); a1x=torch.load(A1_ADAPTER,map_location='cpu',weights_only=False); a1.load_state_dict(a1x['model'],strict=True)
    stem=SpatialPriorStem(); stem.load_state_dict(a1.spatial_prior.state_dict(),strict=True)
    return fusion.to(device).eval().requires_grad_(False),a0.projection.to(device).eval().requires_grad_(False),stem.to(device).eval().requires_grad_(False),fx,a0x,a1x

def source_hashes(fusion,projection,stem):
    return {'fusion':tensor_hash(fusion.state_dict().items()),'clip_projection':tensor_hash(projection.state_dict().items()),'spatial_stem':tensor_hash(stem.state_dict().items())}

def paired_paths(split):
    a,b=cache_paths('middle',split),cache_paths('late',split)
    if len(a)!=len(b): raise RuntimeError('block11/block17 shard drift')
    return list(zip(a,b))

def cache_split(split,device,fusion,projection,stem,processor):
    dest=CACHE/split; dest.mkdir(parents=True,exist_ok=True); expected=8836 if split=='train' else 1106; seen=0
    for pa,pb in paired_paths(split):
        out=dest/pa.name
        if out.exists(): seen+=int(torch.load(out,map_location='cpu',weights_only=False)['end'])-int(torch.load(out,map_location='cpu',weights_only=False)['start']); continue
        a,b=load_shard(pa),load_shard(pb)
        ids=[r['sample_id'] for r in a['records']]
        if ids != [r['sample_id'] for r in b['records']] or not torch.equal(a['targets'],b['targets']): raise RuntimeError('paired cache drift')
        images=[]
        for r in a['records']:
            x=cv2.imread(r['image_path'],cv2.IMREAD_COLOR)
            if x is None: raise OSError(r['image_path'])
            images.append(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))
        pixels=processor.preprocess(images,return_tensors='pt')['pixel_values'].to(device=device,dtype=torch.float32)
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):
            fused=fusion(a['features'].to(device=device,dtype=torch.float32),b['features'].to(device=device,dtype=torch.float32))
            clip=projection(fused); spatial=stem(pixels)
        payload={'schema':'phase6g3a_frozen_feature_cache_v1','split':split,'start':a['start'],'end':a['end'],'records':a['records'],'targets':a['targets'],'clip256':clip.to(torch.bfloat16).cpu(),'spatial256':spatial.to(torch.bfloat16).cpu()}
        if split=='val': payload['original_masks']=a['original_masks']
        tmp=out.with_suffix('.pt.tmp'); torch.save(payload,tmp); os.replace(tmp,out); seen+=len(ids); print(json.dumps({'stage':'CACHE','split':split,'done':seen,'total':expected}),flush=True)
    if seen!=expected: raise RuntimeError(f'{split} cache population {seen} != {expected}')
    dump(dest/'complete.json',{'status':'COMPLETE','split':split,'samples':seen,'shards':len(paired_paths(split)),'feature_shape':[256,24,24],'dtype':'torch.bfloat16','sources_frozen':True})

def cached(split):
    complete=json.load(open(CACHE/split/'complete.json'))
    ps=sorted((CACHE/split).glob('shard_*.pt'))
    if complete['status']!='COMPLETE' or len(ps)!=complete['shards']: raise RuntimeError('incomplete 6G3A cache')
    return ps

def load_cached(p):
    x=torch.load(p,map_location='cpu',weights_only=False)
    if x.get('schema')!='phase6g3a_frozen_feature_cache_v1': raise RuntimeError(p)
    return x

def predict(model,device,keep_low=False):
    rec=[]; low=[]; model.eval()
    with torch.no_grad():
        for p in cached('val'):
            x=load_cached(p); z=model(x['spatial256'].to(device=device,dtype=torch.float32),x['clip256'].to(device=device,dtype=torch.float32))[:,0]
            for i,r in enumerate(x['records']): rec.append({'sample_id':r['sample_id'],**binary_metrics(inverse_logits(z[i],r['geometry']),x['original_masks'][i].to(device))}); low.append(z[i].cpu()) if keep_low else None
    return rec,low

def train_arm(name,model,device):
    root=OUT/name; root.mkdir(parents=True,exist_ok=True); croot=CKPT/name; croot.mkdir(parents=True,exist_ok=True)
    done=root/'training_manifest.json'
    if done.exists():
        selected=torch.load(root/'selected.pt',map_location='cpu',weights_only=False); model.load_state_dict(selected['state_dict']); return finalize_arm(name,model,device,selected)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.); history=[]; best=None
    for epoch in range(1,21):
        began=time.time(); model.train(); rng=random.Random(SEED+epoch); paths=cached('train'); rng.shuffle(paths); sums={'bce':0.,'dice':0.,'total':0.,'samples':0}
        for p in paths:
            x=load_cached(p); order=list(range(len(x['records']))); rng.shuffle(order)
            for start in range(0,len(order),16):
                ix=order[start:start+16]; spatial=x['spatial256'][ix].to(device=device,dtype=torch.float32); clip=x['clip256'][ix].to(device=device,dtype=torch.float32); target=x['targets'][ix,None].to(device=device,dtype=torch.float32)
                opt.zero_grad(set_to_none=True); loss=probe_loss(model(spatial,clip),target); loss['total'].backward(); opt.step(); n=len(ix)
                for k in ('bce','dice','total'): sums[k]+=float(loss[k].detach())*n
                sums['samples']+=n
        vr,_=predict(model,device); vm=summarize_extended(vr); row={'epoch':epoch,'train':{k:sums[k]/sums['samples'] for k in ('bce','dice','total')}|{'samples':sums['samples']},'validation':vm,'seconds':time.time()-began}; history.append(row); dump(root/'history.json',history)
        candidate=(vm['mean_foreground_iou'],vm['mean_foreground_f1']); payload={'schema':'phase6g3a_probe_v1','arm':name,'epoch':epoch,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'selector':candidate}
        torch.save(payload,croot/f'epoch_{epoch}.pt')
        if best is None or candidate>best: best=candidate; torch.save(payload,root/'selected.pt')
        print(json.dumps({'stage':'PROBE_EPOCH','arm':name,**row}),flush=True)
    selected=torch.load(root/'selected.pt',map_location='cpu',weights_only=False); model.load_state_dict(selected['state_dict']); dump(done,{'status':'COMPLETE','selected_epoch':selected['epoch'],'selected_checkpoint_sha256':file_sha256(root/'selected.pt'),'optimizer':'AdamW','lr':1e-3,'weight_decay':0.,'epochs':20,'batch_size':16,'loss':'2*BCE+0.5*Dice','seed':SEED,'selector':'validation mean FG IoU; tie mean FG F1','trainable_parameters':sum(p.numel() for p in model.parameters())})
    return finalize_arm(name,model,device,selected)

def finalize_arm(name,model,device,selected):
    rec,low=predict(model,device,True); metrics=summarize_extended(rec); write_rows(OUT/name/'selected_predictions.jsonl',rec); torch.save({'sample_ids':[r['sample_id'] for r in rec],'low_res_logits':torch.stack(low)},OUT/name/'selected_low_res_logits.pt'); dump(OUT/name/'metrics.json',metrics)
    diag={}
    if name=='a1_aligned':
        sn=cn=rn=n=0.
        with torch.no_grad():
            for p in cached('val'):
                x=load_cached(p); s=x['spatial256'].to(device=device,dtype=torch.float32); c=x['clip256'].to(device=device,dtype=torch.float32); r=model.gamma*model.project(s); sn+=float(s.norm(dim=1).sum()); cn+=float(c.norm(dim=1).sum()); rn+=float(r.norm(dim=1).sum()); n+=s.shape[0]*s.shape[2]*s.shape[3]
        diag={'spatial_patch_norm_mean':sn/n,'clip_patch_norm_mean':cn/n,'aligned_residual_patch_norm_mean':rn/n,'residual_to_clip_norm_ratio':rn/max(cn,1e-12),'gamma':float(model.gamma.detach())}
    return metrics,rec,diag

def read_existing(path): return rows(path)
def paired(left,right):
    if [r['sample_id'] for r in left]!=[r['sample_id'] for r in right]: raise RuntimeError('sample pairing drift')
    return {'iou':paired_statistics([r['foreground_iou'] for r in left],[r['foreground_iou'] for r in right]),'f1':paired_statistics([r['foreground_f1'] for r in left],[r['foreground_f1'] for r in right])}

def analyze(m0,r0,d0,m1,r1,d1):
    existing={
      'block11':read_existing(ROOT/'outputs/phase6g0_multilevel_dense_clip_audit/validation/middle/predictions.jsonl'),
      'block17':read_existing(ROOT/'outputs/phase6g0_multilevel_dense_clip_audit/validation/late/predictions.jsonl'),
      'block22':read_existing(ROOT/'outputs/phase6g0_multilevel_dense_clip_audit/validation/current_hidden_minus2/predictions.jsonl'),
      'block11_17_attention':read_existing(ROOT/'outputs/phase6g2_multilevel_attention/phase6g2a/predictions.jsonl'),
      'global_spatial_attention_adapter':read_existing(ROOT/'outputs/phase6g3_forensic_adapter_audit/a1/selected_predictions.jsonl'),
    }
    metrics={k:summarize_extended(v) for k,v in existing.items()}|{'spatial_prior_stem_only':m0,'position_aligned_fusion':m1}
    comparisons={'stem_vs_block11':paired(r0,existing['block11']),'aligned_vs_global_spatial_attention':paired(r1,existing['global_spatial_attention_adapter']),'aligned_vs_attention_fusion_probe':paired(r1,existing['block11_17_attention']),'aligned_vs_stem':paired(r1,r0)}
    stem_weak=comparisons['stem_vs_block11']['iou']['bootstrap_95_ci'][1]<0
    aligned_gain=comparisons['aligned_vs_global_spatial_attention']['iou']; geometry=(not stem_weak and aligned_gain['mean_difference']>0 and aligned_gain['bootstrap_95_ci'][0]>0)
    decision='SPATIAL_PRIOR_STEM_WEAK' if stem_weak else ('INTERACTION_GEOMETRY_IS_PRIMARY_BOTTLENECK' if geometry else 'SPATIAL_PRIOR_NOT_EFFECTIVELY_COMPLEMENTARY')
    result={'schema':'phase6g3a_results_v1','status':'COMPLETE_STOP','decision':decision,'metrics':metrics,'comparisons':comparisons,'diagnostics':d1,'decision_rule':{'stem_weak':'stem-only vs block11 IoU CI upper < 0','geometry_primary':'stem not weak and aligned vs global adapter IoU mean > 0 with CI lower > 0'},'firewall':{'rectifier_trained':False,'utility_trained':False,'internal_test_accessed':False,'official1000_accessed':False,'ood_accessed':False}}
    dump(OUT/'results.json',result); render(result)

def render(r):
    order=['block11','block17','block22','block11_17_attention','spatial_prior_stem_only','global_spatial_attention_adapter','position_aligned_fusion']; lines=['# Phase 6G.3A — Spatial Prior Stem Diagnostic','','Status: **COMPLETE STOP**. Only internal TRAIN/validation were used. Stem and CLIP-side sources were frozen; Rectifier/Utility and all external sets were untouched.','','| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |','|---|---:|---:|---:|---:|---:|']
    for k in order:
        m=r['metrics'][k]; lines.append(f"| {k} | {m['mean_foreground_iou']:.6f} | {m['median_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m.get('global_foreground_iou',float('nan')):.6f} | {m.get('global_foreground_f1',float('nan')):.6f} |")
    lines += ['','## Frozen-interface detail','','For A1, the selected Phase6G.2A block11+17 fusion and the selected Phase6G.3 A0 1024→256 projection are frozen. The selected Phase6G.3 A1 Spatial Prior Stem is frozen. Only position-wise 1×1 `Project`, scalar `gamma`, and the 1×1 dense head train; no cross-position attention is present.','',f"Aligned diagnostics: `{r['diagnostics']}`",'','## Paired diagnostics','']
    for k,v in r['comparisons'].items(): lines.append(f"- {k}: `{v}`")
    lines += ['',f"```text\n{r['decision']}\n```",'','Raw predictions, checkpoints, histories, provenance, and paired statistics are under `outputs/phase6g3a_spatial_prior_stem_diagnostic/`.']
    (ROOT/'docs/phase6g3a_spatial_prior_stem_diagnostic.md').write_text('\n'.join(lines)+'\n')

def main():
    device=torch.device('cuda:0'); torch.cuda.set_device(device); seed_all(); OUT.mkdir(parents=True,exist_ok=True)
    fusion,projection,stem,fx,a0x,a1x=source_models(device); before=source_hashes(fusion,projection,stem); processor=CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True)
    seed_all(); head=nn.Conv2d(256,1,1); common={k:v.detach().clone() for k,v in head.state_dict().items()}
    protocol={'schema':'phase6g3a_protocol_v1','status':'FROZEN_BEFORE_FIRST_STEP','selected_sources':{'fusion':{'path':str(FUSION.resolve()),'sha256':file_sha256(FUSION),'epoch':fx['epoch']},'clip_projection':{'path':str(A0_ADAPTER.resolve()),'sha256':file_sha256(A0_ADAPTER),'epoch':a0x['epoch'],'component':'projection only'},'spatial_stem':{'path':str(A1_ADAPTER.resolve()),'sha256':file_sha256(A1_ADAPTER),'epoch':a1x['epoch'],'component':'spatial_prior only'}},'source_hashes_before':before,'recipe':{'epochs':20,'batch_size':16,'optimizer':'AdamW','lr':1e-3,'weight_decay':0.,'loss':'2*BCE+0.5*Dice','seed':SEED,'threshold_logit':0.,'selector':'validation mean FG IoU; tie mean FG F1'},'A0_trainable':['1x1 dense head'],'A1_trainable':['1x1 Project','scalar gamma','1x1 dense head'],'shared_dense_head_initialization_exact':True,'firewall':{'rectifier':False,'utility':False,'test':False,'official1000':False,'ood':False}}
    dump(OUT/'protocol.json',protocol)
    cache_split('train',device,fusion,projection,stem,processor); cache_split('val',device,fusion,projection,stem,processor)
    after=source_hashes(fusion,projection,stem)
    if before!=after: raise RuntimeError('frozen source drift during cache construction')
    a0=StemProbe(common).to(device); seed_all(); a1=AlignedFusionProbe(common).to(device)
    m0,r0,d0=train_arm('a0_stem_only',a0,device); m1,r1,d1=train_arm('a1_aligned',a1,device)
    if before!=source_hashes(fusion,projection,stem): raise RuntimeError('frozen source drift')
    dump(OUT/'checkpoint_provenance.json',protocol['selected_sources']|{'source_hashes_before':before,'source_hashes_after':source_hashes(fusion,projection,stem),'frozen_exact':True})
    analyze(m0,r0,d0,m1,r1,d1)

if __name__=='__main__': main()
