#!/usr/bin/env python3
"""Phase6K B0 and matched A/E training on immutable C2 canonical caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.c2_spatial_transfer import AfterProjection, utility_forward_full_input_gradient
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts import phase6e3_c1_native_staged as e3
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, evidence_feature, invalid_record, load_evidence_source, load_rectifier, load_sam_runtime, mask_loss
from scripts import phase4g1q_conditional_utility as q
from model.pcerf import sam_lowres_to_original_normalized
from tools.phase3c1 import geometry_for

CACHE = Path('/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2')
AUDIT = ROOT/'outputs/phase6k0_c2_after'
CKPTS = Path('/data/yz/groundingLMM_official/checkpoints/phase6k0_c2_after')
C2_PROTOCOL = ROOT/'outputs/phase6j0_c2/final_evaluation/protocol.json'
C2_CKPT = ROOT/'checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt'
NATIVE_SELECTOR = ROOT/'outputs/phase6e3_c1_native_staged/joint/selector.json'
CFG = e3.CFG
SEED, BATCH, EPOCHS, LR, WD, CLIP = 3407, 8, 10, 1e-4, 1e-4, 1.0


def require(ok, why):
    if not ok: raise RuntimeError(why)


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def load_cache(split, expected):
    status = json.loads((AUDIT/'cache'/f'{split}.json').read_text())
    require(status['status']=='COMPLETE' and status['n']==len(expected), f'{split} C2 cache incomplete')
    require(status['ids_sha256']==hashlib.sha256('\n'.join(expected).encode()).hexdigest(), f'{split} IDs drift')
    chunks, ids = [], []
    for spec in status['shards']:
        path=Path(spec['path']); require(file_sha256(path)==spec['sha256'],f'{split} shard SHA drift')
        value=torch.load(path,map_location='cpu',weights_only=False); ids+=value['sample_ids']; chunks.append(value)
    require(ids==expected, f'{split} C2 cache order drift')
    require(all(x['c2_sha256']==status['c2_sha256'] for x in chunks), f'{split} C2 SHA drift')
    result={'sample_ids':ids,'c2_sha256':status['c2_sha256'],
            **{key:torch.cat([x[key] for x in chunks]) for key in ('q_seg','valid','seg_count','A','E')}}
    require(result['A'].shape==(len(ids),8,24,24) and result['E'].shape==(len(ids),512,24,24),'A/E cache shape drift')
    require(result['A'].dtype==torch.float32 and result['E'].dtype==torch.bfloat16,
            'A/E cache storage dtype drift')
    require(bool(torch.isfinite(result['A']).all() and torch.isfinite(result['E']).all()),'A/E cache nonfinite')
    require(float((result['A'].flatten(2).sum(-1)-1).abs().max())<1e-5,'A cache head mass drift')
    require(torch.equal(result['valid'],result['seg_count'].eq(1)) and
            bool(torch.isfinite(result['q_seg'][result['valid']]).all()) and
            bool(torch.isnan(result['q_seg'][~result['valid']]).all()),
            'C2 q_seg validity/invalid sentinel drift')
    return result


def provenance():
    c2=json.loads(C2_PROTOCOL.read_text()); native=json.loads(NATIVE_SELECTOR.read_text())
    center=json.loads((ROOT/'outputs/phase6j0_c2/center/status.json').read_text())
    center_result=json.loads((ROOT/'outputs/phase6j0_c2/center/results.json').read_text())
    require(center['status']=='COMPLETE' and center_result['status']=='COMPLETE' and
            center_result['selected_checkpoint_sha256']==c2['selected_checkpoint_sha256'],
            'C2-center completion/checkpoint drift')
    require(c2['status']=='FROZEN' and c2['selected_checkpoint_sha256']==file_sha256(C2_CKPT), 'C2 checkpoint drift')
    require((c2['selected_step'],c2['selected_epoch'])==(3500,7),'C2 selector drift')
    require(native['status']=='COMPLETE' and native['stage']=='joint' and
            native['selected_checkpoint_sha256']==file_sha256(Path(native['selected_checkpoint'])) and
            native['c1_sha256']==e3.C1_SHA and not native['official1000_used_for_selection'] and
            not native['external_ood_used_for_selection'], 'native R1 provenance/firewall drift')
    payload=torch.load(native['selected_checkpoint'],map_location='cpu',weights_only=False)
    require(payload['c1_sha256']==e3.C1_SHA and 'utility_state' in payload and 'rectifier_state' in payload,
            'native joint payload drift')
    return c2,native,payload


def runtime(device,payload):
    utility,_=hc.load_utility('a2',device)
    utility.load_state_dict(payload['utility_state'],strict=True)
    utility.eval().requires_grad_(False)
    rect=load_rectifier(CFG,e3.c1_gamma(),device)
    rect.load_state_dict(payload['rectifier_state'],strict=True)
    rect.eval().requires_grad_(False)
    sam=load_sam_runtime(CFG,device).eval().requires_grad_(False)
    source=load_evidence_source(CFG,'forensic_rect',device).eval().requires_grad_(False)
    frozen={'utility':tensor_state_sha256(utility.state_dict()),
            'rectifier':tensor_state_sha256(rect.state_dict()),
            'sam':tensor_state_sha256(sam.state_dict()),
            'source':tensor_state_sha256(source.state_dict())}
    require(all(not p.requires_grad for m in (utility,rect,sam,source) for p in m.parameters()),'frozen scope drift')
    return utility,rect,sam,source,frozen


def forward(data,cache,idx,store,modules,device,arm=None,need_loss=True):
    utility,rect,sam,source=modules
    ids=[cache['sample_ids'][i] for i in idx.tolist()]
    s64,raw,target,sc,cc=e2.phase4f_spatial_batch(store,ids,device)
    qseg=cache['q_seg'].index_select(0,idx).to(device=device,dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type='cuda',enabled=False):
        raw_low=sam(qseg,s64.to(torch.bfloat16))
    zl=torch.cat([sam_lowres_to_original_normalized(raw_low[j:j+1],store.geometries[sid],
                output_hw=(256,256)) for j,sid in enumerate(ids)]).to(device=device,dtype=torch.bfloat16)
    batch={'S64':s64,'q_seg':qseg,'z_L':zl,
           'clip_geometries':[geometry_for('clip',store.geometries[sid]['original_hw']) for sid in ids],
           'valid_g0':torch.ones(len(ids),dtype=torch.bool,device=device),
           'forensic_present':torch.ones(len(ids),dtype=torch.bool,device=device),
           'forensic_vacuous':torch.zeros(len(ids),dtype=torch.bool,device=device),
           'forensic_off':torch.zeros(len(ids),dtype=torch.bool,device=device)}
    with torch.no_grad():
        base=evidence_feature(source,raw)
    spatial=None if arm is None else cache[arm.arm.upper()].index_select(0,idx).to(device=device,dtype=torch.float32)
    if arm is None: f24,residual=base,None
    else: f24,residual=arm(base,spatial)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        zf=source.dense_head(f24)
        batch['F24']=f24
        batch['z_F24']=zf
        valid=torch.ones(len(idx),576,dtype=torch.bool,device=device)
        r=rect(s64,f24,sc,cc,valid)
        u=utility_forward_full_input_gradient(utility,batch)
    support=r['support'].reshape(len(idx),1,64,64)
    gate=hc.gate_to_sam_grid(u['U'],sc)*support.float()
    adapted=hc.gated_embedding(s64,r['image_embeddings'],gate)
    with torch.autocast(device_type='cuda',enabled=False):
        low=sam(batch['q_seg'],adapted.to(torch.bfloat16))
    loss=mask_loss(low,target,CFG)['total'] if need_loss else None
    return {'F24':f24,'base_F24':base,'z_F24':zf,'rectifier':r['image_embeddings'],'U':u['U'],
            'adapted':adapted,'low':low,'loss':loss,'residual':residual,'target':target,'ids':ids,
            'utility_batch':batch}


def preflight(device,train,dev,train_store,val_store,train_cache,val_cache,modules,frozen):
    idx=torch.tensor([int(train_cache['valid'].nonzero()[0])])
    probes={}
    with torch.no_grad():
        base=forward(train,train_cache,idx,train_store,modules,device)
        historical=hc.utility_forward(modules[0],base['utility_batch'])
    utility_value_error=float((historical['U'].float()-base['U'].float()).abs().max())
    require(utility_value_error==0,f'Phase6K full-input-gradient Utility forward changed values: {utility_value_error}')
    for name in ('a','e'):
        arm=AfterProjection(name).to(device)
        require(all(bool((p==0).all()) for p in arm.parameters()),f'{name} projection not zero initialized')
        with torch.no_grad():
            out=forward(train,train_cache,idx,train_store,modules,device,arm)
        equality={key:float((base[key].float()-out[key].float()).abs().max()) for key in
                  ('F24','z_F24','rectifier','U','adapted','low','loss')}
        require(max(equality.values())==0,f'{name} step0 baseline equivalence failed: {equality}')
        arm.zero_grad(set_to_none=True)
        out=forward(train,train_cache,idx,train_store,modules,device,arm)
        out['U'].float().mean().backward(retain_graph=True)
        utility_input_grad=float(arm.projection.weight.grad.detach().float().norm())
        require(utility_input_grad>0 and bool(torch.isfinite(torch.tensor(utility_input_grad))),
                f'{name} Utility input-gradient path failed')
        arm.zero_grad(set_to_none=True)
        out['loss'].backward()
        grads={key:None if p.grad is None else float(p.grad.detach().float().norm())
               for key,p in arm.named_parameters()}
        require(all(v is not None and torch.isfinite(torch.tensor(v)) and v>0 for v in grads.values()),
                f'{name} projection gradient failed: {grads}')
        require(all(p.grad is None for m in modules for p in m.parameters()),'frozen parameter received gradient')
        probes[name]={'step0_max_abs_error':equality,'gradient_norms':grads,
                      'utility_input_gradient_norm':utility_input_grad,
                      'trainable_names':[k for k,p in arm.named_parameters() if p.requires_grad],
                      'trainable_count':sum(p.numel() for p in arm.parameters())}
    require(frozen=={k:tensor_state_sha256(m.state_dict()) for k,m in zip(
        ('utility','rectifier','sam','source'),modules)},'frozen state changed during preflight')
    dump(AUDIT/'preflight.json',{'status':'PASS','arms':probes,'frozen_sha256':frozen,
         'train_n':len(train_cache['sample_ids']),'train_valid':int(train_cache['valid'].sum()),
         'dev_n':len(val_cache['sample_ids']),'dev_valid':int(val_cache['valid'].sum()),
         'historical_utility_forward_value_max_abs_error':utility_value_error,
         'gradient_route':'Rectifier and Phase6K value-identical Utility input-gradient interface'})
    return probes


def evaluate(data,cache,store,modules,device,arm=None,log_every=200):
    records=[]
    for i,sid in enumerate(cache['sample_ids']):
        if not bool(cache['valid'][i]):
            row=invalid_record(sid,data['original_masks'][i]); row['valid_g0']=False
            row['tn']=int(data['original_masks'][i].numel())-row['fn']
            records.append(row); continue
        with torch.no_grad():
            out=forward(data,cache,torch.tensor([i]),store,modules,device,arm,need_loss=False)
            logits=inverse_sam_logits(out['low'],data['sam_geometries'][i])
            row=metric_record(sid,logits,data['original_masks'][i]); row['valid_g0']=True
            row['tn']=int(data['original_masks'][i].numel())-row['tp']-row['fp']-row['fn']
            records.append(row)
        if (i+1)%log_every==0:
            print(json.dumps({'stage':'DEV','arm':'b0' if arm is None else arm.arm,'done':i+1,'total':len(cache['sample_ids'])}),flush=True)
    require(len(records)==len(cache['sample_ids']),'DEV record population drift')
    return summarize(records),records


def baseline(device):
    c2,native,payload=provenance()
    ids=train_ids(); dev=load_dev('g0')
    train_cache=load_cache('train',ids); val_cache=load_cache('val',dev['sample_ids'])
    require(train_cache['c2_sha256']==val_cache['c2_sha256']==c2['selected_checkpoint_sha256'], 'C2 cache/checkpoint drift')
    train_store=Phase4FStore(CFG,'train'); val_store=Phase4FStore(CFG,'val')
    require(train_store.sample_ids==ids and val_store.sample_ids==dev['sample_ids'],'Phase4F spatial ID order drift')
    require(all(geometry_for('clip',val_store.geometries[sid]['original_hw'])==geom
                for sid,geom in zip(dev['sample_ids'],dev['clip_geometries'])),
            'reconstructed CLIP crop geometry drift')
    modules_and_hash=runtime(device,payload)
    *modules,frozen=modules_and_hash
    train={}
    metric,records=evaluate(dev,val_cache,val_store,modules,device)
    dump(AUDIT/'b0/dev.json',{'status':'COMPLETE','metrics':metric,'records':records,
         'c2_sha256':c2['selected_checkpoint_sha256'],
         'native_sha256':native['selected_checkpoint_sha256'],
         'valid_count':int(val_cache['valid'].sum())})
    preflight(device,train,dev,train_store,val_store,train_cache,val_cache,modules,frozen)
    return train,dev,train_store,val_store,train_cache,val_cache,modules,frozen


def train_arm(name,device):
    require((AUDIT/'preflight.json').exists() and (AUDIT/'b0/dev.json').exists(), 'five preflight gates incomplete')
    c2,native,payload=provenance()
    ids=train_ids(); dev=load_dev('g0')
    train_cache=load_cache('train',ids); val_cache=load_cache('val',dev['sample_ids'])
    train_store=Phase4FStore(CFG,'train'); val_store=Phase4FStore(CFG,'val')
    modules_and_hash=runtime(device,payload); *modules,frozen=modules_and_hash
    train={}
    arm=AfterProjection(name).to(device)
    optimizer=torch.optim.AdamW(arm.parameters(),lr=LR,weight_decay=WD)
    outdir=AUDIT/name; ckptdir=CKPTS/name; outdir.mkdir(parents=True,exist_ok=True); ckptdir.mkdir(parents=True,exist_ok=True)
    require(not list(ckptdir.glob('epoch_*.pt')) and not (outdir/'selector.json').exists(),'arm already started')
    id_to_idx={sid:i for i,sid in enumerate(ids)}; valid=train_cache['valid'].bool()
    history=[]; updates=0
    for epoch in range(1,EPOCHS+1):
        started=time.time(); order=list(ids); random.Random(SEED+1009*epoch).shuffle(order)
        exposures=eligible=0; sumloss=0.; clip_count=0; last_grad=0.; rhos=[]; cosines=[]
        for begin in range(0,len(order),BATCH):
            positions=torch.tensor([id_to_idx[sid] for sid in order[begin:begin+BATCH]])
            idx=positions[valid.index_select(0,positions)]
            exposures+=len(positions); eligible+=len(idx)
            if not len(idx):continue
            optimizer.zero_grad(set_to_none=True)
            out=forward(train,train_cache,idx,train_store,modules,device,arm)
            loss=out['loss']; require(bool(torch.isfinite(loss)),'nonfinite mask loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(arm.parameters(),CLIP)
            require(bool(torch.isfinite(grad)),'nonfinite projection gradient')
            last_grad=float(grad);clip_count+=int(float(grad)>CLIP)
            optimizer.step();updates+=1;sumloss+=float(loss.detach())*len(idx)
            base=out['base_F24'].detach().float().flatten(1)
            changed=out['F24'].detach().float().flatten(1)
            residual=out['residual'].detach().float().flatten(1)
            batch_rho=residual.norm(dim=1)/base.norm(dim=1).clamp_min(1e-12)
            batch_cos=torch.nn.functional.cosine_similarity(base,changed,dim=1)
            rhos.extend(batch_rho.cpu().tolist());cosines.extend(batch_cos.cpu().tolist())
            if updates<=3 or updates%100==0:
                print(json.dumps({'stage':'TRAIN','arm':name,'epoch':epoch,'update':updates,
                                  'loss':float(loss.detach()),'gradient_norm':last_grad,
                                  'rho_median':float(batch_rho.median()),'rho_p95':float(torch.quantile(batch_rho,.95)),
                                  'rho_max':float(batch_rho.max()),'cosine_mean':float(batch_cos.mean())}),flush=True)
        require(exposures==8836 and eligible==int(valid.sum()),'epoch exposure drift')
        pre_path=ckptdir/f'epoch_{epoch}_pre_validation.pt'
        pre_tmp=pre_path.with_suffix('.pt.tmp')
        torch.save({'schema':'phase6k0_after_pre_validation_v1','arm':name,'epoch':epoch,
                    'updates':updates,'projection_state':arm.state_dict(),'optimizer':optimizer.state_dict(),
                    'c2_sha256':c2['selected_checkpoint_sha256'],
                    'native_sha256':native['selected_checkpoint_sha256'],'frozen_sha256':frozen},pre_tmp)
        os.replace(pre_tmp,pre_path)
        metric,records=evaluate(dev,val_cache,val_store,modules,device,arm)
        row={'epoch':epoch,'updates':updates,'train_loss':sumloss/eligible,'dev_mean_fg_iou':metric['mean_foreground_iou'],
             'dev_global_fg_iou':metric['global_foreground_iou'],'eligible':eligible,'invalid':exposures-eligible,
             'sample_order_sha256':hashlib.sha256('\n'.join(order).encode()).hexdigest(),
             'gradient_norm_last':last_grad,'gradient_clip_count':clip_count,
             'gradient_clip_rate':clip_count/max(1,updates-(history[-1]['updates'] if history else 0)),
             'rho_median':float(torch.tensor(rhos).median()),'rho_p95':float(torch.quantile(torch.tensor(rhos),.95)),
             'rho_max':max(rhos),'cosine_mean':sum(cosines)/len(cosines),'seconds':time.time()-started}
        history.append(row); dump(outdir/'history.json',history)
        tmp=ckptdir/f'epoch_{epoch}.pt.tmp'; path=ckptdir/f'epoch_{epoch}.pt'
        torch.save({'schema':'phase6k0_after_projection_v1','arm':name,'epoch':epoch,'projection_state':arm.state_dict(),
                    'optimizer':optimizer.state_dict(),'dev_metrics':metric,'c2_sha256':c2['selected_checkpoint_sha256'],
                    'native_sha256':native['selected_checkpoint_sha256'],'frozen_sha256':frozen},tmp)
        os.replace(tmp,path)
        dump(outdir/f'dev_epoch_{epoch}.json',{'metrics':metric,'records':records})
        print(json.dumps({'stage':'EPOCH_COMPLETE','arm':name,**row}),flush=True)
    require(frozen=={k:tensor_state_sha256(m.state_dict()) for k,m in zip(
        ('utility','rectifier','sam','source'),modules)},'frozen module drift')
    best=max(history,key=lambda x:(x['dev_mean_fg_iou'],-x['epoch']))
    chosen=ckptdir/f"epoch_{best['epoch']}.pt"
    dump(outdir/'selector.json',{'status':'COMPLETE','arm':name,'selected_epoch':best['epoch'],
         'selected_dev_mean_fg_iou':best['dev_mean_fg_iou'],'selected_checkpoint':str(chosen),
         'selected_checkpoint_sha256':file_sha256(chosen),'candidates':history,
         'official1000_used_for_selection':False,'internal_test_used_for_selection':False,
         'external_ood_used_for_selection':False,'frozen_sha256_before_after':frozen})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['baseline','train'],required=True)
    parser.add_argument('--arm',choices=['a','e']);parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args();device=torch.device(args.device);torch.cuda.set_device(device)
    if args.mode=='baseline':baseline(device)
    else:
        require(args.arm is not None,'--arm required');train_arm(args.arm,device)
