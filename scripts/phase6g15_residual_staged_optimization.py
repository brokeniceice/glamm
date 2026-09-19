#!/usr/bin/env python3
"""Phase 6G.15 budget-normalized residual staged optimization arms."""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import phase6g10_train_arm as g10
from scripts.phase6g14_single_matched_translator import build_matched_arms, dump, grad_norms
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from tools.phase4e1 import tensor_state_sha256

OUT=ROOT/'outputs/phase6g15_residual_staged_optimization'
G14=ROOT/'outputs/phase6g14_single_matched_translator'
ARMS=('A1','A2','C1','C2')

class AnchoredProjection(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__(); self.base=copy.deepcopy(base).requires_grad_(False); self.residual=nn.Linear(256,256)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
    def forward(self,x): return self.base(x)+self.residual(x)

def selected_stage1_epoch():
    curve=json.loads((G14/'arms/A1/training_curve.json').read_text())
    candidates=[x for x in curve if int(x['epoch'])<=5]
    return int(max(candidates,key=lambda x:(x['dev_g0_mean_iou'],-x['epoch']))['epoch']),candidates

def base_model(device):
    models,_=build_matched_arms(device); model=models['A1']; epoch,curve=selected_stage1_epoch()
    payload=torch.load(G14/f'arms/A1/checkpoints/epoch_{epoch}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    return model,epoch,curve,payload

def configure(arm,device):
    model,epoch,curve,payload=base_model(device); original=model.rectification.projection
    if arm in ('A1','A2'):
        model.rectification.projection=AnchoredProjection(original).to(device)
    model.requires_grad_(False)
    if arm=='A1': model.rectification.projection.residual.requires_grad_(True)
    elif arm=='A2':
        for n,p in model.named_parameters():
            if not n.startswith('rectification.projection.base.'): p.requires_grad_(True)
        model.rectification.projection.base.requires_grad_(False)
    elif arm=='C1': model.rectification.projection.requires_grad_(True)
    elif arm=='C2': model.requires_grad_(True)
    else: raise ValueError(arm)
    return model,epoch,curve,payload

def effective(model,arm):
    p=model.rectification.projection
    if arm in ('A1','A2'):
        w=p.base.weight.float()+p.residual.weight.float(); b=p.base.bias.float()+p.residual.bias.float()
        rw=p.residual.weight.float(); rb=p.residual.bias.float(); bw=p.base.weight.float()
        extra={'residual_weight_norm':float(rw.norm()),'residual_bias_norm':float(rb.norm()),
               'base_weight_norm':float(bw.norm()),'residual_over_base':float(rw.norm()/bw.norm().clamp_min(1e-12)),
               'base_residual_cosine':float(F.cosine_similarity(bw.flatten(),rw.flatten(),dim=0)) if float(rw.norm()) else 0.0}
    else: w=p.weight.float(); b=p.bias.float(); extra={}
    return w,b,{'effective_weight_norm':float(w.norm()),'effective_bias_norm':float(b.norm()),**extra}

def ownership(model):
    rows=[{'name':n,'shape':list(p.shape),'numel':p.numel()} for n,p in model.named_parameters() if p.requires_grad]
    return {'count':sum(x['numel'] for x in rows),'parameters':rows}

def equivalence(model,arm,device):
    gen=torch.Generator().manual_seed(4815); x=torch.randn(2,4096,256,generator=gen).to(device)
    with torch.no_grad():
        p=model.rectification.projection
        if arm in ('A1','A2'): y=p(x); ref=p.base(x)
        else: y=ref=p(x)
        d=(y.float()-ref.float()); rel=d.norm()/ref.float().norm().clamp_min(1e-12)
    return {'max_abs_error':float(d.abs().max()),'relative_l2_error':float(rel),'passed':float(d.abs().max())==0.0}

def translator_grad_norm(model,arm):
    total=0.0
    target='rectification.projection.residual' if arm in ('A1','A2') else 'rectification.projection'
    for n,p in model.named_parameters():
        if target in n and p.grad is not None: total+=float(p.grad.detach().float().square().sum())
    return math.sqrt(total)

def train(arm,device):
    root=OUT/'arms'/arm
    if (root/'summary.json').exists(): print(json.dumps({'arm':arm,'status':'ALREADY_COMPLETE'}),flush=True); return
    model,base_epoch,stage1_curve,base_payload=configure(arm,device)
    eq=equivalence(model,arm,device)
    if not eq['passed']: raise RuntimeError(f'Stage2 zero residual parity failed: {eq}')
    init={'arm':arm,'stage1_source':str(G14/f'arms/A1/checkpoints/epoch_{base_epoch}.pt'),'stage1_selected_epoch':base_epoch,
          'stage1_candidates':stage1_curve,'stage2_equivalence':eq,'ownership':ownership(model),
          'initial_state_sha256':tensor_state_sha256(model.state_dict()),'global_epochs':[6,7,8,9,10]}
    dump(root/'initialization.json',init)
    fusion=g10.load_fusion(device); projection=g10.load_projection(device); sam=g10.load_sam_runtime(g10.hd.CFG,device)
    p4t,p4v=g10.Phase4FStore(g10.hd.CFG,'train'),g10.Phase4FStore(g10.hd.CFG,'val'); fst,fsv=g10.Store('train'),g10.Store('val'); dev=g10.load_dev('g0')
    ids=p4t.sample_ids; tc=load_c1_cache('train',ids); vc=load_c1_cache('val',dev['sample_ids']); valid=tc['valid'].bool(); where={s:i for i,s in enumerate(ids)}
    params=[p for p in model.parameters() if p.requires_grad]; steps=math.ceil(len(ids)/g10.BATCH)*5
    opt,sch=g10.optimizer_and_scheduler(params,steps); hist=[]; updates=0; start_epoch=1
    curve_path=root/'training_curve.json'
    checkpoints=sorted((root/'checkpoints').glob('epoch_*.pt')) if (root/'checkpoints').exists() else []
    if checkpoints:
        latest=max(checkpoints,key=lambda p:int(p.stem.split('_')[-1])); state=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
        start_epoch=int(state['stage2_epoch'])+1;updates=int(state['updates']);hist=json.loads(curve_path.read_text()) if curve_path.exists() else []
        hist=[x for x in hist if int(x['stage2_epoch'])<start_epoch]
        print(json.dumps({'arm':arm,'status':'RESUME','from_stage2_epoch':start_epoch-1,'updates':updates}),flush=True)
    for local_epoch in range(start_epoch,6):
        global_epoch=local_epoch+5
        began=time.time(); model.train(); order=list(ids); random.Random(g10.SEED+1009*global_epoch).shuffle(order)
        sums={'loss':0.,'gn':0.,'tgn':0.}; count=nstep=0
        for begin in range(0,len(order),g10.BATCH):
            block=order[begin:begin+g10.BATCH]; pos=torch.tensor([where[s] for s in block]); elig=pos[valid.index_select(0,pos)]
            if not len(elig): continue
            batchids=[ids[i] for i in elig.tolist()]; q=tc['q_seg'].index_select(0,elig).to(device=device,dtype=torch.bfloat16)
            opt.zero_grad(set_to_none=True); loss=g10.rectifier_train_step(model,sam,fusion,projection,p4t,fst,q,batchids,device); loss['total'].backward()
            gn=math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None)); tgn=translator_grad_norm(model,arm)
            torch.nn.utils.clip_grad_norm_(params,g10.GRAD_CLIP); opt.step(); sch.step(); updates+=1; nstep+=1; count+=len(batchids)
            sums['loss']+=float(loss['total'].detach())*len(batchids); sums['gn']+=gn; sums['tgn']+=tgn
        met,recs=g10.rectifier_evaluate(model,sam,fusion,projection,p4v,fsv,dev,vc,device); _,_,diag=effective(model,arm)
        row={'stage2_epoch':local_epoch,'global_epoch':global_epoch,'stage2_updates':updates,'total_lineage_epochs':base_epoch+local_epoch,
             'seg_loss':sums['loss']/max(1,count),'dev_g0_mean_iou':met['mean_foreground_iou'],'dev_g0_mean_f1':met['mean_foreground_f1'],
             'gradient_norm_mean':sums['gn']/max(1,nstep),'translator_gradient_norm_mean':sums['tgn']/max(1,nstep),
             'gamma_main':float(model.rectification.gamma.detach()),**diag,'seconds':time.time()-began}
        hist.append(row); dump(curve_path,hist); (root/'checkpoints').mkdir(parents=True,exist_ok=True)
        torch.save({'schema':'phase6g15_stage2_v1','arm':arm,'stage2_epoch':local_epoch,'global_epoch':global_epoch,'updates':updates,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict()},root/'checkpoints'/f'epoch_{local_epoch}.pt')
        g10.write_rows(root/'validation'/f'epoch_{local_epoch}.jsonl',recs); print(json.dumps({'stage':'G15_EPOCH','arm':arm,**row}),flush=True)
    best=max(hist,key=lambda x:(x['dev_g0_mean_iou'],-x['stage2_epoch'])); sel=root/'selected_checkpoint.pt'; sel.write_bytes((root/'checkpoints'/f"epoch_{best['stage2_epoch']}.pt").read_bytes())
    state=torch.load(sel,map_location='cpu',weights_only=False); model.load_state_dict(state['model']); met,recs=g10.rectifier_evaluate(model,sam,fusion,projection,p4v,fsv,dev,vc,device)
    g10.write_rows(root/'validation/selected.jsonl',recs); w,b,diag=effective(model,arm)
    dump(root/'summary.json',{'status':'COMPLETE','arm':arm,'stage1_selected_epoch':base_epoch,'stage2_selected_epoch':best['stage2_epoch'],'metrics':met,
         'effective_map':g10.spectrum(w.detach().cpu()),'effective_bias_norm':float(b.norm()),'diagnostics':diag,'ownership':ownership(model),
         'firewall':{'utility':False,'nonlinear':False,'test':False,'official1000':False,'ood':False}})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=ARMS,required=True); ap.add_argument('--device',required=True); a=ap.parse_args()
    d=torch.device(a.device); torch.cuda.set_device(d); train(a.arm,d)
if __name__=='__main__': main()
