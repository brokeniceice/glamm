#!/usr/bin/env python3
"""Train preregistered small heads over frozen Phase6B feature caches."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6b_classification_attribution"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(labels, prob):
    labels = np.asarray(labels, dtype=np.int64); prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        take = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if take.any(): ece += take.mean() * abs(labels[take].mean() - prob[take].mean())
    return {"n": len(labels), "accuracy": accuracy_score(labels, pred), "balanced_accuracy": balanced_accuracy_score(labels, pred),
            "roc_auc": roc_auc_score(labels, prob), "precision": precision_score(labels, pred, zero_division=0),
            "fake_recall": recall_score(labels, pred), "tnr": tn / (tn + fp), "fpr": fp / (tn + fp),
            "f1": f1_score(labels, pred), "brier": brier_score_loss(labels, prob), "ece": float(ece),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5}


def make_head(in_dim, hidden):
    if hidden is None: return torch.nn.Linear(in_dim, 2)
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2))


def evaluate(model, x, y, batch=2048):
    model.eval(); probs=[]
    with torch.no_grad():
        for i in range(0, len(y), batch):
            probs.append(model(x[i:i+batch]).softmax(1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    return metrics(y.cpu().numpy(), p), p


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    manifest=json.loads((OUT/'feature_manifest.json').read_text())
    if manifest['status']!='COMPLETE' or manifest['integrity']['population_failures']!=0: raise RuntimeError('feature cache incomplete')
    train=torch.load(OUT/'features/train/features.pt',map_location='cpu'); val=torch.load(OUT/'features/val/features.pt',map_location='cpu')
    if train['sample_ids'] != [json.loads(x)['sample_id'] for x in (ROOT/'outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl').read_text().splitlines() if x]: raise RuntimeError('train ID drift')
    if val['sample_ids'] != [json.loads(x)['sample_id'] for x in (ROOT/'outputs/data_audits/unified_forensics_split_v1/val_combined.jsonl').read_text().splitlines() if x]: raise RuntimeError('val ID drift')
    # Respect CUDA_VISIBLE_DEVICES; the assigned physical GPU is exposed as 0.
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    specs={'C1-L':('clip_cls',1024,2048),'C1-S':('clip_cls',1024,128),'C2':('f24_gap',256,512),
           'LP-CLIP':('clip_cls',1024,None),'LP-F24':('f24_gap',256,None)}
    lrs=protocol['training']['learning_rate_candidates']; seeds=protocol['training']['seeds']; epochs=protocol['training']['epochs']; bs=protocol['training']['batch_size']
    all_runs=[]; selected=[]; ckroot=OUT/'checkpoints'; ckroot.mkdir(parents=True,exist_ok=True)
    for arm,(feature,in_dim,hidden) in specs.items():
      xtr=train[feature].float(); ytr=train['labels']; xv=val[feature].float().to(device); yv=val['labels'].to(device)
      for seed in seeds:
        candidates=[]
        for lr in lrs:
          seed_all(seed); model=make_head(in_dim,hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=protocol['training']['weight_decay'],betas=tuple(protocol['training']['betas']))
          steps=epochs*math.ceil(len(ytr)/bs); warm=max(1,int(.05*steps))
          def factor(step):
            if step < warm: return (step+1)/warm
            return .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,steps-warm))))
          sched=torch.optim.lr_scheduler.LambdaLR(opt,factor); gen=torch.Generator().manual_seed(seed); global_step=0
          for epoch in range(1,epochs+1):
            order=torch.randperm(len(ytr),generator=gen); model.train(); total=0.0
            for begin in range(0,len(order),bs):
              idx=order[begin:begin+bs]; xb=xtr[idx].to(device); yb=ytr[idx].to(device)
              opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(model(xb),yb)
              if not torch.isfinite(loss): raise RuntimeError(f'nonfinite {arm} {seed} {lr}')
              loss.backward(); opt.step(); sched.step(); global_step+=1; total += float(loss.detach())*len(idx)
            met,prob=evaluate(model,xv,yv); row={'arm':arm,'feature':feature,'seed':seed,'lr':lr,'epoch':epoch,'train_loss':total/len(ytr),'metrics':met}
            all_runs.append(row); candidates.append((row,{k:v.detach().cpu() for k,v in model.state_dict().items()},prob))
        best=max(candidates,key=lambda z:(z[0]['metrics']['roc_auc'],z[0]['metrics']['accuracy'],-z[0]['epoch'],-z[0]['lr']))
        ck=ckroot/f'{arm}_seed{seed}.pt'; torch.save({'schema':'phase6b_head_v1','arm':arm,'feature':feature,'in_dim':in_dim,'hidden':hidden,'seed':seed,'lr':best[0]['lr'],'epoch':best[0]['epoch'],'state_dict':best[1],'metrics':best[0]['metrics'],'protocol_sha256':sha256_file(OUT/'protocol.json')},ck)
        pred_path=ckroot/f'{arm}_seed{seed}_val_predictions.pt'; torch.save({'sample_ids':val['sample_ids'],'labels':val['labels'],'prob_fake':torch.from_numpy(best[2])},pred_path)
        selected.append({**best[0],'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha256_file(ck),'predictions':str(pred_path.resolve())})
        print(json.dumps({'selected':selected[-1]}),flush=True)
    (OUT/'training_history.json').write_text(json.dumps(all_runs,indent=2)+'\n')
    (OUT/'selected_runs.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','selected_runs':len(selected)}),flush=True)

if __name__=='__main__': main()
