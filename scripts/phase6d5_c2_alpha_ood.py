#!/usr/bin/env python3
"""Phase6D.5 C2/RINE alpha OOD from frozen scores, with GPU-1 inversion audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase6d5_full_ood import MANIFESTS, label, read_image, image_path, metric
from scripts.phase6j0_c2_evaluate import load_c2_model
from tools.phase4c_b import file_sha256

OUT = ROOT/'outputs/phase6d5_c2_alpha_ood'
DOC = ROOT/'docs/phase6d5_c2_alpha_ood.md'
C2_CKPT = ROOT/'checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt'
C2_CFG = ROOT/'configs/phase6j0_c2_preln_cross_attention.yaml'
C2_PROTOCOL = ROOT/'outputs/phase6j0_c2/final_evaluation/protocol.json'
C2_OOD = ROOT/'outputs/phase6j0_c2/final_evaluation/classification_ood'
C2_CENTER = ROOT/'outputs/phase6j0_c2/center/results.json'
RINE_RAW = ROOT/'outputs/phase6d5_full_classification_ood/raw'
RINE_RESULT = ROOT/'outputs/phase6d5_full_classification_ood/results.json'
FUSION = ROOT/'outputs/phase6d5_decision_integration/fusion_parameters.json'
ALPHAS = [round(i/10, 1) for i in range(11)]


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def status(stage, **detail):
    save(OUT/'status.json',{'status':'RUNNING','stage':stage,
          'updated_utc':datetime.now(timezone.utc).isoformat(),**detail})


def frozen():
    protocol=json.loads(C2_PROTOCOL.read_text())
    center=json.loads(C2_CENTER.read_text())
    c2ood=json.loads((C2_OOD/'results.json').read_text())
    rine=json.loads(RINE_RESULT.read_text())
    fusion=json.loads(FUSION.read_text())
    digest=file_sha256(C2_CKPT)
    require(protocol['status']=='FROZEN' and protocol['selected_checkpoint_sha256']==digest and
            center['status']=='COMPLETE' and center['selected_checkpoint_sha256']==digest and
            c2ood['status']=='COMPLETE' and c2ood['checkpoint_sha256']==digest,
            'C2 selected checkpoint or evaluation drift')
    require((protocol['selected_step'],protocol['selected_epoch'])==(3500,7),
            'C2 selector step/epoch drift')
    require(rine['status']=='COMPLETE' and fusion['selected_alpha']==0.3,
            'RINE result or historical alpha drift')
    cfg=yaml.safe_load(C2_CFG.read_text())
    c1cfg=yaml.safe_load((ROOT/'configs/phase6d3_c1_rine_conditioned_p1.yaml').read_text())
    require(cfg['forensics']['rine_checkpoint_sha256']==c1cfg['forensics']['rine_checkpoint_sha256'],
            'C2 and historical RINE checkpoint mismatch')
    return digest,center,c2ood,rine,fusion,cfg


def population(name, c2ood):
    manifest=MANIFESTS[name]
    source=rows(manifest)
    path=Path(c2ood['datasets'][name]['prediction_file'])
    require(file_sha256(manifest)==c2ood['manifests'][name]['sha256'] and
            file_sha256(path)==c2ood['datasets'][name]['prediction_sha256'],
            f'{name} C2 OOD source drift')
    c2=rows(path)
    rine=rows(RINE_RAW/f'{name}.jsonl')
    require(len(source)==len(c2)==len(rine)==c2ood['manifests'][name]['count'],
            f'{name} population count drift')
    require(all(a['sample_id']==b['sample_id']==c['sample_id'] and
                label(a)==b['label']==c['label'] for a,b,c in zip(source,c2,rine)),
            f'{name} paired sample/label/order drift')
    return source,c2,rine


def audit(device_name):
    digest,center,c2ood,_,_,cfg=frozen()
    device=torch.device(device_name); torch.cuda.set_device(device)
    conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
    model,tokenizer,meta=load_c2_model(cfg,C2_CKPT,device,expected_step=3500,expected_epoch=7)
    require(meta['checkpoint_sha256']==digest,'loaded C2 checkpoint drift')
    model.eval().requires_grad_(False)
    processor=CLIPImageProcessor.from_pretrained(cfg['model']['vision_tower'],local_files_only=True)
    backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,max_new_tokens=400)
    overrides={}
    audit_details={}
    for name in MANIFESTS:
        source,c2,_=population(name,c2ood)
        p=np.asarray([x['cls_score_fake'] for x in c2],dtype=np.float64)
        require(np.isfinite(p).all() and ((p>0)&(p<=1)).all(),f'{name} C2 probability invalid')
        targets={int(i) for i in np.flatnonzero(p==1)}
        for value in (1e-5,center['probability_threshold'],0.05,0.5,0.95,0.9999):
            targets.add(int(np.argmin(np.abs(p-value))))
        targets.update((0,min(1,len(p)-1),len(p)//2,len(p)-1))
        groups={(i%2,(i//2)//8) for i in targets}
        exact={}
        abs_errors=[]
        max_probability_error=0.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for rank,group in sorted(groups):
                indices=[2*(group*8+j)+rank for j in range(8)
                         if 2*(group*8+j)+rank<len(source)]
                part=[source[i] for i in indices]
                images=list(pool.map(read_image,part))
                pixels=processor(images=images,return_tensors='pt')['pixel_values']
                samples=[{'image_path':image_path(row),'global_enc_image':pixel,
                          'grounding_enc_image':None,'bboxes':None,'conversations':[''],
                          'masks':None,'label':None,'resize':None,'questions':[],
                          'sampled_classes':[],'cls_label':label(row),'seg_valid':False,
                          'sample_id':row['sample_id'],'source':row.get('generator/source'),
                          'content_category':row.get('generator/source'),
                          'manifest_row':row}
                         for row,pixel in zip(part,pixels)]
                batch=backend._batch_many(samples,'',question=UNIFIED_FORENSICS_QUESTION)
                batch['grounding_enc_images']=None
                with torch.inference_mode():
                    output=model.model_forward(**batch)
                logits=output['cls_logits'].float()
                margin=(logits[:,1]-logits[:,0]).cpu().tolist()
                probability=logits.softmax(-1)[:,1].cpu().tolist()
                for i,m,pr in zip(indices,margin,probability):
                    require(math.isfinite(m),f'{name} nonfinite exact C2 margin')
                    old=float(c2[i]['cls_score_fake'])
                    delta=abs(pr-old)
                    max_probability_error=max(max_probability_error,delta)
                    require(delta<=1e-6,f'{name} C2 probability replay mismatch: '
                            f'{c2[i]["sample_id"]}: old={old}, new={pr}')
                    exact[c2[i]['sample_id']]=float(m)
                    if old<1:
                        reconstructed=math.log(old)-math.log1p(-old)
                        abs_errors.append(abs(reconstructed-m))
        require(all(c2[i]['sample_id'] in exact for i in np.flatnonzero(p==1)),
                f'{name} saturated probabilities missing exact override')
        overrides[name]=exact
        audit_details[name]={'n':len(source),'audited_images':len(exact),
                             'saturated_p_eq_1':int((p==1).sum()),
                             'max_probability_replay_error':max_probability_error,
                             'max_non_saturated_logit_reconstruction_error':max(abs_errors,default=0.),
                             'median_non_saturated_logit_reconstruction_error':float(np.median(abs_errors)) if abs_errors else 0.}
        status('GPU1_AUDIT',dataset=name,**audit_details[name])
    save(OUT/'audit.json',{'status':'PASS','device':device_name,'c2_sha256':digest,
                           'datasets':audit_details,'exact_overrides':overrides})


def scores(y, values):
    return metric([{'label':int(a),'score':float(b)} for a,b in zip(y,values)],'score')


def finalize():
    digest,center,c2ood,rine_result,fusion,_=frozen()
    audit_result=json.loads((OUT/'audit.json').read_text())
    require(audit_result['status']=='PASS' and audit_result['c2_sha256']==digest,
            'GPU1 reconstruction audit incomplete')
    alpha=float(fusion['selected_alpha'])
    cn={'mean':float(center['train_margin_mean']),'std':float(center['train_margin_std'])}
    rn=fusion['normalization']['rine']
    require(cn['std']>0 and rn['std']>0,'invalid TRAIN normalization')
    result={'schema':'phase6d5_c2_alpha_ood_reuse_v1','status':'COMPLETE',
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':{'c2_checkpoint_sha256':digest,
                        'rine_checkpoint_sha256':yaml.safe_load(C2_CFG.read_text())['forensics']['rine_checkpoint_sha256'],
                        'c2_train_normalization':cn,'rine_train_normalization':rn,
                        'historical_validation_selected_alpha':alpha,'alpha_grid':ALPHAS,
                        'formula':'alpha*sR+(1-alpha)*sC; C2 score = logit(frozen OOD probability), with GPU1 exact overrides; each branch normalized by its own internal TRAIN mean/std',
                        'decision':'score>0 => Fake','ood_selection_or_tuning':False,
                        'full_ood_model_rerun':False,'gpu1_exact_audit':True},
            'datasets':{},'mixed_ood_macro':{},'source_files':{},
            'audit':{'path':str((OUT/'audit.json').resolve()),'sha256':file_sha256(OUT/'audit.json'),
                     'datasets':audit_result['datasets']}}
    lines=['# Phase6D.5 — C2/RINE alpha OOD','',
           'Status: **COMPLETE**. Reused the frozen C2 classification OOD probabilities and Phase6D.5 RINE raw margins on identical sample IDs. The C2 raw margin is reconstructed with `logit(p)`; GPU 1 replayed all saturated probabilities and representative original batch-8 groups to check the inversion.','',
           f'The formal alpha is the historical validation-selected **{alpha:.1f}**. The 0.0–1.0 grid is exploratory; no OOD parameter selection was done. C2 and RINE each use their own frozen internal-TRAIN mean/std; a score > 0 predicts Fake.','',
           f'C2 checkpoint SHA256 `{digest}`; RINE checkpoint SHA256 `{result["protocol"]["rine_checkpoint_sha256"]}`.','']
    for name,manifest in MANIFESTS.items():
        source,c2,rine=population(name,c2ood)
        exact=audit_result['exact_overrides'][name]
        p=np.asarray([x['cls_score_fake'] for x in c2],dtype=np.float64)
        margin=np.empty(len(p),dtype=np.float64)
        finite=p<1
        margin[finite]=np.log(p[finite])-np.log1p(-p[finite])
        for i,row in enumerate(c2):
            if row['sample_id'] in exact:
                margin[i]=exact[row['sample_id']]
        require(np.isfinite(margin).all(),f'{name} missing exact C2 margin override')
        rm=np.asarray([x['rine_margin'] for x in rine],dtype=np.float64)
        require(np.isfinite(rm).all(),f'{name} nonfinite RINE margin')
        y=np.asarray([x['label'] for x in c2],dtype=np.int64)
        sc=(margin-cn['mean'])/cn['std']
        sr=(rm-rn['mean'])/rn['std']
        table={str(a):scores(y,a*sr+(1-a)*sc) for a in ALPHAS}
        # Half-ULP float32 brackets isolate uncertainty from saved softmax rounding.
        p32=p.astype(np.float32)
        p_lo=(p+np.nextafter(p32,np.float32(0)).astype(np.float64))/2
        p_hi=(p+np.nextafter(p32,np.float32(1)).astype(np.float64))/2
        p_lo[p32==1]=0.5; p_hi[p32==1]=0.5
        lower=alpha*sr+(1-alpha)*((np.log(p_lo)-np.log1p(-p_lo)-cn['mean'])/cn['std'])
        upper=alpha*sr+(1-alpha)*((np.log(p_hi)-np.log1p(-p_hi)-cn['mean'])/cn['std'])
        is_exact=np.asarray([x['sample_id'] in exact for x in c2],dtype=bool)
        selected=alpha*sr+(1-alpha)*sc
        lower[is_exact]=selected[is_exact];upper[is_exact]=selected[is_exact]
        rounding_ambiguous=int(((lower<=0)&(upper>=0)).sum())
        require(rounding_ambiguous==0,
                f'{name} alpha=0.3 decisions ambiguous under float32 half-ULP rounding')
        pure=table['0.0']; center_metrics=center['classification_ood'][name]
        for key in ('accuracy','tnr','fpr'):
            require(abs(pure[key]-center_metrics[key])<1e-12,
                    f'{name} C2-center {key} decision parity failed')
        if name!='raise998':
            require(abs(pure['fake_recall']-center_metrics['fake_recall'])<1e-12,
                    f'{name} C2-center recall parity failed')
        require(rine_result['datasets'][name]['arms']['D1-alpha']['n']==len(source),
                f'{name} historical Phase6D population drift')
        prediction=OUT/'predictions'/f'{name}.jsonl'; prediction.parent.mkdir(parents=True,exist_ok=True)
        with prediction.open('w') as stream:
            for i,(c,r,x,z) in enumerate(zip(c2,rine,sc,sr)):
                stream.write(json.dumps({'sample_id':c['sample_id'],'label':c['label'],
                    'generator':r.get('generator'),'c2_margin':float(margin[i]),
                    'rine_margin':float(rm[i]),'sC':float(x),'sR':float(z),
                    'alpha_0_3_score':float(alpha*z+(1-alpha)*x)},ensure_ascii=False)+'\n')
        entry={'n':len(source),'alpha_metrics':table,
               'selected_alpha_float32_rounding_ambiguous_count':rounding_ambiguous,
               'prediction_file':str(prediction.resolve()),
               'prediction_sha256':file_sha256(prediction)}
        if name=='genimage':
            generator=np.asarray([str(r.get('generator')) for r in rine])
            entry['per_generator']={g:{str(a):scores(y[generator==g],
                (a*sr+(1-a)*sc)[generator==g]) for a in ALPHAS}
                for g in sorted(set(generator.tolist()))}
        result['datasets'][name]=entry
        result['source_files'][name]={'manifest_sha256':file_sha256(manifest),
            'c2_prediction_sha256':file_sha256(Path(c2ood['datasets'][name]['prediction_file'])),
            'rine_raw_sha256':file_sha256(RINE_RAW/f'{name}.jsonl')}
        d=audit_result['datasets'][name]
        lines += [f'## {name} (n={len(source)})','',
                  f'GPU 1 audit: {d["audited_images"]} images, including all {d["saturated_p_eq_1"]} saturated probabilities; maximum non-saturated `logit(p)` reconstruction difference {d["max_non_saturated_logit_reconstruction_error"]:.6g}. At alpha=0.3, {rounding_ambiguous} decisions straddle zero within the saved float32 probability half-ULP interval.','',
                  '| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        for a in ALPHAS:
            m=table[str(a)]; fmt=lambda v:'-' if v is None else f'{v:.6f}'
            lines.append(f'| {a:.1f} | {fmt(m["accuracy"])} | {fmt(m["roc_auc"])} | '
                         f'{fmt(m["fake_recall"])} | {fmt(m["tnr"])} | {fmt(m["fpr"])} | {fmt(m["f1"])} |')
        lines.append('')
    for a in ALPHAS:
        vals=[result['datasets'][name]['alpha_metrics'][str(a)]
              for name in ('aigi_holmes','genimage','loki')]
        result['mixed_ood_macro'][str(a)]={key:float(np.mean([v[key] for v in vals]))
            for key in ('accuracy','roc_auc','fake_recall','tnr','fpr','f1')}
    lines += ['## Mixed OOD macro (AIGI-Holmes, GenImage, LOKI)','',
              '| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for a in ALPHAS:
        m=result['mixed_ood_macro'][str(a)]
        lines.append(f'| {a:.1f} | {m["accuracy"]:.6f} | {m["roc_auc"]:.6f} | '
                     f'{m["fake_recall"]:.6f} | {m["tnr"]:.6f} | {m["fpr"]:.6f} | {m["f1"]:.6f} |')
    lines += ['', 'RAISE998 contains only Real images, so ROC-AUC and Fake recall are undefined.',
              'GenImage per-generator results and paired per-image scores are retained in the result artifacts.',
              'Precision limit: unsaturated C2 margins are reconstructed from saved float32 probabilities, so ROC-AUC values are numerical approximations to a full raw-logit replay. All 19 saturated cases use exact GPU 1 margins. At formal alpha=0.3, no decision threshold is ambiguous under the saved probability half-ULP intervals.','']
    lines += ['## Formal alpha=0.3 comparison','',
              '| Dataset | C2-center Acc / AUC / F1 | C2+RINE alpha=0.3 Acc / AUC / F1 | Historical C1+RINE alpha=0.3 Acc / AUC / F1 |',
              '|---|---|---|---|']
    for name in MANIFESTS:
        center_m=result['datasets'][name]['alpha_metrics']['0.0']
        c2_m=result['datasets'][name]['alpha_metrics']['0.3']
        c1_m=rine_result['datasets'][name]['arms']['D1-alpha']
        def triplet(m):
            auc='-' if m['roc_auc'] is None else f'{m["roc_auc"]:.6f}'
            return f'{m["accuracy"]:.6f} / {auc} / {m["f1"]:.6f}'
        lines.append(f'| {name} | {triplet(center_m)} | {triplet(c2_m)} | {triplet(c1_m)} |')
    lines += ['',
              'At the frozen alpha=0.3, C2+RINE increases ROC-AUC over C2-center on AIGI-Holmes, GenImage, and LOKI, while fixed-threshold accuracy and F1 decrease on all three. Historical C1+RINE alpha=0.3 has higher accuracy, ROC-AUC, and F1 on those same populations. The alpha sweep remains exploratory.','']
    save(OUT/'results.json',result)
    DOC.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('run','audit','finalize'),default='run')
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    try:
        if args.mode!='finalize':
            status('GPU1_AUDIT',pid=os.getpid(),device=args.device)
            audit(args.device)
        if args.mode!='audit':
            status('FINALIZE',pid=os.getpid())
            finalize()
        save(OUT/'status.json',{'status':'COMPLETE','stage':'COMPLETE_STOP',
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'result':str((OUT/'results.json').resolve()) if args.mode!='audit' else None,
            'report':str(DOC.resolve()) if args.mode!='audit' else None})
    except BaseException as exc:
        save(OUT/'status.json',{'status':'FAILED','stage':'COMPLETE_STOP',
            'updated_utc':datetime.now(timezone.utc).isoformat(),
            'error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    main()
