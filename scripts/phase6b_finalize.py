#!/usr/bin/env python3
"""Aggregate Phase6B internal-only attribution results and write the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6b_classification_attribution'
DOC=ROOT/'docs/phase6b_classification_attribution_results.md'
METRICS=['accuracy','balanced_accuracy','roc_auc','precision','fake_recall','tnr','fpr','f1','brier','ece']


def aggregate(rows):
    result={}
    for k in METRICS:
        v=np.asarray([r['metrics'][k] for r in rows],np.float64)
        result[k]={'mean':float(v.mean()),'std':float(v.std(ddof=1))}
    result['selected']=[{'seed':r['seed'],'lr':r['lr'],'epoch':r['epoch'],'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256']} for r in rows]
    return result


def feature_stats(x,y):
    x=x.double(); y=y.long(); normalized=torch.nn.functional.normalize(x,dim=1)
    classes={}
    sums=[]
    for cls in (0,1):
        z=normalized[y==cls]; s=z.sum(0); n=len(z); sums.append(s)
        within=float(((s@s)-n)/(n*(n-1)))
        classes[str(cls)]={'n':n,'normalized_centroid_norm':float((s/n).norm()),'within_class_cosine':within}
    between=float((sums[0]@sums[1])/(classes['0']['n']*classes['1']['n']))
    normalized_centroid_distance=float((sums[0]/classes['0']['n']-sums[1]/classes['1']['n']).norm())
    raw0=x[y==0].mean(0); raw1=x[y==1].mean(0)
    return {'classes':classes,'between_class_cosine':between,'within_minus_between':{c:classes[c]['within_class_cosine']-between for c in ('0','1')},
            'normalized_centroid_l2_distance':normalized_centroid_distance,'raw_centroid_l2_distance':float((raw0-raw1).norm())}


def pm(v): return f"{v['mean']:.6f} ± {v['std']:.6f}"


def main():
    protocol=json.loads((OUT/'protocol.json').read_text()); fm=json.loads((OUT/'feature_manifest.json').read_text())
    selected=json.loads((OUT/'selected_runs.json').read_text())
    if len(selected)!=15: raise RuntimeError('selected run count mismatch')
    by={a:[r for r in selected if r['arm']==a] for a in ('C1-L','C1-S','C2','LP-CLIP','LP-F24')}
    if any(len(v)!=3 for v in by.values()): raise RuntimeError('seed count mismatch')
    agg={a:aggregate(rows) for a,rows in by.items()}
    val=torch.load(OUT/'features/val/features.pt',map_location='cpu')
    diagnostics={'clip_cls':feature_stats(val['clip_cls'],val['labels']),'f24_gap':feature_stats(val['f24_gap'],val['labels'])}
    def delta(a,b,k): return agg[a][k]['mean']-agg[b][k]['mean']
    attribution={
      'C1-L_minus_C1-S':{k:delta('C1-L','C1-S',k) for k in METRICS},
      'C1-S_minus_C2':{k:delta('C1-S','C2',k) for k in METRICS},
      'C0_reference':{'source':'Phase3D.1 P1-FROZEN generation-derived validation classification','n':2212,'accuracy':0.9864376130198915,
                      'strict_matched_control':False,'other_metrics':'not historically preserved; intentionally not reconstructed by new inference'},
      'interpretation':{
        'head_capacity':'C1-L and C1-S are close; large-head increment is small on internal validation',
        'feature':'capacity-matched CLIP CLS strongly exceeds F24 GAP',
        'f24_signal':'non-random independent signal exists, but it is much weaker than C0 and CLIP CLS',
      }
    }
    decisions={
      'clip_global_classifier':'GO_FOR_FREEZE_CONFIRMATION_C1-L',
      'head_capacity':'SMALL_POSITIVE_NOT_PRIMARY',
      'r1_f24_reuse':'NO-GO_WEAK_FOR_PRIMARY_CLASSIFIER',
      'c3_fusion':'NO-GO_IN_THIS_ROUTE',
      'next_candidate':'C1-L',
      'external_evaluation_started':False,
    }
    results={'schema':'phase6b_classification_attribution_results_v1','status':'COMPLETE_INTERNAL_ONLY','protocol':protocol,
             'feature_manifest':fm,'c0':attribution['C0_reference'],'arms':agg,'attribution':attribution,
             'feature_diagnostics':diagnostics,'decisions':decisions,
             'firewall':{'internal_test_accessed':False,'official1000_accessed':False,'external_accessed':False,'threshold_sweep':False,'c3_trained':False}}
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    rows=[]
    for arm in ('C1-L','C1-S','C2'):
        a=agg[arm]; feature='CLIP CLS' if arm.startswith('C1') else 'F24 GAP'; params={'C1-L':2103298,'C1-S':131458,'C2':132610}[arm]
        rows.append(f"| {arm} | {feature} | {params:,} | {pm(a['accuracy'])} | {pm(a['balanced_accuracy'])} | {pm(a['roc_auc'])} | {pm(a['fake_recall'])} | {pm(a['tnr'])} | {pm(a['fpr'])} | {pm(a['f1'])} | {pm(a['brier'])} | {pm(a['ece'])} |")
    sel=[]
    for arm in ('C1-L','C1-S','C2','LP-CLIP','LP-F24'):
        sel.append('| '+arm+' | '+'; '.join(f"seed {x['seed']}: lr={x['lr']:g}, epoch={x['epoch']}" for x in agg[arm]['selected'])+' |')
    d=diagnostics
    report=f"""# Phase 6B — Classification Attribution / Minimal Improvement

## Outcome

Phase 6B 已在冻结 internal TRAIN/validation 上完成，所有新模型都只是 frozen-feature classification head。没有更新 CLIP、P1/R1、LLM/LoRA、forensic arm、SAM、rectifier 或现有 classifier；没有访问 Internal test、Official1000 或 external benchmark，也没有训练 C3。

结论：**CLIP CLS 是主要有效 representation；C1-L 相对 C1-S 的大头容量增益很小。F24 GAP 明确含有独立 Real/Fake signal，但显著弱于 C0 与 capacity-matched CLIP CLS，不支持进入正式 C3 fusion。下一步候选冻结为 C1-L，但本阶段不自动做 external confirmation。**

## Protocol

- data：冻结 TRAIN 17,672（8,836 Real + 8,836 Fake）；validation 2,212（1,106 + 1,106），原 manifest 顺序、无 resplit；Fake=1。
- feature：`openai/clip-vit-large-patch14-336` revision `{fm['clip']['revision']}`，BF16 frozen penultimate hidden token 0；同一次 forward 的 patch tokens 经 frozen Phase4C-A selected forensic adapter 后 GAP 得 F24 `[256]`。
- integrity：TRAIN/validation extraction failure 均为 0；两 feature 共用 sample ID、label、image 与 order。CLIP parameter hash `{fm['clip']['parameter_sha256']}`；forensic checkpoint SHA256 `{fm['forensic_arm']['checkpoint_sha256']}`；R1 SHA256 `{fm['r1_checkpoint']['sha256']}`。
- heads：C1-L `1024→2048→2`（2,103,298）；C1-S `1024→128→2`（131,458）；C2 `256→512→2`（132,610）。
- optimizer：AdamW，LR candidates `1e-4/3e-4/1e-3`，weight decay `1e-4`，10 epochs，batch 512，5% linear warmup + cosine decay。
- seeds：3407/3408/3409；每 seed 以 validation ROC-AUC 最大选择，tie 依次为 Accuracy、earlier epoch、smaller LR。
- 所有 operating-point metrics 固定 threshold 0.5；ECE 为 15 个 equal-width bins。完整预注册协议见 `outputs/phase6b_classification_attribution/protocol.json`。

## Main internal-validation results

多 seed 数值为 mean ± sample std。C0 是历史 generation-derived architecture reference，只保留 Accuracy，不能当作同 regime 单变量控制。

| Arm | Feature | Head params | Acc | Balanced Acc | ROC-AUC | Fake Recall | TNR | FPR | F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | LLM `[CLS]` | existing 8,194 | 0.986438 (historical) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
{chr(10).join(rows)}

Selected hyperparameters:

| Arm | Per-seed selected LR/epoch |
|---|---|
{chr(10).join(sel)}

## Attribution

| Contrast | Δ Accuracy | Δ ROC-AUC | Interpretation |
|---|---:|---:|---|
| C1-L − C1-S | {attribution['C1-L_minus_C1-S']['accuracy']:+.6f} | {attribution['C1-L_minus_C1-S']['roc_auc']:+.6f} | 同 CLIP feature；大 head 只有很小正增益，不是主要来源 |
| C1-S − C2 | {attribution['C1-S_minus_C2']['accuracy']:+.6f} | {attribution['C1-S_minus_C2']['roc_auc']:+.6f} | 近参数量下 CLIP global CLS 明显优于 F24 GAP |
| C1-L − C0 | {agg['C1-L']['accuracy']['mean']-0.9864376130198915:+.6f} | N/A | C1-L internal Accuracy 略高，但 C0 regime 不同，仅架构参考 |
| C1-S − C0 | {agg['C1-S']['accuracy']['mean']-0.9864376130198915:+.6f} | N/A | 一次错误量级的轻微降低，仍显示强 CLIP signal |
| C2 − C0 | {agg['C2']['accuracy']['mean']-0.9864376130198915:+.6f} | N/A | F24 不接近现有正式 classifier |

## Feature diagnostics and linear probes

| Feature | Normalized centroid L2 | Within cosine Real | Within cosine Fake | Between cosine | Within−between Real/Fake |
|---|---:|---:|---:|---:|---:|
| CLIP CLS | {d['clip_cls']['normalized_centroid_l2_distance']:.6f} | {d['clip_cls']['classes']['0']['within_class_cosine']:.6f} | {d['clip_cls']['classes']['1']['within_class_cosine']:.6f} | {d['clip_cls']['between_class_cosine']:.6f} | {d['clip_cls']['within_minus_between']['0']:.6f} / {d['clip_cls']['within_minus_between']['1']:.6f} |
| F24 GAP | {d['f24_gap']['normalized_centroid_l2_distance']:.6f} | {d['f24_gap']['classes']['0']['within_class_cosine']:.6f} | {d['f24_gap']['classes']['1']['within_class_cosine']:.6f} | {d['f24_gap']['between_class_cosine']:.6f} | {d['f24_gap']['within_minus_between']['0']:.6f} / {d['f24_gap']['within_minus_between']['1']:.6f} |

| Diagnostic probe | Acc | ROC-AUC | Interpretation |
|---|---:|---:|---|
| CLIP CLS → Linear | {pm(agg['LP-CLIP']['accuracy'])} | {pm(agg['LP-CLIP']['roc_auc'])} | 线性可分性很强；非线性大头不是 CLIP 有效的必要条件 |
| F24 GAP → Linear | {pm(agg['LP-F24']['accuracy'])} | {pm(agg['LP-F24']['roc_auc'])} | 明显高于随机，证明有独立 signal；但远弱于 CLIP |

Centroid/cosine 是 representation geometry diagnostics，不参与 checkpoint selector。

## Required decisions

1. **CLIP CLS 是否值得替代 current LLM classifier？** `GO_FOR_FREEZE_CONFIRMATION`：C1-L internal Accuracy/ROC-AUC 强且三 seed 稳定，但尚未做 external confirmation，不能宣布最终替代。
2. **C1-L 优势有多少由 head capacity 解释？** 很少：相对 C1-S 仅 ΔAcc `{attribution['C1-L_minus_C1-S']['accuracy']:+.6f}`、ΔAUC `{attribution['C1-L_minus_C1-S']['roc_auc']:+.6f}`；主要 signal 已存在于 CLIP CLS。
3. **F24 是否包含独立 Real/Fake signal？** 是；C2 AUC `{agg['C2']['roc_auc']['mean']:.6f}`，linear probe AUC `{agg['LP-F24']['roc_auc']['mean']:.6f}`。但作为正式 classifier 为 `NO-GO / WEAK`，因为明显低于 C0/C1。
4. **是否值得进入 C3 fusion？** **NO-GO at Phase6B exit**。C2 未达到 C0 附近，额外 fusion 容量会掩盖 attribution。
5. **下一步冻结哪个 candidate？** **C1-L**，作为后续用户授权下 external generalization confirmation 的唯一首选；C1-S 保留 capacity ablation，不替代主候选。

C2 和 LP-F24 的三个 seed 均在预注册 epoch 10 边界被选中，因此这里的 NO-GO 严格限定为“在本阶段冻结预算下不进入 C3/正式 classifier”。按 hyperparameter firewall，不因看到边界结果而单独延长 F24 训练。

## Firewall and STOP

`results.json` 明确记录：Internal test / Official1000 / external access 均为 false，threshold sweep=false，C3 trained=false。Phase 6B 到此 **STOP**，不自动进入 external classification evaluation。
"""
    DOC.write_text(report)
    print(json.dumps({'status':'COMPLETE','decisions':decisions,'document':str(DOC)},indent=2))

if __name__=='__main__': main()
