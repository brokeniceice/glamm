#!/usr/bin/env python3
"""Phase6K paired DEV/Official analysis, spatial diagnostics, and final report."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.phase6k0_c2_after_train import AUDIT, load_cache, require, dump
from scripts.phase4gf_formal_localization import load_dev
from tools.phase3c1 import transform_mask
from tools.phase4e1 import compare, summarize

DOC=ROOT/'docs/phase6k0_c2_after.md'


def global_paired(left,right,repeats=10000,seed=3407):
    require([x['sample_id'] for x in left]==[x['sample_id'] for x in right],'global pair ID drift')
    l=np.asarray([[r[k] for k in ('tp','fp','fn')] for r in left],dtype=np.float64)
    r=np.asarray([[x[k] for k in ('tp','fp','fn')] for x in right],dtype=np.float64)
    rng=np.random.default_rng(seed);out=[]
    for _ in range(repeats):
        index=rng.integers(0,len(l),size=len(l))
        a=l[index].sum(0);b=r[index].sum(0)
        out.append(a[0]/max(1.,a.sum())-b[0]/max(1.,b.sum()))
    total_l=l.sum(0);total_r=r.sum(0)
    return {'difference':float(total_l[0]/max(1.,total_l.sum())-total_r[0]/max(1.,total_r.sum())),
            'bootstrap_95_ci':[float(x) for x in np.quantile(out,[.025,.975])]}


def diagnostic(dev,cache):
    rows=[]
    for i,sid in enumerate(cache['sample_ids']):
        geometry=dev['clip_geometries'][i]
        gt=transform_mask(dev['original_masks'][i],geometry).float()[None,None]
        inside=F.adaptive_max_pool2d(gt,(24,24))[0,0].bool().flatten()
        a=cache['A'][i].float().reshape(8,576)
        e=cache['E'][i].float().reshape(512,576)
        m=e.square().sum(0).sqrt()
        area=float(inside.float().mean())
        per_head=[]
        for h in range(8):
            p=a[h];p=p/p.sum().clamp_min(1e-12)
            ent=float(-(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum()/math.log(576))
            mass=float(p[inside].sum())
            arg=int(p.argmax())
            auroc=float(roc_auc_score(inside.numpy(),p.numpy())) if inside.any() and (~inside).any() else None
            per_head.append({'normalized_entropy':ent,'max_mass':float(p.max()),
                'top10_mass':float(p.topk(10).values.sum()),'argmax_yx':[arg//24,arg%24],
                'gt_inside_mass':mass,'gt_area_fraction':area,
                'inside_over_outside_density':None if area in (0.,1.) else (mass/area)/max((1-mass)/(1-area),1e-12),
                'patch_auroc':auroc})
        p=m/m.sum().clamp_min(1e-12);mass=float(p[inside].sum())
        e_auc=float(roc_auc_score(inside.numpy(),p.numpy())) if inside.any() and (~inside).any() else None
        rows.append({'sample_id':sid,'valid_c2_g0':bool(cache['valid'][i]),'A_heads':per_head,
                     'E_magnitude':{'gt_inside_mass':mass,'gt_area_fraction':area,
                     'inside_over_outside_density':None if area in (0.,1.) else (mass/area)/max((1-mass)/(1-area),1e-12),
                     'patch_auroc':e_auc}})
    path=AUDIT/'diagnostics/dev_spatial.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def avg(values):return float(np.mean(values)) if values else None
    ah=[h for row in rows for h in row['A_heads']]
    em=[row['E_magnitude'] for row in rows]
    return {'n':len(rows),'A':{key:avg([x[key] for x in ah if x[key] is not None]) for key in
              ('normalized_entropy','max_mass','top10_mass','gt_inside_mass','inside_over_outside_density','patch_auroc')},
            'E':{key:avg([x[key] for x in em if x[key] is not None]) for key in
              ('gt_inside_mass','inside_over_outside_density','patch_auroc')},
            'geometry':'exact CLIP resize-shortest-336 and center-crop, then 24x24 max occupancy'}


def main():
    preflight=json.loads((AUDIT/'preflight.json').read_text());require(preflight['status']=='PASS','preflight missing')
    dev=load_dev('g0');cache=load_cache('val',dev['sample_ids'])
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'checkpoints/GLaMM-FullScope'),use_fast=False)
    tokenizer.add_tokens(['[CLS]','[REAL]','[FAKE]'],special_tokens=True)
    cache_status=json.loads((AUDIT/'cache/val.json').read_text())
    phrase_parse=0
    for item in cache_status['shards']:
        shard=torch.load(item['path'],map_location='cpu',weights_only=False)
        for record in shard['records']:
            generated=tokenizer.decode(record['generated_token_ids'],skip_special_tokens=False)
            tail=generated.split('Target regions:',1)[1].split('[SEG]',1)[0].strip() if 'Target regions:' in generated else ''
            phrase_parse+=bool(tail)
    b0=json.loads((AUDIT/'b0/dev.json').read_text())
    arms={name:json.loads((AUDIT/name/'selector.json').read_text()) for name in ('a','e')}
    selected={}
    for name,selector in arms.items():
        require(selector['status']=='COMPLETE' and not selector['official1000_used_for_selection'],f'{name} selector drift')
        selected[name]=json.loads((AUDIT/name/f"dev_epoch_{selector['selected_epoch']}.json").read_text())
    records={name:value['records'] for name,value in [('b0',b0),*selected.items()]}
    require(all([r['sample_id'] for r in rows]==dev['sample_ids'] for rows in records.values()),'DEV record ID drift')
    valid=[bool(x) for x in cache['valid']]
    require(all([bool(x['valid_g0']) for x in rows]==valid for rows in records.values()),'C2 query invariance drift')
    comparisons={}
    for label,l,r in [('a_minus_b0','a','b0'),('e_minus_b0','e','b0'),('e_minus_a','e','a')]:
        comparisons[label]={'per_image':compare(records[l],records[r]),
                            'global_fg_iou':global_paired(records[l],records[r])}
    diag=diagnostic(dev,cache)
    spatial_rows=[json.loads(line) for line in (AUDIT/'diagnostics/dev_spatial.jsonl').read_text().splitlines() if line]
    alignment_gain={}
    for name in ('a','e'):
        keep=[i for i,row in enumerate(spatial_rows) if row['valid_c2_g0']]
        align=[(sum(spatial_rows[i]['A_heads'][h]['gt_inside_mass'] for h in range(8))/8
                if name=='a' else spatial_rows[i]['E_magnitude']['gt_inside_mass']) for i in keep]
        gain=[records[name][i]['foreground_iou']-records['b0'][i]['foreground_iou'] for i in keep]
        coef,pvalue=spearmanr(align,gain)
        alignment_gain[name]={'spearman_rho':None if np.isnan(coef) else float(coef),
                              'pvalue':None if np.isnan(pvalue) else float(pvalue),'n':len(keep)}
    official=json.loads((AUDIT/'official1000/results.json').read_text())
    require(official['status']=='COMPLETE' and official['population']==1000,'Official1000 incomplete')
    packed=[json.loads(x) for x in (AUDIT/'official1000/packed_records.jsonl').read_text().splitlines() if x]
    official_records={name:[x['metrics'][name] for x in packed] for name in ('b0','a','e')}
    official_global={label:global_paired(official_records[l],official_records[r]) for label,l,r in
                     [('a_minus_b0','a','b0'),('e_minus_b0','e','b0'),('e_minus_a','e','a')]}
    metrics={name:{**summarize(rows),**{key:sum(int(r[key]) for r in rows)
              for key in ('tp','fp','fn','tn')}} for name,rows in records.items()}
    def stable(name):
        key=name+'_minus_b0';d=comparisons[key];o=official['paired'][key]
        return (metrics[name]['mean_foreground_iou']>metrics['b0']['mean_foreground_iou'] and
                metrics[name]['global_foreground_iou']>metrics['b0']['global_foreground_iou'] and
                d['per_image']['foreground_iou']['bootstrap_95_ci'][0]>0 and
                official['metrics'][name]['mean_foreground_iou']>official['metrics']['b0']['mean_foreground_iou'] and
                official['metrics'][name]['global_foreground_iou']>official['metrics']['b0']['global_foreground_iou'] and
                o['foreground_iou']['bootstrap_95_ci'][0]>0)
    verdict='FROZEN C2 SPATIAL TRANSFER SUPPORTED' if stable('a') or stable('e') else 'FROZEN C2 SPATIAL TRANSFER NOT SUPPORTED'
    result={'status':'COMPLETE','schema':'phase6k0_c2_after_v1','verdict':verdict,
            'dev_metrics':metrics,'dev_paired':comparisons,'official1000':official,
            'official_global_paired':official_global,'spatial_diagnostics':diag,
            'alignment_vs_localization_gain':alignment_gain,
            'train_valid':preflight['train_valid'],'dev_valid':preflight['dev_valid'],
            'dev_phrase_parse_success_count':phrase_parse,
            'C2_query_invariant':True,'preflight':preflight}
    dump(AUDIT/'results.json',result)
    def f(v):return f'{v:.6f}'
    lines=['# Phase6K — Frozen C2 Spatial Transfer: After Injection','',
       '本阶段只训练 A/E 各自的零初始化 1×1 投影。C2、Phase4C-A、C1-native R1 与 SAM 权重保持冻结。',
       '', '## 五项预检','',
       f"- C2 selected checkpoint SHA256: `{official['protocol']['c2_sha256']}`；epoch 7 / step 3500。",
       f"- C1-native joint R1 SHA256: `{official['protocol']['native_sha256']}`；内部 DEV 选择，未用 Official1000/OOD。",
       f"- A `8×24×24`；E `512×24×24`；TRAIN {preflight['train_n']} 张，C2 valid {preflight['train_valid']} 张；DEV {preflight['dev_n']} 张，C2 valid {preflight['dev_valid']} 张。",
       f"- DEV `[SEG]` valid rate {preflight['dev_valid']}/{preflight['dev_n']}；phrase parse success {phrase_parse}/{preflight['dev_n']}。Official1000 `[SEG]` exactly-one {official['exactly_one_seg_count']}/1000；phrase parse success {official['phrase_parse_success_count']}/1000。",
       '- C2 捕获开启前后生成 token、q_seg、分类概率和掩码完全一致；A head 质量和及 E context 归约审计见 `outputs/phase6k0_c2_after/cache/capture_preflight.json`。',
       '- A/E step-0 与 B0 的 F24、z_F24、Rectifier、Utility、SAM embedding、mask logits 和 loss 全部相等；投影梯度非零且冻结参数哈希未变。',
       '', '## Q1–Q3：内部 DEV canonical G0','',
       '| 模型 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |','|---|---:|---:|---:|---:|---:|']
    for name in ('b0','a','e'):
        m=metrics[name];lines.append(f"| {name.upper()} | {m['n']} | {f(m['mean_foreground_iou'])} | {f(m['mean_foreground_f1'])} | {f(m['global_foreground_iou'])} | {f(m['global_foreground_f1'])} |")
    lines+=['',f"- A selected epoch {arms['a']['selected_epoch']}，checkpoint SHA256 `{arms['a']['selected_checkpoint_sha256']}`。",
            f"- E selected epoch {arms['e']['selected_epoch']}，checkpoint SHA256 `{arms['e']['selected_checkpoint_sha256']}`。"]
    lines+=['','差值为前者减后者；逐图配对 bootstrap 10,000 次，固定 seed 3407。','',
        '| 比较 | Mean IoU Δ | 95% CI | W/T/L | Wilcoxon p | Global IoU Δ | Global 95% CI |','|---|---:|---|---|---:|---:|---|']
    for key in ('a_minus_b0','e_minus_b0','e_minus_a'):
        p=comparisons[key]['per_image']['foreground_iou'];g=comparisons[key]['global_fg_iou']
        lines.append(f"| {key} | {f(p['mean_difference'])} | [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.3g} | {f(g['difference'])} | [{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}] |")
    lines+=['','选定 epoch 的残差与训练日志：','',
            '| Arm | Epoch | ρ 中位数 | ρ P95 | ρ 最大值 | cos(F,F′) 均值 | 梯度裁剪比例 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('a','e'):
        row=next(x for x in arms[name]['candidates'] if x['epoch']==arms[name]['selected_epoch'])
        lines.append(f"| {name.upper()} | {row['epoch']} | {f(row['rho_median'])} | {f(row['rho_p95'])} | {f(row['rho_max'])} | {f(row['cosine_mean'])} | {f(row['gradient_clip_rate'])} |")
    lines+=['','## Official1000：历史 matched canonical control','',
        'Official1000 仅在 A/E 内部 DEV epoch 冻结后评测；本数据集曾参与历史模型比较，因此不称 untouched final test。',
        '', '| 模型 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |','|---|---:|---:|---:|---:|---:|']
    for name in ('b0','a','e'):
        m=official['metrics'][name]
        lines.append(f"| {name.upper()} | {m['n']} | {f(m['mean_foreground_iou'])} | {f(m['mean_foreground_f1'])} | {f(m['global_foreground_iou'])} | {f(m['global_foreground_f1'])} |")
    lines+=['','| Official 比较 | Mean IoU Δ | 95% CI | W/T/L | Wilcoxon p | Global IoU Δ | Global 95% CI |',
            '|---|---:|---|---|---:|---:|---|']
    for key in ('a_minus_b0','e_minus_b0','e_minus_a'):
        p=official['paired'][key]['foreground_iou'];g=official_global[key]
        lines.append(f"| {key} | {f(p['mean_difference'])} | [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.3g} | {f(g['difference'])} | [{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}] |")
    lines+=['','## Q4–Q6：解释与结论','',
       f"- 空间诊断：A 平均 patch AUROC {diag['A']['patch_auroc']}; E magnitude 平均 patch AUROC {diag['E']['patch_auroc']}。GT 经过 CLIP 336 中心裁剪几何再映射到 24×24；这些只是相关性诊断。",
       f"- GT-inside mass 与逐图 IoU 增益的 Spearman 相关：A {alignment_gain['a']['spearman_rho']}，E {alignment_gain['e']['spearman_rho']}；仅描述相关性。",
       f'- 阶段判断：**{verdict}**。A/E 参数量不同，比较的是信息接口，不能声称严格容量匹配。',
       ('- E 相对 A 的 DEV Mean/Global 均更高，且逐图配对 95% CI 下界大于 0；这支持 E 所携带的加权空间特征比单独的 A 关联更多定位效用，但不证明因果。'
        if metrics['e']['mean_foreground_iou']>metrics['a']['mean_foreground_iou'] and
           metrics['e']['global_foreground_iou']>metrics['a']['global_foreground_iou'] and
           comparisons['e_minus_a']['per_image']['foreground_iou']['bootstrap_95_ci'][0]>0
        else '- E 相对 A 未形成 DEV Mean/Global 与逐图配对 CI 同时支持的稳定优势；不能据此声称 where + what 比 where-only 更有用。'),
       '- 注意力图不是因果 artifact map；逐图空间诊断见 `outputs/phase6k0_c2_after/diagnostics/dev_spatial.jsonl`。',
       ('- 冻结迁移在 DEV 与历史 Official1000 都有稳定正信号，可以另行讨论 Middle injection 或 joint fine-tuning；本阶段不自动启动。'
        if verdict.endswith('SUPPORTED') and 'NOT' not in verdict else
        '- 冻结迁移未形成稳定正信号，目前不建议授权 Middle injection 或 joint fine-tuning；本阶段到此停止。'),
       '', '证据：`outputs/phase6k0_c2_after/results.json`、逐图 DEV/Official1000 记录、每轮 checkpoint 与 selector。','']
    DOC.write_text('\n'.join(lines))

if __name__=='__main__':main()
