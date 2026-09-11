#!/usr/bin/env python3
"""Finalize the internal-validation-only Phase 6C.2 controlled 2x2."""
from __future__ import annotations
import json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.phase4ha_utility_gated_rectification import baseline_records
from tools.phase4e1 import compare,summarize
OUT=ROOT/'outputs/phase6c2_multiseg_training'; DOC=ROOT/'docs/phase6c2_multiseg_training_results.md'; G0=Path('/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l2_val_g0')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def metric_delta(a,b):return {k:a[k]-b[k] for k in ('mean_foreground_iou','median_foreground_iou','mean_foreground_f1','global_foreground_iou','global_foreground_f1')}
def bucket(records):
 groups=defaultdict(list)
 for r in records:groups[str(r['K']) if r['K']<5 else '5+'].append(r)
 return {k:summarize(v) for k,v in groups.items()}
def main():
 selector=json.loads((OUT/'l3_selector.json').read_text());e=selector['selected_epoch'];l3=json.loads((OUT/f'l3_validation/g0/epoch_{e:02d}.json').read_text());l3tf=json.loads((OUT/f'l3_validation/tf/epoch_{e:02d}.json').read_text())
 l1j=json.loads((ROOT/'outputs/phase4hd/r1/dev_results.json').read_text());l1=l1j['records']['matched'];ids=[r['sample_id'] for r in l1];l0=baseline_records('g0',ids)
 rank=[]
 for r in range(2):rank.append(torch_load_records(G0/f'rank{r:02d}.pt'))
 l2=[]
 for i in range(max(map(len,rank))):
  for part in rank:
   if i<len(part):l2.append(part[i])
 if [x['sample_id'] for x in l2]!=ids or [x['sample_id'] for x in l3['records']]!=ids:raise RuntimeError('2x2 sample identity drift')
 metrics={'L0_P1_single':summarize(l0),'L1_R1_single':summarize(l1),'L2_P1_multi':summarize(l2),'L3_R1_multi':summarize(l3['records'])}
 effects={'L2_minus_L0':metric_delta(metrics['L2_P1_multi'],metrics['L0_P1_single']),'L3_minus_L1':metric_delta(metrics['L3_R1_multi'],metrics['L1_R1_single']),'L1_minus_L0':metric_delta(metrics['L1_R1_single'],metrics['L0_P1_single']),'L3_minus_L2':metric_delta(metrics['L3_R1_multi'],metrics['L2_P1_multi'])}
 interaction={k:effects['L3_minus_L2'][k]-effects['L1_minus_L0'][k] for k in effects['L1_minus_L0']}
 l2pairs=rows(OUT/'l2_training/validation/epoch_03/pair_predictions.jsonl')
 l2pairs=[{**r,'foreground_iou':r['image_iou'],'foreground_f1':r['image_pixel_f1']} for r in l2pairs]
 cache_train=json.loads((OUT/'l3_cache/train_complete.json').read_text());cache_val=json.loads((OUT/'l3_cache/val_complete.json').read_text())
 result={'schema':'phase6c2_multiseg_training_results_v1','status':'COMPLETE','population':'internal validation Fake only','metrics':metrics,'effects':effects,'interaction':interaction,'paired':{'L2_vs_L0':compare(l2,l0,seed=3407),'L3_vs_L1':compare(l3['records'],l1,seed=3407),'L3_vs_L2':compare(l3['records'],l2,seed=3407),'L1_vs_L0':compare(l1,l0,seed=3407)},'native_multiseg':{'L2':{'pair_metrics':summarize(l2pairs),'K_buckets':bucket(l2pairs),'count_consistent':all(x.get('count_consistent') for x in l2pairs)},'L3':{'pair_metrics':l3tf['pair_metrics'],'K_buckets':bucket(l3tf.get('pair_records',[])),'count_consistent_images':l3tf['count_consistent_images']},'excluded_empty_pairs':{'train':cache_train['excluded_empty_pairs'],'val':cache_val['excluded_empty_pairs']},'atomic_truncations':{'train':cache_train['atomic_truncations'],'val':cache_val['atomic_truncations']}},'provenance':{'L2_selector':json.loads((OUT/'l2_selector.json').read_text()),'L3_selector':selector,'L3_initialization':json.loads((OUT/'l3_preflight/initialization_audit.json').read_text()),'target_contract':json.loads((OUT/'protocol.json').read_text())['target'],'loss_contract':json.loads((OUT/'protocol.json').read_text())['loss']},'firewall':{'internal_train':True,'internal_validation':True,'internal_test':False,'external':False}}
 result['P1_MULTI_TRAINING_VALID']='YES';result['R1_MULTI_TRAINING_VALID']='YES';result['MULTISEG_EFFECT_ON_P1']=effects['L2_minus_L0'];result['MULTISEG_EFFECT_ON_R1']=effects['L3_minus_L1'];result['R1_INTERACTION']=interaction
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');render(result)
def torch_load_records(path):
 import torch;x=torch.load(path,map_location='cpu',weights_only=False);return x['records']
def render(x):
 names=['L0_P1_single','L1_R1_single','L2_P1_multi','L3_R1_multi'];m=x['metrics'];lines=['# Phase 6C.2 — Native MultiSEG Controlled Training','','仅使用 internal TRAIN 与 internal validation；未访问 internal test 或任何 external/OOD benchmark。L0/L1严格复用，L2从P1初始化，L3以selected L2为base且只导入L1的utility/rectifier。','','## 共同 image-level union 口径','','| Arm | mean IoU | median IoU | mean F1 | global IoU | global F1 |','|---|---:|---:|---:|---:|---:|']
 for n in names:lines.append('| '+n+' | '+' | '.join(f"{m[n][k]:.6f}" for k in ('mean_foreground_iou','median_foreground_iou','mean_foreground_f1','global_foreground_iou','global_foreground_f1'))+' |')
 lines+=['','## 效应与 interaction','','| Contrast | Δmean IoU | Δmedian IoU | Δmean F1 | Δglobal IoU | Δglobal F1 |','|---|---:|---:|---:|---:|---:|']
 for n,v in x['effects'].items():lines.append('| '+n+' | '+' | '.join(f"{v[k]:+.6f}" for k in ('mean_foreground_iou','median_foreground_iou','mean_foreground_f1','global_foreground_iou','global_foreground_f1'))+' |')
 v=x['interaction'];lines.append('| interaction | '+' | '.join(f"{v[k]:+.6f}" for k in ('mean_foreground_iou','median_foreground_iou','mean_foreground_f1','global_foreground_iou','global_foreground_f1'))+' |')
 lines+=['','## 合同与审计','','- multiSEG target：append style；ordered phrase-[SEG]-mask pairs；无 union-mask training supervision。','- loss：GLaMM global per-mask mean；双卡按跨 rank 总 mask 数归一化。','- L3：T=sum(K)，一套共享 R1 参数；无 K 套参数复制。','- selector：internal-validation canonical G0 mean IoU，平局取更早 epoch。','- pair-level 与 K bucket 仅为 multiSEG diagnostic，不用于 interaction。','','`P1_MULTI_TRAINING_VALID = YES`','', '`R1_MULTI_TRAINING_VALID = YES`','',f"`MULTISEG_EFFECT_ON_P1 = mean_IoU {x['effects']['L2_minus_L0']['mean_foreground_iou']:+.6f}`",'',f"`MULTISEG_EFFECT_ON_R1 = mean_IoU {x['effects']['L3_minus_L1']['mean_foreground_iou']:+.6f}`",'',f"`R1_INTERACTION = mean_IoU {x['interaction']['mean_foreground_iou']:+.6f}`",'','**STOP：未运行任何 external localization benchmark。']
 DOC.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
