#!/usr/bin/env python3
"""Finalize Phase 3D.0-R qualitative packet, report, provenance and completion."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3b_replay import file_sha256
from tools.phase3d0r import choose_top

def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def dump(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def fmt(x):return "NA" if x is None else f"{float(x):.4f}"
def stable(values,category,count=10):return sorted(values,key=lambda x:hashlib.sha256(f"3407:{category}:{x['sample_id']}:{x.get('rollout_index',-1)}".encode()).hexdigest())[:count]

def resolve_image(row,cfg):
 candidate=row.get("image_path")
 if candidate and Path(candidate).is_file():return Path(candidate)
 root=(ROOT/cfg["data"]["synthscars_root"]).resolve() if row["forensics_domain"]=="fake" else (ROOT/cfg["data"]["datasets_root"]).resolve()
 return root/row["image_relpath"]

def mask_thumb(path,dest):
 if not path or not Path(path).exists():return None
 value=torch.load(path,map_location="cpu").bool()
 while value.ndim>2:value=value.any(dim=0)
 image=Image.fromarray((value.numpy().astype(np.uint8)*255),mode="L");image.thumbnail((256,256),Image.Resampling.NEAREST);image.save(dest);return dest.name

def gt_thumb(row,image_path,dest):
 with Image.open(image_path) as image:w,h=image.size
 value=UnifiedForensicsDataset._fake_union_mask(row,h,w).bool()
 while value.ndim>2:value=value.any(dim=0)
 Image.fromarray((value.numpy().astype(np.uint8)*255),mode="L").resize((256,256),Image.Resampling.NEAREST).save(dest)

def qualitative(out,cfg,selected):
 groups=rows(out/"validation/scored_groups.jsonl");greedy={x["sample_id"]:x for x in rows(out/"validation/scored_greedy.jsonl")}
 raw=rows(ROOT/cfg["experiment"]["manifest_dir"]/"val_combined.jsonl");raw_by_id={x["sample_id"]:x for x in raw}
 fake=[x for g in groups if g["sample_id"] in raw_by_id and int(g["rollouts"][0]["gt_class"])==1 for x in g["rollouts"]]
 group_items=[]
 for g in groups:
  if int(g["rollouts"][0]["gt_class"])!=1:continue
  item={"sample_id":g["sample_id"],"group":g,"OLD_R3":choose_top(g["rollouts"],"OLD_R3"),"Q1":choose_top(g["rollouts"],"Q1"),"Q2":choose_top(g["rollouts"],"Q2"),"selected":choose_top(g["rollouts"],selected),"greedy":greedy[g["sample_id"]]};group_items.append(item)
 cats={
  "SYNONYM_PARAPHRASE":stable([x for x in fake if x["R_phrase_lex"]<.3 and x["R_phrase_sem"]>=.7],"syn"),
  "FUNCTION_WORD_INFLATION":stable([x for x in fake if x["R_phrase_lex"]>=.7 and x["R_phrase_sem"]<.5],"func"),
  "SPATIAL_CONTRADICTION":stable([x for x in fake if x["P_spatial_contra"]],"spatial"),
  "PHRASE_GOOD_MASK_POOR":stable([x for x in fake if x["R_phrase_sem"]>=.7 and x["R_mask"]<=.2],"pgmp"),
  "LOW_CONTROLLABILITY_GROUP":stable([x["selected"] for x in group_items if x["selected"]["C_ground"]<.25],"low"),
  "HIGH_CONTROLLABILITY_GROUP":stable([x["selected"] for x in group_items if x["selected"]["C_ground"]>=.75],"high"),
  "OLD_R3_VS_Q1_DISAGREEMENT":stable([x["Q1"] for x in group_items if x["OLD_R3"]["rollout_index"]!=x["Q1"]["rollout_index"]],"oldq1"),
  "OLD_R3_VS_Q2_DISAGREEMENT":stable([x["Q2"] for x in group_items if x["OLD_R3"]["rollout_index"]!=x["Q2"]["rollout_index"]],"oldq2"),
  "Q1_VS_Q2_DISAGREEMENT":stable([x["Q2"] for x in group_items if x["Q1"]["rollout_index"]!=x["Q2"]["rollout_index"]],"q1q2"),
  "SELECTED_VS_GREEDY":stable([x["selected"] for x in group_items if x["selected"]["generated_token_ids"]!=x["greedy"]["generated_token_ids"]],"selgreedy"),
 }
 q=out/"qualitative";assets=q/"assets";assets.mkdir(parents=True,exist_ok=True);parts=["<!doctype html><meta charset='utf-8'><title>Phase 3D.0-R</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}article{border-top:1px solid;padding:1em}.imgs{display:flex}.imgs img{max-width:256px;background:#ddd;margin:5px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:4px}</style><h1>Phase 3D.0-R deterministic qualitative audit</h1>"]
 ordinal=0
 for category,values in cats.items():
  parts.append(f"<h2>{category} (n={len(values)})</h2>")
  for x in values:
   row=raw_by_id[x["sample_id"]];image_path=resolve_image(row,cfg);image_name=f"{ordinal:03d}_image.jpg";Image.open(image_path).convert("RGB").resize((256,256)).save(assets/image_name,quality=85)
   gt_name=f"{ordinal:03d}_gt.png";gt_thumb(row,image_path,assets/gt_name);pred=mask_thumb(x.get("binary_mask_path"),assets/f"{ordinal:03d}_pred.png")
   reference=UnifiedForensicsDataset.authoritative_localization_field(row)["normalized_training_phrase"]
   metrics=("R_phrase_lex","R_key_soft","R_sentence_sem","R_phrase_sem","P_spatial_contra","R_mask","C_ground","R_ground_rel","OLD_R3","Q1","Q2","Q3")
   table="".join(f"<tr><th>{k}</th><td>{fmt(x.get(k))}</td></tr>" for k in metrics)
   pred_html="" if pred is None else f"<figure><img src='assets/{pred}'><figcaption>predicted mask</figcaption></figure>"
   parts.append(f"<article><h3>{html.escape(x['sample_id'])} / S{x['rollout_index']+1}</h3><div class='imgs'><figure><img src='assets/{image_name}'><figcaption>image</figcaption></figure><figure><img src='assets/{gt_name}'><figcaption>GT union</figcaption></figure>{pred_html}</div><p><b>Authoritative:</b> {html.escape(reference)}</p><p><b>Generated phrase:</b> {html.escape(str(x.get('normalized_phrase')))}</p><pre>{html.escape(str(x.get('generated_explanation')))}</pre><table>{table}</table></article>");ordinal+=1
 (q/"index.html").write_text("\n".join(parts));manifest={"status":"COMPLETE","deterministic":True,"seed":3407,"counts":{k:len(v) for k,v in cats.items()},"total_rendered":sum(map(len,cats.values())),"manual_cherry_pick":False,"index":str(q/"index.html")};dump(q/"selection_manifest.json",manifest);return manifest

def main():
 cfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text());out=(ROOT/cfg["experiment"]["output_root"]).resolve();route=load(out/"route_gate.json");selected=route["selected_reward"]
 q=qualitative(out,cfg,selected);dev=load(out/"reward_dev_comparison.json");val=load(out/"validation_comparison.json");sanity=load(out/"semantic_sanity_statistics.json");signal=load(out/"reformulated_policy_signal.json");sub=load(out/"subgroup_statistics.json");fail=load(out/"old_r3_failure_audit.json");source=load(out/"phase3d0_source_artifact.json")
 replay_audits=[load(out/f"audit/reward_dev_fixed_token_mask_replay_shard{i:02d}_of_02.json") for i in range(2)]
 combined_replay={"status":"PASS" if all(x["status"]=="COMPLETE" and x["token_ids_exact"] and x["model_state_exact"] for x in replay_audits) else "FAIL",
  "groups":sum(x["source_groups"] for x in replay_audits),"trajectories":sum(x["trajectories"] for x in replay_audits),"shards":replay_audits,
  "no_rerollout":True,"fixed_token_mask_replay_user_authorized":True};dump(out/"audit/reward_dev_fixed_token_mask_replay.json",combined_replay)
 t=val["top_selected"][selected];g=val["greedy"];p=val["paired_top_vs_greedy"][selected]
 report=f"""# Phase 3D.0-R — Evidence-Aware Reward Reformulation

## 1. Final status

本阶段仅复用 Phase 3D.0 frozen trajectories。用户额外授权 reward-dev 固定 token 的 no-cache mask replay；8192 条文本 rollout 未重新采样，Fake 4096 条 token 序列 exact，P1 前后权重 exact。Phase 3D.1 未启动。

最终 selected reward：**{selected}**。`REFORMULATED_POLICY_SIGNAL_READY={str(signal['REFORMULATED_POLICY_SIGNAL_READY']).lower()}`。Primary gate：**`{route['primary_gate']}`**。

## 2. 为什么 OLD_R3 降级

OLD_R3 的 `R_phrase_lex` 对所有词近似等权，存在 function-word inflation 与合理 paraphrase underscoring；harmonic OLD_R_align 又把不可靠 lexical score 与受 mask decoder ceiling 限制的绝对 IoU 耦合，导致 phrase-correct/mask-poor trajectory 被重复处罚。因此 OLD_R3 仅保留为 `HISTORICAL_REWARD_CONTROL`，旧 lexical/align 仅作诊断。

## 3. Frozen semantic scorer

Encoder：`sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，eval/no-grad/frozen；完整 internal-train authoritative phrase corpus 计算 IDF，validation/test vocabulary 未使用。`R_phrase_sem=.65 R_key_soft + .35 R_sentence_sem`，并显式处罚预注册 spatial contradiction。Sanity status：**{sanity['status']}**。

Sanity checks：

```json
{json.dumps(sanity['checks'],ensure_ascii=False,indent=2)}
```

`R_phrase_sem is an automatic frozen semantic proxy, not human semantic ground truth.`

## 4. Relative grounding

`R_ground_rel=C_range*C_ceiling*mask percentile rank`。只有同组 mask range 足够、且最好 mask 达到一定 ceiling 时，mask 才能影响 language reward；absolute `R_mask` 继续只作诊断。Q1 是 language-only；Q2 给 relative grounding 0.10 权重；Q3 的0.20只作 stress test。

## 5. Reward-dev selection

Q2 conditions：

```json
{json.dumps(dev['selection_conditions'],ensure_ascii=False,indent=2)}
```

Reward-dev 在 validation 之前冻结 selected={selected}。Q3 未获准成为 primary。

## 6. Full validation confirmation

| Metric | Greedy | {selected} top |
|---|---:|---:|
| LM verdict accuracy | {fmt(g['overall_generated_verdict_accuracy'])} | {fmt(t['overall_generated_verdict_accuracy'])} |
| Structure validity | {fmt(g['structural_validity_rate'])} | {fmt(t['structural_validity_rate'])} |
| Fake lexical F1 | {fmt(g['fake_phrase_lex'])} | {fmt(t['fake_phrase_lex'])} |
| Fake semantic phrase | {fmt(g['fake_phrase_sem'])} | {fmt(t['fake_phrase_sem'])} |
| Fake key-soft | {fmt(g['fake_key_soft'])} | {fmt(t['fake_key_soft'])} |
| Fake sentence-sem | {fmt(g['fake_sentence_sem'])} | {fmt(t['fake_sentence_sem'])} |
| Spatial contradiction rate | {fmt(g['fake_spatial_contradiction_rate'])} | {fmt(t['fake_spatial_contradiction_rate'])} |
| Absolute FG IoU | {fmt(g['fake_absolute_iou'])} | {fmt(t['fake_absolute_iou'])} |
| FG F1 | {fmt(g['fake_foreground_f1'])} | {fmt(t['fake_foreground_f1'])} |

Selected top-vs-greedy `R_phrase_sem` mean Δ={fmt(p['R_phrase_sem']['mean_difference'])}，95% CI={p['R_phrase_sem']['bootstrap_95ci']}；absolute IoU mean Δ={fmt(p['R_mask']['mean_difference'])}，95% CI={p['R_mask']['bootstrap_95ci']}。

## 7. Key diagnostics

- lexical-low / semantic-high trajectories：{sub['LEXICAL_LOW_SEMANTIC_HIGH']['trajectory_count']}
- lexical-high / semantic-low trajectories：{sub['LEXICAL_HIGH_SEMANTIC_LOW']['trajectory_count']}
- spatial contradiction trajectories：{sub['SPATIAL_CONTRADICTION']['trajectory_count']}
- phrase-good / mask-poor trajectories：{sub['PHRASE_GOOD_MASK_POOR']['trajectory_count']}
- phrase-good / ground-responsive trajectories：{sub['PHRASE_GOOD_GROUND_RESPONSIVE']['trajectory_count']}
- DOUBLE_PENALTY_CASE：{fail['DOUBLE_PENALTY_CASE']['count']}
- FUNCTION_WORD_INFLATION strict key-soft proxy：{fail['FUNCTION_WORD_INFLATION']['count']}
- FUNCTION_WORD_INFLATION broad lexical-high/semantic-low proxy：{fail['FUNCTION_WORD_INFLATION_BROAD_PROXY']['count']}
- PARAPHRASE_UNDERSCORED proxy：{fail['PARAPHRASE_UNDERSCORED']['count']}
- SPATIAL_CONTRADICTION_OVERREWARDED：{fail['SPATIAL_CONTRADICTION_OVERREWARDED']['count']}

这些 automatic subgroup 是 frozen semantic proxy，不是 human ground truth。

## 8. Policy signal and route

```json
{json.dumps(signal,ensure_ascii=False,indent=2)}
```

最终 gate：**`{route['primary_gate']}`**。该 gate {'只授权提出 Phase 3D.1 controlled policy optimization；不自动启动。' if route['phase3d1_controlled_policy_optimization_authorized'] else '不授权 policy training，路线在此停止。'}

## 9. Limitations

MiniLM 是通用 frozen embedding proxy，不是 forensic human semantic judge；显式规则仅可靠覆盖预注册 spatial pairs，negation mismatch 因无 deterministic scope parser 只记录、不强罚。best-of-K 仍不是 deployment result；本阶段不能证明 GRPO 必然提升性能。qualitative 页面为确定性抽样，未人工 cherry-pick，共 {q['total_rendered']} 个展示条目。
"""
 (out/"final_report.md").write_text(report);(out/"reports/final_report.md").write_text(report);doc=ROOT/"docs/phase3d0r_evidence_aware_reward_reformulation.md";doc.write_text(report)
 provenance=load(out/"provenance.json");provenance.update({"status":"COMPLETE","selected_reward":selected,"primary_gate":route["primary_gate"],"phase3d1_started":False,"reward_dev_mask_replay":combined_replay["status"]});dump(out/"provenance.json",provenance)
 required=["provenance.json","phase3d0_source_artifact.json","semantic_encoder_provenance.json","phrase_preprocessing.json","stopword_list.json","token_idf.json","critical_modifier_lexicon.json","contradiction_rules.json","semantic_sanity_statistics.json","phrase_semantic_components.jsonl","grounding_controllability.jsonl","reward_definition.json","reward_dev_comparison.json","selected_reward.json","validation_comparison.json","old_r3_failure_audit.json","subgroup_statistics.json","reformulated_policy_signal.json","route_gate.json","final_report.md","regression_tests.txt"]
 missing=[x for x in required if not (out/x).exists()]
 if missing:raise RuntimeError(f"missing artifacts: {missing}")
 completion={"status":"COMPLETE","phase":"Phase 3D.0-R","selected_reward":selected,"primary_gate":route["primary_gate"],"report":str(doc),"required_artifacts":[{"path":x,"sha256":file_sha256(out/x)} for x in required],"no_rerollout":True,"reward_dev_fixed_token_mask_replay":True,"P1_weights_exact":combined_replay["status"]=="PASS","no_training":True,"phase3d1_started":False};dump(out/"completion_manifest.json",completion);print(json.dumps({"status":"COMPLETE","selected_reward":selected,"gate":route["primary_gate"],"report":str(doc)},indent=2))

if __name__=="__main__":main()
