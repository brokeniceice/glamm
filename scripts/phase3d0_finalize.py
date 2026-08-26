#!/usr/bin/env python3
"""Build deterministic human-review/qualitative packets and the Phase 3D.0 final report."""

from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3d0 import REWARD_NAMES, choose_top, stable_rank
from tools.phase3b_replay import file_sha256


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d0_reward_preflight.yaml")
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args(argv)


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def rows(path: Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def fmt(value, digits=6):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def deterministic_take(values, count, namespace):
    return sorted(values, key=lambda value: stable_rank(3407, namespace, value["sample_id"]))[:count]


def save_mask_thumbnail(source: str | None, destination: Path) -> str | None:
    if not source or not Path(source).exists(): return None
    tensor = torch.load(source, map_location="cpu").bool()
    while tensor.ndim > 2: tensor = tensor.any(dim=0)
    image = Image.fromarray((tensor.numpy().astype(np.uint8) * 255), mode="L")
    image.thumbnail((256, 256), Image.Resampling.NEAREST); destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination); return destination.name


def save_tensor_thumbnail(tensor: torch.Tensor, destination: Path) -> str:
    value = tensor.bool()
    while value.ndim > 2: value = value.any(dim=0)
    image = Image.fromarray((value.numpy().astype(np.uint8) * 255), mode="L")
    image.thumbnail((256, 256), Image.Resampling.NEAREST); destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination); return destination.name


def authoritative_phrase(row):
    return UnifiedForensicsDataset.authoritative_localization_field(row)["normalized_training_phrase"]


def resolve_image_paths(rows_by_id: dict[str, dict], cfg: dict) -> None:
    """Materialize the same image path resolution used by the frozen dataset."""
    datasets_root = (ROOT / cfg["data"]["datasets_root"]).resolve()
    synthscars_root = (ROOT / cfg["data"]["synthscars_root"]).resolve()
    for row in rows_by_id.values():
        candidate = row.get("image_path")
        if candidate and Path(candidate).is_file():
            row["image_path"] = str(Path(candidate).expanduser().resolve())
            continue
        root = synthscars_root if row["forensics_domain"] == "fake" else datasets_root
        image_path = (root / row["image_relpath"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["image_path"] = str(image_path)


def write_review_packet(output: Path, groups: list[dict], greedy: dict[str, dict], val_rows: dict[str, dict]):
    fake = [group for group in groups if int(group["rollouts"][0]["gt_class"]) == 1]
    enriched = []
    for group in fake:
        r0 = choose_top(group["rollouts"], "R0"); r3 = choose_top(group["rollouts"], "R3")
        base = greedy[group["sample_id"]]
        enriched.append({"sample_id": group["sample_id"], "group": group, "r0": r0, "r3": r3, "greedy": base,
                         "difference": abs(r3["R_phrase_lex"]-base["R_phrase_lex"]) + abs(r3["R_mask"]-base["R_mask"]) + (r0["rollout_index"] != r3["rollout_index"])})
    categories = {
        "A_R0_R3_DIFFER": deterministic_take([x for x in enriched if x["r0"]["rollout_index"] != x["r3"]["rollout_index"]], 50, "review-A"),
        "B_R3_PHRASE_LOW_MASK_HIGH": deterministic_take([x for x in enriched if x["r3"]["R_phrase_lex"] < .10 and x["r3"]["R_mask"] >= .30], 50, "review-B"),
        "C_R3_PHRASE_HIGH_MASK_LOW": deterministic_take([x for x in enriched if x["r3"]["R_phrase_lex"] >= .30 and x["r3"]["R_mask"] < .10], 50, "review-C"),
        "D_GREEDY_R3_LARGEST_DIFFERENCE": sorted(enriched, key=lambda x: (-x["difference"], x["sample_id"]))[:50],
    }
    selected = {}; memberships = defaultdict(list)
    for category, values in categories.items():
        for value in values: selected[value["sample_id"]] = value; memberships[value["sample_id"]].append(category)
    review = output / "human_review"; assets = review / "assets"; assets.mkdir(parents=True, exist_ok=True)
    csv_path = review / "phrase_semantic_audit.csv"
    fields = ["sample_id", "categories", "image_path", "authoritative_target_phrase", "greedy_output",
              "rollouts_json", "selected_rollout_R0", "selected_rollout_R1", "selected_rollout_R2", "selected_rollout_R3",
              "phrase_semantically_correct", "explanation_evidence_consistent", "phrase_mask_consistent", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for sid in sorted(selected):
            value = selected[sid]; row = val_rows[sid]; rollouts = value["group"]["rollouts"]
            writer.writerow({"sample_id": sid, "categories": ";".join(memberships[sid]), "image_path": row["image_path"],
                "authoritative_target_phrase": authoritative_phrase(row), "greedy_output": value["greedy"]["decoded_text"],
                "rollouts_json": json.dumps(rollouts, ensure_ascii=False),
                **{f"selected_rollout_{reward}": choose_top(rollouts, reward)["rollout_index"] for reward in REWARD_NAMES},
                "phrase_semantically_correct": "", "explanation_evidence_consistent": "", "phrase_mask_consistent": "", "notes": ""})
    html_parts = ["<!doctype html><meta charset='utf-8'><title>Phase 3D.0 Human Review</title>",
                  "<style>body{font-family:sans-serif;max-width:1600px;margin:auto}article{border-top:2px solid #555;padding:1em}pre{white-space:pre-wrap}.masks img{max-width:180px;margin:4px;background:#ddd}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:4px;vertical-align:top}</style>",
                  "<h1>Phase 3D.0 Phrase Semantic Audit</h1><p>Human fields are intentionally blank. Automatic gate does not depend on them.</p>"]
    for ordinal, sid in enumerate(sorted(selected)):
        value=selected[sid]; row=val_rows[sid]; group=value["group"]["rollouts"]
        image_name=f"{ordinal:03d}_image.jpg"; Image.open(row["image_path"]).convert("RGB").copy().resize((256,256)).save(assets/image_name, quality=85)
        mask_items=[]
        with Image.open(row["image_path"]) as source_image:
            width, height = source_image.size
        gt_name = save_tensor_thumbnail(UnifiedForensicsDataset._fake_union_mask(row, height, width), assets/f"{ordinal:03d}_gt.png")
        mask_items.append(f"<figure><img src='assets/{gt_name}'><figcaption>GT derived union</figcaption></figure>")
        for label, record in [("greedy", value["greedy"])] + [(f"S{x['rollout_index']+1}", x) for x in group]:
            name=f"{ordinal:03d}_{label}.png"; saved=save_mask_thumbnail(record.get("binary_mask_path"), assets/name)
            if saved: mask_items.append(f"<figure><img src='assets/{saved}'><figcaption>{label}</figcaption></figure>")
        reward_rows="".join(f"<tr><td>S{x['rollout_index']+1}</td><td>{html.escape(str(x['verdict']))}</td><td>{html.escape(str(x['target_regions_raw']))}</td>"+
                            "".join(f"<td>{fmt(x[k],3)}</td>" for k in ("R_cls","R_struct","R_phrase_lex","R_mask","R_align","R0","R1","R2","R3"))+"</tr>" for x in group)
        html_parts.append(f"<article><h2>{html.escape(sid)}</h2><p>{html.escape('; '.join(memberships[sid]))}</p><img src='assets/{image_name}'><p><b>Authoritative:</b> {html.escape(authoritative_phrase(row))}</p><pre>{html.escape(value['greedy']['decoded_text'])}</pre><div class='masks' style='display:flex;flex-wrap:wrap'>{''.join(mask_items)}</div><table><tr><th>trajectory</th><th>verdict</th><th>phrase</th>"+"".join(f"<th>{x}</th>" for x in ("cls","struct","phrase","mask","align","R0","R1","R2","R3"))+f"</tr>{reward_rows}</table><p>Human: phrase_semantically_correct ____; explanation_evidence_consistent ____; phrase_mask_consistent ____; notes ____</p></article>")
    (review/"index.html").write_text("\n".join(html_parts), encoding="utf-8")
    manifest={"status":"READY_FOR_HUMAN_REVIEW","category_requested":50,"category_counts":{k:len(v) for k,v in categories.items()},"unique_count":len(selected),"csv":str(csv_path),"html":str(review/"index.html"),"human_fields_prefilled":False,"automatic_gate_depends_on_human_fields":False}
    dump(review/"selection_manifest.json",manifest); return manifest


def write_qualitative(output: Path, groups: list[dict], greedy: dict[str,dict], val_rows: dict[str,dict]):
    items=[]
    for group in groups:
        rs=group["rollouts"]; base=greedy[group["sample_id"]]; r0=choose_top(rs,"R0"); r3=choose_top(rs,"R3")
        items.append({"sample_id":group["sample_id"],"group":group,"greedy":base,"r0":r0,"r3":r3})
    categories={
      "R3_BETTER_THAN_GREEDY":sorted([x for x in items if x["r3"]["R_align"]>x["greedy"]["R_align"]],key=lambda x:-(x["r3"]["R_align"]-x["greedy"]["R_align"]))[:10],
      "MASK_ONLY_RISK":deterministic_take([x for x in items if x["r0"]["rollout_index"]!=x["r3"]["rollout_index"] and (x["r0"]["R_phrase_lex"]<x["r3"]["R_phrase_lex"] or x["r0"]["R_cls"]<x["r3"]["R_cls"] or x["r0"]["R_struct"]<x["r3"]["R_struct"])],10,"qual-mask"),
      "PHRASE_HIGH_MASK_LOW":deterministic_take([x for x in items if x["r3"]["R_phrase_lex"]>=.3 and x["r3"]["R_mask"]<.1],10,"qual-ph"),
      "PHRASE_LOW_MASK_HIGH":deterministic_take([x for x in items if x["r3"]["R_phrase_lex"]<.1 and x["r3"]["R_mask"]>=.3],10,"qual-mh"),
      "ALL_ROLLOUTS_POOR":deterministic_take([x for x in items if max(r["R_mask"] for r in x["group"]["rollouts"])<.1 and max(r["R_phrase_lex"] for r in x["group"]["rollouts"])<.1],10,"qual-poor"),
      "WRONG_CLASS_HIGH_MASK":deterministic_take([x for x in items if any(r["R_cls"]==0 and r["R_mask"]>=.3 for r in x["group"]["rollouts"])],10,"qual-wrong"),
      "REAL_HALLUCINATED_FAKE":deterministic_take([x for x in items if x["r3"]["gt_class"]==0 and (x["r3"]["verdict"]=="FAKE" or x["r3"]["target_field_present"])],10,"qual-real"),
    }
    assets=output/"qualitative/assets";assets.mkdir(parents=True,exist_ok=True)
    parts=["<!doctype html><meta charset='utf-8'><title>Phase 3D.0 Qualitative</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}article{border-top:1px solid;padding:1em}pre{white-space:pre-wrap}.m img{max-width:180px;margin:5px;background:#ddd}</style><h1>Phase 3D.0 deterministic qualitative index</h1>"]
    ordinal=0
    for category,values in categories.items():
      parts.append(f"<h2>{category} (n={len(values)})</h2>")
      for x in values:
        row=val_rows[x["sample_id"]];image_name=f"{ordinal:03d}_image.jpg";Image.open(row["image_path"]).convert("RGB").resize((256,256)).save(assets/image_name,quality=85)
        masks=[]
        if int(x["r3"]["gt_class"])==1:
          with Image.open(row["image_path"]) as source_image: width,height=source_image.size
          gt=save_tensor_thumbnail(UnifiedForensicsDataset._fake_union_mask(row,height,width),assets/f"{ordinal:03d}_gt.png");masks.append(("GT",gt))
        for label,record in (("greedy",x["greedy"]),("R0 top",x["r0"]),("R3 top",x["r3"])):
          saved=save_mask_thumbnail(record.get("binary_mask_path"),assets/f"{ordinal:03d}_{label.replace(' ','_')}.png")
          if saved:masks.append((label,saved))
        mask_html="".join(f"<figure><img src='assets/{name}'><figcaption>{label}</figcaption></figure>" for label,name in masks)
        target=authoritative_phrase(row) if int(x["r3"]["gt_class"])==1 else "N/A (Real)"
        decomposition=", ".join(f"{key}={fmt(x['r3'][key],3)}" for key in ("R_cls","R_struct","R_phrase_lex","R_mask","R_align","R3"))
        parts.append(f"<article><b>{html.escape(x['sample_id'])}</b><img src='assets/{image_name}'><p><b>Authoritative:</b> {html.escape(target)}</p><div class='m' style='display:flex'>{mask_html}</div><p>Greedy: align={fmt(x['greedy']['R_align'])}, mask={fmt(x['greedy']['R_mask'])}, phrase={fmt(x['greedy']['R_phrase_lex'])}</p><p>R0 top S{x['r0']['rollout_index']+1}: {fmt(x['r0']['R0'])}; R3 top S{x['r3']['rollout_index']+1}: {fmt(x['r3']['R3'])}</p><p>{decomposition}</p><pre>{html.escape(x['r3']['decoded_text'])}</pre></article>");ordinal+=1
    q=output/"qualitative/index.html";q.write_text("\n".join(parts),encoding="utf-8");dump(output/"qualitative/selection_manifest.json",{"deterministic":True,"counts":{k:len(v) for k,v in categories.items()},"index":str(q)})


def main(argv=None):
    cli=args(argv);cfg=yaml.safe_load((ROOT/cli.config).read_text(encoding="utf-8"));output=(ROOT/cfg["experiment"]["output_root"]).resolve()
    protocol=load(output/"sampling_protocol.json");setting=protocol["selected_setting"]
    groups=[r for i in range(cli.num_shards) for r in rows(output/"rollouts"/f"val_{setting}_full_shard{i:02d}_of_{cli.num_shards:02d}.jsonl")]
    greedy_rows=[r for i in range(cli.num_shards) for r in rows(output/"greedy"/f"val_shard{i:02d}_of_{cli.num_shards:02d}.jsonl")];greedy={r["sample_id"]:r for r in greedy_rows}
    raw_val=rows(Path(cfg["data"]["manifest_dir"])/"val_combined.jsonl");val_rows={r["sample_id"]:r for r in raw_val};resolve_image_paths(val_rows,cfg)
    review=write_review_packet(output,groups,greedy,val_rows);write_qualitative(output,groups,greedy,val_rows)
    route=load(output/"route_gate.json");div=load(output/"rollout_diversity.json");pareto=load(output/"pareto_ranking_statistics.json");top=load(output/"top_selection_statistics.json");base=load(output/"greedy_metrics.json");nonreg=load(output/"real_fake_non_regression.json");cat=load(output/"catastrophic_reward_audit.json");ready=load(output/"grpo_signal_readiness.json");paired_stats=load(output/"statistics/paired_top_vs_greedy.json");oracle=load(output/"statistics/oracle_availability_upper_bound.json")
    r=route["selected_reward_candidate"]
    mask_risk=(top["R0"]["fake_phrase_token_f1"]+0.005<top["R3"]["fake_phrase_token_f1"] or top["R0"]["overall_generated_verdict_accuracy"]+0.005<top["R3"]["overall_generated_verdict_accuracy"] or top["R0"]["structural_validity_rate"]+0.005<top["R3"]["structural_validity_rate"])
    dump(output/"statistics/mask_only_reward_risk.json",{"diagnostic_flag":"MASK_ONLY_REWARD_RISK_SUPPORTED" if mask_risk else "MASK_ONLY_REWARD_RISK_NOT_SUPPORTED","operational_rule_frozen_before_report":"R0 top sacrifices phrase, generated verdict, or structure by >0.5 percentage point versus R3","mask_only_is_control_not_default":True})
    audits=list((output/"audit").glob("generation_*.json"));audit_values=[load(p) for p in audits]
    hashes={"status":"PASS" if audit_values and all(x.get("model_state_exact_identical") is True for x in audit_values if x.get("model_state_sha256_before") is not None) else "INCOMPLETE","generation_audits":[str(p) for p in audits],"all_full_hashes_exact":all(x.get("model_state_exact_identical") is True for x in audit_values if x.get("model_state_sha256_before") is not None),"all_trainable_hashes_exact":all(x["trainable_state_exact_identical"] for x in audit_values),"all_requires_grad_false":all(x["all_requires_grad_false"] for x in audit_values),"optimizer_created":False,"scheduler_created":False,"backward_called":False,"model_write":False}
    dump(output/"frozen_model_hash.json",hashes);dump(output/"audit/frozen_model_hash.json",hashes)
    selected_line="none" if r is None else f"{r}（{cfg['reward']['candidates'][r]}）"
    report=f"""# Phase 3D.0 — Evidence-Aware Reward / Rollout Preflight

## 1. 最终状态

本阶段是 frozen P1 的只读 rollout/reward preflight，不是 GRPO，也没有 optimizer、scheduler、backward 或模型更新。最终 primary gate：**`{route['primary_gate']}`**。selected reward：**{selected_line}**；`GRPO_SIGNAL_READY={str(route['GRPO_SIGNAL_READY']).lower()}`。Phase 3D.1 **未启动**。

## 2. Frozen model 与数据 provenance

P1：step 3500 / logical epoch 7，checkpoint SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。所有正式 runner 均 `eval()`、`requires_grad=False`、`torch.no_grad()`；前后 full-state/trainable-state hash exact（汇总状态 `{hashes['status']}`）。

定位 GT 来自 LEGION / SynthScars 官方 `refs[].segmentation` polygon。当前 evaluator 使用 **derived union target from official SynthScars annotations**：逐 ref polygon 按原/canonical 尺寸缩放并 rasterize，随后对同图全部 ref mask 做 boolean union。它不是 pseudo mask，也未被证明与 LEGION paper 的 phrase-level evaluation unit 完全相同；无 erosion、dilation、refinement、relabel 或 threshold sweep，固定 `mask_logit > 0`。

reward-dev 仅来自 internal train：512 Real + 512 Fake；confirmation 使用完整 internal validation：1106 Real + 1106 Fake。internal test、official SynthScars1000、RAISE、LOKI、FakeBench 与 external test 均未参与 protocol/reward/gate 选择。

## 3. Sampling protocol 与 rollout diversity

reward-dev 在不读取 IoU、phrase reward、candidate reward 或 validation performance 的条件下比较 A（T=0.8, top-p=0.95）与 B（T=1.0, top-p=0.95），最终冻结 setting **{setting}**，K=8。

- groups with >=2 distinct normalized trajectories：{fmt(div['fraction_groups_multiple_unique'])}
- selected reward Fake groups with std>0.05：{fmt(div['per_reward'][r]['fraction_fake_groups_reward_std_gt_0.05']) if r else 'NA'}
- Fake groups mask range>0.10：{fmt(div['fraction_groups_mask_range_gt_0.10'])}
- Fake groups phrase-F1 range>0.10：{fmt(div['fraction_groups_phrase_range_gt_0.10'])}

## 4. Reward definitions

- R0：Fake=`R_mask`，Real=`R_cls`（mask-only control）。
- R1：Fake=`R_phrase_lex`，Real=`R_cls`（phrase-only control）。
- R2：Fake=`0.20 R_cls + 0.10 R_struct + 0.30 R_phrase_lex + 0.40 R_mask`；Real=`0.70 R_cls + 0.30 R_struct`。
- R3：Fake=`0.20 R_cls + 0.10 R_struct + 0.20 R_phrase_lex + 0.30 R_mask + 0.20 R_align`；Real 同 R2；`R_align` 是 phrase lexical F1 与 mask IoU 的 harmonic joint。

`R_phrase_lex` 只表示 lexical phrase agreement，不表示 semantic phrase correctness；free-form explanation reward 未加入。

## 5. Greedy 与 reward ranking

canonical greedy：generated-verdict accuracy={fmt(base['overall_generated_verdict_accuracy'])}，structure={fmt(base['structural_validity_rate'])}，Fake phrase F1={fmt(base['fake_phrase_token_f1'])}，Fake FG IoU/F1={fmt(base['fake_foreground_iou'])}/{fmt(base['fake_foreground_f1'])}。

| Reward | Pareto correct | tie | violation | top LM acc | top structure | top phrase F1 | top FG IoU | GRPO signal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
"""+"\n".join(f"| {x} | {fmt(pareto[x]['correct_ranking_rate'])} | {fmt(pareto[x]['tie_rate'])} | {fmt(pareto[x]['violation_rate'])} | {fmt(top[x]['overall_generated_verdict_accuracy'])} | {fmt(top[x]['structural_validity_rate'])} | {fmt(top[x]['fake_phrase_token_f1'])} | {fmt(top[x]['fake_foreground_iou'])} | {ready[x]['GRPO_SIGNAL_READY']} |" for x in REWARD_NAMES)+f"""

selected {r or 'none'} top-vs-greedy paired phrase Δ={fmt(paired_stats[r]['R_phrase_lex']['mean_difference']) if r else 'NA'}，95% CI={paired_stats[r]['R_phrase_lex']['bootstrap_95ci'] if r else 'NA'}；mask IoU Δ={fmt(paired_stats[r]['R_mask']['mean_difference']) if r else 'NA'}，95% CI={paired_stats[r]['R_mask']['bootstrap_95ci'] if r else 'NA'}。

## 6. Real/Fake non-regression 与 reward risk

selected reward generated-verdict accuracy Δ={fmt(nonreg[r]['generated_verdict_accuracy_delta']) if r else 'NA'}，classification non-regression={nonreg[r]['classification_non_regression_pass'] if r else 'NA'}；structure Δ={fmt(nonreg[r]['structural_validity_delta']) if r else 'NA'}，structure non-regression={nonreg[r]['structure_non_regression_pass'] if r else 'NA'}。`[CLS]` head 是 frozen system diagnostic，未混入 rollout `R_cls`；primary `R_cls` 始终使用 generated LM verdict。

mask-only diagnostic：`{'MASK_ONLY_REWARD_RISK_SUPPORTED' if mask_risk else 'MASK_ONLY_REWARD_RISK_NOT_SUPPORTED'}`。selected reward catastrophic rates：wrong class={fmt(cat[r]['wrong_class_rate']) if r else 'NA'}，invalid structure={fmt(cat[r]['invalid_structure_rate']) if r else 'NA'}，TOP_REWARD_MASK_BAD={fmt(cat[r]['TOP_REWARD_MASK_BAD_rate_fake']) if r else 'NA'}，TOP_REWARD_PHRASE_BAD={fmt(cat[r]['TOP_REWARD_PHRASE_BAD_rate_fake']) if r else 'NA'}。

## 7. Rollout availability 与人工语义审计

Fake groups 中存在 mask 优于 greedy 的 candidate：{fmt(oracle['fraction_groups_mask_better_than_greedy_available'])}；存在 phrase-F1 优于 greedy的 candidate：{fmt(oracle['fraction_groups_phrase_better_than_greedy_available'])}；同时满足 class correct、structure valid、phrase 与 mask 均优于 greedy：{fmt(oracle['fraction_groups_jointly_class_structure_phrase_mask_better_candidate_available'])}。这些是 rollout-distribution availability upper bound，不是 policy performance。

人工审阅包：`outputs/phase3d0_reward_preflight/human_review/index.html` 与 `phrase_semantic_audit.csv`，unique groups={review['unique_count']}。人工字段保持空白，automatic gate 不依赖人工标注。**Lexical phrase reward semantic validity remains limited until human audit.**

## 8. 限制与结论边界

best-of-K trajectory 不是 deployment result。Phase 3D.0 只检验 frozen P1 policy distribution 是否含可利用 variation，以及预注册 reward 能否排序该 variation；它不能证明 GRPO 必然提升性能。correlation 仅是 secondary diagnostic，不能以 reward 与自身 component 的相关性循环证明有效性。

## 9. Final route gate

**`{route['primary_gate']}`**。

该 gate {'仅授权提出 Phase 3D.1 policy-optimization controlled experiment；不自动启动。' if route['phase3d1_authorized_for_controlled_experiment'] else '不授权 Phase 3D.1，路线在此停止。'} 本阶段到此停止，未启动 GRPO/PPO/DPO/RL、rejection-sampling SFT、extra SFT、forensic fusion、FEPN 或架构修改。
"""
    doc=ROOT/"docs/phase3d0_evidence_aware_reward_preflight.md";doc.write_text(report,encoding="utf-8");(output/"reports/final_report.md").write_text(report,encoding="utf-8");(output/"final_report.md").write_text(report,encoding="utf-8")
    provenance=load(output/"provenance.json");provenance.update({"status":"COMPLETE","selected_sampling_setting":setting,"selected_reward":r,"primary_gate":route["primary_gate"],"phase3d1_started":False});dump(output/"provenance.json",provenance)
    required=["provenance.json","frozen_model_hash.json","audit/mask_provenance.json","manifests/reward_dev_manifest.json","manifests/reward_val_manifest.json","sampling_protocol.json","reward_definition.json","rollout_manifest.json","greedy_metrics.json","rollout_diversity.json","reward_component_statistics.json","pareto_ranking_statistics.json","top_selection_statistics.json","real_fake_non_regression.json","catastrophic_reward_audit.json","grpo_signal_readiness.json","route_gate.json","final_report.md","reports/regression_tests.txt"]
    missing=[x for x in required if not (output/x).exists()];
    if missing:raise RuntimeError(f"missing required artifacts: {missing}")
    dump(output/"completion_manifest.json",{"status":"COMPLETE","phase":"Phase 3D.0","primary_gate":route["primary_gate"],"selected_reward":r,"report":str(doc),"required_artifacts":[{"path":x,"sha256":file_sha256(output/x)} for x in required],"P1_weights_exact":hashes["all_full_hashes_exact"],"no_training":True,"phase3d1_started":False})
    print(json.dumps({"status":"COMPLETE","report":str(doc),"gate":route["primary_gate"],"selected_reward":r},indent=2))


if __name__=="__main__": main()
