#!/usr/bin/env python3
"""Finalize Phase 3D.1 paired statistics, failure analysis, reports, and route gate."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from scripts.phase2a_final_evaluate import file_sha256
from scripts.phase3d0_analyze import paired
from tools.phase3d0r import FrozenSentenceEncoder, content_phrase, content_tokens, grounding_components, semantic_phrase_score


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def write_rows(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as handle:
        for value in values:handle.write(json.dumps(value,ensure_ascii=False)+"\n")
def mean(values): return float(np.mean(values)) if values else None
def median(values): return float(np.median(values)) if values else None


def bootstrap_summary(values, seed=3407, repeats=10000):
    array=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    means=np.empty(repeats)
    for start in range(0,repeats,500):
        count=min(500,repeats-start);indices=rng.integers(0,len(array),size=(count,len(array)))
        means[start:start+count]=array[indices].mean(axis=1)
    return {"n":len(array),"mean":float(array.mean()),"median":float(np.median(array)),
            "bootstrap_95ci":[float(x) for x in np.quantile(means,[.025,.975])]}


def greedy_arm(root, arm):
    selector=load(root/f"evaluation/selector/{arm}_selector.json")
    step=int(selector["optimizer_step"]);base=root/f"evaluation/selector/{arm}_OPT/step_{step:04d}"
    g0=rows(base/"G0/predictions.jsonl");detection=rows(base/"detection/predictions.jsonl")
    semantic=rows(base/"semantic_structure.jsonl")
    return selector,g0,detection,semantic


def selected_rollout_scores(root, arm, manifest_rows, encoder, idf):
    groups=rows(root/f"rollouts/{arm}_OPT_selected_K8.jsonl");by_id={x["sample_id"]:x for x in manifest_rows}
    refs={sid:UnifiedForensicsDataset.authoritative_localization_field(row)["normalized_training_phrase"]
          for sid,row in by_id.items() if int(row["class_label"])==1}
    inputs=set()
    for group in groups:
        if group["sample_id"] not in refs:continue
        for phrase in [refs[group["sample_id"]],*[x.get("normalized_phrase") for x in group["rollouts"]]]:
            inputs.add(content_phrase(phrase));inputs.update(content_tokens(phrase))
    embeddings=encoder.encode(inputs,batch_size=128)
    scored=[];fake_rows=[]
    for group in groups:
        fake=group["sample_id"] in refs
        grounds=grounding_components([float(x["R_mask"]) for x in group["rollouts"]]) if fake else [None]*8
        enriched=[]
        for index,original in enumerate(group["rollouts"]):
            value=dict(original)
            if fake:
                value.update(semantic_phrase_score(refs[group["sample_id"]],value.get("normalized_phrase"),idf,embeddings))
                value.update(grounds[index]);a=float(value["R_phrase_sem"]);b=float(value["R_mask"])
                value["phrase_mask_consistency_semantic_hmean"]=2*a*b/(a+b+1e-8)
                fake_rows.append(value)
            enriched.append(value)
        scored.append({**{k:v for k,v in group.items() if k!="rollouts"},"rollouts":enriched})
    write_rows(root/f"rewards/{arm}_OPT_selected_K8_scored.jsonl",scored)
    return {"groups":len(groups),"fake_trajectories":len(fake_rows),
            "mean_R_ground_rel":mean([x["R_ground_rel"] for x in fake_rows]),
            "mean_phrase_mask_consistency":mean([x["phrase_mask_consistency_semantic_hmean"] for x in fake_rows]),
            "mean_R_phrase_sem":mean([x["R_phrase_sem"] for x in fake_rows]),
            "mean_absolute_foreground_iou":mean([x["R_mask"] for x in fake_rows])}


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase3d1_policy_optimization.yaml").read_text());root=(ROOT/cfg["experiment"]["output_root"]).resolve()
    val_rows=rows(ROOT/cfg["data"]["manifest_dir"]/"val_combined.jsonl")
    rsel,rg0,rdet,rsem=greedy_arm(root,"R3");qsel,qg0,qdet,qsem=greedy_arm(root,"Q2")
    p1=rows(ROOT/"outputs/phase3d0r_reward_reformulation/validation/scored_greedy.jsonl")
    enc_cfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text())["semantic_encoder"]
    encoder=FrozenSentenceEncoder(enc_cfg["model_id"],enc_cfg["revision"],enc_cfg["cache_dir"],"cpu")
    idf=load(ROOT/cfg["reward"]["Q2"]["token_idf"])["values"]
    stochastic={arm:selected_rollout_scores(root,arm,val_rows,encoder,idf) for arm in ("R3","Q2")}

    def index(values):return {x["sample_id"]:x for x in values}
    ri,qi=index(rg0),index(qg0);rs,qs=index(rsem),index(qsem);rd,qd=index(rdet),index(qdet)
    fake_ids=sorted(set(ri)&set(qi)&set(rs)&set(qs));all_ids=sorted(set(rd)&set(qd))
    contrasts={
      "foreground_iou":paired([qi[x]["foreground_iou"] for x in fake_ids],[ri[x]["foreground_iou"] for x in fake_ids]),
      "foreground_f1":paired([qi[x]["foreground_f1"] for x in fake_ids],[ri[x]["foreground_f1"] for x in fake_ids]),
      "R_phrase_sem":paired([qs[x]["R_phrase_sem"] for x in fake_ids],[rs[x]["R_phrase_sem"] for x in fake_ids]),
      "classification_accuracy":paired([float(qd[x]["cls_pred"]==qd[x]["gt_label"]) for x in all_ids],
                                       [float(rd[x]["cls_pred"]==rd[x]["gt_label"]) for x in all_ids]),
    }
    def arm_summary(g0,det,sem):
        return {"foreground_iou":bootstrap_summary([x["foreground_iou"] for x in g0]),
                "foreground_f1":bootstrap_summary([x["foreground_f1"] for x in g0]),
                "R_phrase_sem":bootstrap_summary([x["R_phrase_sem"] for x in sem]),
                "R_key_soft":mean([x["R_key_soft"] for x in sem]),
                "R_sentence_sem":mean([x["R_sentence_sem"] for x in sem]),
                "classification_accuracy":mean([x["cls_pred"]==x["gt_label"] for x in det]),
                "classification_f1":load(root/f"evaluation/selector/{'R3' if g0 is rg0 else 'Q2'}_OPT/step_{int(rsel['optimizer_step'] if g0 is rg0 else qsel['optimizer_step']):04d}/summary.json")["modes"]["detection"]["classification_head"]["f1"],
                "valid_seg_rate":mean([x["usable_seg"] for x in sem]),
                "structure_validity":mean([x["structural_validity"] for x in sem])}
    summaries={"P3D1-R3":arm_summary(rg0,rdet,rsem),"P3D1-Q2":arm_summary(qg0,qdet,qsem)}
    p1fake=[x for x in p1 if int(x["gt_class"])==1]
    summaries["P1-FROZEN"]={"foreground_iou":bootstrap_summary([x["foreground_iou"] for x in p1fake]),
      "foreground_f1":bootstrap_summary([x["foreground_f1"] for x in p1fake]),
      "R_phrase_sem":bootstrap_summary([x["R_phrase_sem"] for x in p1fake]),
      "R_key_soft":mean([x["R_key_soft"] for x in p1fake]),"R_sentence_sem":mean([x["R_sentence_sem"] for x in p1fake]),
      "classification_accuracy":mean([int(x["cls_pred"])==int(x["gt_class"]) for x in p1]),
      "classification_f1":None,"valid_seg_rate":mean([x["usable_seg"] for x in p1fake]),
      "structure_validity":mean([x["structural_validity"] for x in p1fake])}
    dump(root/"statistics/final_metrics.json",{"arms":summaries,"Q2_minus_R3":contrasts,"selected":{"R3":rsel,"Q2":qsel},"stochastic_K8":stochastic})

    gate_cfg=load(ROOT/"configs/phase3d1_route_gate.json");d={k:v["mean_difference"] for k,v in contrasts.items()}
    iou_ci=contrasts["foreground_iou"]["bootstrap_95ci"];f1_ci=contrasts["foreground_f1"]["bootstrap_95ci"];sem_ci=contrasts["R_phrase_sem"]["bootstrap_95ci"]
    class_ok=d["classification_accuracy"]>=gate_cfg["classification_nonregression_tolerance"]
    supported=d["foreground_iou"]>0 and iou_ci[0]>0 and d["foreground_f1"]>0 and d["R_phrase_sem"]>0 and class_ok
    inconclusive=iou_ci[0]<=0<=iou_ci[1] and f1_ci[0]<=0<=f1_ci[1] and sem_ci[0]<=0<=sem_ci[1] and class_ok
    gate=("GATE_Q2_REWARD_MORE_EFFECTIVE_SUPPORTED" if supported else
          "GATE_REWARD_FORMULATION_NOT_PRIMARY_BOTTLENECK" if inconclusive else "GATE_Q2_NOT_BETTER_THAN_R3")
    route={"primary_gate":gate,"conditions":{"q2_iou_point_improved":d["foreground_iou"]>0,"q2_iou_ci_lower_gt_zero":iou_ci[0]>0,
      "q2_f1_point_improved":d["foreground_f1"]>0,"q2_phrase_sem_point_improved":d["R_phrase_sem"]>0,"classification_nonregression":class_ok},
      "Q2_minus_R3":contrasts,"RSFT_run":False,"additional_reward_tuning":False,"phase3d1_complete":True,
      "automatic_next_phase_started":False}
    dump(root/"route_gate.json",route)

    better=[];rbetter=[];both=[]
    for sid in fake_ids:
        rv,qv=ri[sid],qi[sid];rsv,qsv=rs[sid],qs[sid];di=qv["foreground_iou"]-rv["foreground_iou"];ds=qsv["R_phrase_sem"]-rsv["R_phrase_sem"]
        base={"sample_id":sid,"R3_iou":rv["foreground_iou"],"Q2_iou":qv["foreground_iou"],"iou_delta":di,
              "R3_phrase_sem":rsv["R_phrase_sem"],"Q2_phrase_sem":qsv["R_phrase_sem"],"phrase_sem_delta":ds,
              "R3_text":rv.get("decoded_text"),"Q2_text":qv.get("decoded_text")}
        if di>=.10 or ds>=.10:
            category=("spatial_contradiction_avoided" if rsv.get("P_spatial_contra") and not qsv.get("P_spatial_contra") else
                      "paraphrase_recovered" if ds>=.10 else "mask_ceiling_avoided")
            better.append({**base,"category":category})
        if di<=-.10 or ds<=-.10:
            category="semantic_scorer_failure" if ds>0 and di<-.10 else "grounding_over_regularization"
            rbetter.append({**base,"category":category})
        if rv["foreground_iou"]<=.20 and qv["foreground_iou"]<=.20:
            both.append({**base,"category":"segmentation_limitation"})
    failure={"Q2_better_than_R3":{"count":len(better),"examples":better[:100]},"R3_better_than_Q2":{"count":len(rbetter),"examples":rbetter[:100]},
             "both_fail":{"count":len(both),"examples":both[:100]},"oracle_failure":"not diagnosed without a frozen stronger spatial oracle"}
    dump(root/"failure_analysis.json",failure)
    failure_md=f"# Phase 3D.1 Failure Analysis\n\nQ2 better cases: {len(better)}; R3 better cases: {len(rbetter)}; both FG IoU <= 0.20: {len(both)}. Detailed per-sample records are in `failure_analysis.json`.\n"
    (root/"failure_analysis.md").write_text(failure_md,encoding="utf-8")
    fmt=lambda x:f"{x:.6f}"
    report=f"""# Phase 3D.1 — Evidence-Aware Policy Optimization

## 实验边界

核心比较严格限定为 P1-FROZEN、P3D1-R3、P3D1-Q2；没有加入 RSFT。R3/Q2 均从同一 P1 SHA256 `{cfg['source']['checkpoint_sha256']}` 初始化，K=8、2000 optimizer steps、数据、采样、优化器、学习率、scheduler、可训练模块与 checkpoint 间隔一致，唯一研究变量是冻结 reward。

## Selector

R3 选择 step {rsel['optimizer_step']}，Q2 选择 step {qsel['optimizer_step']}。两者只使用 internal validation Fake mean FG IoU，tie-break 为 FG F1；training reward、test、official1000 均未参与。

## 核心结果

| Arm | mean FG IoU | mean FG F1 | R_phrase_sem | CLS accuracy |
|---|---:|---:|---:|---:|
| P1-FROZEN | {fmt(summaries['P1-FROZEN']['foreground_iou']['mean'])} | {fmt(summaries['P1-FROZEN']['foreground_f1']['mean'])} | {fmt(summaries['P1-FROZEN']['R_phrase_sem']['mean'])} | {fmt(summaries['P1-FROZEN']['classification_accuracy'])} |
| P3D1-R3 | {fmt(summaries['P3D1-R3']['foreground_iou']['mean'])} | {fmt(summaries['P3D1-R3']['foreground_f1']['mean'])} | {fmt(summaries['P3D1-R3']['R_phrase_sem']['mean'])} | {fmt(summaries['P3D1-R3']['classification_accuracy'])} |
| P3D1-Q2 | {fmt(summaries['P3D1-Q2']['foreground_iou']['mean'])} | {fmt(summaries['P3D1-Q2']['foreground_f1']['mean'])} | {fmt(summaries['P3D1-Q2']['R_phrase_sem']['mean'])} | {fmt(summaries['P3D1-Q2']['classification_accuracy'])} |

Q2−R3 paired bootstrap：FG IoU Δ={d['foreground_iou']:+.6f}, 95% CI={iou_ci}；FG F1 Δ={d['foreground_f1']:+.6f}, 95% CI={f1_ci}；R_phrase_sem Δ={d['R_phrase_sem']:+.6f}, 95% CI={sem_ci}；classification accuracy Δ={d['classification_accuracy']:+.6f}。

K=8 stochastic evaluation：R3 mean R_ground_rel={stochastic['R3']['mean_R_ground_rel']:.6f}，Q2={stochastic['Q2']['mean_R_ground_rel']:.6f}；semantic phrase-mask harmonic consistency R3={stochastic['R3']['mean_phrase_mask_consistency']:.6f}，Q2={stochastic['Q2']['mean_phrase_mask_consistency']:.6f}。

## 结论与停止门

主门：**{gate}**。该结论只回答 Q2 与 R3 哪个更适合作为当前 forensic MLLM policy-optimization signal，不把 selector 内的 validation 差异外推为未知 test/真实图像泛化结论。

Phase 3D.1 到此停止；未自动启动 FEPN、NPR、FOCAL、架构修改、额外 reward tuning 或 RSFT。
"""
    (root/"final_comparison.md").write_text(report,encoding="utf-8")
    (root/"reward_ablation.md").write_text("# Reward Ablation\n\n"+report.split("## 核心结果",1)[1],encoding="utf-8")
    docs=ROOT/"docs/phase3d1_evidence_aware_policy_optimization.md";shutil.copy2(root/"final_comparison.md",docs)
    checkpoint_metadata={arm:rows(root/f"experiments/{arm}_OPT/checkpoint_metadata.jsonl") for arm in ("R3","Q2")}
    dump(root/"checkpoint_metadata.json",checkpoint_metadata)
    artifacts=["experiment_manifest.json","fairness_manifest.json","reward_version_manifest.json","training_config.json","selector_protocol.json",
               "checkpoint_metadata.json","final_comparison.md","reward_ablation.md","failure_analysis.md","failure_analysis.json","route_gate.json"]
    dump(root/"completion_manifest.json",{"status":"COMPLETE","artifacts":[{"path":str(root/x),"sha256":file_sha256(root/x)} for x in artifacts],
      "selected_checkpoints":{"R3":rsel["selected_checkpoint"],"Q2":qsel["selected_checkpoint"]},"primary_gate":gate,"RSFT_run":False})
    print(json.dumps(route,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
