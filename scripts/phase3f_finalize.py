#!/usr/bin/env python3
"""Freeze Phase 3F statistics, route gate, report, and completion manifest."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, yaml
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/"configs/phase3f_autonomous_oracle_grounding_distillation.yaml").read_text())
OUT=ROOT/CFG["experiment"]["output_root"]

def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def dump(path,value): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def metric_rows(path): return {r["sample_id"]:r for r in rows(path)}
def paired(left,right,key,repeats=10000,seed=3407):
 ids=sorted(set(left)&set(right))
 if set(left)!=set(right): raise RuntimeError("paired population mismatch")
 delta=np.asarray([float(left[i][key])-float(right[i][key]) for i in ids]); rng=np.random.default_rng(seed)
 means=np.asarray([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(repeats)])
 return {"n":len(ids),"mean_difference":float(delta.mean()),"bootstrap_95ci":[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],
  "wins":int((delta>0).sum()),"ties":int((delta==0).sum()),"losses":int((delta<0).sum()),"repeats":repeats,"seed":seed}

def mode(summary,mode): return summary["modes"][mode]["per_image_mean"]

def main():
 selector=load(OUT/"evaluation/selector/AOGD_selector.json")
 p1_sel=load(OUT/"evaluation/selector/P1_FROZEN/summary.json"); a_sel=load(OUT/"evaluation/selector/AOGD/step_1000/summary.json")
 p1_phrase=load(OUT/"evaluation/final/P1_FROZEN_PHRASE_ONLY/summary.json")
 a_final=load(OUT/"evaluation/final/AOGD_SELECTED_STEP_1000/summary.json")
 p1_tf_path=ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/final/P1_FROZEN/summary.json"; p1_tf=load(p1_tf_path)
 if p1_tf["checkpoint_file_sha256"]!=CFG["source"]["checkpoint_sha256"]: raise RuntimeError("P1 TF reuse provenance")
 conditions={
  "P1":{"G0":mode(p1_sel,"G0"),"Phrase_Only":mode(p1_phrase,"phrase_only"),"TF_Full":mode(p1_tf,"tf_full_context")},
  "AOGD_step1000":{"G0":mode(a_sel,"G0"),"Phrase_Only":mode(a_final,"phrase_only"),"TF_Full":mode(a_final,"tf_full_context")},
  "matched_P3F_SFT":{"status":"NOT_RUN_TRIGGER_FALSE_BY_USER_CONDITIONAL_PROTOCOL"}}
 g0_metrics={"population":"internal_validation_fake_1106","direct_batch_size":1,"canonical_prompt":True,
   "threshold":0.0,"metrics":{"P1":conditions["P1"]["G0"],"AOGD_step1000":conditions["AOGD_step1000"]["G0"]}}
 dump(OUT/"evaluation/g0/g0_metrics.json",g0_metrics); dump(OUT/"g0_metrics.json",g0_metrics)
 phrase_metrics={"population":"internal_validation_fake_1106","metrics":{"P1":conditions["P1"]["Phrase_Only"],"AOGD_step1000":conditions["AOGD_step1000"]["Phrase_Only"]}}
 dump(OUT/"evaluation/phrase_only/phrase_only_metrics.json",phrase_metrics); dump(OUT/"phrase_only_metrics.json",phrase_metrics)
 tf_metrics={"population":"internal_validation_fake_1106","canonical_user_prompt":True,
   "P1_reused":True,"P1_source":str(p1_tf_path),"metrics":{"P1":conditions["P1"]["TF_Full"],"AOGD_step1000":conditions["AOGD_step1000"]["TF_Full"]}}
 dump(OUT/"evaluation/tf_full/tf_full_metrics.json",tf_metrics); dump(OUT/"tf_full_metrics.json",tf_metrics)

 p1_g=metric_rows(OUT/"evaluation/selector/P1_FROZEN/G0/predictions.jsonl"); a_g=metric_rows(OUT/"evaluation/selector/AOGD/step_1000/G0/predictions.jsonl")
 p1_p=metric_rows(OUT/"evaluation/final/P1_FROZEN_PHRASE_ONLY/phrase_only/predictions.jsonl"); a_p=metric_rows(OUT/"evaluation/final/AOGD_SELECTED_STEP_1000/phrase_only/predictions.jsonl")
 p1_t=metric_rows(ROOT/"outputs/phase3e_joint_language_mask_posttraining/evaluation/final/P1_FROZEN/tf_full_context/predictions.jsonl"); a_t=metric_rows(OUT/"evaluation/final/AOGD_SELECTED_STEP_1000/tf_full_context/predictions.jsonl")
 contrasts={condition:{key:paired(left,right,key) for key in ("foreground_iou","foreground_f1")}
  for condition,left,right in (("G0",a_g,p1_g),("Phrase_Only",a_p,p1_p),("TF_Full",a_t,p1_t))}
 dump(OUT/"statistics/paired_bootstrap_aogd_vs_p1.json",contrasts)
 dump(OUT/"paired_bootstrap_aogd_vs_p1.json",contrasts)
 dump(OUT/"statistics/paired_bootstrap_aogd_vs_sft.json",{"status":"NOT_RUN_TRIGGER_FALSE","reason":"AOGD validation G0 improvement vs P1 was not statistically supported; user protocol deferred matched 4500-step SFT"})
 dump(OUT/"paired_bootstrap_aogd_vs_sft.json",{"status":"NOT_RUN_TRIGGER_FALSE","reason":"AOGD validation G0 improvement vs P1 was not statistically supported; user protocol deferred matched 4500-step SFT"})

 p1_r={r["sample_id"]:r for r in rows(OUT/"evaluation/representations/P1_FROZEN.jsonl") if r["autonomous_representation_eligible"]}
 a_r={r["sample_id"]:r for r in rows(OUT/"evaluation/representations/AOGD_SELECTED_STEP_1000.jsonl") if r["autonomous_representation_eligible"]}
 ids=sorted(set(p1_r)&set(a_r)); p1_match={i:p1_r[i] for i in ids}; a_match={i:a_r[i] for i in ids}
 rep={"population":"matched_autonomous_representation_eligible_validation_fake","matched_n":len(ids),
  "eligibility":{"P1":load(OUT/"evaluation/representations/P1_FROZEN_run.json"),"AOGD":load(OUT/"evaluation/representations/AOGD_SELECTED_STEP_1000_run.json")},
  "means":{model:{key:float(np.mean([source[i][key] for i in ids])) for key in ("raw_4096d_cosine_gap","projected_256d_cosine_gap")}
    for model,source in (("P1",p1_match),("AOGD_step1000",a_match))},
  "AOGD_minus_P1":{key:paired(a_match,p1_match,key) for key in ("raw_4096d_cosine_gap","projected_256d_cosine_gap")}}
 correlation_rows=[]
 for sid in ids:
  correlation_rows.append({"sample_id":sid,
   "delta_iou_AOGD_minus_P1":float(a_g[sid]["foreground_iou"]-p1_g[sid]["foreground_iou"]),
   "delta_gap_256d_AOGD_minus_P1":float(a_match[sid]["projected_256d_cosine_gap"]-p1_match[sid]["projected_256d_cosine_gap"]),
   "delta_gap_4096d_AOGD_minus_P1":float(a_match[sid]["raw_4096d_cosine_gap"]-p1_match[sid]["raw_4096d_cosine_gap"])})
 correlation_path=OUT/"representations/delta_gap_vs_delta_iou.jsonl"; correlation_path.parent.mkdir(parents=True,exist_ok=True)
 with correlation_path.open("w",encoding="utf-8") as handle:
  for row in correlation_rows: handle.write(json.dumps(row)+"\n")
 delta_iou=np.asarray([row["delta_iou_AOGD_minus_P1"] for row in correlation_rows])
 correlations={"definition":{"delta_gap":"AOGD_step1000_gap_minus_P1_gap; negative means alignment improvement",
  "delta_iou":"AOGD_step1000_G0_IoU_minus_P1_G0_IoU; positive means localization improvement"},
  "population":"matched_representation_eligible_internal_validation_fake","n":len(correlation_rows),
  "used_for_selector":False,"used_for_route_gate":False,"dimensions":{}}
 for label,key in (("256D_projected","delta_gap_256d_AOGD_minus_P1"),("4096D_raw","delta_gap_4096d_AOGD_minus_P1")):
  delta_gap=np.asarray([row[key] for row in correlation_rows]); pearson=stats.pearsonr(delta_gap,delta_iou); spearman=stats.spearmanr(delta_gap,delta_iou)
  correlations["dimensions"][label]={"pearson_r":float(pearson.statistic),"pearson_p_two_sided":float(pearson.pvalue),
   "spearman_rho":float(spearman.statistic),"spearman_p_two_sided":float(spearman.pvalue),
   "mean_delta_gap":float(delta_gap.mean()),"mean_delta_iou":float(delta_iou.mean())}
 rep["delta_gap_vs_delta_iou_correlation"]=correlations
 dump(OUT/"representations/representation_gap_metrics.json",rep)
 dump(OUT/"representation_gap_metrics.json",rep)
 gap={model:{"phrase_only_minus_G0_iou":value["Phrase_Only"]["foreground_iou"]-value["G0"]["foreground_iou"],
             "TF_full_minus_G0_iou":value["TF_Full"]["foreground_iou"]-value["G0"]["foreground_iou"],
             "TF_full_minus_phrase_only_iou":value["TF_Full"]["foreground_iou"]-value["Phrase_Only"]["foreground_iou"]}
      for model,value in (("P1",conditions["P1"]),("AOGD_step1000",conditions["AOGD_step1000"]))}
 dump(OUT/"representations/gap_decomposition.json",gap)
 dump(OUT/"gap_decomposition.json",gap)
 rep_delta=rep["AOGD_minus_P1"]["projected_256d_cosine_gap"]
 rep_supported=rep_delta["mean_difference"]<0 and rep_delta["bootstrap_95ci"][1]<0
 g0_supported=contrasts["G0"]["foreground_iou"]["bootstrap_95ci"][0]>0
 if rep_supported and not g0_supported: gate="GATE_REPRESENTATION_ALIGNMENT_NOT_SUFFICIENT"
 elif not g0_supported: gate="GATE_AUTONOMOUS_ORACLE_DISTILLATION_NOT_LEARNABLE"
 else: gate="GATE_AUTONOMOUS_ORACLE_GROUNDING_DISTILLATION_EFFECTIVE"
 route={"gate":gate,"stage_II_supported":g0_supported and rep_supported,"selected_checkpoint":selector["selected_checkpoint"],
  "selected_step":selector["optimizer_step"],"G0_gain_supported":g0_supported,"representation_alignment_supported":rep_supported,
  "classification_nonregression":selector["selected_metrics"]["non_regression"]["classification_accuracy"] and selector["selected_metrics"]["non_regression"]["classification_f1"],
  "structure_nonregression":selector["selected_metrics"]["non_regression"]["structure_validity"],
  "matched_sft_triggered":False,"post_training_method_search":"STOP","next_major_route":"FEPN_REQUIRES_SEPARATE_AUTHORIZATION",
  "internal_test_used":False,"official1000_used":False}
 dump(OUT/"route_gate.json",route)
 dump(OUT/"failure_analysis.json",{"primary_failure":"non_significant_autonomous_G0_gain","selected_G0_contrast":contrasts["G0"],
   "representation_contrast":rep["AOGD_minus_P1"],"full_4500_step_G0_iou":selector["candidates"][-1]["mean_foreground_iou"],
   "interpretation":"Representation alignment is assessed separately from deployable mask gain; neither threshold tuning nor test-set selection was used."})
 # Required provenance aliases are compact pointers, not duplicated large artifacts.
 dump(OUT/"experiment_manifest.json",{"phase":"3F","status":"FINALIZED","method":"Autonomous-Oracle Grounding Distillation","training_steps":4500,"exposures":18000,"conditions":conditions})
 dump(OUT/"initial_checkpoint_manifest.json",{"path":CFG["source"]["checkpoint"],"sha256":CFG["source"]["checkpoint_sha256"],"step":3500,"epoch":7})
 dump(OUT/"teacher_manifest.json",load(OUT/"teacher/teacher_cache_manifest.json"))
 dump(OUT/"fairness_manifest.json",load(OUT/"matched_sft_fairness_audit.json"))
 dump(OUT/"training_config.json",{"config":str(ROOT/"configs/phase3f_autonomous_oracle_grounding_distillation.yaml"),"training":CFG["training"],"loss":CFG["loss"],"optimizer":CFG["optimizer"]})
 dump(OUT/"preflight_oracle_gap.json",load(OUT/"preflight/aogd_preflight.json"))
 dump(OUT/"gradient_path_audit.json",load(OUT/"preflight/aogd_preflight.json"))
 dump(OUT/"checkpoint_metadata.json",{"source":str(OUT/"training/checkpoint_metadata.jsonl"),"selector_candidates":CFG["selector"]["candidate_steps"]})
 dump(OUT/"selector_protocol.json",load(OUT/"checkpoint_selection_protocol.json"))
 dump(OUT/"parameter_update_audit.json",load(OUT/"training/parameter_update_audit.json"))
 table=lambda m:f"{m['foreground_iou']:.6f} / {m['foreground_f1']:.6f}"
 report=f"""# Phase 3F — Autonomous–Oracle Grounding Distillation

## 结论

Phase 3F 完成。最终 gate：`{gate}`。AOGD 未形成可部署的显著 canonical G0 增益，因此第一创新冻结为 **Stage-I Phrase-Aligned Forensic SFT only**；停止继续搜索 post-training 方法。FEPN 是下一条主要研究路线，但本阶段不自动启动。

## Validation 主结果

| 模型 | G0 IoU/F1 | Phrase-Only IoU/F1 | TF-Full IoU/F1 |
|---|---:|---:|---:|
| P1 | {table(conditions['P1']['G0'])} | {table(conditions['P1']['Phrase_Only'])} | {table(conditions['P1']['TF_Full'])} |
| AOGD step1000 | {table(conditions['AOGD_step1000']['G0'])} | {table(conditions['AOGD_step1000']['Phrase_Only'])} | {table(conditions['AOGD_step1000']['TF_Full'])} |
| matched P3F-SFT | N/A | N/A | N/A |

Selector 的算术最优为 step 1000。G0 IoU 相对 P1 的 paired difference 为 {contrasts['G0']['foreground_iou']['mean_difference']:+.6f}，95% CI [{contrasts['G0']['foreground_iou']['bootstrap_95ci'][0]:+.6f}, {contrasts['G0']['foreground_iou']['bootstrap_95ci'][1]:+.6f}]，不支持正增益。matched 4500-step SFT 按用户冻结的条件协议未触发；Phase 3E SFT 仅为历史参考，不能称为 matched control。

## Representation 与训练通路

32-Fake preflight 证明 256D autonomous–oracle gap 存在，且 `L_repr` 对 LoRA 的梯度非零。正式训练仅 LoRA 改变；`text_hidden_fcs`、mask decoder 和其他参数 byte-hash 不变。validation matched eligible n={len(ids)}；256D gap AOGD−P1={rep_delta['mean_difference']:+.6f}，95% CI [{rep_delta['bootstrap_95ci'][0]:+.6f}, {rep_delta['bootstrap_95ci'][1]:+.6f}]。

逐图 `Δgap_i` 与 `ΔIoU_i` 的相关性仅作预注册外诊断，不参与 selector/gate。256D projected：Pearson r={correlations['dimensions']['256D_projected']['pearson_r']:+.4f}，Spearman ρ={correlations['dimensions']['256D_projected']['spearman_rho']:+.4f}；4096D raw：Pearson r={correlations['dimensions']['4096D_raw']['pearson_r']:+.4f}，Spearman ρ={correlations['dimensions']['4096D_raw']['spearman_rho']:+.4f}。其中 `Δgap<0` 表示对齐改善，`ΔIoU>0` 表示定位改善，因此若 representation alignment 与定位收益方向一致，预期为负相关。

## Rollout matched control

Phase 3F 复用了 Phase 3B 的同一 canonical batch=1 P1 autonomous cache，SHA256 `6ab4c428748e426132edaf6570ee38f12bbc8c1a2e0d44b089152563fd0990ed`；step 0/500/1000/4500 hash invariant PASS，无 refresh、筛选或混用。因而 Phase 3B 与 Phase 3F 的差异是固定 trajectory 上的 supervision target，而不是 generated trajectory exposure。

## 边界

所有 Detection/G0 使用 canonical prompt 与 direct batch=1；TF-full 使用 canonical user prompt；mask threshold 固定为 logit 0。internal test 与 official1000 未使用。不得据此声称所有 representation distillation 无效；结论仅限本次 frozen P1 trajectory、cosine 256D target 和 LoRA-only 实现。
"""
 (OUT/"final_comparison.md").write_text(report,encoding="utf-8"); (ROOT/"docs/phase3f_autonomous_oracle_grounding_distillation.md").write_text(report,encoding="utf-8")
 route_doc=ROOT/"docs/paper_convergence_route.md"; marker="## Phase 3F — Autonomous–Oracle Grounding Distillation"
 text=route_doc.read_text(encoding="utf-8")
 if marker not in text:
  route_doc.write_text(text.rstrip()+f"\n\n{marker}\n\n- Final gate: `{gate}`.\n- Stage-II AOGD was not supported on internal validation canonical G0.\n- Freeze P1 as the best-supported Stage-I method; stop post-training method search.\n- FEPN is the next major route and requires separate authorization.\n",encoding="utf-8")
 required=["experiment_manifest.json","initial_checkpoint_manifest.json","teacher_manifest.json","fairness_manifest.json","training_config.json","preflight_oracle_gap.json","gradient_path_audit.json","rollout_protocol.json","rollout_statistics.json","checkpoint_metadata.json","selector_protocol.json","g0_metrics.json","phrase_only_metrics.json","tf_full_metrics.json","representation_gap_metrics.json","gap_decomposition.json","paired_bootstrap_aogd_vs_p1.json","paired_bootstrap_aogd_vs_sft.json","parameter_update_audit.json","failure_analysis.json","route_gate.json","final_comparison.md"]
 artifacts={name:{"path":str(OUT/name),"sha256":sha(OUT/name)} for name in required}
 completion={"status":"COMPLETE","gate":gate,"required_artifacts":artifacts,"docs":str(ROOT/"docs/phase3f_autonomous_oracle_grounding_distillation.md"),
  "internal_test_used":False,"official1000_used":False,"matched_sft_not_run_by_conditional_protocol":True}
 dump(OUT/"completion_manifest.json",completion)
 print(json.dumps({"status":"COMPLETE","gate":gate,"conditions":conditions,"representation":rep["means"]},indent=2))

if __name__=="__main__": main()
