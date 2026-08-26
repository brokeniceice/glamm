#!/usr/bin/env python3
"""Generate machine-readable analyses and Chinese Phase 4E-1 final reports."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4e1 import compare,dump,rows,summarize
OUT=ROOT/"outputs/phase4e1_tf_fdg_full_method";DOC=ROOT/"docs/phase4e1";DOC.mkdir(parents=True,exist_ok=True)
ARMS=("full","no_teacher","no_forensic","k1","full_clip")
def load(path):return json.loads(Path(path).read_text())
def write(name,text):(DOC/name).write_text(text.strip()+"\n",encoding="utf-8")
def table_history(history,key):
 lines=["| Epoch | Optimizer updates | Train loss | Validation mean FG IoU | Validation global FG IoU |","|---:|---:|---:|---:|---:|"]
 for r in history:
  train=r.get("train") or {};lines.append(f"| {r['epoch']} | {r.get('global_step',r.get('optimizer_updates',0))} | {train.get('mean_loss',train.get('total','—'))} | {r['validation']['mean_foreground_iou']:.6f} | {r['validation']['global_foreground_iou']:.6f} |")
 return "\n".join(lines)
def strata(full_records,p1_records):
 p1={r["sample_id"]:r for r in p1_records};full={r["sample_id"]:r for r in full_records};prediction={r["sample_id"]:r for r in rows(ROOT/"outputs/phase3f_autonomous_oracle_grounding_distillation/evaluation/selector/P1_FROZEN/G0/predictions.jsonl")};manifest={r["sample_id"]:r for r in rows(ROOT/"outputs/data_audits/unified_forensics_split_v1/val_combined.jsonl") if int(r["class_label"])==1}
 ids=[r["sample_id"] for r in full_records];areas=np.asarray([prediction[s]["gt_mask_area_ratio"] for s in ids]);q=np.quantile(areas,[.25,.5,.75])
 def area(a):return "Q1" if a<=q[0] else ("Q2" if a<=q[1] else ("Q3" if a<=q[2] else "Q4"))
 def diff(i):
  d=np.asarray([full[s]["foreground_iou"]-p1[s]["foreground_iou"] for s in i]);return {"n":len(i),"full_mean_iou":float(np.mean([full[s]["foreground_iou"] for s in i])),"p1_mean_iou":float(np.mean([p1[s]["foreground_iou"] for s in i])),"mean_delta":float(d.mean()),"wins":int((d>1e-12).sum()),"ties":int((abs(d)<=1e-12).sum()),"losses":int((d< -1e-12).sum())}
 dimensions={"mask_area":{},"ref_count":{},"source":{},"category":{},"baseline_difficulty":{}}
 for s in ids:
  tags={"mask_area":area(prediction[s]["gt_mask_area_ratio"]),"ref_count":str(min(5,int(prediction[s]["num_official_refs"]))) if int(prediction[s]["num_official_refs"])<5 else "5+","source":str(manifest[s].get("source","unknown")),"category":str(manifest[s].get("content_category") or manifest[s].get("content_type") or "unknown")}
  b=p1[s]["foreground_iou"];tags["baseline_difficulty"]="zero" if b==0 else ("(0,.1]" if b<=.1 else ("(.1,.3]" if b<=.3 else ("(.3,.5]" if b<=.5 else ">.5")))
  for dimension,tag in tags.items():dimensions[dimension].setdefault(tag,[]).append(s)
 return {dimension:{tag:diff(values) for tag,values in groups.items()} for dimension,groups in dimensions.items()}
def finalize_teacher_failed(summaries,available,teacher_history,teacher_selector,qualification):
 """Finalize the protocol-valid route where Full Stage S is prohibited by its teacher gate."""
 write("phase4e1_stageT_training_report.md",f"""# Phase 4E-1 Stage T 训练报告

Teacher 使用全部 8,836 train Fake，5 个完整 epoch；selector 仅使用 validation Fake canonical TF mean FG IoU。

{table_history(teacher_history,'global_step')}

Selected epoch：**{teacher_selector['selected_epoch']}**。internal test 与 official1000 未使用。""")
 write("phase4e1_teacher_qualification.md",f"""# Phase 4E-1 Teacher qualification

- TEACHER_MASK_CAPABILITY：**{qualification['TEACHER_MASK_CAPABILITY']}**
- TEACHER_IMAGE_SPECIFIC_USE：**{qualification['TEACHER_IMAGE_SPECIFIC_USE']}**
- TEACHER_SPATIAL_SPECIFIC_USE：**{qualification['TEACHER_SPATIAL_SPECIFIC_USE']}**
- Teacher vs P1 TF mean IoU delta：{qualification['teacher_vs_p1_tf_mean_iou_delta']:.6f}
- 实际 KD enable matrix：`{json.dumps(qualification['kd_enable'])}`

Teacher capability gate 失败后，冻结协议禁止启动 Full Stage S；qualification 未参与 teacher 重选。""")
 write("phase4e1_full_student_training_report.md","""# Phase 4E-1 Full Student 训练报告

Full Stage S **未启动**。原因是 Full Teacher 的 validation Fake canonical TF mean FG IoU 相对 P1 TF 的下降超过冻结门槛，触发 `TEACHER_MASK_CAPABILITY=FAILED`。因此不存在 Full Student exposures、optimizer updates 或 checkpoint；这属于协议规定的停止路径，不是运行故障。""")
 write("phase4e1_full_results.md","""# Phase 4E-1 Full results

`FULL_METHOD_GAIN=NOT_EVALUATED_TEACHER_FAILED`。Full Student 未获准训练，因此不得构造或报告 Full Student canonical G0、Phrase-only、TF-full 增益。""")
 write("phase4e1_causal_controls.md",f"""# Phase 4E-1 Causal controls

- TEACHER_IMAGE_SPECIFIC_USE：**{qualification['TEACHER_IMAGE_SPECIFIC_USE']}**
- TEACHER_SPATIAL_SPECIFIC_USE：**{qualification['TEACHER_SPATIAL_SPECIFIC_USE']}**
- Student IMAGE_SPECIFIC_UTILIZATION：**NOT_EVALUATED_TEACHER_FAILED**
- Student SPATIAL_SPECIFIC_UTILIZATION：**NOT_EVALUATED_TEACHER_FAILED**
- FORENSIC_SPECIALIZATION_TRANSFER：**NOT_EVALUATED_FULL_STUDENT_ABSENT**

Teacher control 只证明 Teacher 输出依赖 matched image/position，不能替代端到端 Student transfer 证据。""")
 write("phase4e1_slot_collapse_report.md","""# Phase 4E-1 Slot collapse

Full Student 未训练，故 `SLOT_COLLAPSE=NOT_EVALUATED_TEACHER_FAILED`。K=1 Teacher 的独立结果保留在 Tier-1 报告中。""")
 tier={}
 for arm in ARMS:
  q=OUT/"qualification"/arm/"qualification.json"
  if q.exists():
   value=load(q);tier[arm]={"teacher_mask_capability":value["TEACHER_MASK_CAPABILITY"],"teacher_selected_mean_fg_iou":value["conditions"]["matched"]["mean_foreground_iou"],"teacher_vs_p1_tf_mean_iou_delta":value["teacher_vs_p1_tf_mean_iou_delta"]}
  if arm in summaries:tier.setdefault(arm,{})["student_g0"]=summaries[arm]["matched"]
 dump(OUT/"statistics/tier1_summary.json",tier)
 write("phase4e1_tier1_ablation_report.md",f"""# Phase 4E-1 Tier-1 ablations

Full Teacher gate 失败不删除已预注册的 matched Tier-1 结果。可用结果如下；没有 Student summary 的 teacher-dependent arm 是因各自 teacher gate 失败而按协议停止。

```json
{json.dumps(tier,ensure_ascii=False,indent=2)}
```""")
 complexity={}
 for arm in ARMS:
  p=OUT/"preflight"/f"{arm}_integrated_preflight.json"
  if p.exists():
   pre=load(p);complexity[arm]={k:pre.get(k) for k in ("parameter_count","forward_flops_profiled","profile_batch_size","peak_allocated_bytes_forward_backward")}
 dump(OUT/"complexity_report.json",complexity)
 write("phase4e1_complexity_report.md",f"# Phase 4E-1 Complexity\n\n```json\n{json.dumps(complexity,indent=2)}\n```\n\nFLOPs 为 profiler 对注明 batch size 的 forward 统计；peak memory 为 integrated forward/backward preflight 的 allocated bytes。")
 contribution={"FULL_METHOD_GAIN":"NOT_EVALUATED_TEACHER_FAILED","TEACHER_MASK_CAPABILITY":qualification["TEACHER_MASK_CAPABILITY"],"TEACHER_IMAGE_SPECIFIC_USE":qualification["TEACHER_IMAGE_SPECIFIC_USE"],"TEACHER_SPATIAL_SPECIFIC_USE":qualification["TEACHER_SPATIAL_SPECIFIC_USE"],"FULL_STAGE_S":"NOT_STARTED_TEACHER_FAILED","TEACHER_TRANSFER_CONTRIBUTION":"NOT_EVALUATED_FULL_STUDENT_ABSENT","FORENSIC_BRANCH_CONTRIBUTION":"NOT_EVALUATED_FULL_STUDENT_ABSENT","MULTI_QUERY_CONTRIBUTION":"NOT_EVALUATED_FULL_STUDENT_ABSENT","FORENSIC_SPECIALIZATION_TRANSFER":"NOT_EVALUATED_FULL_STUDENT_ABSENT","SLOT_COLLAPSE_STATE":"NOT_EVALUATED_TEACHER_FAILED"}
 dump(OUT/"final_claims.json",contribution)
 write("phase4e1_final_method_report.md",f"""# Phase 4E-1 最终方法报告

Phase 4E-1 触发冻结协议规定的 Teacher capability 停止门：Full Teacher 虽显示 image-specific 与 spatial-specific 使用，但绝对 TF mask capability 未达到 P1 门槛，因此 Full Stage S 未启动，Full method gain 不可评估。

```json
{json.dumps(contribution,ensure_ascii=False,indent=2)}
```

已授权的 Tier-1 arms 仍按 matched 协议完成或触发各自门控。internal test 与 official1000 继续封存，未自动进入 held-out evaluation 或下一 Phase。""")
 dump(OUT/"completion_manifest.json",{"status":"COMPLETE_PROTOCOL_STOP","phase":"Phase 4E-1","stop_reason":"TEACHER_MASK_CAPABILITY_FAILED","arms_with_student_results":available,"reports":[str(p) for p in sorted(DOC.glob("phase4e1_*.md"))],"internal_test_accessed":False,"official1000_accessed":False,"heldout_evaluation_started":False,"next_phase_started":False})
 print(json.dumps({"status":"COMPLETE_PROTOCOL_STOP","claims":contribution},ensure_ascii=False,indent=2))
def main():
 summaries={};available=[]
 for arm in ARMS:
  p=OUT/"final"/arm/"summary.json"
  if p.exists():summaries[arm]=load(p);available.append(arm)
 teacher_history=load(OUT/"training/full/teacher_history.json");teacher_selector=load(OUT/"selectors/full_teacher.json");qualification=load(OUT/"qualification/full/qualification.json")
 if "full" not in summaries:
  if qualification["TEACHER_MASK_CAPABILITY"] != "FAILED":raise RuntimeError("Full final evaluation missing without a failed teacher gate")
  finalize_teacher_failed(summaries,available,teacher_history,teacher_selector,qualification);return
 full=summaries["full"];student_history=load(OUT/"training/full/student_history.json");student_selector=load(OUT/"selectors/full_student.json")
 p1=[]
 for path in sorted((ROOT/"outputs/phase4c_b_evidence_reader/cache/validation/G0").glob("shard_*.pt")):
  import torch;p1 += torch.load(path,map_location="cpu",weights_only=False)["records"]
 p1=[{"sample_id":r["sample_id"],"foreground_iou":r["foreground_iou"],"foreground_f1":r["foreground_f1"]} for r in p1]
 full_records=rows(OUT/"final/full/matched.jsonl");full_vs_p1=compare(full_records,p1);stratified=strata(full_records,p1);dump(OUT/"statistics/full_vs_p1.json",full_vs_p1);dump(OUT/"statistics/full_stratification.json",stratified)
 tier={arm:{"selected_epoch":load(OUT/"selectors"/f"{arm}_student.json")["selected_epoch"],"g0":summaries[arm]["matched"]} for arm in available}
 if "full_clip" in summaries:tier["forensic_vs_clip"]=compare(full_records,rows(OUT/"final/full_clip/matched.jsonl"))
 dump(OUT/"statistics/tier1_summary.json",tier)
 ci=full_vs_p1["foreground_iou"]["bootstrap_95_ci"];full_gain="TRUE" if ci[0]>0 else ("FALSE" if ci[1]<=0 else "INCONCLUSIVE")
 image_ci=full["matched_vs_cross"]["foreground_iou"]["bootstrap_95_ci"];spatial_ci=full["matched_vs_spatial_shuffle"]["foreground_iou"]["bootstrap_95_ci"]
 image="TRUE" if image_ci[0]>0 else "FALSE";spatial="TRUE" if spatial_ci[0]>0 else "FALSE"
 specialization="NOT_AVAILABLE" if "full_clip" not in summaries else ("TRUE" if tier["forensic_vs_clip"]["foreground_iou"]["bootstrap_95_ci"][0]>0 else "FALSE")
 write("phase4e1_stageT_training_report.md",f"""# Phase 4E-1 Stage T 训练报告\n\nTeacher 使用全部 8,836 train Fake，5 个完整 epoch；selector 仅使用 validation Fake canonical TF mean FG IoU。\n\n{table_history(teacher_history,'global_step')}\n\nSelected epoch：**{teacher_selector['selected_epoch']}**。internal test 与 official1000 未使用。""")
 write("phase4e1_teacher_qualification.md",f"""# Phase 4E-1 Teacher qualification\n\n- TEACHER_MASK_CAPABILITY：**{qualification['TEACHER_MASK_CAPABILITY']}**\n- TEACHER_IMAGE_SPECIFIC_USE：**{qualification['TEACHER_IMAGE_SPECIFIC_USE']}**\n- TEACHER_SPATIAL_SPECIFIC_USE：**{qualification['TEACHER_SPATIAL_SPECIFIC_USE']}**\n- Teacher vs P1 TF mean IoU delta：{qualification['teacher_vs_p1_tf_mean_iou_delta']:.6f}\n- 实际 KD enable matrix：`{json.dumps(qualification['kd_enable'])}`\n\nqualification 未参与 teacher 重选。""")
 write("phase4e1_full_student_training_report.md",f"""# Phase 4E-1 Full Student 训练报告\n\nStage S 遍历 exposures=88,360；optimization-eligible valid-G0 exposures=86,900。loss 仅按 batch 内 valid-G0 数归一化。\n\n{table_history(student_history,'optimizer_updates')}\n\nSelected epoch：**{student_selector['selected_epoch']}**；正式 selector population 为全部 1,106 validation Fake。""")
 write("phase4e1_full_results.md",f"""# Phase 4E-1 Full results\n\n| Condition | N | Mean FG IoU | Global FG IoU | Mean FG F1 |\n|---|---:|---:|---:|---:|\n| canonical G0 | {full['matched']['n']} | {full['matched']['mean_foreground_iou']:.6f} | {full['matched']['global_foreground_iou']:.6f} | {full['matched']['mean_foreground_f1']:.6f} |\n| Phrase-only | {full['phrase_only']['n']} | {full['phrase_only']['mean_foreground_iou']:.6f} | {full['phrase_only']['global_foreground_iou']:.6f} | {full['phrase_only']['mean_foreground_f1']:.6f} |\n| TF-full | {full['tf_full']['n']} | {full['tf_full']['mean_foreground_iou']:.6f} | {full['tf_full']['global_foreground_iou']:.6f} | {full['tf_full']['mean_foreground_f1']:.6f} |\n\nFULL_METHOD_GAIN：**{full_gain}**。VALID-G0-ONLY 仅为 diagnostic，不能替代 1,106-sample headline G0。详细 paired/Wilcoxon 与分层结果见 machine-readable statistics。""")
 write("phase4e1_causal_controls.md",f"""# Phase 4E-1 Causal controls\n\n- IMAGE_SPECIFIC_UTILIZATION：**{image}**，matched-cross CI={image_ci}\n- SPATIAL_SPECIFIC_UTILIZATION：**{spatial}**，matched-shuffle CI={spatial_ci}\n- FORENSIC_SPECIALIZATION_TRANSFER：**{specialization}**\n\n所有 control 均在 selected checkpoint 冻结后执行，不参与 selector。""")
 collapse=full.get("slot_collapse",{"SLOT_COLLAPSE":"NOT_AVAILABLE"});write("phase4e1_slot_collapse_report.md",f"# Phase 4E-1 Slot collapse\n\nSLOT_COLLAPSE：**{collapse['SLOT_COLLAPSE']}**。\n\n该诊断不单独用于否定 multi-query hypothesis；K=1 matched arm 同时报告。")
 write("phase4e1_tier1_ablation_report.md",f"# Phase 4E-1 Tier-1 ablations\n\n已完成 arms：{', '.join(available)}。\n\n```json\n{json.dumps(tier,ensure_ascii=False,indent=2)}\n```")
 complexity={}
 for arm in available:
  pre=load(OUT/"preflight"/f"{arm}_integrated_preflight.json");complexity[arm]={k:pre.get(k) for k in ("parameter_count","forward_flops_profiled","profile_batch_size","peak_allocated_bytes_forward_backward")}
 dump(OUT/"complexity_report.json",complexity);write("phase4e1_complexity_report.md",f"# Phase 4E-1 Complexity\n\n```json\n{json.dumps(complexity,indent=2)}\n```\n\nFLOPs 为 profiler 对注明 batch size 的 forward 统计；peak memory 为 integrated forward/backward preflight 的 allocated bytes。")
 contribution={"FULL_METHOD_GAIN":full_gain,"TEACHER_TRANSFER_CONTRIBUTION":"AVAILABLE" if "no_teacher" in summaries else "NOT_AVAILABLE","FORENSIC_BRANCH_CONTRIBUTION":"AVAILABLE" if "no_forensic" in summaries else "NOT_AVAILABLE","MULTI_QUERY_CONTRIBUTION":"AVAILABLE" if "k1" in summaries else "NOT_AVAILABLE","IMAGE_SPECIFIC_UTILIZATION":image,"SPATIAL_SPECIFIC_UTILIZATION":spatial,"FORENSIC_SPECIALIZATION_TRANSFER":specialization,"SLOT_COLLAPSE_STATE":collapse["SLOT_COLLAPSE"]};dump(OUT/"final_claims.json",contribution)
 write("phase4e1_final_method_report.md",f"""# Phase 4E-1 最终方法报告\n\nPhase 4E-1 按冻结协议与 valid-G0 conditional amendment 完成。\n\n```json\n{json.dumps(contribution,ensure_ascii=False,indent=2)}\n```\n\n正式 canonical G0 保留 28 个无 `[SEG]` validation failure；Student 训练没有为 146 个 invalid train sample 构造 surrogate hidden。internal test 与 official1000 继续封存，未自动进入 held-out evaluation 或下一 Phase。""")
 dump(OUT/"completion_manifest.json",{"status":"COMPLETE","phase":"Phase 4E-1","arms_completed":available,"reports":[str(p) for p in sorted(DOC.glob("phase4e1_*.md"))],"internal_test_accessed":False,"official1000_accessed":False,"heldout_evaluation_started":False,"next_phase_started":False})
 print(json.dumps({"status":"COMPLETE","claims":contribution},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
