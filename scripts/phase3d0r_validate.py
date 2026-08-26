#!/usr/bin/env python3
"""One-shot full-validation confirmation for the frozen Phase 3D.0-R scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from scripts.phase3d0_analyze import paired
from tools.phase3b_replay import file_sha256
from tools.phase3d0 import normalize_output
from tools.phase3d0r import FrozenSentenceEncoder, choose_top, content_phrase, content_tokens, grounding_components, new_rewards, semantic_phrase_score

def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def dump(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def write_rows(path,values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8") as h:
  for x in values:h.write(json.dumps(x,ensure_ascii=False)+"\n")
def mean(values):return float(np.mean(values)) if values else None

def embedding_inputs(pairs):
 values=set()
 for a,b in pairs:
  for phrase in (a,b):values.add(content_phrase(phrase));values.update(content_tokens(phrase))
 return values

def enrich(row,reference,idf,embeddings,ground=None):
 value=dict(row);fake=int(value["gt_class"])==1
 if fake:value.update(semantic_phrase_score(reference,value.get("normalized_phrase"),idf,embeddings))
 else:value.update({"R_phrase_sem":0.0,"R_key_soft":0.0,"R_sentence_sem":0.0,"sentence_cosine_raw":None,
   "P_spatial_contra":0.0,"P_neg_contra":0.0,"NEGATION_MISMATCH":False})
 if ground is not None:value.update(ground)
 elif fake:value.update({"R_ground_rel":None,"C_ground":None,"R_rank_mask":None})
 else:value.update({"R_ground_rel":0.0,"C_ground":0.0,"R_rank_mask":0.0})
 value["OLD_R3"]=float(value["R3"]);value.update(new_rewards(value["gt_class"],value));return value

def top_summary(groups,reward):
 selected=[choose_top(x["rollouts"],reward) for x in groups];fake=[x for x in selected if x["gt_class"]==1]
 return {"count":len(selected),"overall_generated_verdict_accuracy":mean([x["R_cls"] for x in selected]),
  "structural_validity_rate":mean([x["R_struct"] for x in selected]),"fake_phrase_lex":mean([x["R_phrase_lex"] for x in fake]),
  "fake_phrase_sem":mean([x["R_phrase_sem"] for x in fake]),"fake_key_soft":mean([x["R_key_soft"] for x in fake]),
  "fake_sentence_sem":mean([x["R_sentence_sem"] for x in fake]),"fake_spatial_contradiction_rate":mean([x["P_spatial_contra"] for x in fake]),
  "fake_absolute_iou":mean([x["R_mask"] for x in fake]),"fake_foreground_f1":mean([x["foreground_f1"] for x in fake]),
  "fake_ground_rel":mean([x["R_ground_rel"] for x in fake])}

def ranks(records,reward):
 order=sorted(records,key=lambda x:(-float(x[reward]),int(x["rollout_index"])))
 return {x["rollout_index"]:i+1 for i,x in enumerate(order)}

def main():
 cfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text());out=(ROOT/cfg["experiment"]["output_root"]).resolve();source=(ROOT/cfg["experiment"]["source_root"]).resolve();manifest=(ROOT/cfg["experiment"]["manifest_dir"]).resolve()
 selected_info=load(out/"selected_reward.json");selected_reward=selected_info["selected_reward"]
 if selected_info["status"]!="FROZEN_BEFORE_FULL_VALIDATION" or selected_reward not in {"Q1","Q2"}:raise RuntimeError("candidate not frozen")
 idf=load(out/"token_idf.json")["values"];val=rows(manifest/"val_combined.jsonl");val_by_id={x["sample_id"]:x for x in val}
 refs={x["sample_id"]:UnifiedForensicsDataset.authoritative_localization_field(x)["normalized_training_phrase"] for x in val if int(x["class_label"])==1}
 raw_groups=[x for i in range(2) for x in rows(source/f"rollouts/val_A_full_shard{i:02d}_of_02.jsonl")]
 raw_greedy=[x for i in range(2) for x in rows(source/f"greedy/val_shard{i:02d}_of_02.jsonl")]
 if len(raw_groups)!=2212 or len(raw_greedy)!=2212 or any(len(x["rollouts"])!=8 for x in raw_groups):raise RuntimeError("validation source incomplete")
 pairs=[]
 for group in raw_groups:
  if group["sample_id"] in refs:pairs.extend((refs[group["sample_id"]],x.get("normalized_phrase")) for x in group["rollouts"])
 for x in raw_greedy:
  if x["sample_id"] in refs:pairs.append((refs[x["sample_id"]],x.get("normalized_phrase")))
 enc=cfg["semantic_encoder"];encoder=FrozenSentenceEncoder(enc["model_id"],enc["revision"],enc["cache_dir"],"cpu");embeddings=encoder.encode(embedding_inputs(pairs),128)
 groups=[];component_rows=[];ground_rows=[]
 for group in raw_groups:
  fake=group["sample_id"] in refs;grounds=grounding_components([float(x["R_mask"]) for x in group["rollouts"]]) if fake else [None]*8
  enriched=[enrich(x,refs.get(group["sample_id"]),idf,embeddings,grounds[i]) for i,x in enumerate(group["rollouts"])]
  groups.append({**{k:v for k,v in group.items() if k!="rollouts"},"rollouts":enriched})
  component_rows.extend({k:x.get(k) for k in ("sample_id","rollout_index","gt_class","normalized_phrase","R_phrase_lex","R_key_soft","R_sentence_sem","sentence_cosine_raw","R_phrase_sem","P_spatial_contra","P_neg_contra","NEGATION_MISMATCH","R_mask","C_ground","R_ground_rel","OLD_R3","Q1","Q2","Q3")} for x in enriched)
  if fake:ground_rows.append({"sample_id":group["sample_id"],**{k:enriched[0][k] for k in ("mask_group_min","mask_group_max","mask_group_range","C_range","C_ceiling","C_ground")}})
 greedy=[enrich(x,refs.get(x["sample_id"]),idf,embeddings,None) for x in raw_greedy];greedy_by_id={x["sample_id"]:x for x in greedy}
 write_rows(out/"validation/scored_groups.jsonl",groups);write_rows(out/"validation/scored_greedy.jsonl",greedy)
 write_rows(out/"phrase_semantic_components.jsonl",component_rows);write_rows(out/"reward_components/phrase_semantic_components.jsonl",component_rows)
 write_rows(out/"grounding_controllability.jsonl",ground_rows);write_rows(out/"group_grounding/grounding_controllability.jsonl",ground_rows)
 comparison={};paired_all={}
 for reward in ("OLD_R3","Q1","Q2","Q3"):
  selected=[dict(choose_top(x["rollouts"],reward),selected_reward=reward) for x in groups];write_rows(out/f"reward_candidates/top_selected_{reward}.jsonl",selected)
  comparison[reward]=top_summary(groups,reward);pairs_by_metric={}
  for key in ("R_cls","R_struct","R_phrase_lex","R_phrase_sem","R_key_soft","R_sentence_sem","P_spatial_contra","R_mask","foreground_f1"):
   pairs2=[(x[key],greedy_by_id[x["sample_id"]][key]) for x in selected if x.get(key) is not None and greedy_by_id[x["sample_id"]].get(key) is not None]
   pairs_by_metric[key]=paired([a for a,b in pairs2],[b for a,b in pairs2])
  paired_all[reward]=pairs_by_metric
 base={"count":len(greedy),"overall_generated_verdict_accuracy":mean([x["R_cls"] for x in greedy]),"structural_validity_rate":mean([x["R_struct"] for x in greedy]),
       "fake_phrase_lex":mean([x["R_phrase_lex"] for x in greedy if x["gt_class"]==1]),"fake_phrase_sem":mean([x["R_phrase_sem"] for x in greedy if x["gt_class"]==1]),
       "fake_key_soft":mean([x["R_key_soft"] for x in greedy if x["gt_class"]==1]),"fake_sentence_sem":mean([x["R_sentence_sem"] for x in greedy if x["gt_class"]==1]),
       "fake_spatial_contradiction_rate":mean([x["P_spatial_contra"] for x in greedy if x["gt_class"]==1]),"fake_absolute_iou":mean([x["R_mask"] for x in greedy if x["gt_class"]==1]),
       "fake_foreground_f1":mean([x["foreground_f1"] for x in greedy if x["gt_class"]==1])}
 validation={"status":"CONFIRMATION_COMPLETE_NO_RETUNING","selected_reward":selected_reward,"greedy":base,"top_selected":comparison,"paired_top_vs_greedy":paired_all,
             "validation_used_to_change_formula":False}
 dump(out/"validation_comparison.json",validation);dump(out/"validation/validation_comparison.json",validation)
 all_rollouts=[x for g in groups for x in g["rollouts"]];fake_rollouts=[x for x in all_rollouts if x["gt_class"]==1]
 subgroup_defs={"LEXICAL_LOW_SEMANTIC_HIGH":lambda x:x["R_phrase_lex"]<.3 and x["R_phrase_sem"]>=.7,
  "LEXICAL_HIGH_SEMANTIC_LOW":lambda x:x["R_phrase_lex"]>=.7 and x["R_phrase_sem"]<.5,
  "SPATIAL_CONTRADICTION":lambda x:bool(x["P_spatial_contra"]),"PHRASE_GOOD_MASK_POOR":lambda x:x["R_phrase_sem"]>=.7 and x["R_mask"]<=.2,
  "PHRASE_GOOD_GROUND_RESPONSIVE":lambda x:x["R_phrase_sem"]>=.7 and x["C_ground"]>=.75}
 subgroup={}
 for name,pred in subgroup_defs.items():
  subset=[x for x in fake_rollouts if pred(x)];subgroup[name]={"trajectory_count":len(subset),"group_count":len({x["sample_id"] for x in subset}),
   "mean_lex":mean([x["R_phrase_lex"] for x in subset]),"mean_sem":mean([x["R_phrase_sem"] for x in subset]),"mean_mask":mean([x["R_mask"] for x in subset]),
   "OLD_R3_selected_count":sum(pred(choose_top(g["rollouts"],"OLD_R3")) for g in groups if g["sample_id"] in refs),
   "selected_new_count":sum(pred(choose_top(g["rollouts"],selected_reward)) for g in groups if g["sample_id"] in refs)}
 dump(out/"subgroup_statistics.json",subgroup);dump(out/"subgroups/subgroup_statistics.json",subgroup)
 failures={"DOUBLE_PENALTY_CASE":[],"FUNCTION_WORD_INFLATION":[],"FUNCTION_WORD_INFLATION_BROAD_PROXY":[],"PARAPHRASE_UNDERSCORED":[],"SPATIAL_CONTRADICTION_OVERREWARDED":[]}
 for group in groups:
  if group["sample_id"] not in refs:continue
  oldrank=ranks(group["rollouts"],"OLD_R3");newrank=ranks(group["rollouts"],selected_reward)
  for x in group["rollouts"]:
   detail={"sample_id":x["sample_id"],"rollout_index":x["rollout_index"],"phrase":x["normalized_phrase"],"R_phrase_lex":x["R_phrase_lex"],"R_phrase_sem":x["R_phrase_sem"],"R_key_soft":x["R_key_soft"],"R_mask":x["R_mask"],"C_ground":x["C_ground"],"OLD_R3":x["OLD_R3"],selected_reward:x[selected_reward]}
   if x["R_phrase_sem"]>=.7 and x["R_mask"]<=.2 and oldrank[x["rollout_index"]]-newrank[x["rollout_index"]]>=2:failures["DOUBLE_PENALTY_CASE"].append(detail)
   if x["R_phrase_lex"]>=.7 and x["R_key_soft"]<.5:failures["FUNCTION_WORD_INFLATION"].append(detail)
   if x["R_phrase_lex"]>=.7 and x["R_phrase_sem"]<.5:failures["FUNCTION_WORD_INFLATION_BROAD_PROXY"].append(detail)
   if x["R_phrase_lex"]<.3 and x["R_phrase_sem"]>=.7:failures["PARAPHRASE_UNDERSCORED"].append(detail)
   if x["P_spatial_contra"] and x["R_phrase_lex"]>=.5:failures["SPATIAL_CONTRADICTION_OVERREWARDED"].append(detail)
 failure_payload={name:{"count":len(values),"examples":sorted(values,key=lambda x:(x["sample_id"],x["rollout_index"]))[:50]} for name,values in failures.items()}
 dump(out/"old_r3_failure_audit.json",failure_payload)
 selected_rows=[choose_top(g["rollouts"],selected_reward) for g in groups];fake_selected=[x for x in selected_rows if x["gt_class"]==1]
 fake_groups=[g for g in groups if g["sample_id"] in refs];std_rate=mean([np.std([x[selected_reward] for x in g["rollouts"]])>.05 for g in fake_groups])
 distinct=mean([len({normalize_output(x["decoded_text"]) for x in g["rollouts"]})>=2 for g in groups])
 low_groups=[g for g in fake_groups if g["rollouts"][0]["C_ground"]<.25]
 low_selected=[choose_top(g["rollouts"],selected_reward) for g in low_groups];low_delta=mean([x["R_phrase_sem"]-greedy_by_id[x["sample_id"]]["R_phrase_sem"] for x in low_selected])
 pair=paired_all[selected_reward];conditions={"groups_multiple_unique":distinct>=.5,"fake_reward_std":std_rate>=.5,
  "classification_nonreg":comparison[selected_reward]["overall_generated_verdict_accuracy"]>=base["overall_generated_verdict_accuracy"]-.005,
  "structure_nonreg":comparison[selected_reward]["structural_validity_rate"]>=base["structural_validity_rate"]-.005,
  "phrase_sem_positive_ci":pair["R_phrase_sem"]["bootstrap_95ci"][0]>0,"no_semantic_phrase_collapse":pair["R_phrase_sem"]["mean_difference"]>=-.005,
  "spatial_contradiction_not_worse":comparison[selected_reward]["fake_spatial_contradiction_rate"]<=base["fake_spatial_contradiction_rate"]+1e-12,
  "low_controllability_phrase_nonreg":low_delta>=-.02}
 ready=all(conditions.values());signal={"selected_reward":selected_reward,"conditions":conditions,"fraction_groups_multiple_unique":distinct,
  "fraction_fake_groups_reward_std_gt_0.05":std_rate,"low_controllability_phrase_delta_vs_greedy":low_delta,"REFORMULATED_POLICY_SIGNAL_READY":ready}
 dump(out/"reformulated_policy_signal.json",signal)
 sanity=load(out/"semantic_sanity_statistics.json")
 if not sanity["semantic_encoder_sanity_pass"]:gate="GATE_SEMANTIC_SCORER_NOT_SUPPORTED"
 elif selected_reward=="Q2" and ready:gate="GATE_REFORMULATED_REWARD_SUPPORTED"
 elif selected_reward=="Q1" and ready:gate="GATE_LANGUAGE_ONLY_REWARD_SUPPORTED"
 else:gate="GATE_REWARD_REFORMULATION_INCONCLUSIVE"
 route={"primary_gate":gate,"selected_reward":selected_reward,"REFORMULATED_POLICY_SIGNAL_READY":ready,
  "phase3d1_controlled_policy_optimization_authorized":gate in {"GATE_REFORMULATED_REWARD_SUPPORTED","GATE_LANGUAGE_ONLY_REWARD_SUPPORTED"},
  "phase3d1_started":False,"OLD_R3":"HISTORICAL_REWARD_CONTROL","R_phrase_lex":"LEXICAL_DIAGNOSTIC_ONLY","OLD_R_ALIGN_DIAGNOSTIC_ONLY":True,
  "R_phrase_sem_is_automatic_frozen_semantic_proxy_not_human_ground_truth":True,"validation_retuning":False}
 dump(out/"route_gate.json",route)
 audit={"status":"PASS","no_new_validation_rollouts":True,"validation_source_files":[{"path":str(source/f"rollouts/val_A_full_shard{i:02d}_of_02.jsonl"),"sha256":file_sha256(source/f"rollouts/val_A_full_shard{i:02d}_of_02.jsonl")} for i in range(2)],
  "exact_source_mask_paths_reused":all(x.get("binary_mask_path") is None or str(x["binary_mask_path"]).startswith(str(source)) for x in all_rollouts),
  "semantic_encoder_frozen":True,"formula_frozen_before_validation":True,"stopwords_frozen":True,"idf_train_only":True,"test_used":False,"phase3d1_started":False}
 dump(out/"audit/validation_invariance.json",audit);print(json.dumps(route,indent=2))
 dump(out/"audit/reporting_proxy_clarification.json",{
  "status":"REPORTING_ONLY_NO_REWARD_OR_GATE_CHANGE",
  "strict_function_word_proxy":"R_phrase_lex>=0.70 and R_key_soft<0.50",
  "broad_display_proxy":"preregistered lexical-high/semantic-low: R_phrase_lex>=0.70 and R_phrase_sem<0.50",
  "reason":"the qualitative page must expose the preregistered failure family even when the stricter key-soft subset is empty",
  "selected_reward_changed":False,"formula_or_threshold_changed":False,"route_gate_changed":False})

if __name__=="__main__":main()
