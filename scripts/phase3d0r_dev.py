#!/usr/bin/env python3
"""Reward-dev semantic sanity, candidate comparison, and pre-validation freeze."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3b_replay import file_sha256, phrase_overlap
from tools.phase3d0 import normalize_output
from tools.phase3d0r import (CURATED_SYNONYMS, FrozenSentenceEncoder, SPATIAL_PAIRS, choose_top,
    content_phrase, content_tokens, grounding_components, new_rewards, semantic_phrase_score)


def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def write_rows(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as h:
        for value in values:h.write(json.dumps(value,ensure_ascii=False)+"\n")
def mean(values): return float(np.mean(values)) if values else None
def rate(values): return float(np.mean(values)) if values else None


def replace_token(phrase, old, new):
    return " ".join(new if token == old else token for token in content_tokens(phrase))


def all_embedding_inputs(pairs):
    values=set()
    for left,right in pairs:
        for phrase in (left,right):
            values.add(content_phrase(phrase));values.update(content_tokens(phrase))
    return values


def sanity_pairs(dev_refs, idf):
    identity=[(x,x) for x in dev_refs]
    case=[(x,x.upper()+" !!!") for x in dev_refs]
    stop=[(x,"the region of "+x+" area") for x in dev_refs]
    spatial=[]
    for phrase in dev_refs:
        tokens=set(content_tokens(phrase))
        for a,b in SPATIAL_PAIRS:
            if a in tokens and b not in tokens:spatial.append((phrase,replace_token(phrase,a,b)));break
            if b in tokens and a not in tokens:spatial.append((phrase,replace_token(phrase,b,a)));break
    top=sorted(idf,key=lambda x:(-idf[x],x))[:max(20,len(idf)//10)]
    unrelated=[]
    for phrase in dev_refs:
        candidates=[x for x in content_tokens(phrase) if x not in {y for pair in SPATIAL_PAIRS for y in pair}]
        if not candidates:continue
        old=candidates[0];rank=int(hashlib.sha256(phrase.encode()).hexdigest()[:8],16)
        options=[x for x in top if x!=old]
        if options:unrelated.append((phrase,replace_token(phrase,old,options[rank%len(options)])))
    synonyms=[]
    for phrase in dev_refs:
        tokens=set(content_tokens(phrase))
        for a,b in CURATED_SYNONYMS:
            if a in tokens:synonyms.append((phrase,replace_token(phrase,a,b)));break
            if b in tokens:synonyms.append((phrase,replace_token(phrase,b,a)));break
    return {"identity":identity,"case_punctuation":case,"stopword":stop,"spatial_contradiction":spatial,
            "unrelated_keyword":unrelated[:512],"curated_synonym":synonyms}


def summarize_top(groups,reward):
    selected=[choose_top(x["rollouts"],reward) for x in groups]
    fake=[x for x in selected if x["gt_class"]==1]
    return {"groups":len(groups),"lm_verdict_accuracy":mean([x["R_cls"] for x in selected]),
      "structure_validity":mean([x["R_struct"] for x in selected]),"fake_phrase_sem":mean([x["R_phrase_sem"] for x in fake]),
      "fake_phrase_lex":mean([x["R_phrase_lex"] for x in fake]),"fake_absolute_iou":mean([x["R_mask"] for x in fake]),
      "fake_ground_rel":mean([x["R_ground_rel"] for x in fake]),"spatial_contradiction_rate_fake":mean([x["P_spatial_contra"] for x in fake])}


def subgroup_top(groups,reward,predicate,key):
    subset=[x for x in groups if x["rollouts"][0]["gt_class"]==1 and predicate(x["rollouts"][0])]
    selected=[choose_top(x["rollouts"],reward) for x in subset]
    return {"groups":len(subset),"mean":mean([x[key] for x in selected])}


def failure_counts(groups,reward):
    fake_groups=[x for x in groups if x["rollouts"][0]["gt_class"]==1]
    selected=[choose_top(x["rollouts"],reward) for x in fake_groups]
    punished=0
    for group in fake_groups:
        records=group["rollouts"]
        semantic_best=max(records,key=lambda row:(float(row["R_phrase_sem"]),-int(row["rollout_index"])))
        order=sorted(records,key=lambda row:(-float(row[reward]),int(row["rollout_index"])))
        rank={row["rollout_index"]:index+1 for index,row in enumerate(order)}
        punished += bool(semantic_best["R_phrase_sem"]>=.7 and semantic_best["R_mask"]<=.2
                         and rank[semantic_best["rollout_index"]]>=3)
    values={"LEXICAL_HIGH_SEMANTIC_LOW":sum(x["R_phrase_lex"]>=.7 and x["R_phrase_sem"]<.5 for x in selected),
      "PHRASE_GOOD_MASK_CEILING_PUNISHED":punished,
      "SPATIAL_CONTRADICTION_SELECTED":sum(bool(x["P_spatial_contra"]) for x in selected)}
    values["aggregate"]=sum(values.values());return values


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase3d0r_reward_reformulation.yaml").read_text());out=(ROOT/cfg["experiment"]["output_root"]).resolve()
    source=(ROOT/cfg["experiment"]["source_root"]).resolve();manifest=(ROOT/cfg["experiment"]["manifest_dir"]).resolve()
    idf=load(out/"token_idf.json")["values"]
    train=rows(manifest/"train_combined.jsonl");train_by_id={x["sample_id"]:x for x in train}
    groups=rows(source/"rollouts/dev_A_text_shard00_of_01.jsonl")
    replay=[x for i in range(2) for x in rows(out/f"reward_dev/mask_replay_fake_A_shard{i:02d}_of_02.jsonl")]
    if len(groups)!=1024 or len(replay)!=512:raise RuntimeError(f"reward-dev incomplete text={len(groups)} masks={len(replay)}")
    replay_by_id={x["sample_id"]:x for x in replay};dev_manifest=load(source/"manifests/reward_dev_manifest.json")
    refs={x["sample_id"]:UnifiedForensicsDataset.authoritative_localization_field(train_by_id[x["sample_id"]])["normalized_training_phrase"]
          for x in dev_manifest["records"] if int(x["class_label"])==1}
    pairs=[]
    for group in groups:
        if group["sample_id"] in refs:
            pairs.extend((refs[group["sample_id"]],r["normalized_phrase"]) for r in group["rollouts"])
    sanity=sanity_pairs(list(refs.values()),idf)
    for value in sanity.values():pairs.extend(value)
    enc_cfg=cfg["semantic_encoder"];encoder=FrozenSentenceEncoder(enc_cfg["model_id"],enc_cfg["revision"],enc_cfg["cache_dir"],"cpu")
    embeddings=encoder.encode(all_embedding_inputs(pairs),batch_size=128)
    scored=[];phrase_rows=[];ground_rows=[]
    for group in groups:
        fake=group["sample_id"] in refs;mask_group=replay_by_id.get(group["sample_id"])
        mask_by_index={} if mask_group is None else {x["rollout_index"]:x for x in mask_group["rollouts"]}
        grounding=None if not fake else grounding_components([float(mask_by_index[i]["R_mask"]) for i in range(8)])
        enriched=[]
        for index,original in enumerate(group["rollouts"]):
            row=dict(original);row["gt_class"]=int(row["gt_class"]);row["R_cls"]=float((row["verdict"]=="FAKE") if fake else (row["verdict"]=="REAL"));row["R_struct"]=float(row["structural_validity"])
            if fake:
                mask=mask_by_index[index];row.update({k:mask[k] for k in ("R_mask","foreground_iou","foreground_f1","fg_bg_miou","binary_mask_path","mask_logits_path")})
                row["R_phrase_lex"]=float(phrase_overlap(refs[group["sample_id"]],row["normalized_phrase"])["normalized_token_f1"])
                row.update(semantic_phrase_score(refs[group["sample_id"]],row["normalized_phrase"],idf,embeddings));row.update(grounding[index]);row["OLD_R3"]=float(mask["R3"])
            else:
                row.update({"R_mask":0.0,"foreground_iou":None,"foreground_f1":None,"fg_bg_miou":None,"binary_mask_path":None,"mask_logits_path":None,
                    "R_phrase_lex":0.0,"R_phrase_sem":0.0,"R_key_soft":0.0,"R_sentence_sem":0.0,"sentence_cosine_raw":None,
                    "P_spatial_contra":0.0,"P_neg_contra":0.0,"NEGATION_MISMATCH":False,"C_ground":0.0,"R_ground_rel":0.0,
                    "R_rank_mask":0.0,"C_range":0.0,"C_ceiling":0.0,"mask_group_min":0.0,"mask_group_max":0.0,"mask_group_range":0.0,
                    "OLD_R3":.7*row["R_cls"]+.3*row["R_struct"]})
            row.update(new_rewards(row["gt_class"],row));enriched.append(row)
            phrase_rows.append({k:row.get(k) for k in ("sample_id","rollout_index","gt_class","normalized_phrase","R_phrase_lex","R_key_soft","R_sentence_sem","sentence_cosine_raw","R_phrase_sem","P_spatial_contra","P_neg_contra","NEGATION_MISMATCH")})
        scored.append({**{k:v for k,v in group.items() if k!="rollouts"},"rollouts":enriched})
        if fake:ground_rows.append({"sample_id":group["sample_id"],**{k:enriched[0][k] for k in ("mask_group_min","mask_group_max","mask_group_range","C_range","C_ceiling","C_ground")}})
    write_rows(out/"reward_dev/scored_groups.jsonl",scored);write_rows(out/"reward_dev/phrase_semantic_components.jsonl",phrase_rows)
    write_rows(out/"reward_dev/grounding_controllability.jsonl",ground_rows)
    stats={}
    originals={}
    for name,values in sanity.items():
        base=[semantic_phrase_score(a,a,idf,embeddings)["R_phrase_sem"] for a,_ in values]
        changed=[semantic_phrase_score(a,b,idf,embeddings)["R_phrase_sem"] for a,b in values]
        drop=np.asarray(base)-np.asarray(changed) if values else np.asarray([]);originals[name]=drop
        stats[name]={"n":len(values),"mean_original":mean(base),"mean_changed":mean(changed),"mean_drop":mean(drop.tolist()),"median_drop":float(np.median(drop)) if len(drop) else None}
    stats["identity"]["pass_rate_score_ge_0.98"]=rate([semantic_phrase_score(a,b,idf,embeddings)["R_phrase_sem"]>=.98 for a,b in sanity["identity"]])
    stats["case_punctuation"]["pass_rate_drop_le_0.02"]=rate([x<=.02+1e-9 for x in originals["case_punctuation"]])
    stats["stopword"]["pass_rate_abs_change_le_0.05"]=rate([abs(x)<=.05+1e-9 for x in originals["stopword"]])
    stats["spatial_contradiction"]["pass_rate_drop_ge_0.20"]=rate([x>=.20-1e-9 for x in originals["spatial_contradiction"]])
    synonym_ok=bool(len(originals["curated_synonym"]) and mean(originals["curated_synonym"].tolist())<mean(originals["unrelated_keyword"].tolist()))
    checks={"S1_identity":stats["identity"]["pass_rate_score_ge_0.98"]>=.98,
      "S2_case_punctuation":stats["case_punctuation"]["pass_rate_drop_le_0.02"]>=.95,
      "S3_stopword":stats["stopword"]["pass_rate_abs_change_le_0.05"]>=.90,
      "S4_spatial":bool(stats["spatial_contradiction"]["n"] and stats["spatial_contradiction"]["pass_rate_drop_ge_0.20"]>=.90),
      "S5_high_idf":bool(stats["unrelated_keyword"]["n"] and stats["unrelated_keyword"]["median_drop"]>0),
      "S6_synonym":synonym_ok}
    sanity_payload={"status":"PASS" if all(checks.values()) else "SEMANTIC_ENCODER_SANITY_WEAK","checks":checks,"statistics":stats,
                    "semantic_encoder_sanity_pass":all(checks.values())}
    dump(out/"semantic_sanity_statistics.json",sanity_payload);dump(out/"reward_dev/semantic_sanity_statistics.json",sanity_payload)
    comparison={name:summarize_top(scored,name) for name in ("OLD_R3","Q1","Q2","Q3")}
    low=lambda row:row["C_ground"]<.25;high=lambda row:row["C_ground"]>=.75
    comparison["subgroups"]={name:{"low_phrase_sem":subgroup_top(scored,name,low,"R_phrase_sem"),
      "high_phrase_sem":subgroup_top(scored,name,high,"R_phrase_sem"),"high_absolute_iou":subgroup_top(scored,name,high,"R_mask")}
      for name in ("OLD_R3","Q1","Q2","Q3")}
    failures={name:failure_counts(scored,name) for name in ("OLD_R3","Q1","Q2","Q3")};comparison["failure_proxies"]=failures
    q1,q2=comparison["Q1"],comparison["Q2"];protocol=load(out/"reward_dev/selection_protocol.json")
    oldf,newf=failures["OLD_R3"],failures["Q2"];aggregate_ok=(newf["aggregate"]<=oldf["aggregate"]-1 if oldf["aggregate"]<10 else newf["aggregate"]<=.9*oldf["aggregate"])
    conditions={"lm_nonreg":q2["lm_verdict_accuracy"]>=q1["lm_verdict_accuracy"]-.005,
      "structure_nonreg":q2["structure_validity"]>=q1["structure_validity"]-.005,
      "phrase_nonreg":q2["fake_phrase_sem"]>=q1["fake_phrase_sem"]-.02,
      "high_ground_mask_help":comparison["subgroups"]["Q2"]["high_absolute_iou"]["mean"]>=comparison["subgroups"]["Q1"]["high_absolute_iou"]["mean"],
      "low_ground_phrase_nonreg":comparison["subgroups"]["Q2"]["low_phrase_sem"]["mean"]>=comparison["subgroups"]["Q1"]["low_phrase_sem"]["mean"]-.02,
      "each_failure_not_above_old":all(newf[k]<=oldf[k] for k in ("LEXICAL_HIGH_SEMANTIC_LOW","PHRASE_GOOD_MASK_CEILING_PUNISHED","SPATIAL_CONTRADICTION_SELECTED")),
      "aggregate_failure_reduction":aggregate_ok}
    selected="Q2" if all(conditions.values()) else "Q1"
    comparison.update({"selection_conditions":conditions,"selected_reward":selected,"semantic_sanity":sanity_payload["status"]})
    dump(out/"reward_dev_comparison.json",comparison);dump(out/"reward_dev/reward_dev_comparison.json",comparison)
    inputs=[out/"semantic_encoder_provenance.json",out/"phrase_preprocessing.json",out/"stopword_list.json",out/"token_idf.json",
            out/"critical_modifier_lexicon.json",out/"contradiction_rules.json",out/"reward_definition.json",out/"reward_dev/selection_protocol.json",
            out/"reward_dev/scored_groups.jsonl"]
    selected_payload={"status":"FROZEN_BEFORE_FULL_VALIDATION","selected_reward":selected,"conditions":conditions,
      "semantic_sanity_pass":sanity_payload["semantic_encoder_sanity_pass"],"Q3_primary_allowed":False,
      "inputs":[{"path":str(x),"sha256":file_sha256(x)} for x in inputs],"validation_used":False,"formula_changed":False}
    prior_path=out/"selected_reward.json"
    correction_path=out/"audit/prevalidation_selection_proxy_correction.json"
    if prior_path.exists() and not correction_path.exists():
            dump(correction_path,{
              "status":"CORRECTED_BEFORE_VALIDATION_RESULT", "validation_result_observed":False,
              "reward_formula_or_weights_changed":False,
              "incorrect_proxy":"counting selected phrase-good/mask-poor trajectories as failures",
              "correct_proxy":"counting semantic-best phrase-good/mask-poor trajectories ranked third or worse because of mask ceiling",
              "reason":"the original direction contradicted the preregistered double-penalty objective"})
    dump(out/"selected_reward.json",selected_payload);print(json.dumps(selected_payload,indent=2))


if __name__=="__main__":main()
