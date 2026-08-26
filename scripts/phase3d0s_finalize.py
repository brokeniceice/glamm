#!/usr/bin/env python3
"""Finalize Phase 3D.0-S from completed frozen GPT judgments only."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3d0s_gpt_audit"
SOURCE = ROOT / "outputs/phase3d0r_reward_reformulation"
EXPECTED_MODEL = "gpt-5.6-luna"
SEED = 3407
BOOTSTRAPS = 10_000


def cli_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-root",default="outputs/phase3d0s_gpt_audit")
    parser.add_argument("--expected-model",default="gpt-5.6-luna")
    parser.add_argument("--docs-path",default="docs/phase3d0s_independent_gpt_semantic_audit.md")
    return parser.parse_args(argv)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    return float(np.mean(values)) if values else None


def median(values):
    return float(np.median(values)) if values else None


def ci_mean(values, seed=SEED):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(BOOTSTRAPS, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def correlation_payload(x, y, seed=SEED):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson": None, "spearman": None,
                "pearson_bootstrap_95ci": None, "spearman_bootstrap_95ci": None}
    pearson = float(pearsonr(x, y).statistic); spearman = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed); p_draws=[]; s_draws=[]
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(x), len(x)); bx=x[index]; by=y[index]
        if np.std(bx) == 0 or np.std(by) == 0:
            continue
        p_draws.append(float(pearsonr(bx, by).statistic)); s_draws.append(float(spearmanr(bx, by).statistic))
    return {"n": len(x), "pearson": pearson, "spearman": spearman,
            "pearson_bootstrap_95ci": [float(v) for v in np.quantile(p_draws,[.025,.975])],
            "spearman_bootstrap_95ci": [float(v) for v in np.quantile(s_draws,[.025,.975])]}


def wilson(success, total, z=1.959963984540054):
    if total == 0: return None
    p=success/total; d=1+z*z/total; c=(p+z*z/(2*total))/d
    h=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [c-h,c+h]


def normalized_sources(mapping, orientation):
    a=mapping["candidate_A_source"]; b=mapping["candidate_B_source"]
    return (a,b) if orientation == "AB" else (b,a)


def normalize_response(row, mapping):
    a_source,b_source=normalized_sources(mapping,row["orientation"])
    winner=row["winner"]
    normalized_winner = a_source if winner == "A" else b_source if winner == "B" else winner
    metrics={
      a_source:{"overall":row["A_score"],"phrase":row["A_phrase_semantic_score"],
                "phrase_mask":row["A_phrase_mask_consistency_score"],"explanation":row["A_explanation_evidence_consistency_score"]},
      b_source:{"overall":row["B_score"],"phrase":row["B_phrase_semantic_score"],
                "phrase_mask":row["B_phrase_mask_consistency_score"],"explanation":row["B_explanation_evidence_consistency_score"]},
    }
    return {**row,"normalized_winner":normalized_winner,"source_metrics":metrics}


def validate(manifest, blind, scores):
    pair_ids={x["pair_id"] for x in manifest["pairs"]}; mappings={x["pair_id"] for x in blind["mappings"]}
    requests=[x["request_id"] for x in scores]
    by_pair=defaultdict(list)
    for row in scores: by_pair[row["pair_id"]].append(row)
    errors=[]
    if manifest["pair_count"] != 500 or len(pair_ids) != 500: errors.append("pair_count")
    if pair_ids != mappings: errors.append("blind_mapping_set")
    if len(scores) != 1050 or len(set(requests)) != 1050: errors.append("request_count_or_uniqueness")
    repeated=0
    for pair_id in pair_ids:
        keys={(x["orientation"],x["replicate"]) for x in by_pair[pair_id]}
        if not {("AB",0),("BA",0)} <= keys: errors.append("missing_order:"+pair_id)
        repeated += ("AB",1) in keys
    if repeated != 50: errors.append("repeat_count")
    allowed={"A","B","TIE","UNCERTAIN"}
    score_fields=["A_score","B_score","A_phrase_semantic_score","B_phrase_semantic_score",
      "A_phrase_mask_consistency_score","B_phrase_mask_consistency_score",
      "A_explanation_evidence_consistency_score","B_explanation_evidence_consistency_score"]
    for row in scores:
        if row["winner"] not in allowed or any(not isinstance(row[k],int) or not 0 <= row[k] <= 4 for k in score_fields):
            errors.append("schema:"+row["request_id"])
        if row.get("response_model") != EXPECTED_MODEL: errors.append("model:"+row["request_id"])
    payload={"status":"PASS" if not errors else "FAIL","pair_count":len(pair_ids),"request_count":len(scores),
             "unique_request_count":len(set(requests)),"order_complete_pairs":sum({("AB",0),("BA",0)} <= {(x["orientation"],x["replicate"]) for x in by_pair[p]} for p in pair_ids),
             "expected_model":EXPECTED_MODEL,"repeatability_pairs":repeated,"all_response_models_exact":all(x.get("response_model")==EXPECTED_MODEL for x in scores),"errors":errors}
    if errors: raise RuntimeError("integrity validation failed: "+repr(errors[:10]))
    return payload


def candidate_for_source(pair, mapping, source):
    return pair["left"] if mapping["candidate_A_source"] == source else pair["right"]


def pairwise_summary(pair_rows, consensus):
    counts=Counter(consensus[p["pair_id"]]["winner"] for p in pair_rows)
    q2=int(counts["Q2"]); comp=int(counts["comparator"]); denominator=q2+comp
    return {"n_pairs":len(pair_rows),"Q2_wins":q2,"comparator_wins":comp,"ties":int(counts["TIE"]),
      "uncertain":int(counts["UNCERTAIN"]),"order_disagreements":int(counts["ORDER_DISAGREEMENT"]),
      "effective_decisive_n":denominator,"gpt_q2_preference_rate":q2/denominator if denominator else None,
      "wilson_95ci":wilson(q2,denominator),"consensus_rule":"AB and BA must map to the same source-level winner; otherwise ORDER_DISAGREEMENT"}


def source_average(ab,ba,source,metric):
    return (ab["source_metrics"][source][metric]+ba["source_metrics"][source][metric])/2


def qualitative_page(pairs, mappings, consensus, observations):
    obs_by_pair={x["pair_id"]:x for x in observations}
    def q2(p): return candidate_for_source(p,mappings[p["pair_id"]],"Q2")
    def comp(p): return candidate_for_source(p,mappings[p["pair_id"]],"comparator")
    categories={
      "GPT prefers Q2 over greedy":[p for p in pairs if p["cohort"]=="Q2_VS_GREEDY" and consensus[p["pair_id"]]["winner"]=="Q2"],
      "GPT prefers greedy over Q2":[p for p in pairs if p["cohort"]=="Q2_VS_GREEDY" and consensus[p["pair_id"]]["winner"]=="comparator"],
      "GPT disagrees with MiniLM":[p for p in pairs if (q2(p)["R_phrase_sem"]-comp(p)["R_phrase_sem"])*(obs_by_pair[p["pair_id"]]["q2_phrase"]-obs_by_pair[p["pair_id"]]["comparator_phrase"])<0],
      "MiniLM false positive":[p for p in pairs if any(c["R_phrase_sem"]>=.7 and s<=1 for c,s in ((q2(p),obs_by_pair[p["pair_id"]]["q2_phrase"]),(comp(p),obs_by_pair[p["pair_id"]]["comparator_phrase"])))],
      "MiniLM false negative":[p for p in pairs if any(c["R_phrase_sem"]<.5 and s>=3 for c,s in ((q2(p),obs_by_pair[p["pair_id"]]["q2_phrase"]),(comp(p),obs_by_pair[p["pair_id"]]["comparator_phrase"])))],
      "OLD_R3 failure fixed by Q2":[p for p in pairs if (p["cohort"]=="Q2_VS_OLD_R3" or p["stratum"]=="OLD_R3_DOUBLE_PENALTY") and consensus[p["pair_id"]]["winner"]=="Q2"],
      "Phrase correct but mask poor":[p for p in pairs if p["stratum"]=="PHRASE_GOOD_MASK_POOR"],
      "Mask improved with correct phrase":[p for p in pairs if q2(p)["R_mask"]>comp(p)["R_mask"] and obs_by_pair[p["pair_id"]]["q2_phrase"]>=3],
    }
    parts=["<!doctype html><meta charset='utf-8'><title>Phase 3D.0-S Qualitative Audit</title>",
      "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}article{border-top:2px solid #777;padding:1em}.imgs{display:flex;gap:8px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{max-width:260px}.text{display:grid;grid-template-columns:1fr 1fr;gap:14px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}</style>",
      "<h1>Phase 3D.0-S Independent GPT Semantic Audit</h1><p>页面已解盲；MiniLM/reward信息仅在GPT评分全部完成后加入，未提供给GPT。</p>"]
    selection={}
    for name,values in categories.items():
        chosen=sorted(values,key=lambda p:hashlib.sha256(f"{SEED}:{name}:{p['pair_id']}".encode()).hexdigest())[:6]
        selection[name]=[p["pair_id"] for p in chosen];parts.append(f"<h2>{html.escape(name)} (shown {len(chosen)} / eligible {len(values)})</h2>")
        for p in chosen:
            pid=p["pair_id"]; o=obs_by_pair[pid]; qc=q2(p); cc=comp(p); assets=p["assets"]
            def rel(x): return "../audit/assets/"+pid+"/"+Path(x).name
            parts.append(f"<article><h3>{pid} · {html.escape(p['sample_id'])}</h3><p>cohort={p['cohort']}; stratum={p['stratum']}; consensus={consensus[pid]['winner']}</p>"
              f"<div class='imgs'><figure><img src='{rel(assets['original'])}'><figcaption>Original</figcaption></figure><figure><img src='{rel(assets['reference'])}'><figcaption>Annotation-derived reference</figcaption></figure><figure><img src='{rel(assets['left'])}'><figcaption>Original Candidate A overlay</figcaption></figure><figure><img src='{rel(assets['right'])}'><figcaption>Original Candidate B overlay</figcaption></figure></div>"
              f"<table><tr><th></th><th>MiniLM phrase</th><th>GPT phrase</th><th>mask IoU diagnostic</th><th>GPT overall</th><th>GPT phrase-mask</th><th>GPT explanation</th></tr>"
              f"<tr><td>Q2</td><td>{qc['R_phrase_sem']:.3f}</td><td>{o['q2_phrase']:.2f}</td><td>{qc['R_mask']:.3f}</td><td>{o['q2_overall']:.2f}</td><td>{o['q2_phrase_mask']:.2f}</td><td>{o['q2_explanation']:.2f}</td></tr>"
              f"<tr><td>Comparator</td><td>{cc['R_phrase_sem']:.3f}</td><td>{o['comparator_phrase']:.2f}</td><td>{cc['R_mask']:.3f}</td><td>{o['comparator_overall']:.2f}</td><td>{o['comparator_phrase_mask']:.2f}</td><td>{o['comparator_explanation']:.2f}</td></tr></table>"
              f"<div class='text'><div><h4>Q2</h4><b>{html.escape(str(qc['normalized_phrase']))}</b><pre>{html.escape(str(qc['generated_explanation']))}</pre></div><div><h4>Comparator</h4><b>{html.escape(str(cc['normalized_phrase']))}</b><pre>{html.escape(str(cc['generated_explanation']))}</pre></div></div>"
              f"<p><b>AB reason:</b> {html.escape(consensus[pid]['ab_reason'])}</p><p><b>BA reason:</b> {html.escape(consensus[pid]['ba_reason'])}</p></article>")
    path=OUT/"qualitative/index.html";path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(parts),encoding="utf-8")
    write_json(OUT/"qualitative/selection_manifest.json",{"deterministic":True,"seed":SEED,"categories":selection,"index":str(path)})


def main(argv=None):
    global OUT,EXPECTED_MODEL
    args=cli_args(argv);OUT=(ROOT/args.output_root).resolve();EXPECTED_MODEL=args.expected_model
    manifest=read_json(OUT/"sampling_manifest.json");blind=read_json(OUT/"blind_mapping.json");scores=read_jsonl(OUT/"parsed_scores.jsonl");provenance=read_json(OUT/"gpt_provenance.json")
    integrity=validate(manifest,blind,scores);write_json(OUT/"audit/response_integrity.json",integrity)
    pairs=manifest["pairs"]; pair_by_id={p["pair_id"]:p for p in pairs};mapping_by_id={m["pair_id"]:m for m in blind["mappings"]}
    normalized=[normalize_response(x,mapping_by_id[x["pair_id"]]) for x in scores]
    by_pair=defaultdict(dict)
    for x in normalized: by_pair[x["pair_id"]][(x["orientation"],x["replicate"])]=x
    consensus={};order_matches=[];observations=[];candidate_obs=[]
    for p in pairs:
        pid=p["pair_id"];ab=by_pair[pid][("AB",0)];ba=by_pair[pid][("BA",0)];match=ab["normalized_winner"]==ba["normalized_winner"]
        order_matches.append(match);winner=ab["normalized_winner"] if match else "ORDER_DISAGREEMENT"
        consensus[pid]={"winner":winner,"order_consistent":match,"ab_winner":ab["normalized_winner"],"ba_winner":ba["normalized_winner"],"ab_reason":ab["reason"],"ba_reason":ba["reason"]}
        obs={"pair_id":pid,"cohort":p["cohort"],"stratum":p["stratum"],"winner":winner}
        for source,prefix in (("Q2","q2"),("comparator","comparator")):
            for metric in ("overall","phrase","phrase_mask","explanation"):
                obs[prefix+"_"+metric]=source_average(ab,ba,source,metric)
            candidate=candidate_for_source(p,mapping_by_id[pid],source)
            candidate_obs.append({"pair_id":pid,"sample_id":p["sample_id"],"cohort":p["cohort"],"stratum":p["stratum"],"source":source,
                "trajectory_key":f"{p['sample_id']}:{source}:{candidate.get('rollout_index')}","R_phrase_sem":candidate["R_phrase_sem"],"R_phrase_lex":candidate["R_phrase_lex"],
                "R_mask":candidate["R_mask"],"R_ground_rel":candidate.get("R_ground_rel"),"gpt_phrase":obs[prefix+"_phrase"],"gpt_overall":obs[prefix+"_overall"],
                "gpt_phrase_mask":obs[prefix+"_phrase_mask"],"gpt_explanation":obs[prefix+"_explanation"]})
        observations.append(obs)
    raw_ab=Counter(by_pair[p["pair_id"]][("AB",0)]["winner"] for p in pairs)
    raw_ba=Counter(by_pair[p["pair_id"]][("BA",0)]["winner"] for p in pairs)
    letter_patterns=Counter((by_pair[p["pair_id"]][("AB",0)]["winner"],by_pair[p["pair_id"]][("BA",0)]["winner"]) for p in pairs)
    order_rate=mean(order_matches);order_payload={"status":"PASS" if order_rate>=.90 else "GPT_ORDER_BIAS_HIGH","n_pairs":500,"consistent_pairs":sum(order_matches),
      "inconsistent_pairs":500-sum(order_matches),"order_consistency":order_rate,"threshold":.90,"exact_source_level_winner_agreement_includes_tie_and_uncertain":True,
      "raw_position_winner_counts":{"AB":dict(raw_ab),"BA":dict(raw_ba)},
      "raw_letter_pair_patterns":{f"{a}_then_{b}":n for (a,b),n in sorted(letter_patterns.items())},
      "position_bias_diagnostic":"Selecting the same letter after candidate swap changes the underlying source winner; A_then_A and B_then_B are direct position-inconsistent patterns.",
      "by_cohort":{name:{"n":len(xs),"consistency":mean([consensus[x["pair_id"]]["order_consistent"] for x in xs])} for name,xs in ((n,[p for p in pairs if p["cohort"]==n]) for n in ("Q2_VS_GREEDY","Q2_VS_OLD_R3","FAILURE_ORIENTED"))}}
    write_json(OUT/"order_bias.json",order_payload);write_json(OUT/"audit/order_bias.json",order_payload)
    repeats=[]
    for pid,values in by_pair.items():
        if ("AB",1) not in values: continue
        first=values[("AB",0)];second=values[("AB",1)];agreement=first["normalized_winner"]==second["normalized_winner"]
        diffs=[]
        for source in ("Q2","comparator"):
            for metric in ("overall","phrase","phrase_mask","explanation"):
                diffs.append(abs(first["source_metrics"][source][metric]-second["source_metrics"][source][metric]))
        repeats.append({"pair_id":pid,"winner_agreement":agreement,"mean_absolute_score_difference":mean(diffs),"max_absolute_score_difference":max(diffs)})
    repeat_rate=mean([x["winner_agreement"] for x in repeats]);repeat_payload={"status":"PASS" if repeat_rate>=.90 else "GPT_REPEATABILITY_WEAK","n_pairs":len(repeats),
      "winner_agreement_count":sum(x["winner_agreement"] for x in repeats),"winner_agreement":repeat_rate,"threshold":.90,
      "mean_absolute_score_difference":mean([x["mean_absolute_score_difference"] for x in repeats]),"median_absolute_score_difference":median([x["mean_absolute_score_difference"] for x in repeats]),
      "max_score_difference_distribution":dict(Counter(x["max_absolute_score_difference"] for x in repeats)),"pairs":repeats}
    write_json(OUT/"repeatability.json",repeat_payload);write_json(OUT/"audit/repeatability.json",repeat_payload)
    primary_g=[p for p in pairs if p["cohort"]=="Q2_VS_GREEDY"];primary_o=[p for p in pairs if p["cohort"]=="Q2_VS_OLD_R3"]
    judge_reliable=order_rate>=.90 and repeat_rate>=.90
    g_summary=pairwise_summary(primary_g,consensus);g_summary.update({"comparison":"Q2_vs_greedy","threshold":.70,"threshold_pass":g_summary["gpt_q2_preference_rate"]>=.70,
      "judge_reliability_pass":judge_reliable,"scientifically_usable":judge_reliable,"validity_note":"Descriptive only; must not be used for a Q2 claim when judge reliability fails."})
    o_summary=pairwise_summary(primary_o,consensus);o_summary.update({"comparison":"Q2_vs_OLD_R3","threshold":.60,"threshold_pass":o_summary["gpt_q2_preference_rate"]>=.60,
      "judge_reliability_pass":judge_reliable,"scientifically_usable":judge_reliable,"validity_note":"Descriptive only; must not be used for a Q2 claim when judge reliability fails."})
    write_json(OUT/"q2_vs_greedy.json",g_summary);write_json(OUT/"statistics/q2_vs_greedy.json",g_summary)
    write_json(OUT/"q2_vs_oldr3.json",o_summary);write_json(OUT/"statistics/q2_vs_oldr3.json",o_summary)
    minilm=[x["R_phrase_sem"] for x in candidate_obs];gpt=[x["gpt_phrase"] for x in candidate_obs]
    unique_groups=defaultdict(list)
    for x in candidate_obs: unique_groups[x["trajectory_key"]].append(x)
    unique=[{**xs[0],"gpt_phrase":mean([x["gpt_phrase"] for x in xs])} for xs in unique_groups.values()]
    corr_all=correlation_payload(minilm,gpt);corr_unique=correlation_payload([x["R_phrase_sem"] for x in unique],[x["gpt_phrase"] for x in unique],SEED+1)
    false_neg=[x for x in candidate_obs if x["R_phrase_sem"]<.5 and x["gpt_phrase"]>=3]
    false_pos=[x for x in candidate_obs if x["R_phrase_sem"]>=.7 and x["gpt_phrase"]<=1]
    correlation={"schema_version":"phase3d0s_minilm_gpt_correlation_v1","primary_analysis":"all blinded candidate observations, order-averaged",
      "all_candidate_observations":corr_all,"unique_trajectory_sensitivity":corr_unique,"threshold_spearman":.60,"threshold_pass":corr_all["spearman"]>=.60,
      "judge_reliability_pass":judge_reliable,"scientifically_usable":judge_reliable,
      "validity_note":"Correlation is descriptive only and must not validate or invalidate MiniLM when the GPT judge reliability gate fails.",
      "minilm_summary":{"mean":mean(minilm),"median":median(minilm),"mean_bootstrap_95ci":ci_mean(minilm)},
      "gpt_phrase_summary":{"mean":mean(gpt),"median":median(gpt),"mean_bootstrap_95ci":ci_mean(gpt,SEED+2)},
      "false_negative_definition":"R_phrase_sem < 0.5 and GPT phrase >= 3","false_negative_count":len(false_neg),"false_negative_rate":len(false_neg)/len(candidate_obs),
      "false_positive_definition":"R_phrase_sem >= 0.7 and GPT phrase <= 1","false_positive_count":len(false_pos),"false_positive_rate":len(false_pos)/len(candidate_obs),
      "dependency_note":"The lexical-high/semantic-low failure stratum contains deterministic replacement; unique-trajectory sensitivity deduplicates repeated trajectory keys."}
    write_json(OUT/"minilm_gpt_correlation.json",correlation);write_json(OUT/"correlation/minilm_gpt_correlation.json",correlation)
    subgroup={}
    names=["LEXICAL_LOW_SEMANTIC_HIGH","LEXICAL_HIGH_SEMANTIC_LOW","SPATIAL_CONTRADICTION","PHRASE_GOOD_MASK_POOR","OLD_R3_DOUBLE_PENALTY"]
    obs_by_id={x["pair_id"]:x for x in observations}
    for name in names:
        ps=[p for p in pairs if p["stratum"]==name];os=[obs_by_id[p["pair_id"]] for p in ps]
        subgroup[name]={"n_pair_observations":len(ps),"unique_trajectory_keys":len({f"{p['sample_id']}:{candidate_for_source(p,mapping_by_id[p['pair_id']],'comparator').get('rollout_index')}" for p in ps}),
          "q2_preference":pairwise_summary(ps,consensus),"q2_gpt_phrase_mean":mean([x["q2_phrase"] for x in os]),"comparator_gpt_phrase_mean":mean([x["comparator_phrase"] for x in os]),
          "q2_gpt_phrase_mask_mean":mean([x["q2_phrase_mask"] for x in os]),"comparator_gpt_phrase_mask_mean":mean([x["comparator_phrase_mask"] for x in os])}
    high=[x for x in candidate_obs if x["source"]=="Q2" and x["R_ground_rel"] is not None and x["R_ground_rel"]>=.75]
    subgroup["HIGH_CONTROLLABILITY_GROUNDING"]={"operational_proxy":"Q2 R_ground_rel >= 0.75","n_candidate_observations":len(high),
      "gpt_phrase_mean":mean([x["gpt_phrase"] for x in high]),"gpt_phrase_mask_mean":mean([x["gpt_phrase_mask"] for x in high]),"gpt_overall_mean":mean([x["gpt_overall"] for x in high])}
    subgroup_payload={"judge_reliability_pass":judge_reliable,"scientifically_usable":judge_reliable,
      "validity_note":"Subgroup results are descriptive only when the GPT judge reliability gate fails.","subgroups":subgroup}
    write_json(OUT/"subgroup_analysis.json",subgroup_payload);write_json(OUT/"subgroups/subgroup_analysis.json",subgroup_payload)
    primary_obs=[x for x in observations if x["cohort"] in ("Q2_VS_GREEDY","Q2_VS_OLD_R3")]
    q2_phrase=[x["q2_phrase"] for x in primary_obs];comp_phrase=[x["comparator_phrase"] for x in primary_obs]
    q2_low=mean([x<=1 for x in q2_phrase]);comp_low=mean([x<=1 for x in comp_phrase])
    collapse={"operational_definition":"Q2 GPT phrase mean is no more than 0.25 points below comparator and Q2 score<=1 rate is no more than 0.05 above comparator",
      "q2_phrase_mean":mean(q2_phrase),"comparator_phrase_mean":mean(comp_phrase),"mean_difference":mean(q2_phrase)-mean(comp_phrase),
      "q2_score_le1_rate":q2_low,"comparator_score_le1_rate":comp_low,"low_score_rate_difference":q2_low-comp_low}
    collapse["no_obvious_semantic_collapse"]=collapse["mean_difference"]>=-.25 and collapse["low_score_rate_difference"]<=.05
    write_json(OUT/"statistics/semantic_collapse.json",collapse)
    qualitative_page(pairs,mapping_by_id,consensus,observations)
    reliable=judge_reliable
    supported=reliable and g_summary["threshold_pass"] and o_summary["threshold_pass"] and correlation["threshold_pass"] and collapse["no_obvious_semantic_collapse"]
    if not reliable: gate="GATE_GPT_AUDIT_FAILED"
    elif supported: gate="GATE_GPT_SEMANTIC_AUDIT_SUPPORTED"
    else: gate="GATE_REWARD_PROXY_WEAK"
    route={"schema_version":"phase3d0s_route_gate_v1","status":"COMPLETE","gate":gate,"terminal":True,
      "criteria":{"gpt_vision_backend_available":True,"order_consistency":{"value":order_rate,"threshold":.90,"pass":order_rate>=.90},
        "repeatability":{"value":repeat_rate,"threshold":.90,"pass":repeat_rate>=.90},"q2_vs_greedy_preference":{"value":g_summary["gpt_q2_preference_rate"],"threshold":.70,"pass":g_summary["threshold_pass"]},
        "q2_vs_oldr3_preference":{"value":o_summary["gpt_q2_preference_rate"],"threshold":.60,"pass":o_summary["threshold_pass"]},
        "minilm_gpt_spearman":{"value":corr_all["spearman"],"threshold":.60,"pass":correlation["threshold_pass"]},"no_obvious_semantic_collapse":{"value":collapse["no_obvious_semantic_collapse"],"pass":collapse["no_obvious_semantic_collapse"]}},
      "authorization":"Phase 3D.1 policy optimization design only; no automatic training" if gate=="GATE_GPT_SEMANTIC_AUDIT_SUPPORTED" else None,
      "rollout_repeated":False,"reward_modified":False,"training_started":False,"phase3d0r_q2_changed":False}
    write_json(OUT/"route_gate.json",route)
    report=f"""# Phase 3D.0-S — Independent GPT Semantic Audit

## 1. 最终结论

最终 gate：`{gate}`。

**由于 position-order consistency 仅为 {order_rate:.4f}，低于 0.90，本阶段按预注册规则判定 GPT judge 不可靠。以下 preference、correlation 与 subgroup 数值只作为故障诊断，不得用于支持或否定 Q2/MiniLM。**

本阶段只对 Phase 3D.0-R 冻结回答进行独立多模态 GPT 外部验证；没有重新 rollout、没有修改 Q2/MiniLM/reward，也没有启动训练。

## 2. GPT 与数据完整性

- judge：`{EXPECTED_MODEL}`，temperature={provenance.get('temperature')}（parameter sent={provenance.get('temperature_parameter_sent')}），reasoning effort={provenance.get('reasoning_effort')}。
- 500 对比较、每对 AB/BA 两次，另有 50 对重复调用，总计 1050 条；失败 0 条。
- 输入包括原图、官方 SynthScars annotation-derived reference overlay、两候选 phrase/explanation/mask overlay。
- GPT 不可见候选来源、reward、MiniLM、IoU、F1 或置信度。

## 3. GPT 自身可靠性

- position-order consistency：{order_rate:.4f}（阈值 0.90，{'PASS' if order_rate>=.90 else 'FAIL'}）。
- repeatability winner agreement：{repeat_rate:.4f}（阈值 0.90，{'PASS' if repeat_rate>=.90 else 'FAIL'}）。
- repeatability mean absolute score difference：{repeat_payload['mean_absolute_score_difference']:.4f}。
- 原始位置选择：AB 调用 A={raw_ab['A']}、B={raw_ab['B']}、TIE={raw_ab['TIE']}；BA 调用 A={raw_ba['A']}、B={raw_ba['B']}、TIE={raw_ba['TIE']}。
- 交换候选后仍选择同一字母：A→A={letter_patterns[('A','A')]}，B→B={letter_patterns[('B','B')]}。这两种模式会改变真实来源胜者，是直接的位置偏差证据。

## 4. Reward ranking validation

本节因 GPT reliability gate 失败而**不可用于科学结论**，仅记录描述性故障统计。

### Q2 vs Greedy

- Q2 wins={g_summary['Q2_wins']}，Greedy wins={g_summary['comparator_wins']}，tie={g_summary['ties']}，uncertain={g_summary['uncertain']}，order disagreement={g_summary['order_disagreements']}。
- decisive preference rate={g_summary['gpt_q2_preference_rate']:.4f}，95% Wilson CI={g_summary['wilson_95ci']}，阈值 0.70，{'PASS' if g_summary['threshold_pass'] else 'FAIL'}。

### Q2 vs OLD_R3

- Q2 wins={o_summary['Q2_wins']}，OLD_R3 wins={o_summary['comparator_wins']}，tie={o_summary['ties']}，uncertain={o_summary['uncertain']}，order disagreement={o_summary['order_disagreements']}。
- decisive preference rate={o_summary['gpt_q2_preference_rate']:.4f}，95% Wilson CI={o_summary['wilson_95ci']}，阈值 0.60，{'PASS' if o_summary['threshold_pass'] else 'FAIL'}。

胜率仅使用 AB 与 BA 在真实来源层面一致、且明确给出 Q2 或 comparator 胜者的 pair；tie、uncertain 和 order disagreement 均不进入分母。

## 5. MiniLM–GPT semantic validation

本节同样因 GPT reliability gate 失败而**不能验证或否定 MiniLM**。

- candidate observations={corr_all['n']}；Pearson={corr_all['pearson']:.4f}，95% bootstrap CI={corr_all['pearson_bootstrap_95ci']}。
- Spearman={corr_all['spearman']:.4f}，95% bootstrap CI={corr_all['spearman_bootstrap_95ci']}，阈值 0.60，{'PASS' if correlation['threshold_pass'] else 'FAIL'}。
- MiniLM false negative={len(false_neg)}；false positive={len(false_pos)}。
- unique-trajectory sensitivity Spearman={corr_unique['spearman']:.4f}。

这里 GPT phrase score 是 AB/BA 对同一真实候选评分的平均值。lexical-high/semantic-low 严格池只有 9 个唯一 mask 轨迹，20 次 failure-oriented 判断含确定性有放回补足；因此同时报告 unique-trajectory sensitivity，不把重复项当作额外独立轨迹。

## 6. Semantic collapse audit

- Q2 GPT phrase mean={collapse['q2_phrase_mean']:.4f}；comparator={collapse['comparator_phrase_mean']:.4f}；Δ={collapse['mean_difference']:.4f}。
- Q2 score<=1 rate={collapse['q2_score_le1_rate']:.4f}；comparator={collapse['comparator_score_le1_rate']:.4f}。
- no obvious semantic collapse={collapse['no_obvious_semantic_collapse']}。

这是审计前未规定数值公式的定性 gate，本次采用保守的显式操作定义：均值退化不超过 0.25/4 分，且低分率增加不超过 5 个百分点。

## 7. 科学边界与下一步

`GATE_GPT_SEMANTIC_AUDIT_SUPPORTED` 只会授权 Phase 3D.1 policy optimization design，不自动授权或启动 GRPO/PPO/DPO/SFT。`GATE_REWARD_PROXY_WEAK` 表示 GPT 自身可靠但 MiniLM proxy 或 ranking gate 未获支持，应停止并审查 semantic proxy。`GATE_GPT_AUDIT_FAILED` 表示 GPT judge 自身不可靠，不能使用其结论。

本次实际进入 `GATE_GPT_AUDIT_FAILED`，因此停止：不授权 Phase 3D.1，不修改 reward，不补做自动调参，也不把描述性胜率解释为 Q2 性能。

定性页面：`outputs/phase3d0s_gpt_audit/qualitative/index.html`。
"""
    (OUT/"final_report.md").write_text(report,encoding="utf-8");(OUT/"reports/final_report.md").write_text(report,encoding="utf-8")
    docs=(ROOT/args.docs_path).resolve();docs.parent.mkdir(parents=True,exist_ok=True);docs.write_text(report,encoding="utf-8")
    completion={"schema_version":"phase3d0s_completion_manifest_v1","status":"COMPLETE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"route_gate":gate,
      "counts":{"pairs":500,"api_responses":1050,"api_failures":0},"hashes":{"sampling_manifest_sha256":sha256_file(OUT/"sampling_manifest.json"),
      "blind_mapping_sha256":sha256_file(OUT/"blind_mapping.json"),"raw_responses_sha256":sha256_file(OUT/"raw_responses.jsonl"),"parsed_scores_sha256":sha256_file(OUT/"parsed_scores.jsonl"),
      "final_report_sha256":sha256_file(OUT/"final_report.md")},"invariants":{"rollout_repeated":False,"reward_modified":False,"training_started":False,"phase3d0r_q2_changed":False}}
    write_json(OUT/"completion_manifest.json",completion)
    print(json.dumps({"gate":gate,"order_consistency":order_rate,"repeatability":repeat_rate,"q2_vs_greedy":g_summary["gpt_q2_preference_rate"],
      "q2_vs_oldr3":o_summary["gpt_q2_preference_rate"],"spearman":corr_all["spearman"],"semantic_collapse_ok":collapse["no_obvious_semantic_collapse"]},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
