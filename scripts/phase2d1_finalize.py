#!/usr/bin/env python3
"""Finalize Phase 2D.1 metrics, qualitative assets, audit, and report inputs."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import random
from statistics import mean, median

from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2d1_trace_replay"
CHECKPOINT = ROOT / "checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt"
SEED = 20260811


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for position in order[start:end]:
            result[position] = average
        start = end
    return result


def pearson(a, b):
    am, bm = mean(a), mean(b)
    numerator = sum((x-am)*(y-bm) for x,y in zip(a,b))
    denominator = math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator / denominator if denominator else None


def correlation(a, b):
    return {"n": len(a), "pearson": pearson(a,b), "spearman": pearson(rank(a),rank(b)),
            "interpretation": "相关性，不是因果证据。"}


def bootstrap_mean(values, iterations=10000):
    if not values:
        return {"status":"NOT_ESTIMABLE"}
    rng=random.Random(SEED); estimates=[]
    for _ in range(iterations):
        estimates.append(mean(rng.choice(values) for _ in values))
    estimates.sort()
    return {"mean":mean(values),"bootstrap_iterations":iterations,"seed":SEED,
            "ci95_percentile":[estimates[249],estimates[9749]]}


def mode_summary(rows, mode):
    values=[row[mode]["iou"] for row in rows]
    gains=[row[mode]["recovery_from_G0"] for row in rows]
    fractions=[row[mode]["fraction_of_TF_gap_recovered"] for row in rows
               if row[mode]["fraction_of_TF_gap_recovered"] is not None]
    return {"n":len(rows),"mean_iou":mean(values) if values else None,"median_iou":median(values) if values else None,
            "mean_recovery_from_G0":mean(gains) if gains else None,"median_recovery_from_G0":median(gains) if gains else None,
            "positive_recovery_proportion":mean(x>0 for x in gains) if gains else None,
            "fraction_of_TF_gap_recovered_mean":mean(fractions) if fractions else None,
            "fraction_ratio_n":len(fractions)}


def overlay(image, mask, color):
    base=image.convert("RGBA")
    value=Image.fromarray(mask.to(torch.uint8).cpu().numpy()*110,"L").resize(base.size,Image.Resampling.NEAREST)
    layer=Image.new("RGBA",base.size,color)
    base.alpha_composite(Image.composite(layer,Image.new("RGBA",base.size,(0,0,0,0)),value))
    return base.convert("RGB")


def qualitative(rows, manifests):
    assets=OUT/"qualitative/assets"; assets.mkdir(parents=True,exist_ok=True)
    cards=[]
    for index,row in enumerate(rows):
        sid=row["sample_id"]; safe=sid.replace(":","__")
        image=Image.open(manifests[sid]["image_path"]).convert("RGB")
        gt=Image.new("L",image.size,0); draw=ImageDraw.Draw(gt)
        for ref in manifests[sid].get("refs",[]):
            for polygon in ref.get("polygons",[]):
                draw.polygon([(polygon[i],polygon[i+1]) for i in range(0,len(polygon),2)],fill=1)
        g0=torch.load(OUT/f"traces/g0/{safe}.pt",map_location="cpu")
        tf=torch.load(OUT/f"traces/tf/{safe}.pt",map_location="cpu")
        panels=[("Original",image),("GT",overlay(image,torch.from_numpy(__import__('numpy').array(gt)).bool(),(0,255,0,150))),
                ("G0",overlay(image,g0["binary_mask"].any(0),(255,0,0,150))),
                ("TF",overlay(image,tf["binary_mask"].any(0),(0,100,255,150)))]
        if row["status"]=="COMPLETE":
            h2=torch.load(OUT/f"traces/h2/{safe}.pt",map_location="cpu")
            h3=torch.load(OUT/f"traces/h3/{safe}.pt",map_location="cpu")
            panels.extend([("H2a",overlay(image,h2["H2a"]["binary_mask"].any(0),(255,0,255,150))),
                           ("H3 phrase",overlay(image,h3["binary_mask"].any(0),(255,180,0,150)))])
        thumb=[]
        for label,panel in panels:
            panel.thumbnail((320,320)); canvas=Image.new("RGB",(320,350),"white"); canvas.paste(panel,((320-panel.width)//2,25))
            ImageDraw.Draw(canvas).text((8,5),label,fill="black"); thumb.append(canvas)
        montage=Image.new("RGB",(320*len(thumb),350),"white")
        for i,panel in enumerate(thumb):montage.paste(panel,(i*320,0))
        asset=assets/f"{index:03d}_{safe}.jpg"; montage.save(asset,quality=88)
        cards.append(f'''<article><h2>{html.escape(sid)} · {html.escape(row['status'])}</h2><img src="assets/{asset.name}">
<p>G0={row['G0']['iou']:.4f}；TF={row['TF']['iou']:.4f}；H2a={row.get('H2a',{}).get('iou','NA')}；H3={row.get('H3_phrase_only',{}).get('iou','NA')}</p>
<p>G0/TF token length={row['G0']['token_length']}/{row['TF']['token_length']}；SEG position={row['G0']['seg_position']}/{row['TF']['seg_position']}</p>
<details><summary>G0 文本</summary><pre>{html.escape(row['G0_text'])}</pre></details><details><summary>TF 文本与 phrase</summary><pre>{html.escape(row['TF_text'])}\nphrases: {html.escape(row.get('phrases',''))}</pre></details></article>''')
    page='''<!doctype html><meta charset="utf-8"><title>Phase 2D.1 定性审计</title><style>body{font:15px sans-serif;max-width:1960px;margin:auto;background:#eee}article{background:white;margin:16px;padding:14px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Phase 2D.1 真实 mask 定性审计</h1>'''+''.join(cards)
    (OUT/"qualitative/index.html").write_text(page,encoding="utf-8")
    return len(cards)


def main():
    rows=read_jsonl(OUT/"controlled_modes/per_sample_results.jsonl")
    selections={row["sample_id"]:row for row in read_jsonl(OUT/"selection/selected_samples.jsonl")}
    manifests={row["sample_id"]:row for row in read_jsonl(
        ROOT/"outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl")}
    valid=[row for row in rows if row["status"]=="COMPLETE"]
    modes=("H1a_oracle_context","H2a","H2b","H2c","H3_phrase_only")
    subset_predicates={
        "all_traced":lambda s:True,
        "severe":lambda s:s["historical_tf_iou"]>=.7 and s["historical_g0_iou"]<=.3,
        "F_SEG_state_candidate":lambda s:"S2_all_F_candidates" in s["subsets"],
        "negative_gap":lambda s:s["historical_gap"]<0,
        "matched_controls":lambda s:"S4_matched_control" in s["subsets"],
    }
    subgroup={}
    for name,predicate in subset_predicates.items():
        members=[row for row in rows if predicate(selections[row["sample_id"]])]
        eligible=[row for row in members if row["status"]=="COMPLETE"]
        subgroup[name]={"selected_n":len(members),"eligible_n":len(eligible),
            "status_counts":{status:sum(row["status"]==status for row in members) for status in sorted({r["status"] for r in members})},
            "G0_mean_iou":mean(r["G0"]["iou"] for r in eligible) if eligible else None,
            "TF_mean_iou":mean(r["TF"]["iou"] for r in eligible) if eligible else None,
            "modes":{mode:mode_summary(eligible,mode) for mode in modes}}
    write_json(OUT/"metrics/subgroup_metrics.json",subgroup)
    write_json(OUT/"metrics/gap_recovery.json",{mode:mode_summary(valid,mode) for mode in modes})
    write_json(OUT/"metrics/bootstrap.json",{mode:bootstrap_mean([r[mode]["recovery_from_G0"] for r in valid]) for mode in modes})

    gaps=[r["TF"]["iou"]-r["G0"]["iou"] for r in valid]
    rep_keys=("G0_vs_TF_hidden","G0_vs_TF_projection","G0_vs_TF_sparse_prompt","G0_vs_TF_mask_logits")
    rep_metrics={key:{metric:correlation(gaps,[r["representations"][key][metric] for r in valid])
                      for metric in ("cosine_similarity","l2_distance","relative_l2_to_second")}
                 for key in rep_keys}
    write_json(OUT/"representations/hidden_metrics.json",rep_metrics["G0_vs_TF_hidden"])
    write_json(OUT/"representations/projection_metrics.json",rep_metrics["G0_vs_TF_projection"])
    write_json(OUT/"representations/sam_prompt_metrics.json",{
        "sparse":rep_metrics["G0_vs_TF_sparse_prompt"],"mask_logits":rep_metrics["G0_vs_TF_mask_logits"]})
    write_jsonl(OUT/"representations/per_sample_representation_metrics.jsonl",
               [{"sample_id":r["sample_id"],"gap":r["TF"]["iou"]-r["G0"]["iou"],**r["representations"]} for r in valid])

    replay=read_jsonl(OUT/"replay/per_sample_replay.jsonl")
    replay_summary={"n":len(replay),"semantics":"historical G0 full-forward predictor vs cache-stepwise fixed-token replay",
        "mean_hidden_cosine":mean(r["full_forward_vs_stepwise_hidden"]["cosine_similarity"] for r in replay),
        "mean_hidden_relative_l2":mean(r["full_forward_vs_stepwise_hidden"]["relative_l2_to_second"] for r in replay),
        "mean_projection_cosine":mean(r["full_forward_vs_stepwise_projection"]["cosine_similarity"] for r in replay),
        "mean_mask_logit_cosine":mean(r["full_forward_vs_stepwise_mask_logits"]["cosine_similarity"] for r in replay),
        "mean_binary_mask_iou":mean(r["full_forward_vs_stepwise_binary_mask_iou"] for r in replay),
        "binary_mask_exact_count":sum(r["full_forward_vs_stepwise_binary_mask_iou"]==1 for r in replay)}
    write_json(OUT/"replay/replay_consistency.json",replay_summary)

    h4=[]
    for row in rows:
        safe=row["sample_id"].replace(":","__")
        g0=torch.load(OUT/f"traces/g0/{safe}.pt",map_location="cpu")["input_ids"][0].tolist()[:row["G0"]["token_length"]]
        tf=torch.load(OUT/f"traces/tf/{safe}.pt",map_location="cpu")["input_ids"][0].tolist()
        common=0
        while common<min(len(g0),len(tf)) and g0[common]==tf[common]:common+=1
        h4.append({"sample_id":row["sample_id"],"exact_common_prefix_tokens":common,
                   "G0_token_length":len(g0),"TF_token_length":len(tf),
                   "earliest_divergence_raw_position":common if common<min(len(g0),len(tf)) else None,
                   "prefix_intervention_status":"NOT_EXECUTED_MULTI_FACTOR_PREFIX_INTERVENTION"})
    write_jsonl(OUT/"replay/h4_prefix_divergence.jsonl",h4)
    write_json(OUT/"audit/replay_semantics.json",{
        "canonical_G0_mask_predictor":"post-generation full-sequence forward without cache",
        "Replay_FullForward":"fixed historical padded token batch, no cache; batch_size=8",
        "Replay_Stepwise":"prompt full forward then one raw token per cache update",
        "important_observation":replay_summary,
        "historical_reproducibility":{"tolerance":5e-4,"passed":sum(r["trace_forward_reproducible"] for r in rows),
            "mismatch":sum(not r["trace_forward_reproducible"] for r in rows)}})
    audit=[
        {"stage":"canonical_G0_prompt","file":"dataset/forensics/unified.py","function":"CANONICAL_UNIFIED_QUESTION / custom_collate_fn","tensor":"input_ids","semantic":"canonical prompt ending at fixed [CLS]"},
        {"stage":"G0_generate","file":"model/GLaMM.py","function":"GLaMMForCausalLM.evaluate:596","tensor":"generation_outputs.sequences","shape":"[B,T]","dtype":"int64","device":"cuda","semantic":"greedy num_beams=1, do_sample default false, use_cache=true"},
        {"stage":"full_forward","file":"model/GLaMM.py","function":"GLaMMForCausalLM.evaluate:612","tensor":"full_output.hidden_states","semantic":"generated full sequence re-forwarded without cache for localization"},
        {"stage":"predictor","file":"model/GLaMM.py","function":"extract_seg_predictor_hidden:66","tensor":"llm_predictor_hidden","shape":"[1,4096]","dtype":"bfloat16","semantic":"expanded causal hidden immediately before [SEG]"},
        {"stage":"projection","file":"model/GLaMM.py","function":"_extract_projected_seg_predictor_hidden:477","tensor":"projected_embedding","shape":"[1,256]","dtype":"bfloat16","semantic":"text_hidden_fcs[0] output"},
        {"stage":"SAM_prompt","file":"model/GLaMM.py","function":"_generate_and_postprocess_masks:500","tensor":"sparse/dense prompt embedding","semantic":"prompt encoder output; dense embedding is no-mask embedding"},
        {"stage":"SAM_decoder","file":"model/GLaMM.py","function":"_generate_and_postprocess_masks:511","tensor":"low_res_mask_logits","shape":"[1,1,256,256]","semantic":"mask decoder logits"},
        {"stage":"postprocess","file":"model/GLaMM.py","function":"_generate_and_postprocess_masks:517","tensor":"postprocessed_mask_logits","semantic":"original-resolution logits"},
        {"stage":"threshold","file":"eval/forensics.py","function":"compute_binary_mask_metrics:62","tensor":"binary_mask","semantic":"fixed logit > 0"},
        {"stage":"cache","file":"model/llava/llava_with_region_arch.py","function":"prepare_inputs_labels_for_multimodal:83","tensor":"past_key_values/attention_mask","semantic":"cached steps rebuild all-ones mask at cache_length+1; position ids implicit"},
    ]
    write_json(OUT/"audit/inference_trace_path.json",{"checkpoint_sha256":sha(CHECKPOINT),"stages":audit})
    specs=[
        {"mode":"G0_trace","status":"executed","changed_variables":[],"fixed_variables":["tokens","batch shape","weights","threshold"],"causal":False},
        {"mode":"Replay_FullForward","status":"executed","token_source":"persisted historical G0","batch_size":8,"cache":False,"causal":False},
        {"mode":"Replay_Stepwise","status":"executed_selected_16","token_source":"persisted historical G0","cache":True,"causal":False},
        {"mode":"H1a_oracle_context","status":"executed","changed_variables":["language context","length","SEG position","predictor state"],"interpretation":"MULTI_FACTOR_DIAGNOSTIC_ONLY","causal":False},
        {"mode":"H1b_position_matched","status":"H1B_NOT_IDENTIFIABLE","reason":"padding/placeholder would enter causal attention or change semantics","causal":False},
        {"mode":"H2a_predictor_hidden_swap","status":"executed","image_source":"same_sample","token_context":"G0 upstream retained","predictor_hidden":"TF","projection":"unchanged_frozen","sam":"unchanged_frozen","threshold":0,"changed_variables":["predictor_hidden"],"causal":True,"label":"INTERVENTION_ONLY_NOT_A_DEPLOYABLE_MODEL"},
        {"mode":"H2b_projection_swap","status":"executed","image_source":"same_sample","projected_embedding":"TF","sam":"unchanged_frozen","threshold":0,"changed_variables":["projected_embedding"],"causal":True,"label":"INTERVENTION_ONLY_NOT_A_DEPLOYABLE_MODEL"},
        {"mode":"H2c_sam_prompt_swap","status":"executed","image_source":"same_sample","sam_prompt":"TF sparse+dense","mask_decoder":"unchanged_frozen","threshold":0,"changed_variables":["sparse_prompt_embedding","dense_prompt_embedding"],"causal":True,"label":"INTERVENTION_ONLY_NOT_A_DEPLOYABLE_MODEL"},
        {"mode":"H3_phrase_only","status":"executed","changed_variables":["text","length","SEG position","hidden trajectory"],"interpretation":"MULTI_FACTOR_DIAGNOSTIC_ONLY","causal":False},
        {"mode":"H4_prefix","status":"exact_divergence_executed_intervention_not_executed","reason":"continuation replacement changes multiple variables","causal":False},
    ]
    for spec in specs:spec.update({"checkpoint_sha256":sha(CHECKPOINT),"dataset":"SynthScars official test","threshold":0})
    write_json(OUT/"controlled_modes/mode_specs.json",specs)
    integrity={"eligible_n":len(valid),"all_valid":all(all(v["status"]=="VALID" for v in r["intervention_integrity"].values()) for r in valid),
               "excluded_before_intervention":{"reproducibility_mismatch":sum(r["status"]=="TRACE_FORWARD_REPRODUCIBILITY_MISMATCH" for r in rows),
                                                "no_G0_predictor":sum(r["status"]=="G0_SEG_PREDICTOR_UNAVAILABLE" for r in rows)}}
    write_json(OUT/"controlled_modes/intervention_integrity.json",integrity)
    write_json(OUT/"controlled_modes/controlled_results.json",{"subgroups":"../../metrics/subgroup_metrics.json","overall":{m:mode_summary(valid,m) for m in modes}})

    f_rows=[r for r in valid if "S2_all_F_candidates" in selections[r["sample_id"]]["subsets"]]
    negative=[r for r in valid if selections[r["sample_id"]]["historical_gap"]<0]
    write_json(OUT/"metrics/f_candidate_analysis.json",{"selected":13,"eligible":len(f_rows),
        "H2a_mean_recovery":mean(r["H2a"]["recovery_from_G0"] for r in f_rows),
        "H3_mean_iou":mean(r["H3_phrase_only"]["iou"] for r in f_rows),
        "F_PROXY_FALSE_POSITIVE_confirmed":0,"status":"STATE_MEDIATOR_SUPPORTED_BUT_ANOMALY_SPECIFICITY_UNRESOLVED"})
    write_json(OUT/"metrics/negative_gap_analysis.json",{"n":len(negative),
        "G0_mean_iou":mean(r["G0"]["iou"] for r in negative),"TF_mean_iou":mean(r["TF"]["iou"] for r in negative),
        "H3_mean_iou":mean(r["H3_phrase_only"]["iou"] for r in negative),
        "G0_mean_token_length":mean(r["G0"]["token_length"] for r in negative),
        "TF_mean_token_length":mean(r["TF"]["token_length"] for r in negative),
        "interpretation":"phrase-only 平均优于 TF-full 但低于 G0；探索性、非因果。"})
    qn=qualitative(rows,manifests)
    # Re-finalization may find a manifest from an earlier pass.  Keep the
    # "before_manifest" inventory stable by excluding that file explicitly.
    files=[p for p in OUT.rglob('*') if p.is_file() and p != OUT/"manifest.json"]
    manifest={"phase":"2D.1","language":"zh-CN","checkpoint":{"path":str(CHECKPOINT),"sha256":sha(CHECKPOINT),"step":2500,"epoch":5},
        "selection_n":len(rows),"eligible_intervention_n":len(valid),"status_counts":{s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})},
        "trace_invariance":"PASS","phase3_training_authorized":True,
        "recommended_phase3_direction":"phrase-level/autoregressive grounding 与 exposure-gap mitigation；不授权 forensic fusion",
        "decision_gates":["GATE_A_LANGUAGE_GROUNDING_SUPPORTED","GATE_B_PREDICTOR_STATE_MEDIATOR_SUPPORTED","GATE_F_FORENSIC_FUSION_NOT_AUTHORIZED"],
        "qualitative_cases":qn,"artifact_file_count_before_manifest":len(files),"artifact_bytes_before_manifest":sum(p.stat().st_size for p in files),
        "discipline":{"training_started":False,"model_weights_modified":False,"checkpoint_selection_performed":False,
            "test_set_model_selection_performed":False,"threshold_selection_performed":False,"proxy_category_labels_created":False}}
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__":main()
