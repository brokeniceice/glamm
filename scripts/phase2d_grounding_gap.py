#!/usr/bin/env python3
"""Build the artifact-driven Phase 2D autoregressive grounding-gap audit."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.phase2d_grounding_taxonomy import analyze_grounding_text, assign_taxonomy
from eval.phase2d_paired_metrics import aggregate_global, distribution, per_image_mask_metrics
from eval.phase2d_scope_validation import assert_exact_identity, index_unique

OUT = ROOT / "outputs/phase2d_grounding_gap"
MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
RAW = ROOT / "outputs/phase2b_legion_parity/official1000_raw"
C2 = ROOT / "outputs/phase2c_forensic_fusion/external/synthscars_official"
CHECKPOINT = ROOT / "checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt"
SEED = 20260811


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_difference(a, b, *, iterations=10000):
    rng = random.Random(SEED)
    a, b = list(a), list(b)
    if not a or not b:
        return {"status": "NOT_ESTIMABLE", "reason": "empty subgroup"}
    estimates = []
    for _ in range(iterations):
        estimates.append(mean(rng.choice(a) for _ in a) - mean(rng.choice(b) for _ in b))
    estimates.sort()
    return {
        "difference_in_mean_gap_a_minus_b": mean(a) - mean(b), "bootstrap_iterations": iterations,
        "bootstrap_seed": SEED, "ci95_percentile": [estimates[249], estimates[9749]],
        "interpretation": "探索性、非因果；置信区间不用于模型选择。",
    }


def subgroup(rows, predicate):
    selected = [row for row in rows if predicate(row)]
    return {
        "n": len(selected),
        "mean_g0_iou": mean(row["G0"]["foreground_iou"] for row in selected) if selected else None,
        "mean_tf_iou": mean(row["TF"]["foreground_iou"] for row in selected) if selected else None,
        "mean_gap": mean(row["paired"]["tf_minus_g0_foreground_iou"] for row in selected) if selected else None,
        "median_gap": median(row["paired"]["tf_minus_g0_foreground_iou"] for row in selected) if selected else None,
    }


def mode_specs():
    fixed = ["Phase2A step2500 checkpoint", "official SynthScars test split", "image input",
             "SEG projection", "SAM", "mask threshold=0"]
    specs = [
        {"mode": "H1", "status": "NOT_IMPLEMENTED_STRICT_INTERFACE_AUDIT", "image_input": "same",
         "language_prefix": "oracle full explanation", "generation_suffix": "autoregressive [SEG] continuation",
         "seg_token_source": "would be autoregressive", "fixed_variables": fixed,
         "changed_variables": ["language prefix", "generation start position", "SEG position unless constrained"],
         "reason": "现有 generate 接口没有只允许从 oracle 前缀生成 [SEG] 且保持历史终止语义的约束；自由续写会同时改变 suffix 长度和 SEG 位置。不能作为单因素实验。"},
        {"mode": "H2", "status": "H2_NOT_IDENTIFIABLE", "image_input": "same",
         "language_prefix": "persisted G0 tokens", "seg_token_source": "controlled/replayed",
         "fixed_variables": fixed,
         "changed_variables": ["KV/cache trajectory", "state construction path"],
         "reason": "把 G0 全序列重新 teacher-force 会同时改变 KV/cache 轨迹、位置处状态构造和推理执行路径；当前接口不能只替换 SEG state。"},
        {"mode": "H3", "status": "NOT_IMPLEMENTED_MULTI_FACTOR_DIAGNOSTIC_ONLY", "image_input": "same",
         "language_prefix": "annotation refs.phrase only", "seg_token_source": "teacher-forced",
         "fixed_variables": fixed,
         "changed_variables": ["language content", "sequence length", "SEG position", "state construction"],
         "reason": "refs.phrase 可得，但相对 TF-full 同时改变多个变量；不能回答单因素因果问题。"},
        {"mode": "H4", "status": "OFFLINE_TOKEN_PREFIX_ANALYSIS_ONLY", "image_input": "not run",
         "fixed_variables": ["persisted G0 tokens", "GT text artifact"],
         "changed_variables": ["not applicable: no intervention executed"],
         "reason": "G0 token 已保存，但 TF token 未保存；不加载 tokenizer/模型不能构造 exact token divergence，且当前 prefix replay 不隔离单一变量。"},
    ]
    for spec in specs:
        spec.update({
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": "07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072",
            "seg_projection": "unchanged / no intervention executed", "sam": "unchanged / no intervention executed",
            "mask_threshold": 0, "causal_interpretation_allowed": False,
        })
    return specs


def make_qualitative(rows, manifests):
    qdir = OUT / "qualitative"
    assets = qdir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r["paired"]["tf_minus_g0_foreground_iou"], reverse=True)
    picks = []
    for label, pool in [
        ("最大正差", ordered[:8]), ("最大负差", ordered[-8:]),
        ("TF高/G0低", [r for r in ordered if r["TF"]["foreground_iou"] >= .7 and r["G0"]["foreground_iou"] <= .3][:8]),
        ("TF高/G0高", [r for r in ordered if r["TF"]["foreground_iou"] >= .7 and r["G0"]["foreground_iou"] >= .7][:6]),
        ("TF低/G0低", [r for r in ordered if r["TF"]["foreground_iou"] <= .3 and r["G0"]["foreground_iou"] <= .3][:6]),
        ("B3正确/G0差", [r for r in ordered if r["classification"]["B3_correct"] and r["G0"]["foreground_iou"] <= .3][:8]),
    ]:
        for row in pool:
            if row["sample_id"] not in {item[1]["sample_id"] for item in picks}:
                picks.append((label, row))
    cards = []
    for idx, (subset, row) in enumerate(picks):
        manifest = manifests[row["sample_id"]]
        source = Path(manifest["image_path"])
        asset = assets / f"{idx:03d}_{source.name.rsplit('.', 1)[0]}.jpg"
        image = Image.open(source).convert("RGB")
        image.thumbnail((480, 480))
        draw = ImageDraw.Draw(image, "RGBA")
        sx, sy = image.width / manifest["image_variants"][0]["image_size"][0], image.height / manifest["image_variants"][0]["image_size"][1]
        for ref in manifest.get("refs", []):
            for polygon in ref.get("polygons", []):
                points = [(polygon[i] * sx, polygon[i + 1] * sy) for i in range(0, len(polygon), 2)]
                draw.polygon(points, fill=(0, 255, 0, 55), outline=(0, 255, 0, 210))
        image.save(asset, quality=88)
        cards.append(f'''<article><h2>{html.escape(subset)} · {html.escape(row['sample_id'])}</h2>
<img src="assets/{asset.name}" alt="原图与GT多边形叠加"><p>绿色区域：GT union mask；历史 artifact 未保存 G0/TF mask，不能重建。</p>
<p>G0 IoU={row['G0']['foreground_iou']:.4f}；TF IoU={row['TF']['foreground_iou']:.4f}；差值={row['paired']['tf_minus_g0_foreground_iou']:.4f}</p>
<p>G0 分类={row['G0']['cls_pred']}；B0={row['classification']['B0_pred']}；B3={row['classification']['B3_pred']}；taxonomy={row['text_analysis']['taxonomy']}</p>
<details><summary>G0 生成文本</summary><pre>{html.escape(row['G0']['generated_text'] or '')}</pre></details>
<details><summary>GT/TF 文本</summary><pre>{html.escape(row['TF']['target_text'] or '')}</pre></details></article>''')
    page = '''<!doctype html><meta charset="utf-8"><title>Phase 2D 定性审计</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto;background:#eee}article{background:white;padding:18px;margin:18px 0}img{max-width:480px}pre{white-space:pre-wrap}</style><h1>Phase 2D 定性审计</h1>''' + "".join(cards)
    (qdir / "index.html").write_text(page, encoding="utf-8")
    return len(cards)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output_dir).resolve()
    manifests = read_jsonl(MANIFEST)
    g0, g1, tf = (read_jsonl(RAW / name / "predictions.jsonl") for name in ("G0", "G1", "tf_full_context"))
    b0 = read_jsonl(C2 / "B0_phase2a/predictions.jsonl")
    b3 = read_jsonl(C2 / "B3_npr_srm/predictions.jsonl")
    for name, candidate in (("G0", g0), ("G1", g1), ("TF", tf), ("B0", b0), ("B3", b3)):
        assert_exact_identity(manifests, candidate, reference_name="official1000_manifest", candidate_name=name)
    mi, gi, g1i, tfi, b0i, b3i = (index_unique(rows, name=name) for rows, name in
        ((manifests, "manifest"), (g0, "G0"), (g1, "G1"), (tf, "TF"), (b0, "B0"), (b3, "B3")))
    paired = []
    for sample_id in mi:
        manifest, gr, g1r, tr = mi[sample_id], gi[sample_id], g1i[sample_id], tfi[sample_id]
        width, height = manifest["image_variants"][0]["image_size"]
        gm, g1m, tm = (per_image_mask_metrics(row, width * height) for row in (gr, g1r, tr))
        text = analyze_grounding_text(gr.get("generated_explanation") or "", manifest.get("refs", []))
        contradiction = gr["lm_verdict_pred"] == "real"
        taxonomy = assign_taxonomy(text, gm["foreground_iou"], tm["foreground_iou"], semantic_contradiction=contradiction)
        text["taxonomy"] = taxonomy
        row = {
            "sample_id": sample_id, "dataset": "SynthScars", "split": "official_test",
            "image_path": manifest["image_path"], "image_identity": manifest["image_identity"],
            "image_sha256": sha(Path(manifest["image_path"])), "target_representation": "union_mask",
            "mask_threshold": 0, "G0": {**gm, "generated_text": gr.get("generated_text"),
                "generated_token_ids": gr.get("generated_token_ids"), "seg_triggered": gr["seg_triggered"],
                "seg_token_position": gr.get("seg_position"), "predicted_mask_reference": None,
                "cls_pred": gr["cls_pred"], "lm_verdict_pred": gr["lm_verdict_pred"]},
            "G1": {**g1m, "seg_triggered": g1r["seg_triggered"], "seg_token_position": g1r.get("seg_position"),
                "predicted_mask_reference": None},
            "TF": {**tm, "target_text": manifest["explanation"], "target_sequence_identifier": sample_id,
                "seg_triggered": tr["seg_triggered"], "seg_token_position": None, "predicted_mask_reference": None},
            "paired": {"tf_minus_g0_foreground_iou": tm["foreground_iou"] - gm["foreground_iou"],
                "tf_minus_g0_per_image_fg_bg_miou": tm["per_image_fg_bg_miou"] - gm["per_image_fg_bg_miou"],
                "tf_minus_g0_foreground_f1": tm["foreground_f1"] - gm["foreground_f1"]},
            "classification": {"B0_pred": b0i[sample_id]["pred"], "B0_correct": b0i[sample_id]["pred"] == b0i[sample_id]["gt"],
                "B3_pred": b3i[sample_id]["pred"], "B3_correct": b3i[sample_id]["pred"] == b3i[sample_id]["gt"],
                "B3_fixed_B0": b0i[sample_id]["pred"] != b0i[sample_id]["gt"] and b3i[sample_id]["pred"] == b3i[sample_id]["gt"],
                "B3_lm_disagree": ("fake" if int(b3i[sample_id]["pred"]) == 1 else "real") != gr["lm_verdict_pred"]},
            "text_analysis": text,
        }
        paired.append(row)
    write_jsonl(OUT / "per_image/official1000_grounding_gap.jsonl", paired)
    severe = [row for row in paired if row["TF"]["foreground_iou"] >= .70 and row["G0"]["foreground_iou"] <= .30]
    write_jsonl(OUT / "per_image/tf_high_g0_low.jsonl", severe)
    gaps = [row["paired"]["tf_minus_g0_foreground_iou"] for row in paired]
    gap_summary = {"metric_semantics": {"paired": "per-image foreground IoU difference",
        "global": "aggregate TP/FP/FN; not a mean or per-image decomposition"}, "foreground_iou_gap": distribution(gaps),
        "global_metrics": {name: aggregate_global(rows) for name, rows in (("G0", g0), ("G1", g1), ("TF", tf))},
        "severe_preregistered": {**subgroup(paired, lambda r: r in severe), "criterion": "TF>=0.70 and G0<=0.30",
            "percentage": len(severe) / len(paired)},
        "ranked_sets": {"top_positive_sample_ids": [r["sample_id"] for r in sorted(paired,key=lambda r:r["paired"]["tf_minus_g0_foreground_iou"],reverse=True)[:25]],
            "near_zero_sample_ids": [r["sample_id"] for r in sorted(paired,key=lambda r:abs(r["paired"]["tf_minus_g0_foreground_iou"]))[:25]],
            "most_negative_sample_ids": [r["sample_id"] for r in sorted(paired,key=lambda r:r["paired"]["tf_minus_g0_foreground_iou"])[:25]]}}
    write_json(OUT / "metrics/paired_gap_summary.json", gap_summary)
    subgroups = {
        "B0_correct": subgroup(paired, lambda r:r["classification"]["B0_correct"]),
        "B0_wrong": subgroup(paired, lambda r:not r["classification"]["B0_correct"]),
        "B3_correct": subgroup(paired, lambda r:r["classification"]["B3_correct"]),
        "B3_fixed_B0": subgroup(paired, lambda r:r["classification"]["B3_fixed_B0"]),
        "B3_correct_G0_bad": subgroup(paired, lambda r:r["classification"]["B3_correct"] and r["G0"]["foreground_iou"]<=.30),
        "B3_LM_disagree": subgroup(paired, lambda r:r["classification"]["B3_lm_disagree"]),
        "target_present": subgroup(paired, lambda r:r["text_analysis"]["text_status"]=="TARGET_PRESENT"),
        "target_missing_or_partial": subgroup(paired, lambda r:r["text_analysis"]["text_status"] in {"MISSING_TARGET","PARTIAL_TARGET"}),
    }
    write_json(OUT / "metrics/subgroup_metrics.json", subgroups)
    b0_correct = [r["paired"]["tf_minus_g0_foreground_iou"] for r in paired if r["classification"]["B0_correct"]]
    b0_wrong = [r["paired"]["tf_minus_g0_foreground_iou"] for r in paired if not r["classification"]["B0_correct"]]
    text_ok = [r["paired"]["tf_minus_g0_foreground_iou"] for r in paired if r["text_analysis"]["text_status"]=="TARGET_PRESENT"]
    text_bad = [r["paired"]["tf_minus_g0_foreground_iou"] for r in paired if r["text_analysis"]["text_status"]!="TARGET_PRESENT"]
    write_json(OUT / "metrics/statistical_tests.json", {"B0_correct_vs_wrong":bootstrap_difference(b0_correct,b0_wrong),
        "target_present_vs_other":bootstrap_difference(text_ok,text_bad)})
    taxonomy_counts = Counter(r["text_analysis"]["taxonomy"] for r in severe)
    taxonomy = {}
    for label, count in taxonomy_counts.items():
        selected=[r for r in severe if r["text_analysis"]["taxonomy"]==label]
        taxonomy[label]={"count":count,"percentage":count/len(severe) if severe else None,
            "mean_G0_iou":mean(r["G0"]["foreground_iou"] for r in selected),"mean_TF_iou":mean(r["TF"]["foreground_iou"] for r in selected),
            "mean_gap":mean(r["paired"]["tf_minus_g0_foreground_iou"] for r in selected),"representative_sample_ids":[r["sample_id"] for r in selected[:5]]}
    write_json(OUT / "text_analysis/grounding_text_metrics.json", {"method":"deterministic_lexical_overlap_v1",
        "all_status_counts":Counter(r["text_analysis"]["text_status"] for r in paired), "limitations":"非语义裁判；同义词、空间关系和指代可能误判。"})
    write_json(OUT / "text_analysis/failure_taxonomy.json", {"scope":"preregistered severe subset","n":len(severe),"categories":taxonomy,
        "status":"DETERMINISTIC_PROXY_REQUIRES_HUMAN_REVIEW"})
    write_jsonl(OUT / "text_analysis/failure_taxonomy_samples.jsonl", severe)
    coupling_rows=[{"sample_id":r["sample_id"],**r["classification"],"G0_iou":r["G0"]["foreground_iou"],"TF_iou":r["TF"]["foreground_iou"],"gap":r["paired"]["tf_minus_g0_foreground_iou"]} for r in paired]
    write_jsonl(OUT / "classification_coupling/coupling_samples.jsonl",coupling_rows)
    write_json(OUT / "classification_coupling/coupling_metrics.json", {"scope_validation":"EXACT_SAMPLE_ID_MATCH_OFFICIAL1000",**subgroups})
    specs=mode_specs(); write_json(OUT / "controlled_modes/mode_specs.json", specs)
    write_json(OUT / "controlled_modes/controlled_mode_metrics.json", {"status":"NO_STRICT_SINGLE_FACTOR_MODE_EXECUTED","modes":{s['mode']:s['status'] for s in specs}})
    write_json(OUT / "audit/repository_audit.json", {"checkpoint":{"path":str(CHECKPOINT),"sha256":sha(CHECKPOINT),"step":2500,"epoch":5},
        "implementations":{"G0":"eval/forensics.py:evaluate_unified_fake_generation_localization","G1":"eval/forensics.py:evaluate_unified_gt_fake_prefix_localization",
        "TF":"eval/forensics_eval.py:GLaMMForensicsBackend.teacher_forced_localization","SEG_hidden":"model/GLaMM.py:_extract_projected_seg_predictor_hidden",
        "projection":"model/GLaMM.py:text_hidden_fcs[0]","SAM":"model/GLaMM.py:_generate_and_postprocess_masks"},
        "persisted":{"G0_G1_TF_rows":1000,"G0_G1_tokens":True,"TF_tokens":False,"predicted_masks":False},"training_started":False})
    write_json(OUT / "audit/inference_mode_audit.json", {"G0":"canonical unified prompt, free autoregressive generation","G1":"same prompt with structural [FAKE] continuation prefix",
        "TF":"teacher-forced [FAKE] + full GT explanation + [SEG] through causal forward","mask_threshold":0,"mode_specs":"../controlled_modes/mode_specs.json"})
    qualitative_count=make_qualitative(paired,mi)
    historical={name:json.loads((RAW/name/"metrics.json").read_text()) for name in ("G0","G1","tf_full_context")}
    reproduced={name:aggregate_global(rows) for name,rows in (("G0",g0),("G1",g1),("tf_full_context",tf))}
    checks={name:{"n_match":reproduced[name]["n"]==historical[name]["num_gt_fake"],
        "global_iou_abs_error":abs(reproduced[name]["global_foreground_iou"]-historical[name]["global_iou"]),
        "global_f1_abs_error":abs(reproduced[name]["global_foreground_f1"]-historical[name]["global_pixel_f1"])} for name in reproduced}
    repro={"status":"MATCH" if all(v["n_match"] and v["global_iou_abs_error"]<1e-12 and v["global_f1_abs_error"]<1e-12 for v in checks.values()) else "REPRODUCIBILITY_MISMATCH",
        "method":"re-aggregate persisted exact TP/FP/FN counts; no repeated model inference","checks":checks}
    write_json(OUT / "metrics/reproducibility.json",repro)
    manifest_out={"phase":"2D","language":"zh-CN","seed":SEED,"checkpoint_sha256":sha(CHECKPOINT),"dataset_manifest_sha256":sha(MANIFEST),
        "scope":{"dataset":"SynthScars","split":"official test","n":1000,"target":"union mask","threshold":0},"reproducibility":repro["status"],
        "artifacts":{"paired":"complete","controlled_inference":"not executed; no strict isolated mode through current interface","hidden_state_diagnostics":"NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS",
            "predicted_mask_visualization":"NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS","qualitative_gt_overlays":qualitative_count},
        "discipline":{"training_started":False,"model_weights_modified":False,"checkpoint_selection_performed":False,"test_set_model_selection_performed":False,"proxy_category_labels_created":False}}
    write_json(OUT / "manifest.json",manifest_out)
    print(json.dumps({"output":str(OUT),"n":len(paired),"severe":len(severe),"repro":repro["status"],"qualitative":qualitative_count},ensure_ascii=False))


if __name__ == "__main__":
    main()
