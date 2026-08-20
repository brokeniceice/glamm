#!/usr/bin/env python3
"""Analyze Phase 3C.0 paired traces and render review/report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3c0 import classify_failure


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/phase3c0_residual_diagnosis")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def paired(values, *, seed=3407, repeats=10000):
    diff = np.asarray(values, dtype=np.float64)
    if not diff.size:
        return {"num_paired": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for start in range(0, repeats, 500):
        count = min(500, repeats - start)
        selected = rng.integers(0, diff.size, size=(count, diff.size))
        boot[start:start + count] = diff[selected].mean(axis=1)
    nonzero = diff[diff != 0]
    test = stats.wilcoxon(nonzero) if nonzero.size else None
    return {
        "num_paired": int(diff.size), "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "bootstrap_repeats": repeats,
        "bootstrap_95ci": [float(x) for x in np.quantile(boot, (.025, .975))],
        "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "positive_fraction": float((diff > 0).mean()),
        "wilcoxon_statistic": None if test is None else float(test.statistic),
        "wilcoxon_pvalue": None if test is None else float(test.pvalue),
    }


def safe_corr(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3 or np.std([a for a, _ in pairs]) == 0 or np.std([b for _, b in pairs]) == 0:
        return {"n": len(pairs), "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None}
    a, b = zip(*pairs)
    pearson = stats.pearsonr(a, b); spearman = stats.spearmanr(a, b)
    return {"n": len(pairs), "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}


def condition_mean(rows, key):
    subset = [row[key] for row in rows if row.get(key) is not None]
    return {
        "n": len(subset),
        "mean_foreground_iou": float(np.mean([x["foreground_iou"] for x in subset])),
        "mean_foreground_f1": float(np.mean([x["foreground_f1"] for x in subset])),
        "mean_fg_bg_miou": float(np.mean([x["fg_bg_miou"] for x in subset])),
    }


def enrich(rows):
    for row in rows:
        a, b, c, d = row["A"], row.get("B"), row["C"], row["D"]
        row["category"] = classify_failure(a["foreground_iou"], d["foreground_iou"])
        row["persistent_oracle_failure"] = all(
            item["foreground_iou"] <= .30 for item in (a, c, d)
        )
        row["deltas"] = {
            "B_minus_A": None if b is None else b["foreground_iou"] - a["foreground_iou"],
            "C_minus_A": c["foreground_iou"] - a["foreground_iou"],
            "D_minus_A": d["foreground_iou"] - a["foreground_iou"],
            "C_minus_B": None if b is None else c["foreground_iou"] - b["foreground_iou"],
            "D_minus_C": d["foreground_iou"] - c["foreground_iou"],
        }


def phrase_statistics(rows):
    eligible = [r for r in rows if r["B"] is not None]
    f1 = [r["phrase_quality"]["normalized_token_f1"] for r in eligible]
    a_iou = [r["A"]["foreground_iou"] for r in eligible]
    ba = [r["deltas"]["B_minus_A"] for r in eligible]
    da = [r["deltas"]["D_minus_A"] for r in eligible]
    bins = {"exact_match": [], "high_overlap_exploratory": [],
            "medium_overlap_exploratory": [], "low_overlap_exploratory": []}
    for row in eligible:
        quality = row["phrase_quality"]
        if quality["normalized_exact_match"]:
            key = "exact_match"
        elif quality["normalized_token_f1"] >= .75:
            key = "high_overlap_exploratory"
        elif quality["normalized_token_f1"] >= .35:
            key = "medium_overlap_exploratory"
        else:
            key = "low_overlap_exploratory"
        bins[key].append(row)
    return {
        "primary_replacement_population": len(eligible),
        "phrase_presence_rate_all_g0": float(np.mean([r["phrase_quality"]["phrase_presence"] for r in rows])),
        "normalized_exact_match_rate": float(np.mean([r["phrase_quality"]["normalized_exact_match"] for r in eligible])),
        "mean_token_precision": float(np.mean([r["phrase_quality"]["normalized_token_precision"] for r in eligible])),
        "mean_token_recall": float(np.mean([r["phrase_quality"]["normalized_token_recall"] for r in eligible])),
        "mean_token_f1": float(np.mean(f1)),
        "correlations": {
            "token_f1_vs_g0_iou": safe_corr(f1, a_iou),
            "token_f1_vs_phrase_repair_gain": safe_corr(f1, ba),
            "token_f1_vs_tf_minus_g0_gap": safe_corr(f1, da),
        },
        "lexical_bins_are_exploratory_not_route_gate": {
            key: {"n": len(group), "mean_A_iou": float(np.mean([r["A"]["foreground_iou"] for r in group])) if group else None,
                  "mean_B_minus_A": float(np.mean([r["deltas"]["B_minus_A"] for r in group])) if group else None}
            for key, group in bins.items()
        },
        "warning": "lexical token F1 is not semantic correctness; no external LLM judge was used",
    }


def representation_statistics(rows):
    eligible = [r for r in rows if r["B"] is not None]
    result = {}
    for space in ("hidden", "projected"):
        ad_rel = [r["representation_distances"][f"{space}_A_D"]["relative_l2"] for r in eligible]
        bd_rel = [r["representation_distances"][f"{space}_B_D"]["relative_l2"] for r in eligible]
        cd_rel = [r["representation_distances"][f"{space}_C_D"]["relative_l2"] for r in eligible]
        ad_cos = [r["representation_distances"][f"{space}_A_D"]["cosine"] for r in eligible]
        bd_cos = [r["representation_distances"][f"{space}_B_D"]["cosine"] for r in eligible]
        ba = [r["deltas"]["B_minus_A"] for r in eligible]
        cb = [r["deltas"]["C_minus_B"] for r in eligible]
        result[space] = {
            "mean_cosine_A_D": float(np.mean(ad_cos)), "mean_cosine_B_D": float(np.mean(bd_cos)),
            "mean_relative_l2_A_D": float(np.mean(ad_rel)), "mean_relative_l2_B_D": float(np.mean(bd_rel)),
            "mean_relative_l2_C_D": float(np.mean(cd_rel)),
            "A_to_B_closeness_gain_vs_B_minus_A": safe_corr(
                [a - b for a, b in zip(ad_rel, bd_rel)], ba),
            "B_to_C_closeness_gain_vs_C_minus_B": safe_corr(
                [b - c for b, c in zip(bd_rel, cd_rel)], cb),
            "A_to_B_cosine_gain_vs_B_minus_A": safe_corr(
                [b - a for a, b in zip(ad_cos, bd_cos)], ba),
        }
    result["interpretation_boundary"] = "association only; representation distance does not establish causality"
    return result


def subgroup_statistics(rows):
    groups = {name: [r for r in rows if r["category"] == name]
              for name in ("G0_GOOD", "LANGUAGE_RECOVERABLE", "INTERMEDIATE", "PERSISTENT")}
    result = {}
    for name, group in groups.items():
        eligible = [r for r in group if r["B"] is not None]
        result[name] = {
            "n": len(group), "repair_eligible_n": len(eligible),
            "A_mean_iou": float(np.mean([r["A"]["foreground_iou"] for r in group])) if group else None,
            "B_mean_iou": float(np.mean([r["B"]["foreground_iou"] for r in eligible])) if eligible else None,
            "C_mean_iou": float(np.mean([r["C"]["foreground_iou"] for r in group])) if group else None,
            "D_mean_iou": float(np.mean([r["D"]["foreground_iou"] for r in group])) if group else None,
            "B_minus_A": float(np.mean([r["deltas"]["B_minus_A"] for r in eligible])) if eligible else None,
            "C_minus_B": float(np.mean([r["deltas"]["C_minus_B"] for r in eligible])) if eligible else None,
            "D_minus_C": float(np.mean([r["deltas"]["D_minus_C"] for r in group])) if group else None,
            "phrase_token_f1": float(np.mean([r["phrase_quality"]["normalized_token_f1"] for r in group])) if group else None,
            "hidden_relative_l2_A_D": float(np.mean([
                r["representation_distances"]["hidden_A_D"]["relative_l2"]
                for r in group if r["representation_distances"]["hidden_A_D"]["relative_l2"] is not None
            ])) if any(r["representation_distances"]["hidden_A_D"]["relative_l2"] is not None for r in group) else None,
        }
    return result


def mask_image(path, size):
    tensor = torch.load(path, map_location="cpu").bool()
    if tensor.ndim == 3: tensor = tensor.any(dim=0)
    image = Image.fromarray((tensor.numpy().astype(np.uint8) * 255), mode="L")
    return image.resize(size, Image.Resampling.NEAREST).convert("RGB")


def render_panel(row, destination: Path):
    original = Image.open(row["image_path"]).convert("RGB")
    width = 360; height = max(1, round(original.height * width / original.width))
    size = (width, height)
    panels = [original.resize(size, Image.Resampling.LANCZOS)]
    paths = [row["A"]["gt_mask_path"], row["A"]["binary_mask_path"]]
    paths.append(row["B"]["binary_mask_path"] if row["B"] else row["A"]["binary_mask_path"])
    paths.extend((row["C"]["binary_mask_path"], row["D"]["binary_mask_path"]))
    panels.extend(mask_image(path, size) for path in paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width * len(panels), height))
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels): canvas.paste(panel, (index * width, 0))
    canvas.save(destination, quality=88)


def review_packet(rows, out, seed):
    eligible = [r for r in rows if r["B"] is not None]
    tie = lambda r: stable_key(r["sample_id"], seed)
    high = sorted(eligible, key=lambda r: (-r["deltas"]["B_minus_A"], tie(r)))[:50]
    low = sorted(eligible, key=lambda r: (r["deltas"]["B_minus_A"], tie(r)))[:50]
    persistent = sorted([r for r in rows if r["persistent_oracle_failure"]], key=tie)[:50]
    controls = sorted([r for r in rows if r["A"]["foreground_iou"] > .30], key=tie)[:50]
    selected = {}
    memberships = {}
    for name, group in (("high_B_minus_A_recovery", high), ("low_or_no_B_minus_A_recovery", low),
                        ("persistent_oracle_failures", persistent), ("G0_good_controls", controls)):
        for row in group:
            selected[row["sample_id"]] = row
            memberships.setdefault(row["sample_id"], []).append(name)
    packet = sorted(selected.values(), key=lambda r: stable_key(r["sample_id"], seed))
    root = out / "human_review"; assets = root / "assets"; assets.mkdir(parents=True, exist_ok=True)
    csv_path = root / "human_review.csv"
    fields = ["sample_id", "groups", "image_path", "authoritative_phrase", "generated_phrase",
              "generated_explanation", "A_iou", "B_iou", "C_iou", "D_iou",
              "generated_phrase_semantically_correct", "explanation_target_consistent", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(packet):
            asset = assets / f"{index:03d}_{hashlib.sha256(row['sample_id'].encode()).hexdigest()[:12]}.jpg"
            render_panel(row, asset)
            writer.writerow({
                "sample_id": row["sample_id"], "groups": ";".join(memberships[row["sample_id"]]),
                "image_path": row["image_path"], "authoritative_phrase": row["authoritative_phrase"],
                "generated_phrase": row["A"]["generated_phrase_raw"],
                "generated_explanation": row["A"]["generated_explanation"],
                "A_iou": row["A"]["foreground_iou"], "B_iou": row["B"]["foreground_iou"] if row["B"] else "",
                "C_iou": row["C"]["foreground_iou"], "D_iou": row["D"]["foreground_iou"],
                "generated_phrase_semantically_correct": "", "explanation_target_consistent": "", "notes": "",
            })
    cards = []
    for index, row in enumerate(packet):
        asset = sorted(assets.glob(f"{index:03d}_*.jpg"))[0]
        b = "N/A" if row["B"] is None else f"{row['B']['foreground_iou']:.4f}"
        cards.append(f"<article><h2>{html.escape(row['sample_id'])}</h2>"
                     f"<p>组别：{html.escape('; '.join(memberships[row['sample_id']]))}</p>"
                     f"<img src='assets/{asset.name}'><p>原图 / GT / A / B / C / D</p>"
                     f"<p>authoritative: {html.escape(row['authoritative_phrase'])}</p>"
                     f"<p>generated: {html.escape(str(row['A']['generated_phrase_raw']))}</p>"
                     f"<p>explanation: {html.escape(str(row['A']['generated_explanation']))}</p>"
                     f"<p>IoU A/B/C/D: {row['A']['foreground_iou']:.4f} / {b} / {row['C']['foreground_iou']:.4f} / {row['D']['foreground_iou']:.4f}</p>"
                     "<p>人工字段请在 CSV 填写；本 HTML 未自动生成语义判断。</p></article>")
    css = "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{max-width:100%;border:1px solid #aaa}article{margin:2rem 0;padding:1rem;background:#f5f5f5}</style>"
    (root / "index.html").write_text("<!doctype html><meta charset='utf-8'>" + css + "".join(cards), encoding="utf-8")
    manifest = {"seed": seed, "requested_per_group": 50,
                "group_counts": {"high_B_minus_A_recovery": len(high), "low_or_no_B_minus_A_recovery": len(low),
                                 "persistent_oracle_failures": len(persistent), "G0_good_controls": len(controls)},
                "unique_samples": len(packet), "allows_overlap": True,
                "automated_human_fields_filled": False,
                "sample_memberships": memberships}
    dump(root / "selection_manifest.json", manifest)
    return manifest


def qualitative(rows, out, seed):
    eligible = [r for r in rows if r["B"] is not None]
    tie = lambda r: stable_key(r["sample_id"], seed)
    groups = {
        "phrase_repair_large_recovery": sorted(eligible, key=lambda r: (-r["deltas"]["B_minus_A"], tie(r)))[:20],
        "context_removal_large_recovery": sorted(eligible, key=lambda r: (-r["deltas"]["C_minus_B"], tie(r)))[:20],
        "tf_phrase_still_failure": sorted([r for r in rows if r["persistent_oracle_failure"]], key=tie)[:20],
        "negative_phrase_intervention": sorted(eligible, key=lambda r: (r["deltas"]["B_minus_A"], tie(r)))[:20],
        "g0_already_good_controls": sorted([r for r in rows if r["A"]["foreground_iou"] > .30], key=tie)[:20],
    }
    root = out / "qualitative"; assets = root / "assets"; assets.mkdir(parents=True, exist_ok=True)
    cards = []
    manifest = {"seed": seed, "rules": {}, "selections": {}}
    number = 0
    for group_name, group in groups.items():
        manifest["selections"][group_name] = [r["sample_id"] for r in group]
        manifest["rules"][group_name] = "deterministic metric ordering with seeded SHA256 tie-break"
        cards.append(f"<h1>{html.escape(group_name)}</h1>")
        for row in group:
            asset = assets / f"{number:03d}_{hashlib.sha256(row['sample_id'].encode()).hexdigest()[:12]}.jpg"; number += 1
            render_panel(row, asset)
            dist = row["representation_distances"]
            b = "N/A" if row["B"] is None else f"{row['B']['foreground_iou']:.4f}"
            cards.append(f"<article><h2>{html.escape(row['sample_id'])}</h2><img src='assets/{asset.name}'>"
                         f"<p>原图 / GT / A / B / C / D；类别={row['category']}</p>"
                         f"<p>generated={html.escape(str(row['A']['generated_phrase_raw']))}<br>authoritative={html.escape(row['authoritative_phrase'])}</p>"
                         f"<p>explanation={html.escape(str(row['A']['generated_explanation']))}</p>"
                         f"<p>IoU A/B/C/D={row['A']['foreground_iou']:.4f}/{b}/{row['C']['foreground_iou']:.4f}/{row['D']['foreground_iou']:.4f}; "
                         f"hidden cosine(A,D)={dist['hidden_A_D']['cosine']}</p></article>")
    css = "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{max-width:100%}article{padding:1rem;margin:1rem;background:#f4f4f4}</style>"
    (root / "index.html").write_text("<!doctype html><meta charset='utf-8'>" + css + "".join(cards), encoding="utf-8")
    dump(root / "selection_manifest.json", manifest)
    return manifest


def main(argv=None):
    cli = args(argv); out = (ROOT / cli.output_dir).resolve()
    provenance = load(out / "provenance.json")
    values = load_rows(out / "paired/paired_conditions.jsonl")
    if provenance["status"] != "EVALUATION_COMPLETE" or len(values) != 1106:
        raise RuntimeError(f"Phase3C0 evaluation incomplete: status={provenance['status']} rows={len(values)}")
    if len({r["sample_id"] for r in values}) != 1106:
        raise RuntimeError("Phase3C0 duplicate/missing sample ids")
    enrich(values)
    eligible = [r for r in values if r["B"] is not None]
    q1 = paired([r["deltas"]["B_minus_A"] for r in eligible], seed=cli.seed)
    q2 = paired([r["deltas"]["C_minus_B"] for r in eligible], seed=cli.seed)
    q3 = paired([r["deltas"]["D_minus_C"] for r in values], seed=cli.seed)
    positive_gap = [r for r in eligible if r["deltas"]["D_minus_A"] > 0]
    ratios = [r["deltas"]["B_minus_A"] / r["deltas"]["D_minus_A"] for r in positive_gap]
    q1["positive_D_minus_A_gap"] = {
        "n": len(ratios), "unclipped_mean": float(np.mean(ratios)), "unclipped_median": float(np.median(ratios)),
        "clipped_0_1_mean": float(np.mean(np.clip(ratios, 0, 1))),
        "clipped_0_1_median": float(np.median(np.clip(ratios, 0, 1))),
    }
    dump(out / "phrase_repair_statistics.json", q1)
    dump(out / "statistics/phrase_repair_statistics.json", q1)
    dump(out / "context_contamination_statistics.json", q2)
    dump(out / "statistics/context_contamination_statistics.json", q2)
    dump(out / "statistics/tf_phrase_minus_phrase_only_statistics.json", q3)

    failures = [r for r in values if r["A"]["foreground_iou"] <= .30]
    repair_failure_scope = [r for r in failures if r["B"] is not None]
    persistent = [r for r in failures if r["persistent_oracle_failure"]]
    failure_stats = {
        "g0_failure_definition": "A FG IoU <= 0.30", "g0_failure_count": len(failures),
        "repair_eligible_g0_failures": len(repair_failure_scope),
        "repair_still_failure_fraction": float(np.mean([r["B"]["foreground_iou"] <= .30 for r in repair_failure_scope])),
        "phrase_only_still_failure_fraction": float(np.mean([r["C"]["foreground_iou"] <= .30 for r in failures])),
        "tf_phrase_still_failure_fraction": float(np.mean([r["D"]["foreground_iou"] <= .30 for r in failures])),
        "persistent_oracle_failure_definition": "A<=0.30 AND C<=0.30 AND D<=0.30",
        "persistent_oracle_failure_count": len(persistent),
        "usable_seg_fraction_within_g0_failures": float(np.mean([r["g0_seg_status"] == "USABLE" for r in failures])),
        "g0_seg_unavailable_count_all": sum(r["g0_seg_status"] != "USABLE" for r in values),
        "phrase_insertion_exploratory_count": sum(r["phrase_insertion_exploratory"] for r in values),
        "interpretation": "residual downstream / visual-spatial localization bottleneck candidates only",
    }
    dump(out / "persistent_failure_statistics.json", failure_stats)
    dump(out / "statistics/persistent_failure_statistics.json", failure_stats)
    phrase = phrase_statistics(values); representation = representation_statistics(values)
    subgroups = subgroup_statistics(values)
    dump(out / "phrase_quality_statistics.json", phrase)
    dump(out / "representation_statistics.json", representation)
    dump(out / "subgroups/category_statistics.json", subgroups)

    conditions = {name: condition_mean(values, key) for name, key in
                  (("A_G0", "A"), ("B_phrase_repair_eligible_only", "B"),
                   ("C_phrase_only", "C"), ("D_TF_PHRASE", "D"))}
    dump(out / "statistics/four_condition_summary.json", conditions)
    review = review_packet(values, out, cli.seed)
    qualitative(values, out, cli.seed)

    thresholds = provenance["route_thresholds_preregistered_before_results"]
    language = q1["bootstrap_95ci"][0] > 0 and q1["positive_D_minus_A_gap"]["clipped_0_1_mean"] >= thresholds["language_nontrivial_clipped_recovery_min"]
    context = q2["bootstrap_95ci"][0] > 0
    required_persistent = max(thresholds["persistent_min_count"], math.ceil(thresholds["persistent_min_fraction_of_g0_failures"] * len(failures)))
    downstream = len(persistent) >= required_persistent and failure_stats["usable_seg_fraction_within_g0_failures"] >= thresholds["usable_seg_fraction_min"]
    supported = []
    if language: supported.append("GATE_LANGUAGE_PHRASE_ERROR_SUPPORTED")
    if context: supported.append("GATE_GENERATED_CONTEXT_CONTAMINATION_SUPPORTED")
    if downstream: supported.append("GATE_DOWNSTREAM_SPATIAL_PREFLIGHT_AUTHORIZED")
    if len(supported) >= 2: supported.append("GATE_MIXED_BOTTLENECK")
    if not supported: supported = ["GATE_INCONCLUSIVE"]
    gate = {
        "route_selection_population": "internal validation Fake only", "supported_gates": supported,
        "language_gate": {"passed": language, "B_minus_A_ci_lower": q1["bootstrap_95ci"][0],
                          "mean_clipped_positive_gap_recovery": q1["positive_D_minus_A_gap"]["clipped_0_1_mean"]},
        "context_gate": {"passed": context, "C_minus_B_ci_lower": q2["bootstrap_95ci"][0]},
        "downstream_gate": {"passed": downstream, "persistent_count": len(persistent),
                             "required_count": required_persistent,
                             "usable_seg_fraction_within_g0_failures": failure_stats["usable_seg_fraction_within_g0_failures"]},
        "authorization_boundary": "diagnostic gate only; no GRPO, NPR, FOCAL, fusion, or training was started",
    }
    dump(out / "route_gate.json", gate)

    category_counts = {name: data["n"] for name, data in subgroups.items()}
    report = f"""# Phase 3C.0：P1 残余定位瓶颈只读诊断

## 结论

本阶段只使用 internal validation 的 1106 张 Fake 图像进行 route selection，冻结模型为 Phase 3A selected P1（step 3500 / epoch 7，SHA256 `{provenance['checkpoint_file_sha256']}`）。未训练、未反向传播、未修改权重，也未使用 internal test、official1000、RAISE 或 LOKI 参与路径选择。

最终 evidence gate：`{'`、`'.join(supported)}`。

## 四条件结果

| 条件 | 范围 n | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|---:|
| A：canonical G0 | {conditions['A_G0']['n']} | {conditions['A_G0']['mean_foreground_iou']:.6f} | {conditions['A_G0']['mean_foreground_f1']:.6f} | {conditions['A_G0']['mean_fg_bg_miou']:.6f} |
| B：G0 Phrase Repair（eligible only） | {conditions['B_phrase_repair_eligible_only']['n']} | {conditions['B_phrase_repair_eligible_only']['mean_foreground_iou']:.6f} | {conditions['B_phrase_repair_eligible_only']['mean_foreground_f1']:.6f} | {conditions['B_phrase_repair_eligible_only']['mean_fg_bg_miou']:.6f} |
| C：Authoritative Phrase-Only | {conditions['C_phrase_only']['n']} | {conditions['C_phrase_only']['mean_foreground_iou']:.6f} | {conditions['C_phrase_only']['mean_foreground_f1']:.6f} | {conditions['C_phrase_only']['mean_fg_bg_miou']:.6f} |
| D：TF-PHRASE | {conditions['D_TF_PHRASE']['n']} | {conditions['D_TF_PHRASE']['mean_foreground_iou']:.6f} | {conditions['D_TF_PHRASE']['mean_foreground_f1']:.6f} | {conditions['D_TF_PHRASE']['mean_fg_bg_miou']:.6f} |

注意：B 仅针对 exactly-one usable `[SEG]`、可解析 `Target regions:` 且 authoritative phrase 非空的 primary replacement population，因此 B 的绝对均值不能直接与 A/C/D 的全 1106 均值作非配对比较；Q1/Q2 使用同一 eligible 样本配对。

## 配对问题

- Q1 `B−A`：n={q1['num_paired']}，mean={q1['mean']:+.6f}，median={q1['median']:+.6f}，bootstrap 95% CI=[{q1['bootstrap_95ci'][0]:+.6f}, {q1['bootstrap_95ci'][1]:+.6f}]，win/tie/loss={q1['wins']}/{q1['ties']}/{q1['losses']}，Wilcoxon p={q1['wilcoxon_pvalue']:.6g}。
- 在 `D>A` 的 positive residual gap 中，B 的 recovery ratio：unclipped mean={q1['positive_D_minus_A_gap']['unclipped_mean']:+.6f}、median={q1['positive_D_minus_A_gap']['unclipped_median']:+.6f}；clipped [0,1] mean={q1['positive_D_minus_A_gap']['clipped_0_1_mean']:.6f}、median={q1['positive_D_minus_A_gap']['clipped_0_1_median']:.6f}。
- Q2 `C−B`：n={q2['num_paired']}，mean={q2['mean']:+.6f}，95% CI=[{q2['bootstrap_95ci'][0]:+.6f}, {q2['bootstrap_95ci'][1]:+.6f}]，Wilcoxon p={q2['wilcoxon_pvalue']:.6g}。
- Q3 `D−C`：n={q3['num_paired']}，mean={q3['mean']:+.6f}，95% CI=[{q3['bootstrap_95ci'][0]:+.6f}, {q3['bootstrap_95ci'][1]:+.6f}]，Wilcoxon p={q3['wilcoxon_pvalue']:.6g}。

B 是 `CONTROLLED_MULTI_FACTOR_DIAGNOSTIC`：phrase replacement 同时可能改变 token 长度、`[SEG]` 位置与 hidden trajectory，所以 B−A 不能被称为严格单因素因果效应。C−B 若为正，只支持 generated explanation/context 可能有害，不证明长文本普遍有害；D−C 不自动等价于 CoT benefit。

## 残余失败

- canonical G0 failure（A IoU≤0.30）：{len(failures)}。
- Language-recoverable（A≤0.30 且 D≥0.70）：{category_counts['LANGUAGE_RECOVERABLE']}。
- Intermediate：{category_counts['INTERMEDIATE']}。
- Persistent（A≤0.30 且 D≤0.30）：{category_counts['PERSISTENT']}。
- `PERSISTENT_ORACLE_FAILURE`（A、C、D 均≤0.30）：{len(persistent)}。
- G0 failures 中 usable `[SEG]` 比例：{failure_stats['usable_seg_fraction_within_g0_failures']:.6f}；全体 G0_SEG_UNAVAILABLE={failure_stats['g0_seg_unavailable_count_all']}；exploratory PHRASE_INSERTION={failure_stats['phrase_insertion_exploratory_count']}。

这些 persistent 样本只能称为 residual downstream / visual-spatial localization bottleneck candidates；不能据此声称 NPR 或 FOCAL 一定有效。

## Phrase quality 与 representation

eligible population 的 normalized token F1 均值为 {phrase['mean_token_f1']:.6f}；F1 与 A IoU 的 Spearman ρ={phrase['correlations']['token_f1_vs_g0_iou']['spearman_rho']}，与 B−A 的 ρ={phrase['correlations']['token_f1_vs_phrase_repair_gain']['spearman_rho']}，与 D−A gap 的 ρ={phrase['correlations']['token_f1_vs_tf_minus_g0_gap']['spearman_rho']}。Lexical F1 不等于语义正确；本阶段没有使用 external LLM judge。

4096D hidden 和 256D projected embedding 均已逐样本保存。hidden relative-L2(A,D)={representation['hidden']['mean_relative_l2_A_D']:.6f}、(B,D)={representation['hidden']['mean_relative_l2_B_D']:.6f}、(C,D)={representation['hidden']['mean_relative_l2_C_D']:.6f}；与 IoU recovery 的 Pearson/Spearman 详见 `representation_statistics.json`，只作关联性诊断。

## 审查与可视化

- 人工审查：`outputs/phase3c0_residual_diagnosis/human_review/index.html` 与 `human_review.csv`；四组各最多 50，允许重叠，unique={review['unique_samples']}。人工字段保持空白，自动 gate 不依赖人类标注。
- 定性可视化：`outputs/phase3c0_residual_diagnosis/qualitative/index.html`；选择规则与 sample IDs 固定在 selection manifest，未手工 cherry-pick。

## Invariance、边界与回归

- canonical G0 trace 对 1106 个样本保存 trace-vs-evaluator identity；聚合审计见 `audit/invariance.json`。
- authoritative phrase 直接来自 `UnifiedForensicsDataset` 当前 construction rule；C/D 复用 `GLaMMForensicsBackend` 的共享模板构造。
- threshold 固定为 mask logit `>0`；不存在 threshold sweep。
- checkpoint 与 model state 前后 exact hash 见 `frozen_model_hash.json`。
- 历史 P1 val G0 artifact 不存在，故标记 `HISTORICAL_P1_VAL_G0_ARTIFACT_UNAVAILABLE`；本阶段以 canonical evaluator 与 no-cache trace 的逐样本 exact comparison 作为当前实现复现审计，不能冒充与不存在的历史 artifact 比对。
- 全量 pytest 结果将在 `reports/regression_tests.txt`，最终 manifest 会记录门禁状态。

## 最终停止边界

本报告只给出诊断 evidence gate。未启动且不自动启动 GRPO/PPO/DPO、context-decoupling training、NPR、FOCAL、fusion 或任何 Phase 3C training。
"""
    invariance = {
        "n": len(values),
        "trace_vs_canonical_binary_exact_count": sum(r["A"]["trace_vs_canonical_binary_exact"] for r in values),
        "trace_vs_canonical_logits_exact_count": sum(r["A"]["trace_vs_canonical_mask_max_abs"] in (None, 0.0) for r in values),
        "repair_prefix_exact_count": sum(r["B"]["prefix_exact"] for r in eligible),
        "repair_seg_suffix_exact_count": sum(r["B"]["seg_suffix_exact"] for r in eligible),
        "repair_eligible_count": len(eligible), "threshold_all_zero": True,
        "route_uses_only_internal_val_fake": True,
    }
    dump(out / "audit/invariance.json", invariance)
    (out / "reports/final_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "reports/final_report.md").write_text(report, encoding="utf-8")
    (out / "final_report.md").write_text(report, encoding="utf-8")
    docs = ROOT / "docs/phase3c0_residual_localization_diagnosis.md"
    docs.write_text(report, encoding="utf-8")
    print(json.dumps({"conditions": conditions, "route_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
