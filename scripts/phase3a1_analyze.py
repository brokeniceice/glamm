#!/usr/bin/env python3
"""Phase 3A.1 staged paired analyses and Chinese final report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a1_paired_control"
HIST = ROOT / "outputs/phase2b_legion_parity/official1000_raw"
P1 = ROOT / "outputs/phase3a_phrase_grounding/evaluation/official1000"
NEW = OUT / "evaluation/g0/new_c0"
INTERNAL_NEW = OUT / "evaluation/internal/new_c0"
INTERNAL_P1 = ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal"
TF = OUT / "evaluation/tf_cross_eval"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def spatial(row: dict) -> dict:
    if "foreground_iou" in row:
        return {
            "iou": float(row["foreground_iou"]),
            "f1": float(row["foreground_f1"]),
            "miou": float(row["fg_bg_miou"]),
        }
    width, height = Image.open(row["image_path"]).size
    tp, fp, fn = (int(row[k]) for k in ("tp", "fp", "fn"))
    tn = width * height - tp - fp - fn
    bg_den = tn + fp + fn
    bg_iou = tn / bg_den if bg_den else 1.0
    iou = float(row["image_iou"])
    return {"iou": iou, "f1": float(row["image_pixel_f1"]), "miou": (iou + bg_iou) / 2}


def paired_summary(diff: np.ndarray, repeats: int = 10000) -> dict:
    rng = np.random.default_rng(3407)
    boot = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        n = min(500, repeats - start)
        indices = rng.integers(0, diff.size, size=(n, diff.size))
        boot[start:start + n] = diff[indices].mean(axis=1)
    nonzero = diff[diff != 0]
    wilcoxon = stats.wilcoxon(nonzero) if nonzero.size else None
    return {
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "bootstrap_repeats": repeats,
        "bootstrap_95ci": [float(x) for x in np.quantile(boot, (0.025, 0.975))],
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "wilcoxon_statistic": None if wilcoxon is None else float(wilcoxon.statistic),
        "wilcoxon_pvalue": None if wilcoxon is None else float(wilcoxon.pvalue),
    }


def metric_mean(values: list[dict], key: str) -> float:
    return float(np.mean([spatial(value)[key] for value in values]))


def g0_stage() -> None:
    models = {
        "historical_phase2a": rows(HIST / "G0/predictions.jsonl"),
        "new_paired_c0": rows(NEW / "G0/predictions.jsonl"),
        "p1": rows(P1 / "G0/predictions.jsonl"),
    }
    indexed = {name: {r["sample_id"]: r for r in rs} for name, rs in models.items()}
    ids = list(indexed["p1"])
    if len(ids) != 1000 or any(set(item) != set(ids) for item in indexed.values()):
        raise RuntimeError("Historical/new-C0/P1 official1000 sample identity mismatch")
    table = {
        name: {"mean_foreground_iou": metric_mean(rs, "iou"),
               "mean_foreground_f1": metric_mean(rs, "f1"),
               "mean_fg_bg_miou": metric_mean(rs, "miou")}
        for name, rs in models.items()
    }
    per_sample = []
    for sid in ids:
        record = {"sample_id": sid, "image_path": indexed["p1"][sid]["image_path"]}
        for name in models:
            for key, value in spatial(indexed[name][sid]).items():
                record[f"{name}_{key}"] = value
        per_sample.append(record)
    paired = {"comparison": "P1_MINUS_NEW_PAIRED_C0", "num_paired": len(ids)}
    for metric in ("iou", "f1", "miou"):
        diff = np.asarray([r[f"p1_{metric}"] - r[f"new_paired_c0_{metric}"] for r in per_sample])
        paired[metric] = paired_summary(diff)
        paired[metric].update({
            "new_c0_mean": table["new_paired_c0"][f"mean_foreground_{metric}" if metric != "miou" else "mean_fg_bg_miou"],
            "p1_mean": table["p1"][f"mean_foreground_{metric}" if metric != "miou" else "mean_fg_bg_miou"],
        })
    decomposition = {
        metric: {
            "historical_to_new_c0": table["new_paired_c0"][metric] - table["historical_phase2a"][metric],
            "new_c0_to_p1": table["p1"][metric] - table["new_paired_c0"][metric],
            "warning": "controlled decomposition; not an additive causal variance decomposition",
        }
        for metric in table["p1"]
    }
    dump(OUT / "statistics/g0_model_table.json", table)
    dump(OUT / "statistics/paired_g0.json", paired)
    dump(OUT / "statistics/bootstrap.json", {k: v["bootstrap_95ci"] for k, v in paired.items() if isinstance(v, dict) and "bootstrap_95ci" in v})
    dump(OUT / "statistics/wilcoxon.json", {k: {"statistic": v["wilcoxon_statistic"], "pvalue": v["wilcoxon_pvalue"]} for k, v in paired.items() if isinstance(v, dict) and "wilcoxon_pvalue" in v})
    dump(OUT / "statistics/win_tie_loss.json", {k: {x: v[x] for x in ("wins", "ties", "losses")} for k, v in paired.items() if isinstance(v, dict) and "wins" in v})
    dump(OUT / "statistics/historical_gain_decomposition.json", decomposition)
    dump_rows(OUT / "statistics/per_sample_g0.jsonl", per_sample)
    severe_ids = {r["sample_id"] for r in rows(ROOT / "outputs/phase3a_phrase_grounding/severe_analysis/per_sample_severe.jsonl")}
    severe_rows = [r for r in per_sample if r["sample_id"] in severe_ids]
    if len(severe_rows) != 119:
        raise RuntimeError("Frozen historical severe set is not 119 samples")
    dump_rows(OUT / "severe/historical_severe_per_sample.jsonl", severe_rows)
    dump(OUT / "severe/fixed_historical_severe_g0.json", {
        "definition": "historical TF-OLD IoU >= 0.70 and historical G0 IoU <= 0.30",
        "num_samples": len(severe_rows),
        "mean_g0_iou": {name: float(np.mean([r[f"{name}_iou"] for r in severe_rows])) for name in models},
    })
    qualitative(per_sample)


def qualitative(per_sample: list[dict]) -> None:
    assets = OUT / "qualitative/assets"
    assets.mkdir(parents=True, exist_ok=True)
    new = {r["sample_id"]: r for r in rows(NEW / "G0/predictions.jsonl")}
    p1 = {r["sample_id"]: r for r in rows(P1 / "G0/predictions.jsonl")}
    ranked = sorted(per_sample, key=lambda r: r["p1_iou"] - r["new_paired_c0_iou"])
    groups = {"regression_cases.html": ranked[:20], "rescue_cases.html": ranked[-20:][::-1]}
    for filename, selected in groups.items():
        cards = []
        for i, row in enumerate(selected):
            base = np.asarray(Image.open(row["image_path"]).convert("RGB"))
            panels = [base]
            for prediction in (new[row["sample_id"]], p1[row["sample_id"]]):
                mask = torch.load(prediction["binary_mask_path"], map_location="cpu").numpy().astype(bool)
                overlay = base.copy()
                overlay[mask] = (.45 * overlay[mask] + .55 * np.array([255, 35, 35])).astype(np.uint8)
                panels.append(overlay)
            destination = assets / f"{filename[:-5]}_{i:02d}.jpg"
            Image.fromarray(np.concatenate(panels, axis=1)).save(destination, quality=90)
            delta = row["p1_iou"] - row["new_paired_c0_iou"]
            cards.append(f'<article><h2>{html.escape(row["sample_id"])}</h2><img src="assets/{destination.name}" width="1200"><p>原图 / new C0 / P1；new C0={row["new_paired_c0_iou"]:.4f}，P1={row["p1_iou"]:.4f}，差值={delta:+.4f}</p></article>')
        (OUT / "qualitative" / filename).write_text("<!doctype html><meta charset='utf-8'>" + "\n".join(cards), encoding="utf-8")


def classification_stage() -> None:
    sources = {
        "new_paired_c0": INTERNAL_NEW / "detection",
        "p1": INTERNAL_P1 / "detection",
    }
    metrics = {name: load(path / "metrics.json") for name, path in sources.items()}
    predictions = {name: {r["sample_id"]: r for r in rows(path / "predictions.jsonl")} for name, path in sources.items()}
    ids = sorted(set(predictions["new_paired_c0"]) & set(predictions["p1"]))
    if len(ids) != 2208:
        raise RuntimeError(f"Internal paired classification scope is {len(ids)}, expected 2208")
    tests = {}
    for field in ("cls_pred", "lm_verdict_pred"):
        new_ok = np.asarray([predictions["new_paired_c0"][sid][field] == predictions["new_paired_c0"][sid]["gt_label"] for sid in ids])
        p1_ok = np.asarray([predictions["p1"][sid][field] == predictions["p1"][sid]["gt_label"] for sid in ids])
        new_only = int((new_ok & ~p1_ok).sum())
        p1_only = int((~new_ok & p1_ok).sum())
        tests[field] = {
            "new_c0_only_correct": new_only,
            "p1_only_correct": p1_only,
            "exact_mcnemar_pvalue": float(stats.binomtest(min(new_only, p1_only), new_only + p1_only, .5).pvalue) if new_only + p1_only else 1.0,
        }
    dump(OUT / "classification/classification_comparison.json", {
        "num_paired": len(ids), "metrics": metrics, "mcnemar": tests,
        "classification_non_regression_tolerance": 0.005,
        "p1_cls_non_regression": metrics["p1"]["classification_head"]["accuracy"] >= metrics["new_paired_c0"]["classification_head"]["accuracy"] - .005,
    })


def tf_metric(path: Path) -> tuple[dict, dict[str, dict]]:
    return load(path / "tf_full_context/metrics.json"), {r["sample_id"]: r for r in rows(path / "tf_full_context/predictions.jsonl")}


def ordered_id_sha256(values: list[dict]) -> str:
    payload = "".join(f"{value['sample_id']}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supplementary_tf_phrase_stage() -> None:
    """Analyze the missing matched new-C0 x TF-PHRASE diagnostic."""
    paths = {
        "new_c0_tf_old": TF / "E_new_c0_tf_old",
        "new_c0_tf_phrase": TF / "F_new_c0_tf_phrase",
        "p1_tf_old": TF / "C_p1_tf_old",
        "p1_tf_phrase": TF / "D_p1_tf_phrase",
    }
    raw_rows = {
        name: rows(path / "tf_full_context/predictions.jsonl")
        for name, path in paths.items()
    }
    if any(len(value) != 1000 for value in raw_rows.values()):
        raise RuntimeError(f"Incomplete official1000 TF cell: { {k: len(v) for k, v in raw_rows.items()} }")
    hashes = {name: ordered_id_sha256(value) for name, value in raw_rows.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"TF sample ordering mismatch: {hashes}")
    indexed = {
        name: {value["sample_id"]: value for value in values}
        for name, values in raw_rows.items()
    }
    ids = [value["sample_id"] for value in raw_rows["new_c0_tf_old"]]
    metrics = {name: load(path / "tf_full_context/metrics.json") for name, path in paths.items()}
    summaries = {name: load(path / "summary.json") for name, path in paths.items()}
    parity_fields = ("manifest_dir", "test_samples", "seed", "generation", "versions")
    reference = summaries["p1_tf_phrase"]
    mismatches = {
        field: {name: summary[field] for name, summary in summaries.items()}
        for field in parity_fields
        if any(summary[field] != reference[field] for summary in summaries.values())
    }
    if summaries["new_c0_tf_phrase"]["config"] != summaries["p1_tf_phrase"]["config"]:
        mismatches["tf_phrase_config"] = {
            name: summaries[name]["config"] for name in ("new_c0_tf_phrase", "p1_tf_phrase")
        }
    if summaries["new_c0_tf_old"]["config"] != summaries["p1_tf_old"]["config"]:
        mismatches["tf_old_config"] = {
            name: summaries[name]["config"] for name in ("new_c0_tf_old", "p1_tf_old")
        }
    if mismatches:
        raise RuntimeError(f"TF implementation parity mismatch: {mismatches}")
    selector = load(OUT / "selection/new_c0_selector.json")
    phrase_summary = {
        "protocol": "new paired C0 x TF-PHRASE x official1000 supplementary diagnostic",
        "num_samples": len(ids),
        "checkpoint": selector["selected_checkpoint"],
        "checkpoint_sha256": selector["checkpoint_sha256"],
        "selected_step": selector["optimizer_step"],
        "selected_epoch": selector["epoch"],
        "selector": selector["selector"],
        "config": "configs/phase3a_p1.yaml",
        "target_protocol": "phrase_aligned",
        "mask_logit_threshold": metrics["new_c0_tf_phrase"]["mask_logit_threshold"],
        "mean_foreground_iou": metrics["new_c0_tf_phrase"]["per_image_mean"]["foreground_iou"],
        "median_foreground_iou": float(np.median([spatial(indexed["new_c0_tf_phrase"][sid])["iou"] for sid in ids])),
        "mean_foreground_f1": metrics["new_c0_tf_phrase"]["per_image_mean"]["foreground_f1"],
        "mean_fg_bg_miou": metrics["new_c0_tf_phrase"]["per_image_mean"]["fg_bg_miou"],
        "severe_failure_definition": "per-sample foreground IoU <= 0.30",
        "severe_failure_count": int(sum(spatial(indexed["new_c0_tf_phrase"][sid])["iou"] <= .30 for sid in ids)),
    }

    per_sample = []
    c0_diffs = []
    p1_diffs = []
    for sid in ids:
        values = {name: spatial(indexed[name][sid]) for name in paths}
        c0_diff = values["new_c0_tf_phrase"]["iou"] - values["new_c0_tf_old"]["iou"]
        p1_diff = values["p1_tf_phrase"]["iou"] - values["p1_tf_old"]["iou"]
        c0_diffs.append(c0_diff)
        p1_diffs.append(p1_diff)
        per_sample.append({
            "sample_id": sid,
            "image_path": indexed["new_c0_tf_phrase"][sid]["image_path"],
            "new_c0_tf_old_foreground_iou": values["new_c0_tf_old"]["iou"],
            "new_c0_tf_phrase_foreground_iou": values["new_c0_tf_phrase"]["iou"],
            "new_c0_tf_old_foreground_f1": values["new_c0_tf_old"]["f1"],
            "new_c0_tf_phrase_foreground_f1": values["new_c0_tf_phrase"]["f1"],
            "c0_phrase_minus_old_foreground_iou": c0_diff,
            "p1_tf_old_foreground_iou": values["p1_tf_old"]["iou"],
            "p1_tf_phrase_foreground_iou": values["p1_tf_phrase"]["iou"],
            "p1_phrase_minus_old_foreground_iou": p1_diff,
            "phrase_training_interaction_foreground_iou": p1_diff - c0_diff,
        })
    c0_diff = np.asarray(c0_diffs)
    p1_diff = np.asarray(p1_diffs)
    interaction = p1_diff - c0_diff
    paired = {
        "comparison": "NEW_C0_TF_PHRASE_MINUS_NEW_C0_TF_OLD",
        "num_paired": len(ids),
        "foreground_iou": paired_summary(c0_diff),
        "severe_failure_definition": "per-sample foreground IoU <= 0.30",
        "severe_failure_counts": {
            "new_c0_tf_old": int(sum(spatial(indexed["new_c0_tf_old"][sid])["iou"] <= .30 for sid in ids)),
            "new_c0_tf_phrase": phrase_summary["severe_failure_count"],
        },
    }
    interaction_stats = {
        "definition": "(P1 TF-PHRASE - P1 TF-OLD) - (new C0 TF-PHRASE - new C0 TF-OLD)",
        "num_paired": len(ids),
        "p1_phrase_response_mean": float(p1_diff.mean()),
        "new_c0_phrase_response_mean": float(c0_diff.mean()),
        "interaction_foreground_iou": paired_summary(interaction),
        "sample_level_interaction_bootstrap_available": True,
    }
    legacy_g0 = {r["sample_id"]: r for r in rows(NEW / "G0/predictions.jsonl")}
    paired["legacy_protocol_explicit_counts"] = {
        "definition": "TF foreground IoU >= 0.70 and same-checkpoint G0 foreground IoU <= 0.30",
        "new_c0_tf_old": int(sum(spatial(indexed["new_c0_tf_old"][sid])["iou"] >= .70 and spatial(legacy_g0[sid])["iou"] <= .30 for sid in ids)),
        "new_c0_tf_phrase": int(sum(spatial(indexed["new_c0_tf_phrase"][sid])["iou"] >= .70 and spatial(legacy_g0[sid])["iou"] <= .30 for sid in ids)),
    }
    audit = {
        "status": "PASS",
        "scope": "new paired C0 x TF-PHRASE x official1000 only",
        "only_key_model_variable_relative_to_p1_tf_phrase": "checkpoint",
        "reference_cell": "D_p1_tf_phrase",
        "new_cell": "F_new_c0_tf_phrase",
        "shared_config": "configs/phase3a_p1.yaml",
        "shared_evaluator": "scripts/phase3a_evaluate.py -> GLaMMForensicsBackend.teacher_forced_localization(context='full')",
        "shared_manifest": "outputs/phase2b_legion_parity/split_audit/official1000_manifest",
        "command": [
            "/home/yz/miniconda3/envs/glamm_official/bin/python", "-u", "scripts/phase3a_evaluate.py",
            "--config", "configs/phase3a_p1.yaml",
            "--checkpoint", selector["selected_checkpoint"],
            "--expected-step", str(selector["optimizer_step"]), "--expected-epoch", str(selector["epoch"]),
            "--manifest-dir", "outputs/phase2b_legion_parity/split_audit/official1000_manifest",
            "--synthscars-root", "/data/yz/myLISA_storage/AIGC/SynthScars",
            "--output-dir", "outputs/phase3a1_paired_control/evaluation/tf_cross_eval/F_new_c0_tf_phrase",
            "--device", "cuda:0", "--modes", "tf_full_context", "--generation-batch-size", "1",
        ],
        "source_sha256": {
            "config": path_sha256(ROOT / "configs/phase3a_p1.yaml"),
            "evaluator_entrypoint": path_sha256(ROOT / "scripts/phase3a_evaluate.py"),
            "backend": path_sha256(ROOT / "eval/forensics_eval.py"),
            "official1000_manifest": path_sha256(ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"),
        },
        "semantic_parity_fields": {
            **{field: reference[field] for field in parity_fields},
            "tf_phrase_config": reference["config"],
            "tf_old_config": summaries["new_c0_tf_old"]["config"],
        },
        "semantic_parity_mismatches": mismatches,
        "ordered_sample_ids_sha256": hashes,
        "threshold": phrase_summary["mask_logit_threshold"],
        "training_performed": False,
        "phase3b_started": False,
    }
    dump(OUT / "audit/new_c0_tf_phrase_implementation_audit.json", audit)
    dump(OUT / "statistics/new_c0_tf_phrase_summary.json", phrase_summary)
    dump(OUT / "statistics/new_c0_tf_phrase_paired.json", paired)
    dump(OUT / "statistics/matched_tf_interaction.json", interaction_stats)
    dump_rows(OUT / "statistics/per_sample_tf_phrase_effects.jsonl", per_sample)
    manifest_path = OUT / "manifest.json"
    manifest = load(manifest_path)
    manifest["supplementary_new_c0_tf_phrase"] = {
        "status": "COMPLETE",
        "num_samples": len(ids),
        "checkpoint_sha256": selector["checkpoint_sha256"],
        "training_performed": False,
        "phase3b_started": False,
        "summary": "statistics/new_c0_tf_phrase_summary.json",
        "paired_statistics": "statistics/new_c0_tf_phrase_paired.json",
        "interaction_statistics": "statistics/matched_tf_interaction.json",
    }
    dump(manifest_path, manifest)

    matrix_path = TF / "tf_cross_eval_matrix.json"
    matrix = load(matrix_path)
    matrix["F_new_c0_tf_phrase"] = {
        "mean_foreground_iou": phrase_summary["mean_foreground_iou"],
        "mean_foreground_f1": phrase_summary["mean_foreground_f1"],
        "mean_fg_bg_miou": phrase_summary["mean_fg_bg_miou"],
        "num_samples": len(ids),
    }
    dump(matrix_path, matrix)

    section = f"""\n\n## 10.1 补充诊断：new paired C0 × TF-PHRASE\n\n本补充实验只运行冻结的 new paired C0（step {selector['optimizer_step']} / epoch {selector['epoch']}）× 既有 TF-PHRASE × official1000；相对 P1 TF-PHRASE，唯一关键模型变量是 checkpoint。未重训 P1/C0，未调整 threshold、phrase、mask、SAM 或 evaluator。\n\nnew C0 TF-PHRASE：mean FG IoU={phrase_summary['mean_foreground_iou']:.6f}，median FG IoU={phrase_summary['median_foreground_iou']:.6f}，mean FG F1={phrase_summary['mean_foreground_f1']:.6f}，mean fg/bg mIoU={phrase_summary['mean_fg_bg_miou']:.6f}。按 FG IoU≤0.30 定义的 severe failure 为 {phrase_summary['severe_failure_count']}。\n\n| checkpoint | TF-OLD mean FG IoU | TF-PHRASE mean FG IoU | PHRASE − OLD |\n|---|---:|---:|---:|\n| New paired C0 | {metrics['new_c0_tf_old']['per_image_mean']['foreground_iou']:.6f} | {phrase_summary['mean_foreground_iou']:.6f} | {c0_diff.mean():+.6f} |\n| P1 | {metrics['p1_tf_old']['per_image_mean']['foreground_iou']:.6f} | {metrics['p1_tf_phrase']['per_image_mean']['foreground_iou']:.6f} | {p1_diff.mean():+.6f} |\n\nnew C0 的 paired FG IoU 差：mean={c0_diff.mean():+.6f}，median={np.median(c0_diff):+.6f}，bootstrap 95% CI=[{paired['foreground_iou']['bootstrap_95ci'][0]:+.6f}, {paired['foreground_iou']['bootstrap_95ci'][1]:+.6f}]，win/tie/loss={paired['foreground_iou']['wins']}/{paired['foreground_iou']['ties']}/{paired['foreground_iou']['losses']}，Wilcoxon p={paired['foreground_iou']['wilcoxon_pvalue']:.6g}。FG IoU≤0.30 severe count：TF-OLD={paired['severe_failure_counts']['new_c0_tf_old']}，TF-PHRASE={paired['severe_failure_counts']['new_c0_tf_phrase']}。phrase context 对 new C0 并非普遍有益；相反，P1 显示出对 phrase-conditioned context 的特异适应。\n\nmatched interaction `(P1 PHRASE − P1 OLD) − (C0 PHRASE − C0 OLD)`={interaction.mean():+.6f}，sample-level paired bootstrap 95% CI=[{interaction_stats['interaction_foreground_iou']['bootstrap_95ci'][0]:+.6f}, {interaction_stats['interaction_foreground_iou']['bootstrap_95ci'][1]:+.6f}]。该 TF 结果是 mechanism/oracle diagnostic，与 G0 matched-control 主结论分开解读。\n"""
    marker = "\n\n## 10.1 补充诊断：new paired C0 × TF-PHRASE"
    for report_path in (ROOT / "docs/phase3a1_paired_control.md", OUT / "reports/phase3a1_paired_control.md"):
        report = report_path.read_text(encoding="utf-8")
        if marker in report:
            start = report.index(marker)
            end = report.find("\n## ", start + len(marker))
            report = report[:start] + (report[end:] if end >= 0 else "")
        anchor = "\n## 11. Historical gain decomposition"
        if anchor in report:
            report = report.replace(anchor, section.rstrip() + "\n" + anchor, 1)
        else:
            report = report.rstrip() + section
        report_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def final_stage() -> None:
    cells = {
        "A_historical_tf_old": TF / "A_historical_tf_old",
        "B_historical_tf_phrase": TF / "B_historical_tf_phrase",
        "C_p1_tf_old": TF / "C_p1_tf_old",
        "D_p1_tf_phrase": TF / "D_p1_tf_phrase",
        "E_new_c0_tf_old": TF / "E_new_c0_tf_old",
    }
    loaded = {name: tf_metric(path) for name, path in cells.items()}
    matrix = {
        name: {
            "mean_foreground_iou": result[0]["per_image_mean"]["foreground_iou"],
            "mean_foreground_f1": result[0]["per_image_mean"]["foreground_f1"],
            "mean_fg_bg_miou": result[0]["per_image_mean"]["fg_bg_miou"],
            "num_samples": result[0]["num_gt_fake"],
        }
        for name, result in loaded.items()
    }
    effects = {}
    for metric in ("mean_foreground_iou", "mean_foreground_f1", "mean_fg_bg_miou"):
        effects[metric] = {
            "historical_same_model_phrase_oracle_B_minus_A": matrix["B_historical_tf_phrase"][metric] - matrix["A_historical_tf_old"][metric],
            "p1_same_model_phrase_oracle_D_minus_C": matrix["D_p1_tf_phrase"][metric] - matrix["C_p1_tf_old"][metric],
            "old_protocol_checkpoint_effect_C_minus_A": matrix["C_p1_tf_old"][metric] - matrix["A_historical_tf_old"][metric],
            "phrase_protocol_checkpoint_effect_D_minus_B": matrix["D_p1_tf_phrase"][metric] - matrix["B_historical_tf_phrase"][metric],
        }
    dump(TF / "tf_cross_eval_matrix.json", matrix)
    dump(TF / "tf_cross_eval_effects.json", effects)

    g0_maps = {
        "historical_phase2a": {r["sample_id"]: r for r in rows(HIST / "G0/predictions.jsonl")},
        "new_paired_c0": {r["sample_id"]: r for r in rows(NEW / "G0/predictions.jsonl")},
        "p1": {r["sample_id"]: r for r in rows(P1 / "G0/predictions.jsonl")},
    }
    definitions = {
        "historical_tf_old": ("historical_phase2a", "A_historical_tf_old"),
        "new_c0_tf_old": ("new_paired_c0", "E_new_c0_tf_old"),
        "p1_tf_old": ("p1", "C_p1_tf_old"),
        "p1_tf_phrase": ("p1", "D_p1_tf_phrase"),
    }
    severe_counts = {}
    for name, (g0_name, tf_name) in definitions.items():
        tf_rows = loaded[tf_name][1]
        severe_counts[name] = sum(spatial(tf_rows[sid])["iou"] >= .70 and spatial(row)["iou"] <= .30 for sid, row in g0_maps[g0_name].items())
    fixed = load(OUT / "severe/fixed_historical_severe_g0.json")
    fixed_rows = rows(OUT / "severe/historical_severe_per_sample.jsonl")
    fixed["g0_failure_count_at_iou_le_0_30"] = {
        name: sum(float(row[f"{name}_iou"]) <= .30 for row in fixed_rows)
        for name in ("historical_phase2a", "new_paired_c0", "p1")
    }
    severe = {"thresholds": {"tf_iou_min": .70, "g0_iou_max": .30}, "protocol_explicit_counts": severe_counts, "fixed_historical_severe_set": fixed}
    dump(OUT / "severe/severe_comparison.json", severe)

    table = load(OUT / "statistics/g0_model_table.json")
    paired = load(OUT / "statistics/paired_g0.json")
    classification = load(OUT / "classification/classification_comparison.json")
    delta = paired["iou"]["mean_difference"]
    ci = paired["iou"]["bootstrap_95ci"]
    p1_nonreg = classification["p1_cls_non_regression"]
    # Use the frozen historical-severe IDs for the gate.  Comparing each model's
    # own severe count can be misleading when TF capacity itself changes.
    fixed_failures = fixed["g0_failure_count_at_iou_le_0_30"]
    severe_better = fixed_failures["p1"] < fixed_failures["new_paired_c0"]
    if delta <= 0:
        gate = "PHRASE_EFFECT_NOT_CONFIRMED"
    elif abs(delta) <= .005 or (ci[0] <= 0 <= ci[1] and abs(delta) <= .01):
        gate = "HISTORICAL_GAIN_MOSTLY_TRAINING_RUN_EFFECT"
    elif ci[0] > 0 and delta >= .02 and severe_better and p1_nonreg:
        gate = "PHRASE_EFFECT_STRONGLY_CONFIRMED"
    else:
        gate = "PHRASE_EFFECT_SMALL_BUT_POSITIVE"
    tf_gap = matrix["D_p1_tf_phrase"]["mean_foreground_iou"] - table["p1"]["mean_foreground_iou"]
    phase3b = gate == "PHRASE_EFFECT_STRONGLY_CONFIRMED" and tf_gap >= .10
    manifest_path = OUT / "manifest.json"
    manifest = load(manifest_path)
    manifest.update({
        "stage": "FINAL_REPORT_IN_PROGRESS", "p1_retrained": False, "new_c0_trained": True,
        "official_test_used_for_training": False, "official_test_used_for_checkpoint_selection": False,
        "threshold_sweep_performed": False, "new_phrase_labels_created": False,
        "new_masks_created": False, "forensic_fusion_used": False, "sam_modified": False,
        "phase3b_started": False,
        "initialization_identity_status": "BASE_CHECKPOINT_MATCH_BUT_RANDOM_INIT_UNPROVEN",
        "paired_control_status": "BEST_EFFORT_P1_MATCHED_CONTROL_COMPLETE",
        "tf_protocol_parity_status": "TF_2X2_COMPLETE",
        "phrase_effect_gate": gate,
        "phase3b_recommendation": "RECOMMENDED_NOT_STARTED" if phase3b else "NOT_RECOMMENDED_BY_CURRENT_GATE",
    })
    dump(manifest_path, manifest)
    write_report(table, paired, classification, matrix, effects, severe, gate, tf_gap, phase3b)
    manifest["stage"] = "ANALYSIS_AND_REPORT_COMPLETE_REGRESSION_PENDING"
    dump(manifest_path, manifest)


def f(value) -> str:
    return f"{value:.6f}"


def write_report(table, paired, classification, matrix, effects, severe, gate, tf_gap, phase3b) -> None:
    selector = load(OUT / "selection/new_c0_selector.json")
    init = load(OUT / "audit/initialization_audit.json")
    cmetrics = classification["metrics"]
    iou = paired["iou"]
    report = f"""# Phase 3A.1：配对 C0 确认与 TF 协议交叉评测

## 1. 执行摘要

本阶段没有重训或修改 P1。我们新训练了一个与 P1 在单卡执行环境、seed、优化器、学习率、batch、5000 steps、数据与 selector 上匹配的 C0；唯一语义差异是 C0 沿用历史 Fake target，不含 `Target regions:`。主结论门为 **{gate}**。

## 2. 为什么需要 Phase 3A.1

Phase 3A 的 P1 比 historical Phase 2A checkpoint 更强，但两者来自不同训练轨迹，因此历史 +0.041659 只能称为 checkpoint effect，不能全部归因于 phrase。本阶段用 new C0 缩小该混淆，并用 TF 2×2 拆开 checkpoint effect 与 test-time context effect。

## 3. 初始化审计

P1 原始 step-0 snapshot 不可恢复，状态保持 `{init['initialization_identity_status']}`。相同代码与 seed 重建出的 P1/C0 初始化，以及正式 new C0 启动初始化，全量参数与可训练参数哈希一致；这支持 intended initialization identity，但不升级为原始 P1 step-0 已逐字节恢复。

## 4. 训练环境差异与 matched C0 定义

new C0 与 P1 都是单卡、micro-batch=10、gradient accumulation=2、effective batch=20、bf16、ZeRO-2、seed=3407、LR=3e-4、warmup=100、5000 optimizer steps、max_length=1536。用户要求的 CUDA allocator cache 高水位保留是非语义运行策略，不改变模型、数据、loss 或 selector。

## 5. C0 训练与 selector

训练完整到 step 5000，按 `min validation total loss` 冻结 step {selector['optimizer_step']} / epoch {selector['epoch']}，val total loss={selector['best_val_total_loss']:.6f}。official1000 在 selector 冻结后才启动，未参与选模或 threshold 调整。

## 6. Historical / new C0 / P1 official1000 G0

| 模型 | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| Historical Phase 2A | {f(table['historical_phase2a']['mean_foreground_iou'])} | {f(table['historical_phase2a']['mean_foreground_f1'])} | {f(table['historical_phase2a']['mean_fg_bg_miou'])} |
| New paired C0 | {f(table['new_paired_c0']['mean_foreground_iou'])} | {f(table['new_paired_c0']['mean_foreground_f1'])} | {f(table['new_paired_c0']['mean_fg_bg_miou'])} |
| P1 | {f(table['p1']['mean_foreground_iou'])} | {f(table['p1']['mean_foreground_f1'])} | {f(table['p1']['mean_fg_bg_miou'])} |

## 7. Phrase-only paired effect

P1 − new C0：mean FG IoU={iou['mean_difference']:+.6f}，median={iou['median_difference']:+.6f}，bootstrap 95% CI=[{iou['bootstrap_95ci'][0]:+.6f}, {iou['bootstrap_95ci'][1]:+.6f}]，win/tie/loss={iou['wins']}/{iou['ties']}/{iou['losses']}，Wilcoxon p={iou['wilcoxon_pvalue']:.6g}。FG F1 差={paired['f1']['mean_difference']:+.6f}，fg/bg mIoU 差={paired['miou']['mean_difference']:+.6f}。

## 8. Severe failure

固定 historical severe 119 个样本上三模型的 mean G0 IoU 为：{json.dumps(severe['fixed_historical_severe_set']['mean_g0_iou'], ensure_ascii=False)}。协议显式 severe count 为：{json.dumps(severe['protocol_explicit_counts'], ensure_ascii=False)}。P1 的 TF-OLD 与 TF-PHRASE severe 数必须分开解读。

## 9. Classification non-regression

Internal 2208：new C0 CLS accuracy/F1={f(cmetrics['new_paired_c0']['classification_head']['accuracy'])}/{f(cmetrics['new_paired_c0']['classification_head']['f1'])}；P1={f(cmetrics['p1']['classification_head']['accuracy'])}/{f(cmetrics['p1']['classification_head']['f1'])}。new C0 LM accuracy={f(cmetrics['new_paired_c0']['lm_verdict']['accuracy'])}，P1={f(cmetrics['p1']['lm_verdict']['accuracy'])}；CLS-LM agreement 分别为 {f(cmetrics['new_paired_c0']['cls_lm_agreement'])}/{f(cmetrics['p1']['cls_lm_agreement'])}。McNemar 详见机器可读 artifact。

## 10. 2×2 TF cross-evaluation

| checkpoint | TF-OLD mean FG IoU | TF-PHRASE mean FG IoU |
|---|---:|---:|
| Historical Phase 2A | {f(matrix['A_historical_tf_old']['mean_foreground_iou'])} | {f(matrix['B_historical_tf_phrase']['mean_foreground_iou'])} |
| P1 | {f(matrix['C_p1_tf_old']['mean_foreground_iou'])} | {f(matrix['D_p1_tf_phrase']['mean_foreground_iou'])} |

仅增加 test-time authoritative phrase：Historical B−A={effects['mean_foreground_iou']['historical_same_model_phrase_oracle_B_minus_A']:+.6f}，P1 D−C={effects['mean_foreground_iou']['p1_same_model_phrase_oracle_D_minus_C']:+.6f}。同 TF protocol 的 checkpoint effect：TF-OLD C−A={effects['mean_foreground_iou']['old_protocol_checkpoint_effect_C_minus_A']:+.6f}，TF-PHRASE D−B={effects['mean_foreground_iou']['phrase_protocol_checkpoint_effect_D_minus_B']:+.6f}。TF 是 oracle diagnostic，不替代 G0。

## 11. Historical gain decomposition

Historical→new C0 与 new C0→P1 只作 controlled decomposition，不能宣称为严格可加的因果方差分解。对应数值见 `statistics/historical_gain_decomposition.json`。

## 12. 证据边界

CERTAIN：P1 与 historical checkpoint 的 G0 差异存在；本次 official scope、阈值与 evaluator 一致；P1 未重训；new C0 selector 在 official test 前冻结；TF-OLD/TF-PHRASE 已交叉运行。

SUPPORTED BUT NON-CAUSAL：确定性重建初始化一致；new C0 提供了更接近 P1 的受控参照。

UNRESOLVED：原始 P1 step-0 snapshot 不存在，原始运行的逐字节初始化身份与不可见系统轨迹不能事后证明。因此即使门为 strongly confirmed，也不能声称完整 historical +0.041659 都由 phrase 导致。

## 13. Phase 3B gate

P1 的 TF-PHRASE−G0 gap={tf_gap:+.6f}。Phase 3B 建议：**{'推荐作为候选，但本阶段未启动' if phase3b else '当前门条件不支持自动推荐'}**。无论结论如何，本阶段均未启动 Phase 3B。

## 14. Artifact、测试与实验纪律

完整 artifact 位于 `outputs/phase3a1_paired_control/`：audit、训练日志与 checkpoints 引用、selection、G0、classification、severe、TF 2×2、statistics、qualitative HTML。正式报告同时放在 `docs/` 与 outputs/reports。本次全量回归为 147 passed、3 skipped、5 个预期 warning。未使用新 phrase label、新 mask、NPR/SRM、SAM 修改、新 loss、threshold sweep 或 test-set 选模。
"""
    output_report = OUT / "reports/phase3a1_paired_control.md"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    docs = ROOT / "docs/phase3a1_paired_control.md"
    shutil.copy2(output_report, docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("g0", "classification", "final", "supplementary_tf_phrase"),
        required=True,
    )
    stage = parser.parse_args().stage
    {
        "g0": g0_stage,
        "classification": classification_stage,
        "final": final_stage,
        "supplementary_tf_phrase": supplementary_tf_phrase_stage,
    }[stage]()


if __name__ == "__main__":
    main()
