#!/usr/bin/env python3
"""Offline alpha sensitivity sweep over frozen Phase 6D.5 full-OOD margins."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase6d5_full_classification_ood"
PARAMETERS = ROOT / "outputs/phase6d5_decision_integration/fusion_parameters.json"
RESULT = SOURCE / "alpha_sensitivity.json"
REPORT = ROOT / "docs/phase6d5_full_ood_alpha_sensitivity.md"
ALPHAS = [round(i / 10, 1) for i in range(11)]


def rows(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = score > 0
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    result = {
        "n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()), "precision": precision,
        "fake_recall": recall, "tnr": tnr, "fpr": 1 - tnr,
        "f1": 2 * precision * recall / max(1e-30, precision + recall),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.0,
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, score))
        result["auprc"] = float(average_precision_score(y, score))
        fpr, tpr, _ = roc_curve(y, score)
        for q in (0.01, 0.05, 0.10):
            result[f"recall_at_fpr_{int(q * 100)}pct"] = float(tpr[fpr <= q].max())
    else:
        result.update(roc_auc=None, auprc=None)
    return result


def main():
    parameters = json.loads(PARAMETERS.read_text())
    normalization = parameters["normalization"]
    selected = float(parameters["selected_alpha"])
    datasets = {}
    normalized = {}
    for name in ("aigi_holmes", "genimage", "loki", "raise998"):
        path = SOURCE / "raw" / f"{name}.jsonl"
        records = rows(path)
        y = np.asarray([r["label"] for r in records], dtype=np.int64)
        c = np.asarray([r["c1_margin"] for r in records], dtype=np.float64)
        r = np.asarray([r["rine_margin"] for r in records], dtype=np.float64)
        sc = (c - normalization["c1"]["mean"]) / normalization["c1"]["std"]
        sr = (r - normalization["rine"]["mean"]) / normalization["rine"]["std"]
        normalized[name] = (records, y, sc, sr)
        datasets[name] = {
            str(alpha): metrics(y, alpha * sr + (1 - alpha) * sc) for alpha in ALPHAS
        }

    records, y, sc, sr = normalized["genimage"]
    per_generator = {}
    for generator in sorted({str(row["generator"]) for row in records}):
        mask = np.asarray([str(row["generator"]) == generator for row in records])
        per_generator[generator] = {
            str(alpha): metrics(y[mask], alpha * sr[mask] + (1 - alpha) * sc[mask])
            for alpha in ALPHAS
        }

    macro_datasets = ("aigi_holmes", "genimage", "loki")
    macro = {}
    for alpha in ALPHAS:
        entries = [datasets[name][str(alpha)] for name in macro_datasets]
        macro[str(alpha)] = {
            key: float(np.mean([entry[key] for entry in entries]))
            for key in ("accuracy", "roc_auc", "fake_recall", "tnr", "fpr", "f1")
        }

    output = {
        "schema": "phase6d5_full_ood_alpha_sensitivity_v1",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "post-hoc exploratory sensitivity; no OOD selection or tuning",
        "formal_validation_selected_alpha": selected,
        "formula": "alpha*sR + (1-alpha)*sC; score>0 => Fake",
        "normalization": normalization,
        "alpha_grid": ALPHAS,
        "source_files": {
            name: {"path": str((SOURCE / "raw" / f"{name}.jsonl").resolve()),
                   "sha256": sha256(SOURCE / "raw" / f"{name}.jsonl")}
            for name in datasets
        },
        "datasets": datasets,
        "mixed_ood_macro": macro,
        "genimage_per_generator": per_generator,
    }
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Phase 6D.5 — Full-OOD Alpha Sensitivity",
        "",
        "Status: **COMPLETE**. This is a post-hoc exploratory sensitivity analysis over frozen per-sample C1/RINE margins. No model inference, fitting, calibration, threshold tuning, or OOD-based selection was performed.",
        "",
        f"The formal internal-validation-selected value remains **alpha={selected:.1f}**. Formula: `alpha*sR + (1-alpha)*sC`, with frozen internal-TRAIN normalization and `score>0 => Fake`.",
    ]
    for name in ("mixed_ood_macro", "aigi_holmes", "genimage", "loki", "raise998"):
        title = "Mixed OOD macro (AIGI-Holmes, GenImage, LOKI)" if name == "mixed_ood_macro" else name
        table = macro if name == "mixed_ood_macro" else datasets[name]
        lines += ["", f"## {title}", "", "| alpha | status | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for alpha in ALPHAS:
            m = table[str(alpha)]
            status = "validation-selected" if alpha == selected else "exploratory"
            auc = "N/A" if m.get("roc_auc") is None else f"{m['roc_auc']:.6f}"
            lines.append(f"| {alpha:.1f} | {status} | {m['accuracy']:.6f} | {auc} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")

    lines += ["", "## GenImage per-generator", ""]
    for generator, table in per_generator.items():
        lines += [f"### {generator}", "", "| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for alpha in ALPHAS:
            m = table[str(alpha)]
            lines.append(f"| {alpha:.1f} | {m['accuracy']:.6f} | {m['roc_auc']:.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
        lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
