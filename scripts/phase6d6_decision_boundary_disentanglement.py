#!/usr/bin/env python3
"""Phase 6D.6: offline boundary-calibration versus frozen RINE fusion audit."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P65 = ROOT / "outputs/phase6d5_decision_integration"
FULL = ROOT / "outputs/phase6d5_full_classification_ood"
OUT = ROOT / "outputs/phase6d6_decision_boundary_disentanglement"
JSON_OUT = OUT / "phase6d6_decision_boundary_disentanglement.json"
DOC_OUT = ROOT / "docs/phase6d6_decision_boundary_disentanglement.md"
VAL = P65 / "raw_scores/internal_validation.jsonl"
PARAMS = P65 / "fusion_parameters.json"
FULL_NAMES = ("aigi_holmes", "genimage", "loki", "raise998")
MIXED_NAMES = ("aigi_holmes", "genimage", "loki")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 3616


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def hard_metrics(y, score, threshold):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = score > threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()), "balanced_accuracy": 0.5 * (recall + tnr),
        "precision": precision, "fake_recall": recall, "tnr": tnr, "fpr": 1 - tnr,
        "f1": 2 * precision * recall / max(1e-30, precision + recall),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": float(threshold),
        "roc_auc": None if len(np.unique(y)) < 2 else float(roc_auc_score(y, score)),
    }


def select_threshold(y, score):
    """Exact threshold sweep for prediction score > tau."""
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    unique = np.unique(score)
    thresholds = np.concatenate((
        [np.nextafter(unique[0], -np.inf)],
        (unique[:-1] + unique[1:]) / 2,
        [unique[-1]],
    ))
    best = None
    tolerance = 1e-15
    for threshold in thresholds:
        metric = hard_metrics(y, score, float(threshold))
        key = (metric["balanced_accuracy"], metric["f1"], -metric["fpr"])
        if best is None or any(
            key[i] > best["selector_key"][i] + tolerance and
            all(abs(key[j] - best["selector_key"][j]) <= tolerance for j in range(i))
            for i in range(3)
        ):
            best = {"threshold": float(threshold), "metrics": metric, "selector_key": key}
    best["selector_key"] = list(best["selector_key"])
    best["candidate_count"] = int(len(thresholds))
    return best


def prediction_metrics(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=bool)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, int((y == 1).sum()))
    return np.asarray([
        float((pred == y).mean()), recall,
        2 * precision * recall / max(1e-30, precision + recall),
    ])


def paired_bootstrap(y, pred_first, pred_second):
    """Stratified paired bootstrap; deltas are second minus first."""
    y = np.asarray(y, dtype=np.int64)
    pred_first = np.asarray(pred_first, dtype=bool)
    pred_second = np.asarray(pred_second, dtype=bool)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
    # Within each label stratum the relevant paired outcome is one of the four
    # (first prediction, second prediction) cells. Multinomial sampling of these
    # cell counts is exactly the categorical, stratified sample-level bootstrap
    # and avoids materializing 100k indices for every replicate.
    cells = []
    for value in (0, 1):
        mask = y == value
        code = pred_first[mask].astype(np.int64) * 2 + pred_second[mask].astype(np.int64)
        count = np.bincount(code, minlength=4)
        cells.append((int(mask.sum()), count / count.sum()))
    for index in range(BOOTSTRAP_REPLICATES):
        real = rng.multinomial(cells[0][0], cells[0][1])
        fake = rng.multinomial(cells[1][0], cells[1][1])
        # Codes: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
        n_real, n_fake = real.sum(), fake.sum()
        first_tn, second_tn = real[0] + real[1], real[0] + real[2]
        first_fp, second_fp = real[2] + real[3], real[1] + real[3]
        first_tp, second_tp = fake[2] + fake[3], fake[1] + fake[3]
        first_fn, second_fn = fake[0] + fake[1], fake[0] + fake[2]
        def from_counts(tp, tn, fp, fn):
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            return np.asarray([
                (tp + tn) / (n_real + n_fake), recall,
                2 * precision * recall / max(1e-30, precision + recall),
            ])
        deltas[index] = from_counts(second_tp, second_tn, second_fp, second_fn) - from_counts(first_tp, first_tn, first_fp, first_fn)
    observed = prediction_metrics(y, pred_second) - prediction_metrics(y, pred_first)
    low, high = np.percentile(deltas, [2.5, 97.5], axis=0)
    names = ("accuracy", "fake_recall", "f1")
    return {
        name: {"delta_second_minus_first": float(observed[i]),
               "ci95": [float(low[i]), float(high[i])]}
        for i, name in enumerate(names)
    }


def main():
    parameters = json.loads(PARAMS.read_text())
    normalization = parameters["normalization"]
    alpha = float(parameters["selected_alpha"])
    if alpha != 0.3:
        raise RuntimeError(f"frozen alpha drift: {alpha}")

    validation = read_jsonl(VAL)
    vy = np.asarray([row["label"] for row in validation], dtype=np.int64)
    vc_raw = np.asarray([row["c1_margin"] for row in validation], dtype=np.float64)
    vr_raw = np.asarray([row["rine_margin"] for row in validation], dtype=np.float64)
    vz_c = (vc_raw - normalization["c1"]["mean"]) / normalization["c1"]["std"]
    vz_r = (vr_raw - normalization["rine"]["mean"]) / normalization["rine"]["std"]
    vfixed = 0.5 * vz_r + 0.5 * vz_c
    valpha = alpha * vz_r + (1 - alpha) * vz_c
    selectors = {
        "C1-valtau": select_threshold(vy, vc_raw),
        "D1-fixed-valtau": select_threshold(vy, vfixed),
        "D1-alpha-valtau": select_threshold(vy, valpha),
    }

    datasets = {}
    paired = {}
    prediction_provenance = {}
    for name in FULL_NAMES:
        raw_path = FULL / "raw" / f"{name}.jsonl"
        records = read_jsonl(raw_path)
        y = np.asarray([row["label"] for row in records], dtype=np.int64)
        c_raw = np.asarray([row["c1_margin"] for row in records], dtype=np.float64)
        r_raw = np.asarray([row["rine_margin"] for row in records], dtype=np.float64)
        z_c = (c_raw - normalization["c1"]["mean"]) / normalization["c1"]["std"]
        z_r = (r_raw - normalization["rine"]["mean"]) / normalization["rine"]["std"]
        fixed = 0.5 * z_r + 0.5 * z_c
        fused = alpha * z_r + (1 - alpha) * z_c
        arms = {
            "C1-raw": (c_raw, 0.0),
            "C1-center": (c_raw, normalization["c1"]["mean"]),
            "C1-valtau": (c_raw, selectors["C1-valtau"]["threshold"]),
            "D1-fixed": (fixed, 0.0),
            "D1-fixed-valtau": (fixed, selectors["D1-fixed-valtau"]["threshold"]),
            "D1-alpha": (fused, 0.0),
            "D1-alpha-valtau": (fused, selectors["D1-alpha-valtau"]["threshold"]),
        }
        datasets[name] = {arm: hard_metrics(y, score, threshold)
                          for arm, (score, threshold) in arms.items()}
        if name == "genimage":
            generators = sorted({str(row.get("generator")) for row in records})
            datasets[name]["per_generator"] = {}
            for generator in generators:
                mask = np.asarray([str(row.get("generator")) == generator for row in records])
                datasets[name]["per_generator"][generator] = {
                    arm: hard_metrics(y[mask], score[mask], threshold)
                    for arm, (score, threshold) in arms.items()
                }
        if name in MIXED_NAMES:
            paired[name] = {
                "C1-center_vs_D1-alpha": paired_bootstrap(
                    y, c_raw > normalization["c1"]["mean"], fused > 0.0),
                "C1-valtau_vs_D1-alpha-valtau": paired_bootstrap(
                    y, c_raw > selectors["C1-valtau"]["threshold"],
                    fused > selectors["D1-alpha-valtau"]["threshold"]),
            }
        prediction_provenance[name] = {
            "path": str(raw_path.resolve()), "sha256": sha256(raw_path),
            "count": len(records), "ordered_sample_id_sha256": hashlib.sha256(
                "\n".join(row["sample_id"] for row in records).encode()).hexdigest(),
        }

    # Predeclared interpretation: boundary correction explains the hard-gain mechanism
    # when centering is no worse than alpha fusion on all three binary OOD datasets for
    # accuracy and F1. Ranking is added when fusion AUC improves on all three. Residual
    # calibrated hard gain requires positive 95% CI for accuracy or F1 on >=2 datasets
    # without a negative CI on either metric elsewhere.
    boundary = all(
        datasets[name]["C1-center"][key] >= datasets[name]["D1-alpha"][key]
        for name in MIXED_NAMES for key in ("accuracy", "f1")
    )
    ranking = all(
        datasets[name]["D1-alpha"]["roc_auc"] > datasets[name]["C1-raw"]["roc_auc"]
        for name in MIXED_NAMES
    )
    comparison = "C1-valtau_vs_D1-alpha-valtau"
    positive = sum(
        any(paired[name][comparison][key]["ci95"][0] > 0 for key in ("accuracy", "f1"))
        for name in MIXED_NAMES
    )
    negative_any = any(
        paired[name][comparison][key]["ci95"][1] < 0
        for name in MIXED_NAMES for key in ("accuracy", "f1")
    )
    hard_after = positive >= 2 and not negative_any

    decisions = {
        "BOUNDARY_CORRECTION_EXPLAINS_C1_GAIN": "YES" if boundary else "NO",
        "RINE_FUSION_ADDS_RANKING": "YES" if ranking else "NO",
        "RINE_FUSION_ADDS_HARD_GAIN_AFTER_CALIBRATION": "YES" if hard_after else "NO",
    }
    output = {
        "schema": "phase6d6_decision_boundary_disentanglement_v1",
        "status": "COMPLETE", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "inference_or_training": False, "formal_frozen_alpha": alpha,
            "normalization": normalization,
            "threshold_selector": ["maximize internal-val balanced accuracy", "tie higher F1", "tie lower FPR"],
            "ood_used_for_selection": False,
            "score_definitions": {
                "C1-raw": "raw C1 margin; threshold 0",
                "C1-center": "raw C1 margin; threshold internal-TRAIN C1 mean",
                "C1-valtau": "raw C1 margin; validation-selected tau_C",
                "D1-fixed": "0.5*zR+0.5*zC; threshold 0",
                "D1-alpha": "0.3*zR+0.7*zC; threshold 0",
                "valtau_suffix": "same continuous score; validation-selected threshold",
            },
            "decision_rules": {
                "boundary": "C1-center accuracy and F1 >= D1-alpha on all 3 mixed OOD datasets",
                "ranking": "D1-alpha AUC > C1 AUC on all 3 mixed OOD datasets",
                "hard_after_calibration": "positive paired CI for accuracy or F1 on >=2 datasets and no negative CI for either elsewhere",
            },
        },
        "provenance": {
            "fusion_parameters": {"path": str(PARAMS.resolve()), "sha256": sha256(PARAMS)},
            "validation_scores": {"path": str(VAL.resolve()), "sha256": sha256(VAL), "count": len(validation)},
            "full_ood_scores": prediction_provenance,
        },
        "selectors": selectors, "datasets": datasets, "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "method": "stratified paired sample-id bootstrap; second minus first",
            "datasets": paired,
        }, "decisions": decisions,
    }
    dump(JSON_OUT, output)

    lines = [
        "# Phase 6D.6 — Decision Boundary Disentanglement", "",
        "Status: **COMPLETE**. No training or inference was performed. All thresholds used internal validation only; full OOD was evaluation-only. The formal frozen fusion remains `alpha=0.3`.", "",
        "## Frozen selectors", "",
        "| Arm | Continuous score | Validation-selected threshold | Val balanced accuracy | Val F1 | Val FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selector_score = {"C1-valtau": "C1 raw margin", "D1-fixed-valtau": "0.5*zR+0.5*zC", "D1-alpha-valtau": "0.3*zR+0.7*zC"}
    for arm, selected_value in selectors.items():
        m = selected_value["metrics"]
        lines.append(f"| {arm} | {selector_score[arm]} | {selected_value['threshold']:.9f} | {m['balanced_accuracy']:.6f} | {m['f1']:.6f} | {m['fpr']:.6f} |")
    for name in FULL_NAMES:
        lines += ["", f"## {name}", "", "| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, m in datasets[name].items():
            if arm == "per_generator":
                continue
            auc = "N/A" if m["roc_auc"] is None else f"{m['roc_auc']:.6f}"
            lines.append(f"| {arm} | {m['accuracy']:.6f} | {auc} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
    lines += ["", "## Paired bootstrap", "", "Deltas are second minus first; 10,000 stratified paired sample-ID bootstrap replicates.", "",
              "| Dataset | Comparison | Metric | Delta | 95% CI |", "|---|---|---|---:|---:|"]
    for name in MIXED_NAMES:
        for comparison_name, values in paired[name].items():
            for metric_name, value in values.items():
                lines.append(f"| {name} | {comparison_name} | {metric_name} | {value['delta_second_minus_first']:.6f} | [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}] |")
    lines += ["", "## GenImage per-generator", ""]
    for generator, values in datasets["genimage"]["per_generator"].items():
        lines += [f"### {generator}", "", "| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, m in values.items():
            lines.append(f"| {arm} | {m['accuracy']:.6f} | {m['roc_auc']:.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
        lines.append("")
    lines += ["## Final decisions", ""] + [f"- `{key} = {value}`" for key, value in decisions.items()]
    lines += ["", "AIGI-Holmes retains the prior leakage disclosure: 779 exact overlaps with internal TRAIN under the active user-authorized override.", ""]
    DOC_OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
