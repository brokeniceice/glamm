#!/usr/bin/env python3
"""Validate and aggregate the frozen final evaluation without running models."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/final_evaluation"
CLASS = OUT / "classification"
LOC = OUT / "localization"
TABLES = OUT / "tables"
FINAL_JSON = OUT / "final_results.json"
FINAL_REPORT = OUT / "final_report.md"
DOC_REPORT = ROOT / "docs/final_evaluation_report.md"
STATUS = OUT / "finalizer_status.json"

LEAKAGE_SUMMARY = ROOT / "outputs/final_eval_datasets/leakage_audit/summary.json"
LEAKAGE_OVERRIDE = ROOT / "outputs/final_eval_datasets/leakage_override.json"
LEAKAGE_EXACT = ROOT / "outputs/final_eval_datasets/leakage_audit/external_vs_internal_train_exact.jsonl"
LEAKAGE_PHASH = ROOT / "outputs/final_eval_datasets/leakage_audit/external_vs_internal_train_phash.jsonl"
AIGI_GEN_EXACT = ROOT / "outputs/final_eval_datasets/leakage_audit/aigi_holmes_vs_genimage_exact.jsonl"
AIGI_GEN_PHASH = ROOT / "outputs/final_eval_datasets/leakage_audit/aigi_holmes_vs_genimage_phash.jsonl"

CLASS_MANIFESTS = {
    "internal": ROOT / "datasets/Internal2208/manifests/eval_manifest.jsonl",
    "aigi_holmes": ROOT / "datasets/AIGI-Holmes/manifests/eval_manifest.jsonl",
    "genimage": ROOT / "datasets/GenImage/manifests/eval_manifest.jsonl",
    "loki": ROOT / "datasets/LOKI/manifests/classification_eval_manifest.jsonl",
    "raise998": ROOT / "datasets/RAISE/manifests/eval_manifest.jsonl",
}
LOC_MANIFESTS = {
    "synthscars": ROOT / "datasets/SynthScars/manifests/eval_manifest.jsonl",
    "loki": ROOT / "datasets/LOKI/manifests/localization_eval_manifest.jsonl",
    "xaigd": ROOT / "datasets/X-AIGD/manifests/eval_manifest.jsonl",
    "pal4vst": ROOT / "datasets/PAL4VST/manifests/eval_manifest.jsonl",
}
PROVENANCE = {
    "Internal2208": ROOT / "datasets/Internal2208/manifests/provenance.json",
    "SynthScars": ROOT / "datasets/SynthScars/manifests/provenance.json",
    "X-AIGD": ROOT / "datasets/X-AIGD/manifests/provenance.json",
    "PAL4VST": ROOT / "datasets/PAL4VST/manifests/provenance.json",
    "LOKI": ROOT / "datasets/LOKI/manifests/provenance.json",
    "AIGI-Holmes": ROOT / "datasets/AIGI-Holmes/manifests/provenance.json",
    "GenImage": ROOT / "datasets/GenImage/manifests/provenance.json",
    "RAISE": ROOT / "datasets/RAISE/manifests/provenance.json",
}

REQUIRED_CLASSIFICATION = {
    ("r1", "aigi_holmes"),
    ("r1", "genimage"),
    ("r1", "loki"),
    ("r1", "raise998"),
    ("legion_retrained", "aigi_holmes"),
    ("legion_retrained", "genimage"),
    ("legion_retrained", "loki"),
    ("legion_retrained", "raise998"),
}
PUBLIC_INTERMEDIATE_CLASSIFICATION_NOTE = (
    "N/A: the released legion_LE intermediate checkpoint is LE-only and has no trained prediction_head"
)
EXPECTED_CLASS_CHECKPOINT = {
    "r1": ("sha256", "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"),
    "legion_retrained": ("canonical_sha256", "f33fda9ddf0e22bcd9bca8dcc998d8a9c0c6421f6bd6a46573044c6cc7365739"),
}
EXPECTED_LOC_CHECKPOINT = {
    "p1": ("sha256", "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"),
    "r1": ("sha256", "9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5"),
    "legion_intermediate": ("hf_revision", "f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa"),
    "legion_retrained": ("canonical_sha256", "6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844"),
}
REQUIRED_LOCALIZATION = {
    (model, dataset)
    for model in ("p1", "r1", "legion_intermediate", "legion_retrained")
    for dataset in ("synthscars", "loki", "xaigd", "pal4vst")
}
EXPECTED_LOC_CONDITION = {
    ("p1", "synthscars"): "G0",
    ("r1", "synthscars"): "G0",
    ("p1", "loki"): "G1",
    ("r1", "loki"): "G1",
    ("p1", "xaigd"): "G1",
    ("r1", "xaigd"): "G1",
    ("p1", "pal4vst"): "G1",
    ("r1", "pal4vst"): "G1",
    ("legion_intermediate", "synthscars"): "L-FREE",
    ("legion_retrained", "synthscars"): "L-FREE",
    ("legion_intermediate", "loki"): "L-FREE",
    ("legion_retrained", "loki"): "L-FREE",
    ("legion_intermediate", "xaigd"): "L-FREE",
    ("legion_retrained", "xaigd"): "L-FREE",
    ("legion_intermediate", "pal4vst"): "L-FREE",
    ("legion_retrained", "pal4vst"): "L-FREE",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, fieldnames: list[str], values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    os.replace(tmp, path)


def manifest_meta(path: Path) -> dict:
    values = rows(path)
    ids = [str(row["sample_id"]) for row in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate sample IDs: {path}")
    labels = Counter(str(row.get("label")) for row in values)
    sources = Counter(str(row.get("generator/source")) for row in values)
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path), "n": len(values),
        "ordered_sample_id_sha256": ordered_id_sha(ids),
        "labels": dict(labels), "sources": dict(sources), "ids": ids,
    }


def norm_metric(metrics: dict, key: str):
    aliases = {
        "specificity_tnr": ("specificity_tnr", "specificity"),
        "roc_auc": ("roc_auc",),
        "auprc": ("auprc",),
        "brier": ("brier",),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in metrics:
            return metrics[candidate]
    return None


def classification_row(model: str, dataset: str, result: dict, execution: str = "RUN_NEW") -> dict:
    m = result["metrics"]
    tnr = norm_metric(m, "specificity_tnr")
    if tnr is None and m.get("tn") is not None and m.get("fp") is not None:
        tnr = float(m["tn"]) / max(1, int(m["tn"]) + int(m["fp"]))
    fpr = m.get("fpr")
    if fpr is None and m.get("tn") is not None and m.get("fp") is not None:
        fpr = float(m["fp"]) / max(1, int(m["tn"]) + int(m["fp"]))
    if fpr is None and tnr is not None:
        fpr = 1.0 - float(tnr)
    return {
        "model": model,
        "dataset": dataset,
        "execution": execution,
        "n": m.get("n"), "real": m.get("real"), "fake": m.get("fake"),
        "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"),
        "specificity_tnr": tnr, "fpr": fpr, "f1": m.get("f1"),
        "roc_auc": norm_metric(m, "roc_auc"), "auprc": norm_metric(m, "auprc"),
        "brier": norm_metric(m, "brier"),
        "tp": m.get("tp"), "tn": m.get("tn"), "fp": m.get("fp"), "fn": m.get("fn"),
        "threshold": m.get("threshold", 0.5),
    }


def load_current_classification() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows_out: list[dict] = []
    raw: dict[tuple[str, str], dict] = {}
    for model, dataset in sorted(REQUIRED_CLASSIFICATION):
        path = CLASS / model / dataset / "results.json"
        if not path.is_file():
            raise RuntimeError(f"missing classification result: {path}")
        value = read_json(path)
        if value.get("status") != "COMPLETE":
            raise RuntimeError(f"classification result incomplete: {path}")
        manifest = manifest_meta(CLASS_MANIFESTS[dataset])
        if value.get("manifest_sha256") != manifest["sha256"]:
            raise RuntimeError(f"classification manifest drift: {model}/{dataset}")
        if int(value.get("metrics", {}).get("n", -1)) != manifest["n"]:
            raise RuntimeError(f"classification N drift: {model}/{dataset}")
        key, expected = EXPECTED_CLASS_CHECKPOINT[model]
        if value.get("checkpoint", {}).get(key) != expected:
            raise RuntimeError(f"classification checkpoint drift: {model}/{dataset}")
        prediction_file = CLASS / model / dataset / "predictions.jsonl"
        predictions = rows(prediction_file)
        prediction_ids = [str(row.get("sample_id")) for row in predictions]
        if prediction_ids != manifest["ids"] or len(prediction_ids) != len(set(prediction_ids)):
            raise RuntimeError(f"classification prediction identity/order drift: {model}/{dataset}")
        raw[(model, dataset)] = value
        rows_out.append(classification_row(model, dataset, value))
    return rows_out, raw


def historical_internal_classification() -> tuple[list[dict], dict]:
    """Reuse the already-frozen Internal2208 results; never runs a model."""
    manifest = manifest_meta(CLASS_MANIFESTS["internal"])
    output_rows: list[dict] = []
    info: dict[str, Any] = {}

    p1_predictions = ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal/detection/predictions.jsonl"
    p1_metrics = ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal/detection/metrics.json"
    invariance = ROOT / "outputs/p1_r1_reusable_matrix/classification_invariance.json"
    if not (p1_predictions.is_file() and p1_metrics.is_file() and invariance.is_file()):
        raise RuntimeError("historical Internal2208 P1/R1 classification artifacts are missing")
    prediction_rows = rows(p1_predictions)
    if [str(row["sample_id"]) for row in prediction_rows] != manifest["ids"]:
        raise RuntimeError("Internal2208 historical P1 sample identity/order drift")
    inv = read_json(invariance)
    if inv.get("conclusion") != "EXACT_REUSE":
        raise RuntimeError("R1 classification invariance is not EXACT_REUSE")
    pm = read_json(p1_metrics)["classification_head"]
    p1_result = {"metrics": {**pm, "n": 2208, "real": 1104, "fake": 1104, "threshold": 0.5}}
    for model, execution in (("p1", "REUSED_HISTORICAL"), ("r1", "EXACT_REUSE_P1")):
        output_rows.append(classification_row(model, "internal", p1_result, execution=execution))
    info["p1_r1"] = {
        "predictions": str(p1_predictions.resolve()), "predictions_sha256": sha256_file(p1_predictions),
        "metrics": str(p1_metrics.resolve()), "metrics_sha256": sha256_file(p1_metrics),
        "invariance": str(invariance.resolve()), "invariance_sha256": sha256_file(invariance),
    }

    legion_result_path = ROOT / "outputs/phase5a4_legion_retrained_controlled/classification/internal/results.json"
    legion_predictions = ROOT / "outputs/phase5a4_legion_retrained_controlled/classification/internal/predictions.jsonl"
    if not (legion_result_path.is_file() and legion_predictions.is_file()):
        raise RuntimeError("historical Internal2208 LEGION-retrained classification artifact is missing")
    legion_rows = rows(legion_predictions)
    if [str(row["sample_id"]) for row in legion_rows] != manifest["ids"]:
        raise RuntimeError("Internal2208 historical LEGION sample identity/order drift")
    legion = read_json(legion_result_path)
    if legion.get("status") != "COMPLETE" or int(legion.get("metrics", {}).get("n", -1)) != 2208:
        raise RuntimeError("historical Internal2208 LEGION result incomplete")
    output_rows.append(classification_row("legion_retrained", "internal", legion, execution="REUSED_HISTORICAL"))
    info["legion_retrained"] = {
        "results": str(legion_result_path.resolve()), "results_sha256": sha256_file(legion_result_path),
        "predictions": str(legion_predictions.resolve()), "predictions_sha256": sha256_file(legion_predictions),
    }
    return output_rows, info


def load_localization() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows_out: list[dict] = []
    raw: dict[tuple[str, str], dict] = {}
    for model, dataset in sorted(REQUIRED_LOCALIZATION):
        path = LOC / model / dataset / "results.json"
        if not path.is_file():
            raise RuntimeError(f"missing localization result: {path}")
        value = read_json(path)
        if value.get("status") != "COMPLETE":
            raise RuntimeError(f"localization incomplete: {path}")
        manifest = manifest_meta(LOC_MANIFESTS[dataset])
        embedded = value.get("manifest", {})
        if embedded.get("sha256") != manifest["sha256"] or int(embedded.get("n", -1)) != manifest["n"]:
            raise RuntimeError(f"localization manifest/N drift: {model}/{dataset}")
        expected_condition = EXPECTED_LOC_CONDITION[(model, dataset)]
        if value.get("inference_condition") != expected_condition:
            raise RuntimeError(f"localization condition drift: {model}/{dataset}: {value.get('inference_condition')} != {expected_condition}")
        metrics = value["metrics"]
        if int(metrics.get("n", -1)) != manifest["n"]:
            raise RuntimeError(f"localization metrics N drift: {model}/{dataset}")
        if value.get("failure_policy", {}).get("version") != "full_n_no_seg_zero_v1":
            raise RuntimeError(f"localization failure-policy drift: {model}/{dataset}")
        key, expected = EXPECTED_LOC_CHECKPOINT[model]
        if value.get("checkpoint", {}).get(key) != expected:
            raise RuntimeError(f"localization checkpoint drift: {model}/{dataset}")
        prediction_file = LOC / model / dataset / "predictions.jsonl"
        predictions = rows(prediction_file)
        prediction_ids = [str(row.get("sample_id")) for row in predictions]
        if prediction_ids != manifest["ids"] or len(prediction_ids) != len(set(prediction_ids)):
            raise RuntimeError(f"localization prediction identity/order drift: {model}/{dataset}")
        tp = sum(int(row["tp"]) for row in predictions)
        fp = sum(int(row["fp"]) for row in predictions)
        fn = sum(int(row["fn"]) for row in predictions)
        recomputed = {
            "mean_foreground_iou": sum(float(row["foreground_iou"]) for row in predictions) / len(predictions),
            "mean_foreground_f1": sum(float(row["foreground_f1"]) for row in predictions) / len(predictions),
            "global_foreground_iou": tp / max(1, tp + fp + fn),
            "global_foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
        for metric_name, metric_value in recomputed.items():
            if abs(float(metrics[metric_name]) - metric_value) > 1e-12:
                raise RuntimeError(f"localization metric drift: {model}/{dataset}/{metric_name}")
        raw[(model, dataset)] = value
        rows_out.append({
            "model": model, "dataset": dataset, "split": value.get("split"),
            "execution": value.get("execution"), "inference_condition": value.get("inference_condition"),
            "n": metrics.get("n"), "gt_type": value.get("gt_type"),
            "empty_gt_images": metrics.get("empty_gt_images", 0),
            "nonempty_gt_images": metrics.get("nonempty_gt_images", metrics.get("n")),
            "mean_foreground_iou": metrics.get("mean_foreground_iou"),
            "median_foreground_iou": metrics.get("median_foreground_iou"),
            "mean_foreground_f1": metrics.get("mean_foreground_f1"),
            "mean_foreground_iou_nonempty_gt": metrics.get("mean_foreground_iou_nonempty_gt"),
            "mean_foreground_f1_nonempty_gt": metrics.get("mean_foreground_f1_nonempty_gt"),
            "global_foreground_iou": metrics.get("global_foreground_iou"),
            "global_foreground_f1": metrics.get("global_foreground_f1"),
            "threshold_logit": metrics.get("threshold_logit"),
            "metric_semantics": value.get("metric_semantics"),
        })
    return rows_out, raw


def unavailable_public_intermediate_rows() -> list[dict]:
    result = []
    for dataset, manifest_path in CLASS_MANIFESTS.items():
        manifest = manifest_meta(manifest_path)
        result.append({
            "model": "legion_intermediate", "dataset": dataset,
            "execution": "N/A_NO_CLASSIFICATION_HEAD", "n": manifest["n"],
            "real": manifest["labels"].get("Real", 0), "fake": manifest["labels"].get("Fake", 0),
            "accuracy": None, "precision": None, "recall": None, "specificity_tnr": None,
            "fpr": None, "f1": None, "roc_auc": None, "auprc": None, "brier": None,
            "tp": None, "tn": None, "fp": None, "fn": None, "threshold": None,
            "availability_note": PUBLIC_INTERMEDIATE_CLASSIFICATION_NOTE,
        })
    return result


def localization_coverage_rows() -> list[dict]:
    output = []
    for model, dataset in sorted(REQUIRED_LOCALIZATION):
        values = rows(LOC / model / dataset / "predictions.jsonl")
        statuses = Counter(str(row.get("status") or "UNKNOWN") for row in values)
        failure_statuses = {
            "EMPTY_OR_NO_SEG", "NO_VALID_QSEG", "NO_SEG", "EMPTY_MASK",
            "INVALID_SHAPE", "MASK_WITHOUT_SEG",
        }
        failures = sum(
            bool(row.get("failure_score_forced_zero"))
            or str(row.get("status") or "") in failure_statuses
            or row.get("seg_triggered") is False
            or row.get("valid_q_seg") is False
            or ("seg_count" in row and int(row.get("seg_count") or 0) <= 0)
            for row in values
        )
        output.append({
            "model": model, "dataset": dataset, "n": len(values),
            "valid": len(values) - failures, "protocol_failures": failures,
            "status_counts": json.dumps(dict(sorted(statuses.items())), ensure_ascii=False, sort_keys=True),
        })
    return output


def p1_r1_paired_rows() -> list[dict]:
    from tools.phase4e1 import compare
    output = []
    for dataset in sorted(LOC_MANIFESTS):
        p1 = rows(LOC / "p1" / dataset / "predictions.jsonl")
        r1 = rows(LOC / "r1" / dataset / "predictions.jsonl")
        if [row["sample_id"] for row in p1] != [row["sample_id"] for row in r1]:
            raise RuntimeError(f"P1/R1 paired localization identity drift: {dataset}")
        statistics = compare(r1, p1, seed=3407)
        for metric in ("foreground_iou", "foreground_f1"):
            deltas = [float(b[metric]) - float(a[metric]) for a, b in zip(p1, r1)]
            stat = statistics[metric]
            output.append({
                "dataset": dataset, "metric": metric, "n": len(deltas),
                "p1_mean": sum(float(row[metric]) for row in p1) / len(p1),
                "r1_mean": sum(float(row[metric]) for row in r1) / len(r1),
                "mean_delta_r1_minus_p1": sum(deltas) / len(deltas),
                "wins": sum(value > 0 for value in deltas),
                "ties": sum(value == 0 for value in deltas),
                "losses": sum(value < 0 for value in deltas),
                "bootstrap_95_ci_low": stat["bootstrap_95_ci"][0],
                "bootstrap_95_ci_high": stat["bootstrap_95_ci"][1],
                "wilcoxon_pvalue": stat["wilcoxon_pvalue"],
            })
    return output


def leakage_payload() -> dict:
    if not LEAKAGE_SUMMARY.is_file() or not LEAKAGE_OVERRIDE.is_file():
        raise RuntimeError("leakage audit/override artifacts missing")
    summary = read_json(LEAKAGE_SUMMARY)
    override = read_json(LEAKAGE_OVERRIDE)
    if summary.get("status") == "BLOCKED_OVERLAP" and override.get("status") != "ACTIVE":
        raise RuntimeError("leakage audit BLOCKED_OVERLAP without ACTIVE override")
    exact = rows(LEAKAGE_EXACT)
    phash = rows(LEAKAGE_PHASH)
    exact_by = Counter(str(row.get("affected_dataset")) for row in exact)
    phash_by = Counter(str(row.get("right_dataset")) for row in phash)
    aigi_exact = rows(AIGI_GEN_EXACT)
    aigi_phash = rows(AIGI_GEN_PHASH)
    return {
        "summary": summary,
        "override": override,
        "external_vs_internal_train": {
            "exact_overlap": len(exact), "phash_overlap": len(phash),
            "exact_by_dataset": dict(sorted(exact_by.items())),
            "phash_by_dataset": dict(sorted(phash_by.items())),
            "exact_artifact": str(LEAKAGE_EXACT.resolve()),
            "phash_artifact": str(LEAKAGE_PHASH.resolve()),
        },
        "aigi_holmes_vs_genimage": {
            "exact_overlap": len(aigi_exact), "phash_overlap": len(aigi_phash),
            "exact_label_pairs": dict(sorted(Counter(f"{r.get('aigi_label')}/{r.get('genimage_label')}" for r in aigi_exact).items())),
            "phash_label_pairs": dict(sorted(Counter(f"{r.get('right_label')}/{r.get('left_label')}" for r in aigi_phash).items())),
            "exact_artifact": str(AIGI_GEN_EXACT.resolve()),
            "phash_artifact": str(AIGI_GEN_PHASH.resolve()),
        },
    }


def integrity_rows() -> list[dict]:
    specs = [
        ("Internal2208", "classification", CLASS_MANIFESTS["internal"]),
        ("SynthScars", "localization", LOC_MANIFESTS["synthscars"]),
        ("X-AIGD", "localization", LOC_MANIFESTS["xaigd"]),
        ("PAL4VST", "localization", LOC_MANIFESTS["pal4vst"]),
        ("LOKI-localization", "localization", LOC_MANIFESTS["loki"]),
        ("LOKI-classification", "classification", CLASS_MANIFESTS["loki"]),
        ("AIGI-Holmes", "classification", CLASS_MANIFESTS["aigi_holmes"]),
        ("GenImage", "classification", CLASS_MANIFESTS["genimage"]),
        ("RAISE998", "classification", CLASS_MANIFESTS["raise998"]),
    ]
    result = []
    for dataset, task, path in specs:
        meta = manifest_meta(path)
        provenance_name = dataset.split("-")[0] if dataset.startswith("LOKI-") else ("RAISE" if dataset == "RAISE998" else dataset)
        provenance_path = PROVENANCE.get(provenance_name)
        provenance = read_json(provenance_path) if provenance_path and provenance_path.is_file() else {}
        result.append({
            "dataset": dataset, "task": task, "n": meta["n"], "manifest": meta["path"],
            "manifest_sha256": meta["sha256"], "ordered_sample_id_sha256": meta["ordered_sample_id_sha256"],
            "release_or_revision": json.dumps(provenance.get("release_or_revision"), ensure_ascii=False),
            "official_source": provenance.get("official_source"),
            "provenance": str(provenance_path.resolve()) if provenance_path and provenance_path.is_file() else None,
        })
    return result


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], values: list[list[Any]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in values:
        result.append("| " + " | ".join(str(x) for x in row) + " |")
    return result


def render_report(class_rows: list[dict], loc_rows: list[dict], leak: dict, integrity: list[dict]) -> str:
    summary = leak["summary"]
    override = leak["override"]
    internal = leak["external_vs_internal_train"]
    independence = leak["aigi_holmes_vs_genimage"]
    lines = [
        "# Final Frozen Evaluation Report",
        "",
        f"Generated: `{now()}`",
        "",
        "## Data leakage warning",
        "",
        f"The frozen leakage audit status is **{summary.get('status')}**. Evaluation proceeded because `leakage_override.json` is **{override.get('status')}**, under the recorded user authorization. This is an override, not a passed leakage audit.",
        "",
        f"External benchmark ↔ internal train: **{internal['exact_overlap']} exact SHA256 overlaps** and **{internal['phash_overlap']} pHash near-overlap pairs** (frozen Hamming threshold from the data audit). No overlapping sample was removed by the evaluation scripts.",
        "",
        "Affected-dataset exact counts: `" + json.dumps(internal["exact_by_dataset"], ensure_ascii=False, sort_keys=True) + "`  ",
        "Affected-dataset pHash counts: `" + json.dumps(internal["phash_by_dataset"], ensure_ascii=False, sort_keys=True) + "`",
        "",
        f"AIGI-Holmes TestSet ↔ GenImage held-out: **{independence['exact_overlap']} exact** and **{independence['phash_overlap']} pHash near-overlap pairs**.",
        "",
        "## Classification",
        "",
        f"LEGION public intermediate classification is **{PUBLIC_INTERMEDIATE_CLASSIFICATION_NOTE}**.",
        "",
    ]
    class_sorted = sorted(class_rows, key=lambda r: (r["dataset"], r["model"]))
    lines += markdown_table(
        ["Dataset", "Model", "Execution", "N", "Real", "Fake", "Accuracy", "F1", "ROC-AUC", "TNR", "FPR"],
        [[r["dataset"], r["model"], r["execution"], r["n"], r["real"], r["fake"], fmt(r["accuracy"]), fmt(r["f1"]), fmt(r["roc_auc"]), fmt(r["specificity_tnr"]), fmt(r["fpr"])] for r in class_sorted],
    )
    lines += [
        "",
        "RAISE998 is a **Real-only OOD false-positive benchmark**. Its meaningful primary quantities are TNR/FPR (and TN/FP); it is not described as a balanced-accuracy benchmark.",
        "",
        "## Localization",
        "",
        "Inference-condition boundary: SynthScars reuses historical **P1/R1 G0**; LOKI reuses historical **P1/R1 G1**; the new X-AIGD and PAL4VST runs use **P1/R1 G1** as explicitly frozen for this final evaluation. LEGION public-intermediate and LEGION-retrained use official **L-FREE** throughout. Therefore P1/R1-vs-LEGION numbers are cross-protocol comparisons rather than identical-prompt comparisons.",
        "",
    ]
    loc_sorted = sorted(loc_rows, key=lambda r: (r["dataset"], r["model"]))
    lines += markdown_table(
        ["Dataset", "Model", "Condition", "Execution", "N", "Empty-GT", "GT type", "Mean FG IoU", "Mean FG IoU (nonempty GT)", "Mean FG F1", "Global FG IoU", "Global FG F1"],
        [[r["dataset"], r["model"], r["inference_condition"], r["execution"], r["n"], r["empty_gt_images"], r["gt_type"], fmt(r["mean_foreground_iou"]), fmt(r["mean_foreground_iou_nonempty_gt"]), fmt(r["mean_foreground_f1"]), fmt(r["global_foreground_iou"]), fmt(r["global_foreground_f1"])] for r in loc_sorted],
    )
    lines += [
        "",
        "### Localization GT semantics",
        "",
        "- **SynthScars Official1000:** official per-reference polygons, unioned by the evaluator at original resolution.",
        "- **X-AIGD labeled_test:** official human perceptual-artifact polygons. Official records with `labels=[]` are retained and evaluated as all-zero GT masks, matching the official evaluator. Raw polygons remain authoritative; rasterization follows the official int32/clamping + `cv2.fillPoly` policy. For paper-level X-AIGD comparison, dataset-global FG IoU/F1 are the primary category-agnostic metrics; per-image means are supplementary.",
        "- **PAL4VST test:** official pixel artifact masks.",
        "- **LOKI229:** union of 687 official regional bounding boxes on 229 Fake images. This is box-derived localization GT and is not semantically equivalent to X-AIGD/PAL4VST perceptual-artifact masks.",
        "",
        "## GenImage per-generator classification",
        "",
    ]
    gen_rows = []
    for row in class_rows:
        if row["dataset"] != "genimage":
            continue
        if row.get("execution") == "N/A_NO_CLASSIFICATION_HEAD":
            continue
        result_path = CLASS / row["model"] / "genimage/results.json"
        result = read_json(result_path)
        for source, metrics in sorted(result.get("metrics", {}).get("per_source", {}).items()):
            gen_rows.append([row["model"], source, metrics.get("n"), metrics.get("real"), metrics.get("fake"), fmt(metrics.get("accuracy")), fmt(metrics.get("f1")), fmt(metrics.get("roc_auc")), fmt(metrics.get("specificity_tnr")), fmt(metrics.get("fpr"))])
    lines += markdown_table(["Model", "Generator", "N", "Real", "Fake", "Accuracy", "F1", "ROC-AUC", "TNR", "FPR"], gen_rows)
    lines += [
        "",
        "## Benchmark integrity",
        "",
    ]
    lines += markdown_table(
        ["Dataset", "Task", "N", "Manifest SHA256", "Revision"],
        [[r["dataset"], r["task"], r["n"], r["manifest_sha256"], r["release_or_revision"]] for r in integrity],
    )
    lines += [
        "",
        "## Final frozen protocol",
        "",
        "Localization: SynthScars Official1000; X-AIGD official `labeled_test`; PAL4VST official `test`; LOKI229. Models: P1, R1, public intermediate LEGION, and LEGION-retrained.",
        "",
        "Classification: Internal2208 (historical frozen reuse), AIGI-Holmes official TestSet, GenImage official held-out partition, LOKI classification, and RAISE998 Real-only. The old project-created `AIGI-test` is not used as the main official benchmark.",
        "",
        "No threshold tuning, benchmark-dependent sample filtering, test resplitting, or model selection is performed by the finalizer.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    state = {"schema": "final_eval_finalizer_v1", "status": "RUNNING", "stage": "preflight", "pid": os.getpid(), "started_at_utc": now()}
    atomic_json(STATUS, state)
    try:
        leak = leakage_payload()
        state["stage"] = "classification"; atomic_json(STATUS, state)
        class_rows, class_raw = load_current_classification()
        internal_rows, internal_info = historical_internal_classification()
        class_rows.extend(internal_rows)
        class_rows.extend(unavailable_public_intermediate_rows())

        state["stage"] = "localization"; atomic_json(STATUS, state)
        loc_rows, loc_raw = load_localization()
        coverage_rows = localization_coverage_rows()
        paired_rows = p1_r1_paired_rows()
        integrity = integrity_rows()

        state["stage"] = "tables"; atomic_json(STATUS, state)
        write_csv(TABLES / "classification_main.csv", [
            "model", "dataset", "execution", "n", "real", "fake", "accuracy", "precision", "recall",
            "specificity_tnr", "fpr", "f1", "roc_auc", "auprc", "brier", "tp", "tn", "fp", "fn", "threshold",
        ], sorted(class_rows, key=lambda r: (r["dataset"], r["model"])))

        genimage_rows = []
        for (model, dataset), result in sorted(class_raw.items()):
            if dataset != "genimage":
                continue
            for source, metrics in sorted(result.get("metrics", {}).get("per_source", {}).items()):
                row = classification_row(model, f"genimage:{source}", {"metrics": metrics})
                row["generator"] = source
                genimage_rows.append(row)
        write_csv(TABLES / "genimage_per_generator.csv", [
            "model", "generator", "n", "real", "fake", "accuracy", "precision", "recall", "specificity_tnr",
            "fpr", "f1", "roc_auc", "auprc", "brier", "tp", "tn", "fp", "fn", "threshold",
        ], genimage_rows)

        raise_row = next(r for r in class_rows if r["model"] == "legion_retrained" and r["dataset"] == "raise998")
        write_csv(TABLES / "raise_real_only.csv", ["model", "dataset", "n", "real", "fake", "tn", "fp", "specificity_tnr", "fpr", "threshold"], [raise_row])

        write_csv(TABLES / "localization_main.csv", [
            "model", "dataset", "split", "execution", "inference_condition", "n", "empty_gt_images",
            "nonempty_gt_images", "gt_type", "mean_foreground_iou", "median_foreground_iou",
            "mean_foreground_f1", "mean_foreground_iou_nonempty_gt", "mean_foreground_f1_nonempty_gt",
            "global_foreground_iou", "global_foreground_f1", "threshold_logit", "metric_semantics",
        ], sorted(loc_rows, key=lambda r: (r["dataset"], r["model"])))
        write_csv(TABLES / "localization_coverage.csv", [
            "model", "dataset", "n", "valid", "protocol_failures", "status_counts",
        ], coverage_rows)
        write_csv(TABLES / "localization_p1_r1_paired.csv", [
            "dataset", "metric", "n", "p1_mean", "r1_mean", "mean_delta_r1_minus_p1",
            "wins", "ties", "losses", "bootstrap_95_ci_low", "bootstrap_95_ci_high", "wilcoxon_pvalue",
        ], paired_rows)

        write_csv(TABLES / "benchmark_integrity.csv", [
            "dataset", "task", "n", "manifest", "manifest_sha256", "ordered_sample_id_sha256",
            "official_source", "release_or_revision", "provenance",
        ], integrity)

        payload = {
            "schema": "final_evaluation_results_v1",
            "status": "COMPLETE",
            "generated_at_utc": now(),
            "leakage": leak,
            "classification": {
                "rows": class_rows,
                "current_results": {f"{m}/{d}": v for (m, d), v in class_raw.items()},
                "historical_internal_reuse": internal_info,
            },
            "localization": {
                "rows": loc_rows,
                "results": {f"{m}/{d}": v for (m, d), v in loc_raw.items()},
                "protocol_boundary": {
                    "synthscars": "P1/R1 G0 vs LEGION L-FREE",
                    "loki": "P1/R1 G1 vs LEGION L-FREE",
                    "xaigd": "P1/R1 G1 vs LEGION L-FREE",
                    "pal4vst": "P1/R1 G1 vs LEGION L-FREE",
                },
                "coverage": coverage_rows,
                "p1_r1_paired": paired_rows,
            },
            "integrity": integrity,
            "tables": {
                "classification_main": str((TABLES / "classification_main.csv").resolve()),
                "genimage_per_generator": str((TABLES / "genimage_per_generator.csv").resolve()),
                "raise_real_only": str((TABLES / "raise_real_only.csv").resolve()),
                "localization_main": str((TABLES / "localization_main.csv").resolve()),
                "localization_coverage": str((TABLES / "localization_coverage.csv").resolve()),
                "localization_p1_r1_paired": str((TABLES / "localization_p1_r1_paired.csv").resolve()),
                "benchmark_integrity": str((TABLES / "benchmark_integrity.csv").resolve()),
            },
        }
        atomic_json(FINAL_JSON, payload)
        report = render_report(class_rows, loc_rows, leak, integrity)
        atomic_text(FINAL_REPORT, report + "\n")
        atomic_text(DOC_REPORT, report + "\n")
        state.update({
            "status": "COMPLETE", "stage": "complete", "updated_at_utc": now(),
            "final_results": str(FINAL_JSON.resolve()), "final_report": str(FINAL_REPORT.resolve()),
            "docs_report": str(DOC_REPORT.resolve()),
        })
        atomic_json(STATUS, state)
    except BaseException as error:
        state.update({"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error), "updated_at_utc": now()})
        atomic_json(STATUS, state)
        raise


if __name__ == "__main__":
    main()
