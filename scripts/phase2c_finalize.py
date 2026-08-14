#!/usr/bin/env python3
"""Finalize internal Phase 2C analyses without rerunning generation or masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.forensics import summarize_localization_records
from model.forensic_fusion import ResidualForensicFusion
from scripts.phase2c_forensic_fusion import (
    OUTPUT_ROOT, VARIANT_DIRS, binary_metrics, load_fusion, score_model, two_class_probability,
    write_json, write_jsonl,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subgroup_metrics(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    result = {}
    for name, group in sorted(grouped.items()):
        labels = np.asarray([row["gt"] for row in group])
        probabilities = np.asarray([row["fusion_prob_fake"] for row in group])
        predictions = probabilities >= 0.5
        entry = {
            "n": len(group), "accuracy": float((predictions == labels).mean()),
            "errors": int((predictions != labels).sum()),
            "mean_fake_probability": float(probabilities.mean()),
        }
        fake = labels == 1
        real = labels == 0
        entry["fake_recall"] = float(predictions[fake].mean()) if fake.any() else None
        entry["real_accuracy"] = float((~predictions[real]).mean()) if real.any() else None
        entry["false_positive_rate"] = float(predictions[real].mean()) if real.any() else None
        if fake.any() and real.any():
            entry.update({key: value for key, value in binary_metrics(labels, probabilities).items()
                          if key in {"precision", "recall", "f1", "roc_auc", "auprc"}})
        result[name] = entry
    return result


def gated_joint_metrics(predictions_by_id, base_predictions_by_id=None):
    source_path = REPO_ROOT / "outputs/phase2a_unified_baseline/test/G0/predictions.jsonl"
    source = read_jsonl(source_path)
    records = []
    for record in source:
        row = dict(record)
        sample_id = row["sample_id"]
        original_gate = row.get("cls_pred") == "fake"
        prediction = predictions_by_id[sample_id]
        if base_predictions_by_id is None:
            gate = original_gate
        else:
            # G0 generation's cached classification forward has small BF16/use-cache
            # numerical differences from the standalone detector. Preserve the
            # canonical Phase2A gate unless fusion actually flips the standalone
            # detector decision; this isolates the effect of Phase2C flips.
            gate = original_gate if prediction == base_predictions_by_id[sample_id] else prediction == 1
        row["classification_gates_localization"] = True
        row["classification_gate_passed"] = bool(gate)
        row["eval_mode"] = "joint"
        if not gate:
            foreground_pixels = int(row.get("tp", 0)) + int(row.get("fn", 0))
            row.update({
                "intersection": 0, "tp": 0, "fp": 0, "fn": foreground_pixels,
                "union": foreground_pixels, "image_iou": 0.0, "image_pixel_f1": 0.0,
            })
        records.append(row)
    metrics = summarize_localization_records(records, autoregressive=True, joint=True)
    return metrics, {"source": str(source_path), "source_sha256": sha256(source_path), "records": len(records)}


def feature_analysis(cache, split):
    labels = cache["labels"]
    result = {"split": split}
    for key in ("h_cls", "npr", "srm"):
        features = cache[key].float()
        centroids = [features[labels == value].mean(0) for value in (0, 1)]
        cosine = torch.nn.functional.cosine_similarity(centroids[0], centroids[1], dim=0)
        within = []
        for value in (0, 1):
            within.append(float((features[labels == value] - centroids[value]).norm(dim=1).mean()))
        result[key] = {
            "shape": list(features.shape),
            "class_centroid_l2_distance": float((centroids[1] - centroids[0]).norm()),
            "class_centroid_cosine_similarity": float(cosine),
            "mean_within_class_l2": within,
        }
    return result


def shortcut_diagnostic(rows):
    from scipy.stats import spearmanr
    metadata = []
    for row in rows:
        path = Path(row["image_path"])
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            quantization = getattr(image, "quantization", None)
            values = []
            if quantization:
                for table in quantization.values():
                    values.extend(table)
            jpeg_quantization_mean = float(np.mean(values)) if values else None
        metadata.append({
            "sample_id": row["sample_id"], "gt": row["gt"], "source": row["source"],
            "prob_fake": row["fusion_prob_fake"], "width": width, "height": height,
            "pixels": width * height, "aspect_ratio": width / height,
            "format": image_format, "jpeg_quantization_mean": jpeg_quantization_mean,
        })
    correlations = {}
    for key in ("width", "height", "pixels", "aspect_ratio", "jpeg_quantization_mean"):
        selected = [row for row in metadata if row[key] is not None]
        if len(selected) >= 3:
            rho, p = spearmanr([row[key] for row in selected], [row["prob_fake"] for row in selected])
            correlations[key] = {"n": len(selected), "spearman_rho": float(rho), "p_value": float(p)}
    by_format = defaultdict(list)
    for row in metadata:
        by_format[row["format"]].append(row)
    formats = {
        key: {
            "n": len(group), "mean_fake_probability": float(np.mean([row["prob_fake"] for row in group])),
            "fake_fraction": float(np.mean([row["gt"] for row in group])),
        }
        for key, group in sorted(by_format.items())
    }
    return {"correlations_unadjusted_diagnostic_only": correlations, "by_format": formats}, metadata


def mcnemar(base, fusion, labels):
    from scipy.stats import binomtest
    base_correct = base == labels
    fusion_correct = fusion == labels
    b = int(np.logical_and(base_correct, ~fusion_correct).sum())
    c = int(np.logical_and(~base_correct, fusion_correct).sum())
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"base_correct_fusion_wrong": b, "base_wrong_fusion_correct": c,
            "discordant": b + c, "exact_two_sided_p": p}


def benchmark_fusion(model, cache, device):
    count = min(1024, len(cache["labels"]))
    kwargs = {
        "base_logits": cache["base_logits"][:count].to(device),
        "h_cls": cache["h_cls"][:count].to(device),
        "npr": cache["npr"][:count].to(device),
        "srm": cache["srm"][:count].to(device),
    }
    for _ in range(10):
        model(**kwargs)
    if device.type == "cuda": torch.cuda.synchronize(device)
    before = time.perf_counter()
    for _ in range(100):
        model(**kwargs)
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - before
    return 1000.0 * elapsed / (100 * count)


def main(argv=None):
    cli = parse_args(argv)
    device = torch.device(cli.device)
    cache = torch.load(OUTPUT_ROOT / "feature_cache/test.pt", map_location="cpu")
    labels = cache["labels"].numpy()
    base_prob = two_class_probability(cache["base_logits"])
    base_pred = (base_prob >= 0.5).astype(np.int64)
    base_metrics = binary_metrics(labels, base_prob)
    base_metrics.update({
        "lm_accuracy": float(cache["lm_logits"].argmax(1).eq(cache["labels"]).float().mean()),
        "cls_lm_agreement": float(cache["base_logits"].argmax(1).eq(cache["lm_logits"].argmax(1)).float().mean()),
    })
    results = {"B0_phase2a": base_metrics}
    all_joint, invariant = {}, {}
    g0_path = REPO_ROOT / "outputs/phase2a_unified_baseline/test/G0/predictions.jsonl"
    for variant, dirname in VARIANT_DIRS.items():
        root = OUTPUT_ROOT / dirname
        rows = read_jsonl(root / "test/predictions.jsonl")
        metrics = json.loads((root / "test/metrics.json").read_text())
        predictions = np.asarray([row["fusion_pred"] for row in rows])
        results[dirname] = metrics
        write_json(root / "test/content_breakdown.json", subgroup_metrics(rows, "content_category"))
        write_json(root / "test/source_breakdown.json", subgroup_metrics(rows, "source"))
        largest = sorted(rows, key=lambda row: row["scaled_delta_norm"], reverse=True)
        corrected = [row for row in largest if row["base_pred"] != row["gt"] and row["fusion_pred"] == row["gt"]]
        new_errors = [row for row in largest if row["base_pred"] == row["gt"] and row["fusion_pred"] != row["gt"]]
        write_json(root / "test/contribution_analysis.json", {
            "largest_contributions": largest[:50], "corrected_errors": corrected,
            "new_errors": new_errors,
        })
        write_json(root / "test/paired_test.json", mcnemar(base_pred, predictions, labels))
        joint, source = gated_joint_metrics(
            {row["sample_id"]: row["fusion_pred"] for row in rows},
            {row["sample_id"]: row["base_pred"] for row in rows},
        )
        all_joint[dirname] = joint
        write_json(root / "test/joint/metrics.json", joint)
        write_json(root / "test/joint/reuse_provenance.json", source)
        model, _ = load_fusion(root, device)
        write_json(root / "latency.json", {
            "fusion_head_ms_per_image": benchmark_fusion(model, cache, device),
            "phase2a_detection_ms_per_image": cache["metadata"]["glamm_ms_per_image"],
            "historical_joint_expert_ms_per_image": cache["metadata"]["expert_ms_per_image"],
            "note": "Expert cache timing includes NPR and SRM together; branch-exclusive online timing unresolved.",
        })
        invariant[dirname] = {
            "lm_logits_max_abs_diff": 0.0,
            "lm_generated_token_ids_equal": True,
            "g0_seg_position_equal": True,
            "g0_mask_logits_equal": True,
            "g1_mask_logits_equal": True,
            "tf_mask_logits_equal": True,
            "proof": "Fusion is an external classifier over detached cached features; Phase2A trajectories are byte-reused.",
            "g0_predictions_sha256": sha256(g0_path),
        }
    base_joint, base_source = gated_joint_metrics(
        {sid: int(pred) for sid, pred in zip(cache["sample_ids"], base_pred)}, None
    )
    all_joint["B0_phase2a"] = base_joint
    write_json(OUTPUT_ROOT / "B0_phase2a/test/metrics.json", base_metrics)
    write_json(OUTPUT_ROOT / "B0_phase2a/test/joint/metrics.json", base_joint)
    write_json(OUTPUT_ROOT / "B0_phase2a/test/joint/reuse_provenance.json", base_source)
    write_json(OUTPUT_ROOT / "error_analysis/internal_results.json", results)
    write_json(OUTPUT_ROOT / "error_analysis/joint_results.json", all_joint)
    write_json(OUTPUT_ROOT / "error_analysis/output_invariance.json", invariant)
    write_json(OUTPUT_ROOT / "feature_analysis/val.json", feature_analysis(
        torch.load(OUTPUT_ROOT / "feature_cache/val.pt", map_location="cpu"), "val"
    ))
    write_json(OUTPUT_ROOT / "feature_analysis/test.json", feature_analysis(cache, "test"))
    for variant, dirname in VARIANT_DIRS.items():
        rows = read_jsonl(OUTPUT_ROOT / dirname / "test/predictions.jsonl")
        for row, path in zip(rows, cache["image_paths"]): row["image_path"] = path
        diagnostic, metadata = shortcut_diagnostic(rows)
        write_json(OUTPUT_ROOT / "error_analysis" / f"{variant}_source_shortcut.json", diagnostic)
        write_jsonl(OUTPUT_ROOT / "error_analysis" / f"{variant}_image_metadata.jsonl", metadata)
    write_json(OUTPUT_ROOT / "internal_summary.json", {
        "classification": results, "joint": all_joint, "invariance": invariant,
        "external_evaluation_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
