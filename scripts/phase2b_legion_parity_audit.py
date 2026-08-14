#!/usr/bin/env python3
"""Read-only Phase 2B parity, split, and localization gap audit.

This script never loads a model.  It hashes released/frozen images, rasterizes
the frozen annotations, and re-scores the already persisted Phase 2A pixel
confusion counts at the fixed logit threshold of zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.distributed import BalancedDistributedForensicsSampler
from dataset.forensics.synthscars import SynthScarsAdapter, polygon_to_mask, polygons_for_target


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    stat = path.stat()
    return {
        "path": str(path), "relative_path": str(path), "basename": path.name,
        "width": width, "height": height, "bytes": stat.st_size, "sha256": sha256(path),
    }


def safe_ratio(a: int | float, b: int | float, *, empty: float = 1.0) -> float:
    return float(a / b) if b else float(empty)


def suite(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    for row in rows:
        if "pixels" not in row:
            raise KeyError("pixels missing from metric row")
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["pixels"]) - int(row["tp"]) - int(row["fp"]) - int(row["fn"]) for row in rows)
    fg_iou = [safe_ratio(r["tp"], r["tp"] + r["fp"] + r["fn"], empty=1.0) for r in rows]
    bg_iou = []
    fg_f1 = []
    for r in rows:
        local_tn = r["pixels"] - r["tp"] - r["fp"] - r["fn"]
        bg_iou.append(safe_ratio(local_tn, local_tn + r["fp"] + r["fn"], empty=1.0))
        fg_f1.append(safe_ratio(2 * r["tp"], 2 * r["tp"] + r["fp"] + r["fn"], empty=1.0))
    return {
        "n": len(rows),
        "IoU_fg_per_image_mean": statistics.fmean(fg_iou),
        "IoU_fg_global": safe_ratio(tp, tp + fp + fn),
        "IoU_bg_per_image_mean": statistics.fmean(bg_iou),
        "IoU_bg_global": safe_ratio(tn, tn + fp + fn),
        "mIoU_fg_bg_per_image": statistics.fmean((f + b) / 2 for f, b in zip(fg_iou, bg_iou)),
        "mIoU_fg_bg_global": (safe_ratio(tp, tp + fp + fn) + safe_ratio(tn, tn + fp + fn)) / 2,
        "PixelF1_fg_per_image_mean": statistics.fmean(fg_f1),
        "PixelF1_fg_global": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "pixel_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "threshold": "mask_logit > 0 (sigmoid > 0.5)",
    }


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> dict:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None}
    rx, ry = rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    rho = numerator / denominator if denominator else None
    try:
        from scipy.stats import spearmanr
        scipy_result = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        p_value = float(scipy_result.pvalue) if math.isfinite(float(scipy_result.pvalue)) else None
    except ImportError:
        p_value = None
    return {"n": len(pairs), "rho": rho, "p_value_two_sided": p_value}


def levenshtein_similarity(a: str, b: str) -> float:
    left, right = a.lower().split(), b.lower().split()
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, 1):
        current = [i]
        for j, y in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return 1 - previous[-1] / max(len(left), len(right), 1)


def bin_name(value: float, boundaries: list[tuple[float, float, str]]) -> str:
    for low, high, name in boundaries:
        if low <= value < high:
            return name
    return boundaries[-1][2]


def summarize_bins(records: list[dict], key: str, order: list[str]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(row)
    output = {}
    for label in order:
        group = grouped.get(label, [])
        output[label] = {
            "n": len(group),
            "TF": suite([r["TF"] for r in group]),
            "G0": suite([r["G0"] for r in group]),
            "G1": suite([r["G1"] for r in group]),
        }
    return output


def source_and_split_audit(output: Path, synth_root: Path, manifest_root: Path):
    adapters = {split: SynthScarsAdapter(synth_root, split) for split in ("train", "test")}
    official_rows = {}
    fingerprints = {}
    for split, adapter in adapters.items():
        rows = list(adapter.iter_manifest_records())
        official_rows[split] = rows
        fingerprints[split] = {}
        for row in rows:
            path = synth_root / row["image_relpath"]
            fingerprints[split][row["sample_id"]] = image_fingerprint(path)

    frozen = {}
    for split in ("train", "val", "test"):
        frozen[split] = [r for r in load_jsonl(manifest_root / f"{split}_combined.jsonl") if r["source"] == "SynthScars"]

    hash_to_frozen = defaultdict(list)
    frozen_fp = {}
    for split, rows in frozen.items():
        for row in rows:
            fp = image_fingerprint(Path(row["image_path"]))
            frozen_fp[row["sample_id"]] = fp
            hash_to_frozen[fp["sha256"]].append((split, row))

    overlap = {official: {current: 0 for current in ("train", "val", "test", "absent")} for official in ("train", "test")}
    mappings = []
    for official_split in ("train", "test"):
        for row in official_rows[official_split]:
            fp = fingerprints[official_split][row["sample_id"]]
            matches = hash_to_frozen.get(fp["sha256"], [])
            if not matches:
                overlap[official_split]["absent"] += 1
            else:
                for current_split, _ in matches:
                    overlap[official_split][current_split] += 1
            if official_split == "test":
                mappings.append({
                    "official_sample_id": row["sample_id"], "official_image": row["image_name"],
                    "sha256": fp["sha256"], "dimensions": [fp["height"], fp["width"]],
                    "bytes": fp["bytes"], "matches": [
                        {"frozen_split": s, "sample_id": m["sample_id"], "image_path": m["image_path"]}
                        for s, m in matches
                    ],
                    "status": "absent" if not matches else "matched",
                })

    matrix = {
        "matching_key": "exact SHA256 (with basename/dimensions/bytes retained in mapping)",
        "official_annotation_rows": {"train": adapters["train"].annotation_count, "test": adapters["test"].annotation_count},
        "official_unique_images": {k: len(v) for k, v in official_rows.items()},
        "frozen_unique_images": {k: len(v) for k, v in frozen.items()},
        "frozen_official_annotation_rows": {
            k: sum(len(row.get("annotation_ids", [])) for row in rows) for k, rows in frozen.items()
        },
        "frozen_artifact_refs": {k: sum(len(row.get("refs", [])) for row in rows) for k, rows in frozen.items()},
        "overlap_matrix": overlap,
        "official_test_in_our_train": overlap["test"]["train"],
        "official_test_in_our_val": overlap["test"]["val"],
        "official_test_in_our_test": overlap["test"]["test"],
        "official_test_absent": overlap["test"]["absent"],
        "official_test_comparison_status": "CONTAMINATED_DIAGNOSTIC" if overlap["test"]["train"] else "ELIGIBLE_SPLIT_PARITY",
    }
    dump(output / "split_audit/official_vs_frozen_overlap.json", matrix)
    dump_jsonl(output / "split_audit/official_test_mapping.jsonl", mappings)

    content = {
        "official_release": {
            "train": {"human": 6253, "object": 1940, "animal": 1183, "scene": 1860, "total": 11236},
            "test": {"human": 587, "object": 162, "animal": 134, "scene": 117, "total": 1000},
            "provenance": "LEGION ICCV 2025 supplementary Table 1",
            "per_image_category_mapping_publicly_available": False,
        },
        "phase2a_frozen": {
            split: dict(Counter(r["content_category"] for r in rows), total=len(rows)) for split, rows in frozen.items()
        },
        "note": "Released JSON has no content-category field; official aggregate counts cannot be assigned per image without an unpublished mapping.",
    }
    dump(output / "split_audit/content_distribution.json", content)

    official_eval_rows = []
    for row in official_rows["test"]:
        value = dict(row)
        value.update({
            "forensics_domain": "fake", "dataset_split": "test", "class_label": 1,
            "content_category": None, "image_path": str(synth_root / row["image_relpath"]),
        })
        official_eval_rows.append(value)
    dump_jsonl(output / "split_audit/official1000_manifest/test_combined.jsonl", official_eval_rows)

    # Same-source annotation equality for all official-train images retained in frozen manifests.
    official_by_hash = {fingerprints["train"][r["sample_id"]]["sha256"]: r for r in official_rows["train"]}
    compared = exact_text = exact_refs = exact_union = 0
    area_diffs = []
    mismatch_examples = []
    for split, rows in frozen.items():
        for frozen_row in rows:
            official = official_by_hash.get(frozen_fp[frozen_row["sample_id"]]["sha256"])
            if not official:
                continue
            compared += 1
            text_equal = official["explanation"] == frozen_row["explanation"]
            refs_equal = official["refs"] == frozen_row["refs"]
            exact_text += text_equal; exact_refs += refs_equal
            height, width = frozen_row["image_variants"][0]["image_size"]
            official_mask = np.zeros((height, width), dtype=bool)
            frozen_mask = np.zeros((height, width), dtype=bool)
            for ref in official["refs"]:
                official_mask |= polygon_to_mask(polygons_for_target(ref, height, width), height, width).astype(bool)
            for ref in frozen_row["refs"]:
                frozen_mask |= polygon_to_mask(polygons_for_target(ref, height, width), height, width).astype(bool)
            union_equal = bool(np.array_equal(official_mask, frozen_mask))
            exact_union += union_equal
            diff = abs(float(official_mask.mean()) - float(frozen_mask.mean()))
            area_diffs.append(diff)
            if not (text_equal and refs_equal and union_equal) and len(mismatch_examples) < 20:
                mismatch_examples.append({"sample_id": frozen_row["sample_id"], "split": split,
                                          "text_equal": text_equal, "refs_equal": refs_equal,
                                          "union_equal": union_equal, "area_ratio_abs_diff": diff})
    annotation = {
        "compared_same_image_count": compared,
        "artifact_text_exact_count": exact_text,
        "refs_polygon_exact_count": exact_refs,
        "union_mask_pixel_exact_count": exact_union,
        "max_union_area_ratio_abs_diff": max(area_diffs, default=None),
        "mismatch_examples": mismatch_examples,
        "artifact_category_comparison": "unresolved: released train/test JSON refs contain no category field",
        "official_test_vs_frozen_same_image_count": sum(len(m["matches"]) > 0 for m in mappings),
    }
    dump(output / "split_audit/annotation_parity.json", annotation)
    return adapters, frozen, matrix


def supervision_audit(output: Path, adapters, frozen_train: list[dict]):
    stats = {}
    for split, adapter in adapters.items():
        counts = [len(row["refs"]) for row in adapter.samples]
        distribution = Counter("4+" if n >= 4 else str(n) for n in counts)
        stats[split] = {
            "images": len(counts), "annotation_rows": adapter.annotation_count,
            "artifact_instances": sum(counts), "distribution": dict(distribution),
            "mean": statistics.fmean(counts), "median": statistics.median(counts),
            "p90": float(np.percentile(counts, 90)), "max": max(counts),
        }

    class RowsOnly:
        rows = frozen_train

    exposure_counts = Counter()
    for epoch in range(10):
        world_size = 2 if epoch < 2 else 1
        for rank in range(world_size):
            sampler = BalancedDistributedForensicsSampler(RowsOnly(), num_replicas=world_size, rank=rank, seed=3407, shuffle=True)
            sampler.set_epoch(epoch)
            # Exactly 500 optimizer windows x (10/world_size) local samples.
            local_take = 10000 // world_size
            indices = list(iter(sampler))[:local_take]
            exposure_counts.update(frozen_train[i]["sample_id"] for i in indices if frozen_train[i]["class_label"] == 1)
    phase2a = {
        "total_sample_exposures_from_canonical_metrics": 100000,
        "total_fake_mask_exposures": sum(exposure_counts.values()),
        "unique_fake_masks_seen": len(exposure_counts),
        "frozen_fake_train_images": sum(r["class_label"] == 1 for r in frozen_train),
        "mean_exposures_per_frozen_fake_image": sum(exposure_counts.values()) / sum(r["class_label"] == 1 for r in frozen_train),
        "min_seen_exposures": min(exposure_counts.values()), "max_seen_exposures": max(exposure_counts.values()),
        "unseen_fake_images": sum(r["class_label"] == 1 for r in frozen_train) - len(exposure_counts),
        "reconstruction": "deterministic BalancedDistributedForensicsSampler, seed=3407; 500 steps/epoch; canonical world-size history 2 for epochs 1-2, then 1",
    }
    result = {
        "phase2a": phase2a,
        "legion_dataset_instances": stats,
        "legion_script_nominal": {
            "epochs": 3, "steps_per_epoch": 703, "batch_size_per_device": 16,
            "epoch_samples_argument": 11236,
            "effective_dataset_length": 11236,
            "important_code_behavior": "HybridSegDataset ignores its sampler index and calls child[0]; LegionGCGDataset then samples an annotation row uniformly with replacement. GCGBaseDataset also overwrites child epoch_samples with len(datas), while HybridSegDataset length remains the requested 11236.",
            "world_size_from_train_sh": "not fixed; DeepSpeed uses visible devices",
            "paper_actual": "8 A100 GPUs, batch size 2 per device",
            "current_code_executed_epochs": 1,
            "current_code_break": "train.py stops before epoch >= 1, so unmodified entry point executes 703 optimizer steps",
            "current_code_per_rank_draws": 703 * 16,
            "exact_global_exposure": "unresolved: public train.sh batch_size=16/device conflicts with paper batch_size=2/device; visible world size is unspecified; released code also stops after one epoch",
        },
    }
    dump(output / "source_audit/supervision_exposure.json", result)


def metric_and_gap_audit(output: Path, manifest_root: Path, predictions_root: Path):
    fake_rows = {
        r["sample_id"]: r for r in load_jsonl(manifest_root / "test_combined.jsonl") if r["source"] == "SynthScars"
    }
    modes = {}
    for mode in ("G0", "G1", "tf_full_context"):
        rows = load_jsonl(predictions_root / mode / "predictions.jsonl")
        modes[mode] = {r["sample_id"]: r for r in rows}
    ids = sorted(set(fake_rows) & set(modes["G0"]) & set(modes["G1"]) & set(modes["tf_full_context"]))
    if len(ids) != len(fake_rows):
        raise ValueError(f"Prediction alignment incomplete: {len(ids)}/{len(fake_rows)}")

    records = []
    for sample_id in ids:
        manifest = fake_rows[sample_id]
        image_path = Path(manifest["image_path"])
        with Image.open(image_path) as image:
            width, height = image.size
        pixels = width * height
        union = np.zeros((height, width), dtype=bool)
        artifact_types = set()
        for ref in manifest["refs"]:
            union |= polygon_to_mask(polygons_for_target(ref, height, width), height, width).astype(bool)
            for key in ("artifact_type", "category", "type"):
                if ref.get(key): artifact_types.add(str(ref[key]))
        components, labels = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
        component_areas = np.bincount(labels.ravel())[1:]
        component_count = int(components - 1)
        foreground = int(union.sum())
        largest_ratio = float(component_areas.max() / foreground) if foreground and len(component_areas) else 0.0
        count = len(manifest["refs"])
        area_ratio = foreground / pixels
        area_bin = bin_name(area_ratio, [(0, .01, "0-1%"), (.01, .02, "1-2%"), (.02, .05, "2-5%"),
                                               (.05, .1, "5-10%"), (.1, .2, "10-20%"), (.2, math.inf, "20%+")])
        count_bin = "4+" if count >= 4 else str(count)
        component_bin = "4+" if component_count >= 4 else str(component_count)
        metric_rows = {}
        for mode in ("G0", "G1", "tf_full_context"):
            prediction = modes[mode][sample_id]
            metric_rows["TF" if mode == "tf_full_context" else mode] = {
                "sample_id": sample_id, "tp": int(prediction["tp"]), "fp": int(prediction["fp"]),
                "fn": int(prediction["fn"]), "pixels": pixels,
            }
        g0, g1 = modes["G0"][sample_id], modes["G1"][sample_id]
        generated = g0.get("generated_explanation", "")
        gt = g0.get("gt_explanation", "")
        len_ratio = len(generated.split()) / max(1, len(gt.split()))
        edit = levenshtein_similarity(generated, gt)
        rec = {
            "sample_id": sample_id, "image_path": str(image_path), "content": manifest["content_category"],
            "gt_area_ratio": area_ratio, "area_bin": area_bin, "artifact_count": count,
            "artifact_count_bin": count_bin, "connected_components": component_count,
            "connected_component_bin": component_bin, "largest_component_over_artifact_area": largest_ratio,
            "artifact_types": sorted(artifact_types) if artifact_types else ["UNAVAILABLE_IN_RELEASED_JSON"],
            **metric_rows,
            "delta_TF_G0": safe_ratio(metric_rows["TF"]["tp"], metric_rows["TF"]["tp"] + metric_rows["TF"]["fp"] + metric_rows["TF"]["fn"]) - safe_ratio(metric_rows["G0"]["tp"], metric_rows["G0"]["tp"] + metric_rows["G0"]["fp"] + metric_rows["G0"]["fn"]),
            "delta_TF_G1": safe_ratio(metric_rows["TF"]["tp"], metric_rows["TF"]["tp"] + metric_rows["TF"]["fp"] + metric_rows["TF"]["fn"]) - safe_ratio(metric_rows["G1"]["tp"], metric_rows["G1"]["tp"] + metric_rows["G1"]["fp"] + metric_rows["G1"]["fn"]),
            "explanation_length_ratio": len_ratio, "token_edit_similarity": edit,
            "exact_prefix_overlap": float(g0.get("exact_prefix_overlap_over_gt", 0.0)),
            "repetition": bool(g0.get("repetition_flag", False)),
            "verdict_error": g0.get("lm_verdict_pred") != "fake",
            "seg_position": g0.get("seg_position"), "seg_triggered": bool(g0.get("seg_triggered")),
            "gt_explanation": gt, "generated_explanation": generated,
        }
        records.append(rec)

    parity = {
        "split": "Phase2A frozen internal fake test (n=1104)",
        "prediction_and_gt": "same persisted binary prediction confusion counts and same union GT; threshold unchanged",
        "overall": {mode: suite([r[mode] for r in records]) for mode in ("TF", "G0", "G1")},
        "by_content": {
            content: {mode: suite([r[mode] for r in records if r["content"] == content]) for mode in ("TF", "G0", "G1")}
            for content in ("human", "animal", "object", "scene")
        },
        "LEGION_exact_metric": None,
        "LEGION_exact_metric_status": "unresolved: public repository lacks Table 2 aggregation/F1 implementation",
        "union_image_localization_note": "These are Phase2A single-prediction versus OR(all released ref polygons), not LEGION phrase-pair validation.",
    }
    dump(output / "metric_parity/phase2a_best_internal.json", parity)
    dump_jsonl(output / "gap_analysis/per_sample.jsonl", records)
    dump(output / "gap_analysis/mask_area_bins.json", summarize_bins(records, "area_bin", ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20%+"]))
    dump(output / "gap_analysis/artifact_count_bins.json", summarize_bins(records, "artifact_count_bin", ["1", "2", "3", "4+"]))
    dump(output / "gap_analysis/connected_component_bins.json", summarize_bins(records, "connected_component_bin", ["1", "2", "3", "4+"]))
    dump(output / "gap_analysis/artifact_type_bins.json", {
        "status": "unresolved", "reason": "Released SynthScars train/test refs contain no Physics/Distortion/Structure field."
    })

    correlations = {}
    predictors = ["token_edit_similarity", "exact_prefix_overlap", "explanation_length_ratio", "gt_area_ratio",
                  "artifact_count", "connected_components", "largest_component_over_artifact_area"]
    for gap in ("delta_TF_G0", "delta_TF_G1"):
        correlations[gap] = {key: spearman([r[gap] for r in records], [r[key] for r in records]) for key in predictors}
    correlations["binary_groups"] = {
        "repetition": {
            str(value).lower(): {"n": len(group), "mean_TF_G0_gap": statistics.fmean(r["delta_TF_G0"] for r in group)}
            for value in (False, True) if (group := [r for r in records if r["repetition"] == value])
        },
        "verdict_error": {
            str(value).lower(): {"n": len(group), "mean_TF_G0_gap": statistics.fmean(r["delta_TF_G0"] for r in group)}
            for value in (False, True) if (group := [r for r in records if r["verdict_error"] == value])
        },
    }
    dump(output / "gap_analysis/explanation_correlations.json", correlations)
    dump(output / "gap_analysis/tf_g0_gap.json", {
        "n": len(records), "mean_TF_minus_G0": statistics.fmean(r["delta_TF_G0"] for r in records),
        "mean_TF_minus_G1": statistics.fmean(r["delta_TF_G1"] for r in records),
        "thresholds_diagnostic_only": {"TF_high": 0.5, "G0_high": 0.2},
    })

    quadrants = defaultdict(list)
    for row in records:
        tf_iou = safe_ratio(row["TF"]["tp"], row["TF"]["tp"] + row["TF"]["fp"] + row["TF"]["fn"])
        g0_iou = safe_ratio(row["G0"]["tp"], row["G0"]["tp"] + row["G0"]["fp"] + row["G0"]["fn"])
        label = ("A_TF_high_G0_high" if tf_iou >= .5 and g0_iou >= .2 else
                 "B_TF_high_G0_low" if tf_iou >= .5 else
                 "D_TF_low_G0_high" if g0_iou >= .2 else "C_TF_low_G0_low")
        quadrants[label].append({**row, "TF_IoU": tf_iou, "G0_IoU": g0_iou,
                                 "G1_IoU": safe_ratio(row["G1"]["tp"], row["G1"]["tp"] + row["G1"]["fp"] + row["G1"]["fn"])})
    summary = {}
    selected = []
    for label in ("A_TF_high_G0_high", "B_TF_high_G0_low", "C_TF_low_G0_low", "D_TF_low_G0_high"):
        group = quadrants[label]
        group.sort(key=lambda r: (-(r["TF_IoU"] - r["G0_IoU"]), r["sample_id"]))
        summary[label] = {"n": len(group), "fraction": len(group) / len(records), "selected": min(20, len(group))}
        for row in group[:20]:
            selected.append({"quadrant": label, **row})
    dump(output / "gap_analysis/quadrants.json", summary)
    dump_jsonl(output / "gap_analysis/selected_cases.jsonl", selected)
    return records


def write_metric_definition(output: Path):
    definition = {
        "fixed_threshold": "prediction mask logit > 0 (equivalent sigmoid > 0.5)",
        "phase2a": {
            "Mean IoU": "mean over images of foreground TP/(TP+FP+FN)",
            "Global IoU": "foreground TP/(TP+FP+FN) after pixel counts are summed over images",
            "Pixel F1": "foreground 2TP/(2TP+FP+FN), reported both per-image mean and global",
        },
        "LEGION_paper_metric": {
            "mIoU": "paper states mean IoU of foreground and background regions",
            "F1": "paper says overall F1; exact class/aggregation formula unresolved",
            "exact_table2_implementation_public": False,
            "required_statement": "public repository does not provide sufficient code to exactly reconstruct Table 2 aggregation",
        },
        "LEGION_public_repo_validation_metric": {
            "giou": "mean foreground IoU over phrase/mask pairs (implemented via per-image phrase mean weighted by number of GT masks)",
            "ciou": "global foreground IoU over all phrase/mask pairs",
            "class_index": 1,
            "foreground_background_mean": False,
            "F1": "not implemented in scripts/loc_exp/train.py validation",
            "silent_zip_truncation_risk": True,
        },
        "union_image_localization": "OR(all GT artifact instance masks) versus Phase2A's one predicted union mask; not automatically a LEGION Table 2 metric",
        "exact_parity_possible": False,
    }
    dump(output / "metric_parity/metric_definition.json", definition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase2b_legion_parity")
    parser.add_argument("--synthscars", type=Path, default=Path("/data/yz/myLISA_storage/AIGC/SynthScars"))
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs/data_audits/unified_forensics_split_v1")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/phase2a_unified_baseline/test")
    args = parser.parse_args()
    write_metric_definition(args.output)
    adapters, frozen, matrix = source_and_split_audit(args.output, args.synthscars, args.manifests)
    supervision_audit(args.output, adapters, load_jsonl(args.manifests / "train_combined.jsonl"))
    metric_and_gap_audit(args.output, args.manifests, args.predictions)
    print(json.dumps({"status": "complete", "output": str(args.output), "official_test_status": matrix["official_test_comparison_status"]}))


if __name__ == "__main__":
    main()
