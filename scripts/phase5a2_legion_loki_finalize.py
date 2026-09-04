#!/usr/bin/env python3
"""Validate and aggregate the authorized LOKI cross-protocol supplement."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tools.phase3c1 import paired_statistics


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/LOKI/legion_localization/manifest.jsonl"
OUT = ROOT / "outputs/phase5a2_legion_loki_reference"
EXPECTED_MANIFEST_SHA = "c9a29854717867419c2386be75bf3cf4c0366f43230e62b653594ca908628bc8"
EXPECTED_ORDERED_SHA = "0b9b182f3b1b2d9b747858250360360018d6f56e4176e39bf20488837e609503"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def summarize(records: list[dict]) -> dict:
    iou = np.asarray([row["foreground_iou"] for row in records], dtype=np.float64)
    f1 = np.asarray([row["foreground_f1"] for row in records], dtype=np.float64)
    tp, fp, fn = (sum(int(row[key]) for row in records) for key in ("tp", "fp", "fn"))
    return {
        "n": len(records), "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)), "mean_foreground_f1": float(f1.mean()),
        "global_foreground_iou": tp / max(1, tp + fp + fn),
        "global_foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn), "threshold_logit": 0.0,
    }


def paired(left: list[dict], right: list[dict]) -> dict:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise RuntimeError("LOKI paired identity/order mismatch")
    return {key: paired_statistics([row[key] for row in left], [row[key] for row in right], seed=3407)
            for key in ("foreground_iou", "foreground_f1")}


def main() -> None:
    manifest = rows(MANIFEST); ids = [str(row["sample_id"]) for row in manifest]
    if len(ids) != 229 or len(set(ids)) != 229 or sha256(MANIFEST) != EXPECTED_MANIFEST_SHA or ordered_sha(ids) != EXPECTED_ORDERED_SHA:
        raise RuntimeError("LOKI frozen manifest drift")
    shard_root = OUT / "original/shards"
    prediction_paths = (shard_root / "0000_0115.predictions.jsonl", shard_root / "0115_0229.predictions.jsonl")
    worker_paths = (shard_root / "0000_0115.worker.json", shard_root / "0115_0229.worker.json")
    workers = [json.loads(path.read_text()) for path in worker_paths]
    if any(worker.get("status") != "COMPLETE" or worker.get("target_kind") != "loki_bbox_union" for worker in workers):
        raise RuntimeError("LOKI LEGION worker incomplete or target drift")
    legion = [row for path in prediction_paths for row in rows(path)]
    if [row["sample_id"] for row in legion] != ids or [row["ordinal"] for row in legion] != list(range(229)):
        raise RuntimeError("LOKI LEGION sample order drift")
    for row, source in zip(legion, manifest):
        if row.get("target_kind") != "loki_bbox_union":
            raise RuntimeError("LOKI target-kind drift")
        with Image.open(row["mask_path"]) as image:
            if image.size != (source["image_size_hw"][1], source["image_size_hw"][0]):
                raise RuntimeError(f"LOKI prediction geometry drift: {row['sample_id']}")
        with Image.open(source["mask_path"]) as image:
            target = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if int(target.sum()) != int(source["mask_foreground_pixels"]):
            raise RuntimeError(f"LOKI GT mask drift: {row['sample_id']}")
    job_path = ROOT / "outputs/p1_r1_reusable_matrix/jobs/loki_g1.json"
    job = json.loads(job_path.read_text()); r1 = job["R1"]["records"]
    if job.get("mode") != "g1" or [row["sample_id"] for row in r1] != ids:
        raise RuntimeError("R1 LOKI G1 artifact drift")
    r1_valid = [bool(row.get("valid_q_seg")) for row in r1]
    legion_valid = [bool(row["status"] == "OK" and row["seg_count"] > 0 and row["has_pred_mask"]) for row in legion]
    common = [index for index, flags in enumerate(zip(r1_valid, legion_valid)) if all(flags)]
    r1_common, legion_common = [r1[i] for i in common], [legion[i] for i in common]
    result = {
        "schema": "phase5a2_legion_loki_cross_protocol_results_v1", "status": "COMPLETE",
        "comparison_status": "CROSS_PROTOCOL_DIAGNOSTIC_R1_G1_VS_LEGION_LFREE",
        "strict_condition_parity": False,
        "warning": "R1 uses known-Fake G1; LEGION uses official image-only L-FREE. Do not treat as a strict main baseline.",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "ordered_sample_id_sha256": ordered_sha(ids), "n": 229,
        "gt": "union of filled LOKI regional xywh bounding boxes at original resolution",
        "r1_g1": {"primary": summarize(r1), "valid_seg_and_mask": sum(r1_valid), "job": str(job_path.resolve()), "job_sha256": sha256(job_path)},
        "legion_lfree": {"primary": summarize(legion), "valid_seg_and_mask": sum(legion_valid),
                            "status_counts": dict(sorted(Counter(row["status"] for row in legion).items())), "workers": workers},
        "descriptive_r1_minus_legion": paired(r1, legion),
        "both_valid_seg_and_mask_diagnostic": {
            "n": len(common), "r1_g1": summarize(r1_common), "legion_lfree": summarize(legion_common),
            "descriptive_r1_minus_legion": paired(r1_common, legion_common),
        },
    }
    path = OUT / "shared_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
