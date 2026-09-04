#!/usr/bin/env python3
"""Validate and aggregate the frozen R1 G0 versus official LEGION L-FREE runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.phase3c1 import paired_statistics


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("original", "jpeg70", "jpeg80", "gaussian5", "gaussian10")
MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
OUT = ROOT / "outputs/phase5a2_legion_public_reference"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def summarize(records: list[dict]) -> dict:
    tp, fp, fn = (sum(int(row[key]) for row in records) for key in ("tp", "fp", "fn"))
    iou = [float(row["foreground_iou"]) for row in records]
    f1 = [float(row["foreground_f1"]) for row in records]
    iou_sorted = sorted(iou)
    middle = len(iou_sorted) // 2
    median = (iou_sorted[middle - 1] + iou_sorted[middle]) / 2 if len(iou_sorted) % 2 == 0 else iou_sorted[middle]
    return {
        "n": len(records),
        "mean_foreground_iou": sum(iou) / len(iou),
        "median_foreground_iou": median,
        "mean_foreground_f1": sum(f1) / len(f1),
        "global_foreground_iou": tp / max(1, tp + fp + fn),
        "global_foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "threshold_logit": 0.0,
    }


def paired(left: list[dict], right: list[dict]) -> dict:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise RuntimeError("paired identity/order mismatch")
    return {
        key: paired_statistics([row[key] for row in left], [row[key] for row in right], seed=3407)
        for key in ("foreground_iou", "foreground_f1")
    }


def legion_records(condition: str, expected_ids: list[str]) -> tuple[list[dict], dict]:
    directory = OUT / condition / "shards"
    expected_files = (directory / "0000_0500.predictions.jsonl", directory / "0500_1000.predictions.jsonl")
    worker_files = (directory / "0000_0500.worker.json", directory / "0500_1000.worker.json")
    for path in (*expected_files, *worker_files):
        if not path.is_file():
            raise RuntimeError(f"missing required final shard file: {path}")
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_files]
    if any(worker.get("status") != "COMPLETE" for worker in workers):
        raise RuntimeError("LEGION shard did not complete")
    records = [row for path in expected_files for row in load_jsonl(path)]
    if [int(row["ordinal"]) for row in records] != list(range(1000)):
        raise RuntimeError(f"LEGION ordinal mismatch: {condition}")
    if [str(row["sample_id"]) for row in records] != expected_ids:
        raise RuntimeError(f"LEGION manifest mismatch: {condition}")
    if any(row.get("condition") != condition for row in records):
        raise RuntimeError(f"LEGION condition-label mismatch: {condition}")
    for row in records:
        mask_path = Path(row["mask_path"])
        if not mask_path.is_file():
            raise RuntimeError(f"missing persisted LEGION mask: {mask_path}")
        with Image.open(mask_path) as image:
            if image.size != (int(row["original_hw"][1]), int(row["original_hw"][0])):
                raise RuntimeError(f"LEGION mask geometry mismatch: {row['sample_id']}")
    coverage = {
        "n": len(records),
        "valid_seg_and_mask": sum(row["status"] == "OK" and row["seg_count"] > 0 and row["has_pred_mask"] for row in records),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in records).items())),
        "seg_count_total": sum(int(row["seg_count"]) for row in records),
    }
    return records, {"workers": workers, "coverage": coverage}


def r1_records(condition: str, expected_ids: list[str]) -> tuple[list[dict], dict]:
    job_name = "official_g0" if condition == "original" else f"official_{condition}"
    job_path = ROOT / "outputs/p1_r1_reusable_matrix/jobs" / f"{job_name}.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    records = job["R1"]["records"]
    if [str(row["sample_id"]) for row in records] != expected_ids:
        raise RuntimeError(f"R1 matrix manifest mismatch: {condition}")
    coverage_path = ROOT / "outputs/phase3c2_p3_robustness/robustness" / condition / "p1/G0.jsonl"
    coverage_rows = load_jsonl(coverage_path)
    if [str(row["sample_id"]) for row in coverage_rows] != expected_ids:
        raise RuntimeError(f"R1 coverage manifest mismatch: {condition}")
    valid = [bool(row["contains_seg_token"] and row["has_pred_mask"]) for row in coverage_rows]
    coverage = {"n": len(valid), "valid_seg_and_mask": sum(valid), "no_seg_or_mask": len(valid) - sum(valid)}
    return records, {"job_path": str(job_path.resolve()), "job_sha256": sha256(job_path), "coverage": coverage, "valid": valid}


def main() -> None:
    manifest_rows = load_jsonl(MANIFEST)
    ids = [str(row["sample_id"]) for row in manifest_rows]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise RuntimeError("shared manifest population drift")
    summary = {
        "schema": "phase5a2_r1_legion_shared_localization_results_v1",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "ordered_sample_id_sha256": ordered_id_sha(ids), "n": len(ids),
        "primary_policy": "all samples; no [SEG] or empty prediction is persisted as an all-zero prediction",
        "conditional_policy": "both_valid_seg_and_mask only; diagnostic and not the primary result",
        "conditions": {},
    }
    all_rows = {}
    for condition in CONDITIONS:
        legion, legion_meta = legion_records(condition, ids)
        r1, r1_meta = r1_records(condition, ids)
        r1_valid, legion_valid = r1_meta.pop("valid"), [row["status"] == "OK" and row["seg_count"] > 0 and row["has_pred_mask"] for row in legion]
        common = [index for index, (a, b) in enumerate(zip(r1_valid, legion_valid)) if a and b]
        if not common:
            raise RuntimeError(f"empty both-valid diagnostic subset: {condition}")
        r1_common, legion_common = ([r1[index] for index in common], [legion[index] for index in common])
        summary["conditions"][condition] = {
            "r1_g0": {"primary": summarize(r1), "coverage": r1_meta["coverage"], "artifact": r1_meta},
            "legion_lfree": {"primary": summarize(legion), **legion_meta},
            "primary_r1_minus_legion": paired(r1, legion),
            "both_valid_seg_and_mask_diagnostic": {
                "n": len(common), "r1": summarize(r1_common), "legion": summarize(legion_common),
                "r1_minus_legion": paired(r1_common, legion_common),
            },
        }
        all_rows[condition] = {"r1": r1, "legion": legion}
    original = all_rows["original"]
    for condition in CONDITIONS[1:]:
        summary["conditions"][condition]["robustness_degradation_original_minus_condition"] = {
            "r1_g0": paired(original["r1"], all_rows[condition]["r1"]),
            "legion_lfree": paired(original["legion"], all_rows[condition]["legion"]),
        }
    output = OUT / "shared_results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
