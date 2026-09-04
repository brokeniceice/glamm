#!/usr/bin/env python3
"""Aggregate retrained LEGION official1000 localization and classification results."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.phase3c1 import paired_statistics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a4_legion_retrained_controlled"
MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
EXPECTED_MANIFEST_SHA = "fff3c3839e54d831955f8055501882f27a408c885e0524c2d7bf5d3edbe8c700"
EXPECTED_ORDER_SHA = "34b136b24365aedf4878a86b3f12856a974991d3615f1c603213792c99a2e114"
EXPECTED_LE_SHA = "6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844"
EXPECTED_CLS_SHA = "f33fda9ddf0e22bcd9bca8dcc998d8a9c0c6421f6bd6a46573044c6cc7365739"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def summarize(records: list[dict]) -> dict:
    iou = [float(row["foreground_iou"]) for row in records]
    f1 = [float(row["foreground_f1"]) for row in records]
    tp, fp, fn = (sum(int(row[key]) for row in records) for key in ("tp", "fp", "fn"))
    values = sorted(iou); middle = len(values) // 2
    median = (values[middle - 1] + values[middle]) / 2 if len(values) % 2 == 0 else values[middle]
    return {"n": len(records), "mean_foreground_iou": sum(iou) / len(iou), "median_foreground_iou": median,
            "mean_foreground_f1": sum(f1) / len(f1), "global_foreground_iou": tp / max(1, tp + fp + fn),
            "global_foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn), "threshold_logit": 0.0}


def paired(left: list[dict], right: list[dict]) -> dict:
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise RuntimeError("paired sample order mismatch")
    return {key: paired_statistics([row[key] for row in left], [row[key] for row in right], seed=3407)
            for key in ("foreground_iou", "foreground_f1")}


def main() -> None:
    manifest = rows(MANIFEST); ids = [row["sample_id"] for row in manifest]
    if len(ids) != 1000 or len(set(ids)) != 1000 or sha256(MANIFEST) != EXPECTED_MANIFEST_SHA or ordered_sha(ids) != EXPECTED_ORDER_SHA:
        raise RuntimeError("official1000 frozen manifest drift")
    le_identity = json.loads((ROOT / "checkpoints/phase5a3_legion_retrained/stage1_le_identity.json").read_text())
    cls_identity = json.loads((ROOT / "checkpoints/phase5a3_legion_retrained/stage2_cls_identity.json").read_text())
    if le_identity.get("canonical_sha256") != EXPECTED_LE_SHA or cls_identity.get("canonical_sha256") != EXPECTED_CLS_SHA:
        raise RuntimeError("LEGION-retrained checkpoint drift")
    shard = OUT / "official1000/original/shards"
    prediction_paths = (shard / "0000_0500.predictions.jsonl", shard / "0500_1000.predictions.jsonl")
    worker_paths = (shard / "0000_0500.worker.json", shard / "0500_1000.worker.json")
    workers = [json.loads(path.read_text()) for path in worker_paths]
    if any(worker.get("status") != "COMPLETE" or worker.get("target_kind") != "synthscars_union" for worker in workers):
        raise RuntimeError("official1000 localization worker incomplete")
    retrained = [row for path in prediction_paths for row in rows(path)]
    if [row["sample_id"] for row in retrained] != ids or [row["ordinal"] for row in retrained] != list(range(1000)):
        raise RuntimeError("official1000 retrained sample order drift")
    for row in retrained:
        with Image.open(row["mask_path"]) as image:
            if image.size != (row["original_hw"][1], row["original_hw"][0]):
                raise RuntimeError(f"prediction geometry drift: {row['sample_id']}")
    r1_job_path = ROOT / "outputs/p1_r1_reusable_matrix/jobs/official_g0.json"
    r1_job = json.loads(r1_job_path.read_text()); r1 = r1_job["R1"]["records"]
    public_root = ROOT / "outputs/phase5a2_legion_public_reference/original/shards"
    public = [row for path in (public_root / "0000_0500.predictions.jsonl", public_root / "0500_1000.predictions.jsonl") for row in rows(path)]
    if [row["sample_id"] for row in r1] != ids or [row["sample_id"] for row in public] != ids:
        raise RuntimeError("official1000 comparison sample order drift")
    valid = lambda values: sum(row["status"] == "OK" and row["seg_count"] > 0 and row["has_pred_mask"] for row in values)
    classification = {}
    for name in ("internal", "aigi_test"):
        path = OUT / "classification" / name / "results.json"
        value = json.loads(path.read_text())
        if value.get("status") != "COMPLETE": raise RuntimeError(f"classification incomplete: {name}")
        classification[name] = value
    result = {
        "schema": "phase5a4_legion_retrained_controlled_results_v1", "status": "COMPLETE",
        "source_commit": "d21535dd45f6fea509337a83095966f0b86ac924",
        "stage1_le_canonical_sha256": EXPECTED_LE_SHA, "stage2_cls_canonical_sha256": EXPECTED_CLS_SHA,
        "official1000_localization": {
            "n": 1000, "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
            "ordered_sample_id_sha256": ordered_sha(ids),
            "policy": "full-N; no [SEG]/empty mask is all-zero; original resolution; logit > 0; multiple masks union",
            "r1_g0": {"primary": summarize(r1), "job": str(r1_job_path.resolve())},
            "legion_public_intermediate_lfree": {"primary": summarize(public), "valid_seg_and_mask": valid(public)},
            "legion_retrained_lfree": {"primary": summarize(retrained), "valid_seg_and_mask": valid(retrained),
                                                "status_counts": dict(sorted(Counter(row["status"] for row in retrained).items())), "workers": workers},
            "r1_minus_legion_retrained": paired(r1, retrained),
            "legion_retrained_minus_public_intermediate": paired(retrained, public),
        },
        "classification": classification,
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT / "results.json")


if __name__ == "__main__": main()
