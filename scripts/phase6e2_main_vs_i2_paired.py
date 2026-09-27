#!/usr/bin/env python3
"""Audit and compare the frozen Phase6E.2 main and I2 Official1000 rows."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4e1 import compare, summarize

BASE = ROOT / "outputs/phase6e2_c1_specific_r1"
E1 = ROOT / "outputs/phase6e1_c1_old_r1_transfer"
MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
OUT = BASE / "main_vs_i2_paired.json"
SEED = 3407


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def global_iou_ci(main: list[dict], i2: list[dict], repeats: int = 10000) -> list[float]:
    """Resample paired images, then aggregate TP/FP/FN within each resample."""
    a = np.asarray([[row[k] for k in ("tp", "fp", "fn")] for row in main], dtype=np.int64)
    b = np.asarray([[row[k] for k in ("tp", "fp", "fn")] for row in i2], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    deltas = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        end = min(start + 500, repeats)
        idx = rng.integers(0, len(main), size=(end - start, len(main)))
        av, bv = a[idx].sum(axis=1), b[idx].sum(axis=1)
        deltas[start:end] = av[:, 0] / av.sum(axis=1) - bv[:, 0] / bv.sum(axis=1)
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])]


def main() -> None:
    paths = {"main": BASE, "i2": BASE / "i2"}
    manifest = rows(MANIFEST)
    c1 = rows(E1 / "c1_records.jsonl")
    expected_ids = [row["sample_id"] for row in manifest]
    require(sha(MANIFEST) == "fff3c3839e54d831955f8055501882f27a408c885e0524c2d7bf5d3edbe8c700", "manifest SHA drift")
    require(len(expected_ids) == len(set(expected_ids)) == 1000, "manifest count/uniqueness drift")
    require(all(row["source"] == "SynthScars" and int(row["class_label"]) == 1 for row in manifest), "manifest population drift")
    require([row["sample_id"] for row in c1] == expected_ids, "C1 record/manifest order drift")

    data = {}
    for arm, folder in paths.items():
        rec = folder / "official1000_new_r1_records.jsonl"
        selected = folder / "selected_checkpoint.pt"
        result = folder / ("phase6e2_c1_specific_r1.json" if arm == "main" else "phase6e2_i2_random_init.json")
        rr, summary = rows(rec), json.loads(result.read_text())
        selector = json.loads((folder / "selector.json").read_text())
        init = json.loads((folder / "initialization_provenance.json").read_text())
        require([row["sample_id"] for row in rr] == expected_ids, f"{arm} record/manifest order drift")
        require(summary["provenance"]["official_records_sha256"] == sha(rec), f"{arm} record SHA drift")
        require(summary["provenance"]["new_r1_sha256"] == selector["selected_checkpoint_sha256"] == sha(selected), f"{arm} checkpoint SHA drift")
        require(summary["provenance"]["c1_sha256"] == init["c1_sha256"] == "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff", f"{arm} C1 drift")
        require(not selector["official1000_used"] and not selector["internal_test_used"], f"{arm} selector firewall drift")
        require(init["frozen_hash_before"] == init["frozen_hash_after"], f"{arm} frozen module drift")
        for row in rr:
            tp, fp, fn = (row[k] for k in ("tp", "fp", "fn"))
            require(all(isinstance(v, int) and v >= 0 for v in (tp, fp, fn)), f"{arm} invalid count")
            require(all(math.isfinite(row[k]) for k in ("foreground_iou", "foreground_f1")), f"{arm} nonfinite metric")
            require(abs(row["foreground_iou"] - tp / max(1, tp + fp + fn)) < 1e-10, f"{arm} IoU/count mismatch")
            require(abs(row["foreground_f1"] - 2 * tp / max(1, 2 * tp + fp + fn)) < 1e-10, f"{arm} F1/count mismatch")
        metrics = summarize(rr)
        name = "C1+new_R1" if arm == "main" else "C1+I2_random_R1"
        require(all(abs(metrics[k] - summary["arms"][name][k]) < 1e-12 for k in metrics if isinstance(metrics[k], float)), f"{arm} summary mismatch")
        data[arm] = {"records": rr, "selector": selector, "init": init,
                     "metrics": metrics, "records_sha256": sha(rec), "checkpoint_sha256": sha(selected),
                     "selected_epoch": selector["selected_epoch"], "selected_dev_g0": selector["selected_dev_g0"]}

    a, b = data["main"]["records"], data["i2"]["records"]
    require(all(x["tp"] + x["fn"] == y["tp"] + y["fn"] for x, y in zip(a, b)), "per-image GT size mismatch")
    require(all(x["valid_q_seg"] == y["valid_q_seg"] for x, y in zip(a, b)), "SEG validity mismatch")
    require(sum(bool(row["valid_q_seg"]) for row in a) == 992, "SEG count drift")
    ia, ib = data["main"]["init"], data["i2"]["init"]
    for key in ("recipe", "actual_population", "trainable", "frozen", "frozen_hash_before", "frozen_hash_after", "c1_sha256"):
        require(ia[key] == ib[key], f"training protocol mismatch: {key}")
    sa, sb = data["main"]["selector"], data["i2"]["selector"]
    require(len(sa["candidates"]) == len(sb["candidates"]) == 10, "training epoch count mismatch")
    for x, y in zip(sa["candidates"], sb["candidates"]):
        for key in ("sample_order_sha256", "traversal_exposures", "optimization_eligible_exposures", "invalid_g0_exposures", "optimizer_updates"):
            require(x[key] == y[key], f"training traversal mismatch: {key}")

    conflict = BASE / "i2/official1000_concurrent_conflict_1581rows.jsonl"
    conflict_rows = rows(conflict)
    require(len(conflict_rows) == 1581 and sha(conflict) != data["i2"]["records_sha256"], "I2 conflict artifact drift")
    paired = compare(a, b, seed=SEED)
    global_ci = global_iou_ci(a, b)
    result = {
        "schema": "phase6e2_main_vs_i2_paired_v1", "status": "PASS", "population": "SynthScars Official1000 Fake-only original RGB canonical G0",
        "direction": "main minus I2", "manifest_sha256": sha(MANIFEST), "n": 1000, "matched_order": True,
        "matched_gt_pixels_per_image": True, "matched_seg_validity": True, "valid_seg": 992, "invalid_seg": 8,
        "training_protocol_matched": True, "training_difference": "Utility/Rectifier initialization only",
        "i2_conflict_artifact": {"path": str(conflict), "rows": len(conflict_rows), "sha256": sha(conflict), "used": False},
        "arms": {name: {k: v for k, v in value.items() if k not in ("records", "selector", "init")} for name, value in data.items()},
        "paired_statistics": paired, "global_foreground_iou_difference": data["main"]["metrics"]["global_foreground_iou"] - data["i2"]["metrics"]["global_foreground_iou"],
        "global_foreground_iou_paired_image_bootstrap_95_ci": global_ci,
        "global_foreground_f1_difference": data["main"]["metrics"]["global_foreground_f1"] - data["i2"]["metrics"]["global_foreground_f1"],
        "bootstrap": {"seed": SEED, "repeats": 10000, "resampling_unit": "paired image", "ci": "percentile 2.5/97.5"},
    }
    temp = OUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT)
    print(json.dumps({"status": "PASS", "paired": paired, "global_iou_ci": global_ci}, ensure_ascii=False))


if __name__ == "__main__":
    main()
