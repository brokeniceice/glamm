#!/usr/bin/env python3
"""Frozen C1-center versus D1-alpha on DFJ-Detect and FakeBench."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6d5_full_ood as replay

OUT = ROOT / "outputs/phase6d6_extra_ood_dfj_fakebench"
DOC = ROOT / "docs/phase6d6_extra_ood_dfj_fakebench.md"
DFJ = Path("/data/yz/DeepfakeJudge/dfj-bench/dfj-detect")
FAKEBENCH = Path("/data/yz/Fakebench")
FUSION = ROOT / "outputs/phase6d5_decision_integration/fusion_parameters.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    os.replace(tmp, path)


def prepare_manifests():
    manifests = OUT / "manifests"
    duplicate_rows_removed = []
    dfj_rows = []
    with (DFJ / "data.jsonl").open() as f:
        for i, line in enumerate(f):
            source = json.loads(line)
            if len(source["images"]) != 1:
                raise RuntimeError(f"DFJ row {i} does not contain exactly one image")
            image = (DFJ / source["images"][0]).resolve()
            dfj_rows.append({
                "sample_id": f"dfj-detect:{source['images'][0]}",
                "image_path": str(image),
                "class_label": 1 if source["answer"].lower() == "fake" else 0,
                "generator/source": source["source"],
            })

    fakebench_rows = []
    for label_name, label_value in (("real", 0), ("fake", 1)):
        csv_path = FAKEBENCH / f"{label_name}_images.csv"
        seen_names = set()
        with csv_path.open(newline="") as f:
            for csv_row_index, row in enumerate(csv.reader(f)):
                if not row or not row[0].strip():
                    continue
                name = row[0].strip()
                if name in seen_names:
                    duplicate_rows_removed.append({
                        "dataset": "fakebench", "label": label_name,
                        "filename": name, "csv_row_index_zero_based": csv_row_index,
                        "reason": "duplicate CSV row for the same physical image",
                    })
                    continue
                seen_names.add(name)
                image = (FAKEBENCH / f"{label_name}_images" / name).resolve()
                generator = name.split("-", 1)[0] if label_value else "real_images"
                fakebench_rows.append({
                    "sample_id": f"fakebench:{label_name}:{name}",
                    "image_path": str(image),
                    "class_label": label_value,
                    "generator/source": generator,
                })

    result = {}
    for name, rows in (("dfj_detect", dfj_rows), ("fakebench", fakebench_rows)):
        ids = [x["sample_id"] for x in rows]
        missing = [x["image_path"] for x in rows if not Path(x["image_path"]).is_file()]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate sample_id in {name}")
        if missing:
            raise RuntimeError(f"{name} has {len(missing)} missing images; first={missing[0]}")
        path = manifests / f"{name}.jsonl"
        write_jsonl(path, rows)
        result[name] = path
    dump(OUT / "manifest_audit.json", {
        "status": "PASS", "created_at_utc": now(),
        "duplicate_rows_removed": duplicate_rows_removed,
        "unlisted_files_policy": "ignored; official CSV/JSONL is the population contract",
        "datasets": {name: {
            "path": str(path.resolve()), "sha256": sha256(path),
            "count": len(replay.read_jsonl(path)),
            "real": sum(replay.label(x) == 0 for x in replay.read_jsonl(path)),
            "fake": sum(replay.label(x) == 1 for x in replay.read_jsonl(path)),
        } for name, path in result.items()},
    })
    return result


def finalize(manifests):
    params = json.loads(FUSION.read_text())
    norm = params["normalization"]
    alpha = float(params["selected_alpha"])
    if alpha != 0.3:
        raise RuntimeError(f"expected frozen alpha=0.3, got {alpha}")
    result = {
        "schema": "phase6d6_extra_ood_dfj_fakebench_v1", "status": "COMPLETE",
        "generated_at_utc": now(),
        "protocol": {
            "checkpoint": str(replay.CKPT.resolve()),
            "checkpoint_sha256": sha256(replay.CKPT),
            "normalization": norm, "normalization_source": str(FUSION.resolve()),
            "normalization_source_sha256": sha256(FUSION),
            "alpha": alpha, "threshold": 0.0,
            "C1-center": "sC=(C1_margin-mu_C_train)/sigma_C_train; sC>0",
            "D1-alpha": "0.3*sR+0.7*sC; score>0",
            "ood_selection_or_tuning": False,
        },
        "datasets": {},
    }
    for name, manifest in manifests.items():
        source = replay.read_jsonl(manifest)
        records = replay.read_jsonl(OUT / "raw" / f"{name}.jsonl")
        if [x["sample_id"] for x in records] != [x["sample_id"] for x in source]:
            raise RuntimeError(f"identity/order mismatch: {name}")
        for row in records:
            sc = (row["c1_margin"] - norm["c1"]["mean"]) / norm["c1"]["std"]
            sr = (row["rine_margin"] - norm["rine"]["mean"]) / norm["rine"]["std"]
            row["c1_center_margin"] = sc
            row["d1_alpha_margin"] = alpha * sr + (1.0 - alpha) * sc
        pred = OUT / "predictions" / f"{name}.jsonl"
        write_jsonl(pred, records)
        arms = {
            "C1-center": replay.metric(records, "c1_center_margin"),
            "D1-alpha": replay.metric(records, "d1_alpha_margin"),
        }
        by_source = {}
        for source_name in sorted({str(x.get("generator")) for x in records}):
            part = [x for x in records if str(x.get("generator")) == source_name]
            by_source[source_name] = {
                "C1-center": replay.metric(part, "c1_center_margin"),
                "D1-alpha": replay.metric(part, "d1_alpha_margin"),
            }
        result["datasets"][name] = {
            "arms": arms, "per_source": by_source,
            "hard_metric_delta_C1_center_minus_D1_alpha": {
                key: arms["C1-center"][key] - arms["D1-alpha"][key]
                for key in ("accuracy", "fake_recall", "tnr", "fpr", "f1")
            },
            "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
            "predictions": str(pred.resolve()), "predictions_sha256": sha256(pred),
        }
    dump(OUT / "results.json", result)
    lines = [
        "# Extra OOD diagnostic: DFJ-Detect and FakeBench", "",
        "Frozen C1/RINE replay only. Internal-TRAIN normalization and alpha=0.3 are unchanged; no OOD selection or tuning was performed.", "",
        "| Dataset | Arm | N | Accuracy | ROC-AUC | Fake Recall | TNR | FPR | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["datasets"].items():
        for arm, m in entry["arms"].items():
            auc = "N/A" if m["roc_auc"] is None else f"{m['roc_auc']:.6f}"
            lines.append(f"| {name} | {arm} | {m['n']} | {m['accuracy']:.6f} | {auc} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} |")
    DOC.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = prepare_manifests()
    replay.OUT = OUT
    replay.MANIFESTS = manifests
    dump(OUT / "worker_status.json", {
        "status": "RUNNING", "pid": os.getpid(), "device": args.device,
        "batch_size": args.batch_size, "started_at_utc": now(),
    })
    try:
        replay.extract(args.device, args.batch_size)
        finalize(manifests)
        dump(OUT / "worker_status.json", {
            "status": "COMPLETE", "pid": os.getpid(), "device": args.device,
            "batch_size": args.batch_size, "updated_at_utc": now(),
        })
    except BaseException as error:
        dump(OUT / "worker_status.json", {
            "status": "FAILED", "pid": os.getpid(), "device": args.device,
            "exception_type": type(error).__name__, "exception": str(error),
            "updated_at_utc": now(),
        })
        raise


if __name__ == "__main__":
    main()
