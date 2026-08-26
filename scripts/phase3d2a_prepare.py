#!/usr/bin/env python3
"""Freeze Phase 3D.2-A checkpoints and populations without model execution."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256

OUT = ROOT / "outputs/phase3d2a_spatial_attribution_audit"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"
SEED = 3407


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def identity(row: dict) -> str:
    value = row.get("image_identity") or row.get("split_group_id") or row.get("image_path") or row["sample_id"]
    return str(value)


def annotation_ids(row: dict) -> list:
    return list(row.get("annotation_ids") or [ref.get("annotation_id") for ref in row.get("refs") or [] if ref.get("annotation_id")])


def manifest_checksum(values: list[dict]) -> str:
    payload = "\n".join(f"{row['sample_id']}\t{identity(row)}" for row in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_root = ROOT / cfg["data"]["manifest_dir"]
    train_path, val_path = data_root / "train_combined.jsonl", data_root / "val_combined.jsonl"
    train, val = rows(train_path), rows(val_path)
    by_id = {row["sample_id"]: row for row in train}
    metric_rows = rows(P3D2 / "training/metrics.jsonl")
    exposure_ids = [sample_id for record in metric_rows for sample_id in record["sample_ids"]]
    frozen_schedule = load(P3D2 / "training/schedule.json")["sample_ids"]
    if exposure_ids != frozen_schedule or len(exposure_ids) != 1000:
        raise RuntimeError("actual Phase 3D.2 metric log does not match its frozen 1000-exposure schedule")
    if any(sample_id not in by_id for sample_id in exposure_ids):
        raise RuntimeError("scheduled sample absent from frozen training manifest")
    seen_unique_ids = list(dict.fromkeys(exposure_ids))
    seen = [by_id[sample_id] for sample_id in seen_unique_ids]
    if any(int(row["class_label"]) != 1 for row in seen):
        raise RuntimeError("non-Fake exposure in Phase 3D.2 schedule")
    seen_identities = {identity(row) for row in seen}
    candidates = [
        row for row in train if int(row["class_label"]) == 1
        and row["sample_id"] not in set(seen_unique_ids) and identity(row) not in seen_identities
    ]
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    holdout = candidates[: min(1000, len(candidates))]
    validation = [row for row in val if int(row["class_label"]) == 1]
    if len(validation) != 1106:
        raise RuntimeError(f"validation Fake population drifted: {len(validation)}")
    if {row["sample_id"] for row in seen} & {row["sample_id"] for row in holdout}:
        raise RuntimeError("SEEN-TRAIN / TRAIN-HOLDOUT sample intersection")
    if {identity(row) for row in seen} & {identity(row) for row in holdout}:
        raise RuntimeError("SEEN-TRAIN / TRAIN-HOLDOUT canonical identity leakage")

    checkpoint_specs = [
        (0, Path(cfg["source"]["checkpoint"]), "P1 Phrase-Aligned Forensic SFT", True, False),
        (50, Path(cfg["experiment"]["checkpoint_root"]) / "step_0050/checkpoint/mp_rank_00_model_states.pt", "Phase 3D.2 Direct Spatial-Path Optimization", False, True),
        (100, Path(cfg["experiment"]["checkpoint_root"]) / "step_0100/checkpoint/mp_rank_00_model_states.pt", "Phase 3D.2 Direct Spatial-Path Optimization", False, True),
        (250, Path(cfg["experiment"]["checkpoint_root"]) / "step_0250/checkpoint/mp_rank_00_model_states.pt", "Phase 3D.2 Direct Spatial-Path Optimization", False, True),
    ]
    checkpoint_records = []
    for step, path, source, selected, diagnostic in checkpoint_specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_records.append({
            "step": step, "checkpoint": artifact(path), "source_experiment": source,
            "formal_selector_status": "SELECTED" if selected else "NOT_SELECTED",
            "diagnostic_only": diagnostic,
        })
    if checkpoint_records[0]["checkpoint"]["sha256"] != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checkpoint hash mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = {
        "status": "FROZEN_READ_ONLY", "phase": "Phase 3D.2-A", "checkpoints": checkpoint_records,
        "formal_phase3d2_selected_step": 0, "reselection_allowed": False,
    }
    dump(OUT / "manifests/checkpoint_audit_manifest.json", checkpoint_manifest)
    dump(OUT / "checkpoint_audit_manifest.json", checkpoint_manifest)

    repeat_counts = Counter(exposure_ids)
    seen_manifest = {
        "status": "FROZEN", "population": "SEEN-TRAIN", "primary_unit": "unique image identity",
        "source_training_log": artifact(P3D2 / "training/metrics.jsonl"),
        "source_schedule": artifact(P3D2 / "training/schedule.json"),
        "total_exposures": len(exposure_ids), "unique_image_count": len(seen),
        "repeated_image_count": sum(count > 1 for count in repeat_counts.values()),
        "repeat_frequency_distribution": dict(sorted(Counter(repeat_counts.values()).items())),
        "sample_ids": [row["sample_id"] for row in seen],
        "image_identities": [identity(row) for row in seen],
        "source_annotation_ids": [annotation_ids(row) for row in seen],
        "checksum": manifest_checksum(seen),
    }
    holdout_manifest = {
        "status": "FROZEN", "population": "TRAIN-HOLDOUT", "seed": SEED,
        "sampling": "deterministic Python random.Random shuffle, first 1000 after canonical-identity exclusion",
        "available_eligible_count": len(candidates), "selected_count": len(holdout),
        "seen_sample_intersection_count": 0, "seen_identity_intersection_count": 0,
        "sample_ids": [row["sample_id"] for row in holdout],
        "image_identities": [identity(row) for row in holdout],
        "source_annotation_ids": [annotation_ids(row) for row in holdout],
        "checksum": manifest_checksum(holdout),
    }
    dump(OUT / "populations/seen_train_manifest.json", seen_manifest)
    dump(OUT / "seen_train_manifest.json", seen_manifest)
    dump(OUT / "populations/train_holdout_manifest.json", holdout_manifest)
    dump(OUT / "train_holdout_manifest.json", holdout_manifest)
    for name, population in (("seen_train", seen), ("train_holdout", holdout), ("validation_fake", validation)):
        jsonl(OUT / f"populations/{name}/test_combined.jsonl", population)
    validation_manifest = {
        "status": "FROZEN", "population": "VALIDATION-FAKE", "count": len(validation),
        "source": artifact(val_path), "checksum": manifest_checksum(validation),
        "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "populations/validation_fake_manifest.json", validation_manifest)

    experiment = {
        "status": "AUTHORIZED_READ_ONLY_DIAGNOSTIC", "phase": "Phase 3D.2-A",
        "objective": "Explain why Phase 3D.2 direct spatial optimization produced no validation localization gain.",
        "authorized_operations": ["read-only checkpoint evaluation", "paired analysis", "soft-mask diagnostics", "predefined subgroup analysis", "diagnostic module swap", "representation logging"],
        "prohibited_operations": ["backward", "optimizer", "scheduler", "parameter update", "checkpoint creation", "training", "threshold tuning", "checkpoint reselection", "internal test", "official1000", "architecture modification", "FEPN", "NPR", "SRM", "FOCAL"],
        "evaluation_runtime_required": {"model_eval": True, "torch_no_grad": True},
        "populations": {"seen_train": len(seen), "train_holdout": len(holdout), "validation_fake": len(validation)},
        "checkpoints": [row["step"] for row in checkpoint_records], "formal_mask_logit_threshold": 0.0,
        "bootstrap_repeats": 10000, "bootstrap_seed": SEED,
        "internal_test_inspected": False, "official1000_inspected": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    dump(OUT / "manifests/experiment_manifest.json", experiment)
    dump(OUT / "experiment_manifest.json", experiment)
    print(json.dumps({"status": "PASS", "seen": len(seen), "holdout": len(holdout), "validation": len(validation)}, indent=2))


if __name__ == "__main__":
    main()
