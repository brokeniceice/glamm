#!/usr/bin/env python3
"""Freeze Phase 4A manifests, baseline provenance, architecture and budget."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.fepn import FEPNv0, parameter_manifest
from tools.phase4a import file_sha256, load_rows


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config_path = ROOT / "configs/phase4a_fepn_evidence_learnability.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["experiment"]["output_root"]
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    for name in ("manifests", "training", "evaluation", "statistics", "qualitative", "audits"):
        (output / name).mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    link = output / "checkpoints"
    if not link.exists() and not link.is_symlink():
        os.symlink(checkpoint_root, link, target_is_directory=True)
    elif link.resolve() != checkpoint_root.resolve():
        raise RuntimeError(f"checkpoint link mismatch: {link}")

    train_path = ROOT / config["data"]["manifest_dir"] / "train_combined.jsonl"
    val_path = ROOT / config["data"]["manifest_dir"] / "val_combined.jsonl"
    if file_sha256(train_path) != config["data"]["train_manifest_sha256"]:
        raise RuntimeError("train manifest hash mismatch")
    if file_sha256(val_path) != config["data"]["val_manifest_sha256"]:
        raise RuntimeError("validation manifest hash mismatch")
    train, val = load_rows(train_path), load_rows(val_path)
    train_ids, val_ids = [str(x["sample_id"]) for x in train], [str(x["sample_id"]) for x in val]
    checks = {
        "train_count": len(train) == int(config["data"]["train_total"]),
        "val_count": len(val) == int(config["data"]["val_total"]),
        "train_unique": len(set(train_ids)) == len(train),
        "val_unique": len(set(val_ids)) == len(val),
        "train_val_disjoint": not (set(train_ids) & set(val_ids)),
        "train_balanced": sum(int(x["class_label"]) == 0 for x in train) == sum(int(x["class_label"]) == 1 for x in train) == 8836,
        "val_balanced": sum(int(x["class_label"]) == 0 for x in val) == sum(int(x["class_label"]) == 1 for x in val) == 1106,
    }
    baseline_checkpoint = ROOT / config["baseline"]["checkpoint"]
    baseline_predictions = ROOT / config["baseline"]["predictions"]
    baseline_logits = ROOT / config["baseline"]["low_res_logits"]
    if file_sha256(baseline_checkpoint) != config["baseline"]["checkpoint_sha256"]:
        raise RuntimeError("CLIP baseline checkpoint hash mismatch")
    if file_sha256(baseline_predictions) != config["baseline"]["predictions_sha256"]:
        raise RuntimeError("CLIP baseline prediction hash mismatch")
    clip_rows = load_rows(baseline_predictions)
    val_fake_ids = [str(x["sample_id"]) for x in val if int(x["class_label"]) == 1]
    clip_ids = [str(x["sample_id"]) for x in clip_rows]
    checks.update({
        "clip_population_exact_order": clip_ids == val_fake_ids,
        "clip_population_unique": len(set(clip_ids)) == 1106,
        "clip_logits_present": baseline_logits.is_file(),
        "threshold_fixed_zero": float(config["evaluation"]["mask_logit_threshold"]) == 0.0,
        "test_not_loaded": True,
        "official1000_not_loaded": True,
    })
    if not all(checks.values()):
        raise RuntimeError(f"Phase 4A prepare audit failed: {[k for k,v in checks.items() if not v]}")

    torch.manual_seed(int(config["experiment"]["seed"]))
    model = FEPNv0(config["preprocess"]["image_mean"], config["preprocess"]["image_std"])
    architecture = parameter_manifest(model, int(config["preprocess"]["input_resolution"]))
    architecture.update({
        "status": "PASS" if architecture["total_parameters"] <= int(config["architecture"]["parameter_limit"]) else "FAIL",
        "parameter_limit": int(config["architecture"]["parameter_limit"]),
        "views": ["normalized_RGB", "fixed_depthwise_Laplacian_residual"],
        "fusion": "channel_concat_then_learnable_1x1_projection",
        "encoder": "lightweight_convolutional_multiscale_top_down_fusion",
    })
    if architecture["status"] != "PASS":
        raise RuntimeError("FEPN-v0 exceeds parameter limit")

    dataset_manifest = {
        "status": "FROZEN", "checks": checks,
        "train": {"path": str(train_path), "sha256": file_sha256(train_path), "n": len(train), "real": 8836, "fake": 8836},
        "validation": {"path": str(val_path), "sha256": file_sha256(val_path), "n": len(val), "real": 1106, "fake": 1106},
        "target": config["data"]["target"], "dense_supervision": "Fake_only",
        "internal_test": "SEALED", "official1000": "SEALED",
    }
    baseline_manifest = {
        "status": "REUSE_EXACT_MATCH", "name": config["baseline"]["name"],
        "checkpoint": str(baseline_checkpoint), "checkpoint_sha256": file_sha256(baseline_checkpoint),
        "predictions": str(baseline_predictions), "predictions_sha256": file_sha256(baseline_predictions),
        "low_res_logits": str(baseline_logits), "validation_fake_n": 1106,
        "population_order_exact": True, "target_constructor": "UnifiedForensicsDataset._fake_union_mask",
        "geometry": config["preprocess"]["geometry"], "inverse_geometry": "tools.phase3c1.inverse_logits",
        "threshold": 0.0, "hyperparameters_retuned": False,
    }
    experiment = {
        "status": "AUTHORIZED_PREPARED", "phase": "Phase 4A", "model": "FEPN-v0",
        "scientific_question": "standalone task-aligned global+dense forensic evidence learnability",
        "P1_involved": False, "LLM_involved": False, "SAM_involved": False, "SEG_predictor_involved": False,
        "current_split_scope": ["train", "internal_validation"], "internal_test_used": False, "official1000_used": False,
        "architecture_sweep": False, "threshold_tuning": False,
    }
    repo_audit = {
        "status": "PASS", "documentation": "docs/phase4a_repo_audit.md",
        "reuse": ["canonical manifests", "union-mask constructor", "CLIP geometry", "dense loss", "binary metrics", "paired bootstrap", "matched CLIP artifacts"],
        "new": ["standalone lightweight FEPN-v0", "balanced global+dense training loop"],
        "baseline_exact_match": True, "checks": checks,
    }
    training_snapshot = json.loads(json.dumps(config))
    dump(output / "experiment_manifest.json", experiment)
    dump(output / "dataset_manifest.json", dataset_manifest)
    dump(output / "training_config.json", training_snapshot)
    dump(output / "repo_audit.json", repo_audit)
    dump(output / "architecture_manifest.json", architecture)
    dump(output / "parameter_count.json", architecture)
    dump(output / "clip_baseline_manifest.json", baseline_manifest)
    dump(output / "manifests/experiment_manifest.json", experiment)
    dump(output / "manifests/dataset_manifest.json", dataset_manifest)
    dump(output / "manifests/clip_baseline_manifest.json", baseline_manifest)
    print(json.dumps({"status": "PASS", "checks": checks, "architecture": architecture,
                      "steps_per_epoch": config["training"]["steps_per_epoch"],
                      "total_steps": config["training"]["total_steps"],
                      "image_exposures": config["training"]["image_exposures"]}, indent=2))


if __name__ == "__main__":
    main()
