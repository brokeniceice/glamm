#!/usr/bin/env python3
"""Freeze Phase 3D.2 manifests before any validation or training."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d2 import parameter_group, stable_fake_schedule

CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["source"]["checkpoint"]).resolve()
    if file_sha256(source) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("frozen P1 checkpoint hash mismatch")
    selector = load_json(ROOT / cfg["source"]["selector"])
    if Path(selector["selected_checkpoint"]).resolve() != source:
        raise RuntimeError("P1 selector does not resolve to the frozen source")

    train_path = (ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl").resolve()
    val_path = (ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl").resolve()
    train_rows, val_rows = rows(train_path), rows(val_path)
    if len({row["sample_id"] for row in train_rows}) != len(train_rows):
        raise RuntimeError("duplicate training sample ids")
    fake = [row for row in train_rows if int(row["class_label"]) == 1]
    for row in fake:
        if not row.get("refs") or any(not ref.get("phrase") for ref in row["refs"]):
            raise RuntimeError(f"Fake row lacks frozen refs/phrase: {row['sample_id']}")
    exposures = int(cfg["training"]["total_fake_image_exposures"])
    schedule = stable_fake_schedule(train_rows, int(cfg["experiment"]["seed"]), exposures)

    state = torch.load(source, map_location="cpu")
    if int(state["optimizer_step"]) != int(cfg["source"]["optimizer_step"]):
        raise RuntimeError("P1 optimizer step mismatch")
    stored_counts = Counter()
    stored_tensors = Counter()
    for name, tensor in state["module"].items():
        if not torch.is_tensor(tensor) or name.endswith("inv_freq") or "position_ids" in name or "positional_encoding_gaussian_matrix" in name:
            continue
        group = parameter_group(name)
        stored_counts[group] += tensor.numel(); stored_tensors[group] += 1
    del state

    val_alias = out / "evaluation/selector/val_as_test_manifest"
    val_alias.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_path, val_alias / "test_combined.jsonl")
    dump(out / "experiment_manifest.json", {
        "status": "PREPARED_NOT_STARTED", "phase": "Phase 3D.2",
        "research_question": "Can direct optimization of the existing spatial grounding pathway improve localization while keeping the language policy fixed?",
        "arms": cfg["experiment"]["arms"], "only_intervention": "direct spatial-path optimization",
        "phase3d1_frozen_gate": "GATE_REWARD_FORMULATION_NOT_PRIMARY_BOTTLENECK",
        "internal_test_inspected": False, "official1000_inspected": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    dump(out / "initial_checkpoint_manifest.json", {
        "status": "FROZEN_BEFORE_TRAINING", "checkpoint": artifact(source),
        "optimizer_step": cfg["source"]["optimizer_step"], "epoch": cfg["source"]["epoch"],
        "model_config": artifact(ROOT / cfg["source"]["model_config"]),
        "stored_parameter_counts": dict(stored_counts), "stored_parameter_tensor_counts": dict(stored_tensors),
        "trainable_modules": cfg["trainable"]["allowed"], "all_other_parameters_frozen": True,
    })
    dump(out / "train_manifest.json", {
        "status": "FROZEN_BEFORE_TRAINING", "source": artifact(train_path), "split": "train",
        "total_images": len(train_rows), "fake_images": len(fake),
        "scheduled_unique_fake_exposures": exposures,
        "scheduled_sample_ids": [train_rows[index]["sample_id"] for index in schedule],
        "source_counts": dict(Counter(row["source"] for row in fake)),
        "mask_provenance": cfg["data"]["mask_provenance"],
        "target_construction": cfg["data"]["target_construction"],
        "rasterizer": "dataset.forensics.synthscars.polygons_for_target + polygon_to_mask",
        "authoritative_phrase_construction": "annotation order, whitespace normalization, exact duplicate deduplication, semicolon join",
        "generated_context_replay": False, "real_mask_loss": False,
    })
    dump(out / "training_config.json", {
        "status": "FROZEN_BEFORE_TRAINING", "optimizer": cfg["optimizer"],
        "training": cfg["training"], "loss": cfg["loss"], "trainable": cfg["trainable"],
        "teacher_forced_context": "[FAKE] explanation\\nTarget regions: <authoritative phrase> [SEG]",
        "language_forward_grad_enabled": False, "validation_retuning_allowed": False,
    })
    dump(out / "checkpoint_selection_protocol.json", {
        "status": "FROZEN_BEFORE_TRAINING", **cfg["selector"],
        "validation_manifest": artifact(val_path), "validation_count": len(val_rows),
        "validation_fake_count": sum(int(row["class_label"]) == 1 for row in val_rows),
        "internal_test_used": False, "official1000_used": False,
    })
    dump(out / "preparation_audit.json", {
        "status": "PASS", "config": artifact(CONFIG), "source_exact": True,
        "fake_only": True, "authoritative_context": True, "generated_replay": False,
        "reward_used": False, "test_sealed": True, "phase3d2_training_started": False,
    })
    print(json.dumps({"status": "PASS", "train": len(train_rows), "fake": len(fake), "val": len(val_rows)}, indent=2))


if __name__ == "__main__":
    main()
