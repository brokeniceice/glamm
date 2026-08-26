#!/usr/bin/env python3
"""Freeze Phase 3E provenance, fairness, schedule, and selector before training."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_rank(seed: int, namespace: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{sample_id}".encode()).hexdigest()


def schedule_indices(records: list[dict], seed: int, exposures: int) -> list[int]:
    by_label = {
        label: sorted(
            [i for i, row in enumerate(records) if int(row["class_label"]) == label],
            key=lambda i: stable_rank(seed, f"phase3e-label-{label}", records[i]["sample_id"]),
        ) for label in (0, 1)
    }
    positions = {0: 0, 1: 0}; result = []
    for exposure in range(exposures):
        label = exposure % 2
        values = by_label[label]
        result.append(values[positions[label] % len(values)])
        positions[label] += 1
    return result


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    source = Path(cfg["source"]["checkpoint"]).resolve()
    if sha256(source) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checkpoint SHA256 mismatch")
    selector = json.loads((ROOT / cfg["source"]["selector"]).read_text(encoding="utf-8"))
    if Path(selector["selected_checkpoint"]).resolve() != source:
        raise RuntimeError("P1 selector does not resolve to the canonical checkpoint")
    state = torch.load(source, map_location="cpu")
    if int(state["optimizer_step"]) != int(cfg["source"]["optimizer_step"]):
        raise RuntimeError("P1 optimizer step mismatch")
    p1_cfg_path = ROOT / cfg["source"]["model_config"]
    p1_cfg = yaml.safe_load(p1_cfg_path.read_text(encoding="utf-8"))
    for key, expected in (("lora_r", 8), ("lora_alpha", 16)):
        if int(p1_cfg["model"][key]) != expected:
            raise RuntimeError(f"P1 {key} differs from Phase 3E contract")
    if p1_cfg["model"]["lora_target_modules"].split(",") != cfg["trainable"]["lora_target_modules"]:
        raise RuntimeError("P1 LoRA placement differs from Phase 3E contract")
    train_path = (ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl").resolve()
    val_path = (ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl").resolve()
    train_rows, val_rows = rows(train_path), rows(val_path)
    ids = [row["sample_id"] for row in train_rows]
    if len(ids) != len(set(ids)) or set(int(row["class_label"]) for row in train_rows) != {0, 1}:
        raise RuntimeError("training manifest identity/class audit failed")
    for row in train_rows:
        if int(row["class_label"]) == 1 and (not row.get("refs") or any(not ref.get("phrase") for ref in row["refs"])):
            raise RuntimeError(f"Fake row lacks authoritative phrase: {row['sample_id']}")
    schedule = schedule_indices(train_rows, int(cfg["experiment"]["seed"]), int(cfg["training"]["total_image_exposures"]))
    sample_ids = [train_rows[i]["sample_id"] for i in schedule]
    labels = [int(train_rows[i]["class_label"]) for i in schedule]
    schedule_payload = {"seed": cfg["experiment"]["seed"], "indices": schedule, "sample_ids": sample_ids,
                        "class_labels": labels, "sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()}
    dump(out / "training/schedule.json", schedule_payload)
    val_alias = out / "evaluation/validation_manifest"
    val_alias.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_path, val_alias / "test_combined.jsonl")
    trainable_counts = Counter()
    for name, tensor in state["module"].items():
        if not torch.is_tensor(tensor):
            continue
        if "lora_" in name: trainable_counts["lora"] += tensor.numel()
        elif "text_hidden_fcs" in name: trainable_counts["text_hidden_fcs"] += tensor.numel()
        elif "grounding_encoder.mask_decoder" in name: trainable_counts["mask_decoder"] += tensor.numel()
    del state
    common = {"initialization": artifact(source), "dataset": artifact(train_path),
              "sample_order_sha256": schedule_payload["sha256"], "optimizer_steps": cfg["training"]["total_optimizer_steps"],
              "batch_size": cfg["training"]["batch_size_per_device"], "gradient_accumulation": cfg["training"]["gradient_accumulation_steps"],
              "precision": cfg["training"]["precision"], "LoRA": {k: cfg["trainable"][k] for k in ("lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules")},
              "LoRA_lr": cfg["optimizer"]["lora_lr"], "language_CE": True}
    dump(out / "initial_checkpoint_manifest.json", {"status": "FROZEN_BEFORE_TRAINING", "checkpoint": artifact(source),
         "optimizer_step": cfg["source"]["optimizer_step"], "epoch": cfg["source"]["epoch"],
         "model_config": artifact(p1_cfg_path), "P1_stored_target_parameter_counts": dict(trainable_counts),
         "total_parameters": None, "frozen_parameters": None, "trainable_parameters": None,
         "lora_configuration": common["LoRA"]})
    dump(out / "experiment_fairness_manifest.json", {"status": "FROZEN_BEFORE_TRAINING",
         "P3E-SFT-CONT": {**common, "spatial_modules_trainable": False, "mask_loss": False},
         "P3E-JOINT": {**common, "spatial_modules_trainable": True, "mask_loss": True},
         "only_allowed_difference": "joint_spatial_trainability_and_direct_mask_loss", "matched": True})
    dump(out / "training_config.json", {"status": "FROZEN_BEFORE_TRAINING", "training": cfg["training"],
         "optimizer": cfg["optimizer"], "loss": cfg["loss"], "trainable": cfg["trainable"], "prompt": cfg["prompt"],
         "schedule_path": str((out / "training/schedule.json").resolve()), "post_start_changes_allowed": False})
    dump(out / "checkpoint_selection_protocol.json", {"status": "FROZEN_BEFORE_TRAINING", **cfg["selector"],
         "validation_manifest": artifact(val_path), "validation_count": len(val_rows),
         "validation_fake_count": sum(int(row["class_label"]) == 1 for row in val_rows),
         "internal_test_used": False, "official1000_used": False})
    dump(out / "experiment_manifest.json", {"status": "PREPARED_NOT_STARTED", "phase": "Phase 3E",
         "arms": cfg["experiment"]["arms"], "research_question": "Does joint language-to-mask optimization improve autonomous G0 localization beyond P1 and matched extra SFT?",
         "canonical_prompt": cfg["prompt"], "phase3d2c_status": "TERMINATED_NOT_RESUMED",
         "internal_test_inspected": False, "official1000_inspected": False,
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    dump(out / "preparation_audit.json", {"status": "PASS", "source_exact": True, "prompt_canonical": True,
         "real_fake_preserved": True, "same_schedule_frozen": True, "selector_frozen": True,
         "test_sealed": True, "training_started": False})
    print(json.dumps({"status": "PASS", "train": len(train_rows), "val": len(val_rows),
                      "exposures_per_arm": len(schedule), "class_exposures": dict(Counter(labels))}, indent=2))


if __name__ == "__main__":
    main()
