#!/usr/bin/env python3
"""Freeze Phase 3D.1 arms, data, rewards, fairness, budget, and selector before training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d1 import canonical_json_sha256


CONFIG = ROOT / "configs/phase3d1_policy_optimization.yaml"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def artifact(path: Path) -> dict:
    value = path.resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {"path": str(value), "sha256": file_sha256(value), "bytes": value.stat().st_size}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    source = Path(cfg["source"]["checkpoint"]).resolve()
    if file_sha256(source) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 source checkpoint checksum mismatch")
    selector = load_json((ROOT / cfg["source"]["selector"]).resolve())
    if Path(selector["selected_checkpoint"]).resolve() != source:
        raise RuntimeError("P1 selector does not resolve to the frozen source checkpoint")

    p1_cfg_path = (ROOT / cfg["source"]["model_config"]).resolve()
    p1_cfg = yaml.safe_load(p1_cfg_path.read_text(encoding="utf-8"))
    manifest_path = (ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl").resolve()
    rows = read_jsonl(manifest_path)
    sample_ids = [str(row["sample_id"]) for row in rows]
    labels = [int(row["class_label"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or set(labels) != {0, 1}:
        raise RuntimeError("training manifest uniqueness/class balance audit failed")

    reward_paths = {
        "R3_implementation": ROOT / cfg["reward"]["R3"]["implementation"],
        "R3_definition": ROOT / cfg["reward"]["R3"]["definition"],
        "Q2_implementation": ROOT / cfg["reward"]["Q2"]["implementation"],
        "Q2_definition": ROOT / cfg["reward"]["Q2"]["definition"],
        "Q2_selected_reward": ROOT / cfg["reward"]["Q2"]["selected_reward"],
        "Q2_semantic_encoder": ROOT / cfg["reward"]["Q2"]["semantic_encoder_provenance"],
        "Q2_token_idf": ROOT / cfg["reward"]["Q2"]["token_idf"],
    }
    rewards = {key: artifact(path) for key, path in reward_paths.items()}
    selected = load_json(reward_paths["Q2_selected_reward"])
    if selected.get("selected_reward") != "Q2" or selected.get("formula_changed") is not False:
        raise RuntimeError("Q2 is not the frozen Phase 3D.0-R candidate")

    common = {
        "dataset": artifact(manifest_path),
        "sampler": cfg["data"]["sampler"],
        "batch_size_prompts_per_device": cfg["training"]["batch_size_prompts_per_device"],
        "rollout_group_size": cfg["algorithm"]["K"],
        "optimizer": cfg["optimizer"],
        "scheduler": cfg["optimizer"]["scheduler"],
        "warmup_steps": cfg["optimizer"]["warmup_steps"],
        "total_optimizer_steps": cfg["training"]["total_optimizer_steps"],
        "trainable": cfg["trainable"],
        "precision": cfg["training"]["precision"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "algorithm": cfg["algorithm"],
        "sampling": cfg["sampling"],
        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
    }
    fairness = {
        "status": "FROZEN_BEFORE_TRAINING",
        "arms": {
            "P3D1-R3": {**common, "reward_version": "R3_phase3d0"},
            "P3D1-Q2": {**common, "reward_version": "Q2_phase3d0r"},
        },
        "only_allowed_difference": "reward_version_and_frozen_reward_formula",
        "common_contract_sha256": canonical_json_sha256(common),
        "matched_outside_reward": True,
        "RSFT_in_initial_comparison": False,
    }
    initial = {
        "status": "FROZEN_BEFORE_TRAINING",
        "checkpoint": artifact(source),
        "optimizer_step": cfg["source"]["optimizer_step"],
        "epoch": cfg["source"]["epoch"],
        "model_config": artifact(p1_cfg_path),
        "trainable_modules": p1_cfg["trainable"],
        "lora": {key: p1_cfg["model"][key] for key in (
            "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
        )},
    }
    train_manifest = {
        "status": "FROZEN_BEFORE_TRAINING", "split": "train",
        "source": artifact(manifest_path), "count": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "class_counts": {"real": labels.count(0), "fake": labels.count(1)},
        "sample_ids_in_source_order": sample_ids,
        "fake_only": False,
    }
    selector_protocol = {
        "status": "FROZEN_BEFORE_TRAINING", **cfg["selector"],
        "checkpoint_steps": list(range(
            cfg["training"]["checkpoint_interval"],
            cfg["training"]["total_optimizer_steps"] + 1,
            cfg["training"]["checkpoint_interval"],
        )),
        "selection_rule": "maximize mean_foreground_iou; exact tie uses mean_foreground_f1",
        "training_reward_used": False,
    }
    reward_manifest = {
        "status": "FROZEN_BEFORE_TRAINING",
        "R3": {"version": "R3_phase3d0", "artifacts": {k: v for k, v in rewards.items() if k.startswith("R3")}},
        "Q2": {"version": "Q2_phase3d0r", "artifacts": {k: v for k, v in rewards.items() if k.startswith("Q2")}},
        "runtime_reward_modification_allowed": False,
        "GPT_score_used": False,
    }
    experiment = {
        "status": "PREPARED_NOT_STARTED", "arms": cfg["experiment"]["arms"],
        "initial_comparison": ["P1-FROZEN", "P3D1-R3", "P3D1-Q2"],
        "optional_arms_deferred": cfg["experiment"]["excluded_initial_arms"],
        "research_question": "Does Q2 reward optimize forensic MLLM more effectively than R3 reward?",
        "physical_gpu_assignment": {"P3D1-R3": 1, "P3D1-Q2": 2},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    training_config = {
        "status": "FROZEN_BEFORE_TRAINING",
        "algorithm": cfg["algorithm"], "sampling": cfg["sampling"],
        "training": cfg["training"], "optimizer": cfg["optimizer"],
        "trainable": cfg["trainable"], "runtime": cfg["runtime"],
        "total_rollouts_per_arm": cfg["training"]["total_optimizer_steps"] * cfg["algorithm"]["K"],
    }

    dump(out / "initial_checkpoint_manifest.json", initial)
    dump(out / "train_manifest.json", train_manifest)
    dump(out / "experiment_manifest.json", experiment)
    dump(out / "fairness_manifest.json", fairness)
    dump(out / "experiment_fairness_manifest.json", fairness)
    dump(out / "reward_version_manifest.json", reward_manifest)
    dump(out / "training_config.json", training_config)
    dump(out / "selector_protocol.json", selector_protocol)
    dump(out / "checkpoint_selection_protocol.json", selector_protocol)
    dump(out / "preparation_audit.json", {
        "status": "PASS", "config": artifact(CONFIG),
        "source_checkpoint_exact": True, "manifest_unique": True,
        "real_fake_preserved": True, "rewards_frozen": True,
        "selector_frozen": True, "fairness_frozen": True,
        "phase3d1_training_started": False,
    })
    print(json.dumps({"status": "PASS", "output": str(out), "train_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
