#!/usr/bin/env python3
"""Pre-training audit gate for Phase 3A.1."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import deepspeed
import torch
import transformers
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a1_paired_control"
AUDIT = OUT / "audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        result[prefix] = value
    return result


def manifest_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = [row["sample_id"] for row in rows]
    return {
        "path": str(path), "sha256": sha256(path), "row_count": len(rows),
        "sample_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "unique_ids": len(set(ids)),
        "real": sum(int(row["class_label"]) == 0 for row in rows),
        "fake": sum(int(row["class_label"]) == 1 for row in rows),
    }


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    c0_path = ROOT / "configs/phase3a1_paired_c0.yaml"
    p1_path = ROOT / "configs/phase3a_p1.yaml"
    c0 = yaml.safe_load(c0_path.read_text()); p1 = yaml.safe_load(p1_path.read_text())
    base = ROOT / c0["model"]["version"]
    base_files = sorted(base.glob("pytorch_model-*.bin"))
    base_hashes = {path.name: sha256(path) for path in base_files}
    initialization = {
        "initialization_identity_status": "BASE_CHECKPOINT_MATCH_BUT_RANDOM_INIT_UNPROVEN",
        "phase2a_step0_status": "PHASE2A_STEP0_NOT_RECOVERABLE",
        "p1_step0_status": "P1_STEP0_NOT_RECOVERABLE",
        "exact_weight_identity": False,
        "shared_base_checkpoint_path": str(base.resolve()),
        "shared_base_checkpoint_file_sha256": base_hashes,
        "phase2a_recorded_git_head": (ROOT / "outputs/phase2a_unified_baseline/git_commit.txt").read_text().strip(),
        "p1_recorded_git_head": (ROOT / "outputs/phase3a_phrase_grounding/training/p1/git_commit.txt").read_text().strip(),
        "random_or_launch_initialized": {
            "classification_head": "torch.nn.Linear + post_init during from_pretrained; absent keys initialized at launch",
            "lora": "PEFT get_peft_model after torch.manual_seed(3407)",
            "new_special_token_rows": "resize_token_embeddings after torch.manual_seed(3407)",
        },
        "seed_set_before_model_construction": 3407,
        "new_c0_policy": (
            "Reconstruct the documented P1 launch deterministically from the same immutable base, seed, "
            "model-building code and single-GPU environment; actual P1 step-0 byte identity remains unprovable."
        ),
    }
    reconstructed = AUDIT / "reconstructed_initializations"
    p1_hash_path = reconstructed / "p1/initialization_hash.json"
    c0_hash_path = reconstructed / "c0/initialization_hash.json"
    if p1_hash_path.exists() and c0_hash_path.exists():
        p1_hash = json.loads(p1_hash_path.read_text())
        c0_hash = json.loads(c0_hash_path.read_text())
        initialization["reconstructed_intended_initialization"] = {
            "p1_full_hash": p1_hash["full_state"]["sha256"],
            "c0_full_hash": c0_hash["full_state"]["sha256"],
            "p1_trainable_hash": p1_hash["trainable_state"]["sha256"],
            "c0_trainable_hash": c0_hash["trainable_state"]["sha256"],
            "exact_reconstruction_match": (
                p1_hash["full_state"]["sha256"] == c0_hash["full_state"]["sha256"]
                and p1_hash["trainable_state"]["sha256"] == c0_hash["trainable_state"]["sha256"]
            ),
            "boundary": "Confirms two current deterministic reconstructions, not actual P1 step-0 recovery.",
        }
    dump(AUDIT / "initialization_audit.json", initialization)

    env = {
        "historical_phase2a": {
            "trajectory": "world_size=2 through durable step 1001, then world_size=1 recovery segments",
            "world_size_changes": True, "effective_global_batch": 20,
            "recorded_git_head": initialization["phase2a_recorded_git_head"],
        },
        "p1": {
            "trajectory": "world_size=1 throughout; checkpoint resumes after external-contention OOMs",
            "world_size_changes": False, "gpu": "NVIDIA RTX A6000",
            "micro_batch": 10, "gradient_accumulation": 2, "effective_global_batch": 20,
            "recorded_git_head": initialization["p1_recorded_git_head"],
        },
        "new_c0": {
            "world_size": 1, "gpu": "NVIDIA RTX A6000", "micro_batch": 10,
            "gradient_accumulation": 2, "effective_global_batch": 20,
            "cuda_cache_policy": "GLAMM_PRESERVE_CUDA_CACHE=1; no separate reservation process",
        },
        "current_software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "deepspeed": deepspeed.__version__,
        },
        "semantic_training_parity_p1_new_c0": {
            "optimizer": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.0,
            "warmup_steps": 100, "scheduler": "linear warmup decay", "total_steps": 5000,
            "bf16": True, "zero_stage": 2, "gradient_clip": 1.0, "seed": 3407,
            "workers": 8, "validation_every_steps": 500, "selector": "min val_total_loss",
        },
        "nonsemantic_execution_difference": {
            "cuda_cache_retention": "enabled for new C0 per user direction; P1 called empty_cache",
            "expected_numerical_effect": "none; allocator reservation policy only",
        },
    }
    dump(AUDIT / "training_environment_diff.json", env)

    manifest_dir = ROOT / c0["data"]["manifest_dir"]
    splits = {split: manifest_summary(manifest_dir / f"{split}_combined.jsonl")
              for split in ("train", "val", "test")}
    data = {
        "status": "EXACT_SHARED_MANIFEST_BYTES",
        "c0_manifest_dir": str(manifest_dir), "p1_manifest_dir": str(manifest_dir),
        "splits": splits, "same_image_preprocessing": True, "same_union_mask_builder": True,
        "same_annotation_source": True, "same_sampler_policy": True,
        "only_target_difference": "P1 inserts authoritative Target regions span before [SEG]",
    }
    dump(AUDIT / "train_manifest_diff.json", data)

    cflat, pflat = flatten(c0), flatten(p1)
    keys = sorted(set(cflat) | set(pflat)); differences = []
    for key in keys:
        if cflat.get(key) != pflat.get(key):
            kind = "phrase_protocol" if key in {
                "forensics.target_protocol", "forensics.target_template",
                "forensics.localization_phrase_insertion", "forensics.localization_phrase_lm_target",
            } else "operational_metadata" if key in {
                "experiment.name", "experiment.runtime_output_dir", "experiment.experiment_type",
                "checkpoint.output_root", "training.cuda_cache_policy",
            } else "FORBIDDEN_SEMANTIC_DIFFERENCE"
            differences.append({"path": key, "c0": cflat.get(key), "p1": pflat.get(key), "kind": kind})
    forbidden = [row for row in differences if row["kind"] == "FORBIDDEN_SEMANTIC_DIFFERENCE"]
    diff = {"status": "PASS" if not forbidden else "FAIL", "differences": differences,
            "forbidden_differences": forbidden}
    dump(AUDIT / "c0_p1_exact_config_diff.json", diff)
    if forbidden:
        raise RuntimeError(f"Forbidden C0/P1 config differences: {forbidden}")

    manifest = {
        "phase": "3A.1", "stage": "AUDIT_COMPLETE_TRAINING_NOT_STARTED", "language": "zh-CN",
        "p1_retrained": False, "new_c0_trained": False,
        "official_test_used_for_training": False,
        "official_test_used_for_checkpoint_selection": False,
        "threshold_sweep_performed": False, "new_phrase_labels_created": False,
        "new_masks_created": False, "forensic_fusion_used": False,
        "sam_modified": False, "phase3b_started": False,
        "initialization_identity_status": initialization["initialization_identity_status"],
        "paired_control_status": "BEST_EFFORT_P1_MATCHED_CONTROL_PENDING",
        "tf_protocol_parity_status": "PENDING", "phrase_effect_gate": "PENDING",
    }
    dump(OUT / "manifest.json", manifest)
    (OUT / "c0_training/configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(c0_path, OUT / "c0_training/configs/phase3a1_paired_c0.yaml")
    summary = """# Phase 3A.1 训练前审计摘要

- Phase 2A 与 P1 均引用当前同一份 GLaMM-FullScope base checkpoint，分片 SHA256 已冻结。
- Phase 2A 与 P1 的真实 step-0 snapshot 均未保存，因此随机初始化模块的 byte identity 无法事后证明。
- 初始化状态：`BASE_CHECKPOINT_MATCH_BUT_RANDOM_INIT_UNPROVEN`。
- new C0 将用 P1 的 seed=3407、当前可重建 launch 顺序、单卡 micro10/GAS2/ZeRO-2 重建 intended initialization；它是 best-effort matched control，不冒充 actual-P1-step0 exact fork。
- train/val/test manifest bytes、sample IDs、图像预处理和 union-mask builder 完全共享。
- 语义配置差异仅为 historical target 与 phrase-aligned target；输出路径/实验名以及用户要求的 allocator cache retention 属于操作性差异。
- official test 尚未使用，允许进入 smoke gate。
"""
    (AUDIT / "audit_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
