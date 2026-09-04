#!/usr/bin/env python3
"""Detached Phase 5A-3 pipeline: download, gated preflights, two stages, report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/LEGION_official"
PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
DEEPSPEED = Path("/home/yz/miniconda3/envs/legion/bin/deepspeed")
LEGION_BIN = Path("/home/yz/miniconda3/envs/legion/bin")
CKPT = ROOT / "checkpoints/phase5a3_legion_retrained"
BASE = CKPT / "base/GLaMM-GranD-Pretrained"
RUN = CKPT / "stage1_run"
STAGE1_LE = CKPT / "stage1_le"
STAGE2 = CKPT / "stage2_cls"
LOGS = CKPT / "logs"
TORCH_EXTENSIONS = CKPT / "torch_extensions"
DATA = ROOT / "outputs/phase5a3_legion_retrained/data"
CLIP = CKPT.parent / "phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
SAM = CKPT.parent / "phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"
DOC = ROOT / "docs/phase5a3_legion_retrained.md"
SOURCE_COMMIT = "d21535dd45f6fea509337a83095966f0b86ac924"
BASE_REVISION = "a2513f97c9404065cfd5849325e61d5d53456441"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_name: str, *, cwd: Path = ROOT, env: dict | None = None) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    full_env.update({
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # DeepSpeed builds fused_adam lazily.  The independent legion env has
        # both ninja and nvcc, but a detached non-conda shell does not put them
        # on PATH.  Keep the build cache on /data (via checkpoints) and use one
        # shared, prebuilt artifact for both distributed ranks.
        "PATH": f"{LEGION_BIN}:{os.environ.get('PATH', '')}",
        "CUDA_HOME": str(LEGION_BIN.parent),
        "LIBRARY_PATH": f"{LEGION_BIN.parent / 'lib'}:{os.environ.get('LIBRARY_PATH', '')}",
        "LD_LIBRARY_PATH": f"{LEGION_BIN.parent / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS),
        "MAX_JOBS": "8",
    })
    if env:
        full_env.update(env)
    log_path = LOGS / log_name
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] COMMAND {json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=full_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log_path}: {command}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(OFFICIAL), *args], text=True).strip()


def freeze_manifest(data_summary: dict) -> dict:
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    if head != SOURCE_COMMIT or status:
        raise RuntimeError(f"Official repository identity/cleanliness gate failed: {head!r} {status!r}")
    train_n = int(data_summary["stage1"]["train_fake"]["usable"])
    world, micro, grad_accum = 2, 1, 8
    manifest = {
        "schema": "phase5a3_execution_manifest_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP", "formal_optimizer_updates": 0,
        "official_source": {"path": str(OFFICIAL), "commit": head, "clean": True},
        "initialization": {
            "stage1": {"base": str(BASE), "hf_revision": BASE_REVISION, "sam": str(SAM),
                       "public_intermediate_legion_LE_used": False},
            "stage2": {"base": str(STAGE1_LE), "public_intermediate_legion_LE_used": False},
        },
        "stage1": {
            "candidate_images": data_summary["stage1"]["train_fake"]["candidate_images"],
            "candidate_annotation_samples": data_summary["stage1"]["train_fake"]["candidate_annotation_samples"],
            "usable": train_n, "excluded": data_summary["stage1"]["train_fake"]["excluded"],
            "epochs": 3, "lora_r": 8, "lr": 1e-4,
            "loss_weights": {"CE": 1.0, "Dice": 0.2, "BCE": 0.4},
            "micro_batch_per_gpu": micro, "world_size": world,
            "gradient_accumulation_steps": grad_accum,
            "nominal_global_batch": world * micro * grad_accum,
            "steps_per_epoch": math.ceil(train_n / (world * micro * grad_accum)),
            "coverage": "each usable annotation contributes exactly once per epoch; final zero-gradient padding slots",
            "selector": "minimum frozen internal-validation teacher-forced total loss; retain all epochs",
        },
        "stage2": {
            "train_total": 17672, "real_1": 8836, "fake_0": 8836,
            "world_size": 1, "per_device_batch": 64, "gradient_accumulation_steps": 1,
            "effective_global_batch": 64, "steps_per_epoch": 277, "total_optimizer_steps": 831,
            "validation_total": 2212, "epochs": 3, "lr": 1e-3,
            "trainable": "prediction_head only", "scheduler": "cosine",
            "selector": "internal validation accuracy; load_best_model_at_end",
        },
        "data_adapter": data_summary,
        "loader_firewall": {
            "source_audit_inputs": data_summary["allowed_inputs_only"] + [
                data_summary["stage1_original_annotation_lookup"]["path"]
            ],
            "training_loader_inputs": [
                data_summary["stage1"]["train_fake"]["output"],
                data_summary["stage1"]["val_fake"]["output"],
                str(DATA / "stage2/train.json"), str(DATA / "stage2/val.json"),
            ],
            "internal_test": False,
            "official1000": False, "LOKI": False, "RAISE": False,
            "AIGI_test": False, "external_benchmark": False,
        },
        "official_wrapper_deviations": [
            "bypass HybridSegDataset random-with-replacement index-0 path for exact full epochs",
            "remove official hard-coded epoch>=1 break while retaining epochs=3 recipe",
            "do not truncate validation annotations to first 1000",
            "supply inert missing prompt constants needed only by unused eager imports",
            "Stage2 preserve official LEGION label semantics Real=1 Fake=0",
            "prebuild DeepSpeed fused_adam once in the independent legion environment and reuse its /data cache across ranks",
        ],
    }
    dump(CKPT / "execution_manifest.json", manifest)
    return manifest


def directory_identity(path: Path) -> dict:
    records = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        records.append({"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256(item)})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(path), "files": records, "canonical_sha256": hashlib.sha256(canonical).hexdigest()}


def render_report(manifest: dict, data: dict, stage1_history: list, stage2_summary: dict,
                  stage1_identity: dict, stage2_identity: dict) -> None:
    pre1 = json.loads((RUN / "preflight.json").read_text())
    pre2_path = Path(stage2_summary["preflight_path"])
    pre2 = json.loads(pre2_path.read_text())
    exclusions = data["stage1"]["train_fake"]["exclusion_reason_occurrences"]
    stage1_rows = "\n".join(
        f"| {row['epoch']} | {row['train_losses']['loss']:.6f} | {row['train_losses']['ce_loss']:.6f} | "
        f"{row['train_losses']['mask_bce_loss']:.6f} | {row['train_losses']['mask_dice_loss']:.6f} | "
        f"{row['validation_losses']['loss']:.6f} |" for row in stage1_history
    )
    stage2_rows = "\n".join(
        f"| {int(round(row['epoch']))} | {row['eval_accuracy']:.6f} | {row['eval_loss']:.6f} |"
        for row in stage2_summary["validation_by_epoch"]
    )
    text = f"""# Phase 5A-3 — LEGION on Frozen Internal Training Data

## 结论

两阶段训练已完成。Stage 1 从官方指定的 `GLaMM-GranD-Pretrained@{BASE_REVISION}` + SAM 初始化；Stage 2 只从本次 Stage 1 merged LE 初始化。公开 intermediate `legion_LE` 未参与初始化。

## 固定身份与数据防火墙

- LEGION source commit: `{SOURCE_COMMIT}`；完成后 clean: `{git('status', '--porcelain') == ''}`
- Stage 1 base: `{BASE}`
- SAM: `{SAM}`
- CLIP fixed local revision: `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- train/val adapter summary SHA256: `{data['canonical_sha256']}`
- official1000 leakage = 0
- internal test leakage = 0
- external benchmark leakage = 0
- public intermediate `legion_LE` initialization = NO

Loader 只接收冻结的 train/val manifest；训练及选模未传入 internal test、official1000、LOKI、RAISE 或其他 external benchmark 路径。

## Stage 1 — Localization + Explanation

官方原始 sample unit 被保留：一条原始 annotation/caption 为一个样本，每个 ref 保留独立 phrase、独立 `<p>…</p> [SEG]` 与独立 polygon mask；不使用 R1 的 combined phrase，不做 union mask。

- frozen Fake images: {data['stage1']['train_fake']['candidate_images']}
- candidate original annotation samples: {data['stage1']['train_fake']['candidate_annotation_samples']}
- usable LE samples: {data['stage1']['train_fake']['usable']}
- excluded: {data['stage1']['train_fake']['excluded']} (`{json.dumps(exclusions, ensure_ascii=False)}`)
- validation annotation samples: {data['stage1']['val_fake']['usable']}
- epochs=3, LoRA r=8, lr=1e-4, CE=1.0, Dice=0.2, BCE=0.4
- micro batch/GPU=1, GPUs=2, grad accumulation=8, effective global batch=16
- steps/epoch={manifest['stage1']['steps_per_epoch']}; zero-gradient padding does not duplicate training contribution
- preflight loss={pre1['losses']['loss']:.6f}; peak GPU memory={pre1['peak_memory_bytes'] / 2**30:.2f} GiB
- trainable parameters={pre1['trainable_parameter_count']:,}; exact list: `{RUN / 'preflight.json'}`
- selector: minimum frozen internal-validation teacher-forced total loss; all three epochs retained
- selected checkpoint: `{manifest['stage1']['selected_checkpoint']}` (val total `{manifest['stage1']['selected_validation_total_loss']:.6f}`)

| Epoch | Train total | Train CE | Train BCE | Train Dice | Val total |
|---:|---:|---:|---:|---:|---:|
{stage1_rows}

Merged `LEGION-retrained-LE`: `{STAGE1_LE}`  
Canonical checkpoint SHA256: `{stage1_identity['canonical_sha256']}`

## Stage 2 — Detection Classification

- initialization: `{STAGE1_LE}`
- train: 17,672 = 8,836 Real (1) + 8,836 Fake (0)
- validation: 2,212 internal validation only
- epochs=3, lr=1e-3, cosine scheduler, one GPU, per-device/effective global batch={stage2_summary['batch_size']}
- trainable: prediction_head only ({stage2_summary['trainable_parameter_count']:,} parameters)
- preflight loss={pre2['loss']:.6f}; peak GPU memory={pre2['peak_memory_bytes'] / 2**30:.2f} GiB

| Epoch | Internal val accuracy | Internal val loss |
|---:|---:|---:|
{stage2_rows}

Best internal-validation checkpoint: `{stage2_summary['best_checkpoint']}`  
Best accuracy: `{stage2_summary['best_metric']:.6f}`  
`LEGION-retrained-CLS`: `{stage2_summary['final_model']}`  
Canonical checkpoint SHA256: `{stage2_identity['canonical_sha256']}`

## Paper–Code Training Configuration Discrepancy

本实验属于 **same-data controlled retraining**，不是对论文原数据与八卡硬件规模的 exact reproduction。

1. Stage 1：论文为 8 GPUs × 2/GPU = global batch 16；此前已完成的 2 GPUs × 1 × accumulation 2 = gb4 run 已标记为 `stage1_gb4_pilot`，不进入论文主比较。本次最终重跑使用 2 GPUs × 1 × accumulation 8，等效 global batch 16。
2. Stage 2：论文 nominal batch 为 8 × 64 = 512，但论文 ProGAN train 约 720k 样本；本项目 controlled train 仅 17,672。机械使用 512 会使每 epoch 仅约 35 次更新，因此保留官方 per-device batch 64、lr=1e-3、3 epochs、cosine、prediction-head-only 和 accuracy selector，采用单卡 effective global batch 64，共 277 steps/epoch、831 updates。
3. 优先冻结相同训练数据与 supervision、无泄漏、官方架构与 trainable modules，并采用与本数据规模相称的 architecture-specific optimization。

## Dependency / launcher deviations

官方仓库保持未修改。外部 wrapper 仅修复：Stage 1 单 epoch hard break、随机有放回 HybridSegDataset、validation first-1000 截断、未使用数据集的缺失 prompt 常量与官方 merge 旧方法名；Stage 2 保持官方 Real=1/Fake=0，并修复 meta-tensor loader 与图像 collator。模型核心、prompt、LoRA/loss/optimizer/scheduler recipe 未修改。

本阶段到此 STOP；未运行 official1000、LOKI、鲁棒性或 R1 对比。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from-stage1-merge", action="store_true")
    parser.add_argument("--resume-from-stage2", action="store_true")
    cli = parser.parse_args()
    CKPT.mkdir(parents=True, exist_ok=True)
    state_path = CKPT / "pipeline_state.json"
    state = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(), "stage": "download_base"}
    dump(state_path, state)
    offline = {"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    try:
        resume_after_training = cli.resume_from_stage1_merge or cli.resume_from_stage2
        if not resume_after_training:
            run([str(PYTHON), str(ROOT / "scripts/phase5a3_download_base.py")], "pipeline_download.log")
            state["stage"] = "prepare_data"; dump(state_path, state)
            run([str(PYTHON), str(ROOT / "scripts/phase5a3_prepare_data.py")], "pipeline_prepare_data.log")
        data = json.loads((DATA / "conversion_summary.json").read_text())
        manifest = freeze_manifest(data)

        common1 = [
            str(ROOT / "scripts/phase5a3_stage1_train.py"), "--base", str(BASE),
            "--dataset-dir", str(DATA / "stage1"), "--vision-pretrained", str(SAM),
            "--vision-tower", str(CLIP), "--run-dir", str(RUN), "--workers", "2",
        ]
        if not resume_after_training:
            state["stage"] = "stage1_preflight"; dump(state_path, state)
            run([str(DEEPSPEED), "--include", "localhost:1,2", "--master_port", "29631",
                 *common1, "--mode", "preflight"], "stage1_preflight.log", env=offline)
            if json.loads((RUN / "preflight.json").read_text())["status"] != "PASS":
                raise RuntimeError("Stage-1 preflight gate did not pass")

            state["stage"] = "stage1_training"; dump(state_path, state)
            run([str(DEEPSPEED), "--include", "localhost:1,2", "--master_port", "29632",
                 *common1, "--mode", "train"], "stage1_training.log", env=offline)

        train_n = data["stage1"]["train_fake"]["usable"]
        steps = math.ceil(train_n / 16)
        history = json.loads((RUN / "history.json").read_text())
        if len(history) != 3:
            raise RuntimeError("Stage-1 selector requires all three completed epochs")
        selected_stage1 = min(history, key=lambda row: row["validation_losses"]["loss"])
        tag = selected_stage1["tag"]
        manifest["stage1"]["selected_checkpoint"] = tag
        manifest["stage1"]["selected_validation_total_loss"] = selected_stage1["validation_losses"]["loss"]
        dump(CKPT / "execution_manifest.json", manifest)
        ds_root = RUN / "deepspeed_checkpoints"
        fp32 = RUN / "merged_input/pytorch_model.bin"
        fp32.parent.mkdir(parents=True, exist_ok=True)
        if not resume_after_training:
            state["stage"] = "stage1_merge_step1"; dump(state_path, state)
            run([str(PYTHON), str(ds_root / "zero_to_fp32.py"), str(ds_root), str(fp32), "--tag", tag],
                "stage1_merge_step1.log", env=offline)
        elif not cli.resume_from_stage2:
            if not fp32.is_file():
                raise RuntimeError("Refusing merge resume: selected fp32 state is missing")
        if not cli.resume_from_stage2:
            state["stage"] = "stage1_merge_step2"; dump(state_path, state)
            run([str(PYTHON), str(ROOT / "scripts/phase5a3_merge_stage1.py"),
                 "--version", str(BASE), "--weight", str(fp32), "--save_path", str(STAGE1_LE),
                 "--vision_pretrained", str(SAM), "--vision-tower", str(CLIP)],
                "stage1_merge_step2.log", env=offline)
        elif not (STAGE1_LE / "pytorch_model.bin.index.json").is_file():
            raise RuntimeError("Refusing Stage-2 resume: merged Stage-1 checkpoint is incomplete")

        common2 = [
            str(ROOT / "scripts/phase5a3_stage2_train.py"), "--stage1-le", str(STAGE1_LE),
            "--vision-pretrained", str(SAM), "--vision-tower", str(CLIP),
            "--train-json", str(DATA / "stage2/train.json"), "--val-json", str(DATA / "stage2/val.json"),
            "--run-dir", str(STAGE2), "--workers", "4",
        ]
        stage2_env = {**offline, "CUDA_VISIBLE_DEVICES": "2"}
        state["stage"] = "stage2_preflight"; dump(state_path, state)
        selected_batch = 64
        run([str(PYTHON), *common2, "--mode", "preflight", "--batch-size", "64"],
            "stage2_preflight_batch64.log", env=stage2_env)
        state["stage"] = "stage2_training"; state["stage2_batch"] = selected_batch; dump(state_path, state)
        run([str(PYTHON), *common2, "--mode", "train", "--batch-size", str(selected_batch)],
            "stage2_training.log", env=stage2_env)
        stage2_summary_path = STAGE2 / "training_summary.json"
        stage2_summary = json.loads(stage2_summary_path.read_text())
        stage2_summary["preflight_path"] = str(STAGE2 / f"preflight_batch{selected_batch}.json")
        dump(stage2_summary_path, stage2_summary)

        state["stage"] = "identities_and_report"; dump(state_path, state)
        stage1_identity = directory_identity(STAGE1_LE)
        stage2_identity = directory_identity(Path(stage2_summary["final_model"]))
        dump(CKPT / "stage1_le_identity.json", stage1_identity)
        dump(CKPT / "stage2_cls_identity.json", stage2_identity)
        history = json.loads((RUN / "history.json").read_text())
        render_report(manifest, data, history, stage2_summary, stage1_identity, stage2_identity)
        if git("status", "--porcelain"):
            raise RuntimeError("Official LEGION repository became dirty")
        state.update({"status": "COMPLETE", "stage": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                      "report": str(DOC), "stage1_le": str(STAGE1_LE),
                      "stage2_cls": stage2_summary["final_model"]})
        dump(state_path, state)
    except Exception as exc:
        state.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                      "failed_at": datetime.now(timezone.utc).isoformat()})
        dump(state_path, state)
        raise


if __name__ == "__main__":
    main()
