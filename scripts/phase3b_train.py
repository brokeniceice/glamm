#!/usr/bin/env python3
"""Matched Phase 3B continuation trainer with a mask-only replay branch."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import deepspeed
import numpy as np
import torch
import yaml
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.forensics.distributed import BalancedDistributedForensicsSampler
from dataset.forensics.unified import UnifiedForensicsDataset
from model.llava import conversation as conversation_lib
from scripts.phase2a_distributed_train import (
    checkpoint as legacy_checkpoint, configure_train_args, deterministic_tensor_collection_hash,
    ds_config, finite_or_raise, make_loader, rank0_write_json, sync_scalar_numerators, validation,
)
from scripts.phase2a_final_evaluate import file_sha256
from tools.distributed_loss import (
    LOSS_COUNT_KEYS, all_reduce_window_counts, batch_supervision_counts,
    globally_normalized_components, sum_window_counts,
)
from tools.phase3b_replay import build_mask_only_batch, deterministic_replay_selection
from scripts.phase1b_preflight import move_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--optimizer-steps", type=int)
    p.add_argument("--resume")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalized_arm_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["experiment"]["name"] = "<ARM>"
    value["experiment"]["runtime_output_dir"] = "<ARM_OUTPUT>"
    value["experiment"]["experiment_type"] = "<MATCHED_ARM>"
    value["checkpoint"]["output_root"] = "<ARM_CHECKPOINT>"
    value["replay"]["context"] = "<ARM_CONTEXT>"
    return value


def paired_config_audit(config: dict, rank: int) -> None:
    if rank != 0: return
    paths = [ROOT / "configs/phase3b_b0_gold_replay.yaml", ROOT / "configs/phase3b_b1_generated_replay.yaml"]
    configs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths]
    matched = normalized_arm_config(configs[0]) == normalized_arm_config(configs[1])
    result = {
        "status": "PASS" if matched else "FAIL",
        "allowed_differences": ["experiment.name", "experiment.runtime_output_dir", "experiment.experiment_type", "checkpoint.output_root", "replay.context"],
        "b0_config": str(paths[0]), "b1_config": str(paths[1]),
        "b0_sha256": file_sha256(paths[0]), "b1_sha256": file_sha256(paths[1]),
    }
    rank0_write_json(ROOT / "outputs/phase3b_generated_replay/audit/paired_config_audit.json", result, rank)
    if not matched:
        raise RuntimeError("B0/B1 config differs outside preregistered arm fields")


def save_interval(engine, checkpoint_root: Path, step: int, state: dict):
    destination = checkpoint_root / f"step_{step:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(destination), tag="checkpoint", client_state=state,
                           save_latest=True, exclude_frozen_parameters=True)


def main(argv=None):
    cli = parse_args(argv)
    config_path = (ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world_size != 1:
        raise ValueError("Phase 3B matched schedule is preregistered for one GPU per arm")
    paired_config_audit(config, rank)
    local_rank = int(os.environ.get("LOCAL_RANK", cli.local_rank))
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    gas = int(config["training"]["gradient_accumulation_steps"])
    cli.gradient_accumulation_steps = gas
    output_dir = (ROOT / config["experiment"]["runtime_output_dir"]).resolve()
    checkpoint_root = (ROOT / config["checkpoint"]["output_root"]).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_dir / "config.yaml")
        (output_dir / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True), encoding="utf-8")
    torch.distributed.barrier()
    seed = int(config["seed"])
    physical_gpu = int(os.environ.get("PHASE3B_PHYSICAL_GPU", "-1"))
    expected_gpu = 0
    if physical_gpu != expected_gpu:
        raise RuntimeError(f"Phase 3B {config['replay']['context']} arm requires physical GPU {expected_gpu}, got {physical_gpu}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    args = configure_train_args(config, cli, local_rank)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    source_path = Path(config["source"]["checkpoint"]).resolve()
    actual_source_hash = file_sha256(source_path)
    if actual_source_hash != config["source"]["checkpoint_sha256"]:
        raise RuntimeError(f"source checkpoint hash mismatch: {actual_source_hash}")
    source_state = torch.load(source_path, map_location="cpu")
    if int(source_state["optimizer_step"]) != int(config["source"]["optimizer_step"]) or int(source_state["epoch"]) != int(config["source"]["epoch"]):
        raise RuntimeError("source checkpoint step/epoch mismatch")
    missing, unexpected = model.load_state_dict(source_state["module"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected source keys: {unexpected[:10]}")
    model.to(device=device, dtype=torch.bfloat16)
    init_hash = deterministic_tensor_collection_hash(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    rank0_write_json(output_dir / "source_initialization_audit.json", {
        "status": "PASS", "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": actual_source_hash,
        "source_optimizer_step": int(source_state["optimizer_step"]), "source_epoch": int(source_state["epoch"]),
        "missing_frozen_key_count": len(missing), "unexpected_keys": unexpected,
        "trainable_state": init_hash, "fresh_optimizer_for_matched_continuation": True,
        "physical_gpu": physical_gpu,
    }, rank)
    groups = glamm_train.build_optimizer_parameter_groups(model, args)
    optimizer_obj = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0)
    total_steps = int(config["training"]["total_optimizer_steps"])
    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=cli, model=model, optimizer=optimizer_obj,
        config=ds_config(args, total_steps, int(config["training"]["warmup_steps"])),
    )
    train_dataset = UnifiedForensicsDataset(
        ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"], synthscars_root=config["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="phrase_aligned",
    )
    val_dataset = UnifiedForensicsDataset(
        ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="val",
        datasets_root=config["data"]["datasets_root"], synthscars_root=config["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="phrase_aligned",
    )
    sampler = BalancedDistributedForensicsSampler(train_dataset, num_replicas=1, rank=0, seed=seed, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    train_loader = make_loader(train_dataset, sampler, tokenizer, batch_size=args.batch_size, workers=cli.workers)
    val_loader = make_loader(val_dataset, val_sampler, tokenizer, batch_size=2, workers=cli.workers)
    cache_path = (ROOT / config["replay"]["cache"]).resolve()
    cache_rows = read_jsonl(cache_path)
    cache = {row["sample_id"]: row for row in cache_rows}
    if len(cache) != sum(int(r["class_label"]) == 1 for r in train_dataset.rows):
        raise RuntimeError("Replay cache does not cover every train Fake sample")
    cache_hash = file_sha256(cache_path)
    rank0_write_json(output_dir / "replay_cache_audit.json", {
        "status": "PASS", "path": str(cache_path), "sha256": cache_hash,
        "records": len(cache), "eligible": sum(r["replay_eligible"] for r in cache_rows),
    }, rank)
    start_step = 0
    if cli.resume:
        loaded, state = engine.load_checkpoint(cli.resume, tag="checkpoint", load_module_strict=False,
                                               load_optimizer_states=True, load_lr_scheduler_states=True)
        if loaded is None: raise RuntimeError(f"failed resume {cli.resume}")
        start_step = int(state["optimizer_step"])
    final_step = total_steps if cli.optimizer_steps is None else min(total_steps, start_step + cli.optimizer_steps)
    interval = int(config["training"]["validation_interval"])
    current_epoch = start_step // interval; sampler.set_epoch(current_epoch)
    iterator = iter(train_loader); engine.train()
    metrics_path = output_dir / "metrics.jsonl"
    if start_step == 0 and metrics_path.exists(): metrics_path.unlink()
    schedule_path = output_dir / "replay_schedule.jsonl"
    if start_step == 0 and schedule_path.exists(): schedule_path.unlink()
    started = time.time()
    for zero_step in range(start_step, final_step):
        step = zero_step + 1; epoch = zero_step // interval
        if epoch != current_epoch:
            current_epoch = epoch; sampler.set_epoch(epoch); iterator = iter(train_loader)
        window = []
        for _ in range(gas):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            window.append(batch)
        candidates = [sid for batch in window for sid, label in zip(batch["sample_ids"], batch["cls_labels"].tolist())
                      if int(label) == 1 and cache[sid]["replay_eligible"]]
        selected = deterministic_replay_selection(candidates, seed=seed, optimizer_step=step,
                                                  fraction=float(config["replay"]["fraction"]))
        selected_set = set(selected)
        main_counts = [batch_supervision_counts(batch) for batch in window]
        main_global = all_reduce_window_counts(sum_window_counts(main_counts), device)
        aux_batches, aux_counts = [], []
        for batch in window:
            ids = [sid for sid in batch["sample_ids"] if sid in selected_set]
            aux = build_mask_only_batch(
                batch, ids, tokenizer=tokenizer,
                replay_records=None if config["replay"]["context"] == "gold" else cache,
            ) if ids else None
            aux_batches.append(aux)
            aux_counts.append({"text_tokens": 0, "classification_samples": 0, "valid_masks": len(ids)})
        aux_global = all_reduce_window_counts(sum_window_counts(aux_counts), device)
        numerators = {k: 0.0 for k in LOSS_COUNT_KEYS}
        aux_numerators = {k: 0.0 for k in LOSS_COUNT_KEYS}
        for batch, counts, aux, a_counts in zip(window, main_counts, aux_batches, aux_counts):
            batch = move_batch(batch, device, torch.bfloat16)
            output = engine(**batch); finite_or_raise(output, step)
            main_scaled = globally_normalized_components(output, counts, main_global, world_size=1, accumulation_steps=gas)
            for loss_key, count_key in LOSS_COUNT_KEYS.items():
                numerators[loss_key] += float(output[loss_key].detach().float()) * counts[count_key]
            # Backpropagate the original branch before constructing replay so
            # the two large causal graphs never coexist in VRAM.  DeepSpeed
            # steps only at engine.step(), hence gradients still sum exactly.
            engine.backward(main_scaled["loss"])
            if aux is not None:
                aux = move_batch(aux, device, torch.bfloat16)
                positions = {sid: i for i, sid in enumerate(batch["sample_ids"])}
                aux["precomputed_grounding_embeddings"] = output["grounding_image_embeddings"][
                    [positions[sid] for sid in aux["sample_ids"]]
                ]
                aux_output = engine(**aux); finite_or_raise(aux_output, f"{step}-replay")
                if float(aux_output["ce_loss"].detach()) != 0.0 or float(aux_output["cls_loss"].detach()) != 0.0:
                    raise RuntimeError("mask-only replay produced nonzero text/classification loss")
                aux_scaled = globally_normalized_components(aux_output, a_counts, aux_global, world_size=1, accumulation_steps=gas)
                for loss_key, count_key in LOSS_COUNT_KEYS.items():
                    aux_numerators[loss_key] += float(aux_output[loss_key].detach().float()) * a_counts[count_key]
                engine.backward(float(config["replay"]["auxiliary_weight"]) * aux_scaled["loss"])
            engine.step()
        record = {
            "optimizer_step": step, "epoch": epoch + 1,
            "original": {k: numerators[k] / max(1, main_global[c]) for k, c in LOSS_COUNT_KEYS.items()},
            "replay": {k: aux_numerators[k] / max(1, aux_global[c]) for k, c in LOSS_COUNT_KEYS.items()},
            "original_supervision_counts": main_global, "replay_supervision_counts": aux_global,
            "eligible_candidates": len(candidates), "selected_replay_count": len(selected),
            "learning_rates": {g.get("name", str(i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)},
        }
        if rank == 0:
            append_jsonl(metrics_path, record)
            append_jsonl(schedule_path, {"optimizer_step": step, "eligible_sample_ids": candidates,
                                         "selected_sample_ids": selected})
            if step % 25 == 0: print(json.dumps(record), flush=True)
        if step % interval == 0:
            val = validation(engine, val_loader, output_dir, epoch + 1, rank, 1, device)
            state = {"optimizer_step": step, "epoch": epoch + 1, "world_size": 1,
                     "effective_global_batch": 20, "source_checkpoint_sha256": actual_source_hash,
                     "replay_cache_sha256": cache_hash, "arm": config["replay"]["context"],
                     "val_total_loss": val["val_total_loss"],
                     # Compatibility field consumed by the frozen evaluator;
                     # Phase 3B selection does not use this loss.
                     "best_val_total_loss": val["val_total_loss"]}
            save_interval(engine, checkpoint_root, step, state)
            rank0_write_json(output_dir / "training_state.json", {**state, "last_val": val}, rank)
            engine.train()
    rank0_write_json(output_dir / "run_summary.json", {
        "status": "COMPLETE" if final_step == total_steps else "PREFLIGHT_COMPLETE",
        "start_step": start_step, "final_step": final_step, "elapsed_seconds": time.time() - started,
        "source_checkpoint_sha256": actual_source_hash, "initial_trainable_state": init_hash,
        "replay_cache_sha256": cache_hash, "replay_context": config["replay"]["context"],
        "physical_gpu": physical_gpu,
    }, rank)
    torch.distributed.barrier()


if __name__ == "__main__": main()
