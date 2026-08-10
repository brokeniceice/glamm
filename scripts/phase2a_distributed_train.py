#!/usr/bin/env python3
"""Config-driven DeepSpeed runner for the finalized Phase 2A baseline."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.distributed import BalancedDistributedForensicsSampler
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    UnifiedForensicsDataset,
)
from eval.forensics import (
    build_forensics_prediction_records,
    compute_binary_mask_metrics,
    compute_empty_prediction_metrics,
    summarize_detection_records,
    summarize_localization_records,
)
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, move_batch
from tools.distributed_loss import (
    LOSS_COUNT_KEYS,
    all_reduce_window_counts,
    batch_supervision_counts,
    globally_normalized_components,
    sum_window_counts,
)


OUTPUT_ROOT = REPO_ROOT / "outputs/phase2a_unified_baseline"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a_unified_baseline_full.yaml")
    parser.add_argument("--mode", choices=("smoke", "train", "resume-check"), required=True)
    parser.add_argument("--optimizer-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-smoke-checkpoint", action="store_true")
    parser.add_argument("--worst-case", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args(argv)


def rank0_write_json(path, value, rank):
    if rank != 0:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure_train_args(config, cli, local_rank):
    args = build_train_args(config)
    args.local_rank = local_rank
    args.freeze_region_encoder = True
    optimizer = config["optimizer"]
    args.lr = float(optimizer["lora_lr"])
    for key in (
        "lora_lr", "classification_head_lr", "text_hidden_fcs_lr", "mask_decoder_lr",
        "embedding_lr", "lm_head_lr",
    ):
        setattr(args, key, float(optimizer[key]))
    args.beta1, args.beta2 = map(float, optimizer["betas"])
    args.batch_size = int(config["training"]["batch_size_per_device"])
    args.grad_accumulation_steps = int(cli.gradient_accumulation_steps)
    args.epochs = int(config["training"]["epochs"])
    args.steps_per_epoch = int(config["training"]["optimizer_steps_per_epoch"])
    args.use_forensics_data = True
    args.forensics_manifest_dir = config["data"]["manifest_dir"]
    args.forensics_datasets_root = config["data"]["datasets_root"]
    args.synthscars_root = config["data"]["synthscars_root"]
    args.workers = int(cli.workers)
    args.no_eval = False
    return args


def ds_config(args, total_steps, warmup_steps):
    return {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "steps_per_print": 1000000,
        # The optimizer object is supplied explicitly below.  Using DeepSpeed's
        # string-based AdamW entry selects FusedAdam and requires a local C++
        # toolchain/ninja even though Phase 1D preregistered ordinary AdamW.
        "zero_allow_untested_optimizer": True,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps, "warmup_min_lr": 0.0,
                "warmup_max_lr": args.lr, "warmup_num_steps": warmup_steps,
                "warmup_type": "linear",
            },
        },
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2, "contiguous_gradients": True, "overlap_comm": True,
            "reduce_scatter": True, "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8,
        },
    }


def make_loader(dataset, sampler, tokenizer, *, batch_size, workers, inference=False):
    kwargs = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": lambda items: custom_collate_fn(
            items, tokenizer=tokenizer, use_mm_start_end=True,
            inference=inference, token_strategy="fixed_cls_query",
        ),
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def worst_case_rank_indices(dataset, tokenizer, *, rank, world_size, required_pairs):
    fake = [
        (len(tokenizer(" ".join(str(row.get("explanation") or "").split()),
                       add_special_tokens=False).input_ids), index)
        for index, row in enumerate(dataset.rows) if int(row["class_label"]) == 1
    ]
    fake.sort(reverse=True)
    real = [index for index, row in enumerate(dataset.rows) if int(row["class_label"]) == 0]
    output = []
    for pair_index in range(required_pairs):
        global_index = pair_index * world_size + rank
        output.extend([real[global_index % len(real)], fake[global_index % len(fake)][1]])
    return output


def selected_parameter_views(model, tokenizer):
    named = dict(model.named_parameters())
    selectors = {
        "classification_head": next(name for name in named if "classification_head.weight" in name),
        "lora": next(name for name in named if "lora_A" in name),
        "text_hidden_fcs": next(name for name in named if "text_hidden_fcs" in name and name.endswith("weight")),
        "mask_decoder": next(
            name for name in named if "grounding_encoder.mask_decoder" in name and name.endswith("weight")
        ),
        "embeddings": next(name for name in named if "embed_tokens.weight" in name),
        "lm_head": next(name for name in named if "lm_head.weight" in name),
    }
    token_ids = [
        tokenizer(token, add_special_tokens=False).input_ids[0]
        for token in ("[CLS]", "[REAL]", "[FAKE]", "[SEG]")
    ]
    output = {}
    for group, name in selectors.items():
        tensor = named[name].detach()
        if group in {"embeddings", "lm_head"}:
            tensor = tensor[token_ids]
        else:
            tensor = tensor.reshape(-1)[:4096]
        output[group] = {
            "name": name, "values": tensor.float().cpu().clone(),
            "row_ids": token_ids if group in {"embeddings", "lm_head"} else None,
        }
    return output


def parameter_delta_and_sync(model, initial, rank, world_size):
    named = dict(model.named_parameters())
    records = {}
    for group, item in initial.items():
        current = named[item["name"]].detach()
        if group in {"embeddings", "lm_head"}:
            current = current[item["row_ids"]]
        else:
            current = current.reshape(-1)[:item["values"].numel()]
        current = current.float().cpu()
        delta = current - item["values"]
        flat = current.reshape(-1).to(torch.device("cuda", torch.cuda.current_device()))
        gathered = [torch.empty_like(flat) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, flat)
        max_rank_diff = max(float((gathered[0] - value).abs().max()) for value in gathered)
        records[group] = {
            "parameter_name": item["name"],
            "delta_l2": float(delta.norm()),
            "delta_max_abs": float(delta.abs().max()),
            "rank_parameter_max_abs_diff": max_rank_diff,
            "value_sha256": hashlib.sha256(current.numpy().tobytes()).hexdigest(),
        }
    return records


def sync_scalar_numerators(local_numerators, device):
    keys = ("ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss")
    tensor = torch.tensor([local_numerators[key] for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return dict(zip(keys, tensor.tolist()))


def finite_or_raise(output, step):
    for key in ("loss", "ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss"):
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError(f"Non-finite {key} at optimizer step {step}")
    if int(output["seg_missing_pred_count"]) or int(output["seg_count_mismatch_count"]):
        raise RuntimeError(f"Required Fake segmentation prediction missing at optimizer step {step}")


def validation(engine, loader, output_dir, epoch, rank, world_size, device):
    engine.eval()
    local_numerators = {key: 0.0 for key in LOSS_COUNT_KEYS}
    local_counts = {key: 0 for key in ("text_tokens", "classification_samples", "valid_masks")}
    detection_records, tf_records = [], []
    with torch.no_grad():
        for batch in loader:
            counts = batch_supervision_counts(batch)
            batch = move_batch(batch, device, torch.bfloat16)
            output = engine(**batch)
            finite_or_raise(output, f"validation-{epoch}")
            for loss_key, count_key in LOSS_COUNT_KEYS.items():
                local_numerators[loss_key] += float(output[loss_key].detach().float()) * counts[count_key]
            for key in local_counts:
                local_counts[key] += counts[key]
            detection_records.extend(build_forensics_prediction_records(batch, output))
            for index, is_valid in enumerate(batch["seg_valid"].detach().cpu().tolist()):
                if not is_valid:
                    continue
                pred_mask, gt_mask = output["pred_masks"][index], batch["masks_list"][index]
                has_prediction = pred_mask is not None and pred_mask.shape[0] > 0
                metrics = (
                    compute_binary_mask_metrics(pred_mask, gt_mask)
                    if has_prediction else compute_empty_prediction_metrics(gt_mask)
                )
                tf_records.append({
                    **metrics, "content_type": batch["content_categories"][index],
                    "seg_triggered": has_prediction, "has_pred_mask": has_prediction,
                })
    global_counts = all_reduce_window_counts(local_counts, device)
    global_numerators = sync_scalar_numerators(local_numerators, device)
    components = {
        loss_key: global_numerators[loss_key] / max(1, global_counts[count_key])
        for loss_key, count_key in LOSS_COUNT_KEYS.items()
    }
    gathered_detection = [None] * world_size; gathered_tf = [None] * world_size
    torch.distributed.all_gather_object(gathered_detection, detection_records)
    torch.distributed.all_gather_object(gathered_tf, tf_records)
    if rank == 0:
        detection_records = [row for rows in gathered_detection for row in rows]
        tf_records = [row for rows in gathered_tf for row in rows]
        detection_metrics = summarize_detection_records(detection_records)
        tf_metrics = summarize_localization_records(tf_records, autoregressive=False)
        metrics = {
            "epoch": epoch, "val_total_loss": sum(components.values()),
            **{f"val_{key}": value for key, value in components.items()},
            "classification": detection_metrics["classification_head"],
            "lm_verdict": detection_metrics["lm_verdict"],
            "cls_lm_agreement": detection_metrics["cls_lm_agreement"],
            "tf_full_context": tf_metrics,
            "global_supervision_counts": global_counts,
        }
        epoch_dir = output_dir / "validation" / f"epoch_{epoch:02d}"
        rank0_write_json(epoch_dir / "metrics.json", metrics, rank)
        with (epoch_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in detection_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        metrics = None
    payload = [metrics]
    torch.distributed.broadcast_object_list(payload, src=0)
    engine.train()
    return payload[0]


def checkpoint(engine, checkpoint_root, *, best, client_state):
    destination = checkpoint_root / ("best" if best else "last")
    destination.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(
        str(destination), tag="checkpoint", client_state=client_state,
        save_latest=True, exclude_frozen_parameters=True,
    )


def main(argv=None):
    cli = parse_args(argv)
    config_path = (REPO_ROOT / cli.config).resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if cli.micro_batch_size is not None:
        config["training"]["batch_size_per_device"] = int(cli.micro_batch_size)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", cli.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    micro_batch = int(config["training"]["batch_size_per_device"])
    target_global_batch = 20
    denominator = micro_batch * world_size
    if target_global_batch % denominator:
        raise ValueError(
            f"Cannot preserve effective global batch {target_global_batch} with "
            f"micro_batch={micro_batch}, world_size={world_size}"
        )
    expected_gas = target_global_batch // denominator
    gas = expected_gas if cli.gradient_accumulation_steps is None else cli.gradient_accumulation_steps
    if micro_batch * world_size * gas != target_global_batch:
        raise ValueError(
            f"Effective global batch changed: {micro_batch}*{world_size}*{gas}"
        )
    cli.gradient_accumulation_steps = gas
    output_dir = OUTPUT_ROOT / (cli.output_subdir or ("preflight/dual" if world_size == 2 else "preflight/single"))
    if cli.mode == "train":
        output_dir = OUTPUT_ROOT
    configured_checkpoint_root = cli.checkpoint_root or config["checkpoint"]["output_root"]
    checkpoint_root = Path(configured_checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    checkpoint_root = checkpoint_root.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_dir / "config.yaml")
        (output_dir / "git_commit.txt").write_text(
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                           capture_output=True, check=True).stdout,
            encoding="utf-8",
        )
    torch.distributed.barrier()

    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    args = configure_train_args(config, cli, local_rank)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    if cli.disable_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.to(device=device, dtype=torch.bfloat16)
    groups = glamm_train.build_optimizer_parameter_groups(model, args)
    total_budget = int(config["training"]["total_optimizer_steps"])
    torch_optimizer = torch.optim.AdamW(
        groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0
    )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=cli, model=model, optimizer=torch_optimizer,
        config=ds_config(args, total_budget, int(config["training"]["warmup_steps"])),
    )
    initial_parameters = selected_parameter_views(engine.module, tokenizer)

    train_dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=args.image_size,
    )
    val_dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="val",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=args.image_size,
    )
    train_sampler = (
        FixedIndexSampler(worst_case_rank_indices(
            train_dataset, tokenizer, rank=rank, world_size=world_size,
            required_pairs=max(
                1, math.ceil((cli.optimizer_steps or 1) * gas * args.batch_size / 2)
            ),
        ))
        if cli.worst_case else
        BalancedDistributedForensicsSampler(
            train_dataset, num_replicas=world_size, rank=rank, seed=seed, shuffle=True
        )
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    train_loader = make_loader(
        train_dataset, train_sampler, tokenizer, batch_size=args.batch_size, workers=cli.workers
    )
    val_loader = make_loader(val_dataset, val_sampler, tokenizer, batch_size=2, workers=cli.workers)

    start_step, best_val = 0, float("inf")
    if cli.resume:
        load_path, state = engine.load_checkpoint(
            cli.resume, tag="checkpoint", load_module_strict=False,
            load_optimizer_states=True, load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(f"Failed to resume from {cli.resume}")
        start_step = int(state["optimizer_step"])
        best_val = float(state.get("best_val_total_loss", float("inf")))
        rank0_write_json(output_dir / "resume_audit.json", {
            "status": "passed", "load_path": load_path, "optimizer_step": start_step,
            "world_size": world_size,
        }, rank)
        if cli.mode == "resume-check":
            torch.distributed.barrier()
            return
    elif cli.mode == "resume-check":
        raise ValueError("resume-check requires --resume")

    requested_steps = cli.optimizer_steps
    final_step = total_budget if requested_steps is None else start_step + requested_steps
    if cli.mode == "train":
        final_step = total_budget
    if final_step > total_budget and cli.mode == "train":
        raise ValueError("Training would exceed preregistered optimizer-step budget")
    steps_per_epoch = int(config["training"]["optimizer_steps_per_epoch"])
    current_epoch = start_step // steps_per_epoch
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(current_epoch)
    train_iterator = iter(train_loader)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("a" if start_step else "w", encoding="utf-8") if rank == 0 else None
    audit_steps = []
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    cumulative_data = cumulative_forward = cumulative_backward = cumulative_optimizer = 0.0
    engine.train()

    for optimizer_step in range(start_step, final_step):
        epoch = optimizer_step // steps_per_epoch
        if epoch != current_epoch:
            current_epoch = epoch
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            train_iterator = iter(train_loader)
        window, counts, sample_rows = [], [], []
        data_started = time.perf_counter()
        for _ in range(gas):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader); batch = next(train_iterator)
            window.append(batch); counts.append(batch_supervision_counts(batch))
            sample_rows.append([
                {"sample_id": sid, "image_path": path, "gt_label": int(label),
                 "content_type": content, "source": source}
                for sid, path, label, content, source in zip(
                    batch["sample_ids"], batch["image_paths"], batch["cls_labels"].tolist(),
                    batch["content_categories"], batch["sources"],
                )
            ])
        cumulative_data += time.perf_counter() - data_started
        global_counts = all_reduce_window_counts(sum_window_counts(counts), device)
        local_numerators = {key: 0.0 for key in LOSS_COUNT_KEYS}
        step_start = time.perf_counter()
        for micro_index, (batch, local_counts) in enumerate(zip(window, counts)):
            batch = move_batch(batch, device, torch.bfloat16)
            if cli.profile_timing:
                torch.cuda.synchronize(device)
            started = time.perf_counter(); output = engine(**batch)
            if cli.profile_timing:
                torch.cuda.synchronize(device)
            cumulative_forward += time.perf_counter() - started
            finite_or_raise(output, optimizer_step + 1)
            scaled = globally_normalized_components(
                output, local_counts, global_counts,
                world_size=world_size, accumulation_steps=gas,
            )
            for loss_key, count_key in LOSS_COUNT_KEYS.items():
                local_numerators[loss_key] += (
                    float(output[loss_key].detach().float()) * local_counts[count_key]
                )
            started = time.perf_counter(); engine.backward(scaled["loss"])
            if cli.profile_timing:
                torch.cuda.synchronize(device)
            cumulative_backward += time.perf_counter() - started
            started = time.perf_counter(); engine.step()
            if cli.profile_timing:
                torch.cuda.synchronize(device)
            cumulative_optimizer += time.perf_counter() - started
        global_numerators = sync_scalar_numerators(local_numerators, device)
        component_values = {
            loss_key: global_numerators[loss_key] / max(1, global_counts[count_key])
            for loss_key, count_key in LOSS_COUNT_KEYS.items()
        }
        record = {
            "optimizer_step": optimizer_step + 1, "epoch": epoch,
            "total_loss": sum(component_values.values()),
            "text_loss": component_values["ce_loss"], "cls_loss": component_values["cls_loss"],
            "mask_bce_loss": component_values["mask_bce_loss"],
            "mask_dice_loss": component_values["mask_dice_loss"],
            "global_supervision_counts": global_counts,
            "learning_rates": {group.get("name", str(i)): float(group["lr"])
                               for i, group in enumerate(optimizer.param_groups)},
            "seconds_this_optimizer_step": time.perf_counter() - step_start,
        }
        if rank == 0:
            metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n"); metrics_handle.flush()
            if (optimizer_step + 1) % (10 if cli.mode == "smoke" else 25) == 0:
                print(json.dumps(record), flush=True)
        if len(audit_steps) < 20:
            audit_steps.append({"optimizer_step": optimizer_step + 1, "micro_batches": sample_rows})

        if cli.mode == "train" and (optimizer_step + 1) % steps_per_epoch == 0:
            val_metrics = validation(
                engine, val_loader, output_dir, epoch + 1, rank, world_size, device
            )
            is_best = float(val_metrics["val_total_loss"]) < best_val
            best_val = min(best_val, float(val_metrics["val_total_loss"]))
            state = {
                "optimizer_step": optimizer_step + 1, "epoch": epoch + 1,
                "best_val_total_loss": best_val, "world_size": world_size,
                "effective_global_batch": 20,
            }
            checkpoint(engine, checkpoint_root, best=False, client_state=state)
            if is_best:
                checkpoint(engine, checkpoint_root, best=True, client_state=state)
            rank0_write_json(output_dir / "training_state.json", {**state, "last_val": val_metrics}, rank)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - run_start
    parameter_audit = parameter_delta_and_sync(engine.module, initial_parameters, rank, world_size)
    local_summary = {
        "rank": rank, "world_size": world_size, "optimizer_steps": final_step - start_step,
        "sample_exposures": (final_step - start_step) * gas * args.batch_size,
        "elapsed_seconds": elapsed, "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "data_loading_seconds": cumulative_data, "forward_seconds": cumulative_forward,
        "backward_seconds": cumulative_backward, "optimizer_seconds": cumulative_optimizer,
        "parameter_sync": parameter_audit, "sample_audit": audit_steps,
    }
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_summary)
    if rank == 0:
        global_samples = (final_step - start_step) * 20
        summary = {
            "mode": cli.mode, "world_size": world_size, "per_device_batch": args.batch_size,
            "gradient_accumulation_steps": gas, "effective_global_batch": 20,
            "optimizer_steps": final_step - start_step, "global_sample_exposures": global_samples,
            "elapsed_seconds": elapsed, "samples_per_second": global_samples / elapsed,
            "optimizer_steps_per_second": (final_step - start_step) / elapsed,
            "seconds_per_optimizer_step": elapsed / max(1, final_step - start_step),
            "ranks": gathered,
            "all_parameter_groups_synchronized": all(
                item["rank_parameter_max_abs_diff"] == 0.0
                for item in parameter_audit.values()
            ),
        }
        rank0_write_json(output_dir / "run_summary.json", summary, rank)
        sampler_records = {f"rank_{item['rank']}": item["sample_audit"] for item in gathered}
        rank0_write_json(output_dir / "distributed_sampler_runtime.json", sampler_records, rank)
    if metrics_handle is not None:
        metrics_handle.close()
    if cli.mode == "smoke" and cli.save_smoke_checkpoint:
        checkpoint(engine, checkpoint_root, best=False, client_state={
            "optimizer_step": final_step, "epoch": current_epoch,
            "best_val_total_loss": best_val, "world_size": world_size,
        })
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
