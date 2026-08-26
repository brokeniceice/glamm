#!/usr/bin/env python3
"""Strict Fake-only direct spatial-path trainer for Phase 3D.2."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.inference_trace import unwrap_glamm
from model.GLaMM import calculate_dice_loss, compute_sigmoid_cross_entropy
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, make_collate, move_batch
from scripts.phase2a_distributed_train import deterministic_tensor_collection_hash
from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d2 import TRAINABLE_GROUPS, is_spatial_trainable, parameter_group, stable_fake_schedule, validate_optimizer_groups


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d2_direct_spatial_path.yaml")
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--optimizer-steps", type=int, default=None)
    parser.add_argument("--run-kind", choices=("official", "preflight"), default="official")
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()


def group_parameter_records(model):
    output = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        item = output.setdefault(group, {"names": [], "parameter_count": 0, "tensor_count": 0})
        item["names"].append(name); item["parameter_count"] += parameter.numel(); item["tensor_count"] += 1
    return output


def group_hashes(model):
    names = sorted({parameter_group(name) for name, _ in model.named_parameters()})
    return {
        group: deterministic_tensor_collection_hash(
            (name, parameter) for name, parameter in model.named_parameters()
            if parameter_group(name) == group
        )
        for group in names
    }


def configure_boundary(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_spatial_trainable(name)
    groups = []
    for group_name in TRAINABLE_GROUPS:
        params = [parameter for name, parameter in model.named_parameters() if parameter_group(name) == group_name]
        if not params or any(not parameter.requires_grad for parameter in params):
            raise RuntimeError(f"invalid trainable boundary for {group_name}")
        groups.append({"name": group_name, "params": params})
    validate_optimizer_groups(groups)
    registered = {id(parameter) for group in groups for parameter in group["params"]}
    for name, parameter in model.named_parameters():
        if (id(parameter) in registered) != is_spatial_trainable(name):
            raise RuntimeError(f"optimizer registration mismatch: {name}")
    return groups


def spatial_forward(core, batch, bce_weight: float, dice_weight: float):
    """Frozen language/image representation followed by the two trainable spatial modules."""
    with torch.no_grad():
        global_images = core._prepare_global_enc_image(batch["global_enc_images"], batch["offset"])
        prepared_ids, prepared_attention, past, embeds, _ = core.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], batch["attention_masks"], None, None,
            global_images, batch["bboxes"],
        )
        decoder = core.model(
            input_ids=prepared_ids, attention_mask=prepared_attention,
            past_key_values=past, inputs_embeds=embeds, use_cache=False,
            output_attentions=False, output_hidden_states=False, return_dict=True,
        )
        frozen_hidden = decoder[0].detach()
        image_embeddings = core.get_grounding_encoder_embs(batch["grounding_enc_images"]).detach()
    pred_embeddings, positions = core._extract_projected_seg_predictor_hidden(
        frozen_hidden, batch["input_ids"], batch["offset"]
    )
    if any(len(value) != 1 for value in positions):
        raise RuntimeError(f"authoritative training row must contain exactly one [SEG]: {[len(x) for x in positions]}")
    pred_masks = core._generate_and_postprocess_masks(
        pred_embeddings, image_embeddings, batch["resize_list"], batch["label_list"]
    )
    bce = frozen_hidden.new_zeros((), dtype=torch.float32)
    dice = frozen_hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for prediction, target, valid in zip(pred_masks, batch["masks_list"], batch["seg_valid"].tolist()):
        if not valid or target is None or target.numel() == 0:
            raise RuntimeError("Phase 3D.2 training batches must be Fake with a nonempty union mask")
        if prediction.shape[0] != target.shape[0]:
            raise RuntimeError(f"mask count mismatch {prediction.shape[0]} vs {target.shape[0]}")
        n = target.shape[0]
        bce = bce + compute_sigmoid_cross_entropy(prediction, target, n) * n
        dice = dice + calculate_dice_loss(prediction, target, n) * n
        count += n
    bce = bce_weight * bce / count
    dice = dice_weight * dice / count
    return bce + dice, bce, dice


def active_snapshot(model):
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def norm_for_group(model, group, *, gradients=False, before=None):
    squares = []
    for name, parameter in model.named_parameters():
        if parameter_group(name) != group:
            continue
        if gradients:
            if parameter.grad is not None:
                squares.append(parameter.grad.detach().float().pow(2).sum().cpu())
        elif before is not None and name in before:
            delta = parameter.detach().float().cpu() - before[name]
            squares.append(delta.pow(2).sum())
    return math.sqrt(sum(float(value) for value in squares)) if squares else 0.0


def save_checkpoint(model, source_module, optimizer, scheduler, root, step, metadata):
    destination = Path(root) / f"step_{step:04d}" / "checkpoint"
    destination.mkdir(parents=True, exist_ok=True)
    module = dict(source_module)
    named = dict(model.named_parameters())
    for name in list(module):
        if name in named and is_spatial_trainable(name):
            module[name] = named[name].detach().cpu()
    payload = {
        "module": module, "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
        "optimizer_step": step, "epoch": int(metadata["logical_epoch"]), "best_val_total_loss": float("nan"),
        "client_state": metadata,
    }
    path = destination / "mp_rank_00_model_states.pt"
    torch.save(payload, path)
    dump(destination.parent / "metadata.json", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
    return path


def main(argv=None):
    cli = parse_args(argv)
    cfg_path = (ROOT / cli.config).resolve(); cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    expected_gpu = int(cfg["runtime"]["training_gpu"])
    if cli.physical_gpu != expected_gpu:
        raise RuntimeError(f"training GPU is frozen to physical GPU {expected_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(expected_gpu)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES violates frozen GPU assignment")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    out = root / ("training" if cli.run_kind == "official" else "preflight/two_step")
    metrics_path = out / "metrics.jsonl"
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]).resolve()
    if cli.run_kind == "official" and metrics_path.exists() and not cli.resume:
        raise RuntimeError("official metrics already exist; archive or resume explicitly")
    out.mkdir(parents=True, exist_ok=True)

    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    args = build_train_args(model_cfg); args.local_rank = 0; args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    model.gradient_checkpointing_disable()
    source_path = Path(cfg["source"]["checkpoint"]).resolve()
    if file_sha256(source_path) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 source hash mismatch")
    source = torch.load(source_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(source["module"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected P1 keys: {unexpected[:10]}")
    groups = configure_boundary(model)
    lr = float(cfg["optimizer"]["learning_rate"])
    for group in groups: group["lr"] = lr
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(map(float, cfg["optimizer"]["betas"])),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    total_steps = int(cfg["training"]["total_optimizer_steps"]); warmup = int(cfg["optimizer"]["warmup_steps"])
    def lr_lambda(completed):
        if completed < warmup: return float(completed + 1) / max(1, warmup)
        return max(0.0, float(total_steps - completed) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.to(device=device, dtype=torch.bfloat16).eval()
    records = group_parameter_records(model); before_hash = group_hashes(model); baseline_active = active_snapshot(model)
    dump(out / "initialization_audit.json", {
        "status": "PASS", "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": file_sha256(source_path), "missing_frozen_key_count": len(missing),
        "unexpected_keys": unexpected, "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_groups": records, "before_hashes": before_hash,
        "optimizer_groups": [group["name"] for group in optimizer.param_groups],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })

    start_step = 0
    if cli.resume:
        resume = torch.load(Path(cli.resume), map_location="cpu")
        model.load_state_dict(resume["module"], strict=False); optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["lr_scheduler"]); start_step = int(resume["optimizer_step"])
    requested = total_steps if cli.optimizer_steps is None else start_step + int(cli.optimizer_steps)
    final_step = min(total_steps, requested)
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="phrase_aligned",
    )
    exposures = total_steps * int(cfg["training"]["effective_batch_size"])
    schedule = stable_fake_schedule(dataset.rows, seed, exposures)
    dump(out / "schedule.json", {"sample_ids": [dataset.rows[i]["sample_id"] for i in schedule], "unique": True})
    collate = make_collate(tokenizer); core = unwrap_glamm(model)
    bce_weight = float(cfg["loss"]["mask_bce_weight"]); dice_weight = float(cfg["loss"]["mask_dice_weight"])
    audit_steps = set(map(int, cfg["training"]["gradient_audit_steps"])); gradient_audit = []
    started = time.time(); torch.cuda.reset_peak_memory_stats(device)
    for zero_step in range(start_step, final_step):
        step = zero_step + 1; step_started = time.time()
        begin = zero_step * int(cfg["training"]["effective_batch_size"])
        indices = schedule[begin:begin + int(cfg["training"]["effective_batch_size"])]
        samples = [dataset[index] for index in indices]
        if any(int(sample["cls_label"]) != 1 or not sample["seg_valid"] for sample in samples):
            raise RuntimeError("non-Fake sample entered spatial loss")
        batch = move_batch(collate(samples), device, torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        loss, bce, dice = spatial_forward(core, batch, bce_weight, dice_weight)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite spatial loss")
        snapshot = active_snapshot(model) if step in audit_steps else None
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], float(cfg["training"]["gradient_clip_norm"])
        ))
        audit_record = None
        if step in audit_steps:
            audit_record = {"optimizer_step": step, "modules": {}}
            for group in records:
                audit_record["modules"][group] = {
                    "grad_present": any(p.grad is not None for n, p in model.named_parameters() if parameter_group(n) == group),
                    "grad_norm": norm_for_group(model, group, gradients=True),
                    "update_norm": 0.0,
                }
        optimizer.step(); scheduler.step()
        if audit_record is not None:
            for group in records:
                audit_record["modules"][group]["update_norm"] = norm_for_group(model, group, before=snapshot)
            gradient_audit.append(audit_record); dump(out / "gradient_audit.json", {"status": "IN_PROGRESS", "records": gradient_audit})
        record = {
            "optimizer_step": step, "sample_ids": [sample["sample_id"] for sample in samples],
            "spatial_loss": float(loss.detach().float().cpu()), "mask_bce_loss": float(bce.detach().float().cpu()),
            "mask_dice_loss": float(dice.detach().float().cpu()), "gradient_norm_before_clip": grad_norm,
            "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
            "elapsed_seconds": time.time() - step_started, "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        append(metrics_path, record); print(json.dumps(record), flush=True)
        if cli.run_kind == "official" and step % int(cfg["training"]["checkpoint_interval"]) == 0:
            metadata = {"arm": "P3D2-SPATIAL", "optimizer_step": step,
                        "logical_epoch": step // int(cfg["training"]["checkpoint_interval"]),
                        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                        "only_trainable_groups": list(TRAINABLE_GROUPS), "physical_gpu": cli.physical_gpu}
            path = save_checkpoint(model, source["module"], optimizer, scheduler, checkpoint_root, step, metadata)
            append(root / "checkpoint_metadata.jsonl", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
    after_hash = group_hashes(model)
    parameter_audit = []
    optimizer_names = {group["name"] for group in optimizer.param_groups}
    for group, before in before_hash.items():
        changed = before["sha256"] != after_hash[group]["sha256"]
        allowed = group in TRAINABLE_GROUPS
        parameter_audit.append({
            "module_name": group, "before_sha256": before["sha256"], "after_sha256": after_hash[group]["sha256"],
            "changed": changed, "requires_grad": allowed, "optimizer_registered": group in optimizer_names,
            "gradient_seen": any(item["modules"][group]["grad_present"] for item in gradient_audit),
            "parameter_delta_norm": norm_for_group(model, group, before=baseline_active) if allowed else 0.0,
        })
        if not allowed and changed:
            raise RuntimeError(f"frozen parameter group changed: {group}")
    dump(out / "parameter_invariance_audit.json", {"status": "PASS", "modules": parameter_audit})
    dump(out / "gradient_audit.json", {"status": "PASS" if final_step == total_steps else "PREFLIGHT_PASS", "records": gradient_audit})
    dump(out / "run_summary.json", {
        "status": "COMPLETE" if cli.run_kind == "official" and final_step == total_steps else "PREFLIGHT_COMPLETE",
        "start_step": start_step, "final_step": final_step, "total_optimizer_steps": total_steps,
        "fake_image_exposures": (final_step - start_step) * int(cfg["training"]["effective_batch_size"]),
        "elapsed_seconds": time.time() - started, "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
    })


if __name__ == "__main__":
    main()
