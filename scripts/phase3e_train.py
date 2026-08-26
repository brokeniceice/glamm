#!/usr/bin/env python3
"""Matched Phase 3E continuation/JOINТ trainer with mandatory gradient audit."""

from __future__ import annotations

import argparse
import hashlib
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
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION, UnifiedForensicsDataset
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, move_batch
from tools.distributed_loss import batch_supervision_counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3e_joint_language_mask_posttraining.yaml")
    parser.add_argument("--arm", choices=("SFT_CONT", "JOINT"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=None, help="Bounded engineering run; never expands budget")
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if "lora_" in name: return "lora"
    if "text_hidden_fcs" in name: return "text_hidden_fcs"
    if "grounding_encoder.mask_decoder" in name: return "mask_decoder"
    return "frozen"


def configure_boundary(model, arm: str) -> dict:
    allowed = {"lora"} if arm == "SFT_CONT" else {"lora", "text_hidden_fcs", "mask_decoder"}
    counts = {key: 0 for key in ("total", "trainable", "frozen", "lora", "text_hidden_fcs", "mask_decoder")}
    tensors = {key: 0 for key in ("lora", "text_hidden_fcs", "mask_decoder")}
    names = {key: [] for key in ("lora", "text_hidden_fcs", "mask_decoder")}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        parameter.requires_grad = group in allowed
        counts["total"] += parameter.numel()
        counts["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
        if group != "frozen":
            counts[group] += parameter.numel(); tensors[group] += 1; names[group].append(name)
    if not all(counts[group] > 0 for group in allowed):
        raise RuntimeError(f"empty allowed parameter group: {counts}")
    if arm == "SFT_CONT" and (any(p.requires_grad for n, p in model.named_parameters() if parameter_group(n) != "lora")):
        raise RuntimeError("SFT boundary leak")
    return {"allowed": sorted(allowed), "parameter_counts": counts, "tensor_counts": tensors,
            "lora_tensor_names": names["lora"]}


def grad_norms(model) -> dict:
    squares = {key: [] for key in ("lora", "text_hidden_fcs", "mask_decoder")}
    present = {key: 0 for key in squares}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group in squares and parameter.grad is not None:
            present[group] += 1; squares[group].append(parameter.grad.detach().float().pow(2).sum())
    values = {key: float(torch.stack(items).sum().sqrt()) if items else 0.0 for key, items in squares.items()}
    values["total"] = math.sqrt(sum(value * value for value in values.values()))
    return {"norms": values, "gradient_tensor_counts": present}


def tensor_hash(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def group_hashes(model) -> dict:
    result = {}
    for group in ("lora", "text_hidden_fcs", "mask_decoder", "frozen"):
        result[group] = tensor_hash((n, p) for n, p in model.named_parameters() if parameter_group(n) == group)
    return result


def make_cpu_batch(dataset, index: int, tokenizer, *, include_spatial: bool):
    sample = dataset[index]
    batch = custom_collate_fn([sample], tokenizer=tokenizer, use_mm_start_end=True,
                              inference=False, token_strategy="fixed_cls_query")
    if not include_spatial:
        batch["grounding_enc_images"] = None
        # Language-only CE intentionally has no segmentation objective.  Marking
        # this false avoids a misleading missing-mask diagnostic from the shared
        # loss helper while leaving the tokenized causal trajectory unchanged.
        batch["masks_list"] = [None]
        batch["seg_valid"] = torch.tensor([False])
    return batch, sample


def make_batch(dataset, index: int, tokenizer, device, dtype, *, include_spatial: bool):
    batch, sample = make_cpu_batch(dataset, index, tokenizer, include_spatial=include_spatial)
    return move_batch(batch, device, dtype), sample


def independent_backward(model, dataset, fake_index, tokenizer, device, dtype, component: str) -> dict:
    model.zero_grad(set_to_none=True)
    # Fixed seed makes all audit trajectories and both arms reproducible.
    torch.manual_seed(99173); torch.cuda.manual_seed_all(99173)
    batch, sample = make_batch(dataset, fake_index, tokenizer, device, dtype, include_spatial=component != "language")
    output = model(**batch)
    language = output["ce_loss"]
    mask = output["mask_loss"]
    loss = {"language": language, "mask": mask, "joint": language + mask}[component]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite {component} audit loss")
    loss.backward()
    result = {"component": component, "sample_id": sample["sample_id"],
              "language_ce": float(language.detach().float()), "mask_loss": float(mask.detach().float()),
              **grad_norms(model)}
    model.zero_grad(set_to_none=True)
    del output, loss, batch
    torch.cuda.empty_cache()
    return result


def run_gradient_audit(model, dataset, tokenizer, device, dtype, output_root: Path) -> dict:
    fake_index = next(i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 1)
    language = independent_backward(model, dataset, fake_index, tokenizer, device, dtype, "language")
    mask = independent_backward(model, dataset, fake_index, tokenizer, device, dtype, "mask")
    joint = independent_backward(model, dataset, fake_index, tokenizer, device, dtype, "joint")
    checks = {
        "language_to_lora_nonzero": language["norms"]["lora"] > 0,
        "language_to_fc_zero": language["norms"]["text_hidden_fcs"] == 0,
        "language_to_decoder_zero": language["norms"]["mask_decoder"] == 0,
        "mask_to_lora_nonzero": mask["norms"]["lora"] > 0,
        "mask_to_fc_nonzero": mask["norms"]["text_hidden_fcs"] > 0,
        "mask_to_decoder_nonzero": mask["norms"]["mask_decoder"] > 0,
        "joint_lora_nonzero": joint["norms"]["lora"] > 0,
        "joint_fc_nonzero": joint["norms"]["text_hidden_fcs"] > 0,
        "joint_decoder_nonzero": joint["norms"]["mask_decoder"] > 0,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "gate_on_failure": "GATE_JOINT_GRADIENT_PATH_BROKEN",
              "fixed_fake_index": fake_index, "audit_A_language_only": language, "audit_B_mask_only": mask,
              "audit_C_joint": joint, "separate_language_gradient_norms": language["norms"],
              "separate_mask_gradient_norms": mask["norms"], "checks": checks,
              "lambda_was_not_changed": True}
    dump(output_root / "audits/gradient_path_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("GATE_JOINT_GRADIENT_PATH_BROKEN")
    return result


def save_checkpoint(model, source_module, optimizer, scheduler, root: Path, step: int, metadata: dict) -> Path:
    destination = root / f"step_{step:04d}" / "checkpoint"
    destination.mkdir(parents=True, exist_ok=True)
    module = dict(source_module)
    for name, parameter in model.named_parameters():
        if parameter_group(name) in metadata["allowed_parameter_groups"]:
            module[name] = parameter.detach().cpu().clone()
    payload = {"module": module, "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
               "optimizer_step": step, "epoch": step // 100, "best_val_total_loss": float("nan"),
               "client_state": metadata}
    path = destination / "mp_rank_00_model_states.pt"
    torch.save(payload, path)
    dump(destination.parent / "metadata.json", {**metadata, "checkpoint": str(path), "sha256": sha256(path)})
    return path


def main(argv=None):
    cli = parse_args(argv)
    cfg_path = (ROOT / cli.config).resolve(); cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    expected_gpu = int(cfg["runtime"][f"{cli.arm}_gpu"])
    if cli.physical_gpu != expected_gpu:
        raise RuntimeError(f"{cli.arm} must use physical GPU {expected_gpu}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(cli.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}, expected {cli.physical_gpu}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    out_root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    arm_out = out_root / "experiments" / cli.arm
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]).resolve() / cli.arm
    arm_out.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    source_path = Path(cfg["source"]["checkpoint"]).resolve()
    if sha256(source_path) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checksum mismatch")
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    args = build_train_args(model_cfg); args.local_rank = 0; args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    source = torch.load(source_path, map_location="cpu")
    if int(source["optimizer_step"]) != int(cfg["source"]["optimizer_step"]): raise RuntimeError("P1 step mismatch")
    missing, unexpected = model.load_state_dict(source["module"], strict=False)
    if unexpected: raise RuntimeError(f"unexpected P1 keys: {unexpected[:10]}")
    boundary = configure_boundary(model, cli.arm)
    # PEFT + gradient checkpointing needs gradients on the embedding *outputs*
    # so LoRA receives backward signals when the embedding weight itself is
    # frozen.  This hook does not make token embeddings trainable.
    model.enable_input_require_grads()
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input-gradient hook unexpectedly unfroze token embeddings")
    model.to(device=device, dtype=torch.bfloat16); model.train()
    initial_hashes = group_hashes(model)
    # Fill the previously unknown live-model counts without changing the frozen protocol.
    manifest_path = out_root / "initial_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"total_parameters": boundary["parameter_counts"]["total"],
                     "frozen_parameters": boundary["parameter_counts"]["frozen"],
                     f"{cli.arm}_trainable_parameters": boundary["parameter_counts"]["trainable"]})
    dump(manifest_path, manifest)
    schedule = json.loads((out_root / "training/schedule.json").read_text(encoding="utf-8"))
    dataset = UnifiedForensicsDataset(ROOT / cfg["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="phrase_aligned")
    if [dataset.rows[i]["sample_id"] for i in schedule["indices"]] != schedule["sample_ids"]:
        raise RuntimeError("frozen schedule no longer resolves exactly")
    if CANONICAL_UNIFIED_QUESTION != cfg["prompt"]["user_question"]:
        raise RuntimeError("canonical prompt mismatch")

    if cli.arm == "JOINT":
        run_gradient_audit(model, dataset, tokenizer, device, torch.bfloat16, out_root)
    elif not (out_root / "audits/gradient_path_audit.json").is_file():
        # The hard joint audit must precede either formal arm; run it by temporarily enabling the spatial groups.
        configure_boundary(model, "JOINT")
        run_gradient_audit(model, dataset, tokenizer, device, torch.bfloat16, out_root)
        boundary = configure_boundary(model, cli.arm)
    if cli.audit_only:
        dump(arm_out / "audit_only_summary.json", {"status": "PASS", "arm": cli.arm, "boundary": boundary,
             "initial_hashes": initial_hashes, "formal_optimizer_updates": 0})
        print(json.dumps({"status": "PASS", "arm": cli.arm, "audit_only": True, "boundary": boundary}, indent=2)); return

    groups = []
    lrs = {"lora": float(cfg["optimizer"]["lora_lr"]), "text_hidden_fcs": float(cfg["optimizer"]["text_hidden_fcs_lr"]),
           "mask_decoder": float(cfg["optimizer"]["mask_decoder_lr"])}
    for group in boundary["allowed"]:
        params = [p for n, p in model.named_parameters() if p.requires_grad and parameter_group(n) == group]
        groups.append({"name": group, "params": params, "lr": lrs[group]})
    optimizer = torch.optim.AdamW(groups, betas=tuple(map(float, cfg["optimizer"]["betas"])),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    total_steps = int(cfg["training"]["total_optimizer_steps"]); warmup = int(cfg["optimizer"]["warmup_steps"])
    def lr_lambda(completed):
        if completed < warmup: return float(completed + 1) / max(1, warmup)
        return max(0.0, float(total_steps - completed) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if cli.resume:
        state = torch.load(Path(cli.resume).resolve(), map_location="cpu")
        model.load_state_dict(state["module"], strict=False); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"]); start_step = int(state["optimizer_step"])
    elif (arm_out / "training_metrics.jsonl").exists():
        raise RuntimeError("existing formal metrics require --resume; no overwrite allowed")
    final_step = total_steps if cli.optimizer_steps is None else min(total_steps, start_step + cli.optimizer_steps)
    gas = int(cfg["training"]["gradient_accumulation_steps"]); interval = int(cfg["training"]["checkpoint_interval"])
    source_module = source["module"]
    dump(arm_out / "initialization_audit.json", {"status": "PASS", "arm": cli.arm, "source": str(source_path),
         "source_sha256": sha256(source_path), "missing_frozen_key_count": len(missing), "unexpected_keys": unexpected,
         "boundary": boundary, "initial_group_hashes": initial_hashes, "canonical_prompt": CANONICAL_UNIFIED_QUESTION,
         "classification_loss_excluded": True, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    metrics_path = arm_out / "training_metrics.jsonl"; started = time.time()
    for zero_step in range(start_step, final_step):
        step_started = time.time(); optimizer.zero_grad(set_to_none=True)
        accum = {"language_ce": 0.0, "mask_bce": 0.0, "mask_dice": 0.0, "mask_loss": 0.0}
        sample_ids = []; labels = []; window = []; local_counts = []
        for micro in range(gas):
            exposure = zero_step * gas + micro
            index = int(schedule["indices"][exposure])
            include_spatial = cli.arm == "JOINT" and int(dataset.rows[index]["class_label"]) == 1
            batch, sample = make_cpu_batch(dataset, index, tokenizer, include_spatial=include_spatial)
            window.append((exposure, batch, sample)); local_counts.append(batch_supervision_counts(batch))
            sample_ids.append(sample["sample_id"]); labels.append(int(sample["cls_label"]))
        global_counts = {key: sum(value[key] for value in local_counts)
                         for key in ("text_tokens", "classification_samples", "valid_masks")}
        for (exposure, batch, sample), counts in zip(window, local_counts):
            # Reset per exposure so LoRA dropout is matched across arms even though only JOINT executes SAM.
            trajectory_seed = seed * 1000003 + exposure
            torch.manual_seed(trajectory_seed); torch.cuda.manual_seed_all(trajectory_seed)
            batch = move_batch(batch, device, torch.bfloat16)
            output = model(**batch)
            language = output["ce_loss"]; mask = output["mask_loss"] if cli.arm == "JOINT" else language * 0.0
            text_scale = counts["text_tokens"] / global_counts["text_tokens"]
            mask_scale = (counts["valid_masks"] / global_counts["valid_masks"]
                          if global_counts["valid_masks"] else 0.0)
            loss = language * text_scale + float(cfg["loss"]["mask_lambda"]) * mask * mask_scale
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {zero_step + 1}")
            loss.backward()
            accum["language_ce"] += float(language.detach().float()) * text_scale
            accum["mask_bce"] += float(output["mask_bce_loss"].detach().float()) * mask_scale if cli.arm == "JOINT" else 0.0
            accum["mask_dice"] += float(output["mask_dice_loss"].detach().float()) * mask_scale if cli.arm == "JOINT" else 0.0
            accum["mask_loss"] += float(mask.detach().float()) * mask_scale
            del output, loss, language, mask, batch
        norms = grad_norms(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                         float(cfg["training"]["gradient_clip_norm"])))
        clipped = grad_norm > float(cfg["training"]["gradient_clip_norm"])
        optimizer.step(); scheduler.step(); step = zero_step + 1
        record = {"optimizer_step": step, "arm": cli.arm, **accum, "total_loss": accum["language_ce"] + accum["mask_loss"],
                  "gradient_norm_before_clip": grad_norm, "per_group_gradient_norms": norms["norms"],
                  "gradient_clipped": clipped, "learning_rates": {g["name"]: float(g["lr"]) for g in optimizer.param_groups},
                  "sample_ids": sample_ids, "class_labels": labels, "global_supervision_counts": global_counts,
                  "aggregation": "P1_exact_text_token_mean_and_valid_mask_mean",
                  "seconds_this_optimizer_step": time.time() - step_started,
                  "mask_loss_to_lora_audit_norm": json.loads((out_root / "audits/gradient_path_audit.json").read_text())["audit_B_mask_only"]["norms"]["lora"] if cli.arm == "JOINT" else None}
        append(metrics_path, record)
        if step <= 5 or step % 10 == 0: print(json.dumps(record), flush=True)
        if step % interval == 0 or step == total_steps:
            metadata = {"arm": cli.arm, "optimizer_step": step, "epoch": step // 100,
                        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                        "allowed_parameter_groups": boundary["allowed"], "training_record": record}
            checkpoint = save_checkpoint(model, source_module, optimizer, scheduler, checkpoint_root, step, metadata)
            append(arm_out / "checkpoint_metadata.jsonl", {**metadata, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)})
    final_hashes = group_hashes(model)
    expected_changed = {"lora": True, "text_hidden_fcs": cli.arm == "JOINT", "mask_decoder": cli.arm == "JOINT", "frozen": False}
    changed = {group: initial_hashes[group] != final_hashes[group] for group in initial_hashes}
    audit = {"status": "PASS" if changed == expected_changed else "FAIL", "arm": cli.arm,
             "initial_hashes": initial_hashes, "final_hashes": final_hashes, "changed": changed,
             "expected_changed": expected_changed, "frozen_delta_zero": not changed["frozen"]}
    dump(arm_out / "parameter_update_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError(f"parameter boundary audit failed: {changed}")
    dump(arm_out / "run_summary.json", {"status": "COMPLETE" if final_step == total_steps else "BOUNDED_RUN_COMPLETE",
         "arm": cli.arm, "start_step": start_step, "final_step": final_step, "total_optimizer_steps": total_steps,
         "image_exposures_this_run": (final_step - start_step) * gas, "elapsed_seconds": time.time() - started,
         "seconds_per_optimizer_step": (time.time() - started) / max(1, final_step - start_step), "parameter_audit": audit})


if __name__ == "__main__":
    main()
