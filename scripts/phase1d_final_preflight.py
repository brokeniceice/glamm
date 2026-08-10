#!/usr/bin/env python3
"""One-batch-only final audit for the preregistered Phase 2A configuration."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    UnifiedForensicsDataset,
)
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import loss_isolation_and_gradient_audit, make_collate, move_batch
from scripts.phase1d_training_policy import configure_args, optimizer_group_audit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a_unified_baseline_full.yaml")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_max_abs(a, b):
    left, right = a.detach().float().cpu(), b.detach().float().cpu()
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max())


def main():
    cli = parse_args()
    config_path = (REPO_ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = REPO_ROOT / "outputs/phase1d_training_policy/final_preflight"
    parameter_dir = REPO_ROOT / "outputs/phase1d_training_policy/parameter_audit"
    output_dir.mkdir(parents=True, exist_ok=True); parameter_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    args = configure_args(config)
    args.freeze_region_encoder = False
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    device = torch.device(cli.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    model.to(device=device, dtype=dtype).eval()

    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=args.image_size,
    )
    real = next(dataset[i] for i, row in enumerate(dataset.rows) if row["forensics_domain"] == "real")
    fake = next(dataset[i] for i, row in enumerate(dataset.rows) if row["forensics_domain"] == "fake")
    mixed_batch = move_batch(make_collate(tokenizer)([real, fake]), device, dtype)

    captured = []
    hook = model.get_output_embeddings().register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().cpu())
    )
    with torch.no_grad():
        before = model(**mixed_batch)
    hook.remove()
    before_lm_logits = captured[-1]

    inference_batch = custom_collate_fn(
        [real, fake], tokenizer=tokenizer, use_mm_start_end=True,
        inference=True, token_strategy="fixed_cls_query",
    )
    inference_batch = move_batch(inference_batch, device, dtype)
    with torch.no_grad():
        before_inference = model(**inference_batch)

    trainable_before = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    region_parameters = glamm_train.freeze_unused_region_encoder(model)
    trainable_after = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    captured = []
    hook = model.get_output_embeddings().register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().cpu())
    )
    with torch.no_grad():
        after = model(**mixed_batch)
        after_inference = model(**inference_batch)
    hook.remove()
    after_lm_logits = captured[0]
    pred_mask_diffs = []
    for left, right in zip(before_inference["pred_masks"], after_inference["pred_masks"]):
        if left is None or right is None:
            pred_mask_diffs.append(0.0 if left is right else float("inf"))
        else:
            pred_mask_diffs.append(tensor_max_abs(left, right))
    equivalence = {
        "classification_logits_max_abs": tensor_max_abs(before["cls_logits"], after["cls_logits"]),
        "lm_verdict_logits_max_abs": tensor_max_abs(
            before["lm_verdict_logits"], after["lm_verdict_logits"]
        ),
        "full_lm_logits_max_abs": tensor_max_abs(before_lm_logits, after_lm_logits),
        "pred_masks_max_abs": max(pred_mask_diffs),
        "loss_max_abs": tensor_max_abs(before["loss"], after["loss"]),
        "ce_loss_max_abs": tensor_max_abs(before["ce_loss"], after["ce_loss"]),
        "cls_loss_max_abs": tensor_max_abs(before["cls_loss"], after["cls_loss"]),
        "mask_bce_loss_max_abs": tensor_max_abs(before["mask_bce_loss"], after["mask_bce_loss"]),
        "mask_dice_loss_max_abs": tensor_max_abs(before["mask_dice_loss"], after["mask_dice_loss"]),
    }
    if any(value != 0.0 for value in equivalence.values()):
        raise AssertionError({"freeze_output_equivalence": equivalence})

    groups = glamm_train.build_optimizer_parameter_groups(model, args)
    group_audit = optimizer_group_audit(groups)
    optimizer_ids = {id(parameter) for group in groups for parameter in group["params"]}
    region_ids = {
        id(parameter) for name, parameter in model.named_parameters() if "region_encoder" in name
    }
    if not optimizer_ids.isdisjoint(region_ids):
        raise AssertionError("region_encoder entered optimizer")
    isolation = loss_isolation_and_gradient_audit(model, dataset, tokenizer, device, dtype)

    model.zero_grad(set_to_none=True)
    result = model(**mixed_batch); result["loss"].backward()
    mixed_group_gradients = {}
    for group in groups:
        squares = [
            parameter.grad.detach().float().pow(2).sum()
            for parameter in group["params"] if parameter.grad is not None
        ]
        mixed_group_gradients[group["name"]] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    if not all(value > 0 for value in mixed_group_gradients.values()):
        raise AssertionError({"expected_mixed_group_gradients": mixed_group_gradients})
    model.zero_grad(set_to_none=True)

    shapes = {
        key: list(value.shape) for key, value in mixed_batch.items() if torch.is_tensor(value)
    }
    report = {
        "status": "passed",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "canonical_prompt_id": CANONICAL_PROMPT_TEMPLATE_ID,
        "canonical_prompt_sha256": CANONICAL_PROMPT_SHA256,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters_before_region_freeze": trainable_before,
        "trainable_region_parameters_before": region_parameters,
        "trainable_parameters_after_region_freeze": trainable_after,
        "trainable_region_parameters_after": 0,
        "optimizer_groups": group_audit,
        "optimizer_duplicate_parameters": 0,
        "region_encoder_in_optimizer": False,
        "freeze_output_equivalence": equivalence,
        "one_batch_loss_and_gradient_audit": isolation,
        "mixed_optimizer_group_gradient_norms": mixed_group_gradients,
        "mixed_batch_tensor_shapes": shapes,
        "formal_training_started": False,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "preflight_report.json").write_text(serialized, encoding="utf-8")
    (parameter_dir / "parameter_audit.json").write_text(serialized, encoding="utf-8")
    (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print(json.dumps({"status": "passed", "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
