#!/usr/bin/env python3
"""Bounded Phase 1D policy ablations and final preflight helpers."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    UnifiedForensicsDataset,
)
from eval.forensics import (
    evaluate_detection,
    evaluate_gt_fake_generation_localization,
    evaluate_joint_localization,
    evaluate_teacher_forced_full_context,
    evaluate_unified_fake_generation_localization,
    write_evaluation_outputs,
)
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, make_collate, move_batch
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_FAKE_TOKEN, DEFAULT_REAL_TOKEN


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1D controlled training-policy ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def optimizer_group_audit(groups):
    seen = set()
    records = []
    for group in groups:
        duplicate = [id(parameter) for parameter in group["params"] if id(parameter) in seen]
        if duplicate:
            raise ValueError(f"Duplicate optimizer parameters in group {group['name']}")
        seen.update(id(parameter) for parameter in group["params"])
        records.append({
            "name": group["name"], "learning_rate": float(group["lr"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "tensor_count": len(group["params"]),
        })
    return records


def snapshot_vocab_weights(model):
    return {
        "embeddings": model.get_input_embeddings().weight.detach().cpu().clone(),
        "lm_head": model.get_output_embeddings().weight.detach().cpu().clone(),
    }


def _summary(values):
    values = values.detach().float().cpu().numpy()
    return {
        "mean_l2_delta": float(values.mean()),
        "median_l2_delta": float(np.median(values)),
        "p95_l2_delta": float(np.percentile(values, 95)),
        "max_l2_delta": float(values.max()),
    }


def vocabulary_drift(initial, final, new_row_ids):
    initial = initial.float(); final = final.detach().float().cpu()
    deltas = (final - initial).norm(dim=1)
    new_ids = torch.as_tensor(sorted(set(new_row_ids)), dtype=torch.long)
    old_mask = torch.ones(deltas.shape[0], dtype=torch.bool); old_mask[new_ids] = False
    initial_norm = initial.norm()
    output = {
        "new_token_rows": {**_summary(deltas[new_ids]), "row_ids": new_ids.tolist()},
        "old_vocabulary_rows": _summary(deltas[old_mask]),
        "relative_delta_norm": float((final - initial).norm() / initial_norm.clamp_min(1e-12)),
    }
    return output


def configure_args(config):
    args = build_train_args(config)
    args.freeze_region_encoder = bool(config["model"]["freeze_region_encoder"])
    optimizer = config["optimizer"]
    args.lr = float(optimizer["lora_lr"])
    for key in (
        "lora_lr", "classification_head_lr", "text_hidden_fcs_lr", "mask_decoder_lr",
        "embedding_lr", "lm_head_lr",
    ):
        setattr(args, key, float(optimizer[key]))
    args.per_sample_text_loss_normalization = bool(
        config["loss"]["per_sample_text_loss_normalization"]
    )
    return args


def run_evaluations(model, tokenizer, dataset, output_dir, device, dtype, max_new_tokens):
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=dtype, use_mm_start_end=True,
        max_new_tokens=max_new_tokens,
    )
    evaluators = (
        ("detection", evaluate_detection),
        ("G0_unified_fake_generate", evaluate_unified_fake_generation_localization),
        ("G1_gt_fake_generate", evaluate_gt_fake_generation_localization),
        ("tf_full_context", evaluate_teacher_forced_full_context),
        ("joint", evaluate_joint_localization),
    )
    outputs = {}
    records_by_mode = {}
    model.eval()
    for directory, evaluator in evaluators:
        records, metrics = evaluator(dataset, backend)
        mode = records[0]["eval_mode"] if records else directory
        write_evaluation_outputs(output_dir / directory, mode, records, metrics)
        outputs[directory] = metrics
        records_by_mode[directory] = records
    g0 = {row["sample_id"]: row for row in records_by_mode["G0_unified_fake_generate"]}
    joint = {row["sample_id"]: row for row in records_by_mode["joint"]}
    comparisons = []
    for sample_id in sorted(g0):
        a, b = g0[sample_id], joint[sample_id]
        comparisons.append({
            "sample_id": sample_id,
            "prompt_hash_equal": a["prompt_sha256"] == b["prompt_sha256"],
            "raw_prompt_equal": a["raw_prompt_text"] == b["raw_prompt_text"],
            "generated_token_ids_equal": a["generated_token_ids"] == b["generated_token_ids"],
            "seg_position_equal": a["seg_position"] == b["seg_position"],
            "intersection_equal_when_gate_passes": (
                a["intersection"] == b["intersection"] if b["classification_gate_passed"] else None
            ),
            "union_equal_when_gate_passes": (
                a["union"] == b["union"] if b["classification_gate_passed"] else None
            ),
            "classification_gate_passed": b["classification_gate_passed"],
        })
    all_pass = all(row["classification_gate_passed"] for row in comparisons)
    equivalence = {
        "canonical_prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
        "canonical_prompt_sha256": CANONICAL_PROMPT_SHA256,
        "all_classification_gates_pass": all_pass,
        "all_generation_inputs_and_outputs_equal": all(
            row["prompt_hash_equal"] and row["raw_prompt_equal"]
            and row["generated_token_ids_equal"] and row["seg_position_equal"]
            for row in comparisons
        ),
        "aggregates_equal_when_all_gates_pass": (
            outputs["G0_unified_fake_generate"]["mean_iou"] == outputs["joint"]["mean_iou"]
            and outputs["G0_unified_fake_generate"]["global_iou"] == outputs["joint"]["global_iou"]
            if all_pass else None
        ),
        "records": comparisons,
    }
    write_json(output_dir / "joint_g0_equivalence.json", equivalence)
    return outputs, records_by_mode, equivalence


def main(argv=None):
    cli = parse_args(argv)
    config_path = (REPO_ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (REPO_ROOT / config["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    args = configure_args(config)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    device = torch.device(cli.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    model.to(device=device, dtype=dtype)
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "region_encoder" in name):
        raise AssertionError("region_encoder remains trainable")
    groups = glamm_train.build_optimizer_parameter_groups(model, args)
    group_audit = optimizer_group_audit(groups)
    optimizer = torch.optim.AdamW(
        groups, betas=tuple(config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    max_steps = int(config["training"]["max_optimizer_steps"])
    warmup = int(config["training"]["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / max(1, warmup) if step < warmup
        else max(0.0, (max_steps - step) / max(1, max_steps - warmup)),
    )
    initial_vocab = snapshot_vocab_weights(model)
    new_row_ids = [
        tokenizer(token, add_special_tokens=False).input_ids[0]
        for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN, "[SEG]")
    ]
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=args.image_size,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size_per_device"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), collate_fn=make_collate(tokenizer),
    )
    iterator = iter(loader)
    metrics_path = output_dir / "metrics.jsonl"
    handle = metrics_path.open("w", encoding="utf-8")
    model.train()
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = move_batch(batch, device, dtype)
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in groups for parameter in group["params"]],
            float(config["optimizer"]["gradient_clip_norm"]),
        )
        optimizer.step(); scheduler.step()
        labels = batch["cls_labels"]
        cls_pred = result["cls_pred"].detach(); lm_pred = result["lm_verdict_pred"].detach()
        real = labels.eq(0); fake = labels.eq(1)
        record = {
            "step": step,
            "total_loss": float(result["loss"].detach().float()),
            "text_loss": float(result["ce_loss"].detach().float()),
            "cls_loss": float(result["cls_loss"].detach().float()),
            "mask_bce_loss": float(result["mask_bce_loss"].detach().float()),
            "mask_dice_loss": float(result["mask_dice_loss"].detach().float()),
            "real_text_loss": float(result["real_text_loss"].detach().float()),
            "fake_text_loss": float(result["fake_text_loss"].detach().float()),
            "classification_accuracy": float(cls_pred.eq(labels).float().mean()),
            "real_verdict_accuracy": float(lm_pred[real].eq(0).float().mean()),
            "fake_verdict_accuracy": float(lm_pred[fake].eq(1).float().mean()),
            "lm_verdict_accuracy": float(lm_pred.eq(labels).float().mean()),
            "cls_lm_agreement": float(cls_pred.eq(lm_pred).float().mean()),
            "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
        print(json.dumps(record), flush=True)
    handle.close()

    drift = {
        "embeddings": vocabulary_drift(
            initial_vocab["embeddings"], model.get_input_embeddings().weight, new_row_ids
        ),
        "lm_head": vocabulary_drift(
            initial_vocab["lm_head"], model.get_output_embeddings().weight, new_row_ids
        ),
    }
    write_json(output_dir / "vocabulary_drift.json", drift)
    evaluation, records, equivalence = run_evaluations(
        model, tokenizer, dataset, output_dir / "evaluation", device, dtype,
        int(config["evaluation"]["max_new_tokens"]),
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    write_json(output_dir / "optimizer_group_audit.json", {
        "total_parameters": total, "trainable_parameters": trainable,
        "frozen_parameters": total - trainable, "groups": group_audit,
        "region_encoder_trainable_parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if "region_encoder" in name and parameter.requires_grad
        ),
        "duplicate_parameter_count": 0,
    })
    state = {
        name: parameter.detach().cpu() for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save({"trainable_state_dict": state, "config": config}, output_dir / "final_checkpoint.pt")
    write_json(output_dir / "summary.json", {
        "config": config, "evaluation": evaluation, "vocabulary_drift": drift,
        "optimizer_groups": group_audit, "joint_g0_equivalence": equivalence,
        "canonical_prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
        "canonical_prompt_sha256": CANONICAL_PROMPT_SHA256,
    })
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print(json.dumps({"status": "complete", "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
