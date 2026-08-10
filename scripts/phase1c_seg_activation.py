#!/usr/bin/env python3
"""Phase 1C diagnosis-first runner; never launches the full baseline training."""

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
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import (
    evaluate_gt_fake_generation_localization,
    evaluate_joint_localization,
    evaluate_known_fake_prompt_generation_localization,
    evaluate_legacy_gt_fake_generation_localization,
    evaluate_teacher_forced_full_context,
    evaluate_unified_fake_generation_localization,
    evaluate_unified_gt_fake_prefix_localization,
    write_evaluation_outputs,
)
from eval.forensics_eval import GLaMMForensicsBackend
from eval.phase1c_seg_diagnosis import (
    audit_weight_tying, module_participates_in_loss, summarize_failure_records,
)
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, make_collate, move_batch
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_FAKE_TOKEN, DEFAULT_REAL_TOKEN


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1C SEG activation diagnosis")
    parser.add_argument("--config", default="configs/phase1c_seg_activation.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-diagnosis", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audits", action="store_true")
    parser.add_argument("--mask8-overfit", action="store_true")
    return parser.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_phase1b_model_and_data(config, device):
    train_args = build_train_args(config)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(train_args)
    model = glamm_train.initialize_model(train_args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, train_args)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[train_args.precision]
    checkpoint_path = REPO_ROOT / config["experiment"]["phase1b_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected Phase 1B checkpoint keys: {unexpected}")
    model.to(device=device, dtype=dtype).eval()
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, train_args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=train_args.image_size,
    )
    return model, tokenizer, dataset, dtype, len(missing)


def save_mode_outputs(root, mode, records, metrics):
    mode_dir = root / mode
    write_evaluation_outputs(mode_dir, mode, records, metrics)
    trace_dir = root / "seg_probability_traces" / mode
    failure_rows = []
    for record in records:
        trace = record.pop("seg_probability_trace", [])
        write_json(trace_dir / f"{record['sample_id']}.json", {
            "sample_id": record["sample_id"], "mode": mode, "trace": trace,
        })
        if record.get("stop_reason") != "SEG_TRIGGERED":
            failure_rows.append(record)
    # Rewrite predictions after moving the step-level trace to dedicated files.
    write_evaluation_outputs(mode_dir, mode, records, metrics)
    write_jsonl(root / f"seg_failure_records_{mode}.jsonl", failure_rows)
    failure_summary = summarize_failure_records(records)
    write_json(root / f"seg_failure_summary_{mode}.json", failure_summary)
    return failure_summary


def run_baseline_diagnosis(config, output_dir, device, *, resume=False):
    model, tokenizer, dataset, dtype, missing = load_phase1b_model_and_data(config, device)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=dtype, use_mm_start_end=True,
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
    )
    root = output_dir / "baseline_diagnosis"
    evaluators = (
        ("G0_unified_fake_generate", "unified_fake_generate", evaluate_unified_fake_generation_localization),
        ("G1_unified_gt_fake_prefix", "unified_prompt_gt_fake_prefix", evaluate_unified_gt_fake_prefix_localization),
        ("G2_gt_fake_generate", "gt_fake_generate", evaluate_known_fake_prompt_generation_localization),
        ("G3_tf_full_context", "tf_full_context", evaluate_teacher_forced_full_context),
        ("joint", "joint", evaluate_joint_localization),
        ("legacy_G2", "legacy_gt_fake_generate", evaluate_legacy_gt_fake_generation_localization),
    )
    summaries = {}
    for directory, mode, evaluator in evaluators:
        existing_dir = root / directory
        existing_metrics = existing_dir / f"forensics_metrics_{mode}.json"
        if resume and existing_metrics.is_file():
            metrics = json.loads(existing_metrics.read_text(encoding="utf-8"))
            failure_path = root / f"seg_failure_summary_{mode}.json"
            summaries[mode] = {"metrics": metrics}
            if failure_path.is_file():
                summaries[mode]["failure_summary"] = json.loads(failure_path.read_text(encoding="utf-8"))
            print(f"[Phase1C] reusing completed {mode}", flush=True)
            continue
        print(f"[Phase1C] evaluating {mode}", flush=True)
        records, metrics = evaluator(dataset, backend)
        if mode == "tf_full_context":
            write_evaluation_outputs(root / directory, mode, records, metrics)
            summaries[mode] = {"metrics": metrics}
        else:
            failure = save_mode_outputs(root, mode, records, metrics)
            # Keep the requested human-readable G0/G1/G2/Joint directory aliases.
            alias = root / directory
            if alias != root / mode:
                alias.mkdir(parents=True, exist_ok=True)
                write_evaluation_outputs(alias, mode, records, metrics)
            summaries[mode] = {"metrics": metrics, "failure_summary": failure}

    fake_indices = [
        index for index, row in enumerate(dataset.rows) if row["forensics_domain"] == "fake"
    ]
    selected_alignment_indices = []
    seen = {}
    for index in fake_indices:
        group = str(dataset.rows[index].get("content_category", "")).lower()
        if seen.get(group, 0) < 2:
            selected_alignment_indices.append(index)
            seen[group] = seen.get(group, 0) + 1
    alignments = []
    for index in selected_alignment_indices:
        print(f"[Phase1C] prefix alignment sample {dataset[index]['sample_id']}", flush=True)
        alignments.append(backend.generated_fake_vs_prefilled_fake_alignment(dataset[index]))
    alignment_summary = {
        "num_samples": len(alignments),
        "num_generated_fake": sum(row["generated_fake"] for row in alignments),
        "num_raw_token_ids_equal": sum(row["raw_token_ids_equal"] for row in alignments),
        "max_hidden_abs_difference": max(
            (row["hidden_max_abs_difference"] for row in alignments
             if row["hidden_max_abs_difference"] is not None), default=None,
        ),
        "max_logits_abs_difference": max(
            (row["logits_max_abs_difference"] for row in alignments
             if row["logits_max_abs_difference"] is not None), default=None,
        ),
        "records": alignments,
    }
    write_json(root / "prompt_prefix_alignment.json", alignment_summary)
    write_json(root / "diagnosis_summary.json", {
        "phase1b_checkpoint": config["experiment"]["phase1b_checkpoint"],
        "checkpoint_missing_frozen_keys": missing,
        "seed": config["experiment"]["seed"],
        "generation": config["generation"],
        "modes": summaries,
        "prompt_prefix_alignment": {key: value for key, value in alignment_summary.items() if key != "records"},
    })
    print(json.dumps({"status": "baseline_diagnosis_complete", "output": str(root)}, indent=2))


def run_parameter_and_text_audits(config, output_dir, device):
    model, tokenizer, dataset, dtype, missing = load_phase1b_model_and_data(config, device)
    parameter_dir = output_dir / "parameter_audit"
    text_dir = output_dir / "text_loss_audit"

    embedding = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    tying = audit_weight_tying(
        embedding, lm_head,
        config_tie_word_embeddings=bool(getattr(model.config, "tie_word_embeddings", False)),
    )

    selected = [dataset[0], dataset[1]]
    if {int(row["cls_label"]) for row in selected} != {0, 1}:
        selected = [
            next(dataset[index] for index in range(len(dataset)) if int(dataset.rows[index]["class_label"]) == label)
            for label in (0, 1)
        ]
    batch = custom_collate_fn(
        selected, tokenizer=tokenizer, inference=False, token_strategy="fixed_cls_query"
    )
    batch = move_batch(batch, device, dtype)
    region_modules = [
        (name, module) for name, module in model.named_modules()
        if name.endswith("region_encoder")
    ]
    hook_counts = {name: 0 for name, _ in region_modules}
    hooks = []
    for name, module in region_modules:
        hooks.append(module.register_forward_hook(
            lambda _module, _inputs, _output, module_name=name:
                hook_counts.__setitem__(module_name, hook_counts[module_name] + 1)
        ))
    model.train()
    model.zero_grad(set_to_none=True)
    result = model(**batch)
    result["loss"].backward()
    for hook in hooks:
        hook.remove()
    region_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if "region_encoder" in name
    ]
    region_grad_sq = sum(
        float(parameter.grad.detach().float().pow(2).sum())
        for _, parameter in region_parameters if parameter.grad is not None
    )
    region_audit = {
        "module_names": [name for name, _ in region_modules],
        "forward_hook_counts": hook_counts,
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in region_parameters if parameter.requires_grad
        ),
        "parameters_with_gradient": sum(parameter.grad is not None for _, parameter in region_parameters),
        "gradient_norm": region_grad_sq ** 0.5,
        "batch_bboxes": batch.get("bboxes"),
        "participates_in_unified_forensics_loss": module_participates_in_loss(
            sum(hook_counts.values()), region_grad_sq ** 0.5
        ),
        "recommendation": "freeze for Unified Forensics" if not sum(hook_counts.values()) and region_grad_sq == 0
                          else "retain pending further audit",
    }
    token_rows = {}
    for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN, "[SEG]"):
        token_id = tokenizer(token, add_special_tokens=False).input_ids[0]
        gradient = embedding.grad[token_id]
        token_rows[token] = {
            "token_id": token_id,
            "gradient_norm_in_mixed_backward": float(gradient.detach().float().norm()),
            "must_remain_trainable_in_row_mask_option": True,
            "is_new_phase0_5_token": token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN),
        }
    write_json(parameter_dir / "trainable_cleanup_audit.json", {
        "phase1b_checkpoint": config["experiment"]["phase1b_checkpoint"],
        "checkpoint_missing_frozen_keys": missing,
        "embedding_lm_head_weight_tying": tying,
        "region_encoder": region_audit,
        "special_token_rows": token_rows,
        "optimizer_group_options": {
            "option_a": "keep full embedding/lm_head trainable at 1e-5 or 3e-5",
            "option_b": "gradient-mask old vocabulary rows; retain [CLS], [REAL], [FAKE], and existing [SEG]",
            "phase1c_status": "audit only; no optimizer policy changed",
        },
    })
    model.zero_grad(set_to_none=True)

    rows = [json.loads(line) for line in (
        REPO_ROOT / config["data"]["full_train_manifest"]
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    totals = {"real": 0, "fake": 0}
    counts = {"real": 0, "fake": 0}
    lengths = {"real": [], "fake": []}
    for row in rows:
        domain = row["forensics_domain"]
        # One terminal conversation separator token is supervised in the
        # current collate parser in addition to the target string tokens.
        supervised = len(tokenizer(
            UnifiedForensicsDataset._target(row), add_special_tokens=False
        ).input_ids) + 1
        totals[domain] += supervised
        counts[domain] += 1
        lengths[domain].append(supervised)
    combined = totals["real"] + totals["fake"]
    write_json(text_dir / "real_fake_lm_token_audit.json", {
        "manifest": config["data"]["full_train_manifest"],
        "sample_counts": counts,
        "supervised_lm_token_totals": totals,
        "mean_supervised_tokens_per_sample": {
            domain: totals[domain] / counts[domain] for domain in ("real", "fake")
        },
        "token_denominator_contribution": {
            domain: totals[domain] / combined for domain in ("real", "fake")
        },
        "fake_to_real_supervised_token_ratio": totals["fake"] / totals["real"],
        "per_sample_text_loss_normalization": {
            "implemented": True,
            "default_enabled": False,
            "formula": "mean_i(mean_t CE_it over supervised tokens)",
        },
    })
    print(json.dumps({"status": "audits_complete", "output": str(output_dir)}, indent=2))


def run_mask8_overfit(config, output_dir, device):
    model, tokenizer, dataset, dtype, _ = load_phase1b_model_and_data(config, device)
    selected_indices = []
    counts = {}
    for index, row in enumerate(dataset.rows):
        if row["forensics_domain"] != "fake":
            continue
        group = str(row.get("content_category", "")).lower()
        if counts.get(group, 0) < 2:
            selected_indices.append(index)
            counts[group] = counts.get(group, 0) + 1
    if counts != {"animal": 2, "human": 2, "object": 2, "scene": 2}:
        raise ValueError(f"Could not select balanced 8-Fake diagnostic: {counts}")
    samples = [dataset[index] for index in selected_indices]
    root = output_dir / "mask8_overfit"
    write_jsonl(root / "dataset_subset.jsonl", [dataset.rows[index] for index in selected_indices])

    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            "grounding_encoder.mask_decoder" in name or "text_hidden_fcs" in name
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["mask8_overfit"]["learning_rate"]),
        betas=tuple(config["training"]["betas"]), weight_decay=float(config["training"]["weight_decay"]),
    )
    loader = DataLoader(
        samples, batch_size=int(config["mask8_overfit"]["batch_size"]), shuffle=False,
        num_workers=0, collate_fn=make_collate(tokenizer),
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=dtype, use_mm_start_end=True,
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
    )
    max_steps = int(config["mask8_overfit"]["max_optimizer_steps"])
    interval = int(config["mask8_overfit"]["evaluation_interval"])
    early_stop_iou = float(config["mask8_overfit"]["early_stop_mean_iou"])
    metrics_path = root / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_handle = metrics_path.open("w", encoding="utf-8")
    iterator = iter(loader)
    running = {"mask_loss": 0.0, "mask_bce_loss": 0.0, "mask_dice_loss": 0.0, "count": 0}

    def evaluate(step):
        model.eval()
        records, metrics = evaluate_teacher_forced_full_context(samples, backend)
        write_evaluation_outputs(root / f"evaluation_step_{step}", "tf_full_context", records, metrics)
        record = {
            "record_type": "mask8_evaluation", "step": step,
            "tf_mean_iou": metrics["mean_iou"], "tf_global_iou": metrics["global_iou"],
            "tf_mean_pixel_f1": metrics["mean_pixel_f1"],
            "tf_global_pixel_f1": metrics["global_pixel_f1"],
            "mean_train_mask_loss_since_last_eval": (
                running["mask_loss"] / running["count"] if running["count"] else None
            ),
            "mean_train_bce_since_last_eval": (
                running["mask_bce_loss"] / running["count"] if running["count"] else None
            ),
            "mean_train_dice_since_last_eval": (
                running["mask_dice_loss"] / running["count"] if running["count"] else None
            ),
        }
        metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n"); metrics_handle.flush()
        print(json.dumps(record), flush=True)
        running.update({"mask_loss": 0.0, "mask_bce_loss": 0.0, "mask_dice_loss": 0.0, "count": 0})
        return metrics

    initial = evaluate(0)
    final = initial
    completed_steps = 0
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device, dtype)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch)
        mask_loss = result["mask_loss"]
        if not torch.isfinite(mask_loss):
            raise FloatingPointError(f"Non-finite mask loss at step {step}")
        mask_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        for key in ("mask_loss", "mask_bce_loss", "mask_dice_loss"):
            running[key] += float(result[key].detach().float().cpu())
        running["count"] += 1
        completed_steps = step
        if step % interval == 0:
            final = evaluate(step)
            if final["mean_iou"] >= early_stop_iou:
                break
    if completed_steps % interval:
        final = evaluate(completed_steps)
    metrics_handle.close()
    state = {
        name: parameter.detach().cpu() for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save({"trainable_state_dict": state}, root / "final_checkpoint.pt")
    write_json(root / "summary.json", {
        "start_checkpoint": config["experiment"]["phase1b_checkpoint"],
        "training_scope": "mask loss only; SAM mask decoder + text_hidden_fcs",
        "selected_content_counts": counts,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "early_stop_mean_iou": early_stop_iou,
        "initial_tf_metrics": initial,
        "final_tf_metrics": final,
    })
    print(json.dumps({"status": "mask8_complete", "steps": completed_steps, "output": str(root)}, indent=2))


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
    if cli.baseline_diagnosis:
        run_baseline_diagnosis(config, output_dir, torch.device(cli.device), resume=cli.resume)
        return
    if cli.audits:
        run_parameter_and_text_audits(config, output_dir, torch.device(cli.device))
        return
    if cli.mask8_overfit:
        run_mask8_overfit(config, output_dir, torch.device(cli.device))
        return
    raise ValueError("Select an explicit bounded Phase 1C action")


if __name__ == "__main__":
    main()
