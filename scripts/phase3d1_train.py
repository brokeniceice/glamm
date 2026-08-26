#!/usr/bin/env python3
"""Resumable matched Phase 3D.1 R3/Q2 group-relative policy trainer."""

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
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import trace_full_forward_batch, unwrap_glamm
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args
from scripts.phase2a_distributed_train import deterministic_tensor_collection_hash
from scripts.phase2a_final_evaluate import file_sha256
from scripts.phase3d0_generate import spatial_metrics, trim_generated
from tools.phase3d0 import parse_structure, reward_components, stable_rank
from tools.phase3d0r import FrozenSentenceEncoder, content_phrase, content_tokens
from tools.phase3d1 import group_relative_advantages, score_q2, score_r3
from tools.utils import IMAGE_TOKEN_INDEX


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d1_policy_optimization.yaml")
    parser.add_argument("--arm", choices=("R3", "Q2"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True)
    parser.add_argument("--optimizer-steps", type=int, default=None, help="Bounded preflight/resume increment")
    parser.add_argument("--resume", default=None, help="Path to a step directory or model-state file")
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scheduled_indices(rows: list[dict], seed: int, steps: int) -> list[int]:
    """Frozen balanced schedule: alternating Real/Fake, deterministic hash order."""
    by_label = {
        label: sorted(
            [index for index, row in enumerate(rows) if int(row["class_label"]) == label],
            key=lambda index: stable_rank(seed, f"phase3d1-train-label-{label}", rows[index]["sample_id"]),
        )
        for label in (0, 1)
    }
    positions = {0: 0, 1: 0}
    output = []
    for step in range(steps):
        label = step % 2
        values = by_label[label]
        output.append(values[positions[label] % len(values)])
        positions[label] += 1
    return output


def stable_group_seed(seed: int, step: int, sample_id: str) -> int:
    raw = stable_rank(seed, f"phase3d1-step-{step:04d}", sample_id)
    return int(raw[:8], 16)


def generated_rows(model, tokenizer, backend, sample, cfg, step: int):
    K = int(cfg["algorithm"]["K"])
    seed = stable_group_seed(int(cfg["experiment"]["seed"]), step, sample["sample_id"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    repeated = [sample] * K
    batch = backend._batch_many(repeated, "", question=UNIFIED_FORENSICS_QUESTION)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            images=batch["global_enc_images"], input_ids=batch["input_ids"], bboxes=batch["bboxes"],
            max_new_tokens=int(cfg["sampling"]["max_new_tokens"]), num_beams=1, do_sample=True,
            temperature=float(cfg["sampling"]["temperature"]), top_p=float(cfg["sampling"]["top_p"]),
            use_cache=True, return_dict_in_generate=True, output_scores=False,
        )
    prompt_len = batch["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    token_rows = [trim_generated(generated.sequences[i, prompt_len:], eos_id) for i in range(K)]
    records = []
    for rollout_index, token_row in enumerate(token_rows):
        ids = [int(value) for value in token_row.detach().cpu().tolist()]
        decoded = tokenizer.decode(token_row[token_row.ne(IMAGE_TOKEN_INDEX)], skip_special_tokens=False).strip()
        parsed = parse_phrase_aligned_generation(decoded)
        records.append({
            "sample_id": sample["sample_id"], "rollout_index": rollout_index,
            "seed": seed, "gt_class": int(sample["cls_label"]),
            "generated_token_ids": ids, "decoded_text": decoded,
            "verdict": parsed["verdict"], "generated_explanation": parsed["explanation"],
            "target_regions_raw": parsed["target_region"], "parse_status": parsed["parse_status"],
            "generation_length": len(ids),
            **parse_structure(parsed, ids, unwrap_glamm(model).seg_token_idx),
        })
    return batch, token_rows, records


def attach_spatial_and_historical_components(model, backend, sample, batch, token_rows, records):
    eligible = [index for index, row in enumerate(records) if row["usable_seg"]]
    traces = {}
    if eligible:
        replay = backend._batch_many([sample] * len(eligible), "", question=UNIFIED_FORENSICS_QUESTION)
        full_rows = [torch.cat((batch["input_ids"][index], token_rows[index])) for index in eligible]
        full_ids = torch.nn.utils.rnn.pad_sequence(
            full_rows, batch_first=True, padding_value=backend.tokenizer.pad_token_id,
        ).to(backend.device)
        originals = [tuple(replay["label_list"][i].shape) for i in range(len(eligible))]
        traces_list = trace_full_forward_batch(model, replay, full_ids, original_sizes=originals)
        traces = dict(zip(eligible, traces_list))
    gt = torch.as_tensor(sample["masks"]).bool() if bool(sample["seg_valid"]) else None
    authoritative = sample["localization_field"]["normalized_training_phrase"] if gt is not None else None
    enriched = []
    core = unwrap_glamm(model)
    for index, record in enumerate(records):
        trace = traces.get(index)
        logits = None if trace is None else trace["postprocessed_mask_logits"]
        metrics = spatial_metrics(logits, gt)
        components = reward_components(
            gt_label=int(sample["cls_label"]), parsed=parse_phrase_aligned_generation(record["decoded_text"]),
            generated_ids=record["generated_token_ids"], seg_token_id=core.seg_token_idx,
            authoritative_phrase=authoritative, foreground_iou=metrics["foreground_iou"],
        )
        enriched.append({**record, **metrics, **components})
    return enriched, authoritative


def q2_embeddings(encoder, authoritative: str | None, rows: list[dict]):
    values = set()
    for phrase in [authoritative, *[row.get("normalized_phrase") for row in rows]]:
        values.add(content_phrase(phrase))
        values.update(content_tokens(phrase))
    return encoder.encode(values, batch_size=128)


def policy_sequence_logprobs(core, batch, token_rows, tokenizer):
    """Differentiable mean generated-token log probability for the whole K group."""
    prompt_len = batch["input_ids"].shape[1]
    full_rows = [torch.cat((batch["input_ids"][i], row)) for i, row in enumerate(token_rows)]
    full_ids = torch.nn.utils.rnn.pad_sequence(
        full_rows, batch_first=True, padding_value=tokenizer.pad_token_id,
    ).to(batch["input_ids"].device)
    attention = full_ids.ne(tokenizer.pad_token_id)
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    labels[~attention] = -100
    global_images = core._prepare_global_enc_image(
        batch["global_enc_images"], batch["offset"],
    )
    (
        prepared_ids, prepared_attention, past_key_values,
        input_embeddings, expanded,
    ) = core.prepare_inputs_labels_for_multimodal(
        full_ids, attention, None, labels, global_images, batch["bboxes"],
    )
    # Run the same multimodal causal decoder as LlavaLlamaForCausalLM.forward,
    # but do not pass labels through its generic forward.  That path computes an
    # unused full-vocabulary SFT cross-entropy tensor before the policy loss and
    # can double peak memory.  The logits and expanded labels below are exactly
    # the quantities needed by the frozen group-relative policy objective.
    decoder_output = core.model(
        input_ids=prepared_ids,
        attention_mask=prepared_attention,
        past_key_values=past_key_values,
        inputs_embeds=input_embeddings,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = core.lm_head(decoder_output[0])
    shifted_labels = expanded[:, 1:]
    shifted_logits = logits[:, :-1]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    chosen = shifted_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(shifted_logits, dim=-1)
    token_logp = (chosen - log_z).masked_fill(~valid, 0)
    counts = valid.sum(dim=1).clamp(min=1)
    return token_logp.sum(dim=1) / counts, counts


def policy_backward(model, backend, sample, token_rows, tokenizer, advantages, micro_batch_size: int):
    """Accumulate one exact group loss over fixed trajectory micro-batches."""
    K = len(token_rows)
    logps, counts = [], []
    loss_value = 0.0
    core = unwrap_glamm(model)
    for start in range(0, K, micro_batch_size):
        end = min(K, start + micro_batch_size)
        rows = token_rows[start:end]
        batch = backend._batch_many([sample] * len(rows), "", question=UNIFIED_FORENSICS_QUESTION)
        values, token_counts = policy_sequence_logprobs(core, batch, rows, tokenizer)
        weights = torch.tensor(advantages[start:end], device=values.device, dtype=values.dtype)
        partial = -(weights.detach() * values).sum() / K
        if not torch.isfinite(partial):
            raise FloatingPointError("non-finite policy loss")
        partial.backward()
        loss_value += float(partial.detach().cpu())
        logps.extend(float(value) for value in values.detach().float().cpu())
        counts.extend(int(value) for value in token_counts.detach().cpu())
        del batch, values, token_counts, weights, partial
    return loss_value, logps, counts


def save_checkpoint(model, optimizer, scheduler, root: Path, step: int, metadata: dict):
    destination = root / f"step_{step:04d}" / "checkpoint"
    destination.mkdir(parents=True, exist_ok=True)
    module = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    payload = {
        "module": module, "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(), "optimizer_step": step,
        # Logical checkpoint interval index, used only by the frozen loader's
        # provenance assertion; policy training itself is step based.
        "epoch": step // 250, "best_val_total_loss": float("nan"),
        "client_state": metadata,
    }
    path = destination / "mp_rank_00_model_states.pt"
    torch.save(payload, path)
    dump(destination.parent / "metadata.json", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
    return path


def resolve_resume(value: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        candidates = [path / "checkpoint/mp_rank_00_model_states.pt", path / "mp_rank_00_model_states.pt"]
        path = next((item for item in candidates if item.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main(argv=None):
    cli = parse_args(argv)
    cfg_path = (ROOT / cli.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    expected_gpu = int(cfg["runtime"][f"{cli.arm}_gpu"])
    if cli.physical_gpu != expected_gpu:
        raise RuntimeError(f"{cli.arm} must use physical GPU {expected_gpu}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(cli.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}, expected {cli.physical_gpu}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    out = (ROOT / cfg["experiment"]["output_root"] / "experiments" / f"{cli.arm}_OPT").resolve()
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]).resolve() / f"{cli.arm}_OPT"
    out.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_path, rollouts_path = out / "training_metrics.jsonl", out / "rollouts.jsonl"
    source_path = Path(cfg["source"]["checkpoint"]).resolve()
    if file_sha256(source_path) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    train_args = build_train_args(model_cfg)
    train_args.local_rank = 0
    train_args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(train_args)
    model = glamm_train.initialize_model(train_args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, train_args)
    source = torch.load(source_path, map_location="cpu")
    if int(source["optimizer_step"]) != int(cfg["source"]["optimizer_step"]):
        raise RuntimeError("source optimizer step mismatch")
    missing, unexpected = model.load_state_dict(source["module"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected P1 keys: {unexpected[:10]}")
    del source
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()  # Policy likelihood and sampling both disable dropout; gradients remain enabled.
    initial_hash = deterministic_tensor_collection_hash(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    groups = glamm_train.build_optimizer_parameter_groups(model, train_args)
    lr = float(cfg["optimizer"]["learning_rate"])
    for group in groups:
        group["lr"] = lr
    optimizer = torch.optim.AdamW(
        groups, lr=lr, betas=tuple(map(float, cfg["optimizer"]["betas"])),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    total_steps = int(cfg["training"]["total_optimizer_steps"])
    warmup = int(cfg["optimizer"]["warmup_steps"])
    def lr_lambda(completed):
        if completed < warmup:
            return float(completed + 1) / max(1, warmup)
        return max(0.0, float(total_steps - completed) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if cli.resume:
        resume_path = resolve_resume(cli.resume)
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["module"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state["optimizer_step"])
    elif metrics_path.exists() or rollouts_path.exists():
        raise RuntimeError(f"existing {cli.arm} training artifacts require --resume or explicit archival")
    final_step = total_steps if cli.optimizer_steps is None else min(total_steps, start_step + cli.optimizer_steps)

    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, train_args.vision_tower, split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=train_args.image_size, target_protocol="phrase_aligned",
    )
    schedule = scheduled_indices(dataset.rows, seed, total_steps)
    dump(out / "schedule.json", {
        "seed": seed, "total_steps": total_steps,
        "sample_ids": [dataset.rows[index]["sample_id"] for index in schedule],
        "class_labels": [int(dataset.rows[index]["class_label"]) for index in schedule],
    })
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )
    encoder = None; idf = None
    if cli.arm == "Q2":
        qcfg = yaml.safe_load((ROOT / "configs/phase3d0r_reward_reformulation.yaml").read_text())
        enc = qcfg["semantic_encoder"]
        encoder = FrozenSentenceEncoder(enc["model_id"], enc["revision"], enc["cache_dir"], "cpu")
        idf = read_json(ROOT / cfg["reward"]["Q2"]["token_idf"])["values"]

    dump(out / "initialization_audit.json", {
        "status": "PASS", "arm": cli.arm, "physical_gpu": cli.physical_gpu,
        "source_checkpoint": str(source_path), "source_checkpoint_sha256": file_sha256(source_path),
        "source_optimizer_step": cfg["source"]["optimizer_step"], "missing_frozen_key_count": len(missing),
        "unexpected_keys": unexpected, "initial_trainable_state": initial_hash,
        "fresh_optimizer": not bool(cli.resume), "config_sha256": file_sha256(cfg_path),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })

    interval = int(cfg["training"]["checkpoint_interval"])
    started = time.time(); completed_rollouts = 0; zero_variance_groups = 0
    for zero_step in range(start_step, final_step):
        step = zero_step + 1; sample = dataset[schedule[zero_step]]; step_started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        # Previous-step gradients are not needed during sampling.  Releasing
        # them here is mathematically identical to clearing them before
        # backward, while leaving more memory for the frozen K=8 rollout.
        optimizer.zero_grad(set_to_none=True)
        batch, token_rows, records = generated_rows(model, tokenizer, backend, sample, cfg, step)
        records, authoritative = attach_spatial_and_historical_components(
            model, backend, sample, batch, token_rows, records,
        )
        if cli.arm == "R3":
            rewards = score_r3(int(sample["cls_label"]), records)
        else:
            embeddings = q2_embeddings(encoder, authoritative, records)
            rewards, records = score_q2(
                int(sample["cls_label"]), records, authoritative_phrase=authoritative,
                idf=idf, embeddings=embeddings,
            )
        advantages = group_relative_advantages(rewards, float(cfg["algorithm"]["advantage_epsilon"]))
        zero_variance_groups += int(not any(advantages))

        del batch
        policy_loss, sequence_logps, token_counts = policy_backward(
            model, backend, sample, token_rows, tokenizer, advantages,
            int(cfg["training"]["policy_trajectory_micro_batch_size"]),
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(cfg["training"]["gradient_clip_norm"]),
        ))
        optimizer.step(); scheduler.step()
        completed_rollouts += len(records)

        rollout_record = {
            "optimizer_step": step, "arm": cli.arm, "sample_id": sample["sample_id"],
            "gt_class": int(sample["cls_label"]), "authoritative_phrase": authoritative,
            "rollouts": [
                {**row, "reward": rewards[index], "advantage": advantages[index],
                 "policy_mean_logprob": sequence_logps[index],
                 "policy_token_count": token_counts[index]}
                for index, row in enumerate(records)
            ],
        }
        append(rollouts_path, rollout_record)
        reward_array = np.asarray(rewards, dtype=float)
        record = {
            "optimizer_step": step, "arm": cli.arm, "sample_id": sample["sample_id"],
            "gt_class": int(sample["cls_label"]), "policy_loss": policy_loss,
            "gradient_norm_before_clip": grad_norm, "reward_mean": float(reward_array.mean()),
            "reward_std": float(reward_array.std(ddof=0)), "reward_min": float(reward_array.min()),
            "reward_max": float(reward_array.max()), "zero_advantage_group": not any(advantages),
            "mean_generation_length": float(np.mean([len(row) for row in token_rows])),
            "learning_rates": {group.get("name", str(index)): float(group["lr"])
                               for index, group in enumerate(optimizer.param_groups)},
            "elapsed_seconds": time.time() - step_started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        append(metrics_path, record)
        print(json.dumps(record), flush=True)
        if step % interval == 0 or step == total_steps:
            metadata = {
                "arm": cli.arm, "optimizer_step": step, "training_loss": record["policy_loss"],
                "reward_mean": record["reward_mean"], "reward_std": record["reward_std"],
                "completed_rollouts": completed_rollouts, "zero_variance_groups": zero_variance_groups,
                "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                "initial_trainable_state": initial_hash, "physical_gpu": cli.physical_gpu,
            }
            checkpoint = save_checkpoint(model, optimizer, scheduler, checkpoint_root, step, metadata)
            append(out / "checkpoint_metadata.jsonl", {**metadata, "checkpoint": str(checkpoint)})

    dump(out / "run_summary.json", {
        "status": "COMPLETE" if final_step == total_steps else "PREFLIGHT_COMPLETE",
        "arm": cli.arm, "start_step": start_step, "final_step": final_step,
        "total_optimizer_steps": total_steps, "completed_rollouts_this_run": completed_rollouts,
        "elapsed_seconds": time.time() - started, "zero_variance_groups_this_run": zero_variance_groups,
        "physical_gpu": cli.physical_gpu, "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
        "initial_trainable_state": initial_hash,
    })


if __name__ == "__main__":
    main()
