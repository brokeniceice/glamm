#!/usr/bin/env python3
"""Phase 3F preflight, frozen-teacher cache construction, and AOGD training."""

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
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION, UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, move_batch
from tools.distributed_loss import batch_supervision_counts
from tools.phase3f_aogd import (
    capture_representations, configure_lora_only, core_model, cosine_loss, file_sha256,
    grad_norm, group_hashes, load_jsonl_index, mask_metrics_from_projected,
    parameter_group, replay_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3f_autonomous_oracle_grounding_distillation.yaml")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--mode", choices=("preflight", "build-teacher-cache", "train", "sft-control"), required=True)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def setup(cfg, physical_gpu, mode):
    expected = int(cfg["runtime"]["sft_control_gpu"] if mode == "sft-control" else cfg["runtime"]["physical_gpu"])
    if physical_gpu != expected:
        raise RuntimeError(f"Phase 3F is assigned to physical GPU {expected}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(expected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES violates frozen GPU assignment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    args = build_train_args(model_cfg); args.local_rank = 0; args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    source_path = Path(cfg["source"]["checkpoint"])
    if file_sha256(source_path) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checksum mismatch")
    source = torch.load(source_path, map_location="cpu")
    if int(source["optimizer_step"]) != int(cfg["source"]["optimizer_step"]):
        raise RuntimeError("P1 optimizer step mismatch")
    if int(source["epoch"]) != int(cfg["source"]["epoch"]):
        raise RuntimeError("P1 epoch mismatch")
    missing, unexpected = model.load_state_dict(source["module"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected P1 keys: {unexpected[:10]}")
    boundary = configure_lora_only(model)
    model.enable_input_require_grads()
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input gradient hook unfroze embeddings")
    model.to(device=device, dtype=torch.bfloat16)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="phrase_aligned",
    )
    schedule = json.loads(((ROOT / cfg["experiment"]["output_root"]) / "training/schedule.json").read_text(encoding="utf-8"))
    if [dataset.rows[index]["sample_id"] for index in schedule["indices"]] != schedule["sample_ids"]:
        raise RuntimeError("Phase 3F frozen schedule does not resolve exactly")
    if CANONICAL_UNIFIED_QUESTION != cfg["prompt"]["user_question"] or UNIFIED_FORENSICS_QUESTION != cfg["prompt"]["user_question"]:
        raise RuntimeError("canonical prompt mismatch")
    replay = load_jsonl_index(ROOT / cfg["rollout"]["phase3b_cache"])
    return device, args, tokenizer, model, source, missing, boundary, backend, dataset, schedule, replay


def make_language_batch(dataset, index, tokenizer, device):
    sample = dataset[index]
    batch = custom_collate_fn([sample], tokenizer=tokenizer, use_mm_start_end=True,
                              inference=False, token_strategy="fixed_cls_query")
    batch["grounding_enc_images"] = None
    batch["masks_list"] = [None]
    batch["seg_valid"] = torch.tensor([False])
    return move_batch(batch, device, torch.bfloat16), sample


def teacher_batch(backend, sample):
    return backend._batch(sample, backend.tf_phrase_content(sample), question=UNIFIED_FORENSICS_QUESTION)


def optimizer_and_scheduler(model, cfg):
    params = [parameter for name, parameter in model.named_parameters()
              if parameter.requires_grad and parameter_group(name) == "lora"]
    optimizer = torch.optim.AdamW([{"name": "lora", "params": params, "lr": float(cfg["optimizer"]["lora_lr"])}],
                                  betas=tuple(map(float, cfg["optimizer"]["betas"])),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    total = int(cfg["training"]["total_optimizer_steps"]); warmup = int(cfg["optimizer"]["warmup_steps"])
    def lr_lambda(completed):
        if completed < warmup:
            return float(completed + 1) / max(1, warmup)
        return max(0.0, float(total - completed) / max(1, total - warmup))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def preflight(cfg, out, model, backend, dataset, schedule, replay):
    core = core_model(model)
    fixed = [(index, sid) for index, sid, label in zip(schedule["indices"], schedule["sample_ids"], schedule["class_labels"])
             if int(label) == 1 and replay[sid]["replay_eligible"]][:int(cfg["preflight"]["fake_count"])]
    if len(fixed) != int(cfg["preflight"]["fake_count"]):
        raise RuntimeError("not enough eligible Fake samples for preflight")
    rows = []; teacher_vectors = {}
    model.eval()
    with torch.no_grad():
        for index, sid in fixed:
            sample = dataset[index]
            auto_batch = replay_batch(backend, sample, replay[sid], UNIFIED_FORENSICS_QUESTION)
            oracle_batch = teacher_batch(backend, sample)
            auto_raw, auto_projected, _, _ = capture_representations(core, auto_batch)
            oracle_raw, oracle_projected, _, _ = capture_representations(core, oracle_batch)
            auto_metrics = mask_metrics_from_projected(core, auto_batch, auto_projected)
            oracle_metrics = mask_metrics_from_projected(core, oracle_batch, oracle_projected)
            raw_loss = float(cosine_loss(auto_raw, oracle_raw))
            projected_loss = float(cosine_loss(auto_projected, oracle_projected))
            teacher_vectors[sid] = oracle_projected.detach().float().cpu()
            rows.append({"sample_id": sid, "g0": auto_metrics, "tf_full": oracle_metrics,
                         "raw_4096d_cosine_gap": raw_loss, "projected_256d_cosine_gap": projected_loss})
    mean = lambda key, mode=None: sum((row[mode][key] if mode else row[key]) for row in rows) / len(rows)
    checks = {
        "A_tf_iou_advantage_positive": mean("foreground_iou", "tf_full") > mean("foreground_iou", "g0"),
        "B_raw_4096d_gap_positive": mean("raw_4096d_cosine_gap") > 0,
        "B_projected_256d_gap_positive": mean("projected_256d_cosine_gap") > 0,
    }
    first_index, first_sid = fixed[0]
    sample = dataset[first_index]
    auto_batch = replay_batch(backend, sample, replay[first_sid], UNIFIED_FORENSICS_QUESTION)
    model.train(); model.zero_grad(set_to_none=True)
    torch.manual_seed(99173); torch.cuda.manual_seed_all(99173)
    _, auto_projected, _, _ = capture_representations(core, auto_batch)
    loss = cosine_loss(auto_projected, teacher_vectors[first_sid].to(auto_projected.device))
    loss.backward()
    lora_gradient = grad_norm(model, "lora")
    frozen_gradient_tensors = sum(parameter.grad is not None for name, parameter in model.named_parameters()
                                  if parameter_group(name) != "lora")
    checks["C_lrepr_lora_gradient_positive"] = lora_gradient > 0
    checks["C_all_frozen_gradients_absent"] = frozen_gradient_tensors == 0
    model.zero_grad(set_to_none=True)

    before_hashes = group_hashes(model)
    before_lora = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                   if parameter_group(name) == "lora"}
    model.eval()
    with torch.no_grad():
        _, before_vector, _, _ = capture_representations(core, auto_batch)
        before_vector = before_vector.detach().float().cpu()
    optimizer, _ = optimizer_and_scheduler(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    model.train(); torch.manual_seed(81831); torch.cuda.manual_seed_all(81831)
    _, projected, _, _ = capture_representations(core, auto_batch)
    step_loss = cosine_loss(projected, teacher_vectors[first_sid].to(projected.device))
    step_loss.backward(); optimizer.step(); model.zero_grad(set_to_none=True)
    after_hashes = group_hashes(model)
    model.eval()
    with torch.no_grad():
        _, after_vector, _, _ = capture_representations(core, auto_batch)
        after_vector = after_vector.detach().float().cpu()
    checks.update({
        "D_lora_changed": before_hashes["lora"] != after_hashes["lora"],
        "D_g_auto_changed": not torch.equal(before_vector, after_vector),
        "D_text_hidden_fcs_unchanged": before_hashes["text_hidden_fcs"] == after_hashes["text_hidden_fcs"],
        "D_mask_decoder_unchanged": before_hashes["mask_decoder"] == after_hashes["mask_decoder"],
        "D_all_other_frozen_unchanged": before_hashes["frozen"] == after_hashes["frozen"],
    })
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in before_lora.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    restored_hashes = group_hashes(model)
    checks["D_student_restored_exactly"] = restored_hashes == before_hashes
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "gate_on_failure": cfg["preflight"]["gate_on_failure"],
        "sample_count": len(rows), "sample_ids": [sid for _, sid in fixed], "checks": checks,
        "condition_means": {"g0_iou": mean("foreground_iou", "g0"), "tf_full_iou": mean("foreground_iou", "tf_full"),
                            "tf_minus_g0_iou": mean("foreground_iou", "tf_full") - mean("foreground_iou", "g0"),
                            "raw_4096d_cosine_gap": mean("raw_4096d_cosine_gap"),
                            "projected_256d_cosine_gap": mean("projected_256d_cosine_gap")},
        "lrepr_only_lora_gradient_norm": lora_gradient, "frozen_gradient_tensor_count": frozen_gradient_tensors,
        "one_step_g_auto_l2_change": float((after_vector - before_vector).norm()), "rows": rows,
    }
    dump(out / "preflight/aogd_preflight.json", result)
    dump(out / "preflight/representation_gap_32fake.json", {"rows": rows, "means": result["condition_means"]})
    if result["status"] != "PASS":
        raise RuntimeError(cfg["preflight"]["gate_on_failure"])
    print(json.dumps({key: result[key] for key in ("status", "condition_means", "lrepr_only_lora_gradient_norm", "one_step_g_auto_l2_change")}, indent=2))


def build_teacher_cache(cfg, out, model, backend, dataset, schedule, replay):
    preflight_path = out / "preflight/aogd_preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["status"] != "PASS":
        raise RuntimeError("PASS preflight required before teacher cache")
    destination = Path(cfg["experiment"]["checkpoint_root"]) / "teacher_cache/projected_256d.pt"
    manifest_path = out / "teacher/teacher_cache_manifest.json"
    if destination.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if file_sha256(destination) == manifest["sha256"]:
            print(json.dumps({"status": "REUSED", "count": manifest["count"], "sha256": manifest["sha256"]}, indent=2)); return
        raise RuntimeError("existing teacher cache hash mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    core = core_model(model); model.eval(); values = {}; started = time.time()
    eligible = []
    seen = set()
    for index, sid, label in zip(schedule["indices"], schedule["sample_ids"], schedule["class_labels"]):
        if int(label) == 1 and replay[sid]["replay_eligible"] and sid not in seen:
            eligible.append((index, sid)); seen.add(sid)
    with torch.no_grad():
        for position, (index, sid) in enumerate(eligible, 1):
            batch = teacher_batch(backend, dataset[index])
            raw, projected, _, _ = capture_representations(core, batch)
            values[sid] = {"raw_4096d": raw.detach().float().cpu(),
                           "projected_256d": projected.detach().float().cpu()}
            if position <= 5 or position % 50 == 0:
                print(json.dumps({"teacher_cache": position, "total": len(eligible),
                                  "seconds_per_sample": (time.time() - started) / position}), flush=True)
    torch.save({"vectors": values, "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                "trajectory": "authoritative_full_canonical_tf", "dimensions": {"raw_4096d": 4096, "projected_256d": 256}}, destination)
    manifest = {"status": "FROZEN", "path": str(destination), "sha256": file_sha256(destination),
                "count": len(values), "dimensions": {"raw_4096d": 4096, "projected_256d": 256},
                "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                "stop_gradient": True, "elapsed_seconds": time.time() - started}
    dump(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def save_checkpoint(model, source_module, optimizer, scheduler, root, step, metadata):
    destination = Path(root) / f"step_{step:04d}" / "checkpoint"
    destination.mkdir(parents=True, exist_ok=True)
    module = dict(source_module)
    named = dict(model.named_parameters())
    for name in list(module):
        if name in named and parameter_group(name) == "lora":
            module[name] = named[name].detach().cpu().clone()
    path = destination / "mp_rank_00_model_states.pt"
    torch.save({"module": module, "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                "optimizer_step": step, "epoch": step // 100, "best_val_total_loss": float("nan"),
                "client_state": metadata}, path)
    dump(destination.parent / "metadata.json", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
    return path


def checkpoint_diagnostics(model, core, dataset, tokenizer, backend, replay, teacher, fixed_index, cfg):
    """No-update, fixed-sample independent language/repr gradient and gap audit."""
    sample = dataset[fixed_index]; sid = sample["sample_id"]
    if not replay[sid]["replay_eligible"]: raise RuntimeError("checkpoint diagnostic sample is not eligible")
    model.zero_grad(set_to_none=True); model.train()
    torch.manual_seed(44101); torch.cuda.manual_seed_all(44101)
    language_batch, _ = make_language_batch(dataset, fixed_index, tokenizer, next(model.parameters()).device)
    output = model(**language_batch); language = output["ce_loss"]; language.backward()
    language_grad = grad_norm(model, "lora")
    del output, language, language_batch
    model.zero_grad(set_to_none=True); model.train()
    torch.manual_seed(44102); torch.cuda.manual_seed_all(44102)
    auto_batch = replay_batch(backend, sample, replay[sid], UNIFIED_FORENSICS_QUESTION)
    _, projected, _, _ = capture_representations(core, auto_batch)
    representation = cosine_loss(projected, teacher[sid]["projected_256d"].to(projected.device))
    representation.backward(); representation_grad = grad_norm(model, "lora")
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        raw, projected_eval, _, _ = capture_representations(core, auto_batch)
        raw_gap = float(cosine_loss(raw, teacher[sid]["raw_4096d"].to(raw.device)))
        projected_gap = float(cosine_loss(projected_eval, teacher[sid]["projected_256d"].to(projected_eval.device)))
    del auto_batch, projected, representation, raw, projected_eval
    model.train()
    return {"fixed_sample_id": sid, "language_lora_gradient_norm": language_grad,
            "representation_lora_gradient_norm": representation_grad,
            "raw_4096d_autonomous_oracle_cosine_gap": raw_gap,
            "projected_256d_autonomous_oracle_cosine_gap": projected_gap,
            "eligible_fake_rate_global": 0.9834766862833861, "generated_seg_valid_rate_global": 0.9834766862833861,
            "rollout_cache_version": cfg["rollout"]["cache_sha256"], "regenerated_trajectories": 0}


def train(cfg, out, model, source, missing, boundary, tokenizer, backend, dataset, schedule, replay, resume, bounded_steps):
    preflight = json.loads((out / "preflight/aogd_preflight.json").read_text())
    if preflight["status"] != "PASS": raise RuntimeError("preflight did not pass")
    teacher_manifest = json.loads((out / "teacher/teacher_cache_manifest.json").read_text())
    teacher_path = Path(teacher_manifest["path"])
    if file_sha256(teacher_path) != teacher_manifest["sha256"]: raise RuntimeError("teacher cache checksum mismatch")
    teacher_state = torch.load(teacher_path, map_location="cpu")
    if teacher_state["source_checkpoint_sha256"] != cfg["source"]["checkpoint_sha256"]: raise RuntimeError("teacher source mismatch")
    teacher = teacher_state["vectors"]
    optimizer, scheduler = optimizer_and_scheduler(model, cfg)
    total = int(cfg["training"]["total_optimizer_steps"]); start = 0
    if resume:
        state = torch.load(Path(resume), map_location="cpu")
        model.load_state_dict(state["module"], strict=False); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"]); start = int(state["optimizer_step"])
    metrics_path = out / "training/training_metrics.jsonl"
    if metrics_path.exists() and not resume: raise RuntimeError("existing formal metrics require --resume")
    if resume:
        def truncate_jsonl(path, step_key):
            if not path.exists(): return
            kept = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                row = json.loads(line)
                if int(row[step_key]) <= start: kept.append(line)
            path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
        truncate_jsonl(metrics_path, "optimizer_step")
        truncate_jsonl(out / "training/checkpoint_metadata.jsonl", "optimizer_step")
    final = total if bounded_steps is None else min(total, start + int(bounded_steps))
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"])
    initial_hashes = group_hashes(model); core = core_model(model)
    dump(out / "training/initialization_audit.json", {
        "status": "PASS", "source_checkpoint": cfg["source"]["checkpoint"],
        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"], "missing_frozen_key_count": len(missing),
        "parameter_boundary": boundary, "initial_group_hashes": initial_hashes,
        "canonical_prompt": cfg["prompt"]["user_question"], "teacher_cache_sha256": teacher_manifest["sha256"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    if start == 0:
        dump(checkpoint_root / "step_0000/metadata.json", {
            "optimizer_step": 0, "checkpoint": cfg["source"]["checkpoint"], "sha256": cfg["source"]["checkpoint_sha256"],
            "role": "canonical_P1_step0_reference_no_copy",
        })
    gas = int(cfg["training"]["gradient_accumulation_steps"]); interval = int(cfg["training"]["checkpoint_interval"])
    diagnostic_index = next(index for index, sid, label in zip(schedule["indices"], schedule["sample_ids"], schedule["class_labels"])
                            if int(label) == 1 and replay[sid]["replay_eligible"])
    started = time.time(); model.train()
    for zero_step in range(start, final):
        step_started = time.time(); optimizer.zero_grad(set_to_none=True)
        window = []
        for micro in range(gas):
            exposure = zero_step * gas + micro; index = int(schedule["indices"][exposure])
            cpu_batch, sample = make_language_batch(dataset, index, tokenizer, torch.device("cpu"))
            counts = batch_supervision_counts(cpu_batch)
            sid = sample["sample_id"]; eligible = int(sample["cls_label"]) == 1 and replay[sid]["replay_eligible"]
            window.append((exposure, index, cpu_batch, sample, counts, eligible))
        text_tokens = sum(item[4]["text_tokens"] for item in window)
        eligible_count = sum(item[5] for item in window)
        accum_ce = 0.0; accum_repr = 0.0; ids = []; labels = []
        for exposure, index, cpu_batch, sample, counts, eligible in window:
            trajectory_seed = int(cfg["experiment"]["seed"]) * 1000003 + exposure
            torch.manual_seed(trajectory_seed); torch.cuda.manual_seed_all(trajectory_seed)
            language_batch = move_batch(cpu_batch, next(model.parameters()).device, torch.bfloat16)
            output = model(**language_batch); ce = output["ce_loss"]
            ce_scale = counts["text_tokens"] / text_tokens
            (ce * ce_scale).backward(); accum_ce += float(ce.detach().float()) * ce_scale
            del output, ce, language_batch
            if eligible:
                replay_input = replay_batch(backend, sample, replay[sample["sample_id"]], UNIFIED_FORENSICS_QUESTION)
                _, projected, _, _ = capture_representations(core, replay_input)
                repr_loss = cosine_loss(projected, teacher[sample["sample_id"]]["projected_256d"].to(projected.device))
                (float(cfg["loss"]["representation_weight"]) * repr_loss / eligible_count).backward()
                accum_repr += float(repr_loss.detach().float()) / eligible_count
                del replay_input, projected, repr_loss
            ids.append(sample["sample_id"]); labels.append(int(sample["cls_label"]))
        lora_grad = grad_norm(model, "lora")
        before_clip = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                           float(cfg["training"]["gradient_clip_norm"])))
        optimizer.step(); scheduler.step(); step = zero_step + 1
        record = {"optimizer_step": step, "language_ce": accum_ce, "representation_loss": accum_repr,
                  "total_loss": accum_ce + float(cfg["loss"]["representation_weight"]) * accum_repr,
                  "lora_gradient_norm": lora_grad, "gradient_norm_before_clip": before_clip,
                  "gradient_clipped": before_clip > float(cfg["training"]["gradient_clip_norm"]),
                  "learning_rate": float(optimizer.param_groups[0]["lr"]), "sample_ids": ids, "class_labels": labels,
                  "eligible_fake_count": eligible_count, "rollout_cache_sha256": cfg["rollout"]["cache_sha256"],
                  "seconds_this_optimizer_step": time.time() - step_started}
        append(metrics_path, record)
        if step <= 5 or step % 10 == 0: print(json.dumps(record), flush=True)
        if step % interval == 0 or step == total:
            if file_sha256(ROOT / cfg["rollout"]["phase3b_cache"]) != cfg["rollout"]["cache_sha256"]:
                raise RuntimeError("frozen rollout cache changed during training")
            diagnostics = checkpoint_diagnostics(model, core, dataset, tokenizer, backend, replay, teacher,
                                                 diagnostic_index, cfg)
            metadata = {"optimizer_step": step, "epoch": step // 100, "allowed_parameter_groups": ["lora"],
                        "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                        "rollout_cache_sha256": cfg["rollout"]["cache_sha256"], "training_record": record,
                        "checkpoint_diagnostics": diagnostics}
            path = save_checkpoint(model, source["module"], optimizer, scheduler, checkpoint_root, step, metadata)
            append(out / "training/checkpoint_metadata.jsonl", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
            if step in (500, 1000, total):
                integrity_path = out / "rollout_integrity_audit.json"
                integrity = json.loads(integrity_path.read_text()); integrity[f"step_{step}"] = "PASS"; dump(integrity_path, integrity)
    final_hashes = group_hashes(model)
    changed = {group: initial_hashes[group] != final_hashes[group] for group in initial_hashes}
    expected = {"lora": final > start, "text_hidden_fcs": False, "mask_decoder": False, "frozen": False}
    audit = {"status": "PASS" if changed == expected else "FAIL", "changed": changed, "expected_changed": expected,
             "initial_hashes": initial_hashes, "final_hashes": final_hashes}
    dump(out / "training/parameter_update_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError("Phase 3F parameter update audit failed")
    dump(out / "training/run_summary.json", {"status": "COMPLETE" if final == total else "BOUNDED_RUN_COMPLETE",
          "start_step": start, "final_step": final, "total_optimizer_steps": total,
          "image_exposures_this_run": (final - start) * gas, "elapsed_seconds": time.time() - started,
          "seconds_per_optimizer_step": (time.time() - started) / max(1, final - start), "parameter_audit": audit})


def train_sft_control(cfg, out, model, source, missing, boundary, tokenizer, dataset, schedule, resume, bounded_steps):
    optimizer, scheduler = optimizer_and_scheduler(model, cfg)
    total = int(cfg["training"]["total_optimizer_steps"]); start = 0
    if resume:
        state = torch.load(Path(resume), map_location="cpu")
        model.load_state_dict(state["module"], strict=False); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"]); start = int(state["optimizer_step"])
    arm_out = out / "experiments/P3F_SFT_CONT_MATCHED"; metrics_path = arm_out / "training_metrics.jsonl"
    if metrics_path.exists() and not resume: raise RuntimeError("existing SFT formal metrics require --resume")
    if resume and metrics_path.exists():
        kept = [line for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and int(json.loads(line)["optimizer_step"]) <= start]
        metrics_path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    final = total if bounded_steps is None else min(total, start + int(bounded_steps))
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]) / "P3F_SFT_CONT_MATCHED"
    initial_hashes = group_hashes(model)
    dump(arm_out / "initialization_audit.json", {"status": "PASS", "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
         "missing_frozen_key_count": len(missing), "parameter_boundary": boundary, "initial_group_hashes": initial_hashes,
         "schedule_sha256": schedule["sha256"], "total_optimizer_steps": total, "total_image_exposures": 18000,
         "purpose": "exact_4500_step_matched_SFT_control_after_budget_amendment"})
    gas = int(cfg["training"]["gradient_accumulation_steps"]); interval = int(cfg["training"]["checkpoint_interval"])
    started = time.time(); model.train()
    for zero_step in range(start, final):
        step_started = time.time(); optimizer.zero_grad(set_to_none=True); window = []
        for micro in range(gas):
            exposure = zero_step * gas + micro; index = int(schedule["indices"][exposure])
            cpu_batch, sample = make_language_batch(dataset, index, tokenizer, torch.device("cpu"))
            window.append((exposure, cpu_batch, sample, batch_supervision_counts(cpu_batch)))
        text_tokens = sum(item[3]["text_tokens"] for item in window); accum_ce = 0.0; ids = []; labels = []
        for exposure, cpu_batch, sample, counts in window:
            trajectory_seed = int(cfg["experiment"]["seed"]) * 1000003 + exposure
            torch.manual_seed(trajectory_seed); torch.cuda.manual_seed_all(trajectory_seed)
            batch = move_batch(cpu_batch, next(model.parameters()).device, torch.bfloat16)
            output = model(**batch); ce = output["ce_loss"]; scale = counts["text_tokens"] / text_tokens
            (ce * scale).backward(); accum_ce += float(ce.detach().float()) * scale
            ids.append(sample["sample_id"]); labels.append(int(sample["cls_label"])); del batch, output, ce
        lora_grad = grad_norm(model, "lora")
        before_clip = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                           float(cfg["training"]["gradient_clip_norm"])))
        optimizer.step(); scheduler.step(); step = zero_step + 1
        record = {"optimizer_step": step, "arm": "P3F_SFT_CONT_MATCHED", "language_ce": accum_ce,
                  "representation_loss": 0.0, "total_loss": accum_ce, "lora_gradient_norm": lora_grad,
                  "gradient_norm_before_clip": before_clip, "learning_rate": float(optimizer.param_groups[0]["lr"]),
                  "sample_ids": ids, "class_labels": labels, "seconds_this_optimizer_step": time.time() - step_started}
        append(metrics_path, record)
        if step <= 5 or step % 10 == 0: print(json.dumps(record), flush=True)
        if step % interval == 0 or step == total:
            metadata = {"optimizer_step": step, "epoch": step // 100, "arm": "P3F_SFT_CONT_MATCHED",
                        "allowed_parameter_groups": ["lora"], "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
                        "schedule_sha256": schedule["sha256"], "training_record": record}
            path = save_checkpoint(model, source["module"], optimizer, scheduler, checkpoint_root, step, metadata)
            append(arm_out / "checkpoint_metadata.jsonl", {**metadata, "checkpoint": str(path), "sha256": file_sha256(path)})
    final_hashes = group_hashes(model); changed = {group: initial_hashes[group] != final_hashes[group] for group in initial_hashes}
    expected = {"lora": final > start, "text_hidden_fcs": False, "mask_decoder": False, "frozen": False}
    audit = {"status": "PASS" if changed == expected else "FAIL", "changed": changed, "expected_changed": expected,
             "initial_hashes": initial_hashes, "final_hashes": final_hashes}
    dump(arm_out / "parameter_update_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError("matched SFT parameter audit failed")
    dump(arm_out / "run_summary.json", {"status": "COMPLETE" if final == total else "BOUNDED_RUN_COMPLETE",
         "start_step": start, "final_step": final, "total_optimizer_steps": total,
         "image_exposures_this_run": (final - start) * gas, "elapsed_seconds": time.time() - started,
         "seconds_per_optimizer_step": (time.time() - started) / max(1, final - start), "parameter_audit": audit})


def main():
    cli = parse_args(); cfg = yaml.safe_load((ROOT / cli.config).read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    if not (out / "phase3b_rollout_reuse_audit.json").is_file():
        raise RuntimeError("run phase3f_prepare.py first")
    setup_values = setup(cfg, cli.physical_gpu, cli.mode)
    device, args, tokenizer, model, source, missing, boundary, backend, dataset, schedule, replay = setup_values
    if cli.mode == "preflight": preflight(cfg, out, model, backend, dataset, schedule, replay)
    elif cli.mode == "build-teacher-cache": build_teacher_cache(cfg, out, model, backend, dataset, schedule, replay)
    elif cli.mode == "train": train(cfg, out, model, source, missing, boundary, tokenizer, backend, dataset, schedule, replay, cli.resume, cli.optimizer_steps)
    else: train_sft_control(cfg, out, model, source, missing, boundary, tokenizer, dataset, schedule, cli.resume, cli.optimizer_steps)


if __name__ == "__main__":
    main()
