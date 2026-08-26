#!/usr/bin/env python3
"""Train one matched Phase 4D-1 arm and run frozen train/validation controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.position_aware_evidence_reader import PositionAwareEvidenceReader
from scripts.phase4c_b_train import evaluate, frozen_hash, load_p1, pair_paths, q_index
from tools.phase4c_b import (decode_low_res, file_sha256, inverse_sam_logits,
                            metric_record, summarize)
from tools.phase4d1 import (canonical_hash, dump, read_jsonl, reader_state_hash,
                           write_csv, write_jsonl)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("pos_clip", "pos_forensic"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def grad_norm(reader, exclude_beta=False):
    values = []
    for name, parameter in reader.named_parameters():
        if exclude_beta and name == "beta":
            continue
        if parameter.grad is not None:
            values.append(float(parameter.grad.detach().float().square().sum()))
    return math.sqrt(sum(values)) if values else 0.0


def optimizer_scheduler(reader, cfg):
    opt = torch.optim.AdamW(
        reader.parameters(),
        lr=float(cfg["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    warm = int(cfg["optimizer"]["warmup_steps"])
    total = int(cfg["optimizer"]["schedule_total_steps"])

    def scale(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, scale)


def cache_paths(cache_root):
    complete = json.loads((cache_root / "train_subset/complete.json").read_text())
    if complete["status"] != "COMPLETE":
        raise RuntimeError("train subset cache is not COMPLETE")
    paths = [Path(row["path"]) for row in complete["manifest"]]
    if sum(row["n"] for row in complete["manifest"]) != 2048:
        raise RuntimeError("train subset cache count mismatch")
    return paths, complete


def feature_key(arm):
    return "clip_features" if arm == "pos_clip" else "forensic_features"


def loss_parts(low, target):
    from tools.phase4c_b import sam_input_loss
    return sam_input_loss(low, target)


def step0_diagnostic(reader, core, path, arm, device):
    shard = torch.load(path, map_location="cpu")
    sums = {k: 0.0 for k in ("bce", "dice", "total", "q_evidence_norm",
                                      "residual_norm", "output_norm")}
    reader.eval()
    with torch.no_grad():
        for i in range(min(4, len(shard["sample_ids"]))):
            q = shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256)
            evidence = shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32)
            value = reader(q, evidence)
            low = decode_low_res(core, value["q_final"].to(torch.bfloat16),
                                 shard["sam_features"][i].to(device=device, dtype=torch.bfloat16))
            losses = loss_parts(low, shard["targets"][i].to(device))
            for key in ("bce", "dice", "total"):
                sums[key] += float(losses[key])
            sums["q_evidence_norm"] += float(value["q_evidence"].float().norm())
            sums["residual_norm"] += float(value["q_residual"].float().norm())
            sums["output_norm"] += float(value["q_final"].float().norm())
    return {key: value / min(4, len(shard["sample_ids"])) for key, value in sums.items()}


def save_checkpoint(path, reader, optimizer, scheduler, step, exposure, init_hash, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "phase4d1_position_reader_v1", "arm": arm, "step": step,
        "exposure": exposure, "reader": reader.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "shared_initial_hash": init_hash,
    }, path)


def train(cfg, bcfg, out, cache_root, checkpoint_root, arm, device, core, model,
          frozen_before):
    paths, complete = cache_paths(cache_root)
    init = torch.load(checkpoint_root / "reader_init.pt", map_location="cpu")
    reader = PositionAwareEvidenceReader().to(device)
    reader.load_state_dict(init["reader"])
    initial_hash = reader_state_hash(reader)
    if initial_hash != init["reader_state_hash"]:
        raise RuntimeError("shared Reader initialization hash mismatch")
    optimizer, scheduler = optimizer_scheduler(reader, cfg)
    arm_root = checkpoint_root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    train_root = out / "training" / arm
    train_root.mkdir(parents=True, exist_ok=True)
    final_checkpoint = arm_root / "step_512.pt"
    diagnostics_path = train_root / "diagnostics.json"
    diagnostics = []
    global_step = 0
    exposure = 0
    if final_checkpoint.exists():
        saved = torch.load(final_checkpoint, map_location="cpu")
        reader.load_state_dict(saved["reader"])
        global_step = int(saved["step"])
        exposure = int(saved["exposure"])
        diagnostics = json.loads(diagnostics_path.read_text())
    else:
        zero = step0_diagnostic(reader, core, paths[0], arm, device)
        diagnostics.append({
            "arm": arm, "step": 0, "exposure": 0, **zero,
            "beta": float(reader.beta), "reader_grad_norm": 0.0,
            "beta_grad_abs": 0.0, "learning_rate": 0.0,
        })
        save_checkpoint(arm_root / "step_0.pt", reader, optimizer, scheduler, 0, 0,
                        initial_hash, arm)
        dump(diagnostics_path, diagnostics)
        reader.train()
        optimizer.zero_grad(set_to_none=True)
        interval = {key: 0.0 for key in ("bce", "dice", "total", "q_evidence_norm",
                                                  "residual_norm", "output_norm")}
        interval_samples = 0
        started = time.time()
        diagnostic_steps = set(cfg["training"]["diagnostic_steps"])
        checkpoint_steps = set(cfg["training"]["checkpoint_steps"])
        order = []
        for path in paths:
            shard = torch.load(path, map_location="cpu")
            for i, sid in enumerate(shard["sample_ids"]):
                q = shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256)
                evidence = shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32)
                value = reader(q, evidence)
                low = decode_low_res(core, value["q_final"].to(torch.bfloat16),
                                     shard["sam_features"][i].to(device=device, dtype=torch.bfloat16))
                losses = loss_parts(low, shard["targets"][i].to(device))
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError("non-finite Phase 4D-1 loss")
                (losses["total"] / int(cfg["training"]["effective_batch_size"])).backward()
                exposure += 1
                order.append(sid)
                interval_samples += 1
                for key in ("bce", "dice", "total"):
                    interval[key] += float(losses[key].detach())
                interval["q_evidence_norm"] += float(value["q_evidence"].detach().float().norm())
                interval["residual_norm"] += float(value["q_residual"].detach().float().norm())
                interval["output_norm"] += float(value["q_final"].detach().float().norm())
                if exposure % int(cfg["training"]["effective_batch_size"]) != 0:
                    continue
                beta_grad = abs(float(reader.beta.grad)) if reader.beta.grad is not None else 0.0
                reader_grad = grad_norm(reader, exclude_beta=True)
                norm = torch.nn.utils.clip_grad_norm_(reader.parameters(),
                                                       float(cfg["training"]["gradient_clip_norm"]))
                if not torch.isfinite(norm):
                    raise RuntimeError("non-finite Phase 4D-1 gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step in diagnostic_steps:
                    row = {key: value / interval_samples for key, value in interval.items()}
                    row.update({
                        "arm": arm, "step": global_step, "exposure": exposure,
                        "beta": float(reader.beta.detach()),
                        "reader_grad_norm": reader_grad,
                        "beta_grad_abs": beta_grad,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": time.time() - started,
                    })
                    diagnostics.append(row)
                    dump(diagnostics_path, diagnostics)
                    print(json.dumps(row), flush=True)
                    interval = {key: 0.0 for key in interval}
                    interval_samples = 0
                if global_step in checkpoint_steps:
                    save_checkpoint(arm_root / f"step_{global_step}.pt", reader, optimizer,
                                    scheduler, global_step, exposure, initial_hash, arm)
        if global_step != int(cfg["training"]["optimizer_steps"]) or exposure != 2048:
            raise RuntimeError(f"formal budget mismatch: step={global_step}, exposure={exposure}")
        if order != json.loads((out / "phase4d1_train_subset.json").read_text())["sample_ids"]:
            raise RuntimeError("runtime training order mismatch")
        dump(train_root / "training_order.json", {
            "n": len(order), "sample_ids_sha256": canonical_hash(order),
            "matches_frozen_subset": True,
        })
    final_hash = reader_state_hash(reader)
    completion_payload = {
        "status": "COMPLETE", "arm": arm, "formal_step": global_step,
        "exposures": exposure, "shared_initial_hash": initial_hash,
        "final_reader_hash": final_hash, "reader_changed": final_hash != initial_hash,
        "beta_final": float(reader.beta.detach()),
        "diagnostic_steps": [row["step"] for row in diagnostics],
        "P1_SAM_hash_unchanged": frozen_hash(core) == frozen_before,
        "P1_checkpoint_hash_unchanged": file_sha256(Path(bcfg["p1"]["checkpoint"])) == bcfg["p1"]["checkpoint_sha256"],
        "selector_used": False, "formal_endpoint": 512,
        "internal_test_access": False, "official1000_access": False,
    }
    dump(train_root / "completion.json", completion_payload)
    return reader, diagnostics, completion_payload


def selected_evidence(cache_root, arm):
    result = {}
    paths, _ = cache_paths(cache_root)
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        for i, sid in enumerate(shard["sample_ids"]):
            result[sid] = shard[feature_key(arm)][i]
    return result


def evaluate_train_subset(reader, core, cache_root, out, arm, device):
    evidence = selected_evidence(cache_root, arm)
    mapping = json.loads((out / "train_cross_image_mapping.json").read_text())["mapping"]
    permutation = torch.tensor(json.loads((out / "spatial_content_permutation.json").read_text())["permutation"], device=device)
    paths, _ = cache_paths(cache_root)
    summaries = {}
    for condition in ("matched", "cross_image", "spatial_shuffle"):
        records = []
        reader.eval()
        with torch.no_grad():
            for path in paths:
                shard = torch.load(path, map_location="cpu")
                for i, sid in enumerate(shard["sample_ids"]):
                    if condition == "cross_image":
                        feature = evidence[mapping[sid]].to(device=device, dtype=torch.float32).unsqueeze(0)
                    else:
                        feature = shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32)
                    if condition == "spatial_shuffle":
                        feature = feature.flatten(2).transpose(1, 2)[:, permutation, :]
                    q = shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256)
                    value = reader(q, feature)
                    low = decode_low_res(core, value["q_final"].to(torch.bfloat16),
                                         shard["sam_features"][i].to(device=device, dtype=torch.bfloat16))
                    # Train-subset sanity is measured in the frozen SAM training
                    # frame because train caches intentionally omit original masks.
                    target = shard["targets"][i].to(device)
                    logits = F.interpolate(
                        low.float(), size=target.shape[-2:], mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    records.append(metric_record(sid, logits,
                                                 target))
        root = out / "train_subset_evaluation" / arm
        write_jsonl(root / f"{condition}.jsonl", records)
        summaries[condition] = summarize(records)
    dump(out / "train_subset_evaluation" / arm / "summary.json", summaries)
    return summaries


def validation_evidence(cfg, arm):
    path = Path(cfg["phase4c_c"]["clip_evidence"] if arm == "pos_clip"
                else cfg["phase4c_c"]["forensic_evidence"])
    state = torch.load(path, map_location="cpu")
    if len(state["sample_ids"]) != 1106:
        raise RuntimeError("validation evidence count mismatch")
    return {sid: state["features"][i] for i, sid in enumerate(state["sample_ids"])}


def evaluate_validation(cfg, bcfg, reader, core, out, arm, device):
    evidence = validation_evidence(cfg, arm)
    q_val = q_index(Path(bcfg["experiment"]["cache_root"]) / "validation", "G0")
    pairs = pair_paths(ROOT / bcfg["frozen_spatial_cache"]["phase3c1_root"], "val")
    mapping = json.loads((out / "validation_cross_image_mapping.json").read_text())["mapping"]
    permutation = torch.tensor(json.loads((out / "spatial_content_permutation.json").read_text())["permutation"], device=device)

    def transform(condition):
        if condition == "cross_image":
            return lambda sid, value: evidence[mapping[sid]].to(device=device, dtype=torch.float32).unsqueeze(0)
        if condition == "spatial_shuffle":
            return lambda sid, value: value.flatten(2).transpose(1, 2)[:, permutation, :]
        if condition == "zero":
            return lambda sid, value: torch.zeros_like(value)
        return None

    summaries = {}
    for condition in cfg["evaluation"]["conditions"]:
        metrics, records, _, _, diagnostics = evaluate(
            reader, None, core, q_val, pairs, device,
            evidence_by_id=evidence, evidence_transform=transform(condition),
        )
        root = out / "validation" / arm
        write_jsonl(root / f"{condition}.jsonl", records)
        dump(root / f"{condition}_metrics.json", metrics)
        summaries[condition] = {"metrics": metrics, "diagnostics": diagnostics}
        print(json.dumps({"arm": arm, "condition": condition,
                          "mean_iou": metrics["mean_foreground_iou"]}), flush=True)
    dump(out / "validation" / arm / "summary.json", summaries)
    return summaries


def main():
    args = cli()
    cfg = yaml.safe_load((ROOT / "configs/phase4d1_position_aware_evidence.yaml").read_text())
    bcfg = yaml.safe_load((ROOT / cfg["phase4c_b"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    cache_root = Path(cfg["experiment"]["cache_root"])
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"])
    preflight = json.loads((out / "preflight_manifest.json").read_text())
    if preflight["status"] != "PASS":
        raise RuntimeError("PASS Phase 4D-1 preflight required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, core = load_p1(bcfg, device)
    frozen_before = frozen_hash(core)
    reader, diagnostics, completion = train(cfg, bcfg, out, cache_root,
                                            checkpoint_root, args.arm, device,
                                            core, model, frozen_before)
    train_summary = evaluate_train_subset(reader, core, cache_root, out, args.arm, device)
    val_summary = evaluate_validation(cfg, bcfg, reader, core, out, args.arm, device)
    if frozen_hash(core) != frozen_before:
        raise RuntimeError("frozen P1/SAM hash changed")
    dump(out / "arm_completion" / f"{args.arm}.json", {
        "status": "COMPLETE", "arm": args.arm, "training": completion,
        "train_subset": train_summary, "validation": val_summary,
        "shared_initial_hash": preflight["reader_init_hash"],
        "internal_test_access": False, "official1000_access": False,
    })
    print(json.dumps({"status": "COMPLETE", "arm": args.arm}, indent=2))


if __name__ == "__main__":
    main()
