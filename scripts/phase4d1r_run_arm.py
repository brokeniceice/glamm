#!/usr/bin/env python3
"""Train and evaluate one corrected Phase 4D-1R arm."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.position_aware_evidence_reader import PositionAwareEvidenceReader
from scripts.phase4c_b_train import frozen_hash, load_p1
from scripts.phase4d1_run_arm import (cache_paths, evaluate_validation, feature_key,
                                     optimizer_scheduler, validation_evidence)
from tools.phase4c_b import decode_low_res, file_sha256, metric_record, sam_input_loss, summarize
from tools.phase4d1 import canonical_hash, dump, reader_state_hash, write_jsonl


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("pos_clip", "pos_forensic"), required=True)
    p.add_argument("--device", required=True)
    return p.parse_args()


def vector_norm(parameters):
    values = [p.grad.detach().float().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.cat(values).norm()) if values else 0.0


def mha_slice_norm(reader, start, end):
    values = []
    if reader.cross_attention.in_proj_weight.grad is not None:
        values.append(reader.cross_attention.in_proj_weight.grad[start:end].float().reshape(-1))
    if reader.cross_attention.in_proj_bias.grad is not None:
        values.append(reader.cross_attention.in_proj_bias.grad[start:end].float().reshape(-1))
    return float(torch.cat(values).norm()) if values else 0.0


def gradient_stats(reader):
    return {
        "reader_body_grad_norm": vector_norm([p for n, p in reader.named_parameters() if n != "beta"]),
        "query_projection_grad_norm": mha_slice_norm(reader, 0, 256),
        "key_projection_grad_norm": mha_slice_norm(reader, 256, 512),
        "value_projection_grad_norm": mha_slice_norm(reader, 512, 768),
        "output_projection_grad_norm": vector_norm(reader.cross_attention.out_proj.parameters()),
        "beta_grad_abs": abs(float(reader.beta.grad)) if reader.beta.grad is not None else 0.0,
    }


def fixed_batch(path, arm, device):
    shard = torch.load(path, map_location="cpu")
    rows = []
    for i in range(4):
        rows.append({
            "sample_id": shard["sample_ids"][i],
            "q": shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256),
            "feature": shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32),
            "sam": shard["sam_features"][i].to(device=device, dtype=torch.bfloat16),
            "target": shard["targets"][i].to(device),
        })
    return rows


def signal_stats(reader, batch, permutation):
    values = defaultdict(float)
    surviving = []
    reader.eval()
    with torch.no_grad():
        for row in batch:
            q, feature = row["q"], row["feature"]
            matched = reader(q, feature)
            shuffled = reader(q, feature.flatten(2).transpose(1, 2)[:, permutation, :])
            m = matched["q_final"].to(torch.bfloat16)
            s = shuffled["q_final"].to(torch.bfloat16)
            values["query_norm"] += float(q.norm())
            values["raw_reader_residual_norm"] += float(matched["q_evidence"].float().norm())
            values["effective_residual_norm"] += float(matched["q_residual"].float().norm())
            values["matched_shuffle_fp32_q_difference"] += float((matched["q_final"] - shuffled["q_final"]).float().norm())
            surviving.append(bool((m != s).any()))
    reader.train()
    n = len(batch)
    result = {k: v / n for k, v in values.items()}
    result["effective_residual_over_query_norm"] = result["effective_residual_norm"] / result["query_norm"]
    result["matched_shuffle_bf16_survival"] = float(np.mean(surviving))
    return result


def step0_gradient(reader, core, batch):
    reader.train(); reader.zero_grad(set_to_none=True)
    sums = defaultdict(float)
    for row in batch:
        value = reader(row["q"], row["feature"])
        low = decode_low_res(core, value["q_final"].to(torch.bfloat16), row["sam"])
        losses = sam_input_loss(low, row["target"])
        (losses["total"] / len(batch)).backward()
        for key in ("total", "bce", "dice"):
            sums[key] += float(losses[key].detach())
    result = {key: value / len(batch) for key, value in sums.items()}
    result.update(gradient_stats(reader))
    reader.zero_grad(set_to_none=True)
    return result


def save_checkpoint(path, reader, optimizer, scheduler, step, exposure, base_hash, corrected_hash, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "phase4d1r_corrected_position_reader_v1", "arm": arm,
        "step": step, "exposure": exposure, "reader": reader.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "parent_initial_hash": base_hash, "corrected_initial_hash": corrected_hash,
        "only_changed_variable": "beta_initialization: 0.0 -> 0.03",
    }, path)


def train(cfg, parent_cfg, bcfg, out, arm, device, core):
    cache_root = Path(cfg["parent_phase4d1"]["cache_root"])
    paths, cache_complete = cache_paths(cache_root)
    init = torch.load(cfg["parent_phase4d1"]["reader_init"], map_location="cpu")
    reader = PositionAwareEvidenceReader().to(device)
    reader.load_state_dict(init["reader"])
    base_hash = reader_state_hash(reader)
    preflight = json.loads((out / "preflight_manifest.json").read_text())
    if base_hash != preflight["parent_reader_init_hash"]:
        raise RuntimeError("parent Reader init hash mismatch")
    reader.beta.data.fill_(float(cfg["intervention"]["corrected_beta_initialization"]))
    corrected_hash = reader_state_hash(reader)
    if (corrected_hash != preflight["corrected_reader_init_hash"] or
            abs(float(reader.beta) - 0.03) > 1e-7):
        raise RuntimeError("corrected Reader init mismatch")

    optimizer, scheduler = optimizer_scheduler(reader, parent_cfg)
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]) / arm
    training_root = out / "training" / arm
    training_root.mkdir(parents=True, exist_ok=True)
    permutation = torch.tensor(json.loads((out / "spatial_content_permutation.json").read_text())["permutation"], device=device)
    batch = fixed_batch(paths[0], arm, device)
    diagnostics = []
    zero = step0_gradient(reader, core, batch)
    zero.update(signal_stats(reader, batch, permutation))
    zero.update({"arm": arm, "step": 0, "exposure": 0, "beta": float(reader.beta),
                 "learning_rate": 0.0, "elapsed_seconds": 0.0})
    diagnostics.append(zero)
    save_checkpoint(checkpoint_root / "step_0.pt", reader, optimizer, scheduler, 0, 0,
                    base_hash, corrected_hash, arm)
    dump(training_root / "diagnostics.json", diagnostics)

    diagnostic_steps = set(cfg["training"]["diagnostic_steps"])
    checkpoint_steps = set(cfg["training"]["checkpoint_steps"])
    interval = defaultdict(float); interval_n = 0
    order = []; exposure = 0; global_step = 0; started = time.time()
    reader.train(); optimizer.zero_grad(set_to_none=True)
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        for i, sid in enumerate(shard["sample_ids"]):
            q = shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256)
            feature = shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32)
            value = reader(q, feature)
            low = decode_low_res(core, value["q_final"].to(torch.bfloat16),
                                 shard["sam_features"][i].to(device=device, dtype=torch.bfloat16))
            losses = sam_input_loss(low, shard["targets"][i].to(device))
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite loss")
            (losses["total"] / int(cfg["training"]["effective_batch_size"])).backward()
            exposure += 1; interval_n += 1; order.append(sid)
            for key in ("total", "bce", "dice"):
                interval[key] += float(losses[key].detach())
            if exposure % int(cfg["training"]["effective_batch_size"]):
                continue
            grads = gradient_stats(reader)
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite gradient")
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step in diagnostic_steps:
                row = {key: value / interval_n for key, value in interval.items()}
                row.update(grads); row.update(signal_stats(reader, batch, permutation))
                row.update({"arm": arm, "step": global_step, "exposure": exposure,
                            "beta": float(reader.beta.detach()),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "elapsed_seconds": time.time() - started})
                diagnostics.append(row); dump(training_root / "diagnostics.json", diagnostics)
                print(json.dumps(row), flush=True)
                interval = defaultdict(float); interval_n = 0
            if global_step in checkpoint_steps:
                save_checkpoint(checkpoint_root / f"step_{global_step}.pt", reader, optimizer,
                                scheduler, global_step, exposure, base_hash, corrected_hash, arm)
    if global_step != 512 or exposure != 2048:
        raise RuntimeError(f"formal endpoint mismatch: {global_step}/{exposure}")
    if canonical_hash(order) != cache_complete["sample_ids_sha256"]:
        raise RuntimeError("training order mismatch")
    dump(training_root / "training_order.json", {"n": len(order),
         "sample_ids_sha256": canonical_hash(order), "matches_parent": True})
    dump(training_root / "completion.json", {
        "status": "COMPLETE", "arm": arm, "formal_step": 512, "exposures": 2048,
        "parent_initial_hash": base_hash, "corrected_initial_hash": corrected_hash,
        "final_reader_hash": reader_state_hash(reader), "beta_final": float(reader.beta),
        "diagnostic_steps": [x["step"] for x in diagnostics], "selector_used": False,
        "internal_test_access": False, "official1000_access": False,
    })
    return reader, diagnostics


def all_train_evidence(cache_root, arm):
    evidence = {}
    paths, _ = cache_paths(cache_root)
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        for i, sid in enumerate(shard["sample_ids"]):
            evidence[sid] = shard[feature_key(arm)][i]
    return paths, evidence


def evaluate_train(reader, core, cache_root, out, arm, device):
    paths, evidence = all_train_evidence(cache_root, arm)
    mapping = json.loads((out / "train_cross_image_mapping.json").read_text())["mapping"]
    permutation = torch.tensor(json.loads((out / "spatial_content_permutation.json").read_text())["permutation"], device=device)
    conditions = ("matched", "cross_image", "spatial_shuffle", "zero")
    records = {c: [] for c in conditions}
    signal = {c: {"bf16": [], "logit": []} for c in conditions[1:]}
    reader.eval()
    with torch.no_grad():
        for path in paths:
            shard = torch.load(path, map_location="cpu")
            for i, sid in enumerate(shard["sample_ids"]):
                q = shard["q_seg"][i].to(device=device, dtype=torch.float32).reshape(1, 256)
                matched_feature = shard[feature_key(arm)][i:i+1].to(device=device, dtype=torch.float32)
                features = {
                    "matched": matched_feature,
                    "cross_image": evidence[mapping[sid]].to(device=device, dtype=torch.float32).unsqueeze(0),
                    "spatial_shuffle": matched_feature.flatten(2).transpose(1, 2)[:, permutation, :],
                    "zero": torch.zeros_like(matched_feature),
                }
                qs = {}; lows = {}
                for condition in conditions:
                    value = reader(q, features[condition]); qs[condition] = value["q_final"].to(torch.bfloat16)
                    lows[condition] = decode_low_res(core, qs[condition], shard["sam_features"][i].to(device=device, dtype=torch.bfloat16))[0, 0]
                    target = shard["targets"][i].to(device)
                    logits = F.interpolate(lows[condition][None, None].float(), size=target.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                    records[condition].append(metric_record(sid, logits, target))
                for condition in conditions[1:]:
                    signal[condition]["bf16"].append(bool((qs["matched"] != qs[condition]).any()))
                    signal[condition]["logit"].append(float((lows["matched"].float() - lows[condition].float()).abs().mean()))
    root = out / "train_subset_evaluation" / arm
    summaries = {}
    for condition in conditions:
        write_jsonl(root / f"{condition}.jsonl", records[condition])
        summaries[condition] = summarize(records[condition])
    signal_summary = {f"matched_vs_{condition}": {
        "n": 2048, "bf16_survival": float(np.mean(signal[condition]["bf16"])),
        "mean_sam_logit_absolute_difference": float(np.mean(signal[condition]["logit"])),
    } for condition in conditions[1:]}
    dump(root / "summary.json", summaries); dump(root / "signal_summary.json", signal_summary)
    return summaries, signal_summary


def main():
    args = cli()
    cfg = yaml.safe_load((ROOT / "configs/phase4d1r_corrected_position_aware.yaml").read_text())
    parent_cfg = yaml.safe_load((ROOT / cfg["parent_phase4d1"]["config"]).read_text())
    bcfg = yaml.safe_load((ROOT / parent_cfg["phase4c_b"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    if json.loads((out / "preflight_manifest.json").read_text())["status"] != "PASS":
        raise RuntimeError("PASS preflight required")
    device = torch.device(args.device); torch.cuda.set_device(device)
    model, core = load_p1(bcfg, device); before = frozen_hash(core)
    reader, diagnostics = train(cfg, parent_cfg, bcfg, out, args.arm, device, core)
    train_summary, train_signal = evaluate_train(reader, core, Path(cfg["parent_phase4d1"]["cache_root"]), out, args.arm, device)
    validation = evaluate_validation(parent_cfg, bcfg, reader, core, out, args.arm, device)
    if frozen_hash(core) != before:
        raise RuntimeError("frozen P1/SAM mutated")
    if file_sha256(Path(bcfg["p1"]["checkpoint"])) != bcfg["p1"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checkpoint mutated")
    dump(out / "arm_completion" / f"{args.arm}.json", {
        "status": "COMPLETE", "arm": args.arm, "formal_step": 512,
        "train_subset": train_summary, "train_signal": train_signal,
        "validation": validation, "P1_SAM_hash_unchanged": True,
        "P1_checkpoint_hash_unchanged": True, "internal_test_access": False,
        "official1000_access": False, "phase4d2_started": False,
    })
    print(json.dumps({"status": "COMPLETE", "arm": args.arm}, indent=2))


if __name__ == "__main__":
    main()
