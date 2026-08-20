#!/usr/bin/env python3
"""Train and evaluate the fixed one-layer Phase 3C.1 spatial probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3c1 import (
    SOURCES, binary_metrics, inverse_logits, paired_statistics, probe_loss, summarize,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--config", default="configs/phase3c1_spatial_probe.yaml")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("val", "test", "official1000"), default="val")
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def config_and_root(path):
    config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    output = (ROOT / config["experiment"]["output_root"]).resolve()
    return config, output


def shards(output: Path, source: str, split: str) -> list[Path]:
    root = output / "cache" / source / split
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if complete["status"] != "COMPLETE" or not complete["source_parameter_hash_exact"]:
        raise RuntimeError(f"invalid cache completion gate: {root}")
    values = sorted(root.glob("shard_*.pt"))
    if len(values) != int(complete["shards"]):
        raise RuntimeError(f"cache shard count mismatch: {len(values)} != {complete['shards']}")
    return values


def load_shard(path: Path):
    value = torch.load(path, map_location="cpu")
    if value.get("schema") != "phase3c1_spatial_cache_v1":
        raise RuntimeError(f"unexpected cache schema: {path}")
    return value


def selected_path(output, source):
    return output / "probes" / source / "selected.pt"


def original_records(probe, paths, device, *, keep_low_logits=False):
    probe.eval()
    records, low_logits = [], []
    with torch.no_grad():
        for path in paths:
            shard = load_shard(path)
            features = shard["features"]
            for index, metadata in enumerate(shard["records"]):
                feature = features[index:index + 1].to(device=device, dtype=torch.float32)
                low = probe(feature)[0, 0]
                original_logits = inverse_logits(low, metadata["geometry"])
                original = shard["original_masks"][index].to(device)
                metric = binary_metrics(original_logits, original)
                records.append({
                    "sample_id": metadata["sample_id"], "source": metadata["source"],
                    "geometry": metadata["geometry"], **metric,
                })
                if keep_low_logits:
                    low_logits.append(low.detach().cpu())
    return records, low_logits


def train(config, output, source, device):
    train_paths, val_paths = shards(output, source, "train"), shards(output, source, "val")
    cache_meta = json.loads((output / "cache" / source / "train" / "complete.json").read_text())
    channels = int(cache_meta["feature_shape"][0])
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(3407)
    probe = nn.Conv2d(channels, 1, kernel_size=1, bias=True).to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["probe"]["learning_rate"]),
        weight_decay=float(config["probe"]["weight_decay"]),
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_ids = {id(p) for p in trainable}
    if optimizer_ids != trainable_ids or sum(p.numel() for p in trainable) != channels + 1:
        raise RuntimeError("probe-only optimizer contract violated")
    batch_size = int(config["probe"]["batch_size"][source])
    epochs = int(config["probe"]["epochs"])
    history, best = [], None
    probe_root = output / "probes" / source
    probe_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.time()
        probe.train()
        rng = random.Random(3407 + epoch)
        epoch_paths = list(train_paths); rng.shuffle(epoch_paths)
        sums = {"bce": 0.0, "dice": 0.0, "total": 0.0, "samples": 0}
        for shard_path in epoch_paths:
            shard = load_shard(shard_path)
            order = list(range(len(shard["records"]))); rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                features = shard["features"][indices].to(device=device, dtype=torch.float32)
                targets = shard["targets"][indices, None].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                losses = probe_loss(probe(features), targets)
                losses["total"].backward()
                optimizer.step()
                count = len(indices)
                for key in ("bce", "dice", "total"):
                    sums[key] += float(losses[key].detach()) * count
                sums["samples"] += count
        if sums["samples"] != int(config["data"]["train_fake"]):
            raise RuntimeError(f"traditional epoch sample count mismatch: {sums['samples']}")
        val_records, _ = original_records(probe, val_paths, device)
        val = summarize(val_records)
        row = {
            "epoch": epoch, "traditional_full_data_pass": True,
            "train_samples": sums["samples"], "batch_size": batch_size,
            "train_bce": sums["bce"] / sums["samples"],
            "train_dice": sums["dice"] / sums["samples"],
            "train_total": sums["total"] / sums["samples"],
            "val": val, "seconds": time.time() - started,
        }
        history.append(row); dump(probe_root / "history.json", history)
        candidate = (val["mean_foreground_iou"], val["mean_foreground_f1"])
        payload = {
            "schema": "phase3c1_linear_probe_v1", "source": source, "epoch": epoch,
            "channels": channels, "state_dict": probe.state_dict(), "optimizer": optimizer.state_dict(),
            "selector": candidate, "hyperparameters": config["probe"],
            "trainable_parameter_count": sum(p.numel() for p in trainable),
            "optimizer_parameter_ids_equal_trainable_ids": True,
        }
        torch.save(payload, probe_root / "last.pt")
        if best is None or candidate > best:
            best = candidate
            torch.save(payload, selected_path(output, source))
        print(json.dumps({"source": source, **row}, ensure_ascii=False), flush=True)
    selected = torch.load(selected_path(output, source), map_location="cpu")
    manifest = {
        "status": "COMPLETE", "source": source, "architecture": "Conv2d(C,1,1,bias=True)",
        "channels": channels, "trainable_parameters": channels + 1, "seed": 3407,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.0,
        "epochs": epochs, "full_data_passes": epochs, "early_stopping": False,
        "batch_size": batch_size, "loss": "2.0*BCE+0.5*Dice",
        "selected_epoch": int(selected["epoch"]), "selected_val": {
            "mean_foreground_iou": selected["selector"][0],
            "mean_foreground_f1": selected["selector"][1],
        },
    }
    dump(probe_root / "training_manifest.json", manifest)


def deterministic_permutation(size: int, sample_id: str):
    digest = hashlib.sha256(f"3407:{sample_id}:spatial".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    return torch.randperm(size, generator=generator)


def persistent_ids(output):
    destination = output / "audit" / "persistent476_ids.json"
    phase3c0 = ROOT / "outputs/phase3c0_residual_diagnosis/paired/paired_conditions.jsonl"
    rows = [json.loads(line) for line in phase3c0.read_text(encoding="utf-8").splitlines() if line]
    ids = [
        row["sample_id"] for row in rows
        if float(row["A"]["foreground_iou"]) <= 0.30
        and float(row["C"]["foreground_iou"]) <= 0.30
        and float(row["D"]["foreground_iou"]) <= 0.30
    ]
    if len(ids) != 476:
        raise RuntimeError(f"frozen Phase 3C.0 persistent membership mismatch: {len(ids)}")
    payload = {
        "source_artifact": str(phase3c0.resolve()),
        "membership_rule_from_phase3c0": "stored A<=0.30 AND stored C<=0.30 AND stored D<=0.30",
        "membership_uses_phase3c1_results": False, "n": len(ids), "sample_ids": ids,
    }
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["sample_ids"] != ids:
            raise RuntimeError("persistent membership changed")
    dump(destination, payload)
    return set(ids)


def evaluate(config, output, source, device, split="val"):
    val_paths = shards(output, source, split)
    checkpoint = torch.load(selected_path(output, source), map_location="cpu")
    probe = nn.Conv2d(int(checkpoint["channels"]), 1, 1, bias=True)
    probe.load_state_dict(checkpoint["state_dict"]); probe.to(device).eval()
    normal, low = original_records(probe, val_paths, device, keep_low_logits=True)
    originals, metadata = [], []
    for path in val_paths:
        shard = load_shard(path)
        originals.extend(shard["original_masks"]); metadata.extend(shard["records"])
    if [row["sample_id"] for row in normal] != [row["sample_id"] for row in metadata]:
        raise RuntimeError("validation ordering mismatch")
    controls = {"spatial_shuffle": [], "sample_shuffle": [], "global_broadcast": []}
    for index, meta in enumerate(metadata):
        current = low[index]
        permutation = deterministic_permutation(current.numel(), meta["sample_id"])
        spatial = current.flatten()[permutation].reshape_as(current)
        sample = low[(index - 1) % len(low)]  # fixed deterministic cyclic permutation
        broadcast = current.mean().expand_as(current)
        for name, value in (("spatial_shuffle", spatial), ("sample_shuffle", sample),
                            ("global_broadcast", broadcast)):
            logits = inverse_logits(value.to(device), meta["geometry"])
            metric = binary_metrics(logits, originals[index].to(device))
            controls[name].append({"sample_id": meta["sample_id"], "source": source, **metric})
    if split != "val":
        records = normal
        destination = output / "confirmatory" / split / source
        write_jsonl(destination / "predictions.jsonl", records)
        dump(destination / "metrics.json", {
            **summarize(records), "split": split,
            "label": "CONFIRMATORY_ONLY_NOT_USED_FOR_ROUTE_SELECTION",
            "selected_probe_frozen_before_evaluation": True,
        })
        print(json.dumps({"source": source, "split": split, "metrics": summarize(records)},
                         ensure_ascii=False, indent=2))
        return
    persistent = persistent_ids(output)
    validation_dir = output / "validation" / source
    write_jsonl(validation_dir / "normal_predictions.jsonl", normal)
    torch.save({"sample_ids": [row["sample_id"] for row in normal],
                "low_res_logits": torch.stack(low)}, validation_dir / "selected_low_res_logits.pt")
    for name, records in controls.items():
        write_jsonl(output / "controls" / source / f"{name}.jsonl", records)
    dump(output / "controls" / source / "manifest.json", {
        "selected_probe_reused_without_retraining": True,
        "spatial_shuffle": "deterministic per-sample permutation of HxW; applied to low logits, exactly equivalent to permuting channel vectors before a shared 1x1 linear probe",
        "sample_shuffle": "deterministic one-position cyclic permutation; applying the same 1x1 probe before permutation is exactly equivalent to feature permutation before probe",
        "global_broadcast": "mean low logit broadcast; for Conv2d(C,1,1,bias=True), mean(probe(feature)) equals probe(mean(feature)) exactly",
        "geometry": "each control prediction uses the target sample inverse geometry; unseen crop exterior forced background",
        "seed": 3407,
    })
    normal_persistent = [row for row in normal if row["sample_id"] in persistent]
    metrics = {"all_val_fake": summarize(normal), "persistent476": summarize(normal_persistent)}
    dump(validation_dir / "metrics.json", metrics)
    stats = {}
    for name, records in controls.items():
        by_id = {row["sample_id"]: row for row in records}
        stats[name] = {}
        for population, selected in (
            ("all_val_fake", normal), ("persistent476", normal_persistent),
        ):
            stats[name][population] = paired_statistics(
                [row["foreground_iou"] for row in selected],
                [by_id[row["sample_id"]]["foreground_iou"] for row in selected],
            )
    dump(output / "statistics" / source / "negative_controls.json", stats)
    print(json.dumps({"source": source, "metrics": metrics}, ensure_ascii=False, indent=2))


def main(argv=None):
    cli = parse_args(argv)
    config, output = config_and_root(cli.config)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if cli.command == "train":
        train(config, output, cli.source, device)
    else:
        evaluate(config, output, cli.source, device, cli.split)


if __name__ == "__main__":
    main()
