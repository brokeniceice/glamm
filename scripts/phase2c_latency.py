#!/usr/bin/env python3
"""Bounded online latency audit for frozen NPR/SRM and fusion heads."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npr_expert.transforms import build_npr_transform
from scripts.phase2c_forensic_fusion import OUTPUT_ROOT, VARIANT_DIRS, load_config, load_expert, load_fusion, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    return parser.parse_args(argv)


def benchmark(call, device, iterations=5):
    for _ in range(2): call()
    if device.type == "cuda": torch.cuda.synchronize(device)
    before = time.perf_counter()
    for _ in range(iterations): call()
    if device.type == "cuda": torch.cuda.synchronize(device)
    return time.perf_counter() - before


def main(argv=None):
    cli = parse_args(argv)
    device = torch.device(cli.device)
    cache = torch.load(OUTPUT_ROOT / "feature_cache/test.pt", map_location="cpu")
    count = min(cli.samples, len(cache["labels"]))
    transform = build_npr_transform(training=False)
    images = []
    for path in cache["image_paths"][:count]:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    expert, _ = load_expert(load_config("configs/phase2c_npr_srm.yaml"), device)
    # Batch-size one matches the formal Phase 2A detector latency protocol.
    tensors = [image.unsqueeze(0).to(device) for image in images]
    with torch.no_grad():
        npr_seconds = benchmark(lambda: [expert.extract_npr_features(x) for x in tensors], device)
        srm_seconds = benchmark(lambda: [expert.extract_srm_features(x) for x in tensors], device)
        joint_seconds = benchmark(lambda: [expert.extract_branch_features(x) for x in tensors], device)
    phase2a_ms = float(cache["metadata"]["glamm_ms_per_image"])
    report = {
        "samples": count, "batch_size": 1, "preprocessing_excluded": True,
        "phase2a_baseline_ms_per_image": phase2a_ms,
        "npr_ms_per_image": 1000 * npr_seconds / (5 * count),
        "srm_ms_per_image": 1000 * srm_seconds / (5 * count),
        "npr_srm_joint_ms_per_image": 1000 * joint_seconds / (5 * count),
        "models": {},
    }
    for variant, dirname in VARIANT_DIRS.items():
        model, _ = load_fusion(OUTPUT_ROOT / dirname, device)
        kwargs = {
            "base_logits": cache["base_logits"][:count].to(device),
            "h_cls": cache["h_cls"][:count].to(device),
            "npr": cache["npr"][:count].to(device), "srm": cache["srm"][:count].to(device),
        }
        with torch.no_grad(): seconds = benchmark(lambda: model(**kwargs), device, iterations=100)
        fusion_ms = 1000 * seconds / (100 * count)
        expert_ms = (report["npr_ms_per_image"] if variant == "npr" else
                     report["srm_ms_per_image"] if variant == "srm" else
                     report["npr_srm_joint_ms_per_image"])
        total = phase2a_ms + expert_ms + fusion_ms
        report["models"][dirname] = {
            "fusion_ms_per_image": fusion_ms, "extra_expert_plus_fusion_ms": expert_ms + fusion_ms,
            "estimated_total_ms_per_image": total,
            "latency_increase_percent": 100 * (total - phase2a_ms) / phase2a_ms,
        }
    write_json(OUTPUT_ROOT / "latency_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
