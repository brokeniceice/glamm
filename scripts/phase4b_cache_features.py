#!/usr/bin/env python3
"""Precompute immutable pre-logit FEPN global features for train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.fepn import FEPNv0
from tools.phase4a import Phase4ADataset, phase4a_collate
from tools.phase4b import dump, file_sha256, tensor_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4b_global_fepn_injection.yaml")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    cli = parser.parse_args()
    cfg = yaml.safe_load((ROOT / cli.config).read_text(encoding="utf-8"))
    if cli.physical_gpu != int(cfg["runtime"]["cache_gpu"]):
        raise RuntimeError("feature cache GPU differs from frozen config")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", str(cli.physical_gpu)):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}, expected {cli.physical_gpu}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    fepn_path = Path(cfg["fepn"]["checkpoint"])
    if file_sha256(fepn_path) != cfg["fepn"]["checkpoint_sha256"]:
        raise RuntimeError("FEPN checkpoint hash mismatch")
    phase4a_cfg = yaml.safe_load((ROOT / "configs/phase4a_fepn_evidence_learnability.yaml").read_text())
    model = FEPNv0(phase4a_cfg["preprocess"]["image_mean"], phase4a_cfg["preprocess"]["image_std"])
    state = torch.load(fepn_path, map_location="cpu"); model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
    cache_root = Path(cfg["fepn"]["cache_root"]); cache_root.mkdir(parents=True, exist_ok=True)
    records = {}; started_all = time.time()
    for split in ("train", "val"):
        destination = cache_root / f"{split}.pt"
        if destination.exists():
            saved = torch.load(destination, map_location="cpu")
            records[split] = {"path": str(destination), "sha256": file_sha256(destination),
                              "feature_sha256": tensor_sha256(saved["features"]), "count": len(saved["sample_ids"]),
                              "shape": list(saved["features"].shape), "dtype": str(saved["features"].dtype),
                              "reused_existing": True}
            continue
        dataset = Phase4ADataset(phase4a_cfg, split)
        loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.workers,
                            collate_fn=phase4a_collate, pin_memory=True, persistent_workers=cli.workers > 0)
        sample_ids=[]; image_paths=[]; labels=[]; features=[]; started=time.time()
        with torch.no_grad():
            for batch_number, batch in enumerate(loader, 1):
                images = batch["images"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dense, _ = model.encode(images)
                    pooled = dense.mean(dim=(2, 3))
                if tuple(pooled.shape[1:]) != (128,) or not torch.isfinite(pooled).all():
                    raise RuntimeError("invalid FEPN pooled feature")
                sample_ids.extend(batch["sample_ids"]); image_paths.extend(batch["image_paths"])
                labels.extend(int(value) for value in batch["labels"].tolist())
                features.append(pooled.to(torch.float16).cpu())
                if batch_number % 50 == 0:
                    print(f"phase4b-cache split={split} rows={len(sample_ids)}/{len(dataset)}", flush=True)
        matrix = torch.cat(features)
        payload = {"sample_ids": sample_ids, "image_paths": image_paths, "class_labels": labels,
                   "features": matrix, "feature_location": "dense_features.mean((2,3))",
                   "fepn_checkpoint_sha256": cfg["fepn"]["checkpoint_sha256"],
                   "preprocess_sha256": hashlib.sha256(json.dumps(phase4a_cfg["preprocess"], sort_keys=True).encode()).hexdigest()}
        torch.save(payload, destination)
        records[split] = {"path": str(destination), "sha256": file_sha256(destination),
                          "feature_sha256": tensor_sha256(matrix), "count": len(sample_ids),
                          "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                          "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
                          "elapsed_seconds": time.time() - started, "reused_existing": False}
    out = ROOT / cfg["experiment"]["output_root"]
    manifest = {"status": "FROZEN", "fepn_checkpoint_sha256": cfg["fepn"]["checkpoint_sha256"],
                "feature_location": "classification-MLP input: dense_features.mean((2,3))",
                "feature_dim": 128, "feature_dtype": "torch.float16", "splits": records,
                "internal_test_cached": False, "official1000_cached": False,
                "total_elapsed_seconds": time.time() - started_all}
    dump(out / "fepn_global_feature_cache_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
