"""Shared Phase 4C-A artifact, metric, and cached-feature helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(named) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_paths(phase3c1_root: Path, split: str) -> list[Path]:
    root = phase3c1_root / "cache" / "clip" / split
    complete = json.loads((root / "complete.json").read_text())
    if complete["status"] != "COMPLETE" or not complete["source_parameter_hash_exact"]:
        raise RuntimeError(f"invalid frozen CLIP cache: {root}")
    paths = sorted(root.glob("shard_*.pt"))
    if len(paths) != int(complete["shards"]):
        raise RuntimeError("CLIP cache shard-count mismatch")
    return paths


def load_shard(path: Path):
    value = torch.load(path, map_location="cpu")
    if value.get("schema") != "phase3c1_spatial_cache_v1" or value.get("source") != "clip":
        raise RuntimeError(f"invalid CLIP cache payload: {path}")
    return value


def summarize_extended(records):
    if not records:
        raise ValueError("empty metric records")
    iou = np.asarray([r["foreground_iou"] for r in records], np.float64)
    f1 = np.asarray([r["foreground_f1"] for r in records], np.float64)
    tp = sum(int(r["tp"]) for r in records); fp = sum(int(r["fp"]) for r in records)
    fn = sum(int(r["fn"]) for r in records)
    return {
        "n": len(records),
        "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)),
        "mean_foreground_f1": float(f1.mean()),
        "global_foreground_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "global_foreground_f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "threshold_logit": 0.0,
        "aggregate_counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def paired_bootstrap(left, right, repeats=10000, seed=3407):
    if len(left) != len(right) or not left:
        raise ValueError("paired inputs differ or are empty")
    a = np.asarray(left, np.float64); b = np.asarray(right, np.float64); delta = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, np.float64)
    # Chunked to avoid a 10k x 1106 temporary allocation.
    for start in range(0, repeats, 500):
        n = min(500, repeats - start)
        index = rng.integers(0, len(delta), size=(n, len(delta)))
        boot[start:start+n] = delta[index].mean(1)
    eps = 1e-12
    return {
        "n": len(delta), "bootstrap_repeats": repeats,
        "mean_difference": float(delta.mean()), "median_difference": float(np.median(delta)),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wins": int((delta > eps).sum()), "ties": int((np.abs(delta) <= eps).sum()),
        "losses": int((delta < -eps).sum()),
    }

