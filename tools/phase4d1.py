"""Shared utilities for Phase 4D-1 minimal position-aware evidence test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from tools.phase4c_b import paired, tensor_hash


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values),
                    encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reader_state_hash(reader) -> str:
    return tensor_hash(reader.state_dict().items())


def tensor_summary(value: torch.Tensor):
    x = value.detach().float()
    return {
        "shape": list(x.shape),
        "count": x.numel(),
        "rms": float(x.square().mean().sqrt()),
        "std": float(x.std(unbiased=False)),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


class RunningTensorSummary:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: torch.Tensor):
        x = value.detach().double()
        self.count += x.numel()
        self.total += float(x.sum())
        self.square += float(x.square().sum())
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))

    def result(self, shape):
        mean = self.total / self.count
        variance = max(0.0, self.square / self.count - mean * mean)
        return {
            "shape_per_sample": list(shape),
            "count": self.count,
            "rms": math.sqrt(self.square / self.count),
            "std": math.sqrt(variance),
            "mean": mean,
            "min": self.minimum,
            "max": self.maximum,
        }


def fixed_derangement(ids, seed=3407):
    values = list(ids)
    rng = random.Random(seed)
    for _ in range(10000):
        candidate = values.copy()
        rng.shuffle(candidate)
        if all(a != b for a, b in zip(values, candidate)):
            return dict(zip(values, candidate))
    raise RuntimeError("failed to construct fixed derangement")


def paired_with_wilcoxon(left, right, repeats=10000, seed=3407):
    result = paired(left, right, repeats=repeats, seed=seed)
    for metric, key in (("foreground_iou", "foreground_iou"),
                        ("foreground_f1", "foreground_f1")):
        delta = np.asarray([a[key] - b[key] for a, b in zip(left, right)],
                           dtype=np.float64)
        if np.all(np.abs(delta) <= 1e-12):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided",
                            method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        result[metric]["wilcoxon_statistic"] = statistic
        result[metric]["wilcoxon_p_value"] = p_value
    return result
