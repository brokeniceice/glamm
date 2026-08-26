"""Shared data, metric, hashing and batching utilities for Phase 4A."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import CLIPImageProcessor

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3c1 import geometry_for, transform_mask


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


class Phase4ADataset(Dataset):
    def __init__(self, config: dict, split: str, *, return_original: bool = False) -> None:
        root = Path(__file__).resolve().parents[1]
        self.root = root
        self.config = config
        self.split = split
        self.return_original = return_original
        manifest = root / config["data"]["manifest_dir"] / f"{split}_combined.jsonl"
        self.rows = load_rows(manifest)
        self.processor = CLIPImageProcessor.from_pretrained(config["preprocess"]["source"])
        self.datasets_root = (root / config["data"]["datasets_root"]).resolve()
        self.synthscars_root = (root / config["data"]["synthscars_root"]).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_image(self, row: dict) -> Path:
        candidate = Path(str(row.get("image_path") or "")).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        base = self.synthscars_root if row["forensics_domain"] == "fake" else self.datasets_root
        path = base / str(row["image_relpath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self.resolve_image(row)
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            pixels = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        label = int(row["class_label"])
        geometry = geometry_for("clip", [height, width])
        original = target = None
        if label == 1:
            original = UnifiedForensicsDataset._fake_union_mask(row, height, width).bool().any(dim=0)
            target = transform_mask(original, geometry).float().unsqueeze(0)
        return {
            "sample_id": str(row["sample_id"]),
            "image_path": str(path),
            "image": pixels,
            "label": label,
            "dense_target": target,
            "original_mask": original if self.return_original else None,
            "geometry": geometry,
            "source": row.get("source"),
        }


def phase4a_collate(rows: list[dict]) -> dict:
    fake_positions = [index for index, row in enumerate(rows) if row["label"] == 1]
    dense_targets = (
        torch.stack([rows[index]["dense_target"] for index in fake_positions])
        if fake_positions else None
    )
    return {
        "sample_ids": [row["sample_id"] for row in rows],
        "image_paths": [row["image_path"] for row in rows],
        "images": torch.stack([row["image"] for row in rows]),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
        "fake_positions": torch.tensor(fake_positions, dtype=torch.long),
        "dense_targets": dense_targets,
        "original_masks": [row["original_mask"] for row in rows],
        "geometries": [row["geometry"] for row in rows],
        "sources": [row["source"] for row in rows],
    }


class BalancedBatchSampler(Sampler[list[int]]):
    """Deterministic half-Real/half-Fake batches with one pass per epoch."""

    def __init__(self, rows: list[dict], batch_size: int, seed: int, epoch: int) -> None:
        if batch_size % 2:
            raise ValueError("balanced batch size must be even")
        self.real = [i for i, row in enumerate(rows) if int(row["class_label"]) == 0]
        self.fake = [i for i, row in enumerate(rows) if int(row["class_label"]) == 1]
        if len(self.real) != len(self.fake):
            raise ValueError("Phase 4A requires matched Real/Fake population")
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.real) / self.half)

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + 1009 * self.epoch)
        real, fake = self.real.copy(), self.fake.copy()
        rng.shuffle(real); rng.shuffle(fake)
        for start in range(0, len(real), self.half):
            batch = real[start:start + self.half] + fake[start:start + self.half]
            rng.shuffle(batch)
            yield batch


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    prediction = logits > 0
    truth = labels.astype(bool)
    tp = int((prediction & truth).sum()); fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum()); tn = int((~prediction & ~truth).sum())
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(labels)), "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc_score(labels, logits)),
        "logit_mean": float(np.mean(logits)), "logit_std": float(np.std(logits)),
        "real_accuracy": float(tn / max(1, tn + fp)), "fake_accuracy": float(tp / max(1, tp + fn)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
