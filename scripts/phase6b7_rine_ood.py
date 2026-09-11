#!/usr/bin/env python3
"""Frozen RINE-on-C1 OOD inference and paired confirmation."""

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

import cv2
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPImageProcessor, CLIPVisionModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.rine_on_c1 import RINE_OFFICIAL_COMMIT, RINEOnHFCLIP  # noqa: E402
from scripts.final_eval_classification import MANIFESTS, scope  # noqa: E402


OUT = ROOT / "outputs/phase6b7_rine_ood"
PROTOCOL = OUT / "protocol.json"
CHECKPOINT = ROOT / "outputs/phase6b6_rine_training/selected_checkpoint.pt"
CHECKPOINT_SHA = "5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562"
CLIP_PATH = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
RINE_REPO = ROOT / "external/RINE_official"
BASELINE_ROOT = ROOT / "outputs/final_evaluation/classification/legion_retrained"
DATASETS = ("aigi_holmes", "genimage", "loki", "raise998")
EXPECTED = {"aigi_holmes": 99999, "genimage": 100000, "loki": 2217, "raise998": 998}
BATCH_SIZE = 128
CHUNK_SIZE = 2048
SEED = 3407


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "infer", "finalize"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def repo_gate() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(RINE_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(RINE_REPO), "status", "--porcelain"], text=True
    ).strip()
    if commit != RINE_OFFICIAL_COMMIT or dirty:
        raise RuntimeError(f"RINE source identity drift: {commit}, {dirty!r}")


def baseline_path(name: str) -> Path:
    return BASELINE_ROOT / name / "predictions.jsonl"


def prepare() -> None:
    if PROTOCOL.exists():
        raise RuntimeError("Frozen protocol already exists and will not be overwritten")
    repo_gate()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Selected Phase 6B.6 checkpoint drift")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if checkpoint.get("epoch") != 1:
        raise RuntimeError("Selected epoch is not 1")
    sources = {}
    for name in DATASETS:
        rows = scope(name)
        baseline = jsonl(baseline_path(name))
        if len(rows) != EXPECTED[name] or len(baseline) != EXPECTED[name]:
            raise RuntimeError(f"Population drift for {name}")
        if [row["sample_id"] for row in rows] != [row["sample_id"] for row in baseline]:
            raise RuntimeError(f"Baseline/manifest order mismatch for {name}")
        if [row["gt"] for row in rows] != [int(row["gt"]) for row in baseline]:
            raise RuntimeError(f"Baseline/manifest label mismatch for {name}")
        sources[name] = {
            "n": len(rows), "manifest": str(MANIFESTS[name].resolve()),
            "manifest_sha256": sha256(MANIFESTS[name]),
            "c1_exact_legion_retrained_predictions": str(baseline_path(name).resolve()),
            "baseline_predictions_sha256": sha256(baseline_path(name)),
        }
    protocol = {
        "schema": "phase6b7_rine_ood_protocol_v1",
        "status": "FROZEN_BEFORE_OOD_INFERENCE",
        "checkpoint": {"path": str(CHECKPOINT.resolve()), "sha256": CHECKPOINT_SHA,
                       "selected_epoch": 1},
        "model": {"name": "RINE-on-C1", "q": 2, "projection_dim": 1024,
                  "xi_training_only": 0.2, "blocks": 24,
                  "backbone": "frozen CLIP ViT-L/14@336",
                  "preprocessing": "same RGB CLIPImageProcessor as C1-Exact"},
        "baseline": {
            "name": "C1-Exact / LEGION-retrained",
            "reason": "Phase 6B.1b confirmed the prediction heads are bitwise identical; Phase 6B.3 froze this exact stream as F1",
        },
        "inference": {"batch_size": BATCH_SIZE, "chunk_size": CHUNK_SIZE,
                      "dtype": "BF16 CLIP + FP32 RINE head", "seed": SEED,
                      "threshold": 0.5, "positive_class": "Fake=1"},
        "datasets": sources,
        "reporting": {"metrics": ["Accuracy", "ROC-AUC", "Fake recall", "TNR",
                                            "FPR", "F1"],
                      "paired": ["RINE-only correct", "C1-only correct",
                                 "prediction disagreement", "exact McNemar p"],
                      "genimage_per_generator": True,
                      "raise998_primary": ["TNR", "FPR"]},
        "firewall": {"training": False, "threshold_tuning": False,
                     "calibration": False, "checkpoint_reselection": False,
                     "architecture_change": False, "fusion": False, "AIDE": False},
    }
    atomic_json(PROTOCOL, protocol)
    atomic_json(OUT / "status.json", {"status": "PREPARED", "protocol_sha256": sha256(PROTOCOL)})
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


class OODImages(Dataset):
    def __init__(self, rows: list[dict], processor: CLIPImageProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to decode frozen OOD sample: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixels, "index": index}


def load_model(device: torch.device) -> RINEOnHFCLIP:
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Selected checkpoint changed after protocol freeze")
    vision = CLIPVisionModel.from_pretrained(CLIP_PATH, local_files_only=True)
    vision = vision.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    model = RINEOnHFCLIP(vision).to(device)
    saved = torch.load(CHECKPOINT, map_location="cpu")
    if saved.get("epoch") != 1:
        raise RuntimeError("Checkpoint epoch drift")
    model.rine.load_state_dict(saved["rine_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


def infer(names: list[str], device_name: str, workers: int) -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Prepare protocol before inference")
    protocol_hash = sha256(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_OOD_INFERENCE":
        raise RuntimeError("Protocol is not frozen")
    repo_gate()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(device_name); torch.cuda.set_device(device)
    model = load_model(device)
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    started = time.monotonic()
    for name in names:
        source = scope(name)
        if len(source) != protocol["datasets"][name]["n"]:
            raise RuntimeError(f"Protocol population drift for {name}")
        dataset = OODImages(source, processor)
        shard_dir = OUT / "predictions" / name / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for lower in range(0, len(dataset), CHUNK_SIZE):
            upper = min(len(dataset), lower + CHUNK_SIZE)
            destination = shard_dir / f"{lower:06d}_{upper:06d}.pt"
            if destination.exists():
                saved = torch.load(destination, map_location="cpu")
                expected_ids = [row["sample_id"] for row in source[lower:upper]]
                if saved.get("sample_ids") != expected_ids or len(saved.get("prob_fake", [])) != upper - lower:
                    raise RuntimeError(f"Invalid resume shard: {destination}")
                continue
            loader = DataLoader(
                Subset(dataset, range(lower, upper)), batch_size=BATCH_SIZE,
                shuffle=False, num_workers=workers, pin_memory=True, drop_last=False,
                persistent_workers=workers > 0,
            )
            probabilities, indices = [], []
            with torch.inference_mode():
                for batch in loader:
                    pixels = batch["pixel_values"].to(
                        device=device, dtype=torch.bfloat16, non_blocking=True
                    )
                    logits, _, _ = model(pixels)
                    probabilities.append(torch.sigmoid(logits).squeeze(1).float().cpu())
                    indices.extend(batch["index"].tolist())
            if indices != list(range(lower, upper)):
                raise RuntimeError(f"Inference order drift for {name} {lower}:{upper}")
            payload = {
                "schema": "phase6b7_rine_predictions_shard_v1", "dataset": name,
                "lower": lower, "upper": upper,
                "sample_ids": [row["sample_id"] for row in source[lower:upper]],
                "labels": torch.tensor([row["gt"] for row in source[lower:upper]]),
                "sources": [row["source"] for row in source[lower:upper]],
                "prob_fake": torch.cat(probabilities),
                "checkpoint_sha256": CHECKPOINT_SHA, "threshold": 0.5,
            }
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary); os.replace(temporary, destination)
            done = upper
            elapsed = time.monotonic() - started
            event = {"event": "ood_progress", "dataset": name, "done": done,
                     "total": len(dataset), "elapsed_seconds": elapsed}
            print(json.dumps(event), flush=True)
            atomic_json(OUT / "status.json", {
                "status": "RUNNING", "pid": os.getpid(), "dataset": name,
                "done": done, "total": len(dataset), "elapsed_seconds": elapsed,
                "protocol_sha256": protocol_hash,
            })
        atomic_json(OUT / "predictions" / name / "complete.json", {
            "status": "COMPLETE", "n": len(dataset), "shard_count": math.ceil(len(dataset) / CHUNK_SIZE),
            "checkpoint_sha256": CHECKPOINT_SHA, "protocol_sha256": protocol_hash,
        })
    model.close_hooks()
    atomic_json(OUT / "worker_complete.json", {
        "status": "COMPLETE", "datasets": names, "runtime_seconds": time.monotonic() - started,
        "protocol_sha256": protocol_hash, "checkpoint_sha256": CHECKPOINT_SHA,
    })


def metric(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    tp = int(((labels == 1) & predictions).sum()); tn = int(((labels == 0) & ~predictions).sum())
    fp = int(((labels == 0) & predictions).sum()); fn = int(((labels == 1) & ~predictions).sum())
    result = {
        "n": int(len(labels)), "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()),
        "accuracy": float((labels == predictions).mean()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "fake_recall": float(recall_score(labels, predictions, zero_division=0)),
        "tnr": float(tn / max(1, tn + fp)), "fpr": float(fp / max(1, tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5,
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return result


def paired(labels: np.ndarray, c1_prob: np.ndarray, rine_prob: np.ndarray) -> dict:
    c1_pred = c1_prob >= 0.5; rine_pred = rine_prob >= 0.5
    c1_correct = c1_pred == labels; rine_correct = rine_pred == labels
    rine_only = int((rine_correct & ~c1_correct).sum())
    c1_only = int((c1_correct & ~rine_correct).sum())
    discordant = rine_only + c1_only
    return {
        "both_correct": int((c1_correct & rine_correct).sum()),
        "rine_only_correct": rine_only, "c1_only_correct": c1_only,
        "both_wrong": int((~c1_correct & ~rine_correct).sum()),
        "prediction_disagreement": int((c1_pred != rine_pred).sum()),
        "mcnemar_exact_p": float(binomtest(min(rine_only, c1_only), discordant, 0.5).pvalue)
        if discordant else 1.0,
    }


def consolidate(name: str, source: list[dict]) -> dict:
    parts = []
    shard_dir = OUT / "predictions" / name / "shards"
    for lower in range(0, len(source), CHUNK_SIZE):
        upper = min(len(source), lower + CHUNK_SIZE)
        path = shard_dir / f"{lower:06d}_{upper:06d}.pt"
        if not path.exists():
            raise RuntimeError(f"Missing prediction shard: {path}")
        parts.append(torch.load(path, map_location="cpu"))
    ids = sum((part["sample_ids"] for part in parts), [])
    expected_ids = [row["sample_id"] for row in source]
    if ids != expected_ids:
        raise RuntimeError(f"Consolidated sample order drift for {name}")
    labels = torch.cat([part["labels"] for part in parts])
    probabilities = torch.cat([part["prob_fake"] for part in parts])
    sources = sum((part["sources"] for part in parts), [])
    payload = {"schema": "phase6b7_rine_predictions_v1", "dataset": name,
               "sample_ids": ids, "labels": labels, "sources": sources,
               "prob_fake": probabilities, "checkpoint_sha256": CHECKPOINT_SHA}
    path = OUT / "predictions" / name / "predictions.pt"
    torch.save(payload, path)
    return payload


def finalize() -> None:
    protocol_hash = sha256(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Checkpoint drift at finalization")
    results = {}
    for name in DATASETS:
        source = scope(name)
        rine = consolidate(name, source)
        baseline = jsonl(baseline_path(name))
        if rine["sample_ids"] != [row["sample_id"] for row in baseline]:
            raise RuntimeError(f"C1/RINE pairing mismatch for {name}")
        labels = rine["labels"].numpy().astype(np.int64)
        if not np.array_equal(labels, np.asarray([row["gt"] for row in baseline], dtype=np.int64)):
            raise RuntimeError(f"C1/RINE label mismatch for {name}")
        c1_prob = np.asarray([row["prob_fake"] for row in baseline], dtype=np.float64)
        rine_prob = rine["prob_fake"].numpy().astype(np.float64)
        dataset_result = {
            "metrics": {"C1-Exact / LEGION-retrained": metric(labels, c1_prob),
                        "RINE-on-C1": metric(labels, rine_prob)},
            "paired": paired(labels, c1_prob, rine_prob),
            "prediction_file": str((OUT / "predictions" / name / "predictions.pt").resolve()),
            "prediction_file_sha256": sha256(OUT / "predictions" / name / "predictions.pt"),
            "manifest_sha256": protocol["datasets"][name]["manifest_sha256"],
        }
        if name == "genimage":
            per_generator = {}
            source_values = np.asarray(rine["sources"], dtype=object)
            for generator in sorted(set(source_values.tolist())):
                take = source_values == generator
                per_generator[generator] = {
                    "metrics": {
                        "C1-Exact / LEGION-retrained": metric(labels[take], c1_prob[take]),
                        "RINE-on-C1": metric(labels[take], rine_prob[take]),
                    },
                    "paired": paired(labels[take], c1_prob[take], rine_prob[take]),
                }
            dataset_result["per_generator"] = per_generator
        results[name] = dataset_result

    mixed = ("aigi_holmes", "genimage", "loki")
    aggregate = {}
    for model_name in ("C1-Exact / LEGION-retrained", "RINE-on-C1"):
        aggregate[model_name] = {
            key: float(np.mean([results[name]["metrics"][model_name][key] for name in mixed]))
            for key in ("accuracy", "roc_auc", "fake_recall", "tnr", "fpr", "f1")
        }
    aggregate["delta_rine_minus_c1"] = {
        key: aggregate["RINE-on-C1"][key] - aggregate["C1-Exact / LEGION-retrained"][key]
        for key in aggregate["RINE-on-C1"]
    }
    auc_wins = sum(
        results[name]["metrics"]["RINE-on-C1"]["roc_auc"]
        > results[name]["metrics"]["C1-Exact / LEGION-retrained"]["roc_auc"]
        for name in mixed
    )
    accuracy_wins = sum(
        results[name]["metrics"]["RINE-on-C1"]["accuracy"]
        > results[name]["metrics"]["C1-Exact / LEGION-retrained"]["accuracy"]
        for name in mixed
    )
    gen = results["genimage"]["per_generator"]
    generator_summary = {
        "count": len(gen),
        "accuracy_wins": sum(v["metrics"]["RINE-on-C1"]["accuracy"]
                             > v["metrics"]["C1-Exact / LEGION-retrained"]["accuracy"] for v in gen.values()),
        "auc_wins": sum(v["metrics"]["RINE-on-C1"]["roc_auc"]
                        > v["metrics"]["C1-Exact / LEGION-retrained"]["roc_auc"] for v in gen.values()),
        "recall_wins": sum(v["metrics"]["RINE-on-C1"]["fake_recall"]
                           > v["metrics"]["C1-Exact / LEGION-retrained"]["fake_recall"] for v in gen.values()),
        "tnr_non_decrease": sum(v["metrics"]["RINE-on-C1"]["tnr"]
                                >= v["metrics"]["C1-Exact / LEGION-retrained"]["tnr"] for v in gen.values()),
    }
    final = {
        "schema": "phase6b7_rine_ood_results_v1", "status": "COMPLETE",
        "protocol_sha256": protocol_hash, "checkpoint_sha256": CHECKPOINT_SHA,
        "datasets": results, "mixed_ood_macro": aggregate,
        "summary": {"mixed_accuracy_wins": accuracy_wins, "mixed_auc_wins": auc_wins,
                    "mixed_dataset_count": len(mixed),
                    "genimage_generator_summary": generator_summary,
                    "raise998_rine_fpr": results["raise998"]["metrics"]["RINE-on-C1"]["fpr"],
                    "raise998_c1_fpr": results["raise998"]["metrics"]["C1-Exact / LEGION-retrained"]["fpr"]},
        "firewall": {"training": False, "threshold_tuning": False, "calibration": False,
                     "checkpoint_reselection": False, "architecture_change": False,
                     "fusion": False, "AIDE": False},
    }
    atomic_json(OUT / "results.json", final)
    atomic_json(OUT / "status.json", {"status": "COMPLETE", "protocol_sha256": protocol_hash,
                                      "checkpoint_sha256": CHECKPOINT_SHA})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parsed = args()
    if parsed.command == "prepare":
        prepare()
    elif parsed.command == "infer":
        infer(parsed.datasets or list(DATASETS), parsed.device, parsed.workers)
    else:
        finalize()


if __name__ == "__main__":
    main()
