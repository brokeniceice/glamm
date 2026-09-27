#!/usr/bin/env python3
"""Retrain the pinned official RINE implementation on project data at 224px.

Set PYTHONPATH to include the pinned OpenAI CLIP checkout, isolated ftfy deps,
and external/RINE_official. The official model, transforms, and SupCon are imported
directly; this file only adapts manifests and records results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.models import Model
from src.utils import SupConLoss, get_transforms, seed_everything


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rine_official224_retrain"
ASSETS = Path("/data/yz/myLISA_storage/experiments/rine_official_224")
WEIGHTS = ASSETS / "ViT-L-14.pt"
CLIP_REPO = ASSETS / "openai_clip_source"
RINE_REPO = ROOT / "external/RINE_official"
EXPECTED_WEIGHTS_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
EXPECTED_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
EXPECTED_RINE_COMMIT = "9b7fd5857cc205d0412be6aeee0d7611b95bd620"
TRAIN = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/train.json"
VAL = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/val.json"
EVAL = {
    "internal_test": ROOT / "datasets/Internal2208/manifests/eval_manifest.jsonl",
    "aigi_holmes": ROOT / "datasets/AIGI-Holmes/manifests/eval_manifest.jsonl",
    "genimage": ROOT / "datasets/GenImage/manifests/eval_manifest.jsonl",
    "loki": ROOT / "datasets/LOKI/manifests/classification_eval_manifest.jsonl",
    "raise998": ROOT / "datasets/RAISE/manifests/eval_manifest.jsonl",
}
EXPECTED_N = {"internal_test": 2208, "aigi_holmes": 99999, "genimage": 100000,
              "loki": 2217, "raise998": 998}
HEAD_KEYS = ("alpha", "proj1.", "proj2.", "head.")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def training_rows(path: Path) -> list[dict]:
    rows = records(path)
    return [{"sample_id": r["sample_id"], "image_path": r["image_path"],
             "gt": 1 - int(r["label"]), "source": r.get("source", "internal")}
            for r in rows]


def evaluation_rows(name: str) -> list[dict]:
    rows = records(EVAL[name])
    out = [{"sample_id": r["sample_id"], "image_path": r["image_path"],
            "gt": int(r["label"] == "Fake"),
            "source": r.get("generator/source", "unknown")}
           for r in rows]
    if len(out) != EXPECTED_N[name] or len({r["sample_id"] for r in out}) != len(out):
        raise RuntimeError(f"Evaluation population drift: {name}")
    return out


class Images(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, row["gt"], index


def metrics(rows: list[dict]) -> dict:
    y = np.asarray([r["gt"] for r in rows], dtype=np.int8)
    p = np.asarray([r["prob_fake"] for r in rows], dtype=np.float64)
    q = (p >= 0.5).astype(np.int8)
    tp = int(((y == 1) & (q == 1)).sum())
    tn = int(((y == 0) & (q == 0)).sum())
    fp = int(((y == 0) & (q == 1)).sum())
    fn = int(((y == 1) & (q == 0)).sum())
    result = {"n": len(rows), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
              "accuracy": float((y == q).mean()), "fake_recall": tp / max(1, tp + fn),
              "tnr": tn / max(1, tn + fp), "fpr": fp / max(1, tn + fp),
              "precision": tp / max(1, tp + fp),
              "f1": 2 * tp / max(1, 2 * tp + fp + fn),
              "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5}
    if len(set(y.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["auprc"] = float(average_precision_score(y, p))
    return result


def check_protocol() -> dict:
    path = OUT / "protocol.json"
    protocol = json.loads(path.read_text())
    if sha(WEIGHTS) != protocol["provenance"]["clip_weights_sha256"]:
        raise RuntimeError("CLIP weights changed")
    if commit(CLIP_REPO) != EXPECTED_CLIP_COMMIT or commit(RINE_REPO) != EXPECTED_RINE_COMMIT:
        raise RuntimeError("Pinned upstream checkout changed")
    for key, path in {"train": TRAIN, "validation": VAL, **EVAL}.items():
        if sha(path) != protocol["manifests"][key]["sha256"]:
            raise RuntimeError(f"Manifest changed: {key}")
    return protocol


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Protocol already exists")
    if sha(WEIGHTS) != EXPECTED_WEIGHTS_SHA:
        raise RuntimeError("Official 224 CLIP weight hash mismatch")
    if commit(CLIP_REPO) != EXPECTED_CLIP_COMMIT or commit(RINE_REPO) != EXPECTED_RINE_COMMIT:
        raise RuntimeError("Official source checkout mismatch")
    train = training_rows(TRAIN)
    val = training_rows(VAL)
    if len(train) != 17672 or len(val) != 2212:
        raise RuntimeError("Internal manifest count drift")
    small = []
    for row in train:
        with Image.open(row["image_path"]) as image:
            width, height = image.size
        if width < 224 or height < 224:
            small.append(row["sample_id"])
    if len(small) != 40:
        raise RuntimeError(f"Small-image census drift: {len(small)}")
    eligible = [r for r in train if r["sample_id"] not in set(small)]
    if len({r["sample_id"] for r in eligible}) != len(eligible):
        raise RuntimeError("Duplicate training IDs")
    manifests = {key: {"path": str(path.resolve()), "sha256": sha(path)}
                 for key, path in {"train": TRAIN, "validation": VAL, **EVAL}.items()}
    for key in EVAL:
        manifests[key]["n"] = len(evaluation_rows(key))
    protocol = {
        "schema": "rine_official224_project_retraining_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "provenance": {"rine_commit": EXPECTED_RINE_COMMIT,
                       "openai_clip_commit": EXPECTED_CLIP_COMMIT,
                       "clip_weights": str(WEIGHTS), "clip_weights_sha256": EXPECTED_WEIGHTS_SHA,
                       "official_model": str(RINE_REPO / "src/models.py"),
                       "official_transforms_and_supcon": str(RINE_REPO / "src/utils.py")},
        "model": {"backbone": "OpenAI CLIP ViT-L/14 (224px)", "block_hooks": "all 24 ln_2 outputs, CLS index 0",
                  "nproj": 2, "proj_dim": 1024, "factor": 0.2,
                  "trainable": ["alpha", "proj1", "proj2", "head"], "frozen": "all CLIP parameters"},
        "training": {"seed": 0, "batch_size": 128, "epochs": 1, "optimizer": "Adam",
                     "lr": 0.001, "weight_decay": 0, "loss": "BCEWithLogitsLoss(sum)+0.2*official SupConLoss",
                     "transform": "official get_transforms()[0]: blur/JPEG, RandomCrop224, HFlip, CLIP norm",
                     "source_n": len(train), "eligible_n": len(eligible), "excluded_small_ids": small,
                     "exclusion_reason": "official RandomCrop(224) raises when width or height < 224"},
        "evaluation": {"transform": "official get_transforms()[1]: CenterCrop224, CLIP norm, no resize",
                       "threshold": 0.5, "positive_class": "Fake", "checkpoint": "epoch 1 fixed",
                       "no_ood_selection": True},
        "manifests": manifests,
    }
    atomic(OUT / "protocol.json", protocol)
    atomic(OUT / "status.json", {"status": "PREPARED", "protocol_sha256": sha(OUT / "protocol.json")})
    print(json.dumps({"status": "PREPARED", "eligible_n": len(eligible), "excluded_n": len(small)}), flush=True)


def make_model(device: str) -> Model:
    model = Model(backbone=(str(WEIGHTS), 1024), nproj=2, proj_dim=1024, device=device)
    model.to(device)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 6323201:
        raise RuntimeError("Official head parameter count mismatch")
    return model


def train(device: str, workers: int) -> None:
    protocol = check_protocol()
    checkpoint = OUT / "checkpoint_epoch1.pt"
    if checkpoint.exists():
        raise RuntimeError("Checkpoint already exists; refusing to overwrite")
    seed_everything(0)
    train_transform, eval_transform, _ = get_transforms()
    excluded = set(protocol["training"]["excluded_small_ids"])
    rows = [r for r in training_rows(TRAIN) if r["sample_id"] not in excluded]
    if len(rows) != protocol["training"]["eligible_n"]:
        raise RuntimeError("Training scope changed")
    loader = DataLoader(Images(rows, train_transform), batch_size=128, shuffle=True,
                        num_workers=workers, pin_memory=True, drop_last=False)
    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bce = BCEWithLogitsLoss(reduction="sum")
    supcon = SupConLoss()
    atomic(OUT / "status.json", {"status": "TRAINING", "protocol_sha256": sha(OUT / "protocol.json"),
                                 "device": device, "pid": os.getpid(), "step": 0})
    model.train()
    started = time.monotonic()
    seen = 0
    for step, (pixels, targets, _) in enumerate(loader, 1):
        pixels = pixels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, embedding = model(pixels)
        loss = bce(logits, targets.float().view(-1, 1)) + 0.2 * supcon(
            F.normalize(embedding).unsqueeze(1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seen += len(targets)
        if step == 1 or step % 10 == 0 or step == len(loader):
            event = {"status": "TRAINING", "step": step, "steps": len(loader),
                     "seen": seen, "loss": float(loss.detach()), "seconds": round(time.monotonic()-started, 1)}
            atomic(OUT / "status.json", event)
            print(json.dumps(event), flush=True)
    if seen != len(rows):
        raise RuntimeError("Training image count mismatch")
    head = {key: value.detach().cpu() for key, value in model.state_dict().items()
            if key == "alpha" or key.startswith(("proj1.", "proj2.", "head."))}
    torch.save({"epoch": 1, "official_rine_commit": EXPECTED_RINE_COMMIT,
                "openai_clip_commit": EXPECTED_CLIP_COMMIT,
                "clip_weights_sha256": EXPECTED_WEIGHTS_SHA,
                "protocol_sha256": sha(OUT / "protocol.json"),
                "trainable_state_dict": head}, checkpoint)
    atomic(OUT / "checkpoint_identity.json", {"path": str(checkpoint), "sha256": sha(checkpoint),
                                               "epoch": 1, "train_n": seen})
    atomic(OUT / "status.json", {"status": "TRAINED", "checkpoint_sha256": sha(checkpoint),
                                 "train_n": seen, "seconds": round(time.monotonic()-started, 1)})
    print(json.dumps({"status": "TRAINED", "sha256": sha(checkpoint)}), flush=True)
    del model, loader
    torch.cuda.empty_cache()
    infer_one("validation", training_rows(VAL), eval_transform, device, workers)


def load_trained(device: str) -> Model:
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    path = Path(identity["path"])
    if sha(path) != identity["sha256"]:
        raise RuntimeError("Trained checkpoint hash mismatch")
    model = make_model(device)
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol_sha256"] != sha(OUT / "protocol.json"):
        raise RuntimeError("Checkpoint protocol mismatch")
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if unexpected or any(not key.startswith("clip.") for key in missing):
        raise RuntimeError(f"Checkpoint key mismatch: {missing[:4]} {unexpected[:4]}")
    return model.eval()


def infer_one(name: str, rows: list[dict], transform, device: str, workers: int) -> None:
    result_dir = OUT / "evaluation" / name
    predictions = result_dir / "predictions.jsonl"
    result = result_dir / "results.json"
    if result.exists():
        saved = json.loads(result.read_text())
        if saved.get("status") == "COMPLETE":
            print(json.dumps({"dataset": name, "status": "ALREADY_COMPLETE"}), flush=True)
            return
    old = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()] if predictions.exists() else []
    if [r["sample_id"] for r in old] != [r["sample_id"] for r in rows[:len(old)]]:
        raise RuntimeError(f"Prediction prefix mismatch: {name}")
    model = load_trained(device)
    dataset = Images(rows, transform)
    batch_size = 16 if name == "raise998" else 128
    loader = DataLoader(Subset(dataset, range(len(old), len(dataset))), batch_size=batch_size,
                        shuffle=False, num_workers=min(workers, 4) if name == "raise998" else workers,
                        pin_memory=True, drop_last=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with predictions.open("a") as stream, torch.inference_mode():
        for pixels, targets, indices in loader:
            logits, _ = model(pixels.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).view(-1).float().cpu().tolist()
            for index, target, prob in zip(indices.tolist(), targets.tolist(), probs):
                if int(target) != rows[index]["gt"]:
                    raise RuntimeError(f"Label mismatch: {name}/{index}")
                stream.write(json.dumps({"sample_id": rows[index]["sample_id"],
                                         "gt": int(target), "source": rows[index]["source"],
                                         "prob_fake": prob}) + "\n")
            stream.flush()
            done = int(indices[-1]) + 1
            if done % 2048 < batch_size or done == len(rows):
                print(json.dumps({"dataset": name, "done": done, "total": len(rows),
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    all_rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if len(all_rows) != len(rows) or [r["sample_id"] for r in all_rows] != [r["sample_id"] for r in rows]:
        raise RuntimeError(f"Incomplete or reordered predictions: {name}")
    summary = {"status": "COMPLETE", "dataset": name, "n": len(rows),
               "protocol_sha256": sha(OUT / "protocol.json"),
               "checkpoint_sha256": json.loads((OUT / "checkpoint_identity.json").read_text())["sha256"],
               "metrics": metrics(all_rows)}
    if name == "genimage":
        summary["per_generator"] = {source: metrics([r for r in all_rows if r["source"] == source])
                                    for source in sorted({r["source"] for r in all_rows})}
    atomic(result, summary)
    print(json.dumps({"dataset": name, "status": "COMPLETE", "metrics": summary["metrics"]}), flush=True)


def evaluate(names: list[str], device: str, workers: int) -> None:
    check_protocol()
    _, transform, _ = get_transforms()
    for name in names:
        rows = training_rows(VAL) if name == "validation" else evaluation_rows(name)
        infer_one(name, rows, transform, device, workers)


def finalize() -> None:
    check_protocol()
    names = ("validation", *EVAL)
    results = {}
    for name in names:
        path = OUT / "evaluation" / name / "results.json"
        if not path.exists():
            raise RuntimeError(f"Missing result: {name}")
        result = json.loads(path.read_text())
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"Incomplete result: {name}")
        results[name] = result["metrics"]
    atomic(OUT / "results.json", {"status": "COMPLETE", "model": "official RINE 224 retrained on project data",
                                  "results": results, "checkpoint": json.loads((OUT / "checkpoint_identity.json").read_text())})
    atomic(OUT / "status.json", {"status": "COMPLETE", "datasets": list(names)})
    print(json.dumps({"status": "COMPLETE", "datasets": list(names)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "eval", "finalize"))
    parser.add_argument("--datasets", nargs="+", choices=("validation", *EVAL))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "train": train(args.device, args.workers)
    elif args.command == "eval": evaluate(args.datasets or list(EVAL), args.device, args.workers)
    else: finalize()


if __name__ == "__main__":
    main()
