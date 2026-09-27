#!/usr/bin/env python3
"""Manifest adapter for the pinned official NPR model and 50-epoch recipe.

Launch training with torchrun --standalone --nproc_per_node=2.  Dataset handling,
DDP, and result recording live here; the network is imported from upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/NPR_official"
sys.path.insert(0, str(UPSTREAM))
from networks.resnet import resnet50  # noqa: E402 -- the pinned upstream model

UPSTREAM_COMMIT = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"
OUT = ROOT / "outputs/npr_official_retrain"
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
ImageFile.LOAD_TRUNCATED_IMAGES = True  # upstream data/datasets.py


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def source_commit() -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()


def records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def rows(name: str) -> list[dict]:
    if name in ("train", "validation"):
        data = records(TRAIN if name == "train" else VAL)
        result = [{"sample_id": r["sample_id"], "image_path": r["image_path"],
                   "gt": 1 - int(r["label"]), "source": r.get("source", "internal")}
                  for r in data]
        required = 17672 if name == "train" else 2212
    else:
        data = records(EVAL[name])
        result = [{"sample_id": r["sample_id"], "image_path": r["image_path"],
                   "gt": int(r["label"] == "Fake"),
                   "source": r.get("generator/source", "unknown")}
                  for r in data]
        required = EXPECTED_N[name]
    if len(result) != required or len({r["sample_id"] for r in result}) != required:
        raise RuntimeError(f"Population drift or duplicate ID: {name}")
    return result


def transforms_for(training: bool):
    # Exactly the active operations in upstream data/datasets.py:binary_dataset.
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224) if training else transforms.Lambda(lambda image: image),
        transforms.RandomHorizontalFlip() if training else transforms.Lambda(lambda image: image),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class Images(Dataset):
    def __init__(self, data: list[dict], transform):
        self.data, self.transform = data, transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        with Image.open(item["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, item["gt"], index


def metrics(data: list[dict]) -> dict:
    y = np.asarray([r["gt"] for r in data], dtype=np.int8)
    p = np.asarray([r["prob_fake"] for r in data], dtype=np.float64)
    q = (p > 0.5).astype(np.int8)  # official validate.py uses strict >
    tp = int(((y == 1) & (q == 1)).sum())
    tn = int(((y == 0) & (q == 0)).sum())
    fp = int(((y == 0) & (q == 1)).sum())
    fn = int(((y == 1) & (q == 0)).sum())
    out = {"n": len(data), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
           "accuracy": float((y == q).mean()), "fake_recall": tp / max(1, tp + fn),
           "tnr": tn / max(1, tn + fp), "fpr": fp / max(1, tn + fp),
           "precision": tp / max(1, tp + fp),
           "f1": 2 * tp / max(1, 2 * tp + fp + fn),
           "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5}
    if len(set(y.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    return out


def validate_protocol() -> dict:
    protocol = json.loads((OUT / "protocol.json").read_text())
    if source_commit() != UPSTREAM_COMMIT:
        raise RuntimeError("Pinned upstream commit changed")
    files = {"train": TRAIN, "validation": VAL, **EVAL,
             "upstream_model": UPSTREAM / "networks/resnet.py",
             "upstream_transform": UPSTREAM / "data/datasets.py",
             "upstream_trainer": UPSTREAM / "networks/trainer.py"}
    for name, path in files.items():
        if sha(path) != protocol["files"][name]["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")
    return protocol


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Protocol already exists")
    if source_commit() != UPSTREAM_COMMIT:
        raise RuntimeError("Unexpected official NPR source commit")
    for name in ("train", "validation", *EVAL):
        rows(name)
    files = {"train": TRAIN, "validation": VAL, **EVAL,
             "upstream_model": UPSTREAM / "networks/resnet.py",
             "upstream_transform": UPSTREAM / "data/datasets.py",
             "upstream_trainer": UPSTREAM / "networks/trainer.py"}
    protocol = {
        "schema": "npr_official_project_retraining_v1", "status": "FROZEN_BEFORE_TRAINING",
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "upstream networks.resnet.resnet50(pretrained=False,num_classes=1)",
        "training": {"seed": 100, "epochs": 50, "global_batch_size": 32,
                     "optimizer": "Adam", "learning_rate": 0.0002, "beta1": 0.9,
                     "lr_decay": "multiply by 0.9 at end of 0-based epochs 10,20,30,40",
                     "loss": "BCEWithLogitsLoss", "checkpoint": "epoch 50 fixed",
                     "transform": "Resize((256,256)), RandomCrop(224), RandomHorizontalFlip, ToTensor, ImageNet norm",
                     "augmentation": "official default: no blur or JPEG"},
        "evaluation": {"transform": "Resize((256,256)), no crop, no flip, ToTensor, ImageNet norm",
                       "positive_class": "Fake", "threshold": 0.5,
                       "no_test_or_ood_selection": True},
        "distributed": {"gpus": 2, "sync_batch_norm": True,
                        "per_gpu_batch_size": 16, "shuffled_each_epoch": True},
        "files": {name: {"path": str(path.resolve()), "sha256": sha(path)}
                  for name, path in files.items()},
        "counts": {name: len(rows(name)) for name in ("train", "validation", *EVAL)},
    }
    atomic(OUT / "protocol.json", protocol)
    atomic(OUT / "status.json", {"status": "PREPARED", "protocol_sha256": sha(OUT / "protocol.json")})
    print(json.dumps({"status": "PREPARED", "train_n": protocol["counts"]["train"]}), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(workers: int) -> None:
    protocol = validate_protocol()
    if (OUT / "checkpoint_epoch50.pt").exists():
        raise RuntimeError("Final checkpoint already exists")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.get_world_size() != 2:
        raise RuntimeError("Frozen protocol requires 2 GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(100)
    data = rows("train")
    sampler = DistributedSampler(data, num_replicas=2, rank=rank, shuffle=True, seed=100,
                                 drop_last=False)
    loader = DataLoader(Images(data, transforms_for(True)), batch_size=16, sampler=sampler,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    net = resnet50(pretrained=False, num_classes=1)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    model = DistributedDataParallel(net, device_ids=[local_rank])
    optim = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    latest = OUT / "checkpoint_latest.pt"
    start_epoch = 0
    if latest.exists():
        resume = torch.load(latest, map_location="cpu")
        if resume["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise RuntimeError("Resume protocol mismatch")
        model.module.load_state_dict(resume["model"])
        optim.load_state_dict(resume["optimizer"])
        start_epoch = resume["epoch"]
        if rank == 0:
            print(json.dumps({"status": "RESUMED", "epoch": start_epoch}), flush=True)
    dist.barrier()
    started = time.monotonic()
    for epoch in range(start_epoch, 50):
        sampler.set_epoch(epoch)
        model.train()
        train_sum = torch.zeros(2, device=device)
        for step, (pixels, labels, _) in enumerate(loader, 1):
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            logits = model(pixels).squeeze(1)
            loss = loss_fn(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_sum += torch.tensor([float(loss.detach()) * len(labels), len(labels)], device=device)
            if rank == 0 and (step == 1 or step % 100 == 0):
                atomic(OUT / "status.json", {"status": "TRAINING", "epoch": epoch + 1,
                                             "step": step, "steps": len(loader), "pid": os.getpid(),
                                             "seconds": round(time.monotonic() - started, 1)})
        dist.all_reduce(train_sum)
        if epoch % 10 == 0 and epoch != 0:  # upstream train.py, end of epoch
            for group in optim.param_groups:
                group["lr"] = max(1e-6, group["lr"] * 0.9)
        dist.barrier()
        if rank == 0:
            state = {"epoch": epoch + 1, "model": model.module.state_dict(),
                     "optimizer": optim.state_dict(), "protocol_sha256": sha(OUT / "protocol.json")}
            temp = latest.with_suffix(".pt.tmp")
            torch.save(state, temp)
            os.replace(temp, latest)
            event = {"status": "TRAINING", "epoch": epoch + 1,
                     "train_loss": float(train_sum[0] / train_sum[1]),
                     "lr_after_epoch": optim.param_groups[0]["lr"],
                     "seconds": round(time.monotonic() - started, 1)}
            atomic(OUT / "status.json", event)
            print(json.dumps(event), flush=True)
        dist.barrier()
    if rank == 0:
        final = OUT / "checkpoint_epoch50.pt"
        os.replace(latest, final)
        atomic(OUT / "checkpoint_identity.json", {"path": str(final), "sha256": sha(final),
                                                  "epoch": 50, "train_n": len(data)})
        atomic(OUT / "status.json", {"status": "TRAINED", "checkpoint_sha256": sha(final)})
        print(json.dumps({"status": "TRAINED", "checkpoint_sha256": sha(final)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def load_model(device: str):
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    path = Path(identity["path"])
    if sha(path) != identity["sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol_sha256"] != sha(OUT / "protocol.json") or checkpoint["epoch"] != 50:
        raise RuntimeError("Checkpoint provenance mismatch")
    model = resnet50(pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def infer_one(name: str, model, device: str, workers: int, batch_size: int) -> None:
    data = rows(name)
    folder = OUT / "evaluation" / name
    predictions = folder / "predictions.jsonl"
    result = folder / "results.json"
    if result.exists() and json.loads(result.read_text()).get("status") == "COMPLETE":
        print(json.dumps({"dataset": name, "status": "ALREADY_COMPLETE"}), flush=True)
        return
    old = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()] if predictions.exists() else []
    if [r["sample_id"] for r in old] != [r["sample_id"] for r in data[:len(old)]]:
        raise RuntimeError(f"Prediction prefix mismatch: {name}")
    dataset = Images(data, transforms_for(False))
    loader = DataLoader(Subset(dataset, range(len(old), len(data))), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with predictions.open("a") as stream, torch.inference_mode():
        for pixels, targets, indices in loader:
            probs = torch.sigmoid(model(pixels.to(device, non_blocking=True))).view(-1).cpu().tolist()
            for index, target, prob in zip(indices.tolist(), targets.tolist(), probs):
                if int(target) != data[index]["gt"]:
                    raise RuntimeError(f"Label mismatch: {name}/{index}")
                stream.write(json.dumps({"sample_id": data[index]["sample_id"],
                                         "gt": int(target), "source": data[index]["source"],
                                         "prob_fake": prob}) + "\n")
            stream.flush()
            done = int(indices[-1]) + 1
            if done % 2048 < batch_size or done == len(data):
                print(json.dumps({"dataset": name, "done": done, "total": len(data),
                                  "seconds": round(time.monotonic() - started, 1)}), flush=True)
    all_rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if len(all_rows) != len(data) or [r["sample_id"] for r in all_rows] != [r["sample_id"] for r in data]:
        raise RuntimeError(f"Incomplete or reordered predictions: {name}")
    summary = {"status": "COMPLETE", "dataset": name, "n": len(data),
               "protocol_sha256": sha(OUT / "protocol.json"),
               "checkpoint_sha256": json.loads((OUT / "checkpoint_identity.json").read_text())["sha256"],
               "metrics": metrics(all_rows)}
    if name == "genimage":
        summary["per_generator"] = {source: metrics([r for r in all_rows if r["source"] == source])
                                    for source in sorted({r["source"] for r in all_rows})}
    atomic(result, summary)
    print(json.dumps({"dataset": name, "status": "COMPLETE", "metrics": summary["metrics"]}), flush=True)


def evaluate(names: list[str], device: str, workers: int, batch_size: int) -> None:
    validate_protocol()
    model = load_model(device)
    for name in names:
        infer_one(name, model, device, min(workers, 4) if name == "raise998" else workers,
                  min(batch_size, 16) if name == "raise998" else batch_size)


def finalize() -> None:
    validate_protocol()
    names = ("validation", *EVAL)
    results = {}
    for name in names:
        path = OUT / "evaluation" / name / "results.json"
        if not path.exists():
            raise RuntimeError(f"Missing complete result: {name}")
        result = json.loads(path.read_text())
        if result["status"] != "COMPLETE" or result["n"] != len(rows(name)):
            raise RuntimeError(f"Incomplete result: {name}")
        results[name] = result["metrics"]
    atomic(OUT / "results.json", {"status": "COMPLETE", "model": "official NPR retrained on project data",
                                  "results": results,
                                  "checkpoint": json.loads((OUT / "checkpoint_identity.json").read_text())})
    atomic(OUT / "status.json", {"status": "COMPLETE", "datasets": list(names)})
    print(json.dumps({"status": "COMPLETE", "datasets": list(names)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "eval", "finalize"))
    parser.add_argument("--datasets", nargs="+", choices=("validation", *EVAL))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "train": train(args.workers)
    elif args.command == "eval": evaluate(args.datasets or list(("validation", *EVAL)),
                                        args.device, args.workers, args.batch_size)
    else: finalize()


if __name__ == "__main__":
    main()
