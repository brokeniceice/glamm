#!/usr/bin/env python3
"""Evaluate C1-Exact on the frozen internal validation split and write Phase 6B.1a artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

import phase5a3_stage2_train as stage2


ROOT = Path(__file__).resolve().parents[1]
C1L_RESULTS = ROOT / "outputs/phase6b_classification_attribution/results.json"


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
    temporary.replace(path)


def metrics(labels: np.ndarray, prob_fake: np.ndarray) -> dict:
    pred = (prob_fake >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, prob_fake)),
        "fake_recall": float(tp / (tp + fn)),
        "tnr": float(tn / (tn + fp)),
        "fpr": float(fp / (tn + fp)),
        "f1": float(f1_score(labels, pred)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vision-pretrained", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    summary_path = args.run_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("best_metric") is None:
        raise RuntimeError("C1-Exact training is not complete")

    cli = argparse.Namespace(
        stage1_le=str(args.run_dir / "final_model"),
        vision_pretrained=args.vision_pretrained,
        vision_tower=args.vision_tower,
        train_json=args.train_json,
        val_json=args.val_json,
        run_dir=str(args.run_dir), batch_size=64, workers=args.workers, seed=3407,
    )
    _, _, model, trainable = stage2.load(cli)
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = stage2.FrozenClsDataset(args.val_json, args.vision_tower)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.workers)
    labels_official, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["global_enc_images"].to(device=device, dtype=torch.bfloat16)
            logits = model(global_enc_images=pixels, inference_cls=True)["logits"].float()
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())
            labels_official.extend(batch["cls_gt_list"].tolist())
    # Official LEGION uses Real=1/Fake=0. Report all paper-facing metrics with Fake=1.
    labels = 1 - np.asarray(labels_official, dtype=np.int64)
    prob_fake = np.asarray(probabilities, dtype=np.float64)
    exact_metrics = metrics(labels, prob_fake)

    c1l = json.loads(C1L_RESULTS.read_text(encoding="utf-8"))["arms"]["C1-L"]
    c1l_metrics = {key: value["mean"] for key, value in c1l.items()
                   if isinstance(value, dict) and "mean" in value}
    selected = min(
        (row for row in summary["validation_by_epoch"]
         if row["eval_accuracy"] == summary["best_metric"]),
        key=lambda row: row["step"],
    )
    result = {
        "schema": "phase6b1a_exact_stage2_control_v1", "status": "COMPLETE",
        "candidate": "C1-Exact",
        "architecture": "frozen CLIP penultimate CLS 1024 -> Linear(2048) -> ReLU -> Linear(2)",
        "contract": {
            "batch_size": 64, "gradient_accumulation_steps": 1,
            "effective_global_batch": 64, "learning_rate": 0.001,
            "weight_decay": 0.0, "epochs": 3, "scheduler": "cosine",
            "selector": "maximum internal-validation Accuracy; earliest checkpoint on tie",
            "threshold": 0.5, "seed": 3407,
            "optimizer_trainer": "same transformers Trainer path as LEGION-retrained Stage2",
            "labels_during_training": {"real": 1, "fake": 0},
            "feature_path": "online frozen CLIP BF16 penultimate CLS",
        },
        "population": {"train": 17672, "validation": len(dataset)},
        "trainable_parameter_count": sum(item[2] for item in trainable),
        "validation_by_epoch": summary["validation_by_epoch"],
        "selected_epoch": selected["epoch"], "selected_step": selected["step"],
        "selected_checkpoint": summary["best_checkpoint"],
        "final_model": str((args.run_dir / "final_model").resolve()),
        "metrics": exact_metrics,
        "c1_l_three_seed_mean": c1l_metrics,
        "delta_c1_exact_minus_c1_l_mean": {
            key: exact_metrics[key] - c1l_metrics[key]
            for key in ("accuracy", "roc_auc", "fake_recall", "tnr", "fpr", "f1")
        },
        "artifacts": {
            "training_summary": str(summary_path.resolve()),
            "training_summary_sha256": sha256(summary_path),
            "validation_manifest": str(Path(args.val_json).resolve()),
            "validation_manifest_sha256": sha256(Path(args.val_json)),
        },
        "deviations_from_legion_retrained_stage2": [],
        "firewall": {"internal_train": True, "internal_validation": True,
                     "internal_test": False, "external_benchmarks": False},
    }
    atomic_json(args.run_dir / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
