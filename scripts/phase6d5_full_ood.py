#!/usr/bin/env python3
"""Single-GPU resumable full-OOD replay for frozen Phase 6D.5 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model

OUT = ROOT / "outputs/phase6d5_full_classification_ood"
CFG = ROOT / "configs/phase6d3_c1_rine_conditioned_p1.yaml"
CKPT = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
FUSION = ROOT / "outputs/phase6d5_decision_integration/fusion_parameters.json"
MANIFESTS = {
    "aigi_holmes": ROOT / "datasets/AIGI-Holmes/manifests/eval_manifest.jsonl",
    "genimage": ROOT / "datasets/GenImage/manifests/eval_manifest.jsonl",
    "loki": ROOT / "datasets/LOKI/manifests/classification_eval_manifest.jsonl",
    "raise998": ROOT / "datasets/RAISE/manifests/eval_manifest.jsonl",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def label(row):
    value = row.get("class_label", row.get("label"))
    if isinstance(value, (int, bool)):
        return int(value)
    return int(str(value).strip().lower() == "fake")


def image_path(row):
    path = Path(row["image_path"])
    if not path.is_file():
        raise OSError(f"missing image: {path}")
    return str(path.resolve())


def read_image(row):
    path = image_path(row)
    value = cv2.imread(path, cv2.IMREAD_COLOR)
    if value is None:
        raise OSError(f"decode failed: {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def metric(records, score_key):
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    score = np.asarray([r[score_key] for r in records], dtype=np.float64)
    pred = score > 0
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    result = {
        "n": len(records), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()), "precision": precision,
        "fake_recall": recall, "tnr": tnr, "fpr": 1 - tnr,
        "f1": 2 * precision * recall / max(1e-30, precision + recall),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.0,
    }
    if len(set(y.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(y, score))
        result["auprc"] = float(average_precision_score(y, score))
        fpr, tpr, _ = roc_curve(y, score)
        for q in (0.01, 0.05, 0.10):
            result[f"recall_at_fpr_{int(q * 100)}pct"] = float(tpr[fpr <= q].max())
    else:
        result.update(roc_auc=None, auprc=None)
    return result


def extract(device_name, batch_size):
    config = yaml.safe_load(CFG.read_text())
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, metadata = load_model(
        config, CKPT, device, expected_step=2500, expected_epoch=5
    )
    processor = CLIPImageProcessor.from_pretrained(
        config["model"]["vision_tower"], local_files_only=True
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, max_new_tokens=400
    )

    for dataset, manifest in MANIFESTS.items():
        source = read_jsonl(manifest)
        destination = OUT / "raw" / f"{dataset}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = read_jsonl(destination) if destination.exists() else []
        existing_ids = [r["sample_id"] for r in existing]
        source_ids = [r["sample_id"] for r in source]
        if existing_ids != source_ids[:len(existing_ids)]:
            raise RuntimeError(f"resume prefix mismatch: {dataset}")
        pending = source[len(existing):]
        with destination.open("a") as handle, ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(pending), batch_size):
                part = pending[start:start + batch_size]
                images = list(pool.map(read_image, part))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                samples = [{
                    "image_path": image_path(row), "global_enc_image": pixel,
                    "grounding_enc_image": None, "bboxes": None, "conversations": [""],
                    "masks": None, "label": None, "resize": None, "questions": [],
                    "sampled_classes": [], "cls_label": label(row), "seg_valid": False,
                    "sample_id": row["sample_id"], "source": row.get("generator/source"),
                    "content_category": row.get("generator/source"), "manifest_row": row,
                } for row, pixel in zip(part, pixels)]
                batch = backend._batch_many(samples, "", question=UNIFIED_FORENSICS_QUESTION)
                batch["grounding_enc_images"] = None
                with torch.inference_mode():
                    output = model.model_forward(**batch)
                c1 = (output["cls_logits"][:, 1] - output["cls_logits"][:, 0]).float().cpu().tolist()
                rine = model._last_rine_binary_logits.float().cpu().tolist()
                for row, c1_margin, rine_margin in zip(part, c1, rine):
                    handle.write(json.dumps({
                        "sample_id": row["sample_id"], "dataset": dataset,
                        "label": label(row), "generator": row.get("generator/source"),
                        "image_path": image_path(row), "c1_margin": c1_margin,
                        "rine_margin": rine_margin,
                    }, ensure_ascii=False) + "\n")
                handle.flush()
                completed = len(existing) + min(start + batch_size, len(pending))
                if (start // batch_size + 1) % 50 == 0:
                    print(json.dumps({"dataset": dataset, "done": completed, "total": len(source)}), flush=True)
        records = read_jsonl(destination)
        if [r["sample_id"] for r in records] != source_ids:
            raise RuntimeError(f"identity/order mismatch: {dataset}")
        dump(OUT / "raw" / f"{dataset}.complete.json", {
            "status": "COMPLETE", "count": len(records),
            "manifest_sha256": sha256(manifest),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        })


def finalize():
    parameters = json.loads(FUSION.read_text())
    normalization = parameters["normalization"]
    alpha = float(parameters["selected_alpha"])
    result = {
        "schema": "phase6d5_full_classification_ood_v1", "status": "COMPLETE",
        "generated_at_utc": now(), "checkpoint": str(CKPT.resolve()),
        "checkpoint_sha256": sha256(CKPT), "fusion_parameters": parameters,
        "protocol": {
            "device": "cuda:1", "checkpoint_epoch": 5, "checkpoint_step": 2500,
            "prompt": "canonical unified fixed [CLS] query", "threshold": 0,
            "normalization": "frozen Phase6D.5 internal-TRAIN mean/std",
            "alpha": "frozen Phase6D.5 internal-validation selection",
            "ood_selection_or_tuning": False,
        }, "datasets": {}, "manifests": {},
    }
    for dataset, manifest in MANIFESTS.items():
        records = read_jsonl(OUT / "raw" / f"{dataset}.jsonl")
        source = read_jsonl(manifest)
        if [r["sample_id"] for r in records] != [r["sample_id"] for r in source]:
            raise RuntimeError(f"finalize identity/order mismatch: {dataset}")
        for row in records:
            sc = (row["c1_margin"] - normalization["c1"]["mean"]) / normalization["c1"]["std"]
            sr = (row["rine_margin"] - normalization["rine"]["mean"]) / normalization["rine"]["std"]
            row["d1_fixed_margin"] = 0.5 * sc + 0.5 * sr
            row["d1_alpha_margin"] = alpha * sr + (1 - alpha) * sc
        prediction = OUT / "predictions" / f"{dataset}.jsonl"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        arms = {
            "C1": metric(records, "c1_margin"),
            "D1-fixed": metric(records, "d1_fixed_margin"),
            "D1-alpha": metric(records, "d1_alpha_margin"),
        }
        entry = {"arms": arms, "prediction_file": str(prediction.resolve()),
                 "prediction_sha256": sha256(prediction)}
        if dataset == "genimage":
            generators = sorted({str(r["generator"]) for r in records})
            entry["per_generator"] = {
                generator: {
                    "C1": metric([r for r in records if str(r["generator"]) == generator], "c1_margin"),
                    "D1-fixed": metric([r for r in records if str(r["generator"]) == generator], "d1_fixed_margin"),
                    "D1-alpha": metric([r for r in records if str(r["generator"]) == generator], "d1_alpha_margin"),
                } for generator in generators
            }
        result["datasets"][dataset] = entry
        result["manifests"][dataset] = {
            "path": str(manifest.resolve()), "sha256": sha256(manifest), "count": len(source)
        }
    dump(OUT / "results.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "extract", "finalize"), default="run")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("run", "extract"):
        dump(OUT / "worker_status.json", {
            "status": "RUNNING", "pid": os.getpid(), "device": args.device,
            "batch_size": args.batch_size, "started_at_utc": now(),
        })
        try:
            extract(args.device, args.batch_size)
            if args.mode == "run":
                finalize()
            dump(OUT / "worker_status.json", {
                "status": "COMPLETE", "pid": os.getpid(), "device": args.device,
                "batch_size": args.batch_size, "updated_at_utc": now(),
            })
        except BaseException as error:
            dump(OUT / "worker_status.json", {
                "status": "FAILED", "pid": os.getpid(), "device": args.device,
                "exception_type": type(error).__name__, "exception": str(error),
                "updated_at_utc": now(),
            })
            raise
    else:
        finalize()


if __name__ == "__main__":
    main()
