#!/usr/bin/env python3
"""Resumable official LEGION Stage-2 classification on frozen comparison sets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from transformers import CLIPImageProcessor


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/LEGION_official"
INTERNAL_MANIFEST = ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl"
INTERNAL_R1 = ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal/detection/predictions.jsonl"
AIGI_SOURCE = ROOT / "datasets/AIGI-Holmes-Dataset/dataset/test.jsonl"
AIGI_R1 = ROOT / "outputs/phase3a1_paired_control/evaluation/external_classification/p1/aigi_test/predictions.jsonl"
EXPECTED = {
    "internal": {"n": 2208, "manifest_sha": "b5c16ff15c59f7bf4f327530d89668347bdec3cc002d4e334264fdcf97d856fa", "r1_sha": "00204b50f3481a47576f2d434b8227c6325bf06c92813942da74d2e5af94fadc", "batch": 1},
    "aigi_test": {"n": 1731, "manifest_sha": "375e25309c844033cfb1dd5169260f6291c3410e3397ea6ed68b2b4f63bb162a", "r1_sha": "461cf7ce0af247c1c6c3424a895c6f1839fe048d825b9afbc40d3ed0679502aa", "batch": 8},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def ordered_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def load_scope(name: str) -> tuple[list[dict], Path]:
    if name == "internal":
        manifest, historical = rows(INTERNAL_MANIFEST), rows(INTERNAL_R1)
        if sha256(INTERNAL_MANIFEST) != EXPECTED[name]["manifest_sha"] or sha256(INTERNAL_R1) != EXPECTED[name]["r1_sha"]:
            raise RuntimeError("internal frozen artifact drift")
        by_id = {row["sample_id"]: row for row in historical}
        if [row["sample_id"] for row in manifest] != [row["sample_id"] for row in historical]:
            raise RuntimeError("internal historical R1 order differs from frozen manifest")
        scope = [{
            "sample_id": row["sample_id"], "image_path": by_id[row["sample_id"]]["image_path"],
            "gt": int(row["class_label"]), "source": row.get("source"),
        } for row in manifest]
        source = INTERNAL_MANIFEST
    elif name == "aigi_test":
        source_rows, historical = rows(AIGI_SOURCE), rows(AIGI_R1)
        if sha256(AIGI_SOURCE) != EXPECTED[name]["manifest_sha"] or sha256(AIGI_R1) != EXPECTED[name]["r1_sha"]:
            raise RuntimeError("AIGI-test frozen artifact drift")
        if len(source_rows) != len(historical):
            raise RuntimeError("AIGI-test count mismatch")
        scope = []
        for index, (source_row, old) in enumerate(zip(source_rows, historical)):
            expected_id = f"aigi_test:{index:06d}"
            gt = int(isinstance(source_row.get("mask"), str) and bool(source_row.get("mask", "").strip()))
            image = source_row.get("images") or []
            resolved = str((ROOT / "datasets/AIGI-Holmes-Dataset" / str(image[0]).removeprefix("./")).resolve())
            if old["sample_id"] != expected_id or Path(old["image_path"]).resolve() != Path(resolved) or old["gt_label"] != ("fake" if gt else "real"):
                raise RuntimeError(f"AIGI-test row drift at {index}")
            scope.append({"sample_id": expected_id, "image_path": old["image_path"], "gt": gt, "source": old.get("source")})
        source = AIGI_SOURCE
    else:
        raise ValueError(name)
    if len(scope) != EXPECTED[name]["n"] or len({row["sample_id"] for row in scope}) != len(scope):
        raise RuntimeError(f"{name} population drift")
    return scope, source


def load_official(model_dir: Path, device: torch.device):
    repo = str(OFFICIAL.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("phase5a4_official_cls_eval", OFFICIAL / "scripts/cls/eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    saved = sys.argv
    sys.argv = [saved[0], "--version", str(model_dir), "--pretrained", "--precision", "bf16",
                "--vision_tower", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
                "--vision_pretrained", "/data/yz/myLISA_storage/checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"]
    try:
        args = module.parse_args()
    finally:
        sys.argv = saved
    model, _ = module.load_model(args)
    model = model.to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(args.vision_tower, local_files_only=True)
    return model, processor


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = ((probabilities >= .5).astype(np.int64) == labels).astype(np.float64)
    total = len(labels)
    value = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if selected.any():
            value += selected.sum() / total * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(value)


def summarize(records: list[dict]) -> dict:
    y = np.asarray([row["gt"] for row in records], dtype=np.int64)
    p = np.asarray([row["prob_fake"] for row in records], dtype=np.float64)
    pred = (p >= .5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "n": len(y), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()), "precision": float(precision), "recall": float(recall),
        "specificity": float(tn / max(1, tn + fp)), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)), "ece_15": ece(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": .5,
    }


def paired_against_r1(name: str, records: list[dict]) -> dict:
    old_path = INTERNAL_R1 if name == "internal" else AIGI_R1
    old = rows(old_path)
    if [row["sample_id"] for row in old] != [row["sample_id"] for row in records]:
        raise RuntimeError(f"{name} paired order mismatch")
    old_correct = np.asarray([row["cls_pred"] == row["gt_label"] for row in old])
    new_correct = np.asarray([row["pred"] == ("fake" if row["gt"] else "real") for row in records])
    old_right_new_wrong = int((old_correct & ~new_correct).sum())
    old_wrong_new_right = int((~old_correct & new_correct).sum())
    discordant = old_right_new_wrong + old_wrong_new_right
    return {
        "r1_is_exact_p1_reuse": True,
        "r1_artifact": str(old_path.resolve()), "r1_artifact_sha256": sha256(old_path),
        "r1_correct_legion_wrong": old_right_new_wrong,
        "r1_wrong_legion_correct": old_wrong_new_right,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": float(binomtest(min(old_right_new_wrong, old_wrong_new_right), discordant, .5).pvalue) if discordant else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    status_path = args.output_root / "classification_worker.json"
    state = {"schema": "phase5a4_legion_retrained_cls_worker_v1", "status": "RUNNING", "started_at_utc": utc_now(), "pid": os.getpid(), "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "model_dir": str(args.model_dir.resolve())}
    atomic_json(status_path, state)
    started = time.monotonic()
    try:
        model, processor = load_official(args.model_dir.resolve(), device)
        for name in ("internal", "aigi_test"):
            scope, source_path = load_scope(name)
            output = args.output_root / "classification" / name
            prediction_path = output / "predictions.jsonl"
            existing = rows(prediction_path) if prediction_path.exists() else []
            indexed = {row["sample_id"]: row for row in existing}
            if any(sample_id not in {row["sample_id"] for row in scope} for sample_id in indexed):
                raise RuntimeError(f"{name} resume artifact scope drift")
            pending = [row for row in scope if row["sample_id"] not in indexed]
            batch_size = EXPECTED[name]["batch"]
            for start in range(0, len(pending), batch_size):
                batch_rows = pending[start:start + batch_size]
                images = []
                for row in batch_rows:
                    bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise OSError(row["image_path"])
                    images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pixels = processor.preprocess(images, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    logits = model(global_enc_images=pixels, inference_cls=True)["logits"].float()
                    fake_probabilities = torch.softmax(logits, dim=-1)[:, 0].cpu().tolist()
                values = [{
                    **row, "prob_fake": float(probability), "pred": "fake" if probability >= .5 else "real",
                    "official_label_semantics": "Real=1,Fake=0",
                } for row, probability in zip(batch_rows, fake_probabilities)]
                append_jsonl(prediction_path, values)
                if (start + len(batch_rows)) % 100 == 0 or start + len(batch_rows) == len(pending):
                    print(json.dumps({"dataset": name, "new_done": start + len(batch_rows), "pending": len(pending)}), flush=True)
            complete = rows(prediction_path)
            by_id = {row["sample_id"]: row for row in complete}
            ordered = [by_id[row["sample_id"]] for row in scope]
            if len(by_id) != len(scope):
                raise RuntimeError(f"{name} result incomplete")
            prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
            result = {
                "schema": "phase5a4_legion_retrained_cls_result_v1", "status": "COMPLETE", "dataset": name,
                "source_manifest": str(source_path.resolve()), "source_manifest_sha256": sha256(source_path),
                "ordered_sample_id_sha256": ordered_sha([row["sample_id"] for row in ordered]),
                "batch_size": batch_size, "checkpoint": str(args.model_dir.resolve()),
                "metrics": summarize(ordered), "paired_vs_r1": paired_against_r1(name, ordered),
            }
            atomic_json(output / "results.json", result)
        torch.cuda.synchronize(device)
        state.update({"status": "COMPLETE", "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    except BaseException as error:
        state.update({"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error)})
        raise
    finally:
        state.update({"finished_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started})
        atomic_json(status_path, state)


if __name__ == "__main__":
    main()
