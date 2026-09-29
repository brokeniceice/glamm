#!/usr/bin/env python3
"""Calibrate C2 on internal TRAIN, then reuse frozen test/OOD scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts import phase6d5_decision_integration as reference
from scripts.phase6j0_c2_evaluate import load_c2_model

PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"
OUT = ROOT / "outputs/phase6j0_c2/center"
TRAIN = ROOT / "outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl"
TEST = ROOT / "outputs/phase6j0_c2/evaluation/test/detection/predictions.jsonl"
OOD = ROOT / "outputs/phase6j0_c2/final_evaluation/classification_ood/results.json"
FINAL = ROOT / "outputs/phase6j0_c2/final_evaluation/results.json"
CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
CFG = ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml"
REPORT = ROOT / "docs/final_evaluation_report.md"
DATASETS = ("aigi_holmes", "genimage", "loki", "raise998")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def selected_checkpoint() -> tuple[int, int, str]:
    final = json.loads(FINAL.read_text())
    require(final["status"] == "COMPLETE", "C2 scoped evaluation incomplete")
    digest = sha(CKPT)
    require(final["selected_checkpoint_sha256"] == digest, "C2 checkpoint drift")
    return int(final["selected_step"]), int(final["selected_epoch"]), digest


def worker(rank: int, batch_size: int) -> None:
    step, epoch, checkpoint_sha = selected_checkpoint()
    device = torch.device(f"cuda:{(0, 2)[rank]}")
    torch.cuda.set_device(device)
    config = yaml.safe_load(CFG.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_c2_model(config, CKPT, device,
                                            expected_step=step, expected_epoch=epoch)
    require(meta["checkpoint_sha256"] == checkpoint_sha, "Loaded C2 checkpoint drift")
    processor = CLIPImageProcessor.from_pretrained(config["model"]["vision_tower"],
                                                   local_files_only=True)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device,
                                    dtype=torch.bfloat16, max_new_tokens=400)
    source = rows(TRAIN)
    assigned = [row for i, row in enumerate(source) if i % 2 == rank]
    path = OUT / f"train_rank{rank}.jsonl"
    existing = rows(path) if path.exists() else []
    require([row["sample_id"] for row in existing] ==
            [row["sample_id"] for row in assigned[:len(existing)]],
            f"TRAIN rank {rank} resume prefix drift")
    require(all(isinstance(row.get("margin"), (int, float)) and math.isfinite(row["margin"])
                for row in existing), f"TRAIN rank {rank} invalid prior margin")
    pending = assigned[len(existing):]
    with path.open("a", buffering=1) as handle, ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(pending), batch_size):
            part = pending[start:start + batch_size]
            images = list(pool.map(reference.image, part))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            samples = [{"image_path": reference.image_path(row),
                        "global_enc_image": pixel, "grounding_enc_image": None,
                        "bboxes": None, "conversations": [""], "masks": None,
                        "label": None, "resize": None, "questions": [],
                        "sampled_classes": [], "cls_label": reference.label(row),
                        "seg_valid": False, "sample_id": row["sample_id"],
                        "source": row.get("source", row.get("generator/source")),
                        "content_category": row.get("content_category", row.get("generator/source")),
                        "manifest_row": row}
                       for row, pixel in zip(part, pixels)]
            batch = backend._batch_many(samples, "", question=UNIFIED_FORENSICS_QUESTION)
            batch["grounding_enc_images"] = None
            with torch.inference_mode():
                output = model.model_forward(**batch)
            margins = (output["cls_logits"][:, 1] - output["cls_logits"][:, 0]).float().cpu().tolist()
            for row, margin in zip(part, margins):
                require(math.isfinite(margin), f"Nonfinite C2 TRAIN margin: {row['sample_id']}")
                handle.write(json.dumps({"sample_id": row["sample_id"],
                                         "label": reference.label(row), "margin": margin}) + "\n")
            if (start // batch_size + 1) % 50 == 0 or start + batch_size >= len(pending):
                print(json.dumps({"rank": rank, "done": len(existing) + min(start + batch_size, len(pending)),
                                  "total": len(assigned)}), flush=True)
    complete = rows(path)
    require([row["sample_id"] for row in complete] ==
            [row["sample_id"] for row in assigned], f"TRAIN rank {rank} incomplete")
    save(OUT / f"train_rank{rank}.complete.json",
         {"status": "COMPLETE", "rank": rank, "count": len(assigned),
          "checkpoint_sha256": checkpoint_sha, "manifest_sha256": sha(TRAIN),
          "predictions_sha256": sha(path)})


def metrics(records: list[dict], threshold: float, score_key: str,
            label_key: str, *, strict: bool) -> dict:
    y = np.asarray([int(row[label_key] == "fake") if label_key == "gt_label"
                    else int(row[label_key]) for row in records], dtype=np.int64)
    scores = np.asarray([float(row[score_key]) for row in records], dtype=np.float64)
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            "Invalid frozen classification probabilities")
    pred = (scores > threshold) if strict else (scores >= threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    tnr = tn / max(1, tn + fp)
    real_only = not bool(y.any())
    return {"n": len(y), "real": int((y == 0).sum()), "fake": int(y.sum()),
            "accuracy": float((pred == y).mean()),
            "f1": None if real_only else 2 * precision * recall / max(1e-30, precision + recall),
            "roc_auc": None if real_only else float(roc_auc_score(y, scores)),
            "fake_recall": None if real_only else recall,
            "tnr": tnr, "fpr": 1 - tnr, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "threshold_probability": threshold,
            "at_threshold_ties": int((scores == threshold).sum())}


def checked_records(path: Path, manifest: Path, *, test: bool = False) -> list[dict]:
    records = rows(path)
    population = rows(manifest)
    require([str(row["sample_id"]) for row in records] ==
            [str(row["sample_id"]) for row in population],
            f"Frozen score identity/order drift: {path}")
    if test:
        require(all((row["gt_label"] == "fake") == bool(int(man["class_label"]))
                    for row, man in zip(records, population)), "Internal test label drift")
    else:
        require(all(row["label"] == int(man["label"] == "Fake")
                    for row, man in zip(records, population)), f"OOD label drift: {path}")
    return records


def format_row(name: str, value: dict) -> str:
    def fmt(number):
        return "-" if number is None else f"{number:.6f}"
    return (f"| {name} | {value['n']} | {fmt(value['accuracy'])} | {fmt(value['f1'])} | "
            f"{fmt(value['roc_auc'])} | {fmt(value['fake_recall'])} | "
            f"{fmt(value['tnr'])} | {fmt(value['fpr'])} |")


def append_report(result: dict) -> None:
    sections = (
        ("### Internal2208（Real 1104，Fake 1104）", result["internal_test"]),
        ("### AIGI-Holmes TestSet（Real 50,000，Fake 49,999）", result["classification_ood"]["aigi_holmes"]),
        ("### GenImage held-out（Real 50,000，Fake 50,000）", result["classification_ood"]["genimage"]),
        ("### LOKI 分类（Real 900，Fake 1,317）", result["classification_ood"]["loki"]),
        ("### RAISE998（Real 998，Fake 0）", result["classification_ood"]["raise998"]),
    )
    lines = REPORT.read_text().splitlines()
    for heading, value in sections:
        require(lines.count(heading) == 1, f"C2-center report heading drift: {heading}")
        start = lines.index(heading)
        separator = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("|---")), None)
        require(separator is not None, f"C2-center report table missing: {heading}")
        end = separator + 1
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        require(not any("| C2-center |" in row for row in lines[separator:end]),
                f"C2-center already in table: {heading}")
        raw = next((i for i in range(separator + 1, end) if lines[i].startswith("| C2-raw |")), None)
        require(raw is not None, f"C2-raw row missing: {heading}")
        lines.insert(raw + 1, format_row("C2-center", value))
    model_description = (
        f"- **C2-center** 复用同一 C2 checkpoint 和逐图分类分数；内部训练集 "
        f"{result['train_count']:,} 张图像的 H2 Fake−Real logit margin 均值为 "
        f"`{result['train_margin_mean']:.9f}`。固定规则 `margin > mean`，对应 Fake 概率阈值 "
        f"`{result['probability_threshold']:.9g}`。仅分类判定改变，Official1000 定位与 C2-raw 相同。")
    anchor = "- **C2-raw**"
    require(sum(line.startswith(anchor) for line in lines) == 1, "C2-raw description missing")
    lines.insert(next(i for i, line in enumerate(lines) if line.startswith(anchor)) + 1,
                 model_description)
    source = ("- C2-center 使用冻结 internal-TRAIN 均值校准，内部测试和 OOD 逐图分数精确复用 C2-raw；"
              "没有在测试/OOD 上选阈值。计数、清单及 checkpoint 哈希见"
              "[C2-center 汇总](../outputs/phase6j0_c2/center/results.json)。")
    anchor = "- C2-raw 的内部测试"
    require(sum(line.startswith(anchor) for line in lines) == 1, "C2-raw source missing")
    lines.insert(next(i for i, line in enumerate(lines) if line.startswith(anchor)) + 1, source)
    updated = "\n".join(lines).rstrip() + "\n"
    temporary = REPORT.with_suffix(".md.tmp")
    temporary.write_text(updated)
    os.replace(temporary, REPORT)


def finalize() -> None:
    step, epoch, checkpoint_sha = selected_checkpoint()
    source = rows(TRAIN)
    require(len(source) == 17672, "Internal TRAIN population drift")
    by_id = {}
    for rank in (0, 1):
        complete = json.loads((OUT / f"train_rank{rank}.complete.json").read_text())
        path = OUT / f"train_rank{rank}.jsonl"
        require(complete["status"] == "COMPLETE" and
                complete["checkpoint_sha256"] == checkpoint_sha and
                complete["manifest_sha256"] == sha(TRAIN) and
                complete["predictions_sha256"] == sha(path), f"TRAIN rank {rank} provenance drift")
        records = rows(path)
        expected = [row for i, row in enumerate(source) if i % 2 == rank]
        require(len(records) == complete["count"] and
                [row["sample_id"] for row in records] == [row["sample_id"] for row in expected],
                f"TRAIN rank {rank} ID/order drift")
        for record, manifest in zip(records, expected):
            require(record["label"] == reference.label(manifest), "TRAIN label drift")
            require(record["sample_id"] not in by_id, "Duplicate TRAIN sample")
            by_id[record["sample_id"]] = record
    ordered = [by_id[row["sample_id"]] for row in source]
    margins = np.asarray([row["margin"] for row in ordered], dtype=np.float64)
    require(np.isfinite(margins).all(), "Nonfinite TRAIN margin")
    mean = float(margins.mean())
    threshold = 1 / (1 + math.exp(-mean)) if mean >= 0 else math.exp(mean) / (1 + math.exp(mean))
    frozen = json.loads(FINAL.read_text())
    ood = json.loads(OOD.read_text())
    require(ood["status"] == "COMPLETE" and ood["checkpoint_sha256"] == checkpoint_sha,
            "C2 OOD result/checkpoint drift")
    test_manifest = ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl"
    test = checked_records(TEST, test_manifest, test=True)
    require(len(test) == frozen["internal_test"]["samples"], "Internal test count drift")
    raw_test = metrics(test, .5, "cls_prob_fake", "gt_label", strict=False)
    test_reference = frozen["internal_test"]["modes"]["detection"]["classification_head"]
    for key in ("accuracy", "f1", "roc_auc", "fake_recall", "tnr"):
        if key in test_reference:
            require(abs(raw_test[key] - test_reference[key]) < 1e-9,
                    f"Internal test raw metric drift: {key}")
    test_center = metrics(test, threshold, "cls_prob_fake", "gt_label", strict=True)
    datasets = {}
    score_provenance = {"internal_test": {"path": str(TEST.resolve()), "sha256": sha(TEST),
                                          "manifest_sha256": sha(test_manifest)}}
    for name in DATASETS:
        manifest = Path(ood["manifests"][name]["path"])
        prediction = Path(ood["datasets"][name]["prediction_file"])
        require(sha(manifest) == ood["manifests"][name]["sha256"] and
                sha(prediction) == ood["datasets"][name]["prediction_sha256"],
                f"{name} frozen OOD provenance drift")
        records = checked_records(prediction, manifest)
        require(len(records) == ood["datasets"][name]["classification_head"]["n"],
                f"{name} OOD count drift")
        raw = metrics(records, .5, "cls_score_fake", "label", strict=False)
        reference_metrics = ood["datasets"][name]["classification_head"]
        for key in ("accuracy", "fake_recall", "tnr", "roc_auc"):
            if raw[key] is not None and reference_metrics[key] is not None:
                require(abs(raw[key] - reference_metrics[key]) < 1e-9,
                        f"{name} raw metric drift: {key}")
        datasets[name] = metrics(records, threshold, "cls_score_fake", "label", strict=True)
        score_provenance[name] = {"path": str(prediction.resolve()), "sha256": sha(prediction),
                                  "manifest_sha256": sha(manifest)}
    result = {"schema": "phase6j0_c2_center_v1", "status": "COMPLETE",
              "selected_checkpoint_sha256": checkpoint_sha, "selected_step": step,
              "selected_epoch": epoch, "train_count": len(source),
              "train_manifest_sha256": sha(TRAIN), "train_margin_mean": mean,
              "train_margin_std": float(margins.std()),
              "probability_threshold": threshold,
              "calibration": "internal TRAIN mean C2 H2 fake-minus-real logit margin",
              "no_test_or_ood_selection": True,
              "test_and_ood_execution": "EXACT_REUSE_FROZEN_C2_RAW_SCORES",
              "internal_test": test_center, "classification_ood": datasets,
              "score_provenance": score_provenance}
    save(OUT / "results.json", result)
    append_report(result)
    save(OUT / "status.json", {"status": "COMPLETE", "result": str((OUT / "results.json").resolve()),
                               "report": str(REPORT.resolve())})


def supervisor(batch_size: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "status.json", {"status": "RUNNING", "stage": "TRAIN_CALIBRATION"})
    try:
        processes = []
        for rank in (0, 1):
            with (OUT / f"worker_rank{rank}.log").open("a", buffering=1) as log:
                process = subprocess.Popen([PY, __file__, "--mode", "worker", "--rank", str(rank),
                                            "--batch-size", str(batch_size)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            processes.append(process)
        codes = [process.wait() for process in processes]
        require(codes == [0, 0], f"C2 TRAIN calibration workers failed: {codes}")
        save(OUT / "status.json", {"status": "RUNNING", "stage": "REUSE_FROZEN_TEST_OOD"})
        finalize()
    except Exception as exc:
        save(OUT / "status.json", {"status": "FAILED", "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("supervisor", "worker", "finalize"), required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "worker":
        require(args.rank is not None, "worker rank missing")
        worker(args.rank, args.batch_size)
    elif args.mode == "finalize":
        finalize()
    else:
        supervisor(args.batch_size)


if __name__ == "__main__":
    main()
