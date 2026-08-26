#!/usr/bin/env python3
"""Evaluate all Phase 3D.1 interval checkpoints on frozen internal validation and select one."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.phase3a_metrics import parse_phrase_aligned_generation
from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3d0 import parse_structure
from tools.phase3d0r import FrozenSentenceEncoder, content_phrase, content_tokens, semantic_phrase_score


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def semantic_summary(predictions, manifest_rows, encoder, idf):
    by_id = {row["sample_id"]: row for row in manifest_rows}
    pairs = []
    parsed_rows = []
    for prediction in predictions:
        reference = UnifiedForensicsDataset.authoritative_localization_field(
            by_id[prediction["sample_id"]]
        )["normalized_training_phrase"]
        parsed = parse_phrase_aligned_generation(prediction.get("decoded_text") or "")
        phrase = parsed.get("target_region")
        pairs.append((reference, phrase))
        parsed_rows.append((prediction, parsed))
    values = set()
    for left, right in pairs:
        for phrase in (left, right):
            values.add(content_phrase(phrase)); values.update(content_tokens(phrase))
    embeddings = encoder.encode(values, batch_size=128)
    semantic, structure = [], []
    detail = []
    for (reference, phrase), (prediction, parsed) in zip(pairs, parsed_rows):
        score = semantic_phrase_score(reference, phrase, idf, embeddings)
        struct = parse_structure(parsed, prediction.get("generated_token_ids") or [], 32004)
        semantic.append(score); structure.append(struct)
        detail.append({
            "sample_id": prediction["sample_id"], "reference_phrase": reference,
            "generated_phrase": phrase, **score, **struct,
        })
    return {
        "R_phrase_sem": float(np.mean([row["R_phrase_sem"] for row in semantic])),
        "R_key_soft": float(np.mean([row["R_key_soft"] for row in semantic])),
        "R_sentence_sem": float(np.mean([row["R_sentence_sem"] for row in semantic])),
        "structure_validity": float(np.mean([row["structural_validity"] for row in structure])),
        "malformed_output_rate": float(np.mean([not row["structural_validity"] for row in structure])),
        "valid_seg_rate": float(np.mean([row["usable_seg"] for row in structure])),
    }, detail


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("R3", "Q2"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    cli = parser.parse_args(argv)
    expected = 1 if cli.arm == "R3" else 2
    if cli.physical_gpu != expected:
        raise RuntimeError(f"frozen assignment for {cli.arm} is GPU {expected}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != str(cli.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}, expected {cli.physical_gpu}")
    cfg = yaml.safe_load((ROOT / "configs/phase3d1_policy_optimization.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    source_val = (ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl").resolve()
    manifest_rows = rows(source_val)
    val_as_test = root / "evaluation/selector/val_as_test_manifest"
    val_as_test.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_val, val_as_test / "test_combined.jsonl")
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]).resolve() / f"{cli.arm}_OPT"
    selection_root = root / "evaluation/selector" / f"{cli.arm}_OPT"
    qcfg = yaml.safe_load((ROOT / "configs/phase3d0r_reward_reformulation.yaml").read_text())
    enc = qcfg["semantic_encoder"]
    encoder = FrozenSentenceEncoder(enc["model_id"], enc["revision"], enc["cache_dir"], "cpu")
    idf = load(ROOT / cfg["reward"]["Q2"]["token_idf"])["values"]
    candidates = []
    interval = int(cfg["training"]["checkpoint_interval"])
    total = int(cfg["training"]["total_optimizer_steps"])
    for step in range(interval, total + 1, interval):
        checkpoint = checkpoint_root / f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = selection_root / f"step_{step:04d}"
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            command = [
                sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"),
                "--config", str(ROOT / cfg["source"]["model_config"]),
                "--checkpoint", str(checkpoint), "--output-dir", str(output),
                "--manifest-dir", str(val_as_test), "--device", "cuda:0",
                "--modes", "detection", "G0", "--expected-step", str(step),
                "--expected-epoch", str(step // interval),
                "--generation-batch-size", str(cli.generation_batch_size),
                "--skip-spatial-save", "--reset",
            ]
            subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
        summary = load(summary_path)
        g0 = load(output / "G0/metrics.json")
        predictions = rows(output / "G0/predictions.jsonl")
        semantic, detail = semantic_summary(predictions, manifest_rows, encoder, idf)
        with (output / "semantic_structure.jsonl").open("w", encoding="utf-8") as handle:
            for row in detail:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        detection = summary["modes"]["detection"]["classification_head"]
        candidate = {
            "optimizer_step": step, "logical_epoch": step // interval,
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "mean_foreground_iou": g0["per_image_mean"]["foreground_iou"],
            "mean_foreground_f1": g0["per_image_mean"]["foreground_f1"],
            "classification_accuracy": detection["accuracy"],
            "classification_f1": detection["f1"],
            **semantic,
        }
        dump(output / "selection_metrics.json", candidate)
        candidates.append(candidate)
    selected = max(candidates, key=lambda row: (
        row["mean_foreground_iou"], row["mean_foreground_f1"], -row["optimizer_step"],
    ))
    result = {
        "status": "FROZEN_AFTER_COMPLETE_INTERNAL_VALIDATION",
        "arm": f"P3D1-{cli.arm}",
        "selector": "max_internal_val_fake_mean_foreground_iou_then_mean_foreground_f1",
        "selected_checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "optimizer_step": selected["optimizer_step"],
        "logical_epoch": selected["logical_epoch"],
        "selected_metrics": selected, "candidates": candidates,
        "training_reward_used": False, "test_used": False,
        "official1000_used": False, "validation_reward_retuning": False,
    }
    dump(root / "evaluation/selector" / f"{cli.arm}_selector.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
