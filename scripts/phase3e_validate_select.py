#!/usr/bin/env python3
"""Evaluate frozen Phase 3E candidates on internal validation and select with non-regression gates."""

from __future__ import annotations

import argparse
import json
import os
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


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def rows(path: Path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def semantic_summary(predictions, manifest_rows, encoder, idf):
    by_id = {row["sample_id"]: row for row in manifest_rows}; pairs = []; parsed_rows = []
    for prediction in predictions:
        reference = UnifiedForensicsDataset.authoritative_localization_field(by_id[prediction["sample_id"]])["normalized_training_phrase"]
        parsed = parse_phrase_aligned_generation(prediction.get("decoded_text") or "")
        pairs.append((reference, parsed.get("target_region"))); parsed_rows.append((prediction, parsed))
    values = set()
    for left, right in pairs:
        for phrase in (left, right): values.add(content_phrase(phrase)); values.update(content_tokens(phrase))
    embeddings = encoder.encode(values, batch_size=128); semantic = []; structure = []; detail = []
    for (reference, phrase), (prediction, parsed) in zip(pairs, parsed_rows):
        score = semantic_phrase_score(reference, phrase, idf, embeddings)
        struct = parse_structure(parsed, prediction.get("generated_token_ids") or [], 32004)
        semantic.append(score); structure.append(struct)
        detail.append({"sample_id": prediction["sample_id"], "reference_phrase": reference,
                       "generated_phrase": phrase, **score, **struct})
    mean = lambda key, values: float(np.mean([row[key] for row in values]))
    return {"R_phrase_sem": mean("R_phrase_sem", semantic), "R_key_soft": mean("R_key_soft", semantic),
            "R_sentence_sem": mean("R_sentence_sem", semantic),
            "structure_validity": mean("structural_validity", structure),
            "malformed_output_rate": float(np.mean([not row["structural_validity"] for row in structure])),
            "valid_seg_rate": mean("usable_seg", structure)}, detail


def evaluate(checkpoint: Path, step: int, epoch: int, output: Path, cfg, physical_gpu: int):
    if not (output / "summary.json").is_file():
        command = [sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(ROOT / cfg["source"]["model_config"]),
                   "--checkpoint", str(checkpoint), "--output-dir", str(output),
                   "--manifest-dir", str((ROOT / cfg["experiment"]["output_root"] / "evaluation/validation_manifest").resolve()),
                   "--device", "cuda:0", "--modes", "detection", "G0", "--expected-step", str(step),
                   "--expected-epoch", str(epoch), "--generation-batch-size", "1", "--skip-spatial-save", "--reset",
                   "--detection-user-prompt", "canonical"]
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=("SFT_CONT", "JOINT"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True); cli = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
    if cli.physical_gpu != int(cfg["runtime"][f"{cli.arm}_gpu"]): raise RuntimeError("GPU assignment mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(cli.physical_gpu)): raise RuntimeError("visible GPU mismatch")
    root = (ROOT / cfg["experiment"]["output_root"]).resolve(); eval_root = root / "evaluation/selector"
    manifest_rows = rows(ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl")
    qcfg = yaml.safe_load((ROOT / "configs/phase3d0r_reward_reformulation.yaml").read_text())
    enc = qcfg["semantic_encoder"]; encoder = FrozenSentenceEncoder(enc["model_id"], enc["revision"], enc["cache_dir"], "cpu")
    idf = load(ROOT / "outputs/phase3d0r_reward_reformulation/token_idf.json")["values"]
    p1 = Path(cfg["source"]["checkpoint"]).resolve(); p1_out = eval_root / "P1_FROZEN"
    evaluate(p1, int(cfg["source"]["optimizer_step"]), int(cfg["source"]["epoch"]), p1_out, cfg, cli.physical_gpu)

    def metrics(output, checkpoint, step):
        g0 = load(output / "G0/metrics.json"); summary = load(output / "summary.json")
        predictions = rows(output / "G0/predictions.jsonl"); semantic, detail = semantic_summary(predictions, manifest_rows, encoder, idf)
        with (output / "semantic_structure.jsonl").open("w", encoding="utf-8") as handle:
            for row in detail: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        detection = summary["modes"]["detection"]["classification_head"]
        value = {"optimizer_step": step, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                 "mean_foreground_iou": g0["per_image_mean"]["foreground_iou"],
                 "mean_foreground_f1": g0["per_image_mean"]["foreground_f1"],
                 "median_foreground_iou": float(np.median([r["foreground_iou"] for r in predictions])),
                 "median_foreground_f1": float(np.median([r["foreground_f1"] for r in predictions])),
                 "classification_accuracy": detection["accuracy"], "classification_f1": detection["f1"],
                 "classification_confusion": {k: detection[k] for k in ("tp", "tn", "fp", "fn")}, **semantic}
        dump(output / "selection_metrics.json", value); return value

    baseline = metrics(p1_out, p1, 0); candidates = []
    for step in cfg["selector"]["candidate_steps"]:
        if int(step) == 0: continue
        checkpoint = Path(cfg["experiment"]["checkpoint_root"]) / cli.arm / f"step_{int(step):04d}/checkpoint/mp_rank_00_model_states.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        output = eval_root / cli.arm / f"step_{int(step):04d}"
        evaluate(checkpoint, int(step), int(step) // int(cfg["training"]["checkpoint_interval"]), output, cfg, cli.physical_gpu)
        value = metrics(output, checkpoint, int(step))
        value["non_regression"] = {
            "classification_accuracy": value["classification_accuracy"] >= baseline["classification_accuracy"] - float(cfg["selector"]["classification_accuracy_max_drop"]),
            "classification_f1": value["classification_f1"] >= baseline["classification_f1"] - float(cfg["selector"]["classification_f1_max_drop"]),
            "structure_validity": value["structure_validity"] >= baseline["structure_validity"] - float(cfg["selector"]["structure_validity_max_drop"]),
            "no_seg_or_malformed_collapse": value["valid_seg_rate"] > 0 and value["malformed_output_rate"] < 1,
        }
        value["eligible"] = all(value["non_regression"].values()); dump(output / "selection_metrics.json", value); candidates.append(value)
    # Phase3E step 0 is the frozen P1 initialization and is an explicit
    # preregistered candidate, so an arm must fall back when every trained
    # checkpoint is worse even if those checkpoints pass non-regression.
    baseline["eligible"] = True
    baseline["non_regression"] = {"initialization_reference": True}
    eligible = [baseline, *[row for row in candidates if row["eligible"]]]
    selected = max(eligible, key=lambda row: (row["mean_foreground_iou"], row["mean_foreground_f1"], -row["optimizer_step"]))
    result = {"status": "FROZEN_AFTER_COMPLETE_INTERNAL_VALIDATION", "arm": cli.arm,
              "selector": "max_val_fake_G0_mean_FG_IoU_then_F1_after_preregistered_nonregression_gates",
              "baseline": baseline, "selected_checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
              "optimizer_step": selected["optimizer_step"], "selected_metrics": selected, "candidates": candidates,
              "fallback_to_P1": selected is baseline, "training_loss_used": False, "TF_used": False,
              "internal_test_used": False, "official1000_used": False, "threshold_tuned": False,
              "detection_user_prompt": "canonical", "g0_user_prompt": "canonical", "generation_batch_size": 1}
    dump(eval_root / f"{cli.arm}_selector.json", result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
