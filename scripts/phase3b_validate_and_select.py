#!/usr/bin/env python3
"""Run frozen internal-val G0 at all Phase 3B intervals and freeze the selector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3b_replay import file_sha256, phrase_overlap


def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def load_jsonl(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--generation-batch-size", type=int, default=8); p.add_argument("--reset", action="store_true")
    p.add_argument("--available-only", action="store_true",
                   help="Evaluate checkpoints currently present; freeze only after all 8 exist.")
    cli = p.parse_args()
    config_path = (ROOT / cli.config).resolve(); config = yaml.safe_load(config_path.read_text())
    arm = "b0_gold_replay" if config["replay"]["context"] == "gold" else "b1_generated_replay"
    selection_root = ROOT / "outputs/phase3b_generated_replay/selection" / arm
    val_manifest = selection_root / "val_as_test_manifest"; val_manifest.mkdir(parents=True, exist_ok=True)
    source_val = ROOT / config["data"]["manifest_dir"] / "val_combined.jsonl"
    shutil.copy2(source_val, val_manifest / "test_combined.jsonl")
    checkpoint_root = (ROOT / config["checkpoint"]["output_root"]).resolve()
    interval = int(config["training"]["validation_interval"])
    total = int(config["training"]["total_optimizer_steps"])
    candidates = []
    rows_by_id = {r["sample_id"]: r for r in load_jsonl(source_val)}
    for step in range(interval, total + 1, interval):
        epoch = step // interval
        ckpt = checkpoint_root / f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
        ready_marker = checkpoint_root / f"step_{step:04d}/latest"
        if not ckpt.exists() or not ready_marker.exists():
            if cli.available_only: continue
            raise FileNotFoundError(ckpt)
        out = selection_root / f"step_{step:04d}"
        summary = out / "summary.json"
        if cli.reset or not summary.exists():
            cmd = [sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(config_path),
                   "--checkpoint", str(ckpt), "--output-dir", str(out), "--manifest-dir", str(val_manifest),
                   "--device", cli.device, "--modes", "detection", "G0", "--expected-step", str(step),
                   "--expected-epoch", str(epoch), "--generation-batch-size", str(cli.generation_batch_size), "--reset"]
            cmd.append("--skip-spatial-save")
            subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())
        summary_value = load_json(summary)
        g0 = load_json(out / "G0/metrics.json")
        predictions = load_jsonl(out / "G0/predictions.jsonl")
        phrase = []
        for row in predictions:
            authoritative = UnifiedForensicsDataset.authoritative_localization_field(rows_by_id[row["sample_id"]])["normalized_training_phrase"]
            phrase.append(phrase_overlap(authoritative, row.get("generated_localization_phrase")))
        lengths = [len(row.get("generated_token_ids") or []) for row in predictions]
        diag = {
            "target_presence_rate": sum(row.get("generated_localization_phrase") is not None for row in predictions) / len(predictions),
            "normalized_exact_rate": sum(x["normalized_exact_match"] for x in phrase) / len(phrase),
            "mean_token_f1": sum(x["normalized_token_f1"] for x in phrase) / len(phrase),
            "generation_length_mean": sum(lengths) / len(lengths), "generation_length_max": max(lengths),
            "seg_trigger_rate": g0.get("seg_trigger_rate"),
        }
        candidate = {
            "optimizer_step": step, "epoch": epoch, "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": file_sha256(ckpt),
            "val_fake_g0_mean_foreground_iou": g0["per_image_mean"]["foreground_iou"],
            "val_fake_g0_mean_foreground_f1": g0["per_image_mean"]["foreground_f1"],
            "classification": summary_value["modes"]["detection"]["classification_head"],
            "lm_verdict": summary_value["modes"]["detection"]["lm_verdict"],
            "cls_lm_agreement": summary_value["modes"]["detection"]["cls_lm_agreement"],
            "generation_diagnostics": diag,
        }
        dump(out / "selection_metrics.json", candidate); candidates.append(candidate)
    if not candidates:
        print("No available interval checkpoints"); return
    selected = max(candidates, key=lambda x: (x["val_fake_g0_mean_foreground_iou"], x["val_fake_g0_mean_foreground_f1"]))
    selector = {
        "status": ("FROZEN_BEFORE_INTERNAL_TEST_AND_OFFICIAL1000" if len(candidates) == total // interval
                   else "PARTIAL_VALIDATION_NOT_FROZEN"),
        "arm": arm, "selector": "max_internal_val_fake_g0_mean_foreground_iou_then_mean_foreground_f1",
        "selected_checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
        "optimizer_step": selected["optimizer_step"], "epoch": selected["epoch"],
        "selected_metrics": selected, "candidates": candidates,
        "test_used_for_selection": False, "official1000_used_for_selection": False,
    }
    name = f"{arm}_selector.json" if len(candidates) == total // interval else f"{arm}_partial_validation.json"
    dump(ROOT / "outputs/phase3b_generated_replay/selection" / name, selector)
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
