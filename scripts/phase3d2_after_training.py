#!/usr/bin/env python3
"""Run the frozen Phase 3D.2 selector and final validation evaluations."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase3d2_direct_spatial_path.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    summary = root / "training/run_summary.json"
    if not summary.is_file() or load(summary).get("status") != "COMPLETE":
        raise RuntimeError("Phase 3D.2 training is not complete")
    candidates = list(map(int, cfg["selector"]["candidate_steps"]))
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"]).resolve()
    val_alias = root / "evaluation/selector/val_as_test_manifest"
    model_config = ROOT / cfg["source"]["model_config"]

    def spec(step):
        if step == 0:
            return Path(cfg["source"]["checkpoint"]), int(cfg["source"]["optimizer_step"]), int(cfg["source"]["epoch"])
        interval = int(cfg["training"]["checkpoint_interval"])
        return checkpoint_root / f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt", step, step // interval

    def eval_partition(gpu, steps):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["GLAMM_PRESERVE_CUDA_CACHE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        for step in steps:
            checkpoint, expected_step, epoch = spec(step)
            output = root / f"evaluation/tf_phrase/step_{step:04d}"
            command = [
                PYTHON, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(model_config),
                "--checkpoint", str(checkpoint), "--output-dir", str(output),
                "--manifest-dir", str(val_alias), "--device", "cuda:0", "--modes", "tf_full_context",
                "--expected-step", str(expected_step), "--expected-epoch", str(epoch),
                "--skip-spatial-save", "--reset",
            ]
            log = root / f"logs/tf_step_{step:04d}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    partitions = [candidates[::2], candidates[1::2]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(eval_partition, gpu, steps) for gpu, steps in zip(cfg["runtime"]["evaluation_gpus"], partitions)]
        for future in futures: future.result()

    records = []
    for step in candidates:
        checkpoint, expected_step, epoch = spec(step)
        metrics = load(root / f"evaluation/tf_phrase/step_{step:04d}/tf_full_context/metrics.json")
        records.append({
            "optimizer_step": step, "checkpoint": str(checkpoint), "stored_optimizer_step": expected_step,
            "logical_epoch": epoch, "mean_foreground_iou": metrics["per_image_mean"]["foreground_iou"],
            "mean_foreground_f1": metrics["per_image_mean"]["foreground_f1"],
        })
    selected = max(records, key=lambda row: (row["mean_foreground_iou"], row["mean_foreground_f1"], -row["optimizer_step"]))
    selector = {
        "status": "FROZEN_AFTER_COMPLETE_INTERNAL_VALIDATION", "arm": "P3D2-SPATIAL",
        "selector": cfg["selector"]["primary"], "tie_breaker": cfg["selector"]["tie_breaker"],
        "selected_checkpoint": selected["checkpoint"], "optimizer_step": selected["optimizer_step"],
        "selected_metrics": selected, "candidates": records,
        "G0_used": False, "training_loss_used": False, "internal_test_used": False, "official1000_used": False,
    }
    dump(root / "evaluation/selector/selector.json", selector)

    def final_eval(gpu, arm, checkpoint, expected_step, epoch):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["GLAMM_PRESERVE_CUDA_CACHE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        output = root / f"evaluation/g0/{arm}"
        command = [
            PYTHON, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(model_config),
            "--checkpoint", str(checkpoint), "--output-dir", str(output), "--manifest-dir", str(val_alias),
            "--device", "cuda:0", "--modes", "detection", "G0", "--expected-step", str(expected_step),
            "--expected-epoch", str(epoch), "--generation-batch-size", str(cfg["evaluation"]["generation_batch_size"]),
            "--skip-spatial-save", "--reset",
        ]
        with (root / f"logs/{arm}_g0.log").open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    selected_checkpoint, selected_step, selected_epoch = spec(int(selected["optimizer_step"]))
    jobs = [
        (cfg["runtime"]["evaluation_gpus"][0], "P1_FROZEN", Path(cfg["source"]["checkpoint"]),
         int(cfg["source"]["optimizer_step"]), int(cfg["source"]["epoch"])),
        (cfg["runtime"]["evaluation_gpus"][1], "P3D2_SPATIAL", selected_checkpoint, selected_step, selected_epoch),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(final_eval, *job) for job in jobs]
        for future in futures: future.result()

    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(cfg["runtime"]["evaluation_gpus"][0])
    with (root / "logs/behavioral_invariance.log").open("w", encoding="utf-8") as handle:
        subprocess.run([
            PYTHON, str(ROOT / "scripts/phase3d2_behavior_audit.py"),
            "--selected-checkpoint", str(selected_checkpoint), "--selected-step", str(selected_step),
            "--selected-epoch", str(selected_epoch),
        ], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    subprocess.run([PYTHON, str(ROOT / "scripts/phase3d2_finalize.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
