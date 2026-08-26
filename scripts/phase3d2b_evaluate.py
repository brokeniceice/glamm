#!/usr/bin/env python3
"""Run the six frozen matched-prompt validation evaluations for Phase 3D.2-B."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    consistency = load(OUT / "matched_protocol_consistency_audit.json")
    if consistency.get("status") != "PASS":
        raise RuntimeError("matched consistency audit did not pass; full validation is forbidden")
    checkpoint_manifest = load(OUT / "checkpoint_manifest.json")
    specs = checkpoint_manifest["checkpoints"]
    if [row["candidate_step"] for row in specs] != [0, 50, 100, 150, 200, 250]:
        raise RuntimeError("candidate set drift")
    manifest_dir = OUT / "manifests/validation_fake"
    if len((manifest_dir / "test_combined.jsonl").read_text(encoding="utf-8").splitlines()) != 1106:
        raise RuntimeError("validation Fake alias drift")
    model_config = ROOT / "configs/phase3a_p1.yaml"

    def evaluate_partition(physical_gpu: int, rows: list[dict]) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        env["GLAMM_PRESERVE_CUDA_CACHE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        for record in rows:
            step = int(record["candidate_step"])
            output = OUT / f"evaluation/matched_tf_phrase/step_{step:04d}"
            command = [
                PYTHON, str(ROOT / "scripts/phase3a_evaluate.py"),
                "--config", str(model_config), "--checkpoint", record["checkpoint"]["path"],
                "--output-dir", str(output), "--manifest-dir", str(manifest_dir),
                "--device", "cuda:0", "--modes", "tf_full_context",
                "--expected-step", str(record["stored_optimizer_step"]),
                "--expected-epoch", str(record["logical_epoch"]),
                "--tf-user-prompt", "canonical", "--skip-spatial-save", "--reset",
            ]
            log = OUT / f"logs/matched_step_{step:04d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    partitions = [specs[::2], specs[1::2]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate_partition, gpu, rows) for gpu, rows in zip((1, 2), partitions)]
        for future in futures:
            future.result()
    for record in specs:
        step = int(record["candidate_step"])
        summary = load(OUT / f"evaluation/matched_tf_phrase/step_{step:04d}/summary.json")
        if summary.get("tf_user_prompt") != "canonical" or summary.get("test_samples") != 1106:
            raise RuntimeError(f"matched evaluation summary contract failed at step {step}")
    print(json.dumps({"status": "PASS", "evaluated_steps": [row["candidate_step"] for row in specs]}, indent=2))


if __name__ == "__main__":
    main()
