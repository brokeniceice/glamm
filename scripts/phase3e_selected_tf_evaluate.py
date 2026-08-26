#!/usr/bin/env python3
"""Run canonical matched TF-PHRASE for a selected Phase 3E arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=("P1_FROZEN", "SFT_CONT", "JOINT"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True); cli = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(cli.physical_gpu)): raise RuntimeError("visible GPU mismatch")
    cfg = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    if cli.arm == "P1_FROZEN":
        checkpoint = Path(cfg["source"]["checkpoint"]).resolve(); expected_step = int(cfg["source"]["optimizer_step"]); expected_epoch = int(cfg["source"]["epoch"])
    else:
        selector = json.loads((root / f"evaluation/selector/{cli.arm}_selector.json").read_text())
        checkpoint = Path(selector["selected_checkpoint"]); selected_step = int(selector["optimizer_step"])
        if selected_step == 0:
            expected_step = int(cfg["source"]["optimizer_step"]); expected_epoch = int(cfg["source"]["epoch"])
        else:
            expected_step = selected_step; expected_epoch = selected_step // int(cfg["training"]["checkpoint_interval"])
    output = root / "evaluation/final" / cli.arm
    if not (output / "summary.json").is_file():
        command = [sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(ROOT / cfg["source"]["model_config"]),
                   "--checkpoint", str(checkpoint), "--output-dir", str(output),
                   "--manifest-dir", str(root / "evaluation/validation_manifest"), "--device", "cuda:0",
                   "--modes", "tf_full_context", "--expected-step", str(expected_step), "--expected-epoch", str(expected_epoch),
                   "--generation-batch-size", "1", "--skip-spatial-save", "--tf-user-prompt", "canonical", "--reset"]
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
    print(json.dumps({"status": "COMPLETE", "arm": cli.arm, "checkpoint": str(checkpoint),
                      "expected_step": expected_step, "expected_epoch": expected_epoch,
                      "output": str(output), "tf_user_prompt": "canonical"}, indent=2))


if __name__ == "__main__": main()
