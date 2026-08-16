#!/usr/bin/env python3
"""Evaluate frozen Phase 3B selectors on internal test, then official1000."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--arm", choices=("b0_gold_replay", "b1_generated_replay"), required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--generation-batch-size", type=int, default=1)
    p.add_argument("--reset", action="store_true"); cli = p.parse_args()
    config = ROOT / "configs" / ("phase3b_b0_gold_replay.yaml" if cli.arm.startswith("b0") else "phase3b_b1_generated_replay.yaml")
    selector = load(ROOT / "outputs/phase3b_generated_replay/selection" / f"{cli.arm}_selector.json")
    common = [sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(config),
              "--checkpoint", selector["selected_checkpoint"], "--device", cli.device,
              "--expected-step", str(selector["optimizer_step"]), "--expected-epoch", str(selector["epoch"]),
              "--generation-batch-size", str(cli.generation_batch_size), "--modes", "detection", "G0", "tf_full_context"]
    env = os.environ.copy()
    internal = ROOT / "outputs/phase3b_generated_replay/evaluation/internal" / cli.arm
    command = common + ["--output-dir", str(internal)] + (["--reset"] if cli.reset else [])
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    official = ROOT / "outputs/phase3b_generated_replay/evaluation/official1000" / cli.arm
    command = common + ["--output-dir", str(official),
                        "--manifest-dir", str(ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest"),
                        "--synthscars-root", "/data/yz/myLISA_storage/AIGC/SynthScars"] + (["--reset"] if cli.reset else [])
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__": main()
