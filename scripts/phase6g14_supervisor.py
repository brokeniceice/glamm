#!/usr/bin/env python3
"""Detached sequential Phase6G.14 supervisor for GPU 1."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def main():
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    for arm in ("A0", "A1"):
        subprocess.run(
            [PY, str(ROOT / "scripts/phase6g14_single_matched_translator.py"),
             "--arm", arm, "--device", "cuda:1"],
            cwd=ROOT, env=env, check=True,
        )
    subprocess.run(
        [PY, str(ROOT / "scripts/phase6g14_finalize.py")],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    main()
