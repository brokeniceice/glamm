#!/usr/bin/env python3
"""Run match-checkpoint L-FREE localization on X-AIGD or PAL4VST."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import final_eval_localization_supervisor as base

MODEL = ROOT / "checkpoints/legion_retrained_match/stage1_le"
IDENTITY = ROOT / "checkpoints/legion_retrained_match/stage1_le_identity.json"
OUT = ROOT / "outputs/legion_retrained_match_evaluation/localization"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("xaigd", "pal4vst"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    identity = json.loads(IDENTITY.read_text())
    base.OUT = OUT

    def match_identity(model: str):
        if model != "legion_retrained": return base.checkpoint_identity(model)
        return {"path": str(MODEL.resolve()), "canonical_sha256": identity["canonical_sha256"],
                "kind": "huggingface_directory", "role": "legion_retrained_match_stage1_LE"}

    base.checkpoint_identity = match_identity
    base.run_legion("legion_retrained", args.dataset, args.device)


if __name__ == "__main__": main()
