#!/usr/bin/env python3
"""Use the existing Phase 3A evaluator with the new C2 model class."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from model.c2_preln_cross_attention import C2GLaMMForCausalLM
from scripts import phase3a_evaluate as evaluator
from scripts import phase2a_final_evaluate as base_evaluator


def load_c2_model(config, checkpoint_path, device, **kwargs):
    if config["experiment"]["name"] != "phase6j0_c2_preln_cross_attention":
        raise ValueError("C2 evaluator received an unrelated config")
    glamm_train.GLaMMForCausalLM = C2GLaMMForCausalLM
    # The existing loader's C1 branch creates the heads and invokes the
    # subclass's override, which adds the C2 attention block.
    loader_config = copy.deepcopy(config)
    loader_config["forensics"]["conditioning"] = "frozen_rine_q2_direct_embedding_token"
    return base_evaluator.load_model(loader_config, checkpoint_path, device, **kwargs)


def main():
    evaluator.load_model = load_c2_model
    evaluator.main()


if __name__ == "__main__":
    main()
