#!/usr/bin/env python3
"""Launch the pinned official Stage-1 merge without modifying its repository."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/LEGION_official"

# The pinned repository eagerly imports dataset families that the merge never
# uses.  Its utils module does not define their prompt constants, and its
# bundled dataset package omits grefer.  Supply the same inert import shim used
# by the external training launcher, then execute the official merge verbatim.
sys.path.insert(0, str(OFFICIAL))
import dataset.utils as dataset_utils_package  # noqa: E402
import dataset.utils.utils as dataset_prompts  # noqa: E402

dataset_utils_package.__path__.append(str(ROOT / "dataset/utils"))
for name in ("CAPTION_QUESTIONS", "REGION_QUESTIONS", "REGION_GROUP_QUESTIONS", "SEG_QUESTIONS"):
    if not hasattr(dataset_prompts, name):
        setattr(dataset_prompts, name, [])

sys.path.insert(0, str(OFFICIAL / "scripts/loc_exp"))
# The merge script retained GLaMM's pre-rename initializer while the pinned
# LEGION model renamed it to initialize_legion_model.  Alias only at runtime;
# both names invoke the exact official LEGION implementation.
from model.Legion import LegionModel  # noqa: E402

if not hasattr(LegionModel, "initialize_glamm_model"):
    LegionModel.initialize_glamm_model = LegionModel.initialize_legion_model
runpy.run_path(str(OFFICIAL / "scripts/merge_weights/merge_lora_weights.py"), run_name="__main__")
