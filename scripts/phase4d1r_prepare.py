#!/usr/bin/env python3
"""Freeze the single-variable Phase 4D-1R protocol before training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.position_aware_evidence_reader import PositionAwareEvidenceReader
from tools.phase4c_b import file_sha256
from tools.phase4d1 import canonical_hash, dump, reader_state_hash


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase4d1r_corrected_position_aware.yaml").read_text())
    parent_cfg = yaml.safe_load((ROOT / cfg["parent_phase4d1"]["config"]).read_text())
    bcfg = yaml.safe_load((ROOT / parent_cfg["phase4c_b"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    ckpt = Path(cfg["experiment"]["checkpoint_root"])
    parent = ROOT / cfg["parent_phase4d1"]["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    completion = json.loads((parent / "completion_manifest.json").read_text())
    if completion["status"] != "COMPLETE":
        raise RuntimeError("completed Phase 4D-1 required")
    p1 = Path(bcfg["p1"]["checkpoint"])
    if file_sha256(p1) != bcfg["p1"]["checkpoint_sha256"]:
        raise RuntimeError("P1 hash mismatch")

    subset = json.loads((parent / "phase4d1_train_subset.json").read_text())
    cache_complete = json.loads((Path(cfg["parent_phase4d1"]["cache_root"]) / "train_subset/complete.json").read_text())
    if subset["selected_n"] != 2048 or cache_complete["n"] != 2048:
        raise RuntimeError("frozen train population mismatch")
    if subset["sample_ids_sha256"] != cache_complete["sample_ids_sha256"]:
        raise RuntimeError("cache/subset order hash mismatch")

    init = torch.load(cfg["parent_phase4d1"]["reader_init"], map_location="cpu")
    reader = PositionAwareEvidenceReader()
    reader.load_state_dict(init["reader"])
    if reader_state_hash(reader) != init["reader_state_hash"] or float(reader.beta) != 0.0:
        raise RuntimeError("parent Reader initialization mismatch")
    parent_hash = reader_state_hash(reader)
    reader.beta.data.fill_(float(cfg["intervention"]["corrected_beta_initialization"]))
    corrected_hash = reader_state_hash(reader)

    files = {}
    for name in ("train_cross_image_mapping.json", "validation_cross_image_mapping.json",
                 "spatial_content_permutation.json"):
        payload = json.loads((parent / name).read_text())
        files[name] = {"path": str(parent / name), "sha256": canonical_hash(payload)}
        dump(out / name, payload)
    spatial = json.loads((parent / "spatial_content_permutation.json").read_text())
    if spatial.get("semantics") != "permute feature content only; fixed P_2D remains at lattice slots":
        raise RuntimeError("spatial shuffle semantics mismatch")

    payload = {
        "status": "PASS",
        "single_changed_variable": "beta_initialization: 0.0 -> 0.03",
        "beta_basis": cfg["intervention"]["selection_rule"],
        "performance_selection_used": False,
        "parent_reader_init_path": cfg["parent_phase4d1"]["reader_init"],
        "parent_reader_init_hash": parent_hash,
        "corrected_reader_init_hash": corrected_hash,
        "corrected_beta": 0.03,
        "train_subset_n": 2048,
        "train_subset_hash": subset["sample_ids_sha256"],
        "cache_order_hash": cache_complete["sample_ids_sha256"],
        "validation_n": 1106,
        "reused_mapping_files": files,
        "optimizer": parent_cfg["optimizer"],
        "loss": parent_cfg["loss"],
        "training": cfg["training"],
        "P1_checkpoint": {"path": str(p1), "sha256": bcfg["p1"]["checkpoint_sha256"]},
        "internal_test_access": False,
        "official1000_access": False,
        "training_started": False,
        "phase4d2_started": False,
    }
    dump(out / "preflight_manifest.json", payload)
    (out / "phase4d1r_preflight.md").write_text(
        "# Phase 4D-1R Preflight\n\nStatus: **PASS**\n\n"
        "- Only changed variable: `beta initialization: 0.0 -> 0.03`.\n"
        "- `0.03` is the smallest preregistered Phase 4D-1B alpha with POS-FORENSIC BF16 survival versus baseline q >=80%; it was not selected by IoU, matched-vs-shuffle, or binary masks.\n"
        f"- Parent Reader init hash: `{parent_hash}`; corrected two-arm init hash: `{corrected_hash}`.\n"
        f"- Frozen 2,048-sample order hash: `{subset['sample_ids_sha256']}`.\n"
        "- Architecture, positional encoding, caches, mappings, optimizer, scheduler, loss, batch size, seed, evaluator, inverse geometry, threshold, and validation population are inherited unchanged from Phase 4D-1.\n"
        "- Formal endpoint is step512; no checkpoint selector.\n"
        "- Internal test and official1000 remain sealed; Phase 4D-2 is not started.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
