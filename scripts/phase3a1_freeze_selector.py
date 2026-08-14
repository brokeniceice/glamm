#!/usr/bin/env python3
"""Freeze the paired C0 selector before any Phase 3A.1 official-test inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a1_paired_control"
METRICS = OUT / "c0_training/run/metrics.jsonl"
BEST = Path("/data/yz/groundingLMM_official/checkpoints/phase3a1_paired_control/c0/best/checkpoint/mp_rank_00_model_states.pt")
LAST = Path("/data/yz/groundingLMM_official/checkpoints/phase3a1_paired_control/c0/last/checkpoint/mp_rank_00_model_states.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = [json.loads(line) for line in METRICS.read_text(encoding="utf-8").splitlines() if line]
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != list(range(1, 5001)):
        raise RuntimeError("C0 canonical metrics are not the exact contiguous steps 1..5000")
    best = torch.load(BEST, map_location="cpu")
    last = torch.load(LAST, map_location="cpu")
    if int(last.get("optimizer_step", -1)) != 5000:
        raise RuntimeError("C0 last checkpoint is not durable step 5000")
    selector = {
        "status": "FROZEN_BEFORE_OFFICIAL_TEST",
        "selector": "min_validation_total_loss",
        "selected_checkpoint": str(BEST),
        "checkpoint_sha256": sha256(BEST),
        "optimizer_step": int(best["optimizer_step"]),
        "epoch": int(best["epoch"]),
        "best_val_total_loss": float(best["best_val_total_loss"]),
        "training_completed_step": 5000,
        "official_test_used_for_selection": False,
        "validation_localization_used_for_selection": False,
        "training_protocol": "P1-matched new C0; historical language target",
    }
    dump(OUT / "selection/new_c0_selector.json", selector)
    dump(OUT / "selection/c0_selector.json", selector)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "stage": "C0_SELECTOR_FROZEN_OFFICIAL_TEST_NOT_STARTED",
        "new_c0_trained": True,
        "new_c0_selector_frozen": True,
        "official_test_started": False,
    })
    dump(manifest_path, manifest)
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
