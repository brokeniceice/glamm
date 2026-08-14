#!/usr/bin/env python3
"""Record immutable hashes for completed Phase 2C results without changing them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase2c_forensic_fusion"
MANIFEST = OUTPUT / "freeze_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    result_files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path != MANIFEST)
    source_files = [
        ROOT / "configs/phase2c_npr.yaml",
        ROOT / "configs/phase2c_srm.yaml",
        ROOT / "configs/phase2c_npr_srm.yaml",
        ROOT / "model/forensic_fusion.py",
        ROOT / "npr_expert/official_npr_srm.py",
        ROOT / "scripts/phase2c_forensic_fusion.py",
        ROOT / "scripts/phase2c_verify_cache.py",
        ROOT / "scripts/phase2c_finalize.py",
        ROOT / "scripts/phase2c_external_evaluate.py",
        ROOT / "scripts/phase2c_latency.py",
        ROOT / "scripts/phase2c_output_invariance.py",
        ROOT / "tests/test_phase2c_forensic_fusion.py",
    ]
    checkpoints = [
        ROOT / "checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt",
        ROOT / "checkpoints_stage1/official_npr_srm_20260802_062826/best.pth",
    ]
    selections = {}
    for variant in ("B1_npr", "B2_srm", "B3_npr_srm"):
        path = OUTPUT / variant / "train/selection.json"
        selections[variant] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "status": "PHASE2C_RESULTS_FROZEN",
        "frozen_on": "2026-08-11",
        "phase2b1_does_not_modify_listed_files": True,
        "result_file_count": len(result_files),
        "result_total_bytes": sum(path.stat().st_size for path in result_files),
        "result_files": [record(path) for path in result_files],
        "source_files": [record(path) for path in source_files],
        "frozen_checkpoints": [record(path) for path in checkpoints],
        "validation_only_selections": selections,
        "external_used_for_selection": False,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": len(result_files), "manifest": str(MANIFEST)}))


if __name__ == "__main__":
    main()
