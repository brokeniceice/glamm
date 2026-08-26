#!/usr/bin/env python3
"""Freeze Phase 4E-1 valid-G0 populations and immutable provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/phase4e1_tf_fdg_full_method.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    replay_path = ROOT / CFG["data"]["replay_cache"]
    replay = rows(replay_path)
    train_valid = [str(row["sample_id"]) for row in replay if row["replay_eligible"]]
    train_invalid = [str(row["sample_id"]) for row in replay if not row["replay_eligible"]]
    val_root = ROOT / CFG["data"]["validation_context_cache"] / "G0"
    val_records = []
    for path in sorted(val_root.glob("shard_*.pt")):
        import torch
        val_records.extend(torch.load(path, map_location="cpu", weights_only=False)["records"])
    val_valid = [str(row["sample_id"]) for row in val_records if row["valid_q_seg"]]
    val_invalid = [str(row["sample_id"]) for row in val_records if not row["valid_q_seg"]]
    groups = {
        "train_valid_g0_ids": train_valid,
        "train_invalid_g0_ids": train_invalid,
        "val_valid_g0_ids": val_valid,
        "val_invalid_g0_ids": val_invalid,
    }
    expected = {"train_valid_g0_ids": 8690, "train_invalid_g0_ids": 146,
                "val_valid_g0_ids": 1078, "val_invalid_g0_ids": 28}
    if len(replay) != 8836 or len(val_records) != 1106:
        raise RuntimeError("formal population count mismatch")
    if any(len(groups[key]) != count for key, count in expected.items()):
        raise RuntimeError("valid-G0 count mismatch")
    if set(train_valid) & set(train_invalid) or set(val_valid) & set(val_invalid):
        raise RuntimeError("valid/invalid population overlap")
    for name, ids in groups.items():
        dump(OUT / "manifests" / f"{name}.json", {
            "schema": "phase4e1_frozen_id_set_v1", "name": name,
            "count": len(ids), "ordered_ids_sha256": canonical_hash(ids), "sample_ids": ids,
        })
    referenced = [
        CFG["p1"]["checkpoint"], CFG["evidence"]["forensic_checkpoint"],
        CFG["evidence"]["clip_checkpoint"], str(replay_path), str(val_root / "complete.json"),
    ]
    hashes = {str(Path(path).resolve()): file_hash(Path(path)) for path in referenced}
    checks = {
        "counts_exact": True, "sets_disjoint": True,
        "train_union_exact": len(set(train_valid) | set(train_invalid)) == 8836,
        "val_union_exact": len(set(val_valid) | set(val_invalid)) == 1106,
        "p1_hash_exact": hashes[str(Path(CFG["p1"]["checkpoint"]).resolve())] == CFG["p1"]["checkpoint_sha256"],
        "forensic_checkpoint_hash_exact": hashes[str(Path(CFG["evidence"]["forensic_checkpoint"]).resolve())] == CFG["evidence"]["forensic_checkpoint_sha256"],
        "clip_checkpoint_hash_exact": hashes[str(Path(CFG["evidence"]["clip_checkpoint"]).resolve())] == CFG["evidence"]["clip_checkpoint_sha256"],
        "prompt_hash_exact": {row["prompt_sha256"] for row in replay} == {CFG["p1"]["prompt_sha256"]},
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "approved_amendment": "valid-G0 conditional optimization",
        "counts": {key: len(value) for key, value in groups.items()},
        "hashes": {key: canonical_hash(value) for key, value in groups.items()},
        "source_file_hashes": hashes, "checks": checks,
        "hidden_extraction_rule": CFG["p1"]["hidden_extraction"],
        "canonical_prompt": CFG["p1"]["prompt"], "canonical_prompt_sha256": CFG["p1"]["prompt_sha256"],
        "batch_rule": "loss mean over n_valid_g0 only; all-invalid batch has no backward/optimizer/scheduler advance",
        "traversal_exposures": 88360, "optimization_eligible_valid_g0_exposures": 86900,
        "shared_by_all_stage_s_arms": CFG["arms"]["tier1"] + CFG["arms"]["tier2_priority"],
        "formal_validation_population": 1106, "valid_g0_only_diagnostic_population": 1078,
        "internal_test_accessed": False, "official1000_accessed": False,
        "formal_optimizer_updates": 0,
    }
    dump(OUT / "preflight" / "approved_protocol_amendment.json", result)
    dump(OUT / "experiment_manifest.json", {
        "phase": "Phase 4E-1", "status": "PREFLIGHT_POPULATIONS_FROZEN",
        "frozen_protocol_files": CFG["protocol"], "config": "configs/phase4e1_tf_fdg_full_method.yaml",
        "test_seal": CFG["seal"], "formal_optimizer_updates": 0,
    })
    dump(OUT / "preflight" / "full_clip_evidence_source.json", {
        "status": "FROZEN",
        "feature_source": "frozen P1 CLIP hidden-state-minus-2 patch lattice",
        "projection": "Phase 4C-A projection-only Conv2d(1024,256,1)",
        "projection_checkpoint": CFG["evidence"]["clip_checkpoint"],
        "projection_checkpoint_sha256": CFG["evidence"]["clip_checkpoint_sha256"],
        "selected_epoch": CFG["evidence"]["clip_selected_epoch"],
        "input_tensor_shape": [1024, 24, 24], "output_tensor_shape": [256, 24, 24],
        "input_normalization": "CLIPImageProcessor ResizeShortest336 CenterCrop336 Rescale Normalize v1",
        "output_normalization": "none beyond learned projection",
        "cache_manifest_train": "outputs/phase3c1_spatial_probe/cache/clip/train/complete.json",
        "cache_manifest_val": "outputs/phase3c1_spatial_probe/cache/clip/val/complete.json",
        "matched_difference_from_full_forensic": "evidence source only",
        "internal_test_accessed": False, "official1000_accessed": False,
    })
    if result["status"] != "PASS":
        raise RuntimeError("Phase 4E-1 population preflight failed")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
