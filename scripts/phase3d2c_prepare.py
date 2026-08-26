#!/usr/bin/env python3
"""Freeze Phase 3D.2-C populations and checkpoint attribution objects."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256

OUT = ROOT / "outputs/phase3d2c_spatial_generalization_attribution"
P3D2A = ROOT / "outputs/phase3d2a_spatial_attribution_audit"
P3D2B = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    gate_b = load(P3D2B / "route_gate.json")
    if gate_b["primary_gate"] != "GATE_MATCHED_SPATIAL_OPTIMIZATION_NO_VALIDATION_GAIN":
        raise RuntimeError("Phase 3D.2-B prerequisite gate mismatch")
    seen_manifest = load(P3D2A / "seen_train_manifest.json")
    holdout_manifest = load(P3D2A / "train_holdout_manifest.json")
    if seen_manifest["total_exposures"] != seen_manifest["unique_image_count"] or seen_manifest["unique_image_count"] != 1000:
        raise RuntimeError("SEEN-TRAIN exposure/identity contract drift")
    if holdout_manifest["selected_count"] != 1000 or holdout_manifest["seed"] != 3407:
        raise RuntimeError("TRAIN-HOLDOUT frozen contract drift")
    if set(seen_manifest["sample_ids"]) & set(holdout_manifest["sample_ids"]):
        raise RuntimeError("sample leakage between SEEN-TRAIN and TRAIN-HOLDOUT")
    if set(seen_manifest["image_identities"]) & set(holdout_manifest["image_identities"]):
        raise RuntimeError("identity leakage between SEEN-TRAIN and TRAIN-HOLDOUT")
    populations = {
        "seen_train": P3D2A / "populations/seen_train/test_combined.jsonl",
        "train_holdout": P3D2A / "populations/train_holdout/test_combined.jsonl",
        "validation_fake": P3D2B / "manifests/validation_fake/test_combined.jsonl",
    }
    expected = {"seen_train": 1000, "train_holdout": 1000, "validation_fake": 1106}
    for name, source in populations.items():
        values = rows(source)
        if len(values) != expected[name] or len({row["sample_id"] for row in values}) != expected[name]:
            raise RuntimeError(f"{name} population drift")
        destination = OUT / f"populations/{name}/test_combined.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        dump(OUT / f"manifests/{name}_population.json", {
            "status": "REUSED_FROZEN", "population": name, "count": len(values),
            "source": artifact(source), "alias": artifact(destination),
            "sample_ids": [row["sample_id"] for row in values],
        })
    dump(OUT / "manifests/seen_train_manifest.json", seen_manifest)
    dump(OUT / "seen_train_manifest.json", seen_manifest)
    dump(OUT / "manifests/train_holdout_manifest.json", holdout_manifest)
    dump(OUT / "train_holdout_manifest.json", holdout_manifest)

    source_checkpoints = load(P3D2B / "checkpoint_manifest.json")["checkpoints"]
    by_step = {int(row["candidate_step"]): row for row in source_checkpoints}
    selected_steps = [0, 50, 100, 250]
    checkpoints = []
    for step in selected_steps:
        row = dict(by_step[step])
        row["attribution_role"] = {
            0: "original_P1_baseline", 50: "early_training_state",
            100: "matched_validation_significant_degradation_point", 250: "final_late_stage_diagnostic",
        }[step]
        row["formal_selector_eligible_in_phase3d2c"] = False
        checkpoints.append(row)
    checkpoint_manifest = {
        "status": "FROZEN_READ_ONLY", "phase": "Phase 3D.2-C", "checkpoints": checkpoints,
        "primary_attribution_steps": selected_steps,
        "phase3d2b_steps_150_200_preserved_but_not_reevaluated": True,
        "formal_checkpoint_reselection_allowed": False,
    }
    dump(OUT / "manifests/checkpoint_attribution_manifest.json", checkpoint_manifest)
    dump(OUT / "checkpoint_attribution_manifest.json", checkpoint_manifest)
    experiment = {
        "status": "AUTHORIZED_READ_ONLY_ATTRIBUTION", "phase": "Phase 3D.2-C",
        "research_question": "Why did Phase 3D.2 spatial optimization not yield matched validation localization gain?",
        "prerequisite_gate": gate_b["primary_gate"],
        "protocol": "phase3d2_training_matched_tf_phrase_v1",
        "checkpoints": selected_steps, "populations": expected,
        "bootstrap_repeats": 10000, "bootstrap_seed": 3407, "binary_mask_logit_threshold": 0.0,
        "allowed": ["matched read-only evaluation", "soft-mask diagnostics", "predefined subgroup attribution", "in-memory module swap", "representation logging"],
        "prohibited": ["training", "backward", "optimizer", "scheduler", "parameter update", "checkpoint creation", "checkpoint promotion", "threshold search", "internal test", "official1000", "G0", "FEPN", "NPR", "SRM", "FOCAL", "architecture modification"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "internal_test_used": False, "official1000_used": False, "G0_used": False,
    }
    dump(OUT / "manifests/experiment_manifest.json", experiment)
    dump(OUT / "experiment_manifest.json", experiment)
    print(json.dumps({"status": "PASS", "steps": selected_steps, "populations": expected}, indent=2))


if __name__ == "__main__":
    main()
