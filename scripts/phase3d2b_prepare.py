#!/usr/bin/env python3
"""Freeze the read-only Phase 3D.2-B matched re-evaluation contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256

OUT = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    historical = load(P3D2 / "evaluation/selector/selector.json")
    steps = [0, 50, 100, 150, 200, 250]
    records = []
    for step in steps:
        if step == 0:
            path = Path(cfg["source"]["checkpoint"])
            stored_step, epoch, source = int(cfg["source"]["optimizer_step"]), int(cfg["source"]["epoch"]), "Phase 3A P1"
        else:
            path = Path(cfg["experiment"]["checkpoint_root"]) / f"step_{step:04d}/checkpoint/mp_rank_00_model_states.pt"
            stored_step, epoch, source = step, step // 50, "Phase 3D.2 Direct Spatial-Path Optimization"
        records.append({
            "candidate_step": step, "stored_optimizer_step": stored_step, "logical_epoch": epoch,
            "checkpoint": artifact(path), "source_experiment": source,
            "historical_selector_status": "SELECTED" if step == 0 else "NOT_SELECTED",
            "matched_protocol_use": "RETROSPECTIVE_CANDIDATE_ONLY",
        })
    if records[0]["checkpoint"]["sha256"] != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checkpoint hash mismatch")
    checkpoint_manifest = {
        "status": "FROZEN_READ_ONLY", "phase": "Phase 3D.2-B", "checkpoints": records,
        "historical_selected_step": 0, "checkpoint_modification_allowed": False,
        "checkpoint_promotion_allowed": False,
    }
    dump(OUT / "manifests/checkpoint_manifest.json", checkpoint_manifest)
    dump(OUT / "checkpoint_manifest.json", checkpoint_manifest)
    protocol = {
        "status": "FROZEN_BEFORE_MATCHED_EVALUATION", "protocol_id": "phase3d2_training_matched_tf_phrase_v1",
        "only_changed_variable": "teacher-forced user prompt",
        "training_user_prompt": "Determine whether this image is authentic and explain the forensic evidence.",
        "matched_evaluation_user_prompt": "Determine whether this image is authentic and explain the forensic evidence.",
        "legacy_evaluation_user_prompt": "Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.",
        "assistant_target": "[FAKE] explanation\\nTarget regions: <authoritative phrase> [SEG]",
        "target_protocol": "phrase_aligned", "mask_target": "per_image_all_ref_union",
        "mask_logit_threshold": float(cfg["evaluation"]["mask_logit_threshold"]),
        "metric_aggregation": "per-image foreground IoU/F1 mean and median",
        "population": "frozen internal validation Fake", "population_count": 1106,
        "model_eval_required": True, "torch_no_grad_required": True,
    }
    dump(OUT / "manifests/matched_protocol_spec.json", protocol)
    dump(OUT / "matched_protocol_spec.json", protocol)
    val_source = ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl"
    validation_fake = [row for row in jsonl_rows(val_source) if int(row["class_label"]) == 1]
    if len(validation_fake) != 1106:
        raise RuntimeError(f"frozen validation Fake count drifted: {len(validation_fake)}")
    validation_alias = OUT / "manifests/validation_fake/test_combined.jsonl"
    write_jsonl(validation_alias, validation_fake)
    dump(OUT / "manifests/validation_population.json", {
        "status": "FROZEN", "count": len(validation_fake), "source": artifact(val_source),
        "sample_ids": [row["sample_id"] for row in validation_fake],
        "alias": artifact(validation_alias), "internal_test_used": False, "official1000_used": False,
    })
    experiment = {
        "status": "AUTHORIZED_READ_ONLY_MATCHED_REEVALUATION", "phase": "Phase 3D.2-B",
        "research_question": "Do existing Phase 3D.2 checkpoints improve localization under a training-matched teacher-forced protocol?",
        "phase3d2a_gate": "GATE_PHASE3D2_IMPLEMENTATION_MISMATCH_FOUND",
        "candidate_steps": steps, "historical_selected_step": int(historical["optimizer_step"]),
        "historical_results_preserved": True, "retrospective_selector_only": True,
        "allowed_population": "internal_validation_fake_1106_only",
        "prohibited": ["training", "backward", "optimizer", "scheduler", "parameter update", "checkpoint creation", "threshold tuning", "G0", "internal test", "official1000", "seen-train", "train-holdout", "soft-mask", "subgroup", "module-swap", "architecture modification"],
        "bootstrap_repeats": 10000, "bootstrap_seed": 3407,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "internal_test_used": False, "official1000_used": False, "G0_used": False,
    }
    dump(OUT / "manifests/experiment_manifest.json", experiment)
    dump(OUT / "experiment_manifest.json", experiment)
    print(json.dumps({"status": "PASS", "checkpoints": steps}, indent=2))


if __name__ == "__main__":
    main()
