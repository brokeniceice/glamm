#!/usr/bin/env python3
"""Finalize the Phase 3D.2-A Audit-A stop without running downstream attribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256

OUT = ROOT / "outputs/phase3d2a_spatial_attribution_audit"
GATE = "GATE_PHASE3D2_IMPLEMENTATION_MISMATCH_FOUND"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    consistency = load(OUT / "train_eval_consistency_audit.json")
    if consistency["status"] != "MISMATCH_FOUND" or consistency.get("gate") != GATE:
        raise RuntimeError("Audit A mismatch stop is not established")
    experiment = load(OUT / "experiment_manifest.json")
    experiment.update({
        "status": "COMPLETE_STOPPED_AFTER_AUDIT_A", "primary_gate": GATE,
        "downstream_attribution_executed": False, "repair_or_rerun_executed": False,
        "training_or_parameter_update_performed": False,
    })
    dump(OUT / "experiment_manifest.json", experiment)
    dump(OUT / "manifests/experiment_manifest.json", experiment)
    diagnosis = {
        "status": "STOPPED_AFTER_AUDIT_A", "primary_attribution_gate": GATE,
        "failure_class": "train_eval_implementation_mismatch",
        "material_mismatch": {
            "component": "language user prompt before the identical authoritative assistant target",
            "training_question": "Determine whether this image is authentic and explain the forensic evidence.",
            "evaluation_question": "Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.",
            "code_path": {
                "training": "UnifiedForensicsDataset._conversation uses CANONICAL_UNIFIED_USER_CONTENT; Phase 3D.2 consumes dataset conversations",
                "evaluation": "teacher_forced_localization -> _causal_forward -> _batch; _batch defaults to FORENSICS_QUESTION",
            },
            "fixed_sample_result": {
                "n": len(consistency["sample_records"]),
                "raw_input_token_sequences_equal": sum(row["input_ids_equal"] for row in consistency["sample_records"]),
                "gt_masks_exact_equal": sum(row["gt_mask_pixel_exact_equal"] for row in consistency["sample_records"]),
                "hidden_max_abs_diff_range": [
                    min(row["frozen_seg_hidden"]["max_abs_diff"] for row in consistency["sample_records"]),
                    max(row["frozen_seg_hidden"]["max_abs_diff"] for row in consistency["sample_records"]),
                ],
                "mask_logit_max_abs_diff_range": [
                    min(row["postprocessed_mask_logits"]["max_abs_diff"] for row in consistency["sample_records"]),
                    max(row["postprocessed_mask_logits"]["max_abs_diff"] for row in consistency["sample_records"]),
                ],
            },
        },
        "scientific_consequence": "Phase 3D.2 training checkpoints were selected with a TF-PHRASE evaluator that did not reproduce the training user prompt. Current mechanism attribution is paused; the prior numerical selector table remains historical fact, but it cannot answer matched train/eval learnability or generalization.",
        "not_inferred": [
            "Phase 3D.2 would improve under a corrected evaluator",
            "direct spatial optimization effectively fits seen training samples",
            "direct spatial optimization overfits",
            "mask decoder or text_hidden_fcs is the primary bottleneck",
            "representation redesign or a forensic branch is required",
        ],
        "repair_attempted": False, "retraining_attempted": False, "checkpoint_reselection_attempted": False,
        "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "failure_attribution.json", diagnosis)
    route = {
        "status": "STOPPED_WAITING_FOR_AUTHORIZATION", "phase": "Phase 3D.2-A",
        "primary_gate": GATE,
        "decision": "Pause Phase 3D.2 scientific interpretation until a separately authorized protocol-resolution stage.",
        "downstream_attribution_executed": False,
        "automatic_fix_or_rerun_authorized": False,
        "P1_formal_checkpoint_unchanged": True, "phase3d2_formal_selected_step_unchanged": 0,
        "next_route": "Request separate authorization for a matched evaluator/protocol audit; do not repair, retrain, retune, or reselect automatically.",
    }
    dump(OUT / "route_gate.json", route)

    skipped = {
        "status": "NOT_RUN_STOPPED_BY_AUDIT_A", "blocking_gate": GATE,
        "reason": "The preregistered stop condition forbids mechanism attribution after a material train/eval mismatch.",
        "internal_test_used": False, "official1000_used": False,
    }
    required = {
        "population_metrics.json": "checkpoint x population evaluation",
        "population_paired_bootstrap.json": "paired checkpoint statistics",
        "loss_metric_trajectory.json": "training-loss versus metric trajectory",
        "soft_mask_diagnostics.json": "soft-mask diagnostics",
        "subgroup_attribution.json": "predefined subgroup attribution",
        "module_swap_attribution.json": "read-only module swap attribution",
        "representation_change_audit.json": "representation change audit",
    }
    for filename, analysis in required.items():
        dump(OUT / filename, {**skipped, "analysis": analysis})
    # Mirror structured-directory outputs without pretending that analyses ran.
    for rel, filename in (
        ("evaluation/population_metrics.json", "population_metrics.json"),
        ("statistics/population_paired_bootstrap.json", "population_paired_bootstrap.json"),
        ("statistics/loss_metric_trajectory.json", "loss_metric_trajectory.json"),
        ("soft_masks/soft_mask_diagnostics.json", "soft_mask_diagnostics.json"),
        ("subgroups/subgroup_attribution.json", "subgroup_attribution.json"),
        ("module_swap/module_swap_attribution.json", "module_swap_attribution.json"),
        ("representations/representation_change_audit.json", "representation_change_audit.json"),
    ):
        dump(OUT / rel, load(OUT / filename))
    figure = OUT / "figures/loss_metric_trajectory.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 3.2))
    plt.axis("off")
    plt.text(0.5, 0.60, "Not run: Audit A found a material train/eval mismatch", ha="center", va="center", fontsize=13)
    plt.text(0.5, 0.40, GATE, ha="center", va="center", fontsize=10, family="monospace")
    plt.tight_layout(); plt.savefig(figure, dpi=160); plt.close()
    (OUT / "loss_metric_trajectory.png").write_bytes(figure.read_bytes())

    report = f"""# Phase 3D.2-A — Spatial Optimization Attribution Audit

## Final status

`STOPPED_AFTER_AUDIT_A`; primary gate: **`{GATE}`**.

This phase was a read-only diagnostic. All model execution used `model.eval()` and `torch.no_grad()`. No backward, optimizer, scheduler, parameter update, checkpoint creation, threshold search, checkpoint reselection, architecture modification, internal test access, or official1000 access occurred.

## Scientific question

The authorized question was whether Phase 3D.2 failed because it did not fit, did not generalize, used a mismatched interface/evaluator, or contained only localized signal. The preregistered ordering required train/eval consistency before any mechanism attribution.

## Checkpoint provenance

The frozen diagnostic checkpoints were step 0, 50, 100 and 250. The formal Phase 3D.2 selector remains step 0/P1. No checkpoint was promoted or reselected. See `checkpoint_audit_manifest.json`.

## Audit A — train/eval consistency

The authoritative assistant target matches: `[FAKE] explanation\\nTarget regions: <authoritative phrase> [SEG]`. Image preprocessing, all-ref union GT mask construction, SAM postprocess and the formal logit threshold also match. The user prompt does not match:

- training: `Determine whether this image is authentic and explain the forensic evidence.`
- TF-PHRASE evaluation: `Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.`

The training dataset builds the canonical unified prompt, while `teacher_forced_localization()` calls `_causal_forward()`, which calls `_batch()` without overriding its legacy `FORENSICS_QUESTION` default.

Across 8 fixed preregistered training samples, 0/8 raw token sequences were equal, while 8/8 GT masks were pixel-exact. Frozen `[SEG]` hidden max-absolute differences ranged from {diagnosis['material_mismatch']['fixed_sample_result']['hidden_max_abs_diff_range'][0]:.6f} to {diagnosis['material_mismatch']['fixed_sample_result']['hidden_max_abs_diff_range'][1]:.6f}; postprocessed mask-logit max-absolute differences ranged from {diagnosis['material_mismatch']['fixed_sample_result']['mask_logit_max_abs_diff_range'][0]:.6f} to {diagnosis['material_mismatch']['fixed_sample_result']['mask_logit_max_abs_diff_range'][1]:.6f}. This is material for language-conditioned spatial evaluation.

## Populations

Before Audit A completed, read-only manifest preparation recovered 1,000 unique SEEN-TRAIN identities, froze a deterministic 1,000-image TRAIN-HOLDOUT with zero sample/identity overlap, and confirmed the frozen 1,106-image VALIDATION-FAKE population. No checkpoint evaluation was run on these populations after the mismatch was known.

## Downstream analyses

Checkpoint x population metrics, paired bootstrap, loss-vs-metric trajectory, soft-mask diagnostics, subgroup analysis, module-swap attribution and representation-change attribution are all `NOT_RUN_STOPPED_BY_AUDIT_A`. Their output files are explicit stop-status records, not scientific measurements.

## Scientific consequence

The Phase 3D.2 historical validation numbers and step-0 formal selection remain recorded facts. However, because the selector evaluator did not reproduce the training prompt, they cannot establish whether the trained spatial path learned under a matched protocol, failed to fit, overfit, or failed to generalize. Engineering parameter trainability remains established by the prior gradient/update audits; metric-relevant localization learnability/generalization remains unresolved.

This audit does not show that a corrected evaluator would improve Phase 3D.2, does not authorize re-evaluation or checkpoint reselection, and does not support FEPN/NPR/SRM/FOCAL or any architecture conclusion.

## Route decision

Phase 3D.2-A stops here. A separately authorized protocol-resolution stage would be required to define any matched re-evaluation. No code repair, retraining, additional steps, loss/LR/threshold tuning, module training, external baseline work or held-out test evaluation is authorized automatically.
"""
    report_path = OUT / "final_attribution_report.md"
    report_path.write_text(report, encoding="utf-8")
    tracked = [
        OUT / "experiment_manifest.json", OUT / "checkpoint_audit_manifest.json",
        OUT / "train_eval_consistency_audit.json", OUT / "train_eval_consistency_audit.md",
        OUT / "seen_train_manifest.json", OUT / "train_holdout_manifest.json",
        *(OUT / name for name in required), OUT / "loss_metric_trajectory.png",
        OUT / "failure_attribution.json", OUT / "route_gate.json", report_path,
    ]
    completion = {
        "status": "COMPLETE_STOPPED_AFTER_AUDIT_A", "primary_gate": GATE,
        "downstream_attribution_executed": False, "internal_test_used": False, "official1000_used": False,
        "training_or_parameter_update_performed": False,
        "artifacts": [artifact(path) for path in tracked],
    }
    dump(OUT / "completion_manifest.json", completion)
    print(json.dumps({"status": completion["status"], "gate": GATE}, indent=2))


if __name__ == "__main__":
    main()
