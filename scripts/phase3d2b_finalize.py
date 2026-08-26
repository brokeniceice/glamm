#!/usr/bin/env python3
"""Finalize matched validation statistics and route gate for Phase 3D.2-B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256
from scripts.phase3d0_analyze import paired

OUT = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
P3D2A = ROOT / "outputs/phase3d2a_spatial_attribution_audit"
STEPS = [0, 50, 100, 150, 200, 250]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def index(values: list[dict]) -> dict[str, dict]:
    result = {row["sample_id"]: row for row in values}
    if len(result) != len(values):
        raise RuntimeError("duplicate sample id in prediction file")
    return result


def metric_summary(values: list[dict]) -> dict:
    iou = np.asarray([float(row["foreground_iou"]) for row in values])
    f1 = np.asarray([float(row["foreground_f1"]) for row in values])
    return {
        "n": len(values), "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)), "mean_foreground_f1": float(f1.mean()),
        "median_foreground_f1": float(np.median(f1)), "mask_logit_threshold": 0.0,
    }


def main() -> None:
    consistency = load(OUT / "matched_protocol_consistency_audit.json")
    if consistency["status"] != "PASS":
        raise RuntimeError("cannot finalize full evaluation after failed consistency audit")
    legacy_selector = load(P3D2 / "evaluation/selector/selector.json")
    matched_rows = {}
    for step in STEPS:
        path = OUT / f"evaluation/matched_tf_phrase/step_{step:04d}/tf_full_context/predictions.jsonl"
        values = rows(path)
        if len(values) != 1106:
            raise RuntimeError(f"step {step} prediction count drift: {len(values)}")
        matched_rows[step] = values
    ids = sorted(index(matched_rows[0]))
    if len(ids) != 1106 or any(sorted(index(matched_rows[step])) != ids for step in STEPS):
        raise RuntimeError("matched paired population mismatch")
    metrics = {step: metric_summary(matched_rows[step]) for step in STEPS}
    base = metrics[0]
    records = []
    for step in STEPS:
        row = {"candidate_step": step, **metrics[step]}
        row["delta_mean_foreground_iou_vs_step0"] = row["mean_foreground_iou"] - base["mean_foreground_iou"]
        row["delta_mean_foreground_f1_vs_step0"] = row["mean_foreground_f1"] - base["mean_foreground_f1"]
        records.append(row)
    matched_metrics = {
        "status": "COMPLETE_MATCHED_VALIDATION", "protocol": "phase3d2_training_matched_tf_phrase_v1",
        "population": "frozen_internal_validation_fake", "population_count": 1106,
        "candidate_metrics": records,
        "historical_legacy_candidate_metrics": legacy_selector["candidates"],
        "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "evaluation/matched_validation_metrics.json", matched_metrics)
    dump(OUT / "matched_validation_metrics.json", matched_metrics)

    base_idx = index(matched_rows[0])
    contrasts = {}
    for step in STEPS[1:]:
        current = index(matched_rows[step])
        contrasts[str(step)] = {
            "candidate_step": step, "reference_step": 0,
            "foreground_iou": paired(
                [current[sid]["foreground_iou"] for sid in ids],
                [base_idx[sid]["foreground_iou"] for sid in ids],
            ),
            "foreground_f1": paired(
                [current[sid]["foreground_f1"] for sid in ids],
                [base_idx[sid]["foreground_f1"] for sid in ids],
            ),
        }
    paired_output = {
        "status": "COMPLETE", "population_count": len(ids), "bootstrap_repeats": 10000,
        "bootstrap_seed": 3407, "contrasts": contrasts,
    }
    dump(OUT / "statistics/paired_bootstrap.json", paired_output)
    dump(OUT / "paired_bootstrap.json", paired_output)

    selected = max(records, key=lambda row: (
        row["mean_foreground_iou"], row["mean_foreground_f1"], -row["candidate_step"]
    ))
    selected_step = int(selected["candidate_step"])
    selector = {
        "status": "MATCHED_PROTOCOL_RETROSPECTIVE_ONLY", "selector": "matched TF-PHRASE mean FG IoU",
        "tie_breaker": "matched TF-PHRASE mean FG F1", "candidate_steps": STEPS,
        "historical_selected_step": 0, "matched_retrospective_selected_step": selected_step,
        "whether_selection_changes": selected_step != 0, "selected_metrics": selected,
        "checkpoint_promotion_authorized": False, "G0_used": False, "training_loss_used": False,
        "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "matched_retrospective_selector.json", selector)

    comparisons = {}
    for step in (0, 250):
        legacy = rows(P3D2 / f"evaluation/tf_phrase/step_{step:04d}/tf_full_context/predictions.jsonl")
        legacy_idx, matched_idx = index(legacy), index(matched_rows[step])
        if sorted(legacy_idx) != ids:
            raise RuntimeError(f"legacy/matched population mismatch at step {step}")
        comparisons[str(step)] = {
            "candidate_step": step, "legacy_metrics": metric_summary(legacy),
            "matched_metrics": metrics[step],
            "matched_minus_legacy": {
                "foreground_iou": paired(
                    [matched_idx[sid]["foreground_iou"] for sid in ids],
                    [legacy_idx[sid]["foreground_iou"] for sid in ids],
                ),
                "foreground_f1": paired(
                    [matched_idx[sid]["foreground_f1"] for sid in ids],
                    [legacy_idx[sid]["foreground_f1"] for sid in ids],
                ),
            },
        }
    audit_a = load(P3D2A / "train_eval_consistency_audit.json")
    fixed_records = audit_a["sample_records"]
    protocol_comparison = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY", "interpretation": "same model, different evaluation context; not model improvement",
        "validation_comparisons": comparisons,
        "fixed_8_sample_legacy_vs_training_context": {
            "source": str((P3D2A / "train_eval_consistency_audit.json").resolve()),
            "seg_hidden_max_abs_diff": {
                "min": min(row["frozen_seg_hidden"]["max_abs_diff"] for row in fixed_records),
                "mean": float(np.mean([row["frozen_seg_hidden"]["max_abs_diff"] for row in fixed_records])),
                "max": max(row["frozen_seg_hidden"]["max_abs_diff"] for row in fixed_records),
            },
            "postprocessed_mask_logit_max_abs_diff": {
                "min": min(row["postprocessed_mask_logits"]["max_abs_diff"] for row in fixed_records),
                "mean": float(np.mean([row["postprocessed_mask_logits"]["max_abs_diff"] for row in fixed_records])),
                "max": max(row["postprocessed_mask_logits"]["max_abs_diff"] for row in fixed_records),
            },
        },
        "full_validation_hidden_or_logit_tensors_logged": False,
    }
    dump(OUT / "protocol_comparison/legacy_matched_protocol_comparison.json", protocol_comparison)
    dump(OUT / "legacy_matched_protocol_comparison.json", protocol_comparison)

    selected_contrast = contrasts.get(str(selected_step))
    significant_positive = selected_step > 0 and selected_contrast["foreground_iou"]["mean_difference"] > 0 \
        and selected_contrast["foreground_iou"]["bootstrap_95ci"][0] > 0
    any_positive = any(contrasts[str(step)]["foreground_iou"]["mean_difference"] > 0 for step in STEPS[1:])
    best_trained = max(records[1:], key=lambda row: (
        row["mean_foreground_iou"], row["mean_foreground_f1"], -row["candidate_step"]
    ))
    best_trained_ci = contrasts[str(best_trained["candidate_step"])]["foreground_iou"]["bootstrap_95ci"]
    if significant_positive:
        gate = "GATE_MATCHED_SPATIAL_OPTIMIZATION_EFFECTIVE"
        interpretation = "overturned"
        next_question = "Does matched-context spatial improvement transfer to autonomous G0 inference?"
    elif any_positive:
        gate = "GATE_MATCHED_SPATIAL_SIGNAL_INCONCLUSIVE"
        interpretation = "inconclusive"
        next_question = "Seen-train/holdout/loss trajectory attribution requires separate authorization."
    elif best_trained_ci[1] < 0:
        gate = "GATE_MATCHED_SPATIAL_OPTIMIZATION_HARMFUL"
        interpretation = "supported_and_strengthened_as_harmful"
        next_question = "No optimization expansion is authorized."
    else:
        gate = "GATE_MATCHED_SPATIAL_OPTIMIZATION_NO_VALIDATION_GAIN"
        interpretation = "supported"
        next_question = "Phase 3D.2-A downstream attribution may be proposed but is not automatically authorized."
    route = {
        "status": "COMPLETE_STOPPED", "primary_gate": gate,
        "phase3d2_negative_interpretation": interpretation,
        "historical_selected_step": 0, "matched_retrospective_selected_step": selected_step,
        "whether_selection_changes": selected_step != 0,
        "next_scientific_question": next_question,
        "next_phase_authorized": False, "automatic_G0_started": False,
        "checkpoint_promotion_authorized": False, "training_authorized": False,
        "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "route_gate.json", route)

    table = "\n".join(
        f"| {row['candidate_step']} | {row['mean_foreground_iou']:.6f} | {row['delta_mean_foreground_iou_vs_step0']:+.6f} | "
        f"{row['mean_foreground_f1']:.6f} | {row['delta_mean_foreground_f1_vs_step0']:+.6f} |"
        for row in records
    )
    ci_table = "\n".join(
        f"| {step} | {contrasts[str(step)]['foreground_iou']['mean_difference']:+.6f} | "
        f"[{contrasts[str(step)]['foreground_iou']['bootstrap_95ci'][0]:+.6f}, {contrasts[str(step)]['foreground_iou']['bootstrap_95ci'][1]:+.6f}] | "
        f"{contrasts[str(step)]['foreground_iou']['wins']}/{contrasts[str(step)]['foreground_iou']['ties']}/{contrasts[str(step)]['foreground_iou']['losses']} | "
        f"{contrasts[str(step)]['foreground_f1']['mean_difference']:+.6f} | "
        f"[{contrasts[str(step)]['foreground_f1']['bootstrap_95ci'][0]:+.6f}, {contrasts[str(step)]['foreground_f1']['bootstrap_95ci'][1]:+.6f}] |"
        for step in STEPS[1:]
    )
    legacy_table = "\n".join(
        f"| {step} | {comparisons[str(step)]['legacy_metrics']['mean_foreground_iou']:.6f} | "
        f"{comparisons[str(step)]['matched_metrics']['mean_foreground_iou']:.6f} | "
        f"{comparisons[str(step)]['matched_minus_legacy']['foreground_iou']['mean_difference']:+.6f} | "
        f"{comparisons[str(step)]['legacy_metrics']['mean_foreground_f1']:.6f} | "
        f"{comparisons[str(step)]['matched_metrics']['mean_foreground_f1']:.6f} | "
        f"{comparisons[str(step)]['matched_minus_legacy']['foreground_f1']['mean_difference']:+.6f} |"
        for step in (0, 250)
    )
    historical_table = "\n".join(
        f"| {row['optimizer_step']} | {row['mean_foreground_iou']:.6f} | {row['mean_foreground_f1']:.6f} |"
        for row in legacy_selector["candidates"]
    )
    report = f"""# Phase 3D.2-B — Matched Spatial-Path Re-evaluation

## Final status and boundary

Read-only matched-protocol retrospective evaluation completed and stopped. All inference used `model.eval()` and `torch.no_grad()`; no training, checkpoint modification, threshold tuning, G0, internal test, official1000, seen/holdout, subgroup, soft-mask or module-swap work was performed.

## Consistency hard gate

`PASS`: 8/8 fixed samples had exact raw token sequences, `[SEG]` count/index, GT masks, `[SEG]` hidden, 256D projection, native mask logits and postprocessed mask logits. Every reported numerical difference was exactly 0.

## Matched validation metrics

| Step | Matched TF FG IoU | Delta vs step0 | Matched TF FG F1 | Delta vs step0 |
|---:|---:|---:|---:|---:|
{table}

Medians are retained in `matched_validation_metrics.json`.

## Historical legacy evaluator table (preserved)

| Step | Legacy TF FG IoU | Legacy TF FG F1 |
|---:|---:|---:|
{historical_table}

## Paired statistics versus step 0

| Step | IoU delta | IoU 95% CI | W/T/L | F1 delta | F1 95% CI |
|---:|---:|---:|---:|---:|---:|
{ci_table}

## Historical and retrospective selectors

The historical legacy-mismatched selector remains step 0. The matched-protocol retrospective selector is step {selected_step}; selection change = `{str(selected_step != 0).lower()}`. This is retrospective-only and does not correct or overwrite the original selector and does not promote a checkpoint.

## Legacy versus matched evaluator

| Step | Legacy IoU | Matched IoU | Matched-Legacy IoU | Legacy F1 | Matched F1 | Matched-Legacy F1 |
|---:|---:|---:|---:|---:|---:|---:|
{legacy_table}

These are same-model context effects, not model improvements. They support sensitivity to the full conversational trajectory but are not a strict causal law.

## Route decision

Primary gate: **`{gate}`**. Phase 3D.2 negative interpretation status: **`{interpretation}`**.

No next phase is automatically authorized. In particular, no G0, checkpoint promotion, test evaluation, additional training, attribution audit or architecture work may start without a new explicit instruction.
"""
    report_path = OUT / "final_matched_reevaluation.md"
    report_path.write_text(report, encoding="utf-8")
    experiment = load(OUT / "experiment_manifest.json")
    experiment.update({
        "status": "COMPLETE_STOPPED", "primary_gate": gate,
        "matched_retrospective_selected_step": selected_step,
        "training_or_parameter_update_performed": False,
    })
    dump(OUT / "experiment_manifest.json", experiment)
    dump(OUT / "manifests/experiment_manifest.json", experiment)
    required = [
        OUT / "experiment_manifest.json", OUT / "checkpoint_manifest.json", OUT / "matched_protocol_spec.json",
        OUT / "matched_protocol_consistency_audit.json", OUT / "matched_validation_metrics.json",
        OUT / "paired_bootstrap.json", OUT / "matched_retrospective_selector.json",
        OUT / "legacy_matched_protocol_comparison.json", OUT / "route_gate.json", report_path,
    ]
    completion = {
        "status": "COMPLETE_STOPPED", "primary_gate": gate,
        "historical_selected_step": 0, "matched_retrospective_selected_step": selected_step,
        "training_or_parameter_update_performed": False, "G0_used": False,
        "internal_test_used": False, "official1000_used": False,
        "artifacts": [artifact(path) for path in required],
    }
    dump(OUT / "completion_manifest.json", completion)
    print(json.dumps({"status": completion["status"], "gate": gate, "matched_selected_step": selected_step}, indent=2))


if __name__ == "__main__":
    main()
