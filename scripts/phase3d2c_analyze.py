#!/usr/bin/env python3
"""Analyze Phase 3D.2-C populations, soft masks, subgroups, swaps, and route gate."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3d0_analyze import paired

OUT = ROOT / "outputs/phase3d2c_spatial_generalization_attribution"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
P3D2B = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
STEPS = [0, 50, 100, 250]
POPS = ["seen_train", "train_holdout", "validation_fake"]
SOFT_KEYS = [
    "binary_cross_entropy", "soft_dice", "soft_iou", "mean_probability_foreground",
    "mean_probability_background", "foreground_background_probability_margin",
    "mean_logit_foreground", "mean_logit_background",
]
SUBGROUP_MIN_N = 30


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def idx(values):
    result = {row["sample_id"]: row for row in values}
    if len(result) != len(values): raise RuntimeError("duplicate sample IDs")
    return result


def summary(values):
    iou = np.asarray([row["foreground_iou"] for row in values], dtype=float)
    f1 = np.asarray([row["foreground_f1"] for row in values], dtype=float)
    return {
        "n": len(values), "mean_foreground_iou": float(iou.mean()), "median_foreground_iou": float(np.median(iou)),
        "mean_foreground_f1": float(f1.mean()), "median_foreground_f1": float(np.median(f1)),
    }


def paired_rows(left, right, key):
    li, ri = idx(left), idx(right); ids = sorted(set(li) & set(ri))
    if len(ids) != len(li) or len(ids) != len(ri): raise RuntimeError("paired population mismatch")
    return paired([li[sid][key] for sid in ids], [ri[sid][key] for sid in ids])


def soft_paired(left, right, key):
    li, ri = idx(left), idx(right); ids = sorted(li)
    return paired([li[sid]["soft_mask"][key] for sid in ids], [ri[sid]["soft_mask"][key] for sid in ids])


def area_bin(row):
    ratio = float(row["gt_mask_area_ratio"])
    if ratio <= .01: return "<=1%"
    if ratio <= .05: return ">1%-<=5%"
    if ratio <= .20: return ">5%-<=20%"
    return ">20%"


def ref_bin(row):
    count = int(row["num_official_refs"])
    return "1" if count == 1 else ("2" if count == 2 else ">=3")


def difficulty(row):
    value = float(row["foreground_iou"])
    return "hard_<0.10" if value < .10 else ("medium_0.10-<0.50" if value < .50 else "easy_>=0.50")


def main() -> None:
    reproduction = load(OUT / "validation_reproducibility_audit.json")
    if reproduction["status"] != "PASS":
        raise RuntimeError("validation reproduction gate failed")
    predictions = {
        pop: {step: rows(OUT / f"evaluation/{pop}/step_{step:04d}/tf_full_context/predictions.jsonl") for step in STEPS}
        for pop in POPS
    }
    expected = {"seen_train": 1000, "train_holdout": 1000, "validation_fake": 1106}
    for pop in POPS:
        base_ids = [row["sample_id"] for row in predictions[pop][0]]
        if len(base_ids) != expected[pop] or len(set(base_ids)) != expected[pop]: raise RuntimeError(f"{pop} size drift")
        for step in STEPS:
            if [row["sample_id"] for row in predictions[pop][step]] != base_ids: raise RuntimeError(f"{pop}/{step} order drift")
            if not all("soft_mask" in row for row in predictions[pop][step]): raise RuntimeError("soft diagnostics absent")

    population_records = []
    for pop in POPS:
        for step in STEPS:
            population_records.append({"population": pop, "step": step, **summary(predictions[pop][step])})
    metrics = {
        "status": "COMPLETE", "protocol": "canonical_matched_tf_phrase",
        "population_metrics": population_records, "validation_reproducibility": reproduction,
    }
    dump(OUT / "evaluation/population_metrics.json", metrics); dump(OUT / "population_metrics.json", metrics)

    contrasts = {}
    for pop in POPS:
        contrasts[pop] = {}
        for step in STEPS[1:]:
            contrasts[pop][str(step)] = {
                "step": step, "reference_step": 0,
                "foreground_iou": paired_rows(predictions[pop][step], predictions[pop][0], "foreground_iou"),
                "foreground_f1": paired_rows(predictions[pop][step], predictions[pop][0], "foreground_f1"),
            }
    paired_output = {"status": "COMPLETE", "bootstrap_repeats": 10000, "bootstrap_seed": 3407, "populations": contrasts}
    dump(OUT / "statistics/population_paired_bootstrap.json", paired_output)
    dump(OUT / "population_paired_bootstrap.json", paired_output)

    soft = {"status": "COMPLETE", "formal_binary_threshold_unchanged": True, "threshold_search_performed": False,
            "summaries": {}, "step250_minus_step0": {}}
    for pop in POPS:
        soft["summaries"][pop] = {}
        for step in STEPS:
            soft["summaries"][pop][str(step)] = {
                key: {"mean": float(np.mean([row["soft_mask"][key] for row in predictions[pop][step]])),
                      "median": float(np.median([row["soft_mask"][key] for row in predictions[pop][step]]))}
                for key in SOFT_KEYS
            }
        soft["step250_minus_step0"][pop] = {
            key: soft_paired(predictions[pop][250], predictions[pop][0], key) for key in SOFT_KEYS
        }
    dump(OUT / "soft_masks/soft_mask_diagnostics.json", soft); dump(OUT / "soft_mask_diagnostics.json", soft)

    training = rows(P3D2 / "training/metrics.jsonl")
    by_step = {int(row["optimizer_step"]): row for row in training}
    loss_records = []
    metric_lookup = {(row["population"], row["step"]): row for row in population_records}
    for step in STEPS:
        row = {
            "step": step,
            "training_loss": None if step == 0 else {
                "spatial_loss": by_step[step]["spatial_loss"], "mask_bce_loss": by_step[step]["mask_bce_loss"],
                "mask_dice_loss": by_step[step]["mask_dice_loss"],
                "gradient_norm_before_clip": by_step[step].get("gradient_norm_before_clip"),
                "preceding_50_step_mean_spatial_loss": float(np.mean([
                    value["spatial_loss"] for value in training[max(0, step - 50):step]
                ])),
            },
            "population_metrics": {
                pop: {key: metric_lookup[(pop, step)][key] for key in (
                    "mean_foreground_iou", "mean_foreground_f1"
                )} for pop in POPS
            },
        }
        loss_records.append(row)
    first50 = float(np.mean([row["spatial_loss"] for row in training[:50]]))
    last50 = float(np.mean([row["spatial_loss"] for row in training[-50:]]))
    trajectory = {
        "status": "COMPLETE_EXISTING_LOGS_ONLY", "records": loss_records,
        "loss_summary": {"first_50_mean_spatial_loss": first50, "last_50_mean_spatial_loss": last50,
                         "last50_minus_first50": last50 - first50},
        "new_training_performed": False,
    }
    dump(OUT / "loss_analysis/loss_metric_trajectory.json", trajectory); dump(OUT / "loss_metric_trajectory.json", trajectory)
    fig, left = plt.subplots(figsize=(8.2, 4.8))
    steps_all = [row["optimizer_step"] for row in training]
    left.plot(steps_all, [row["spatial_loss"] for row in training], color="#777777", alpha=.45, label="spatial loss")
    left.set_xlabel("optimizer step (existing log)"); left.set_ylabel("training spatial loss")
    right = left.twinx()
    for pop, color in (("seen_train", "#0072B2"), ("train_holdout", "#009E73"), ("validation_fake", "#D55E00")):
        right.plot(STEPS, [metric_lookup[(pop, step)]["mean_foreground_iou"] for step in STEPS],
                   marker="o", color=color, label=f"{pop} FG IoU")
    right.set_ylabel("matched mean FG IoU")
    handles1, labels1 = left.get_legend_handles_labels(); handles2, labels2 = right.get_legend_handles_labels()
    right.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout(); figure = OUT / "figures/loss_metric_trajectory.png"; figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180); plt.close(fig)
    (OUT / "loss_metric_trajectory.png").write_bytes(figure.read_bytes())

    # Predefined subgroup definitions were frozen in source before results: area, ref count, P1 difficulty, source.
    subgroup = {"status": "COMPLETE", "minimum_n_for_stability": SUBGROUP_MIN_N,
                "definitions": ["gt_mask_area_ratio", "official_ref_count", "step0_difficulty", "manifest_source"],
                "populations": {}, "stable_localized_signals": []}
    subgroup_index = defaultdict(dict)
    for pop in POPS:
        base, final = idx(predictions[pop][0]), idx(predictions[pop][250])
        definitions = {
            "gt_mask_area_ratio": lambda row: area_bin(row), "official_ref_count": lambda row: ref_bin(row),
            "step0_difficulty": lambda row: difficulty(row), "manifest_source": lambda row: str(row.get("source")),
        }
        subgroup["populations"][pop] = {}
        for definition, function in definitions.items():
            buckets = defaultdict(list)
            for sid, row in base.items(): buckets[function(row)].append(sid)
            output = []
            for label, ids in sorted(buckets.items()):
                iou = paired([final[sid]["foreground_iou"] for sid in ids], [base[sid]["foreground_iou"] for sid in ids])
                f1 = paired([final[sid]["foreground_f1"] for sid in ids], [base[sid]["foreground_f1"] for sid in ids])
                item = {
                    "label": label, "n": len(ids),
                    "step0_mean_foreground_iou": float(np.mean([base[sid]["foreground_iou"] for sid in ids])),
                    "step250_mean_foreground_iou": float(np.mean([final[sid]["foreground_iou"] for sid in ids])),
                    "foreground_iou": iou, "foreground_f1": f1,
                    "low_n": len(ids) < SUBGROUP_MIN_N,
                }
                output.append(item); subgroup_index[(definition, label)][pop] = item
            subgroup["populations"][pop][definition] = output
    for (definition, label), per_pop in subgroup_index.items():
        supportive = [pop for pop, item in per_pop.items() if item["n"] >= SUBGROUP_MIN_N and
                      item["foreground_iou"]["mean_difference"] > 0 and item["foreground_iou"]["bootstrap_95ci"][0] > 0]
        reproducible = ("seen_train" in supportive and "train_holdout" in supportive) or \
                       ("train_holdout" in supportive and "validation_fake" in supportive)
        if reproducible:
            subgroup["stable_localized_signals"].append({"definition": definition, "label": label, "supportive_populations": supportive})
    dump(OUT / "subgroups/subgroup_attribution.json", subgroup); dump(OUT / "subgroup_attribution.json", subgroup)

    # Module A/D reuse the main step0/step250 evaluations; B/C are diagnostic in-memory hybrids.
    module = {"status": "COMPLETE", "combinations": {}, "interpretation": None}
    combo_sources = {
        "A": {pop: predictions[pop][0] for pop in POPS},
        "D": {pop: predictions[pop][250] for pop in POPS},
        "B": {pop: rows(OUT / f"module_swap/B/{pop}/predictions.jsonl") for pop in POPS},
        "C": {pop: rows(OUT / f"module_swap/C/{pop}/predictions.jsonl") for pop in POPS},
    }
    for combo, per_pop in combo_sources.items():
        module["combinations"][combo] = {}
        for pop, values in per_pop.items():
            record = summary(values)
            record["vs_A"] = {
                "foreground_iou": paired_rows(values, combo_sources["A"][pop], "foreground_iou"),
                "foreground_f1": paired_rows(values, combo_sources["A"][pop], "foreground_f1"),
            }
            module["combinations"][combo][pop] = record
    positive_standalone = []
    for combo in ("B", "C"):
        for pop in POPS:
            contrast = module["combinations"][combo][pop]["vs_A"]["foreground_iou"]
            if contrast["mean_difference"] > 0 and contrast["bootstrap_95ci"][0] > 0:
                positive_standalone.append({"combination": combo, "population": pop, "contrast": contrast})
    module["positive_standalone_signals"] = positive_standalone
    module["invariance_audits"] = {
        combo: load(OUT / f"module_swap/{combo}/invariance_audit.json") for combo in ("B", "C")
    }
    if not positive_standalone:
        module["interpretation"] = "No evidence that either updated component contains a significant standalone FG-IoU gain."
    else:
        module["interpretation"] = "At least one diagnostic hybrid contains a population-specific positive signal; this suggests module-level contribution but is not causal proof."
    dump(OUT / "module_swap/module_swap_attribution.json", module); dump(OUT / "module_swap_attribution.json", module)

    representation = load(OUT / "representation_change_audit.json")
    primary = {pop: contrasts[pop]["250"] for pop in POPS}
    seen_iou, seen_f1 = primary["seen_train"]["foreground_iou"], primary["seen_train"]["foreground_f1"]
    hold_iou = primary["train_holdout"]["foreground_iou"]
    val_iou = primary["validation_fake"]["foreground_iou"]
    seen_positive = seen_iou["mean_difference"] > 0 and seen_iou["bootstrap_95ci"][0] > 0 and \
                    seen_f1["mean_difference"] > 0 and seen_f1["bootstrap_95ci"][0] > 0
    hold_positive = hold_iou["mean_difference"] > 0 and hold_iou["bootstrap_95ci"][0] > 0
    val_positive = val_iou["mean_difference"] > 0 and val_iou["bootstrap_95ci"][0] > 0
    if not seen_positive:
        gate = "GATE_DIRECT_SPATIAL_OPTIMIZATION_NO_EFFECTIVE_FITTING"
    elif not hold_positive and not val_positive:
        gate = "GATE_SPATIAL_OPTIMIZATION_OVERFITS"
    elif hold_positive and not val_positive:
        gate = "GATE_TRAIN_VALIDATION_GENERALIZATION_GAP"
    else:
        gate = "GATE_LOCALIZED_SPATIAL_SIGNAL_DETECTED" if subgroup["stable_localized_signals"] else "GATE_SPATIAL_OPTIMIZATION_OVERFITS"
    soft_seen = soft["step250_minus_step0"]["seen_train"]
    soft_positive = (
        soft_seen["soft_iou"]["mean_difference"] > 0 and soft_seen["soft_iou"]["bootstrap_95ci"][0] > 0
    ) or (
        soft_seen["soft_dice"]["mean_difference"] > 0 and soft_seen["soft_dice"]["bootstrap_95ci"][0] > 0
    ) or (
        soft_seen["binary_cross_entropy"]["mean_difference"] < 0 and soft_seen["binary_cross_entropy"]["bootstrap_95ci"][1] < 0
    )
    route_closed = gate == "GATE_DIRECT_SPATIAL_OPTIMIZATION_NO_EFFECTIVE_FITTING"
    failure = {
        "status": "COMPLETE", "primary_pattern": gate,
        "step250_minus_step0": primary, "soft_seen_fitting_signal": soft_positive,
        "stable_localized_signals": subgroup["stable_localized_signals"],
        "positive_module_swap_signals": positive_standalone,
        "engineering_trainability": "SUPPORTED_BY_PHASE3D2_PARAMETER_AND_GRADIENT_AUDITS",
        "metric_relevant_fitting": "SUPPORTED" if seen_positive else "UNSUPPORTED",
    }
    dump(OUT / "failure_attribution.json", failure)
    route = {
        "status": "COMPLETE_STOPPED", "primary_gate": gate,
        "secondary_localized_signal": bool(subgroup["stable_localized_signals"]),
        "soft_mask_signal_on_seen": soft_positive, "standalone_module_signal": bool(positive_standalone),
        "existing_direct_spatial_path_route_closed": route_closed,
        "representation_or_architecture_investigation_authorized": False,
        "sufficient_evidence_to_claim_new_architecture_required": False,
        "next_route_recommendation": (
            "Discuss representation sufficiency/objective-interface mismatch and paper convergence; no automatic FEPN."
            if route_closed else "Use the identified generalization pattern to propose a separately authorized diagnostic."
        ),
        "automatic_next_phase_started": False, "internal_test_used": False, "official1000_used": False,
    }
    dump(OUT / "route_gate.json", route)
    print(json.dumps({"status": "COMPLETE", "gate": gate, "route_closed": route_closed}, indent=2))


if __name__ == "__main__":
    main()
