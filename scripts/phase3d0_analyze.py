#!/usr/bin/env python3
"""Preregistered confirmation analysis and route gate for Phase 3D.0."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tools.phase3d0 import (
    COMPONENT_NAMES, REWARD_NAMES, choose_top, group_diversity, pareto_statistics,
)
from tools.phase3b_replay import file_sha256


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d0_reward_preflight.yaml")
    parser.add_argument("--setting", choices=("A", "B"), help="must match frozen sampling protocol")
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paired(left, right, *, seed=3407, repeats=10000) -> dict:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float); delta = a - b
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(0, repeats, 500):
        idx = rng.integers(0, len(delta), size=(min(500, repeats - len(chunks)*500), len(delta)))
        chunks.append(delta[idx].mean(axis=1))
    boot = np.concatenate(chunks)
    nonzero = delta[np.abs(delta) > 1e-12]
    try: wilcoxon = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    except ValueError: wilcoxon = None
    return {
        "n": len(delta), "mean_difference": float(delta.mean()), "median_difference": float(np.median(delta)),
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "bootstrap_95ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()), "wilcoxon_pvalue": wilcoxon,
    }


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def rate(rows, predicate): return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def trajectory_summary(rows: list[dict]) -> dict:
    fake = [row for row in rows if int(row["gt_class"]) == 1]; real = [row for row in rows if int(row["gt_class"]) == 0]
    return {
        "count": len(rows), "overall_generated_verdict_accuracy": mean(rows, "R_cls"),
        "real_generated_verdict_accuracy": mean(real, "R_cls"), "fake_generated_verdict_accuracy": mean(fake, "R_cls"),
        "structural_validity_rate": mean(rows, "R_struct"),
        "fake_structural_validity_rate": mean(fake, "R_struct"),
        "real_structural_validity_rate": mean(real, "R_struct"),
        "fake_phrase_token_f1": mean(fake, "R_phrase_lex"), "fake_foreground_iou": mean(fake, "R_mask"),
        "fake_foreground_f1": mean(fake, "foreground_f1"), "fake_fg_bg_miou": mean(fake, "fg_bg_miou"),
        "fake_align": mean(fake, "R_align"),
        "real_fake_structure_hallucination_rate": rate(real, lambda row: row["verdict"] == "FAKE" or row["target_field_present"] or row["usable_seg"]),
        "seg_missing_rate": rate(rows, lambda row: not row["usable_seg"]),
        "target_region_missing_rate": rate(rows, lambda row: not row["target_field_present"]),
    }


def add_system_metrics(summary: dict, rows: list[dict], greedy_by_id: dict[str, dict]) -> None:
    cls_correct = [int(greedy_by_id[row["sample_id"]]["cls_pred"]) == int(row["gt_class"]) for row in rows]
    real = [row for row in rows if int(row["gt_class"]) == 0]
    fake = [row for row in rows if int(row["gt_class"]) == 1]
    summary.update({
        "frozen_cls_accuracy": float(np.mean(cls_correct)),
        "frozen_cls_real_accuracy": rate(real, lambda row: int(greedy_by_id[row["sample_id"]]["cls_pred"]) == 0),
        "frozen_cls_fake_accuracy": rate(fake, lambda row: int(greedy_by_id[row["sample_id"]]["cls_pred"]) == 1),
        "generated_lm_fake_recall": mean(fake, "R_cls"),
        "generated_lm_real_fpr": rate(real, lambda row: row["verdict"] == "FAKE"),
        "frozen_cls_lm_agreement": rate(rows, lambda row: (
            (row["verdict"] == "FAKE" and int(greedy_by_id[row["sample_id"]]["cls_pred"]) == 1)
            or (row["verdict"] == "REAL" and int(greedy_by_id[row["sample_id"]]["cls_pred"]) == 0)
        )),
        "cls_head_is_frozen_diagnostic_not_R_cls": True,
    })


def correlations(rows: list[dict], reward: str) -> dict:
    targets = list(COMPONENT_NAMES) + ["generation_length", "length_normalized_log_probability"]
    result = {}
    for target in targets:
        pairs = [(float(row[reward]), float(row[target])) for row in rows if row.get(target) is not None]
        if len(pairs) < 3:
            result[target] = {"n": len(pairs), "pearson": None, "spearman": None}; continue
        x, y = map(np.asarray, zip(*pairs))
        if np.std(x) == 0 or np.std(y) == 0:
            pearson = spearman = None
        else:
            pearson = float(stats.pearsonr(x, y).statistic); spearman = float(stats.spearmanr(x, y).statistic)
        result[target] = {"n": len(pairs), "pearson": pearson, "spearman": spearman}
    return result


def main(argv=None):
    cli = args(argv); cfg = yaml.safe_load((ROOT / cli.config).read_text(encoding="utf-8"))
    output = (ROOT / cfg["experiment"]["output_root"]).resolve()
    protocol = json.loads((output / "sampling_protocol.json").read_text(encoding="utf-8"))
    setting = cli.setting or protocol["selected_setting"]
    if setting != protocol["selected_setting"] or protocol["status"] != "FROZEN_BEFORE_FULL_VALIDATION":
        raise RuntimeError("confirmation setting does not match frozen sampling protocol")
    reward_definition = json.loads((output / "reward_definition.json").read_text(encoding="utf-8"))
    if reward_definition["status"] != "FROZEN_BEFORE_FULL_VALIDATION": raise RuntimeError("reward not frozen")
    rollout_paths = [output / "rollouts" / f"val_{setting}_full_shard{i:02d}_of_{cli.num_shards:02d}.jsonl" for i in range(cli.num_shards)]
    greedy_paths = [output / "greedy" / f"val_shard{i:02d}_of_{cli.num_shards:02d}.jsonl" for i in range(cli.num_shards)]
    for path in rollout_paths + greedy_paths:
        if not path.exists(): raise FileNotFoundError(path)
    groups = [row for path in rollout_paths for row in load_jsonl(path)]
    greedy = [row for path in greedy_paths for row in load_jsonl(path)]
    groups.sort(key=lambda row: row["sample_id"]); greedy.sort(key=lambda row: row["sample_id"])
    if len(groups) != 2212 or len(greedy) != 2212 or len({r["sample_id"] for r in groups}) != 2212:
        raise RuntimeError(f"full validation incomplete groups={len(groups)} greedy={len(greedy)}")
    greedy_by_id = {row["sample_id"]: row for row in greedy}
    if set(greedy_by_id) != {row["sample_id"] for row in groups}: raise RuntimeError("greedy/rollout ID mismatch")
    if any(len(group["rollouts"]) != 8 for group in groups): raise RuntimeError("K != 8")
    fake_groups = [group["rollouts"] for group in groups if int(group["rollouts"][0]["gt_class"]) == 1]
    all_rollouts = [row for group in groups for row in group["rollouts"]]

    group_stats = []
    for group in groups:
        value = group_diversity(group["rollouts"]); value["sample_id"] = group["sample_id"]
        value["gt_class"] = int(group["rollouts"][0]["gt_class"]); group_stats.append(value)
    with (output / "diversity/group_diversity.jsonl").open("w", encoding="utf-8") as handle:
        for row in group_stats: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    diversity = {
        "groups": len(group_stats),
        "fraction_groups_multiple_unique": rate(group_stats, lambda row: row["number_unique_normalized_outputs"] >= 2),
        "fraction_groups_mask_range_gt_0.10": rate([r for r in group_stats if r["gt_class"] == 1], lambda row: row["mask_iou_range"] > .10),
        "fraction_groups_phrase_range_gt_0.10": rate([r for r in group_stats if r["gt_class"] == 1], lambda row: row["phrase_f1_range"] > .10),
        "per_reward": {},
    }
    for reward in REWARD_NAMES:
        diversity["per_reward"][reward] = {
            "fraction_groups_reward_std_gt_0": rate(group_stats, lambda row, r=reward: row[f"{r}_std"] > 0),
            "fraction_fake_groups_reward_std_gt_0.05": rate([x for x in group_stats if x["gt_class"] == 1], lambda row, r=reward: row[f"{r}_std"] > .05),
            "mean_group_reward_std": mean(group_stats, f"{reward}_std"),
        }
    dump(output / "diversity/rollout_diversity.json", diversity); dump(output / "rollout_diversity.json", diversity)

    pareto = {reward: pareto_statistics(fake_groups, reward) for reward in REWARD_NAMES}
    dump(output / "ranking/pareto_ranking_statistics.json", pareto); dump(output / "pareto_ranking_statistics.json", pareto)
    greedy_summary = trajectory_summary(greedy); add_system_metrics(greedy_summary, greedy, greedy_by_id)
    dump(output / "greedy/greedy_metrics.json", greedy_summary); dump(output / "greedy_metrics.json", greedy_summary)

    top_rows = {}; top_statistics = {}; paired_statistics = {}; catastrophic = {}
    for reward in REWARD_NAMES:
        selected = [dict(choose_top(group["rollouts"], reward), selected_reward=reward) for group in groups]
        top_rows[reward] = selected; top_statistics[reward] = trajectory_summary(selected)
        add_system_metrics(top_statistics[reward], selected, greedy_by_id)
        metrics = {}
        for key in ("R_cls", "R_struct", "R_phrase_lex", "R_mask", "foreground_f1", "fg_bg_miou"):
            pairs = [(row[key], greedy_by_id[row["sample_id"]][key]) for row in selected if row.get(key) is not None and greedy_by_id[row["sample_id"]].get(key) is not None]
            metrics[key] = paired([a for a, _ in pairs], [b for _, b in pairs])
        paired_statistics[reward] = metrics
        fake_selected = [row for row in selected if row["gt_class"] == 1]
        top_mask_bad = sum(row["R_mask"] < .10 and max(x["R_mask"] for x in next(g["rollouts"] for g in groups if g["sample_id"] == row["sample_id"])) >= .30 for row in fake_selected)
        top_phrase_bad = sum(row["R_phrase_lex"] < .10 and max(x["R_phrase_lex"] for x in next(g["rollouts"] for g in groups if g["sample_id"] == row["sample_id"])) >= .30 for row in fake_selected)
        catastrophic[reward] = {
            "wrong_class_rate": rate(selected, lambda row: row["R_cls"] == 0),
            "invalid_structure_rate": rate(selected, lambda row: row["R_struct"] == 0),
            "seg_missing_rate": rate(selected, lambda row: not row["usable_seg"]),
            "target_region_missing_rate": rate(selected, lambda row: not row["target_field_present"]),
            "TOP_REWARD_MASK_BAD_count": top_mask_bad, "TOP_REWARD_MASK_BAD_rate_fake": top_mask_bad / len(fake_selected),
            "TOP_REWARD_PHRASE_BAD_count": top_phrase_bad, "TOP_REWARD_PHRASE_BAD_rate_fake": top_phrase_bad / len(fake_selected),
        }
        path = output / "reward_candidates" / f"top_selected_{reward}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump(output / "statistics/top_selection_statistics.json", top_statistics); dump(output / "top_selection_statistics.json", top_statistics)
    dump(output / "statistics/paired_top_vs_greedy.json", paired_statistics)
    dump(output / "statistics/catastrophic_reward_audit.json", catastrophic); dump(output / "catastrophic_reward_audit.json", catastrophic)

    nonreg = {}; readiness = {}
    min_unique = float(cfg["gate"]["fraction_groups_multiple_unique_min"])
    min_std = float(cfg["gate"]["fraction_fake_selected_reward_std_gt_005_min"])
    min_pareto = float(cfg["gate"]["pareto_correct_ranking_min"])
    max_cls = float(cfg["gate"]["generated_verdict_regression_max"]); max_struct = float(cfg["gate"]["structural_validity_regression_max"])
    collapse_min = float(cfg["gate"]["collapse_mean_regression_min"])
    for reward in REWARD_NAMES:
        cls_delta = top_statistics[reward]["overall_generated_verdict_accuracy"] - greedy_summary["overall_generated_verdict_accuracy"]
        struct_delta = top_statistics[reward]["structural_validity_rate"] - greedy_summary["structural_validity_rate"]
        phrase_pair = paired_statistics[reward]["R_phrase_lex"]; mask_pair = paired_statistics[reward]["R_mask"]
        phrase_collapse = phrase_pair["mean_difference"] < collapse_min and phrase_pair["bootstrap_95ci"][1] < 0
        mask_collapse = mask_pair["mean_difference"] < collapse_min and mask_pair["bootstrap_95ci"][1] < 0
        nonreg[reward] = {
            "generated_verdict_accuracy_delta": cls_delta,
            "classification_non_regression_pass": cls_delta >= -max_cls,
            "structural_validity_delta": struct_delta,
            "structure_non_regression_pass": struct_delta >= -max_struct,
            "frozen_cls_head_unchanged_by_selection": True,
            "frozen_cls_head_accuracy": rate(greedy, lambda row: int(row["cls_pred"]) == int(row["gt_class"])),
        }
        conditions = {
            "groups_multiple_unique": diversity["fraction_groups_multiple_unique"] >= min_unique,
            "fake_reward_std": diversity["per_reward"][reward]["fraction_fake_groups_reward_std_gt_0.05"] >= min_std,
            "pareto_ranking": (pareto[reward]["correct_ranking_rate"] or 0) >= min_pareto,
            "classification_non_regression": nonreg[reward]["classification_non_regression_pass"],
            "structure_non_regression": nonreg[reward]["structure_non_regression_pass"],
            "no_phrase_collapse": not phrase_collapse, "no_mask_collapse": not mask_collapse,
        }
        readiness[reward] = {"conditions": conditions, "GRPO_SIGNAL_READY": all(conditions.values())}
    dump(output / "statistics/real_fake_non_regression.json", nonreg); dump(output / "real_fake_non_regression.json", nonreg)
    dump(output / "statistics/grpo_signal_readiness.json", readiness); dump(output / "grpo_signal_readiness.json", readiness)

    oracle = {"fake_groups": len(fake_groups), "max_mask_mean": float(np.mean([max(r["R_mask"] for r in group) for group in fake_groups])),
              "max_phrase_mean": float(np.mean([max(r["R_phrase_lex"] for r in group) for group in fake_groups]))}
    conjunctive = 0; mask_better = phrase_better = 0
    for group in fake_groups:
        base = greedy_by_id[group[0]["sample_id"]]
        mask_better += any(row["R_mask"] > base["R_mask"] for row in group)
        phrase_better += any(row["R_phrase_lex"] > base["R_phrase_lex"] for row in group)
        conjunctive += any(row["R_cls"] == 1 and row["R_struct"] == 1 and row["R_phrase_lex"] > base["R_phrase_lex"] and row["R_mask"] > base["R_mask"] for row in group)
    oracle.update({"fraction_groups_mask_better_than_greedy_available": mask_better/len(fake_groups),
                   "fraction_groups_phrase_better_than_greedy_available": phrase_better/len(fake_groups),
                   "fraction_groups_jointly_class_structure_phrase_mask_better_candidate_available": conjunctive/len(fake_groups),
                   "interpretation": "rollout-distribution availability upper bound; not policy performance"})
    dump(output / "statistics/oracle_availability_upper_bound.json", oracle)
    component_stats = {key: {"mean": mean(all_rollouts, key), "min": min(float(r[key]) for r in all_rollouts), "max": max(float(r[key]) for r in all_rollouts)} for key in COMPONENT_NAMES}
    dump(output / "reward_components/reward_component_statistics.json", component_stats); dump(output / "reward_component_statistics.json", component_stats)
    corr = {reward: correlations(all_rollouts, reward) for reward in REWARD_NAMES}; dump(output / "statistics/reward_correlations.json", corr)

    supported = [reward for reward in REWARD_NAMES if readiness[reward]["GRPO_SIGNAL_READY"]]
    if supported:
        best_pareto = max(pareto[r]["correct_ranking_rate"] for r in supported)
        candidates = [r for r in supported if pareto[r]["correct_ranking_rate"] == best_pareto]
        candidates.sort(key=lambda r: (catastrophic[r]["TOP_REWARD_MASK_BAD_rate_fake"] + catastrophic[r]["TOP_REWARD_PHRASE_BAD_rate_fake"],
                                       -top_statistics[r]["fake_align"], -top_statistics[r]["fake_foreground_iou"], r))
        selected_reward = candidates[0]
    else: selected_reward = None
    stable_positive = False if selected_reward is None else (
        paired_statistics[selected_reward]["R_phrase_lex"]["bootstrap_95ci"][0] > 0
        or paired_statistics[selected_reward]["R_mask"]["bootstrap_95ci"][0] > 0
    )
    mask_tolerance = float(cfg["gate"]["mask_only_not_worse_tolerance"])
    evidence = [r for r in ("R2", "R3") if readiness[r]["GRPO_SIGNAL_READY"]]
    mask_only_sufficient = bool(readiness["R0"]["GRPO_SIGNAL_READY"] and evidence and
        pareto["R0"]["correct_ranking_rate"] >= max(pareto[r]["correct_ranking_rate"] for r in evidence) - mask_tolerance and
        top_statistics["R0"]["fake_align"] >= max(top_statistics[r]["fake_align"] for r in evidence) - mask_tolerance)
    if diversity["fraction_groups_multiple_unique"] < min_unique or max(diversity["per_reward"][r]["fraction_fake_groups_reward_std_gt_0.05"] for r in REWARD_NAMES) < min_std:
        gate = "GATE_ROLLOUT_DIVERSITY_INSUFFICIENT"
    elif mask_only_sufficient and selected_reward == "R0": gate = "GATE_MASK_ONLY_REWARD_SUFFICIENT"
    elif selected_reward in {"R2", "R3"} and stable_positive: gate = "GATE_EVIDENCE_REWARD_PREFLIGHT_SUPPORTED"
    elif selected_reward is None: gate = "GATE_REWARD_PREFLIGHT_NOT_SUPPORTED"
    else: gate = "GATE_INCONCLUSIVE"
    route = {
        "primary_gate": gate, "selected_reward_candidate": selected_reward,
        "GRPO_SIGNAL_READY": False if selected_reward is None else readiness[selected_reward]["GRPO_SIGNAL_READY"],
        "stable_positive_phrase_or_mask_evidence": stable_positive,
        "mask_only_sufficient": mask_only_sufficient,
        "phase3d1_authorized_for_controlled_experiment": gate == "GATE_EVIDENCE_REWARD_PREFLIGHT_SUPPORTED",
        "phase3d1_started": False,
        "best_of_k_is_not_deployment_result": True,
        "lexical_phrase_reward_semantic_validity_remains_limited_until_human_audit": True,
        "input_artifacts": [{"path": str(p), "sha256": file_sha256(p)} for p in rollout_paths + greedy_paths],
    }
    dump(output / "route_gate.json", route)
    dump(output / "rollout_manifest.json", {"status": "COMPLETE", "setting": setting, "K": 8,
         "groups": len(groups), "trajectories": len(all_rollouts), "rollout_paths": [str(p) for p in rollout_paths],
         "greedy_paths": [str(p) for p in greedy_paths]})
    print(json.dumps(route, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
