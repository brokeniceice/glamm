#!/usr/bin/env python3
"""Freeze the reward-independent Phase 3D.0 sampling protocol from reward-dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tools.phase3d0 import normalize_output, normalized_edit_distance
from tools.phase3b_replay import canonical_json_sha256, file_sha256


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d0_reward_preflight.yaml")
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args(argv)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(groups: list[dict]) -> dict:
    if len(groups) != 1024 or len({row["sample_id"] for row in groups}) != 1024:
        raise RuntimeError(f"sampling setting requires exactly 1024 unique dev groups, got {len(groups)}")
    unique_normalized = []; unique_phrases = []; structural = []; pairwise = []; lengths = []
    for group in groups:
        rollouts = group["rollouts"]
        if len(rollouts) != 8: raise RuntimeError(f"K != 8 for {group['sample_id']}")
        normalized = [normalize_output(row["decoded_text"]) for row in rollouts]
        phrases = [normalize_output(row.get("normalized_phrase")) for row in rollouts]
        unique_normalized.append(len(set(normalized)))
        unique_phrases.append(len({value for value in phrases if value}))
        structural.extend(bool(row["structural_validity"]) for row in rollouts)
        lengths.extend(int(row["generation_length"]) for row in rollouts)
        pairwise.extend(normalized_edit_distance(rollouts[i]["decoded_text"], rollouts[j]["decoded_text"])
                        for i in range(8) for j in range(i + 1, 8))
    return {
        "groups": len(groups), "trajectories": len(groups) * 8,
        "fraction_groups_multiple_unique_normalized": float(np.mean(np.asarray(unique_normalized) >= 2)),
        "mean_unique_normalized_outputs": float(np.mean(unique_normalized)),
        "mean_unique_target_phrases": float(np.mean(unique_phrases)),
        "usable_structural_validity_rate": float(np.mean(structural)),
        "mean_pairwise_normalized_edit_distance": float(np.mean(pairwise)),
        "generation_length_mean": float(np.mean(lengths)),
        "generation_length_p99": float(np.quantile(lengths, .99)),
        "max_length_hit_rate": float(np.mean(np.asarray(lengths) >= 400)),
    }


def main(argv=None):
    cli = args(argv); config_path = (ROOT / cli.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["experiment"]["output_root"]).resolve()
    summaries = {}; inputs = {}
    for setting in ("A", "B"):
        paths = [output / "rollouts" / f"dev_{setting}_text_shard{i:02d}_of_{cli.num_shards:02d}.jsonl"
                 for i in range(cli.num_shards)]
        missing = [str(path) for path in paths if not path.exists()]
        if missing: raise FileNotFoundError(missing)
        groups = [row for path in paths for row in rows(path)]
        summaries[setting] = summarize(groups)
        inputs[setting] = [{"path": str(path), "sha256": file_sha256(path)} for path in paths]
    tolerance = float(cfg["sampling"]["structural_validity_tolerance_absolute"])
    max_length_hit_rate = float(cfg["sampling"]["max_length_hit_rate_max"])
    best_validity = max(value["usable_structural_validity_rate"] for value in summaries.values())
    eligible = [key for key, value in summaries.items()
                if value["usable_structural_validity_rate"] >= best_validity - tolerance
                and value["max_length_hit_rate"] <= max_length_hit_rate]
    if not eligible:
        raise RuntimeError("Both preregistered sampling settings failed generation-stability eligibility")
    selected = max(eligible, key=lambda key: (
        summaries[key]["fraction_groups_multiple_unique_normalized"],
        summaries[key]["mean_unique_normalized_outputs"],
        summaries[key]["mean_unique_target_phrases"],
        summaries[key]["mean_pairwise_normalized_edit_distance"],
        -ord(key),
    ))
    protocol = {
        "status": "FROZEN_BEFORE_FULL_VALIDATION",
        "selection_population": "deterministic internal-train reward-dev 512 Real + 512 Fake",
        "selected_setting": selected,
        "selected_generation_config": {
            **cfg["sampling"]["settings"][selected], "K": 8,
            "max_new_tokens": int(cfg["sampling"]["max_new_tokens"]), "num_beams": 1,
            "same_canonical_prompt": True, "same_special_token_semantics": True,
        },
        "setting_summaries": summaries,
        "selection_rule": {
            "reward_independent": True,
            "mask_iou_used": False, "phrase_reward_used": False,
            "final_candidate_reward_used": False, "validation_performance_used": False,
            "structural_validity_tolerance_absolute": tolerance,
            "max_length_hit_rate_max": max_length_hit_rate,
            "eligible_settings": eligible,
            "priority": ["fraction_groups_multiple_unique_normalized", "mean_unique_normalized_outputs",
                         "mean_unique_target_phrases", "mean_pairwise_normalized_edit_distance", "A_tie_break"],
        },
        "input_artifacts": inputs, "config_sha256": file_sha256(config_path),
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)
    dump(output / "sampling/sampling_protocol.json", protocol)
    dump(output / "sampling_protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
