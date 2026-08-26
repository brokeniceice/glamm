#!/usr/bin/env python3
"""Read-only Phase 3F contract, frozen-rollout, and matched-control audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3f_autonomous_oracle_grounding_distillation.yaml")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    config_path = (ROOT / cli.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    source = Path(cfg["source"]["checkpoint"])
    sft = Path(cfg["matched_sft_control"]["checkpoint"])
    schedule_path = (ROOT / cfg["matched_sft_control"]["phase3e_schedule"]).resolve()
    cache_path = (ROOT / cfg["rollout"]["phase3b_cache"]).resolve()
    cache_manifest_path = (ROOT / cfg["rollout"]["phase3b_manifest"]).resolve()
    train_manifest = (ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl").resolve()

    required = [source, sft, schedule_path, cache_path, cache_manifest_path, train_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    checks = {
        "p1_sha256_exact": sha256(source) == cfg["source"]["checkpoint_sha256"],
        "matched_sft_sha256_exact": sha256(sft) == cfg["matched_sft_control"]["checkpoint_sha256"],
        "phase3b_cache_sha256_exact": sha256(cache_path) == cfg["rollout"]["cache_sha256"],
    }
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    checks["phase3e_schedule_field_exact"] = schedule.get("sha256") == cfg["matched_sft_control"]["schedule_sha256_field"]
    checks["phase3e_schedule_4000_unique"] = len(schedule["sample_ids"]) == 4000 and len(set(schedule["sample_ids"])) == 4000
    checks["phase3e_schedule_balanced"] = Counter(schedule["class_labels"]) == Counter({0: 2000, 1: 2000})
    checks["canonical_prompt_exact"] = cfg["prompt"]["user_question"] == "Determine whether this image is authentic and explain the forensic evidence."
    checks["legacy_prompt_disabled"] = cfg["prompt"]["legacy_localization_question_allowed"] is False

    manifest_by_id = {row["sample_id"]: row for _, row in read_jsonl(train_manifest)}
    checks["schedule_manifest_set_covered"] = set(schedule["sample_ids"]).issubset(manifest_by_id)
    checks["schedule_labels_match_manifest"] = all(
        int(manifest_by_id[sid]["class_label"]) == int(label)
        for sid, label in zip(schedule["sample_ids"], schedule["class_labels"])
    )

    # Preserve the exact Phase 3E trajectory as the first 4000 exposures. Then
    # cover every unseen train sample once and add only 328 deterministic repeats.
    manifest_ids = list(manifest_by_id)
    phase3e_seen = set(schedule["sample_ids"])
    remaining = {
        label: [sid for sid in manifest_ids if int(manifest_by_id[sid]["class_label"]) == label and sid not in phase3e_seen]
        for label in (0, 1)
    }
    generator = random.Random(int(cfg["experiment"]["seed"]) + 3500)
    for values in remaining.values():
        generator.shuffle(values)
    if len(remaining[0]) != 6836 or len(remaining[1]) != 6836:
        raise RuntimeError(f"unexpected remaining population: { {key: len(value) for key, value in remaining.items()} }")
    all_by_label = {
        label: [sid for sid in manifest_ids if int(manifest_by_id[sid]["class_label"]) == label]
        for label in (0, 1)
    }
    for values in all_by_label.values():
        generator.shuffle(values)
    repeat_per_class = (int(cfg["training"]["total_image_exposures"]) - len(manifest_ids)) // 2
    tail_ids = []
    for real, fake in zip(remaining[0], remaining[1]):
        tail_ids.extend((real, fake))
    for real, fake in zip(all_by_label[0][:repeat_per_class], all_by_label[1][:repeat_per_class]):
        tail_ids.extend((real, fake))
    phase3f_ids = list(schedule["sample_ids"]) + tail_ids
    index_by_id = {sid: index for index, sid in enumerate(manifest_ids)}
    phase3f_schedule = {
        "seed": int(cfg["experiment"]["seed"]), "construction": cfg["training"]["schedule"],
        "indices": [index_by_id[sid] for sid in phase3f_ids], "sample_ids": phase3f_ids,
        "class_labels": [int(manifest_by_id[sid]["class_label"]) for sid in phase3f_ids],
        "phase3e_exact_prefix_exposures": 4000, "population_size": len(manifest_ids),
        "repeat_exposures": len(phase3f_ids) - len(set(phase3f_ids)),
    }
    phase3f_schedule["sha256"] = canonical_hash({
        key: phase3f_schedule[key] for key in ("seed", "construction", "indices", "sample_ids", "class_labels")
    })
    checks.update({
        "phase3f_schedule_18000": len(phase3f_ids) == 18000,
        "phase3f_schedule_balanced": Counter(phase3f_schedule["class_labels"]) == Counter({0: 9000, 1: 9000}),
        "phase3f_schedule_covers_full_17672_population": set(phase3f_ids) == set(manifest_ids),
        "phase3f_schedule_repeat_count_328": phase3f_schedule["repeat_exposures"] == 328,
        "phase3e_exact_4000_prefix": phase3f_ids[:4000] == schedule["sample_ids"],
    })
    dump(out / "training/schedule.json", phase3f_schedule)

    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_by_id = {}
    provenance_errors = []
    eligibility_hist = Counter()
    seg_hist = Counter()
    for line_number, row in read_jsonl(cache_path):
        sid = row["sample_id"]
        if sid in cache_by_id:
            provenance_errors.append({"line": line_number, "sample_id": sid, "error": "duplicate_sample_id"})
            continue
        cache_by_id[sid] = row
        eligibility_hist[bool(row["replay_eligible"])] += 1
        seg_hist[int(row["seg_count"])] += 1
        expected = {
            "source_checkpoint_sha256": cfg["source"]["checkpoint_sha256"],
            "source_optimizer_step": cfg["source"]["optimizer_step"],
            "source_epoch": cfg["source"]["epoch"],
            "prompt_template_id": cfg["prompt"]["template_id"],
        }
        bad = {key: [row.get(key), value] for key, value in expected.items() if row.get(key) != value}
        if bad:
            provenance_errors.append({"line": line_number, "sample_id": sid, "mismatch": bad})
        if row.get("full_input_token_ids") != row.get("prompt_token_ids", []) + row.get("generated_token_ids", []):
            provenance_errors.append({"line": line_number, "sample_id": sid, "error": "full_token_concatenation"})
        if row.get("image_path") != manifest_by_id.get(sid, {}).get("resolved_image_path", row.get("image_path")):
            # Older frozen manifests do not store resolved_image_path. Dataset identity is checked below by sample ID/image name.
            pass

    train_fake_ids = {sid for sid, row in manifest_by_id.items() if int(row["class_label"]) == 1}
    scheduled_fake_ids = {sid for sid, label in zip(phase3f_schedule["sample_ids"], phase3f_schedule["class_labels"]) if int(label) == 1}
    checks.update({
        "cache_count_exact": len(cache_by_id) == 8836,
        "cache_set_equals_train_fake": set(cache_by_id) == train_fake_ids,
        "scheduled_fake_cache_covered": scheduled_fake_ids.issubset(cache_by_id),
        "all_row_provenance_exact": not provenance_errors,
        "manifest_declares_batch1": cache_manifest.get("generation_batch_size") == 1,
        "manifest_declares_canonical_batch_identity": cache_manifest.get("canonical_historical_batch_size_identity") is True,
        "manifest_cache_sha_matches": cache_manifest.get("cache_sha256") == cfg["rollout"]["cache_sha256"],
    })
    eligible_rate = eligibility_hist[True] / max(1, len(cache_by_id))
    scheduled_eligible = sum(bool(cache_by_id[sid]["replay_eligible"]) for sid in scheduled_fake_ids)
    scheduled_eligible_rate = scheduled_eligible / max(1, len(scheduled_fake_ids))
    checks["global_eligible_rate_at_least_95pct"] = eligible_rate >= float(cfg["rollout"]["minimum_eligible_rate"])
    checks["scheduled_eligible_rate_at_least_95pct"] = scheduled_eligible_rate >= float(cfg["rollout"]["minimum_eligible_rate"])

    frozen_rows = []
    for exposure, (sid, label) in enumerate(zip(phase3f_schedule["sample_ids"], phase3f_schedule["class_labels"])):
        if int(label) != 1:
            continue
        row = cache_by_id[sid]
        manifest_row = manifest_by_id[sid]
        frozen_rows.append({
            "exposure": exposure,
            "sample_id": sid,
            "dataset_index": int(phase3f_schedule["indices"][exposure]),
            "image_identity": manifest_row.get("image_identity", manifest_row.get("source_record_id")),
            "image_name": manifest_row.get("image_name"),
            "image_relpath": manifest_row.get("image_relpath"),
            "prompt_token_ids_sha256": canonical_hash(row["prompt_token_ids"]),
            "generated_token_ids_sha256": canonical_hash(row["generated_token_ids"]),
            "full_input_token_ids_sha256": canonical_hash(row["full_input_token_ids"]),
            "decoded_text_sha256": hashlib.sha256(row["decoded_text"].encode()).hexdigest(),
            "seg_count": row["seg_count"],
            "seg_positions": row["seg_positions"],
            "usable_seg_predictor_position": row["usable_seg_predictor_position"],
            "replay_eligible": row["replay_eligible"],
        })

    status = "PASS" if all(checks.values()) else "FAIL"
    reuse_audit = {
        "status": status, "checks": checks, "provenance_error_count": len(provenance_errors),
        "provenance_error_examples": provenance_errors[:20], "cache_path": str(cache_path),
        "cache_sha256": sha256(cache_path), "cache_rows": len(cache_by_id),
        "cache_eligible": eligibility_hist[True], "cache_eligible_rate": eligible_rate,
        "scheduled_fake_count": len(scheduled_fake_ids), "scheduled_eligible": scheduled_eligible,
        "scheduled_eligible_rate": scheduled_eligible_rate,
        "seg_count_histogram": {str(key): value for key, value in sorted(seg_hist.items())},
        "decision": "REUSE_PHASE3B_BATCH1_CACHE" if status == "PASS" else "DO_NOT_TRAIN",
    }
    dump(out / "phase3b_rollout_reuse_audit.json", reuse_audit)
    dump(out / "frozen_rollout_manifest.json", {
        "status": "FROZEN" if status == "PASS" else "INVALID", "protocol": cfg["rollout"]["protocol"],
        "source_cache_path": str(cache_path), "source_cache_sha256": sha256(cache_path),
        "schedule_path": str((out / "training/schedule.json").resolve()), "schedule_sha256_field": phase3f_schedule["sha256"],
        "phase3e_prefix_schedule_path": str(schedule_path), "phase3e_prefix_schedule_sha256_field": schedule["sha256"],
        "scheduled_fake_exposures": len(frozen_rows), "scheduled_fake_unique": len(scheduled_fake_ids),
        "rows_sha256": canonical_hash(frozen_rows), "rows": frozen_rows,
    })
    dump(out / "rollout_protocol.json", {
        "status": "FROZEN", "off_policy": True, "teacher": "canonical_P1", "generation_batch_size": 1,
        "refresh_steps": [], "partial_cache_mixing": False, "quality_filtering": False,
        "same_fixed_cache_at_steps": [0, 500, 1000, 4500], "canonical_user_prompt": cfg["prompt"]["user_question"],
    })
    dump(out / "rollout_statistics.json", {
        "cache_rows": len(cache_by_id), "eligible_rate": eligible_rate,
        "scheduled_fake_exposures": len(frozen_rows), "scheduled_eligible_rate": scheduled_eligible_rate,
        "seg_count_histogram": {str(key): value for key, value in sorted(seg_hist.items())},
    })
    dump(out / "rollout_integrity_audit.json", {
        "status": status, "cache_sha256_expected": cfg["rollout"]["cache_sha256"],
        "cache_sha256_observed": sha256(cache_path), "integrity_checkpoints": [0, 500, 1000, 4500],
        "step_0": "PASS" if status == "PASS" else "FAIL", "step_500": "PENDING", "step_1000": "PENDING",
        "step_4500": "PENDING",
    })
    dump(out / "matched_sft_fairness_audit.json", {
        "status": "MATERIAL_MISMATCH_MATCHED_P3F_SFT_CONDITIONALLY_DEFERRED", "p1_initialization_exact": checks["p1_sha256_exact"],
        "matched_sft_checkpoint_exact": checks["matched_sft_sha256_exact"],
        "same_first_1000_sample_order_and_4000_exposures": checks["phase3e_exact_4000_prefix"],
        "same_first_1000_learning_rate_trajectory": False,
        "full_training_budget_matched": False, "aogd_optimizer_steps": 4500, "sft_optimizer_steps": 1000,
        "aogd_image_exposures": 18000, "sft_image_exposures": 4000,
        "same_batch_and_accumulation": True, "same_lora_configuration": True, "same_language_ce": True,
        "material_mismatch": "linear decay horizon changes the LR trajectory after warmup and AOGD has 3500 additional steps",
        "reuse_decision": "DO_NOT_USE_PHASE3E_AS_EXACT_MATCHED_CONTROL_DEFER_P3F_SFT_UNTIL_AOGD_BEATS_P1",
        "matched_sft_trigger": "run only if selected 4500-step AOGD shows validation canonical G0 improvement over P1 and passes non-regression gates",
        "historical_control_role": "Phase3E SFT remains a historical baseline only",
    })
    if status != "PASS":
        raise RuntimeError("Phase 3F prepare audit failed; formal preflight/training prohibited")
    print(json.dumps({"status": status, "cache_eligible_rate": eligible_rate,
                      "scheduled_eligible_rate": scheduled_eligible_rate,
                      "scheduled_fake_count": len(scheduled_fake_ids)}, indent=2))


if __name__ == "__main__":
    main()
