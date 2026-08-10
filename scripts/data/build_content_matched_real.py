#!/usr/bin/env python3
"""Merge real-image content labels and select a SynthScars-matched real set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("human", "animal", "object", "scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/real_images_unified_v1/candidate_train_manifest.jsonl",
    )
    parser.add_argument(
        "--pass-gpt-predictions",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_gpt_pass_v1/predictions.jsonl",
    )
    parser.add_argument(
        "--synth-content-predictions",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1/synthscars_content_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_matched_real_v1",
    )
    parser.add_argument("--selection-seed", default="unified-forensics-real-v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def nested_counts(rows: Iterable[Mapping[str, Any]], first: str, second: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row[first])][str(row[second])] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


def merge_real_labels(
    real_rows: list[Mapping[str, Any]], pass_rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    pass_by_id = {row["sample_id"]: row for row in pass_rows}
    if len(pass_by_id) != len(pass_rows):
        raise ValueError("Duplicate sample IDs in PASS GPT predictions")
    expected_pass_ids = {row["sample_id"] for row in real_rows if row["source"] == "PASS"}
    if set(pass_by_id) != expected_pass_ids:
        missing = expected_pass_ids - set(pass_by_id)
        extra = set(pass_by_id) - expected_pass_ids
        raise ValueError(f"PASS GPT coverage mismatch: missing={len(missing)} extra={len(extra)}")

    merged = []
    for raw in real_rows:
        row = dict(raw)
        original_category = row["content_category"]
        if row["source"] == "PASS":
            prediction = pass_by_id[row["sample_id"]]
            category = prediction["gpt_label"]
            row.update({
                "content_category_original": original_category,
                "content_category": category,
                "content_label_source": "gpt",
                "content_label_model": prediction["model"],
                "content_label_confidence": prediction["gpt_confidence"],
                "content_label_reason": prediction["gpt_reason"],
                "content_label_response_id": prediction.get("response_id"),
            })
        else:
            category = original_category
            if category not in CATEGORIES:
                raise ValueError(f"Invalid existing category for {row['sample_id']}: {category}")
            row.update({
                "content_category_original": original_category,
                "content_label_model": None,
                "content_label_confidence": None,
                "content_label_reason": None,
                "content_label_response_id": None,
            })
        if category not in (*CATEGORIES, "ambiguous"):
            raise ValueError(f"Invalid merged category for {row['sample_id']}: {category}")
        row["content_matching_eligible"] = category in CATEGORIES
        merged.append(row)
    if len({row["sample_id"] for row in merged}) != len(merged):
        raise ValueError("Duplicate sample IDs in merged real manifest")
    return merged


def fake_target_counts(synth_rows: list[Mapping[str, Any]]) -> Counter:
    if len({row["sample_id"] for row in synth_rows}) != len(synth_rows):
        raise ValueError("Duplicate sample IDs in SynthScars content predictions")
    counts = Counter(row["predicted_category"] for row in synth_rows)
    unexpected = set(counts) - set(CATEGORIES)
    if unexpected:
        raise ValueError(f"Unexpected SynthScars categories: {sorted(unexpected)}")
    return counts


def stable_key(sample_id: str, category: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{sample_id}".encode("utf-8")).hexdigest()


def select_matched_real(
    merged_rows: list[Mapping[str, Any]], target_counts: Mapping[str, int], seed: str
) -> list[dict[str, Any]]:
    selected = []
    for category in CATEGORIES:
        candidates = [row for row in merged_rows if row["content_category"] == category]
        target = int(target_counts.get(category, 0))
        if len(candidates) < target:
            raise ValueError(f"Insufficient real {category}: available={len(candidates)} target={target}")
        candidates.sort(key=lambda row: stable_key(row["sample_id"], category, seed))
        for rank, raw in enumerate(candidates[:target]):
            row = dict(raw)
            row["content_matching_target_category"] = category
            row["content_matching_selection_rank"] = rank
            row["content_matching_selection_seed"] = seed
            selected.append(row)
    selected.sort(key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise AssertionError("Selection contains duplicate real images")
    return selected


def main() -> None:
    args = parse_args()
    real_path = args.real_manifest.expanduser().resolve()
    pass_path = args.pass_gpt_predictions.expanduser().resolve()
    synth_path = args.synth_content_predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    real_rows = load_jsonl(real_path)
    pass_rows = load_jsonl(pass_path)
    synth_rows = load_jsonl(synth_path)
    merged = merge_real_labels(real_rows, pass_rows)
    eligible = [row for row in merged if row["content_matching_eligible"]]
    targets = fake_target_counts(synth_rows)
    selected = select_matched_real(merged, targets, args.selection_seed)

    atomic_write_jsonl(output_dir / "real_content_labels_all.jsonl", merged)
    atomic_write_jsonl(output_dir / "real_content_matching_eligible.jsonl", eligible)
    atomic_write_jsonl(output_dir / "selected_real_manifest.jsonl", selected)
    atomic_write_json(output_dir / "target_fake_content_counts.json", dict(targets))
    summary = {
        "schema_version": "content_matched_real_v1",
        "selection_seed": args.selection_seed,
        "real_input_count": len(real_rows),
        "pass_gpt_count": len(pass_rows),
        "eligible_real_count": len(eligible),
        "excluded_ambiguous_count": sum(row["content_category"] == "ambiguous" for row in merged),
        "real_content_counts": dict(Counter(row["content_category"] for row in merged)),
        "real_source_content_counts": nested_counts(merged, "source", "content_category"),
        "fake_target_counts": dict(targets),
        "selected_real_count": len(selected),
        "selected_content_counts": dict(Counter(row["content_category"] for row in selected)),
        "selected_source_counts": dict(Counter(row["source"] for row in selected)),
        "selected_source_content_counts": nested_counts(selected, "source", "content_category"),
        "real_manifest_sha256": sha256_file(real_path),
        "pass_gpt_predictions_sha256": sha256_file(pass_path),
        "synth_content_predictions_sha256": sha256_file(synth_path),
        "content_label_policy": {
            "PASS": "GPT label",
            "other_real_sources": "existing source-manifest label",
            "ambiguous": "retained in audit and excluded from matched selection",
            "SynthScars_target_distribution": "existing frozen CLIP content prediction",
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
