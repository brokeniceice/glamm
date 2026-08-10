#!/usr/bin/env python3
"""Create deterministic 8:1:1 balanced real/fake splits with pHash grouping."""

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
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-manifest", type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_matched_real_v1/selected_real_manifest.jsonl",
    )
    parser.add_argument(
        "--synth-manifest", type=Path,
        default=REPO_ROOT / "outputs/data_audits/synthscars_image_grouped_v1/train_manifest.jsonl",
    )
    parser.add_argument(
        "--synth-content-predictions", type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1/synthscars_content_predictions.jsonl",
    )
    parser.add_argument(
        "--phash-pairs", type=Path,
        default=REPO_ROOT / "outputs/data_audits/image_phash_v1/near_duplicate_pairs.jsonl",
    )
    parser.add_argument(
        "--official-synth-test", type=Path,
        default=REPO_ROOT / "outputs/data_audits/synthscars_image_grouped_v1/test_manifest.jsonl",
    )
    parser.add_argument(
        "--raise-heldout-real", type=Path,
        default=REPO_ROOT / "outputs/data_audits/real_images_unified_v1/heldout_test_manifest.jsonl",
    )
    parser.add_argument(
        "--phash-summary", type=Path,
        default=REPO_ROOT / "outputs/data_audits/image_phash_v1/summary.json",
    )
    parser.add_argument(
        "--official-test-leakage-pairs", type=Path,
        default=REPO_ROOT / "outputs/data_audits/official_synth_test_leakage_v1/official_test_near_duplicate_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/data_audits/unified_forensics_split_v1",
    )
    parser.add_argument("--split-seed", default="unified-forensics-split-v1")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def category_split_targets(count: int) -> dict[str, int]:
    raw = {split: count * RATIOS[split] for split in SPLITS}
    targets = {split: int(raw[split]) for split in SPLITS}
    remaining = count - sum(targets.values())
    tie_order = {"val": 0, "test": 1, "train": 2}
    ranked = sorted(SPLITS, key=lambda split: (-(raw[split] - targets[split]), tie_order[split]))
    for split in ranked[:remaining]:
        targets[split] += 1
    return targets


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def duplicate_groups(sample_ids: set[str], pairs: list[Mapping[str, Any]]) -> list[list[str]]:
    union_find = UnionFind(sample_ids)
    for pair in pairs:
        left, right = pair["left_sample_id"], pair["right_sample_id"]
        if left in sample_ids and right in sample_ids:
            union_find.union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        groups[union_find.find(sample_id)].append(sample_id)
    return [sorted(group) for group in groups.values()]


def assign_category_groups(
    rows: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
    category: str,
    seed: str,
) -> dict[str, str]:
    by_id = {row["sample_id"]: row for row in rows if row["content_category"] == category}
    targets = category_split_targets(len(by_id))
    remaining = dict(targets)
    groups = duplicate_groups(set(by_id), pairs)
    groups.sort(key=lambda group: (-len(group), stable_key("|".join(group), seed)))
    assignments: dict[str, str] = {}
    for group in groups:
        size = len(group)
        eligible = [split for split in SPLITS if remaining[split] >= size]
        if not eligible:
            raise RuntimeError(f"Cannot place pHash group of size {size} for {category}")
        split = min(
            eligible,
            key=lambda name: (
                -(remaining[name] / max(targets[name], 1)),
                stable_key("|".join(group) + ":" + name, seed),
            ),
        )
        for sample_id in group:
            assignments[sample_id] = split
        remaining[split] -= size
    if any(remaining.values()):
        raise AssertionError(f"Unfilled split targets for {category}: {remaining}")
    return assignments


def assign_rows(
    rows: list[Mapping[str, Any]], pairs: list[Mapping[str, Any]], domain: str, seed: str
) -> list[dict[str, Any]]:
    assignments = {}
    for category in CATEGORIES:
        assignments.update(assign_category_groups(rows, pairs, category, f"{seed}:{domain}:{category}"))
    result = []
    groups = duplicate_groups({row["sample_id"] for row in rows}, pairs)
    group_by_id = {}
    for group in groups:
        group_id = hashlib.sha256("|".join(group).encode("utf-8")).hexdigest()[:16]
        for sample_id in group:
            group_by_id[sample_id] = (group_id, len(group))
    for raw in rows:
        row = dict(raw)
        group_id, group_size = group_by_id[row["sample_id"]]
        row.update({
            "forensics_domain": domain,
            "dataset_split": assignments[row["sample_id"]],
            "split_seed": seed,
            "split_group_id": group_id,
            "split_group_size": group_size,
        })
        result.append(row)
    return result


def synth_rows_with_content(
    manifest: list[Mapping[str, Any]], predictions: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    prediction_by_id = {row["sample_id"]: row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Duplicate SynthScars content prediction IDs")
    if set(prediction_by_id) != {row["sample_id"] for row in manifest}:
        raise ValueError("SynthScars manifest/content prediction coverage mismatch")
    result = []
    for raw in manifest:
        row = dict(raw)
        prediction = prediction_by_id[row["sample_id"]]
        row["content_category"] = prediction["predicted_category"]
        row["content_label_source"] = "frozen_clip_content_prediction"
        row["content_label_margin"] = prediction["top1_top2_margin"]
        result.append(row)
    return result


def split_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["dataset_split"]][row["content_category"]] += 1
    return {split: dict(counts[split]) for split in SPLITS}


def remove_official_test_leakage(
    real: list[Mapping[str, Any]],
    fake: list[Mapping[str, Any]],
    leakage_pairs: list[Mapping[str, Any]],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    excluded_fake_ids = set()
    official_domain = "synthscars_official_test"
    for pair in leakage_pairs:
        left_domain, right_domain = pair["left_domain"], pair["right_domain"]
        if official_domain not in {left_domain, right_domain} or left_domain == right_domain:
            continue
        excluded_fake_ids.add(
            pair["right_sample_id"] if left_domain == official_domain else pair["left_sample_id"]
        )
    fake_by_id = {row["sample_id"]: row for row in fake}
    missing = excluded_fake_ids - set(fake_by_id)
    if missing:
        raise ValueError(f"Official-test leakage IDs missing from fake pool: {len(missing)}")
    excluded_fake = [fake_by_id[sample_id] for sample_id in sorted(excluded_fake_ids)]
    exclusion_counts = Counter(row["content_category"] for row in excluded_fake)

    excluded_real_ids = set()
    for category, count in exclusion_counts.items():
        candidates = [row for row in real if row["content_category"] == category]
        candidates.sort(key=lambda row: stable_key(row["sample_id"], seed + ":balance-exclusion"))
        excluded_real_ids.update(row["sample_id"] for row in candidates[:count])
    filtered_real = [dict(row) for row in real if row["sample_id"] not in excluded_real_ids]
    filtered_fake = [dict(row) for row in fake if row["sample_id"] not in excluded_fake_ids]
    audit = {
        "policy": "Keep official SynthScars test untouched; exclude overlapping train-side fake and category-matched real samples.",
        "excluded_fake_count": len(excluded_fake_ids),
        "excluded_fake_ids": sorted(excluded_fake_ids),
        "excluded_fake_content_counts": dict(exclusion_counts),
        "excluded_real_count": len(excluded_real_ids),
        "excluded_real_ids": sorted(excluded_real_ids),
        "post_mitigation_cross_leakage_count": sum(
            (pair["left_sample_id"] in {row["sample_id"] for row in filtered_fake})
            or (pair["right_sample_id"] in {row["sample_id"] for row in filtered_fake})
            for pair in leakage_pairs
            if official_domain in {pair["left_domain"], pair["right_domain"]}
            and pair["left_domain"] != pair["right_domain"]
        ),
    }
    return filtered_real, filtered_fake, audit


def main() -> None:
    args = parse_args()
    paths = {
        "real": args.real_manifest.expanduser().resolve(),
        "synth": args.synth_manifest.expanduser().resolve(),
        "synth_content": args.synth_content_predictions.expanduser().resolve(),
        "phash_pairs": args.phash_pairs.expanduser().resolve(),
        "official_synth_test": args.official_synth_test.expanduser().resolve(),
        "raise_heldout_real": args.raise_heldout_real.expanduser().resolve(),
        "phash_summary": args.phash_summary.expanduser().resolve(),
        "official_test_leakage_pairs": args.official_test_leakage_pairs.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    real = load_jsonl(paths["real"])
    fake = synth_rows_with_content(load_jsonl(paths["synth"]), load_jsonl(paths["synth_content"]))
    pairs = load_jsonl(paths["phash_pairs"])
    if len(real) != len(fake):
        raise ValueError(f"Real/fake counts are not balanced: {len(real)} != {len(fake)}")
    if Counter(row["content_category"] for row in real) != Counter(
        row["content_category"] for row in fake
    ):
        raise ValueError("Real/fake content distributions do not match")

    leakage_pairs = load_jsonl(paths["official_test_leakage_pairs"])
    real, fake, leakage_exclusions = remove_official_test_leakage(
        real, fake, leakage_pairs, args.split_seed
    )
    if leakage_exclusions["post_mitigation_cross_leakage_count"]:
        raise AssertionError("Official-test leakage remains after exclusions")
    if Counter(row["content_category"] for row in real) != Counter(
        row["content_category"] for row in fake
    ):
        raise AssertionError("Real/fake distributions differ after leakage exclusions")
    atomic_write_json(output_dir / "official_test_overlap_exclusions.json", leakage_exclusions)

    assigned_real = assign_rows(real, pairs, "real", args.split_seed)
    assigned_fake = assign_rows(fake, pairs, "fake", args.split_seed)
    for split in SPLITS:
        real_split = sorted(
            (row for row in assigned_real if row["dataset_split"] == split),
            key=lambda row: stable_key(row["sample_id"], args.split_seed + ":shuffle"),
        )
        fake_split = sorted(
            (row for row in assigned_fake if row["dataset_split"] == split),
            key=lambda row: stable_key(row["sample_id"], args.split_seed + ":shuffle"),
        )
        combined = sorted(
            [*real_split, *fake_split],
            key=lambda row: stable_key(row["sample_id"], args.split_seed + ":combined"),
        )
        atomic_write_jsonl(output_dir / f"{split}_real.jsonl", real_split)
        atomic_write_jsonl(output_dir / f"{split}_fake.jsonl", fake_split)
        atomic_write_jsonl(output_dir / f"{split}_combined.jsonl", combined)

    official_test = load_jsonl(paths["official_synth_test"])
    official_test_ids = {row["sample_id"] for row in official_test}
    if official_test_ids & {row["sample_id"] for row in assigned_fake}:
        raise ValueError("SynthScars official test overlaps train-source IDs")
    official_test = [
        {**row, "forensics_domain": "fake", "dataset_split": "official_test"}
        for row in official_test
    ]
    atomic_write_jsonl(output_dir / "official_synthscars_test.jsonl", official_test)

    phash_summary = load_json(paths["phash_summary"])
    corrupt_ids = {row["sample_id"] for row in phash_summary.get("decode_failures", [])}
    raise_rows = load_jsonl(paths["raise_heldout_real"])
    clean_raise = [
        {**row, "forensics_domain": "real", "dataset_split": "raise_heldout"}
        for row in raise_rows if row["sample_id"] not in corrupt_ids
    ]
    atomic_write_jsonl(output_dir / "raise_heldout_real_clean.jsonl", clean_raise)
    atomic_write_json(output_dir / "raise_heldout_exclusions.json", sorted(corrupt_ids))

    real_counts, fake_counts = split_counts(assigned_real), split_counts(assigned_fake)
    if real_counts != fake_counts:
        raise AssertionError("Real/fake per-split content counts differ")
    split_sizes = {
        split: {
            "real": sum(row["dataset_split"] == split for row in assigned_real),
            "fake": sum(row["dataset_split"] == split for row in assigned_fake),
        }
        for split in SPLITS
    }
    summary = {
        "schema_version": "unified_forensics_split_v1",
        "split_seed": args.split_seed,
        "requested_ratios": RATIOS,
        "split_sizes": split_sizes,
        "real_content_counts": real_counts,
        "fake_content_counts": fake_counts,
        "official_synthscars_test_count": len(official_test),
        "raise_heldout_clean_count": len(clean_raise),
        "raise_heldout_excluded_corrupt_count": len(corrupt_ids),
        "internal_pool_count_per_domain_after_official_leakage_exclusion": len(real),
        "official_test_overlap_exclusions": leakage_exclusions,
        "phash_group_count_real": sum(row["split_group_size"] > 1 for row in assigned_real),
        "phash_group_count_fake": sum(row["split_group_size"] > 1 for row in assigned_fake),
        "manifest_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "output_manifest_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.glob("*.jsonl"))
        },
        "evaluation_policy": {
            "test": "balanced internal 10% real/fake split",
            "official_synthscars_test": "separate untouched 1000-image fake benchmark",
            "raise_heldout": "separate clean real benchmark; two corrupt TIFFs excluded",
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
