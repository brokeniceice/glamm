#!/usr/bin/env python3
"""Build unified real-image manifests and source-level integrity audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.real_images import UnifiedRealImageAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audits/real_images_unified_v1"),
    )
    parser.add_argument("--thumbnails-per-source", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=2500)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
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


def enrich_common_sha256(adapter: UnifiedRealImageAdapter, progress_every: int) -> None:
    ffhq_samples = [sample for sample in adapter.samples if sample["source"] == "FFHQ"]
    for index, sample in enumerate(adapter.samples, start=1):
        identity = sample["identity_hash"]
        if identity["algorithm"] == "sha256":
            sample["content_sha256"] = identity["value"]
        else:
            sample["content_sha256"] = file_sha256(adapter.datasets_root / sample["image_relpath"])
        if progress_every > 0 and index % progress_every == 0:
            print(f"sha256 identity: {index}/{len(adapter)} records", flush=True)
    if ffhq_samples:
        print(f"computed SHA256 for {len(ffhq_samples)} FFHQ images", flush=True)


def audit_adapter(adapter: UnifiedRealImageAdapter, progress_every: int) -> Dict[str, Any]:
    missing = []
    unreadable = []
    size_mismatches = []
    extension_counts = Counter()
    mode_counts = Counter()
    source_mode_counts: Dict[str, Counter] = defaultdict(Counter)
    source_counts = Counter()
    role_counts = Counter()
    category_counts = Counter()
    source_category_counts: Dict[str, Counter] = defaultdict(Counter)
    widths, heights = [], []
    total_bytes = 0

    for index, sample in enumerate(adapter.samples, start=1):
        path = adapter.datasets_root / sample["image_relpath"]
        source_counts[sample["source"]] += 1
        role_counts[sample["source_role"]] += 1
        category_counts[sample["content_category"]] += 1
        source_category_counts[sample["source"]][sample["content_category"]] += 1
        extension_counts[path.suffix.lower()] += 1
        if not path.is_file():
            missing.append(sample["sample_id"])
            continue
        actual_bytes = path.stat().st_size
        total_bytes += actual_bytes
        expected_bytes = sample.get("bytes")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            size_mismatches.append({
                "sample_id": sample["sample_id"],
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            })
        try:
            with Image.open(path) as image:
                widths.append(image.width)
                heights.append(image.height)
                mode_counts[image.mode] += 1
                source_mode_counts[sample["source"]][image.mode] += 1
        except Exception as error:
            unreadable.append({"sample_id": sample["sample_id"], "error": str(error)})
        if progress_every > 0 and (index % progress_every == 0 or index == len(adapter)):
            print(f"real image headers: {index}/{len(adapter)}", flush=True)

    sha_groups: Dict[str, list] = defaultdict(list)
    for sample in adapter.samples:
        sha_groups[sample["content_sha256"]].append(sample["sample_id"])
    exact_duplicates = [
        {"sha256": sha256, "sample_ids": sample_ids}
        for sha256, sample_ids in sha_groups.items()
        if len(sample_ids) > 1
    ]

    return {
        "schema_version": "real_images_audit_v1",
        "sample_count": len(adapter),
        "source_counts": dict(source_counts),
        "role_counts": dict(role_counts),
        "content_category_counts": dict(category_counts),
        "source_content_category_counts": {
            source: dict(counts) for source, counts in source_category_counts.items()
        },
        "extension_counts": dict(extension_counts),
        "image_mode_counts": dict(mode_counts),
        "source_image_mode_counts": {
            source: dict(counts) for source, counts in source_mode_counts.items()
        },
        "image_width": {
            "min": min(widths) if widths else None,
            "median": float(np.median(widths)) if widths else None,
            "max": max(widths) if widths else None,
        },
        "image_height": {
            "min": min(heights) if heights else None,
            "median": float(np.median(heights)) if heights else None,
            "max": max(heights) if heights else None,
        },
        "total_image_bytes": total_bytes,
        "missing_images": missing,
        "unreadable_image_headers": unreadable,
        "size_mismatches": size_mismatches,
        "exact_sha256_duplicate_groups": exact_duplicates,
    }


def write_thumbnails(adapter: UnifiedRealImageAdapter, output_dir: Path, count: int) -> list:
    if count <= 0:
        return []
    source_indices: Dict[str, list] = defaultdict(list)
    for index, sample in enumerate(adapter.samples):
        source_indices[sample["source"]].append(index)
    records = []
    for source, indices in source_indices.items():
        selected = np.linspace(0, len(indices) - 1, num=min(count, len(indices)), dtype=int)
        source_dir = output_dir / "thumbnails" / source
        source_dir.mkdir(parents=True, exist_ok=True)
        for output_index, selected_index in enumerate(selected):
            sample = adapter.get_sample(indices[int(selected_index)], decode_mask=False)
            with Image.open(sample["image_path"]) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
            output_name = f"{output_index:03d}_{Path(sample['image_name']).stem}.jpg"
            thumbnail.save(source_dir / output_name, quality=88)
            records.append({
                "sample_id": sample["sample_id"],
                "source": source,
                "content_category": sample["content_category"],
                "output_file": str(Path("thumbnails") / source / output_name),
            })
    return records


def exact_duplicate_exclusions(adapter: UnifiedRealImageAdapter, audit: Mapping[str, Any]) -> list:
    by_id = {sample["sample_id"]: sample for sample in adapter.samples}
    exclusions = []
    for group in audit["exact_sha256_duplicate_groups"]:
        ordered = sorted(group["sample_ids"], key=lambda sample_id: by_id[sample_id]["source_record_index"])
        kept = ordered[0]
        for excluded in ordered[1:]:
            exclusions.append({
                "excluded_sample_id": excluded,
                "kept_sample_id": kept,
                "sha256": group["sha256"],
                "reason": "exact_content_duplicate",
            })
    return exclusions


def main() -> None:
    args = parse_args()
    datasets_root = args.datasets_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    adapter = UnifiedRealImageAdapter(datasets_root)
    enrich_common_sha256(adapter, args.progress_every)

    all_candidate_records = list(adapter.iter_manifest_records("candidate_train"))
    atomic_write_jsonl(output_dir / "candidate_train_all_manifest.jsonl", all_candidate_records)
    atomic_write_jsonl(output_dir / "heldout_test_manifest.jsonl", adapter.iter_manifest_records("heldout_test"))
    audit = audit_adapter(adapter, args.progress_every)
    exclusions = exact_duplicate_exclusions(adapter, audit)
    excluded_ids = {record["excluded_sample_id"] for record in exclusions}
    deduplicated_candidates = [
        record for record in all_candidate_records if record["sample_id"] not in excluded_ids
    ]
    atomic_write_jsonl(output_dir / "candidate_train_manifest.jsonl", deduplicated_candidates)
    atomic_write_json(output_dir / "exact_duplicate_exclusions.json", exclusions)
    thumbnails = write_thumbnails(adapter, output_dir, args.thumbnails_per_source)
    atomic_write_json(output_dir / "audit.json", audit)
    atomic_write_json(output_dir / "thumbnail_index.json", thumbnails)

    source_manifest_sha256 = {
        source: file_sha256(path) for source, path in adapter.source_manifest_paths.items()
    }
    summary = {
        "schema_version": "real_images_unified_v1",
        "datasets_root": str(datasets_root),
        "candidate_train_raw_count": audit["role_counts"].get("candidate_train", 0),
        "candidate_train_exact_deduplicated_count": len(deduplicated_candidates),
        "heldout_test_count": audit["role_counts"].get("heldout_test", 0),
        "source_counts": audit["source_counts"],
        "content_category_counts": audit["content_category_counts"],
        "source_manifest_sha256": source_manifest_sha256,
        "missing_image_count": len(audit["missing_images"]),
        "unreadable_image_header_count": len(audit["unreadable_image_headers"]),
        "size_mismatch_count": len(audit["size_mismatches"]),
        "exact_sha256_duplicate_group_count": len(audit["exact_sha256_duplicate_groups"]),
        "exact_duplicate_exclusion_count": len(exclusions),
        "thumbnail_count": len(thumbnails),
        "remaining_before_balanced_selection": [
            "PASS content classification and manual audit",
            "cross-source perceptual/embedding near-duplicate audit",
            "content-matched selection equal to the audited SynthScars visual-identity count",
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
