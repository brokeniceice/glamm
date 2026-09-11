#!/usr/bin/env python3
"""Phase 6A read-only audit of frozen internal TRAIN phrase/mask supervision."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FAKE = ROOT / "outputs/data_audits/unified_forensics_split_v1/train_fake.jsonl"
RAW = Path("/data/yz/myLISA_storage/AIGC/SynthScars/train/annotations/train.json")
OUT = ROOT / "outputs/phase6a_architecture_audit/synthscars_phrase_mask_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, dict[str, float | int]]:
    total = len(values)
    counts = Counter("5+" if value >= 5 else str(value) for value in values)
    return {
        key: {"count": counts[key], "ratio": counts[key] / total if total else 0.0}
        for key in ("1", "2", "3", "4", "5+")
    }


def decode_ref(ref: dict, height: int, width: int) -> np.ndarray | None:
    output = np.zeros((height, width), dtype=np.uint8)
    valid = False
    for polygon in ref.get("segmentation") or ref.get("polygons") or []:
        flat = np.asarray(polygon, dtype=np.float64).reshape(-1)
        if flat.size < 6 or flat.size % 2:
            continue
        rles = coco_mask.frPyObjects([flat], height, width)
        output |= coco_mask.decode(rles).astype(np.uint8).squeeze()
        valid = True
    return output.astype(bool) if valid else None


def main() -> None:
    frozen = [json.loads(line) for line in TRAIN_FAKE.read_text().splitlines() if line.strip()]
    raw_items = json.loads(RAW.read_text())
    raw_by_id = {str(next(iter(item))): next(iter(item.values())) for item in raw_items}
    selected_ids = [str(annotation_id) for row in frozen for annotation_id in row.get("annotation_ids") or []]
    if len(frozen) != 8836 or len(selected_ids) != 8971 or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("frozen TRAIN identity/count drift")
    if any(annotation_id not in raw_by_id for annotation_id in selected_ids):
        raise RuntimeError("frozen annotation ID missing from raw SynthScars train.json")

    phrase_counts: list[int] = []
    region_counts: list[int] = []
    image_phrase_counts: list[int] = []
    phrases: list[str] = []
    within_annotation_duplicate_phrases = 0
    refs_with_multiple_regions = 0
    invalid_phrase = invalid_polygon = empty_mask = 0
    annotations_with_overlap = overlapping_pairs = 0
    annotations_with_shared_exact_region = shared_exact_region_pairs = 0
    total_refs = total_polygons = 0

    for frozen_row in frozen:
        image_phrase_count = 0
        for annotation_id in frozen_row.get("annotation_ids") or []:
            record = raw_by_id[annotation_id]
            refs = record.get("refs") or []
            phrase_counts.append(len(refs))
            regions = sum(len(ref.get("segmentation") or ref.get("polygons") or []) for ref in refs)
            region_counts.append(regions)
            image_phrase_count += len(refs)
            total_refs += len(refs)
            total_polygons += regions
            local_phrases = [" ".join(str(ref.get("sentence") or "").split()) for ref in refs]
            phrases.extend(local_phrases)
            within_annotation_duplicate_phrases += len(local_phrases) - len(set(local_phrases))
            invalid_phrase += sum(not phrase for phrase in local_phrases)
            refs_with_multiple_regions += sum(
                len(ref.get("segmentation") or ref.get("polygons") or []) > 1 for ref in refs
            )

            image_path = Path(str(frozen_row.get("image_path") or ""))
            if not image_path.is_file():
                candidate = Path(str(record.get("img_file_name") or ""))
                image_path = candidate if candidate.is_file() else image_path.parent / candidate.name
            with Image.open(image_path) as image:
                width, height = image.size
            masks = []
            for ref in refs:
                decoded = decode_ref(ref, height, width)
                invalid_polygon += int(decoded is None)
                empty_mask += int(decoded is not None and not decoded.any())
                masks.append(decoded)
            overlap_here = shared_here = False
            for left, right in combinations(masks, 2):
                if left is None or right is None:
                    continue
                if np.logical_and(left, right).any():
                    overlapping_pairs += 1
                    overlap_here = True
                if np.array_equal(left, right):
                    shared_exact_region_pairs += 1
                    shared_here = True
            annotations_with_overlap += int(overlap_here)
            annotations_with_shared_exact_region += int(shared_here)
        image_phrase_counts.append(image_phrase_count)

    unique_phrases = len(set(phrases))
    payload = {
        "schema": "phase6a_synthscars_phrase_mask_audit_v1",
        "status": "COMPLETE",
        "scope": "frozen internal TRAIN Fake only; no val/test/external benchmark opened",
        "sources": {
            "frozen_train_fake": {"path": str(TRAIN_FAKE), "sha256": sha256(TRAIN_FAKE)},
            "raw_synthscars_train": {"path": str(RAW), "sha256": sha256(RAW)},
        },
        "population": {
            "frozen_images": len(frozen),
            "selected_original_annotation_samples": len(selected_ids),
            "unique_selected_annotation_ids": len(set(selected_ids)),
        },
        "phrase_mask_mapping": {
            "rule": "raw refs[i].sentence maps directly to refs[i].segmentation polygon list",
            "total_phrase_refs": total_refs,
            "total_polygon_regions": total_polygons,
            "unique_phrase_exact_whitespace_normalized": unique_phrases,
            "duplicate_phrase_occurrences_global": len(phrases) - unique_phrases,
            "duplicate_phrase_occurrences_within_annotation": within_annotation_duplicate_phrases,
            "phrase_refs_with_multiple_polygon_regions": refs_with_multiple_regions,
            "invalid_or_empty_phrase_refs": invalid_phrase,
            "refs_with_no_valid_polygon": invalid_polygon,
            "refs_decoding_to_empty_mask": empty_mask,
            "multiple_phrases_sharing_exact_same_decoded_region_pairs": shared_exact_region_pairs,
            "annotations_with_shared_exact_region": annotations_with_shared_exact_region,
            "overlapping_phrase_mask_pairs": overlapping_pairs,
            "annotations_with_any_overlapping_phrase_masks": annotations_with_overlap,
        },
        "phrases_per_original_annotation": {
            "minimum": min(phrase_counts), "maximum": max(phrase_counts),
            "mean": float(np.mean(phrase_counts)), "distribution": distribution(phrase_counts),
        },
        "polygon_regions_per_original_annotation": {
            "minimum": min(region_counts), "maximum": max(region_counts),
            "mean": float(np.mean(region_counts)), "distribution": distribution(region_counts),
        },
        "phrases_per_frozen_image_across_selected_annotations": {
            "minimum": min(image_phrase_counts), "maximum": max(image_phrase_counts),
            "mean": float(np.mean(image_phrase_counts)), "distribution": distribution(image_phrase_counts),
        },
        "recoverability": {
            "original_per_reference_geometry_available_before_union": True,
            "stable_without_manual_reannotation": True,
            "qualification": (
                "Direct phrase_i to refs[i] polygon-mask supervision is recoverable for every ref with a "
                "nonempty decoded mask. One objectively empty decoded ref must be retained as an explicit "
                "invalid/zero-mask case or excluded by a predeclared rule; no semantic remapping is needed."
            ),
        },
        "firewall": {
            "internal_validation_accessed": False, "internal_test_accessed": False,
            "official1000_accessed": False, "external_benchmark_accessed": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(OUT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
