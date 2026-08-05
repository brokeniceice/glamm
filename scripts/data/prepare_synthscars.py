#!/usr/bin/env python3
"""Build image-grouped SynthScars manifests and detailed data audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.synthscars import SynthScarsAdapter, polygons_for_target, ref_signature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/SynthScars"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audits/synthscars_image_grouped_v1"),
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--visualizations-per-split",
        type=int,
        default=50,
        help="Deterministically write this many union-mask overlays for each split.",
    )
    return parser.parse_args()


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


def polygon_bounds(polygons: Iterable[Iterable[float]]) -> List[float]:
    xs, ys = [], []
    for polygon in polygons:
        values = list(polygon)
        xs.extend(values[0::2])
        ys.extend(values[1::2])
    return [min(xs), min(ys), max(xs), max(ys)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_report(adapter: SynthScarsAdapter) -> List[Dict[str, Any]]:
    report = []
    sample_indices = {sample["sample_id"]: index for index, sample in enumerate(adapter.samples)}
    for grouped in adapter.duplicate_image_samples:
        signatures = [ref_signature(ref) for ref in grouped["refs"]]
        signature_counts = Counter(signatures)
        annotation_ref_counts = Counter(ref["annotation_id"] for ref in grouped["refs"])
        masks_by_annotation: Dict[str, np.ndarray] = {}
        decoded = adapter.get_sample(sample_indices[grouped["sample_id"]], decode_masks=True)
        for ref, ref_mask in zip(grouped["refs"], decoded["ref_masks"]):
            annotation_id = ref["annotation_id"]
            current = masks_by_annotation.setdefault(annotation_id, np.zeros_like(ref_mask, dtype=bool))
            current |= ref_mask.astype(bool)
        annotation_ids = grouped["annotation_ids"]
        pairwise_mask_iou = []
        for left_index, left_id in enumerate(annotation_ids):
            for right_id in annotation_ids[left_index + 1 :]:
                left = masks_by_annotation[left_id]
                right = masks_by_annotation[right_id]
                union = np.logical_or(left, right).sum()
                intersection = np.logical_and(left, right).sum()
                pairwise_mask_iou.append({
                    "annotation_ids": [left_id, right_id],
                    "iou": float(intersection / union) if union else None,
                })
        report.append({
            "sample_id": grouped["sample_id"],
            "image_name": grouped["image_name"],
            "annotation_ids": annotation_ids,
            "annotation_ref_counts": dict(annotation_ref_counts),
            "captions_identical": len(set(grouped["annotation_captions"])) == 1,
            "total_refs": len(grouped["refs"]),
            "unique_exact_refs": len(signature_counts),
            "repeated_exact_ref_instances": sum(count - 1 for count in signature_counts.values()),
            "pairwise_annotation_mask_iou": pairwise_mask_iou,
        })
    return report


def audit_split(adapter: SynthScarsAdapter, progress_every: int) -> Dict[str, Any]:
    missing_images = []
    corrupt_images = []
    empty_ref_masks = []
    empty_ref_explanations = []
    empty_union_masks = []
    out_of_bounds_polygons = []
    boundary_roundoff_polygons = []
    ref_counts = []
    mask_area_ratios = []
    polygon_count = 0

    for index, grouped in enumerate(adapter.samples):
        image_path = adapter.root / grouped["image_relpath"]
        if not image_path.is_file():
            missing_images.append(grouped["sample_id"])
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as error:
            corrupt_images.append({"sample_id": grouped["sample_id"], "error": str(error)})
            continue

        for ref in grouped["refs"]:
            polygon_count += len(ref["polygons"])
            if not ref["explanation"]:
                empty_ref_explanations.append({
                    "sample_id": grouped["sample_id"],
                    "ref_id": ref["ref_id"],
                    "phrase": ref["phrase"],
                })
            bounds = polygon_bounds(polygons_for_target(ref, height, width))
            excess = max(0.0, -bounds[0], -bounds[1], bounds[2] - width, bounds[3] - height)
            issue = {
                    "sample_id": grouped["sample_id"],
                    "ref_id": ref["ref_id"],
                    "bounds": bounds,
                    "image_size": [height, width],
                    "max_excess_pixels": excess,
                }
            if excess > 1e-6:
                out_of_bounds_polygons.append(issue)
            elif excess > 0:
                boundary_roundoff_polygons.append(issue)

        decoded = adapter.get_sample(index, decode_masks=True)
        ref_counts.append(len(decoded["ref_masks"]))
        for ref, ref_mask in zip(grouped["refs"], decoded["ref_masks"]):
            if not np.any(ref_mask):
                empty_ref_masks.append({"sample_id": grouped["sample_id"], "ref_id": ref["ref_id"]})
        union_mask = decoded["union_evidence_mask"]
        if not np.any(union_mask):
            empty_union_masks.append(grouped["sample_id"])
        mask_area_ratios.append(float(union_mask.mean()))

        if progress_every > 0 and ((index + 1) % progress_every == 0 or index + 1 == len(adapter)):
            print(f"{adapter.split}: audited {index + 1}/{len(adapter)} images", flush=True)

    ratios = np.asarray(mask_area_ratios, dtype=np.float64)
    return {
        "schema_version": "synthscars_audit_v1",
        "split": adapter.split,
        "annotation_count": adapter.annotation_count,
        "unique_image_count": len(adapter),
        "unique_file_name_count": len({variant["image_name"] for sample in adapter.samples for variant in sample["image_variants"]}),
        "multi_file_variant_identity_count": sum(len(sample["image_variants"]) > 1 for sample in adapter.samples),
        "duplicate_image_count": len(adapter.duplicate_image_samples),
        "ref_count": int(sum(ref_counts)),
        "polygon_count": polygon_count,
        "refs_per_image": {
            "min": min(ref_counts) if ref_counts else None,
            "max": max(ref_counts) if ref_counts else None,
            "mean": float(np.mean(ref_counts)) if ref_counts else None,
        },
        "union_mask_area_ratio": {
            "min": float(ratios.min()) if ratios.size else None,
            "p01": float(np.quantile(ratios, 0.01)) if ratios.size else None,
            "median": float(np.median(ratios)) if ratios.size else None,
            "p99": float(np.quantile(ratios, 0.99)) if ratios.size else None,
            "max": float(ratios.max()) if ratios.size else None,
            "mean": float(ratios.mean()) if ratios.size else None,
        },
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "empty_ref_masks": empty_ref_masks,
        "empty_ref_explanations": empty_ref_explanations,
        "empty_union_masks": empty_union_masks,
        "out_of_bounds_polygons": out_of_bounds_polygons,
        "boundary_roundoff_polygons": boundary_roundoff_polygons,
    }


def write_visualizations(adapter: SynthScarsAdapter, output_dir: Path, count: int) -> List[Dict[str, Any]]:
    if count <= 0 or not len(adapter):
        return []
    count = min(count, len(adapter))
    indices = np.linspace(0, len(adapter) - 1, num=count, dtype=int).tolist()
    visual_dir = output_dir / "visualizations" / adapter.split
    visual_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for output_index, sample_index in enumerate(indices):
        sample = adapter.get_sample(sample_index, decode_masks=True)
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB")
        mask = sample["union_evidence_mask"].astype(bool)
        array = np.asarray(image).copy()
        red = np.zeros_like(array)
        red[..., 0] = 255
        array[mask] = (0.55 * array[mask] + 0.45 * red[mask]).astype(np.uint8)
        overlay = Image.fromarray(array)
        overlay.thumbnail((768, 768), Image.Resampling.LANCZOS)
        output_name = f"{output_index:03d}_{Path(sample['image_name']).stem}.jpg"
        overlay.save(visual_dir / output_name, quality=90)
        records.append({
            "sample_id": sample["sample_id"],
            "output_file": str(Path("visualizations") / adapter.split / output_name),
            "annotation_ids": sample["annotation_ids"],
            "ref_count": len(sample["refs"]),
            "invalid_ref_ids": sample["invalid_ref_ids"],
            "union_mask_area_ratio": float(mask.mean()),
        })
    return records


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    combined_summary = {"schema_version": "synthscars_image_grouped_v1", "root": str(root), "splits": {}}
    for split in args.splits:
        adapter = SynthScarsAdapter(root, split=split)
        atomic_write_jsonl(output_dir / f"{split}_manifest.jsonl", adapter.iter_manifest_records())
        audit = audit_split(adapter, args.progress_every)
        duplicates = duplicate_report(adapter)
        atomic_write_json(output_dir / f"{split}_audit.json", audit)
        atomic_write_json(output_dir / f"{split}_duplicate_images.json", duplicates)
        visualization_index = write_visualizations(adapter, output_dir, args.visualizations_per_split)
        atomic_write_json(output_dir / f"{split}_visualizations.json", visualization_index)
        combined_summary["splits"][split] = {
            "annotation_sha256": file_sha256(adapter.annotation_path),
            "annotation_count": adapter.annotation_count,
            "unique_image_count": len(adapter),
            "unique_file_name_count": audit["unique_file_name_count"],
            "multi_file_variant_identity_count": audit["multi_file_variant_identity_count"],
            "duplicate_image_count": len(adapter.duplicate_image_samples),
            "ref_count": audit["ref_count"],
            "polygon_count": audit["polygon_count"],
            "missing_image_count": len(audit["missing_images"]),
            "corrupt_image_count": len(audit["corrupt_images"]),
            "empty_ref_mask_count": len(audit["empty_ref_masks"]),
            "empty_ref_explanation_count": len(audit["empty_ref_explanations"]),
            "empty_union_mask_count": len(audit["empty_union_masks"]),
            "out_of_bounds_polygon_count": len(audit["out_of_bounds_polygons"]),
            "boundary_roundoff_polygon_count": len(audit["boundary_roundoff_polygons"]),
            "visualization_count": len(visualization_index),
        }
    atomic_write_json(output_dir / "summary.json", combined_summary)
    print(json.dumps(combined_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
