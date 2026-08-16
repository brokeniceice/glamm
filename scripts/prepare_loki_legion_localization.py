#!/usr/bin/env python3
"""Prepare the 229-sample LOKI localization scope used by LEGION Table 2.

LOKI releases regional artifact boxes rather than pixel-level masks.  LEGION
rasterizes every ``problems.regional[].region`` xywh box as a filled rectangle
and unions the rectangles for image-level localization evaluation.  This tool
reproduces that public conversion while preserving the source annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--loki-root", default="datasets/LOKI")
    parser.add_argument("--annotation", default="open_ended_vqa.json")
    parser.add_argument("--output-dir", default="legion_localization")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv=None):
    cli = parse_args(argv)
    loki_root = (ROOT / cli.loki_root).resolve()
    annotation_path = (loki_root / cli.annotation).resolve()
    output_dir = (loki_root / cli.output_dir).resolve()
    rows = json.loads(annotation_path.read_bytes().decode("utf-16"))
    if len(rows) != 229:
        raise ValueError(f"Expected the official 229-row LOKI scope, got {len(rows)}")

    manifest = []
    mask_hashes = {}
    total_boxes = 0
    for index, row in enumerate(rows):
        relative_image = Path(row["image_path"])
        image_path = (loki_root / relative_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size

        regions = row.get("problems", {}).get("regional", [])
        if not regions:
            raise ValueError(f"{relative_image}: no regional annotation")
        union = np.zeros((height, width), dtype=np.uint8)
        normalized_regions = []
        for region_index, region in enumerate(regions):
            bbox = region.get("region") or []
            if len(bbox) != 4:
                raise ValueError(f"{relative_image}: invalid region {bbox!r}")
            x, y, box_width, box_height = [float(value) for value in bbox]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(width - 1, int(x + box_width))
            y2 = min(height - 1, int(y + box_height))
            if x2 < x1 or y2 < y1:
                raise ValueError(f"{relative_image}: empty clipped region {bbox!r}")
            # Exact public LEGION semantics: filled, inclusive OpenCV rectangle.
            cv2.rectangle(union, (x1, y1), (x2, y2), 1, thickness=-1)
            normalized_regions.append({
                "region_index": region_index,
                "bbox_xywh": [x, y, box_width, box_height],
                "rasterized_xyxy_inclusive": [x1, y1, x2, y2],
                "description": " ".join(str(region.get("desc") or "").split()),
            })
        if not union.any():
            raise ValueError(f"{relative_image}: derived union mask is empty")

        stem = image_path.stem
        mask_path = output_dir / "masks" / f"{stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), union * 255):
            raise OSError(f"Failed to write {mask_path}")
        mask_hashes[mask_path.name] = sha256(mask_path)
        total_boxes += len(normalized_regions)
        manifest.append({
            "schema_version": "loki_legion_bbox_union_v1",
            "sample_id": f"loki:localization:{stem}",
            "index": index,
            "image_path": str(image_path),
            "image_relpath": str(relative_image),
            "mask_path": str(mask_path),
            "image_size_hw": [height, width],
            "mask_foreground_pixels": int(union.sum()),
            "mask_fraction": float(union.mean()),
            "regions": normalized_regions,
            "global_description": " ".join(str(
                (row.get("problems", {}).get("global") or [{}])[0].get("desc") or ""
            ).split()),
            "source_annotation_id": row.get("id"),
            "source": "LOKI",
            "gt_semantics": "union of filled regional xywh bounding boxes, matching LEGION generate_loki_mask",
        })

    write_jsonl(output_dir / "manifest.jsonl", manifest)
    classification_path = loki_root / "true_or_false.json"
    classification_rows = json.loads(classification_path.read_text(encoding="utf-8"))
    classification_images = {str(row["image_path"]) for row in classification_rows}
    localization_images = {str(row["image_path"]) for row in rows}
    summary = {
        "schema_version": "loki_legion_bbox_union_v1",
        "loki_root": str(loki_root),
        "source_annotation": str(annotation_path),
        "source_annotation_sha256": sha256(annotation_path),
        "source_annotation_encoding": "UTF-16",
        "num_images": len(manifest),
        "num_regional_boxes": total_boxes,
        "classification_unique_images": len(classification_images),
        "localization_classification_image_overlap": len(
            localization_images & classification_images
        ),
        "localization_images_absent_from_classification_scope": len(
            localization_images - classification_images
        ),
        "missing_images": 0,
        "empty_masks": 0,
        "mask_semantics": "filled inclusive rectangles from problems.regional[].region xywh; per-image union",
        "pixel_level_human_masks_available": False,
        "legion_table2_comparison_scope": True,
        "official_loki_commit": "9b2dac636e660aa5fd158be7888abaf3dd268140",
        "audited_legion_commit": "d21535dd45f6fea509337a83095966f0b86ac924",
        "mask_sha256": mask_hashes,
    }
    dump_json(output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "mask_sha256"}, indent=2))


if __name__ == "__main__":
    main()
