#!/usr/bin/env python3
"""Materialize and freeze X-AIGD's pinned official labeled_test parquet."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = (ROOT / "datasets/X-AIGD").resolve()
PARQUET = DATASET / "raw/labeled_test-00000-of-00001.parquet"
EXPECTED_BYTES = 3_488_049_189
REVISION = "92180f32030507ab54a40d6f1b88f39d6cec8178"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    if PARQUET.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"incomplete X-AIGD parquet: {PARQUET.stat().st_size}/{EXPECTED_BYTES}")
    images = DATASET / "raw/images"
    images.mkdir(parents=True, exist_ok=True)
    annotation_rows = []
    manifest = []
    generators = Counter()
    artifact_categories = Counter()
    parquet = pq.ParquetFile(PARQUET)
    index = 0
    for batch in parquet.iter_batches(batch_size=16):
        for record in batch.to_pylist():
            image_record = record.pop("image")
            payload = image_record.get("bytes")
            if payload is None:
                raise RuntimeError(f"row {index} has no embedded image bytes")
            generator = str(record["generator"])
            uid = str(record["uid"])
            extension = ".png" if str(record.get("image_format", "PNG")).upper() == "PNG" else ".jpg"
            relative = Path("images") / safe(generator) / f"{safe(uid)}{extension}"
            output = DATASET / "raw" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if sha256_file(output) != sha256_bytes(payload):
                    raise RuntimeError(f"existing extracted image differs: {output}")
            else:
                output.write_bytes(payload)
            with Image.open(io.BytesIO(payload)) as decoded:
                width, height = decoded.size
            if (width, height) != (int(record["width"]), int(record["height"])):
                raise RuntimeError(f"dimension mismatch at row {index}")
            labels = record.get("labels") or []
            categories = [str(label["label"]) for label in labels]
            polygon_count = sum(len(label.get("points") or []) > 0 for label in labels)
            generators[generator] += 1
            artifact_categories.update(categories)
            sample_id = f"x-aigd:labeled_test:{safe(generator)}:{safe(uid)}"
            annotation_rows.append({
                "sample_id": sample_id,
                "official_parquet_row": index,
                **record,
            })
            manifest.append({
                "sample_id": sample_id,
                "dataset": "X-AIGD",
                "split": "labeled_test",
                "image_path": str(output.resolve()),
                "label": "Fake",
                "generator/source": generator,
                "gt_annotation_path": str(PARQUET.resolve()),
                "annotation_type": "official human artifact polygons",
                "gt_type": "polygon",
                "sha256": sha256_bytes(payload),
                "width": width,
                "height": height,
                "original_relative_path": str(relative),
                "official_uid": uid,
                "official_parquet_row": index,
                "artifact_categories": categories,
                "polygon_count": polygon_count,
            })
            index += 1
            if index % 250 == 0:
                print(f"X-AIGD {index}/2419", flush=True)
    if len(manifest) != 2419 or len({row["sample_id"] for row in manifest}) != 2419:
        raise RuntimeError(f"X-AIGD population/identity drift: {len(manifest)}")
    annotation_target = DATASET / "processed/labeled_test_records.jsonl"
    manifest_target = DATASET / "manifests/eval_manifest.jsonl"
    atomic_jsonl(annotation_target, annotation_rows)
    atomic_jsonl(manifest_target, manifest)
    atomic_json(DATASET / "manifests/provenance.json", {
        "dataset": "X-AIGD",
        "official_source": "https://huggingface.co/datasets/Coxy7/X-AIGD",
        "download_date": "2026-09-03",
        "release_or_revision": REVISION,
        "paper": "Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection",
        "split_used": "labeled_test",
        "raw_root": str((DATASET / "raw").resolve()),
        "sample_count": {"Fake": 2419, "total": 2419, "per_generator": dict(sorted(generators.items()))},
        "annotation_type": "human artifact polygons embedded in official parquet",
        "official_parquet_bytes": PARQUET.stat().st_size,
        "official_parquet_sha256": sha256_file(PARQUET),
        "eval_manifest_sha256": sha256_file(manifest_target),
        "artifact_category_counts": dict(sorted(artifact_categories.items())),
        "notes": "Only labeled_test was downloaded. The official parquet is immutable raw data. Extracted image files preserve the embedded encoded bytes exactly. No polygon was rasterized; the JSONL under processed is a reversible row-level convenience copy.",
    })
    print("X-AIGD manifest: COMPLETE")


if __name__ == "__main__":
    main()
