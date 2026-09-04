#!/usr/bin/env python3
"""Freeze official downloaded final-evaluation splits after extraction."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "datasets").resolve()
OUT = ROOT / "outputs/final_eval_datasets"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def inspect(path: Path) -> tuple[str, int, int]:
    digest = sha256(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        image.verify()
    return digest, int(width), int(height)


def enrich(rows: list[dict], workers: int = 12) -> list[dict]:
    def one(row: dict) -> dict:
        digest, width, height = inspect(Path(row["image_path"]))
        return {**row, "sha256": digest, "width": width, "height": height}
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(one, rows), 1):
            completed.append(row)
            if index % 5000 == 0 or index == len(rows):
                print(f"hash/decode {index}/{len(rows)}", flush=True)
    return completed


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def freeze_aigi() -> dict:
    dataset = DATA / "AIGI-Holmes"
    extracted = dataset / "raw/extracted/TestSet"
    files = image_files(extracted)
    rows = []
    for path in files:
        relative = path.relative_to(extracted)
        if len(relative.parts) != 3 or relative.parts[1] not in {"0_real", "1_fake"}:
            raise RuntimeError(f"unexpected AIGI-Holmes member: {relative}")
        generator, label_dir, _ = relative.parts
        rows.append({
            "sample_id": f"aigi-holmes:test:{generator}:{label_dir}:{relative.name}",
            "dataset": "AIGI-Holmes", "split": "official_TestSet",
            "image_path": str(path.resolve()), "label": "Fake" if label_dir == "1_fake" else "Real",
            "generator/source": generator, "gt_annotation_path": None,
            "annotation_type": "official directory binary class label",
            "original_relative_path": str(Path("TestSet") / relative),
        })
    rows = enrich(rows)
    counts = Counter(row["label"] for row in rows)
    if len(rows) != 99_999 or counts != Counter({"Real": 50_000, "Fake": 49_999}):
        raise RuntimeError(f"AIGI-Holmes population drift: {len(rows)} {counts}")
    target = dataset / "manifests/eval_manifest.jsonl"
    atomic_jsonl(target, rows)
    provenance = {
        "dataset": "AIGI-Holmes", "official_source": "https://huggingface.co/datasets/zzy0123/AIGI-Holmes-Dataset",
        "download_date": "2026-09-03", "release_or_revision": "3e856ce5ed44ac3b578bf36434829ea42953be02",
        "paper": "AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models",
        "split_used": "official TestSet", "raw_root": str((dataset / "raw").resolve()),
        "sample_count": {**counts, "total": len(rows), "per_generator_and_label": dict(Counter((r["generator/source"], r["label"]) for r in rows))},
        "annotation_type": "official directory binary class label", "archive": str((dataset / "raw/TestSet.zip").resolve()),
        "archive_bytes": (dataset / "raw/TestSet.zip").stat().st_size, "archive_sha256": sha256(dataset / "raw/TestSet.zip"),
        "eval_manifest_sha256": sha256(target), "notes": "No resampling or cleaning; official TestSet only. One Janus fake is absent in the pinned archive, so actual N=99,999 is retained.",
    }
    # JSON object keys cannot be tuples.
    provenance["sample_count"]["per_generator_and_label"] = {
        f"{generator}/{label}": count for (generator, label), count in Counter((r["generator/source"], r["label"]) for r in rows).items()
    }
    atomic_json(dataset / "manifests/provenance.json", provenance)
    return {"n": len(rows), "counts": dict(counts), "manifest": str(target)}


def freeze_genimage() -> dict:
    dataset = DATA / "GenImage"
    extracted = dataset / "raw/extracted/test"
    files = image_files(extracted)
    rows = []
    for path in files:
        relative = path.relative_to(extracted)
        if len(relative.parts) != 3 or relative.parts[1] not in {"ai", "nature"}:
            raise RuntimeError(f"unexpected GenImage member: {relative}")
        generator_partition, label_dir, _ = relative.parts
        generator = generator_partition.removesuffix("_imagenet")
        rows.append({
            "sample_id": f"genimage:test:{generator_partition}:{label_dir}:{relative.name}",
            "dataset": "GenImage", "split": "official_test_heldout",
            "image_path": str(path.resolve()), "label": "Fake" if label_dir == "ai" else "Real",
            "generator/source": generator, "gt_annotation_path": None,
            "annotation_type": "official directory binary class label",
            "original_relative_path": str(Path("test") / relative), "official_split_name": "test",
        })
    rows = enrich(rows)
    counts = Counter(row["label"] for row in rows)
    if len(rows) != 100_000 or counts != Counter({"Real": 50_000, "Fake": 50_000}):
        raise RuntimeError(f"GenImage population drift: {len(rows)} {counts}")
    target = dataset / "manifests/eval_manifest.jsonl"
    atomic_jsonl(target, rows)
    per = Counter((r["generator/source"], r["label"]) for r in rows)
    archive = dataset / "raw/huggingface_test/genimage_test.zip"
    atomic_json(dataset / "manifests/provenance.json", {
        "dataset": "GenImage", "official_source": "https://huggingface.co/datasets/jzousz/GenImage",
        "download_date": "2026-09-04", "release_or_revision": "71c983e6262684bc2c6b6af99582e8f568c259a5",
        "paper": "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image",
        "split_used": "pinned official test held-out archive", "raw_root": str((dataset / "raw").resolve()),
        "sample_count": {**dict(counts), "total": len(rows), "per_generator_and_label": {f"{a}/{b}": n for (a,b),n in per.items()}},
        "annotation_type": "official directory binary class label", "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
        "eval_manifest_sha256": sha256(target), "notes": "No train data and no project resplitting; generator identity is retained.",
    })
    return {"n": len(rows), "counts": dict(counts), "manifest": str(target)}


def freeze_pal4vst() -> dict:
    dataset = DATA / "PAL4VST"
    extracted = dataset / "raw/extracted"
    rows = []
    task_counts = Counter()
    for image_root in sorted(extracted.glob("*/images/test")):
        task = image_root.parents[1].name
        label_root = image_root.parents[1] / "labels/test"
        for image in image_files(image_root):
            candidates = [label_root / (image.stem + suffix) for suffix in (".png", ".jpg", ".jpeg")]
            labels = [path for path in candidates if path.is_file()]
            if len(labels) != 1:
                raise RuntimeError(f"PAL4VST GT mapping is not unique: {image} -> {labels}")
            rows.append({
                "sample_id": f"pal4vst:test:{task}:{image.name}", "dataset": "PAL4VST", "split": "official_test",
                "image_path": str(image.resolve()), "label": "Fake", "generator/source": task,
                "gt_annotation_path": str(labels[0].resolve()), "annotation_type": "official pixel artifact mask",
                "gt_type": "pixel_mask", "original_relative_path": str(image.relative_to(extracted)),
                "original_gt_relative_path": str(labels[0].relative_to(extracted)),
            })
            task_counts[task] += 1
    rows = enrich(rows)
    if not rows or len(rows) != sum(task_counts.values()):
        raise RuntimeError("PAL4VST official test is empty")
    target = dataset / "manifests/eval_manifest.jsonl"
    atomic_jsonl(target, rows)
    archive = dataset / "raw/specific_tasks.zip"
    atomic_json(dataset / "manifests/provenance.json", {
        "dataset": "PAL4VST", "official_source": "https://github.com/owenzlz/PAL4VST and author Google Drive specific_tasks.zip",
        "download_date": "2026-09-03", "release_or_revision": "GitHub 9db472581715024e4fb69af7ffd2d64b5230bbef; Drive file 1h2geaBGrQVNKrjNPUs0oWdE5vXhdwTn_",
        "paper": "Perceptual Artifacts Localization for Image Synthesis Tasks", "split_used": "official test only",
        "raw_root": str((dataset / "raw").resolve()), "sample_count": {"Fake": len(rows), "total": len(rows), "per_task": dict(task_counts)},
        "annotation_type": "official pixel artifact masks", "archive": str(archive.resolve()), "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive), "eval_manifest_sha256": sha256(target),
        "notes": "The full official archive contains train/val/test; the frozen evaluator manifest includes test only and preserves raw images/masks.",
    })
    return {"n": len(rows), "counts": {"Fake": len(rows)}, "tasks": dict(task_counts), "manifest": str(target)}


def main() -> None:
    extraction = json.loads((OUT / "extraction_status.json").read_text())
    if extraction.get("status") != "COMPLETE":
        raise RuntimeError(f"extraction gate not complete: {extraction}")
    status = {"schema": "final_eval_downloaded_prepare_v1", "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    atomic_json(OUT / "downloaded_prepare_status.json", status)
    try:
        results = {"AIGI-Holmes": freeze_aigi(), "GenImage": freeze_genimage(), "PAL4VST": freeze_pal4vst()}
        status.update({"status": "COMPLETE", "datasets": results})
    except BaseException as error:
        status.update({"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error)})
        raise
    finally:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(OUT / "downloaded_prepare_status.json", status)


if __name__ == "__main__":
    main()
