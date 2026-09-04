#!/usr/bin/env python3
"""Freeze manifests for already-present final-evaluation datasets.

This script performs data inventory/integrity work only.  It never imports or
runs a model and never computes a performance metric.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "datasets").resolve()
OUT = ROOT / "outputs/final_eval_datasets"
DATE = "2026-09-03"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path, cached: dict | None = None) -> tuple[int, int]:
    if cached and cached.get("width") and cached.get("height"):
        return int(cached["width"]), int(cached["height"])
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"cannot decode {path}")
        return int(image.shape[1]), int(image.shape[0])
    with Image.open(path) as image:
        return image.size


def base_row(sample_id: str, dataset: str, split: str, image: Path, label: str,
             source: str | None, annotation: Path | None, annotation_type: str,
             cached: dict | None = None, known_sha: str | None = None, **extra) -> dict:
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    width, height = image_size(image, cached)
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "split": split,
        "image_path": str(image),
        "label": label,
        "generator/source": source,
        "gt_annotation_path": str(annotation.resolve()) if annotation else None,
        "annotation_type": annotation_type,
        "sha256": known_sha or sha256(image),
        "width": width,
        "height": height,
        **extra,
    }


def phash_cache() -> dict[str, dict]:
    path = ROOT / "outputs/data_audits/image_phash_v1/phash_records.jsonl"
    return {row["sample_id"]: row for row in read_jsonl(path)}


def synthscars(cache: dict[str, dict]) -> None:
    source_manifest = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
    rows = []
    annotation = DATA / "SynthScars/test/annotations/test.json"
    for item in read_jsonl(source_manifest):
        image = Path(item["image_path"])
        rows.append(base_row(
            item["sample_id"], "SynthScars", "official_test", image, "Fake",
            item.get("generator") or "SynthScars", annotation, "official polygons",
            cache.get(item["sample_id"]), gt_type="polygon",
            original_relative_path=item["image_relpath"],
            official_annotation_ids=item["annotation_ids"],
        ))
    if len(rows) != 1000 or len({row["sample_id"] for row in rows}) != 1000:
        raise RuntimeError("SynthScars Official1000 population drift")
    target = DATA / "SynthScars/manifests/eval_manifest.jsonl"
    atomic_jsonl(target, rows)

    archive = DATA / ".downloads/SynthScars/SynthScars.zip"
    mismatches = []
    with zipfile.ZipFile(archive) as bundle:
        members = {info.filename: info for info in bundle.infolist() if not info.is_dir()}
        expected = ["SynthScars/test/annotations/test.json"] + [
            "SynthScars/test/images/" + Path(row["image_path"]).name for row in rows
        ]
        for member in expected:
            local = DATA / member
            info = members.get(member)
            crc = 0
            with local.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    crc = zlib.crc32(chunk, crc)
            if info is None or info.file_size != local.stat().st_size or info.CRC != (crc & 0xFFFFFFFF):
                mismatches.append(member)
    atomic_json(OUT / "synthscars_archive_integrity.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "expected_test_files": 1001,
        "mismatches": mismatches,
    })
    if mismatches:
        raise RuntimeError(f"SynthScars archive mismatch: {mismatches[:3]}")
    atomic_json(DATA / "SynthScars/manifests/provenance.json", {
        "dataset": "SynthScars",
        "official_source": "https://huggingface.co/datasets/khr0516/SynthScars",
        "download_date": "2026-03-21 (archive mtime; audited 2026-09-03)",
        "release_or_revision": "ee1cd553c2403f551fb1da60745cb2cdcb975e74",
        "paper": "LEGION: Learning to Ground and Explain for Synthetic Image Detection",
        "split_used": "official test",
        "raw_root": str((DATA / "SynthScars/raw").resolve()),
        "sample_count": {"Fake": 1000, "total": 1000},
        "annotation_type": "official per-reference polygons (union is evaluator-derived)",
        "archive_sha256": sha256(archive),
        "source_manifest_sha256": sha256(source_manifest),
        "eval_manifest_sha256": sha256(target),
        "notes": "Existing extraction matches the official archive by member size and ZIP CRC for all 1000 test images plus test.json. Train remains for historical training dependencies but is excluded from this evaluation manifest.",
    })


def internal2208(cache: dict[str, dict]) -> None:
    source_manifest = ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl"
    rows = []
    for item in read_jsonl(source_manifest):
        candidate = item.get("image_path")
        if candidate and Path(candidate).is_file():
            image = Path(candidate)
        elif int(item["class_label"]):
            image = DATA / "SynthScars" / item["image_relpath"]
        else:
            image = DATA / item["image_relpath"]
        rows.append(base_row(
            item["sample_id"], "Internal2208", "frozen_internal_test", image,
            "Fake" if int(item["class_label"]) else "Real", item.get("source"), None,
            "classification label", cache.get(item["sample_id"]),
            known_sha=item.get("content_sha256"), original_relative_path=item["image_relpath"],
        ))
    counts = Counter(row["label"] for row in rows)
    if len(rows) != 2208 or counts != Counter({"Real": 1104, "Fake": 1104}):
        raise RuntimeError(f"Internal2208 drift: {len(rows)} {counts}")
    directory = DATA / "Internal2208/manifests"
    target = directory / "eval_manifest.jsonl"
    atomic_jsonl(target, rows)
    atomic_json(directory / "provenance.json", {
        "dataset": "Internal2208",
        "official_source": "project frozen unified-forensics split",
        "download_date": None,
        "release_or_revision": "unified_forensics_split_v1",
        "paper": None,
        "split_used": "frozen internal test",
        "raw_root": str(DATA),
        "sample_count": {"Real": 1104, "Fake": 1104, "total": 2208},
        "annotation_type": "binary class label",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "eval_manifest_sha256": sha256(target),
        "notes": "Existing frozen test is recorded without modification.",
    })


def loki(cache: dict[str, dict]) -> None:
    root = (ROOT / "datasets/LOKI").resolve()
    raw_annotation = root / "open_ended_vqa.json"
    loc_source = root / "legion_localization/manifest.jsonl"
    loc_rows = []
    box_count = 0
    for item in read_jsonl(loc_source):
        box_count += len(item["regions"])
        loc_rows.append(base_row(
            item["sample_id"], "LOKI", "frozen_localization_229", Path(item["image_path"]),
            "Fake", "LOKI", raw_annotation, "official bounding boxes",
            cache.get(item["sample_id"]), gt_type="bounding_box",
            original_relative_path=item["image_relpath"],
            official_annotation_id=item["source_annotation_id"],
            boxes_xywh=[region["bbox_xywh"] for region in item["regions"]],
            derived_union_mask_path=item["mask_path"],
        ))
    if len(loc_rows) != 229 or box_count != 687:
        raise RuntimeError(f"LOKI localization drift: {len(loc_rows)} images/{box_count} boxes")
    manifest_dir = root / "manifests"
    atomic_jsonl(manifest_dir / "localization_eval_manifest.jsonl", loc_rows)

    records = json.loads((root / "true_or_false.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for item in records:
        if item.get("modality") == "image-text":
            grouped[item["image_path"]].append(item)
    cls_rows = []
    for relative, questions in sorted(grouped.items()):
        labels = []
        for item in questions:
            answer = item["answer"].strip().lower()
            if "ask_fake" in item["question_type"]:
                labels.append(answer == "yes")
            elif "ask_real" in item["question_type"]:
                labels.append(answer == "no")
        if not labels or len(set(labels)) != 1:
            raise RuntimeError(f"ambiguous LOKI label: {relative}")
        sample_id = f"loki:{relative}"
        cls_rows.append(base_row(
            sample_id, "LOKI", "official_image_true_or_false", root / relative,
            "Fake" if labels[0] else "Real", "LOKI", root / "true_or_false.json",
            "binary class label", cache.get(sample_id), original_relative_path=relative,
            official_question_ids=[item["id"] for item in questions],
        ))
    cls_counts = Counter(row["label"] for row in cls_rows)
    if len(cls_rows) != 2217 or cls_counts != Counter({"Fake": 1317, "Real": 900}):
        raise RuntimeError(f"LOKI classification drift: {len(cls_rows)} {cls_counts}")
    atomic_jsonl(manifest_dir / "classification_eval_manifest.jsonl", cls_rows)
    atomic_json(manifest_dir / "provenance.json", {
        "dataset": "LOKI",
        "official_source": "https://huggingface.co/datasets/bczhou/LOKI",
        "official_code": "https://github.com/opendatalab/LOKI",
        "download_date": "existing extraction audited 2026-09-03",
        "release_or_revision": {"huggingface": "314ddacc5080b024d6b8d962b448065cd54c9f42", "git": "9b2dac636e660aa5fd158be7888abaf3dd268140"},
        "paper": "LOKI: A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models",
        "split_used": ["official image true-or-false", "frozen 229-image open-ended localization subset"],
        "raw_root": str(root / "raw"),
        "sample_count": {"classification": {"Real": 900, "Fake": 1317, "total": 2217}, "localization": {"Fake": 229, "boxes": 687}},
        "annotation_type": "classification question pairs; localization xywh bounding boxes",
        "annotation_sha256": {"true_or_false.json": sha256(root / "true_or_false.json"), "open_ended_vqa.json": sha256(raw_annotation)},
        "notes": "Raw official boxes are retained. Rasterized union masks remain derived data only. The official media archive is being reacquired solely to verify the existing media extraction.",
    })


def raise998(cache: dict[str, dict]) -> None:
    source_manifest = ROOT / "outputs/data_audits/unified_forensics_split_v1/raise_heldout_real_clean.jsonl"
    rows = []
    for item in read_jsonl(source_manifest):
        rows.append(base_row(
            item["sample_id"], "RAISE", "frozen_raise_998", DATA / item["image_relpath"],
            "Real", "RAISE-1k", None, "real-only class label", cache.get(item["sample_id"]),
            known_sha=item["content_sha256"], original_relative_path=item["image_relpath"],
        ))
    if len(rows) != 998 or len({row["sample_id"] for row in rows}) != 998:
        raise RuntimeError("RAISE998 population drift")
    manifest_dir = DATA / "RAISE/manifests"
    target = manifest_dir / "eval_manifest.jsonl"
    atomic_jsonl(target, rows)
    atomic_json(manifest_dir / "provenance.json", {
        "dataset": "RAISE",
        "official_source": "https://loki.disi.unitn.it/RAISE/download.html",
        "download_date": "existing official TIFF downloads audited 2026-09-03",
        "release_or_revision": "RAISE-1k official TIFF selection manifest",
        "paper": "RAISE - A Raw Images Dataset for Digital Image Forensics",
        "split_used": "historically frozen 998 decodable images from the official 1000-image RAISE-1k selection",
        "raw_root": str((DATA / "RAISE/raw").resolve()),
        "sample_count": {"Real": 998, "Fake": 0, "total": 998},
        "annotation_type": "real-only identity",
        "source_selection_manifest": str(DATA / "RAISE-1k/manifests/selected.jsonl"),
        "source_selection_manifest_sha256": sha256(DATA / "RAISE-1k/manifests/selected.jsonl"),
        "eval_manifest_sha256": sha256(target),
        "excluded_corrupt_ids": ["real:RAISE-1k:r0515a051t", "real:RAISE-1k:r0bf7f938t"],
        "notes": "No resampling. The same frozen 998 identities are retained; the two historically documented undecodable TIFFs remain outside the evaluator manifest.",
    })


def main() -> None:
    cache = phash_cache()
    synthscars(cache)
    internal2208(cache)
    loki(cache)
    raise998(cache)
    atomic_json(OUT / "existing_datasets_complete.json", {
        "status": "COMPLETE",
        "datasets": ["SynthScars Official1000", "Internal2208", "LOKI localization/classification", "RAISE998"],
        "model_evaluation_run": False,
    })
    print("existing dataset manifests: COMPLETE")


if __name__ == "__main__":
    main()
