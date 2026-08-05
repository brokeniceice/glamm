#!/usr/bin/env python3
"""Compute deterministic pHash values and candidate near-duplicate pairs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/real_images_unified_v1/candidate_train_manifest.jsonl",
    )
    parser.add_argument(
        "--real-heldout-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/real_images_unified_v1/heldout_test_manifest.jsonl",
    )
    parser.add_argument(
        "--synth-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/synthscars_image_grouped_v1/train_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/image_phash_v1",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hamming-threshold", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def load_complete_cache(path: Path, expected_count: int) -> list | None:
    if not path.is_file():
        return None
    try:
        records = load_jsonl(path)
    except Exception:
        return None
    return records if len(records) == expected_count else None


def build_items(datasets_root: Path, real_manifest: Path, heldout_manifest: Path, synth_manifest: Path) -> list:
    datasets_root = datasets_root.expanduser().resolve()
    items = []
    for manifest, domain in ((real_manifest, "real_candidate"), (heldout_manifest, "real_heldout")):
        for row in load_jsonl(manifest):
            items.append({
                "sample_id": row["sample_id"],
                "domain": domain,
                "source": row["source"],
                "image_path": str(datasets_root / row["image_relpath"]),
            })
    for row in load_jsonl(synth_manifest):
        items.append({
            "sample_id": row["sample_id"],
            "domain": "synthscars_fake",
            "source": "SynthScars",
            "image_path": str(datasets_root / "SynthScars" / row["image_relpath"]),
        })
    return items


def perceptual_hash(path: str) -> tuple[str, int, int]:
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OSError(f"OpenCV could not decode TIFF: {path}")
        height, width = gray.shape
        resized_array = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            width, height = image.size
            resized = image.resize((32, 32), Image.Resampling.LANCZOS)
        resized_array = np.asarray(resized, dtype=np.float32)
    coefficients = cv2.dct(resized_array)[:8, :8]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    bits = coefficients >= threshold
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}", width, height


def hash_one(item: Mapping[str, Any]) -> Dict[str, Any]:
    phash, width, height = perceptual_hash(item["image_path"])
    return {**item, "phash64": phash, "width": width, "height": height}


def hash_one_safe(item: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return hash_one(item)
    except Exception as error:
        return {**item, "phash64": None, "error": f"{type(error).__name__}: {error}"}


def chunk_values(value: int) -> Sequence[int]:
    # Five disjoint chunks guarantee at least one identical chunk for pairs
    # with Hamming distance <= 4 (pigeonhole principle).
    widths = (13, 13, 13, 13, 12)
    chunks, shift = [], 0
    for width in widths:
        chunks.append((value >> shift) & ((1 << width) - 1))
        shift += width
    return chunks


def near_duplicate_pairs(records: Sequence[Mapping[str, Any]], threshold: int) -> list:
    records = [record for record in records if record.get("phash64")]
    values = [int(record["phash64"], 16) for record in records]
    buckets: Dict[tuple, list] = defaultdict(list)
    for index, value in enumerate(values):
        for chunk_index, chunk in enumerate(chunk_values(value)):
            buckets[(chunk_index, chunk)].append(index)
    candidate_pairs = set()
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                candidate_pairs.add((min(left, right), max(left, right)))
    pairs = []
    for left, right in sorted(candidate_pairs):
        distance = (values[left] ^ values[right]).bit_count()
        if distance <= threshold:
            pairs.append({
                "left_sample_id": records[left]["sample_id"],
                "right_sample_id": records[right]["sample_id"],
                "left_domain": records[left]["domain"],
                "right_domain": records[right]["domain"],
                "left_source": records[left]["source"],
                "right_source": records[right]["source"],
                "hamming_distance": distance,
            })
    return pairs


def write_pair_visualizations(records: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], output_dir: Path) -> list:
    by_id = {record["sample_id"]: record for record in records}
    visual_dir = output_dir / "pair_visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for pair_index, pair in enumerate(pairs):
        canvas = Image.new("RGB", (1024, 560), "white")
        for side_index, key in enumerate(("left_sample_id", "right_sample_id")):
            record = by_id[pair[key]]
            try:
                with Image.open(record["image_path"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                x = side_index * 512 + (512 - image.width) // 2
                y = (500 - image.height) // 2
                canvas.paste(image, (x, y))
            except Exception:
                pass
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 505), f"d={pair['hamming_distance']} {pair['left_sample_id']}", fill="black")
        draw.text((517, 505), pair["right_sample_id"], fill="black")
        output_name = f"pair_{pair_index:03d}.jpg"
        canvas.save(visual_dir / output_name, quality=88)
        index.append({**pair, "output_file": str(Path("pair_visualizations") / output_name)})
    return index


def main() -> None:
    args = parse_args()
    items = build_items(args.datasets_root, args.real_manifest, args.real_heldout_manifest, args.synth_manifest)
    missing = [item["image_path"] for item in items if not Path(item["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[0]}")
    output_dir = args.output_dir.expanduser().resolve()
    regular_items = [item for item in items if Path(item["image_path"]).suffix.lower() not in {".tif", ".tiff"}]
    tiff_items = [item for item in items if Path(item["image_path"]).suffix.lower() in {".tif", ".tiff"}]
    regular_cache = output_dir / "regular_phash_records.jsonl"
    tiff_cache = output_dir / "tiff_phash_records.jsonl"
    regular_records = load_complete_cache(regular_cache, len(regular_items))
    if regular_records is None:
        regular_records = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, record in enumerate(executor.map(hash_one, regular_items), start=1):
                regular_records.append(record)
                if index % 2000 == 0 or index == len(regular_items):
                    print(f"pHash regular: {index}/{len(regular_items)}", flush=True)
        atomic_write_jsonl(regular_cache, regular_records)
    else:
        print(f"pHash regular: reused {len(regular_records)} cached records", flush=True)

    tiff_records = load_complete_cache(tiff_cache, len(tiff_items))
    if tiff_records is None:
        tiff_records = []
        # Spawn isolates libtiff from the prior threaded JPEG/PNG phase. The
        # worker is recycled frequently because this libtiff build becomes
        # unreliable after a long sequence of high-resolution TIFF decodes.
        with mp.get_context("spawn").Pool(processes=1, maxtasksperchild=20) as pool:
            for tiff_index, record in enumerate(pool.imap(hash_one_safe, tiff_items, chunksize=1), start=1):
                tiff_records.append(record)
                if tiff_index % 100 == 0 or tiff_index == len(tiff_items):
                    print(f"pHash TIFF: {tiff_index}/{len(tiff_items)}", flush=True)
        atomic_write_jsonl(tiff_cache, tiff_records)
    else:
        print(f"pHash TIFF: reused {len(tiff_records)} cached records", flush=True)
    records = regular_records + tiff_records
    pairs = near_duplicate_pairs(records, args.hamming_threshold)
    failures = [record for record in records if not record.get("phash64")]
    atomic_write_jsonl(output_dir / "phash_records.jsonl", records)
    atomic_write_jsonl(output_dir / "near_duplicate_pairs.jsonl", pairs)
    visualizations = write_pair_visualizations(records, pairs, output_dir)
    pair_types = Counter()
    for pair in pairs:
        left, right = pair["left_domain"], pair["right_domain"]
        pair_type = f"within_{left}" if left == right else "cross_" + "__".join(sorted((left, right)))
        pair_types[pair_type] += 1
    summary = {
        "schema_version": "image_phash_v1",
        "algorithm": "32x32 grayscale DCT, top-left 8x8 median threshold, 64 bits",
        "hamming_threshold": args.hamming_threshold,
        "record_count": len(records),
        "domain_counts": dict(Counter(record["domain"] for record in records)),
        "near_duplicate_pair_count": len(pairs),
        "decode_failure_count": len(failures),
        "decode_failures": [
            {"sample_id": record["sample_id"], "image_path": record["image_path"], "error": record["error"]}
            for record in failures
        ],
        "pair_type_counts": dict(pair_types),
        "pair_visualization_count": len(visualizations),
        "note": "pHash pairs are review candidates and are never auto-deleted.",
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
