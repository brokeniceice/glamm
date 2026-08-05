#!/usr/bin/env python3
"""Select and download a reproducible subset of PASS v3 from its official URL list."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/PASS"))
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def load_metadata(path):
    with path.open(encoding="utf-8") as handle:
        return {row["hash"]: row for row in csv.DictReader(handle)}


def select_records(root, count, seed):
    metadata = load_metadata(root / "metadata" / "pass_metadata.csv")
    records = []
    seen = set()
    with (root / "metadata" / "pass_urls.txt").open(encoding="utf-8") as handle:
        for url in handle:
            url = url.strip()
            image_hash = Path(url).stem
            if image_hash in metadata and image_hash not in seen:
                record = dict(metadata[image_hash])
                record.update({"image_hash": image_hash, "download_url": url})
                records.append(record)
                seen.add(image_hash)
    if len(records) < count:
        raise RuntimeError(f"Only {len(records)} URL/metadata pairs, need {count}")
    random.Random(seed).shuffle(records)
    return records[:count], len(records)


def write_manifest(root, records, candidate_count, seed):
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "PASS v3.0",
        "doi": "10.5281/zenodo.6615455",
        "license": "CC BY 4.0",
        "seed": seed,
        "candidate_count": candidate_count,
        "selected_count": len(records),
    }
    with (manifests_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_image(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(record, images_dir):
    expected = record["image_hash"]
    destination = images_dir / f"{expected}.jpg"
    if valid_image(destination):
        return "existing", expected
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(record["download_url"], headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if not valid_image(temporary):
                raise OSError("downloaded file is not a valid image")
            temporary.replace(destination)
            return "downloaded", expected
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(attempt + 1)
    return "failed", expected, str(last_error)


def download_all(root, records, workers):
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, record, images_dir) for record in records]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result[0]] += 1
            if result[0] == "failed":
                failures.append({"image_hash": result[1], "error": result[2]})
            if index % 500 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)
    with (root / "manifests" / "download_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")

    checksum_path = root / "manifests" / "checksums_sha256.jsonl"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(images_dir.glob("*.jpg")):
            record = {"image_hash": path.stem, "sha256": file_hash(path), "bytes": path.stat().st_size}
            handle.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    records, candidate_count = select_records(args.root, args.count, args.seed)
    write_manifest(args.root, records, candidate_count, args.seed)
    print(json.dumps({"candidate_count": candidate_count, "selected_count": len(records)}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
