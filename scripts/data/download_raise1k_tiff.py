#!/usr/bin/env python3
"""Download the official RAISE-1k TIFF images listed in RAISE_1k.csv."""

import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/RAISE-1k"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def load_records(root):
    csv_path = root / "metadata" / "RAISE_1k.csv"
    records = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append({
                "raise_id": row["File"],
                "download_url": row["TIFF"],
                "file_name": Path(row["TIFF"]).name,
                "raw_nef_url": row["NEF"],
                "device": row["Device"],
                "lens": row["Lens"],
                "image_size": row["Image Size"],
                "date_shot": row["Date Shot"],
                "keywords": row["Keywords"],
            })
    if len(records) != 1000 or len({record["raise_id"] for record in records}) != 1000:
        raise RuntimeError("RAISE-1k CSV does not contain exactly 1000 unique records")
    return records


def write_manifests(root, records):
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "RAISE-1k",
        "official_page": "https://loki.disi.unitn.it/RAISE/download.html",
        "selected_format": "TIFF",
        "raw_nef_downloaded": False,
        "intended_role": "camera-native real held-out evaluation; not initial training",
        "count": len(records),
    }
    with (manifests_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def valid_image(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def sha256sum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(record, images_dir):
    destination = images_dir / record["file_name"]
    if valid_image(destination):
        return "existing", record["raise_id"]
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(record["download_url"], headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if not valid_image(temporary):
                raise OSError("downloaded file is not a valid TIFF image")
            temporary.replace(destination)
            return "downloaded", record["raise_id"]
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    return "failed", record["raise_id"], str(last_error)


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
                failures.append({"raise_id": result[1], "error": result[2]})
            if index % 50 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)
    with (root / "manifests" / "download_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")
    with (root / "manifests" / "checksums_sha256.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(images_dir.glob("*.TIF")):
            handle.write(json.dumps({"file_name": path.name, "sha256": sha256sum(path), "bytes": path.stat().st_size}) + "\n")


def main():
    args = parse_args()
    records = load_records(args.root)
    write_manifests(args.root, records)
    print(json.dumps({"selected_count": len(records), "format": "TIFF"}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
