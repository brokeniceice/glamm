#!/usr/bin/env python3
"""Select and download a reproducible subset of official FFHQ 1024x1024 images."""

import argparse
import hashlib
import json
import random
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/FFHQ"))
    parser.add_argument("--count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def select_records(root, count, seed):
    metadata = json.loads((root / "metadata" / "ffhq-dataset-v2.json").read_text())
    candidates = []
    for image_id, record in metadata.items():
        if record["category"] != "training":
            continue
        selected = {
            "image_id": int(image_id),
            "category": record["category"],
            "metadata": record["metadata"],
            "image": record["image"],
        }
        candidates.append(selected)
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} training candidates, need {count}")
    random.Random(seed).shuffle(candidates)
    return candidates[:count], len(candidates)


def write_manifest(root, records, candidate_count, seed):
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "FFHQ",
        "official_repository": "https://github.com/NVlabs/ffhq-dataset",
        "source_split": "training",
        "seed": seed,
        "candidate_count": candidate_count,
        "selected_count": len(records),
        "usage_note": "FFHQ is not intended for facial-recognition development.",
    }
    with (manifests_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_image(path, expected_md5):
    if not path.is_file() or path.stat().st_size == 0 or md5sum(path) != expected_md5:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def direct_url(file_url):
    drive_id = urllib.parse.parse_qs(urllib.parse.urlparse(file_url).query)["id"][0]
    return f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"


def download_one(record, images_dir):
    image = record["image"]
    destination = images_dir / Path(image["file_path"]).name
    expected_md5 = image["file_md5"]
    if valid_image(destination, expected_md5):
        return "existing", record["image_id"]
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(".png.part")
    last_error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(direct_url(image["file_url"]), headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size != image["file_size"]:
                raise OSError(f"size mismatch: expected {image['file_size']}, got {temporary.stat().st_size}")
            if not valid_image(temporary, expected_md5):
                raise OSError("MD5 mismatch or invalid image")
            temporary.replace(destination)
            return "downloaded", record["image_id"]
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    return "failed", record["image_id"], str(last_error)


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
                failures.append({"image_id": result[1], "error": result[2]})
            if index % 250 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)
    with (root / "manifests" / "download_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")


def main():
    args = parse_args()
    records, candidate_count = select_records(args.root, args.count, args.seed)
    write_manifest(args.root, records, candidate_count, args.seed)
    print(json.dumps({"candidate_count": candidate_count, "selected_count": len(records)}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
