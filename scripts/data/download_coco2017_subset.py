#!/usr/bin/env python3
"""Select and download a content-balanced subset of official COCO 2017 train images."""

import argparse
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


ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
SCENE_WORDS = {
    "airport", "beach", "building", "city", "field", "forest", "harbor", "kitchen",
    "lake", "landscape", "mountain", "ocean", "park", "river", "road", "room", "sea",
    "sky", "snow", "street", "valley",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/COCO2017"))
    parser.add_argument("--human", type=int, default=2240)
    parser.add_argument("--object", type=int, default=680)
    parser.add_argument("--animal", type=int, default=440)
    parser.add_argument("--scene", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def load_and_select(root, counts, seed):
    annotations_dir = root / "annotations"
    instances = json.loads((annotations_dir / "instances_train2017.json").read_text())
    captions_data = json.loads((annotations_dir / "captions_train2017.json").read_text())
    category_names = {row["id"]: row["name"] for row in instances["categories"]}
    licenses = {row["id"]: row for row in instances["licenses"]}
    categories_by_image = defaultdict(set)
    for annotation in instances["annotations"]:
        categories_by_image[annotation["image_id"]].add(category_names[annotation["category_id"]])
    captions_by_image = defaultdict(list)
    for annotation in captions_data["annotations"]:
        captions_by_image[annotation["image_id"]].append(annotation["caption"])

    candidates = {category: [] for category in counts}
    metadata = {}
    for image in instances["images"]:
        image_id = image["id"]
        labels = categories_by_image[image_id]
        caption_text = " ".join(captions_by_image[image_id]).lower()
        if "person" in labels:
            category = "human"
        elif labels & ANIMALS:
            category = "animal"
        elif any(word in caption_text.split() for word in SCENE_WORDS):
            category = "scene"
        else:
            category = "object"
        record = dict(image)
        record.update({
            "category": category,
            "instance_categories": sorted(labels),
            "captions": captions_by_image[image_id],
            "license_metadata": licenses.get(image.get("license")),
            "download_url": f"http://images.cocodataset.org/train2017/{image['file_name']}",
        })
        candidates[category].append(image_id)
        metadata[image_id] = record

    rng = random.Random(seed)
    selected = []
    for category, count in counts.items():
        rng.shuffle(candidates[category])
        if len(candidates[category]) < count:
            raise RuntimeError(f"Only {len(candidates[category])} candidates for {category}, need {count}")
        selected.extend(metadata[image_id] for image_id in candidates[category][:count])
    return selected, {key: len(value) for key, value in candidates.items()}


def write_manifests(root, records, candidate_counts, counts, seed):
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "COCO 2017",
        "official_download_page": "https://cocodataset.org/#download",
        "split": "train2017",
        "seed": seed,
        "candidate_counts": candidate_counts,
        "selected_counts": counts,
        "total": len(records),
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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(record, images_dir):
    destination = images_dir / record["file_name"]
    if valid_image(destination):
        return "existing", record["id"]
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
            return "downloaded", record["id"]
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(attempt + 1)
    return "failed", record["id"], str(last_error)


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
            if index % 500 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)
    with (root / "manifests" / "download_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")
    with (root / "manifests" / "checksums_sha256.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(images_dir.glob("*.jpg")):
            handle.write(json.dumps({"file_name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}) + "\n")


def main():
    args = parse_args()
    counts = {"human": args.human, "object": args.object, "animal": args.animal, "scene": args.scene}
    records, candidate_counts = load_and_select(args.root, counts, args.seed)
    write_manifests(args.root, records, candidate_counts, counts, args.seed)
    print(json.dumps({"candidate_counts": candidate_counts, "selected_counts": counts}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
