#!/usr/bin/env python3
"""Select and download a reproducible real-image subset from Open Images V7."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


HUMAN_LABELS = {"Person"}
ANIMAL_LABELS = {"Animal", "Fauna", "Mammal", "Bird", "Fish", "Reptile", "Amphibian", "Insect", "Pet", "Wildlife"}
SCENE_LABELS = {
    "Architecture", "Beach", "Building", "City", "Coast", "Field", "Forest",
    "Garden", "House", "Indoor", "Landscape", "Mountain", "Natural landscape",
    "Nature", "Outdoor recreation", "Park", "Road", "Room", "Rural area", "Sea",
    "Sky", "Street", "Urban area", "Water",
}
OBJECT_LABELS = {
    "Aircraft", "Auto part", "Boat", "Car", "Clothing", "Cuisine", "Dish",
    "Electronics", "Fashion accessory", "Flower", "Food", "Footwear", "Furniture",
    "Home appliance", "Land vehicle", "Meal", "Musical instrument", "Plant", "Produce",
    "Sports equipment", "Tableware", "Tool", "Toy", "Vehicle",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/OpenImagesV7"))
    parser.add_argument("--human", type=int, default=4000)
    parser.add_argument("--object", type=int, default=1500)
    parser.add_argument("--animal", type=int, default=1000)
    parser.add_argument("--scene", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def load_inputs(root):
    metadata_dir = root / "metadata"
    with (metadata_dir / "oidv7-class-descriptions.csv").open(encoding="utf-8") as handle:
        names = dict(csv.reader(handle))

    positive_labels = defaultdict(set)
    with (metadata_dir / "oidv7-val-annotations-human-imagelabels.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["Confidence"]) == 1.0:
                positive_labels[row["ImageID"]].add(names.get(row["LabelName"], row["LabelName"]))

    with (metadata_dir / "validation-images-with-rotation.csv").open(encoding="utf-8") as handle:
        metadata = {row["ImageID"]: row for row in csv.DictReader(handle)}
    return positive_labels, metadata


def select_ids(labels_by_id, counts, seed):
    candidates = {category: [] for category in counts}
    for image_id, labels in labels_by_id.items():
        if labels & HUMAN_LABELS:
            category = "human"
        elif labels & ANIMAL_LABELS:
            category = "animal"
        elif labels & SCENE_LABELS:
            category = "scene"
        elif labels & OBJECT_LABELS:
            category = "object"
        else:
            continue
        candidates[category].append(image_id)

    rng = random.Random(seed)
    selected = {}
    for category, count in counts.items():
        rng.shuffle(candidates[category])
        if len(candidates[category]) < count:
            raise RuntimeError(f"Only {len(candidates[category])} candidates for {category}, need {count}")
        selected[category] = candidates[category][:count]
    return selected, {key: len(value) for key, value in candidates.items()}


def write_manifests(root, selected, candidate_counts, labels_by_id, metadata, seed):
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for category in ("human", "object", "animal", "scene"):
        for image_id in selected[category]:
            record = dict(metadata[image_id])
            record.update({
                "category": category,
                "labels": sorted(labels_by_id[image_id]),
                "download_url": f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg",
            })
            records.append(record)

    manifest_path = manifests_dir / "selected.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "Open Images V7",
        "official_download_page": "https://storage.googleapis.com/openimages/web/download_v7.html",
        "split": "validation",
        "seed": seed,
        "candidate_counts": candidate_counts,
        "selected_counts": {key: len(value) for key, value in selected.items()},
        "total": len(records),
    }
    with (manifests_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return records


def download_one(record, images_dir):
    destination = images_dir / f"{record['ImageID']}.jpg"
    if valid_image(destination):
        return "existing", record["ImageID"]
    temporary = destination.with_suffix(".jpg.part")
    request = urllib.request.Request(record["download_url"], headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if not valid_image(temporary):
            raise OSError("downloaded file is not a valid image")
        temporary.replace(destination)
        return "downloaded", record["ImageID"]
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return "failed", record["ImageID"], str(error)


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
                failures.append({"ImageID": result[1], "error": result[2]})
            if index % 500 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)

    failure_path = root / "manifests" / "download_failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")
    with (root / "manifests" / "checksums_sha256.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(images_dir.glob("*.jpg")):
            handle.write(json.dumps({"ImageID": path.stem, "sha256": sha256sum(path), "bytes": path.stat().st_size}) + "\n")


def main():
    args = parse_args()
    counts = {"human": args.human, "object": args.object, "animal": args.animal, "scene": args.scene}
    labels_by_id, metadata = load_inputs(args.root)
    selected, candidate_counts = select_ids(labels_by_id, counts, args.seed)
    records = write_manifests(args.root, selected, candidate_counts, labels_by_id, metadata, args.seed)
    print(json.dumps({"candidate_counts": candidate_counts, "selected_counts": counts}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
