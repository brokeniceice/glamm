#!/usr/bin/env python3
"""Fetch a licensed, pre-2022 animal-photo subset from the official iNaturalist API."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


GROUPS = {
    "mammal": (40151, 350),
    "bird": (3, 300),
    "reptile": (26036, 100),
    "amphibian": (20978, 100),
    "ray_finned_fish": (47178, 100),
    "insect": (47158, 50),
}
API_URL = "https://api.inaturalist.org/v1/observations"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/iNaturalist"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def api_request(taxon_id, page):
    params = {
        "quality_grade": "research",
        "photos": "true",
        "taxon_id": str(taxon_id),
        "d2": "2021-12-31",
        "photo_license": "cc0,cc-by,cc-by-nc",
        "per_page": "200",
        "order_by": "id",
        "order": "desc",
        "page": str(page),
    }
    request = urllib.request.Request(f"{API_URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def photo_record(observation, group):
    photo = observation["photos"][0]
    url = photo["url"].replace("/square.", "/large.")
    extension = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
    return {
        "group": group,
        "observation_id": observation["id"],
        "observed_on": observation.get("observed_on"),
        "created_at": observation.get("created_at"),
        "place_guess": observation.get("place_guess"),
        "taxon": observation.get("taxon"),
        "observer": observation.get("user", {}).get("login"),
        "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
        "photo_id": photo["id"],
        "photo_license": photo.get("license_code"),
        "photo_attribution": photo.get("attribution"),
        "download_url": url,
        "file_name": f"{observation['id']}_{photo['id']}{extension}",
    }


def create_selection(root):
    records = []
    used_photos = set()
    for group, (taxon_id, target) in GROUPS.items():
        group_records = []
        attempts = 0
        while len(group_records) < target and attempts < 20:
            payload = api_request(taxon_id, attempts + 1)
            for observation in payload["results"]:
                if not observation.get("photos"):
                    continue
                record = photo_record(observation, group)
                if record["photo_id"] in used_photos:
                    continue
                used_photos.add(record["photo_id"])
                group_records.append(record)
                if len(group_records) == target:
                    break
            attempts += 1
            time.sleep(1)
        if len(group_records) != target:
            raise RuntimeError(f"Collected {len(group_records)}/{target} records for {group}")
        records.extend(group_records)

    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "iNaturalist",
        "api": API_URL,
        "selection_constraints": {
            "quality_grade": "research",
            "latest_observation_date": "2021-12-31",
            "photo_licenses": ["cc0", "cc-by", "cc-by-nc"],
        },
        "selected_counts": {group: count for group, (_, count) in GROUPS.items()},
        "total": len(records),
    }
    with (manifests_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return records


def load_or_create_selection(root, refresh):
    manifest = root / "manifests" / "selected.jsonl"
    if manifest.is_file() and not refresh:
        return [json.loads(line) for line in manifest.open(encoding="utf-8")]
    return create_selection(root)


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
        return "existing", record["photo_id"]
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(record["download_url"], headers={"User-Agent": "groundingLMM-data-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if not valid_image(temporary):
                raise OSError("downloaded file is not a valid image")
            temporary.replace(destination)
            return "downloaded", record["photo_id"]
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(attempt + 1)
    return "failed", record["photo_id"], str(last_error)


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
                failures.append({"photo_id": result[1], "error": result[2]})
            if index % 100 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)} downloaded={counts['downloaded']} existing={counts['existing']} failed={counts['failed']}", flush=True)
    with (root / "manifests" / "download_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry")
    with (root / "manifests" / "checksums_sha256.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(images_dir.iterdir()):
            if path.is_file() and ".part" not in path.name:
                handle.write(json.dumps({"file_name": path.name, "sha256": sha256sum(path), "bytes": path.stat().st_size}) + "\n")


def main():
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    records = load_or_create_selection(args.root, args.refresh_selection)
    print(json.dumps({"selected_count": len(records), "groups": dict((key, value[1]) for key, value in GROUPS.items())}, indent=2))
    if not args.select_only:
        download_all(args.root, records, args.workers)


if __name__ == "__main__":
    main()
