#!/usr/bin/env python3
"""Frozen SHA256 and pHash leakage gates for final evaluation manifests."""

from __future__ import annotations

import json
import os
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.audit_image_phash import chunk_values, perceptual_hash


DATA = (ROOT / "datasets").resolve()
OUT = ROOT / "outputs/final_eval_datasets/leakage_audit"
THRESHOLD = 4
MANIFESTS = {
    "SynthScars-Official1000": DATA / "SynthScars/manifests/eval_manifest.jsonl",
    "X-AIGD-labeled_test": DATA / "X-AIGD/manifests/eval_manifest.jsonl",
    "PAL4VST-test": DATA / "PAL4VST/manifests/eval_manifest.jsonl",
    "LOKI-localization": DATA / "LOKI/manifests/localization_eval_manifest.jsonl",
    "LOKI-classification": DATA / "LOKI/manifests/classification_eval_manifest.jsonl",
    "AIGI-Holmes-TestSet": DATA / "AIGI-Holmes/manifests/eval_manifest.jsonl",
    "GenImage-heldout": DATA / "GenImage/manifests/eval_manifest.jsonl",
    "RAISE998": DATA / "RAISE/manifests/eval_manifest.jsonl",
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
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


def resolved_train() -> list[dict]:
    source = ROOT / "outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl"
    result = []
    for row in rows(source):
        explicit = row.get("image_path")
        if explicit:
            path = Path(explicit)
        elif int(row["class_label"]):
            path = DATA / "SynthScars" / row["image_relpath"]
        else:
            path = DATA / row["image_relpath"]
        result.append({"sample_id": row["sample_id"], "image_path": str(path.resolve()),
                       "sha256": row.get("content_sha256") or file_sha256(path), "domain": "internal_train"})
    if len(result) != 17_672 or len({row["sample_id"] for row in result}) != len(result):
        raise RuntimeError("internal train population drift")
    return result


def hash_records(name: str, source: list[dict]) -> list[dict]:
    cache = OUT / "phash" / f"{name}.jsonl"
    if cache.is_file():
        cached = rows(cache)
        if [r["sample_id"] for r in cached] == [r["sample_id"] for r in source]:
            return cached
    def one(row: dict) -> dict:
        value, width, height = perceptual_hash(row["image_path"])
        return {"sample_id": row["sample_id"], "image_path": row["image_path"], "phash64": value,
                "width": width, "height": height, "label": row.get("label"), "dataset": name}
    output = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for index, value in enumerate(executor.map(one, source), 1):
            output.append(value)
            if index % 5000 == 0 or index == len(source):
                print(f"pHash {name} {index}/{len(source)}", flush=True)
    atomic_jsonl(cache, output)
    return output


def bucket(records: list[dict]) -> dict[tuple[int, int], list[tuple[int, dict]]]:
    result = defaultdict(list)
    for record in records:
        value = int(record["phash64"], 16)
        for index, part in enumerate(chunk_values(value)):
            result[(index, part)].append((value, record))
    return result


def cross_pairs(left: list[dict], right: list[dict], left_name: str, right_name: str) -> list[dict]:
    indexed = bucket(left)
    pairs = {}
    for rr in right:
        value = int(rr["phash64"], 16)
        candidates = {}
        for index, part in enumerate(chunk_values(value)):
            for lv, lr in indexed.get((index, part), []):
                candidates[lr["sample_id"]] = (lv, lr)
        for lv, lr in candidates.values():
            distance = (lv ^ value).bit_count()
            if distance <= THRESHOLD:
                key = (lr["sample_id"], rr["sample_id"])
                pairs[key] = {"left_dataset": left_name, "left_sample_id": key[0],
                              "right_dataset": right_name, "right_sample_id": key[1],
                              "hamming_distance": distance, "left_label": lr.get("label"), "right_label": rr.get("label")}
    return list(pairs.values())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT.parent / "leakage_status.json"
    status = {"schema": "final_eval_leakage_audit_v1", "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "phash_hamming_threshold": THRESHOLD}
    atomic_json(status_path, status)
    try:
        train = resolved_train()
        manifests = {name: rows(path) for name, path in MANIFESTS.items()}
        for name, values in manifests.items():
            if len({row["sample_id"] for row in values}) != len(values):
                raise RuntimeError(f"duplicate sample IDs in {name}")
            if any(not Path(row["image_path"]).is_file() for row in values):
                raise RuntimeError(f"missing image in {name}")
        train_sha = defaultdict(list)
        for row in train:
            if row.get("sha256"):
                train_sha[row["sha256"]].append(row["sample_id"])
        exact = []
        for name, values in manifests.items():
            for row in values:
                for train_id in train_sha.get(row.get("sha256"), []):
                    exact.append({"affected_dataset": name, "external_sample_id": row["sample_id"], "internal_train_sample_id": train_id, "sha256": row["sha256"]})
        atomic_jsonl(OUT / "external_vs_internal_train_exact.jsonl", exact)

        train_phash = hash_records("InternalTrain17672", train)
        near = []
        for name, values in manifests.items():
            # Avoid hashing the same LOKI image twice; identity remains traceable to both manifests.
            phashes = hash_records(name, values)
            near.extend(cross_pairs(train_phash, phashes, "InternalTrain17672", name))
        atomic_jsonl(OUT / "external_vs_internal_train_phash.jsonl", near)

        aigi = hash_records("AIGI-Holmes-TestSet", manifests["AIGI-Holmes-TestSet"])
        genimage = hash_records("GenImage-heldout", manifests["GenImage-heldout"])
        aigi_gen_exact = []
        gen_sha = defaultdict(list)
        for row in manifests["GenImage-heldout"]:
            gen_sha[row["sha256"]].append(row)
        for row in manifests["AIGI-Holmes-TestSet"]:
            for other in gen_sha.get(row["sha256"], []):
                aigi_gen_exact.append({"aigi_sample_id": row["sample_id"], "genimage_sample_id": other["sample_id"],
                                       "sha256": row["sha256"], "aigi_label": row["label"], "genimage_label": other["label"]})
        atomic_jsonl(OUT / "aigi_holmes_vs_genimage_exact.jsonl", aigi_gen_exact)
        aigi_gen_near = cross_pairs(genimage, aigi, "GenImage-heldout", "AIGI-Holmes-TestSet")
        atomic_jsonl(OUT / "aigi_holmes_vs_genimage_phash.jsonl", aigi_gen_near)

        result = {"status": "PASS" if not exact and not near else "BLOCKED_OVERLAP",
                  "internal_train_n": len(train), "external_counts": {k: len(v) for k,v in manifests.items()},
                  "external_vs_internal_train": {"exact_overlap": len(exact), "phash_overlap": len(near)},
                  "benchmark_independence": {"aigi_holmes_vs_genimage_exact": len(aigi_gen_exact), "aigi_holmes_vs_genimage_phash": len(aigi_gen_near)},
                  "policy": "No external sample deletion. Any internal-train overlap blocks model evaluation."}
        atomic_json(OUT / "summary.json", result)
        status.update(result)
        if result["status"] != "PASS":
            raise RuntimeError(f"evaluation leakage gate blocked: exact={len(exact)} pHash={len(near)}")
        status["status"] = "COMPLETE"
    except BaseException as error:
        status.setdefault("status", "FAILED")
        if status["status"] == "RUNNING": status["status"] = "FAILED"
        status.update({"exception_type": type(error).__name__, "exception": str(error)})
        raise
    finally:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)


if __name__ == "__main__":
    main()
