#!/usr/bin/env python3
"""Audit SynthScars official test for pHash leakage against all prior pools."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("audit_image_phash", REPO_ROOT / "scripts/data/audit_image_phash.py")
PHASH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-test-manifest", type=Path,
        default=REPO_ROOT / "outputs/data_audits/synthscars_image_grouped_v1/test_manifest.jsonl",
    )
    parser.add_argument(
        "--existing-phash-records", type=Path,
        default=REPO_ROOT / "outputs/data_audits/image_phash_v1/phash_records.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/data_audits/official_synth_test_leakage_v1",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--hamming-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = PHASH.load_jsonl(args.official_test_manifest.expanduser().resolve())
    items = [{
        "sample_id": row["sample_id"],
        "domain": "synthscars_official_test",
        "source": "SynthScars",
        "image_path": row["image_path"],
    } for row in manifest]
    cache = output_dir / "official_test_phash_records.jsonl"
    official_records = PHASH.load_complete_cache(cache, len(items))
    if official_records is None:
        official_records = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, record in enumerate(executor.map(PHASH.hash_one_safe, items), start=1):
                official_records.append(record)
                if index % 100 == 0 or index == len(items):
                    print(f"official test pHash: {index}/{len(items)}", flush=True)
        PHASH.atomic_write_jsonl(cache, official_records)
    else:
        print(f"reused {len(official_records)} official-test pHash records", flush=True)

    existing = PHASH.load_jsonl(args.existing_phash_records.expanduser().resolve())
    all_pairs = PHASH.near_duplicate_pairs(existing + official_records, args.hamming_threshold)
    leakage_pairs = [pair for pair in all_pairs if "synthscars_official_test" in {
        pair["left_domain"], pair["right_domain"]
    }]
    PHASH.atomic_write_jsonl(output_dir / "official_test_near_duplicate_pairs.jsonl", leakage_pairs)
    visualizations = PHASH.write_pair_visualizations(existing + official_records, leakage_pairs, output_dir)
    failures = [row for row in official_records if not row.get("phash64")]
    pair_types = Counter(
        "within_official_test" if pair["left_domain"] == pair["right_domain"]
        else "cross_official_test__" + (
            pair["right_domain"] if pair["left_domain"] == "synthscars_official_test" else pair["left_domain"]
        )
        for pair in leakage_pairs
    )
    summary = {
        "schema_version": "official_synth_test_leakage_v1",
        "official_test_count": len(official_records),
        "decode_failure_count": len(failures),
        "near_duplicate_pair_count": len(leakage_pairs),
        "pair_type_counts": dict(pair_types),
        "cross_split_leakage_pair_count": sum(
            pair["left_domain"] != pair["right_domain"] for pair in leakage_pairs
        ),
        "visualization_count": len(visualizations),
        "hamming_threshold": args.hamming_threshold,
        "pass_condition": not failures and not any(
            pair["left_domain"] != pair["right_domain"] for pair in leakage_pairs
        ),
    }
    PHASH.atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
