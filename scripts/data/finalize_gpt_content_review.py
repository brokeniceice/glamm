#!/usr/bin/env python3
"""Freeze GPT labels as the final labels for the sampled content review set."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_LABELS = frozenset(("human", "animal", "object", "scene", "ambiguous"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt-labels-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_gpt_v1",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def finalize(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("No GPT predictions found")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in GPT predictions")
    final_rows = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        label = row["gpt_label"]
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid GPT label for {row['sample_id']}: {label}")
        clip_label = row["clip_label"]
        confusion[clip_label][label] += 1
        final_rows.append({
            "review_index": row["review_index"],
            "sample_id": row["sample_id"],
            "source": row["source"],
            "domain": row["domain"],
            "final_content_label": label,
            "label_authority": "gpt",
            "gpt_model": row["model"],
            "gpt_confidence": row["gpt_confidence"],
            "gpt_reason": row["gpt_reason"],
            "clip_label": clip_label,
            "clip_margin": row["clip_margin"],
            "clip_gpt_agreement": clip_label == label,
            "human_label_ignored": row.get("human_label"),
            "human_status_ignored": row.get("human_status"),
            "response_id": row.get("response_id"),
        })
    final_rows.sort(key=lambda row: row["review_index"])
    agreement_count = sum(row["clip_gpt_agreement"] for row in final_rows)
    summary = {
        "schema_version": "final_content_review_gpt_v1",
        "policy": "GPT label is authoritative for every sampled review image; human labels are ignored.",
        "sample_count": len(final_rows),
        "label_counts": dict(Counter(row["final_content_label"] for row in final_rows)),
        "domain_label_counts": {
            domain: dict(Counter(
                row["final_content_label"] for row in final_rows if row["domain"] == domain
            ))
            for domain in sorted({row["domain"] for row in final_rows})
        },
        "clip_gpt_agreement_count": agreement_count,
        "clip_gpt_agreement_rate": agreement_count / len(final_rows),
        "clip_to_gpt_confusion": {key: dict(value) for key, value in confusion.items()},
        "important_scope_note": "These GPT labels cover only the sampled review set, not all 32,561 CLIP-labeled images.",
    }
    return final_rows, summary


def main() -> None:
    args = parse_args()
    labels_dir = args.gpt_labels_dir.expanduser().resolve()
    failures_path = labels_dir / "failures.json"
    if failures_path.is_file():
        failures = json.load(failures_path.open(encoding="utf-8"))
        if failures:
            raise RuntimeError(f"Cannot finalize with {len(failures)} failed API requests")
    rows = load_jsonl(labels_dir / "predictions.jsonl")
    final_rows, summary = finalize(rows)
    atomic_write_jsonl(labels_dir / "final_review_labels.jsonl", final_rows)
    atomic_write_json(labels_dir / "final_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
