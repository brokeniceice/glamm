#!/usr/bin/env python3
"""Repair the duplicate-only I2 finalizer race, then finalize without inference."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "outputs/phase6e2_c1_specific_r1/i2/official1000_new_r1_records.jsonl"
C1 = ROOT / "outputs/phase6e1_c1_old_r1_transfer/c1_records.jsonl"
AUDIT = ROOT / "outputs/phase6e2_c1_specific_r1/i2/duplicate_repair_audit.json"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, values) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    os.replace(tmp, path)


def main() -> None:
    source = rows(RECORDS)
    expected = [row["sample_id"] for row in rows(C1)]
    grouped = defaultdict(list)
    for row in source:
        grouped[row["sample_id"]].append(row)
    if set(grouped) != set(expected) or len(expected) != 1000:
        raise RuntimeError("I2 Official1000 identity set is incomplete or drifted")
    conflicts = []
    for sid, values in grouped.items():
        canonical = json.dumps(values[0], sort_keys=True, separators=(",", ":"))
        if any(json.dumps(row, sort_keys=True, separators=(",", ":")) != canonical for row in values[1:]):
            conflicts.append(sid)
    if conflicts:
        raise RuntimeError(f"I2 duplicate predictions conflict: {conflicts[:10]}")
    ordered = [grouped[sid][0] for sid in expected]
    atomic_jsonl(RECORDS, ordered)
    atomic_json(AUDIT, {
        "schema": "phase6e2_i2_duplicate_repair_v1", "status": "PASS",
        "cause": "two authorized detached supervisors concurrently appended identical deterministic predictions",
        "source_rows": len(source), "unique_rows": len(ordered),
        "duplicate_rows_removed": len(source) - len(ordered), "conflicting_duplicate_ids": 0,
        "ordered_ids_match_phase6e1_c1_records": True, "inference_rerun": False,
    })
    os.environ["PHASE6E2_ARM"] = "i2"
    from scripts.phase6e2_official_finalize import finalize
    finalize()
    from scripts.phase6e2_combine import main as combine
    combine()


if __name__ == "__main__":
    main()
