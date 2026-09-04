#!/usr/bin/env python3
"""Freeze the unique-image internal-validation Fake manifest for L-FREE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/data_audits/unified_forensics_split_v1/val_fake.jsonl"
OUT = ROOT / "outputs/phase5a3_stage1_epoch_lfree_validation/manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1106 or len({row["sample_id"] for row in rows}) != 1106:
        raise RuntimeError("frozen internal-validation Fake population drift")
    for row in rows:
        if row.get("dataset_split") != "val" or row.get("forensics_domain") != "fake":
            raise RuntimeError(f"scope drift: {row.get('sample_id')}")
        if not Path(row["image_path"]).is_file() or not row.get("refs"):
            raise RuntimeError(f"missing image/GT refs: {row.get('sample_id')}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "internal_val_fake.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    identity = hashlib.sha256(("\n".join(row["sample_id"] for row in rows) + "\n").encode()).hexdigest()
    provenance = {
        "schema": "phase5a3_stage1_epoch_lfree_manifest_v1",
        "population": "frozen internal validation Fake unique images",
        "n": 1106,
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "ordered_sample_id_sha256": identity,
        "gt": "per-image union of all official SynthScars refs polygons at original resolution",
        "input": "official LEGION image-only L-FREE; no GT phrase, explanation, or authenticity",
        "failure_policy": "no [SEG] or empty prediction is all-zero and counted in full N",
        "threshold": "mask logit > 0; multiple masks union",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
