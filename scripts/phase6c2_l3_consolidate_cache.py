#!/usr/bin/env python3
"""Consolidate immutable L3 cache shards into canonical random-access arrays."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = Path("/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l3_l2_sources")
OUT = Path("/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l3_l2_canonical")
AUDIT = ROOT / "outputs/phase6c2_multiseg_training/l3_cache"
L2_SHA = "dbd7daa8322fe77c8bbfd80223a98ec1e6a4a2c64b4de3b09b09e592ef71d6dd"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_ids_from_rank_partitions(rank_ids: dict[int, list[str]]) -> list[str]:
    """Invert extraction's exact ``fake_indices[rank::world]`` partition."""
    output = []
    for position in range(max(map(len, rank_ids.values()))):
        for rank in sorted(rank_ids):
            if position < len(rank_ids[rank]):
                output.append(rank_ids[rank][position])
    return output


def consolidate(split: str) -> None:
    complete = json.loads((AUDIT / f"{split}_complete.json").read_text())
    if complete.get("status") != "COMPLETE" or complete.get("L2_checkpoint_sha256") != L2_SHA:
        raise RuntimeError(f"{split} source cache is not frozen COMPLETE")
    destination = OUT / split
    done = destination / "complete.json"
    if done.exists():
        saved = json.loads(done.read_text())
        for key in ("q_seg", "target1024_packed", "index"):
            p = destination / saved["files"][key]["name"]
            if not p.is_file() or sha256(p) != saved["files"][key]["sha256"]:
                raise RuntimeError(f"consolidated {split} cache drift: {key}")
        print(json.dumps({"split": split, "status": "EXACT_REUSE"})); return
    destination.mkdir(parents=True, exist_ok=True)
    by_id, rank_ids = {}, {}
    for path in sorted((SRC / split).glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("L2_checkpoint_sha256") != L2_SHA:
            raise RuntimeError(f"L2 provenance drift: {path}")
        offsets = shard["offsets"].tolist()
        rank_ids.setdefault(int(shard["rank"]), [])
        for i, record in enumerate(shard["records"]):
            sid = record["sample_id"]
            if sid in by_id: raise RuntimeError(f"duplicate sample: {sid}")
            by_id[sid] = (record, shard["q_seg"][offsets[i]:offsets[i+1]].clone(),
                         shard["target1024_packed"][offsets[i]:offsets[i+1]].clone())
            rank_ids[int(shard["rank"])].append(sid)
    ids = canonical_ids_from_rank_partitions(rank_ids)
    if set(ids) != set(by_id) or len(ids) != int(complete["images"]):
        raise RuntimeError(f"{split} canonical sample identity mismatch")
    total = sum(int(by_id[sid][0]["K"]) for sid in ids)
    q_path = destination / "q_seg.float16.bin"
    target_path = destination / "target1024_packed.uint8.bin"
    q = np.memmap(q_path, mode="w+", dtype=np.float16, shape=(total, 256))
    target = np.memmap(target_path, mode="w+", dtype=np.uint8, shape=(total, 131072))
    records, offsets, cursor = [], [0], 0
    for sid in ids:
        record, q_value, target_value = by_id[sid]
        k = int(record["K"])
        if len(q_value) != k or len(target_value) != k: raise RuntimeError(f"K mismatch: {sid}")
        q[cursor:cursor+k] = q_value.float().numpy().astype(np.float16)
        target[cursor:cursor+k] = target_value.numpy()
        records.append(record); cursor += k; offsets.append(cursor)
    q.flush(); target.flush(); del q, target
    index = destination / "index.json"
    index.write_text(json.dumps({
        "schema": "phase6c2_l3_canonical_cache_index_v1", "split": split,
        "sample_ids": ids, "offsets": offsets, "records": records,
        "target_shape": [total, 131072], "q_shape": [total, 256],
        "target_order": "canonical image order then annotation/ref order",
    }, ensure_ascii=False) + "\n")
    result = {
        "schema": "phase6c2_l3_canonical_cache_v1", "status": "COMPLETE", "split": split,
        "images": len(ids), "masks": total, "L2_checkpoint_sha256": L2_SHA,
        "files": {key: {"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                  for key, p in (("q_seg", q_path), ("target1024_packed", target_path), ("index", index))},
        "identity": {"sample_order_exact": True, "pair_order_exact": True, "union_used": False},
        "firewall": {"internal_test_accessed": False, "external_accessed": False},
    }
    done.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"split": split, "status": "COMPLETE", "images": len(ids), "masks": total}))


if __name__ == "__main__":
    consolidate("train")
    consolidate("val")
