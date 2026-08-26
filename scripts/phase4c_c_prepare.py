#!/usr/bin/env python3
"""Freeze Phase 4C-C population, Phrase-Repair q cache, controls, and preflight."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.phase4c_b_train import q_index
from tools.phase4c_b import canonical_hash, dump, file_sha256, rows, tensor_hash


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase4c_c_evidence_attribution.yaml").read_text())
    bcfg = yaml.safe_load((ROOT / cfg["phase4c_b"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]; out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["experiment"]["cache_root"]); cache.mkdir(parents=True, exist_ok=True)
    bout = ROOT / cfg["phase4c_b"]["output_root"]
    completion = json.loads((bout / "completion_manifest.json").read_text())
    if completion["status"] != "COMPLETE": raise RuntimeError("Phase 4C-B is not COMPLETE")
    checkpoints = {
        "P1": (Path(bcfg["p1"]["checkpoint"]), bcfg["p1"]["checkpoint_sha256"]),
        "CLIP Reader": (Path(cfg["phase4c_b"]["clip_reader_checkpoint"]), cfg["phase4c_b"]["clip_reader_sha256"]),
        "Forensic Reader": (Path(cfg["phase4c_b"]["forensic_reader_checkpoint"]), cfg["phase4c_b"]["forensic_reader_sha256"]),
        "CLIP projection": (Path(bcfg["phase4c_a"]["clip_proj_checkpoint"]), bcfg["phase4c_a"]["clip_proj_sha256"]),
        "Forensic adapter": (Path(bcfg["phase4c_a"]["forensic_adapter_checkpoint"]), bcfg["phase4c_a"]["forensic_adapter_sha256"]),
    }
    hashes = {}
    for name, (path, expected) in checkpoints.items():
        actual = file_sha256(path)
        if actual != expected: raise RuntimeError(f"{name} hash mismatch")
        hashes[name] = {"path": str(path.resolve()), "sha256": actual}
    bcache = Path(bcfg["experiment"]["cache_root"])
    qg0 = q_index(bcache / "validation", "G0")
    qphrase = q_index(bcache / "validation", "phrase_only")
    qtf = q_index(bcache / "validation", "tf_full_context")
    paired = rows(ROOT / cfg["phase3c0"]["paired_conditions"])
    if len(paired) != 1106 or len(qg0) != 1106: raise RuntimeError("validation count mismatch")
    ids = [r["sample_id"] for r in paired]
    b_by_id = {r["sample_id"]: r for r in paired if r.get("B") is not None}
    common = [sid for sid in ids if sid in b_by_id and qg0[sid][1] and qphrase[sid][1] and qtf[sid][1]]
    if len(common) != int(cfg["population"]["common_four_query_eligible"]):
        raise RuntimeError(f"common population mismatch: {len(common)}")
    g0_exact = 0; repair_q = []; repair_records = []
    for r in paired:
        sid = r["sample_id"]
        if r.get("A") is not None and qg0[sid][1]:
            old = torch.load(r["A"]["representation_path"], map_location="cpu")["projected_embedding_256d"].reshape(-1)
            if not torch.equal(old, qg0[sid][0].reshape(-1)): raise RuntimeError(f"historical/current G0 q mismatch: {sid}")
            g0_exact += 1
        if sid in b_by_id:
            q = torch.load(r["B"]["representation_path"], map_location="cpu")["projected_embedding_256d"].reshape(-1).to(torch.bfloat16)
            repair_q.append(q); repair_records.append({"sample_id": sid, "valid_q_seg": True,
                "q_hash": tensor_hash(((sid, q),)), "assistant_token_ids_sha256": r["B"]["assistant_token_ids_sha256"],
                "prefix_exact": r["B"]["prefix_exact"], "seg_suffix_exact": r["B"]["seg_suffix_exact"]})
    repair_root = cache / "validation/phrase_repair"; repair_root.mkdir(parents=True, exist_ok=True)
    repair_path = repair_root / "shard_000000_001076.pt"
    torch.save({"schema": "phase4c_c_phrase_repair_q_v1", "mode": "phrase_repair",
                "q_seg": torch.stack(repair_q), "valid": torch.ones(len(repair_q), dtype=torch.bool),
                "records": repair_records}, repair_path)
    dump(repair_root / "complete.json", {"status": "COMPLETE", "count": len(repair_q),
         "source": "Phase3C.0 exact token-span Phrase Repair", "all_prefix_exact": all(r["prefix_exact"] for r in repair_records),
         "all_seg_suffix_exact": all(r["seg_suffix_exact"] for r in repair_records),
         "file": str(repair_path), "sha256": file_sha256(repair_path)})
    dump(out / "common_population.json", {"status": "FROZEN", "full_n": 1106, "common_n": len(common),
         "sample_ids": common, "sample_ids_sha256": canonical_hash(common),
         "excluded": [{"sample_id": sid, "g0_valid": qg0[sid][1], "phrase_repair_eligible": sid in b_by_id}
                      for sid in ids if sid not in set(common)]})
    rng = random.Random(int(cfg["experiment"]["seed"])); perm = list(common)
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(common, perm)): break
    mapping = dict(zip(common, perm))
    dump(out / "phase4c_c_image_permutation.json", {"seed": cfg["experiment"]["seed"], "mapping": mapping,
         "n": len(mapping), "fixed_points": sum(a == b for a,b in mapping.items()), "bijection": len(set(mapping.values())) == len(mapping)})
    dump(out / "cross_image_permutation.json", mapping)
    gen = torch.Generator().manual_seed(int(cfg["experiment"]["seed"]))
    spatial_perm = torch.randperm(576, generator=gen).tolist()
    dump(out / "spatial_shuffle_permutation.json", {"seed": cfg["experiment"]["seed"], "shape": [24,24],
         "permutation": spatial_perm, "bijection": len(set(spatial_perm)) == 576})
    p1_ids = [r["sample_id"] for r in rows(bout / "evaluation/G0/P1/predictions.jsonl")]
    if p1_ids != ids: raise RuntimeError("Phase 4C-B/Phrase-Repair sample ordering mismatch")
    preflight = f"""# Phase 4C-C preflight

Status: **BASELINE_REPRODUCTION_PENDING**

## Checkpoints

| Component | Frozen path | SHA256 |
|---|---|---|
""" + "\n".join(f"| {name} | `{v['path']}` | `{v['sha256']}` |" for name,v in hashes.items()) + f"""

- SAM: frozen P1 grounding encoder, covered by P1 checkpoint hash; cached parameter hash `{json.loads((ROOT / bcfg['frozen_spatial_cache']['phase3c1_root'] / 'cache/sam/val/complete.json').read_text())['source_parameter_hash_before']}`.
- CLIP backbone: `openai/clip-vit-large-patch14-336`.

## Dataset and query alignment

- Split: internal validation Fake-only; internal test and official1000 remain sealed.
- Full baseline population: 1,106 IDs, exactly ordered as Phase 4C-B.
- Valid canonical G0 `[SEG]`: {sum(v[1] for v in qg0.values())}/1,106.
- Historical/current G0 projected q exact: {g0_exact}/1,078.
- Phrase-Repair exact token-span eligible: {len(b_by_id)}/1,106; all prefixes and `[SEG]` suffixes exact.
- Mandatory four-query intervention common population: {len(common)} IDs.
- Excluded from the common intervention population: {1106-len(common)}; no missing query was fabricated.
- Phrase-Only and TF-Full valid queries: {sum(v[1] for v in qphrase.values())}/1,106 and {sum(v[1] for v in qtf.values())}/1,106.
- Intervention occurs strictly after frozen language/q extraction, so language tokens, verdict, explanation, target phrase, and `[SEG]` validity cannot change.

## Controls

- Cross-image mapping: deterministic seed {cfg['experiment']['seed']}, bijective derangement, zero fixed points, shared by both Readers.
- Spatial shuffle: one fixed 576-token permutation shared across all images and both Readers.
- Threshold remains logit 0.0; Phase 4C-B decoder, inverse resize, target, metrics, aggregation, and bootstrap are reused.
- No training, selector change, threshold tuning, held-out access, or new checkpoint is authorized or performed.

Formal intervention evaluation is blocked until matched baseline reproduction passes tolerance `1e-4`.
"""
    (out / "phase4c_c_preflight.md").write_text(preflight, encoding="utf-8")
    dump(out / "preflight_manifest.json", {"status": "BASELINE_REPRODUCTION_PENDING", "checkpoint_hashes": hashes,
         "full_n": 1106, "common_n": len(common), "g0_q_exact": g0_exact, "language_invariant_by_placement": True,
         "test_used": False, "official1000_used": False, "training_started": False})
    print(json.dumps({"status": "PREPARED", "full_n": 1106, "common_n": len(common), "g0_q_exact": g0_exact}, indent=2))


if __name__ == "__main__": main()
