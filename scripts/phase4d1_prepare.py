#!/usr/bin/env python3
"""Prepare the frozen subset, position scale audit, and shared Reader init."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.position_aware_evidence_reader import PositionAwareEvidenceReader
from scripts.phase4c_b_train import pair_paths, q_index
from tools.phase4c_b import file_sha256, load_source_model, load_spatial_shard, source_feature
from tools.phase4d1 import (RunningTensorSummary, canonical_hash, dump,
                           fixed_derangement, reader_state_hash, tensor_summary)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/phase4d1_position_aware_evidence.yaml").read_text())
    bcfg = yaml.safe_load((ROOT / cfg["phase4c_b"]["config"]).read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    cache_root = Path(cfg["experiment"]["cache_root"])
    checkpoint_root = Path(cfg["experiment"]["checkpoint_root"])
    out.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["experiment"]["seed"])
    seed_all(seed)

    # Hard provenance gates.
    bout = ROOT / cfg["phase4c_b"]["output_root"]
    cout = ROOT / cfg["phase4c_c"]["output_root"]
    for path in (bout / "completion_manifest.json", cout / "completion_manifest.json"):
        state = json.loads(path.read_text())
        if state["status"] != "COMPLETE":
            raise RuntimeError(f"required parent phase is not COMPLETE: {path}")
    p1_path = Path(bcfg["p1"]["checkpoint"])
    if file_sha256(p1_path) != bcfg["p1"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checkpoint hash mismatch")

    q_train = q_index(Path(bcfg["experiment"]["cache_root"]))
    source_root = ROOT / bcfg["frozen_spatial_cache"]["phase3c1_root"]
    train_pairs = pair_paths(source_root, "train")
    eligible = []
    manifest_order = []
    for cp, sp in train_pairs:
        clip = load_spatial_shard(cp, "clip")
        sam = load_spatial_shard(sp, "sam")
        if [x["sample_id"] for x in clip["records"]] != [x["sample_id"] for x in sam["records"]]:
            raise RuntimeError("train CLIP/SAM identity mismatch")
        for i, meta in enumerate(clip["records"]):
            sid = meta["sample_id"]
            if sid in q_train and q_train[sid][1]:
                # Train SAM caches persist the same official-union target in the
                # frozen 1024x1024 training frame, not original_masks.
                mask = sam["targets"][i]
                area = float(mask.float().mean())
                manifest_order.append(sid)
                eligible.append({"sample_id": sid, "mask_area_ratio": area})
    if len(eligible) != int(cfg["data"]["eligible_train_fake"]):
        raise RuntimeError(f"eligible count mismatch: {len(eligible)}")

    ranked = sorted(eligible, key=lambda x: (x["mask_area_ratio"], x["sample_id"]))
    quartiles = np.array_split(np.arange(len(ranked)), int(cfg["data"]["mask_area_quartiles"]))
    rng = random.Random(seed)
    selected_meta = {}
    quartile_summary = []
    for quartile, indices in enumerate(quartiles):
        candidates = [ranked[int(i)] for i in indices]
        chosen = rng.sample(candidates, int(cfg["data"]["samples_per_quartile"]))
        for row in chosen:
            selected_meta[row["sample_id"]] = {**row, "mask_area_quartile": quartile}
        quartile_summary.append({
            "quartile": quartile,
            "candidate_n": len(candidates),
            "selected_n": len(chosen),
            "candidate_area_min": min(x["mask_area_ratio"] for x in candidates),
            "candidate_area_max": max(x["mask_area_ratio"] for x in candidates),
        })
    training_order = [sid for sid in manifest_order if sid in selected_meta]
    if len(training_order) != int(cfg["data"]["train_subset"]):
        raise RuntimeError("selected train subset count mismatch")
    subset_records = [{"training_index": i, **selected_meta[sid]}
                      for i, sid in enumerate(training_order)]
    subset_payload = {
        "status": "FROZEN",
        "seed": seed,
        "selection_before_model_run": True,
        "eligible_n": len(eligible),
        "selected_n": len(training_order),
        "selection": "512 deterministic random samples from each frozen 1024x1024 union-target area rank quartile; training order restored to frozen manifest order",
        "mask_area_frame": "frozen SAM 1024x1024 train target; same official polygon-derived all-reference union",
        "quartiles": quartile_summary,
        "sample_ids": training_order,
        "sample_ids_sha256": canonical_hash(training_order),
        "records": subset_records,
        "internal_test_access": False,
        "official1000_access": False,
    }
    dump(out / "phase4d1_train_subset.json", subset_payload)

    # A single initialization is serialized, then loaded by both arms.
    reader = PositionAwareEvidenceReader()
    init_payload = {
        "schema": "phase4d1_shared_reader_init_v1",
        "seed": seed,
        "reader": reader.state_dict(),
        "reader_state_hash": reader_state_hash(reader),
        "position_summary": tensor_summary(reader.position_2d),
    }
    init_path = checkpoint_root / "reader_init.pt"
    torch.save(init_payload, init_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_source = load_source_model(bcfg, "clip_reader", device)
    forensic_source = load_source_model(bcfg, "forensic_reader", device)
    clip_summary = RunningTensorSummary()
    forensic_summary = RunningTensorSummary()
    selected = set(training_order)
    observed_order = []
    subset_cache = cache_root / "train_subset"
    subset_cache.mkdir(parents=True, exist_ok=True)
    shard_manifest = []
    with torch.no_grad():
        for shard_index, (cp, sp) in enumerate(train_pairs):
            clip = load_spatial_shard(cp, "clip")
            sam = load_spatial_shard(sp, "sam")
            indices = [i for i, row in enumerate(clip["records"])
                       if row["sample_id"] in selected]
            if not indices:
                continue
            raw = clip["features"][indices].to(device=device, dtype=torch.float32)
            f_clip = source_feature(clip_source, raw, "clip_reader").float().cpu()
            f_forensic = source_feature(forensic_source, raw, "forensic_reader").float().cpu()
            clip_summary.update(f_clip)
            forensic_summary.update(f_forensic)
            sample_ids = [clip["records"][i]["sample_id"] for i in indices]
            observed_order.extend(sample_ids)
            payload = {
                "schema": "phase4d1_train_subset_cache_v1",
                "sample_ids": sample_ids,
                "q_seg": torch.stack([q_train[sid][0].reshape(-1) for sid in sample_ids]).to(torch.bfloat16),
                "clip_features": f_clip.to(torch.bfloat16),
                "forensic_features": f_forensic.to(torch.bfloat16),
                "sam_features": sam["features"][indices],
                "targets": sam["targets"][indices],
                "records": [sam["records"][i] for i in indices],
            }
            path = subset_cache / f"shard_{shard_index:03d}.pt"
            torch.save(payload, path)
            shard_manifest.append({"path": str(path), "n": len(sample_ids), "sample_ids": sample_ids})
    if observed_order != training_order:
        raise RuntimeError("selected cache order differs from frozen training order")
    dump(subset_cache / "complete.json", {
        "status": "COMPLETE", "n": len(observed_order), "shards": len(shard_manifest),
        "sample_ids_sha256": canonical_hash(observed_order), "manifest": shard_manifest,
    })

    position = init_payload["position_summary"]
    clip_stats = clip_summary.result([256, 24, 24])
    forensic_stats = forensic_summary.result([256, 24, 24])
    ratios = {
        "rms_position_over_clip": position["rms"] / clip_stats["rms"],
        "rms_position_over_forensic": position["rms"] / forensic_stats["rms"],
    }
    scale_valid = all(0.01 <= x <= 100.0 for x in ratios.values()) and min(
        clip_stats["std"], forensic_stats["std"], position["std"]
    ) > 1e-8
    scale = {
        "status": "PASS" if scale_valid else "POSITION_SCALE_INVALID",
        "population": "frozen 2048 train subset",
        "clip": clip_stats,
        "forensic": forensic_stats,
        "position": position,
        "ratios": ratios,
        "validity_rule": "all RMS ratios in [0.01,100] and all standard deviations >1e-8",
        "validation_metric_used_for_scale": False,
    }
    dump(out / "position_scale_preflight.json", scale)
    (out / "phase4d1_position_scale_preflight.md").write_text(
        "# Phase 4D-1 Position Scale Preflight\n\n"
        f"Status: **{scale['status']}**\n\n"
        "| Tensor | Shape | RMS | STD | Mean |\n|---|---|---:|---:|---:|\n"
        f"| F_CLIP | 256×24×24 | {clip_stats['rms']:.6f} | {clip_stats['std']:.6f} | {clip_stats['mean']:.6f} |\n"
        f"| F_FORENSIC | 256×24×24 | {forensic_stats['rms']:.6f} | {forensic_stats['std']:.6f} | {forensic_stats['mean']:.6f} |\n"
        f"| P_2D | 576×256 | {position['rms']:.6f} | {position['std']:.6f} | {position['mean']:.6f} |\n\n"
        f"- RMS(P)/RMS(F_CLIP): `{ratios['rms_position_over_clip']:.6f}`\n"
        f"- RMS(P)/RMS(F_FORENSIC): `{ratios['rms_position_over_forensic']:.6f}`\n"
        "- No validation IoU or threshold was read or tuned.\n",
        encoding="utf-8",
    )
    if not scale_valid:
        raise RuntimeError("POSITION_SCALE_INVALID")

    # Reuse Phase 4C-C mappings exactly where defined; deterministically extend only
    # its 30 excluded validation IDs so the complete 1,106 population is evaluable.
    val_ids = []
    for _, sp in pair_paths(source_root, "val"):
        sam = load_spatial_shard(sp, "sam")
        val_ids.extend(row["sample_id"] for row in sam["records"])
    saved = json.loads((ROOT / cfg["phase4c_c"]["cross_image_mapping"]).read_text())["mapping"]
    excluded = [sid for sid in val_ids if sid not in saved]
    extension = fixed_derangement(excluded, seed)
    full_mapping = {**saved, **extension}
    if set(full_mapping) != set(val_ids) or set(full_mapping.values()) != set(val_ids):
        raise RuntimeError("full validation cross-image mapping is not bijective")
    dump(out / "validation_cross_image_mapping.json", {
        "seed": seed, "n": len(full_mapping), "phase4c_c_reused_n": len(saved),
        "deterministic_extension_n": len(extension), "mapping": full_mapping,
        "fixed_points": sum(k == v for k, v in full_mapping.items()), "bijection": True,
    })
    dump(out / "train_cross_image_mapping.json", {
        "seed": seed, "n": len(training_order),
        "mapping": fixed_derangement(training_order, seed), "fixed_points": 0,
    })
    spatial = json.loads((ROOT / cfg["phase4c_c"]["spatial_permutation"]).read_text())
    if len(spatial["permutation"]) != 576 or len(set(spatial["permutation"])) != 576:
        raise RuntimeError("invalid Phase 4C-C spatial permutation")
    dump(out / "spatial_content_permutation.json", {
        **spatial,
        "semantics": "permute feature content only; fixed P_2D remains at lattice slots",
        "forbidden_semantics": "do not permute (F+P) tokens together",
    })

    preflight = {
        "status": "PASS",
        "p1": {"path": str(p1_path), "sha256": bcfg["p1"]["checkpoint_sha256"]},
        "feature_shape": [256, 24, 24],
        "position_shape": [576, 256],
        "position_type": "fixed_2d_sincos",
        "reader_init_path": str(init_path),
        "reader_init_hash": init_payload["reader_state_hash"],
        "two_arm_init_equality_required_at_runtime": True,
        "train_subset_hash": subset_payload["sample_ids_sha256"],
        "train_subset_n": len(training_order),
        "validation_n": len(val_ids),
        "scale_status": scale["status"],
        "internal_test_access": False,
        "official1000_access": False,
        "training_started": False,
    }
    dump(out / "preflight_manifest.json", preflight)
    (out / "phase4d1_preflight.md").write_text(
        "# Phase 4D-1 Preflight\n\nStatus: **PASS**\n\n"
        f"- Feature shape: `256×24×24`; position shape: `576×256`.\n"
        f"- Position: fixed non-trainable 2D sine/cosine code.\n"
        f"- Shared Reader init hash: `{init_payload['reader_state_hash']}`.\n"
        f"- Frozen train subset: {len(training_order)} IDs; hash `{subset_payload['sample_ids_sha256']}`.\n"
        f"- Scale preflight: `{scale['status']}`.\n"
        "- Both arms must load the same serialized initialization and identical sample order.\n"
        "- Internal test and official1000 were not accessed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "subset_n": len(training_order),
                      "scale": scale, "reader_init_hash": init_payload["reader_state_hash"]},
                     indent=2))


if __name__ == "__main__":
    main()
