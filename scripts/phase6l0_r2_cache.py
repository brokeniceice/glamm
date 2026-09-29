#!/usr/bin/env python3
"""Capture A/E/r_prime/q together for the immutable C2-native R2 cache.

Phase6K supplies canonical IDs and provenance; its tensors may differ from a
new forward with a different batch grouping and are never mixed with R2 data.
Phase3C.1 source grids remain immutable references after parity checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase3c2_p3 import dataset_for
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase6l0_r2_preflight import CKPT, EXPECTED_SHA, OUT, load_native_runtime, require, c2_native_sam_state, dump
from tools.phase4c_b import file_sha256
from tools.phase3c1 import geometry_for

PHASE6K = ROOT / "outputs/phase6k0_c2_after/cache"
SPATIAL = ROOT / "outputs/phase3c1_spatial_probe/cache"
R2_DATA = Path("/data/yz/groundingLMM_official/cache/phase6l0_r2")


def hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def audit_spatial(split: str, expected: list[str]) -> dict:
    references = {}
    for source, shape in (("sam", (256, 64, 64)), ("clip", (1024, 24, 24))):
        root = SPATIAL / source / split
        complete_path = root / "complete.json"
        complete = json.loads(complete_path.read_text())
        require(complete["status"] == "COMPLETE" and complete["source_parameter_hash_exact"] and
                tuple(complete["feature_shape"]) == shape and complete["samples"] == len(expected),
                f"{split} {source} spatial source incomplete or shape drift")
        paths = sorted(root.glob("shard_*.pt"))
        require(len(paths) == complete["shards"], f"{split} {source} spatial shard count drift")
        ids, specs = [], []
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["schema"] == "phase3c1_spatial_cache_v1" and payload["source"] == source and
                    tuple(payload["features"].shape[1:]) == shape, f"bad {source} spatial shard {path}")
            require(all(row["geometry"] == geometry_for(source, row["geometry"]["original_hw"])
                        for row in payload["records"]), f"{source} geometry drift in {path}")
            ids.extend(str(row["sample_id"]) for row in payload["records"])
            specs.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        require(ids == expected, f"{split} {source} canonical spatial ID/order drift")
        references[source] = {"complete_path": str(complete_path.resolve()),
                              "complete_sha256": file_sha256(complete_path), "source_parameter_hash": complete["source_parameter_hash_before"],
                              "ids_sha256": hash_ids(ids), "shape": list(shape), "shards": specs}
    return references


def run(args) -> None:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    require((OUT / "preflight_subset.json").exists(), "C2-native source/step0 preflight missing")
    subset = json.loads((OUT / "preflight_subset.json").read_text())
    require(subset["status"] == "PASS" and subset["c2_sha256"] == EXPECTED_SHA,
            "C2-native source/step0 preflight invalid")
    require(file_sha256(CKPT) == EXPECTED_SHA, "C2 checkpoint SHA drift")
    expected = train_ids() if args.split == "train" else load_dev("g0")["sample_ids"]
    phase6k = json.loads((PHASE6K / f"{args.split}.json").read_text())
    require(phase6k["status"] == "COMPLETE" and phase6k["n"] == len(expected) and
            phase6k["ids_sha256"] == hash_ids(expected) and phase6k["c2_sha256"] == EXPECTED_SHA,
            "Phase6K C2 canonical cache incomplete")
    spatial = audit_spatial(args.split, expected)
    state, _ = c2_native_sam_state()
    sam = load_native_runtime(state, device)
    require(file_sha256(OUT / "c2_sam_runtime.pt") == subset["runtime_sha256"], "C2 SAM runtime SHA drift")
    cfg = yaml.safe_load((ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml").read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_c2_model(cfg, CKPT, device, expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]))
    ds = dataset_for(tokenizer, cfg, args.split)
    fake = [i for i, row in enumerate(ds.rows) if int(row["class_label"]) == 1]
    require([str(ds.rows[i]["sample_id"]) for i in fake] == expected, "C2 dataset canonical ID/order drift")
    root = R2_DATA / args.split
    root.mkdir(parents=True, exist_ok=True)
    captured = []
    handle = model.c2_cross_attention.register_forward_hook(
        lambda _m, _i, output: captured.append(output.detach().cpu().clone()))
    model.c2_cross_attention.capture_spatial_intermediates = True
    done, specs, valid_count = 0, [], 0
    try:
        for shard_no, old_spec in enumerate(phase6k["shards"]):
            old_path = Path(old_spec["path"])
            require(file_sha256(old_path) == old_spec["sha256"], f"Phase6K shard SHA drift: {old_path}")
            old = torch.load(old_path, map_location="cpu", weights_only=False)
            n = len(old["sample_ids"])
            require(old["sample_ids"] == expected[done:done+n] and old["c2_sha256"] == EXPECTED_SHA,
                    "Phase6K shard canonical order drift")
            target = root / f"shard_{shard_no:06d}.pt"
            if target.exists():
                payload = torch.load(target, map_location="cpu", weights_only=False)
                require(payload["sample_ids"] == old["sample_ids"] and payload["c2_sha256"] == EXPECTED_SHA and
                        payload["phase6k_sha256"] == old_spec["sha256"] and payload["r_prime"].shape == (n, 4096) and
                        payload["A"].shape == (n, 8, 24, 24) and payload["E"].shape == (n, 512, 24, 24) and
                        payload["q_seg"].shape == (n, 256) and payload["z_L"].shape == (n, 1, 256, 256),
                        "R2 cache resume drift")
            else:
                r_values, z_values, a_values, e_values, q_values, count_values = [], [], [], [], [], []
                current_spatial_ordinal, current_sam = -1, None
                for begin in range(0, n, args.batch_size):
                    end = min(begin + args.batch_size, n)
                    ordinals = list(range(done + begin, done + end))
                    samples = [ds[fake[i]] for i in ordinals]
                    captured.clear()
                    model.c2_cross_attention.last_spatial_intermediates = None
                    with torch.no_grad():
                        results = backend.generate_localization_batch(samples, provide_gt_fake=False,
                                                                       generation_mode="unified_fake_generate")
                    cap = model.c2_cross_attention.last_spatial_intermediates
                    require(bool(captured) and cap is not None and captured[-1].shape == (len(samples), 1, 4096),
                            f"C2 r_prime capture missing or shape drift: forwards={len(captured)}")
                    require(cap["A"].shape == (len(samples), 8, 24, 24) and
                            cap["E"].shape == (len(samples), 512, 24, 24), "C2 A/E capture shape drift")
                    seg_counts = [0 if r["projected_seg_embeddings"] is None else len(r["projected_seg_embeddings"]) for r in results]
                    q_current = torch.full((len(samples), 256), float("nan"), dtype=torch.bfloat16)
                    for j, result in enumerate(results):
                        if seg_counts[j] == 1:
                            q_current[j] = result["projected_seg_embeddings"][0].detach().cpu().to(torch.bfloat16)
                    require(bool(torch.isfinite(cap["A"]).all() and torch.isfinite(cap["E"]).all() and
                                 torch.isfinite(q_current[torch.tensor(seg_counts).eq(1)]).all()),
                            "C2 cache contains nonfinite valid tensors")
                    require(float((cap["A"].flatten(2).sum(-1)-1).abs().max()) < 1e-5,
                            "C2 A head mass drift")
                    a_values.append(cap["A"].to(torch.float32))
                    e_values.append(cap["E"].to(torch.bfloat16))
                    q_values.append(q_current)
                    count_values.extend(seg_counts)
                    r_values.append(captured[-1][:, 0].to(torch.bfloat16))
                    # S64 source is indexed by canonical ordinal in 32-image spatial shards.
                    s_values = []
                    for ordinal in ordinals:
                        spatial_ordinal = ordinal // 32
                        if spatial_ordinal != current_spatial_ordinal:
                            spatial_path = Path(spatial["sam"]["shards"][spatial_ordinal]["path"])
                            current_sam = torch.load(spatial_path, map_location="cpu", weights_only=False)
                            current_spatial_ordinal = spatial_ordinal
                        local = ordinal - current_sam["start"]
                        require(current_sam["records"][local]["sample_id"] == expected[ordinal],
                                "S64 canonical ordinal mismatch")
                        s_values.append(current_sam["features"][local])
                    z = torch.zeros((len(samples), 1, 256, 256), dtype=torch.bfloat16)
                    valid_local = [j for j, count in enumerate(seg_counts) if count == 1]
                    if valid_local:
                        q = q_current[valid_local].to(device=device, dtype=torch.bfloat16)
                        s64 = torch.stack(s_values)[valid_local].to(device=device, dtype=torch.bfloat16)
                        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                            low = sam(q, s64)
                        z[valid_local] = low.detach().cpu().to(torch.bfloat16)
                    z_values.append(z)
                payload = {"schema": "phase6l0_r2_cache_shard_v1", "split": args.split,
                           "sample_ids": old["sample_ids"], "valid_c2_g0": torch.tensor(count_values).eq(1),
                           "seg_count": torch.tensor(count_values, dtype=torch.long),
                           "q_seg": torch.cat(q_values), "A": torch.cat(a_values), "E": torch.cat(e_values),
                           "r_prime": torch.cat(r_values),
                           "z_L": torch.cat(z_values), "target_indices": torch.arange(done, done+n),
                           "phase6k_path": str(old_path.resolve()), "phase6k_sha256": old_spec["sha256"],
                           "c2_sha256": EXPECTED_SHA, "sam_runtime_sha256": subset["runtime_sha256"]}
                temporary = target.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, target)
            valid_count += int(payload["valid_c2_g0"].sum())
            specs.append({"path": str(target.resolve()), "sha256": file_sha256(target), "n": n})
            done += n
            print(json.dumps({"split": args.split, "done": done, "total": len(expected)}), flush=True)
        require(done == len(expected), "R2 cache coverage drift")
        dump(OUT / f"{args.split}.json", {
            "status": "COMPLETE", "schema": "phase6l0_r2_cache_v1", "split": args.split,
            "n": done, "valid_c2_g0": valid_count, "invalid_c2_g0": done-valid_count,
            "ids_sha256": hash_ids(expected), "c2_sha256": EXPECTED_SHA,
            "sam_runtime_sha256": subset["runtime_sha256"], "phase6k_index_sha256": file_sha256(PHASE6K / f"{args.split}.json"),
            "phase6k_shards": phase6k["shards"], "spatial_references": spatial, "shards": specs,
            "firewall": {"internal_test": False, "official1000": False, "external_ood": False}})
    finally:
        handle.remove()
        model.c2_cross_attention.capture_spatial_intermediates = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    run(parser.parse_args())
