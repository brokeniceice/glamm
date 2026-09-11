#!/usr/bin/env python3
"""Cache selected-L2 ordered multiSEG states for Phase 6C.2 L3.

Only internal train/validation Fake rows are reachable.  The selected L2 is
frozen and this script performs no backward/optimizer operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import UnifiedForensicsDataset
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, move_batch
from tools.phase4c_b import file_sha256

CONFIG = ROOT / "configs/phase6c2_l2_p1_multi.yaml"
L2 = Path("/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2/best/checkpoint/mp_rank_00_model_states.pt")
L2_SHA = "dbd7daa8322fe77c8bbfd80223a98ec1e6a4a2c64b4de3b09b09e592ef71d6dd"
OUT = Path("/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l3_l2_sources")
AUDIT = ROOT / "outputs/phase6c2_multiseg_training/l3_cache"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--images-per-shard", type=int, default=64)
    parser.add_argument("--max-images-per-rank", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    return parser.parse_args()


def tensor_sha256(named) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def collate(tokenizer, rows):
    return custom_collate_fn(
        rows, tokenizer=tokenizer, use_mm_start_end=True,
        inference=False, token_strategy="fixed_cls_query",
    )


def packed_1024(masks: torch.Tensor | None) -> torch.Tensor:
    if masks is None or not masks.numel():
        return torch.empty((0, 1024 * 1024 // 8), dtype=torch.uint8)
    value = F.interpolate(masks[:, None].float(), (1024, 1024), mode="nearest")[:, 0].bool()
    packed = np.packbits(value.numpy().reshape(len(value), -1), axis=1)
    return torch.from_numpy(packed.copy())


def write_shard(split_dir: Path, rank: int, shard_index: int, records, q_values, targets):
    offsets = [0]
    for value in q_values:
        offsets.append(offsets[-1] + len(value))
    q = torch.cat(q_values) if offsets[-1] else torch.empty((0, 256), dtype=torch.bfloat16)
    target = torch.cat(targets) if offsets[-1] else torch.empty((0, 1024 * 1024 // 8), dtype=torch.uint8)
    payload = {
        "schema": "phase6c2_l3_l2_multiseg_cache_v1",
        "split": split_dir.name, "rank": rank,
        "records": records, "offsets": torch.tensor(offsets, dtype=torch.long),
        "q_seg": q, "target1024_packed": target,
        "L2_checkpoint": str(L2), "L2_checkpoint_sha256": L2_SHA,
        "target_order": "annotation order after empty-mask filtering and atomic suffix truncation",
    }
    path = split_dir / f"rank{rank:02d}_shard{shard_index:05d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary); temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "images": len(records), "masks": offsets[-1]}


def main():
    cli = parse_args()
    if file_sha256(L2) != L2_SHA:
        raise RuntimeError("selected L2 checkpoint drift")
    local_rank = int(os.environ.get("LOCAL_RANK", cli.local_rank))
    if local_rank < 0:
        raise RuntimeError("launch with torchrun")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    cfg = yaml.safe_load(CONFIG.read_text())
    torch.manual_seed(int(cfg["experiment"]["seed"])); torch.cuda.manual_seed_all(int(cfg["experiment"]["seed"]))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    args = build_train_args(cfg); args.local_rank = local_rank; args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    state = torch.load(L2, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["module"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected L2 keys: {unexpected}")
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    frozen_before = tensor_sha256(model.named_parameters())

    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, args.vision_tower, split=cli.split,
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=args.image_size, target_protocol="native_multiseg",
    )
    fake_indices = [index for index, row in enumerate(dataset.rows) if row["forensics_domain"] == "fake"]
    local_indices = fake_indices[rank::world]
    if cli.max_images_per_rank is not None:
        local_indices = local_indices[:cli.max_images_per_rank]
    loader = DataLoader(
        Subset(dataset, local_indices), batch_size=cli.batch_size, shuffle=False,
        num_workers=cli.workers, collate_fn=partial(collate, tokenizer), pin_memory=True,
    )
    split_dir = cli.output_root / cli.split; split_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(split_dir.glob(f"rank{rank:02d}_shard*.pt"))
    if existing_paths and not cli.resume:
        raise RuntimeError(f"rank {rank} cache output already exists")
    shards = []
    images = masks = excluded_empty_pairs = atomic_truncations = 0
    for path in existing_paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("L2_checkpoint_sha256") != L2_SHA or saved.get("rank") != rank:
            raise RuntimeError(f"resume shard provenance drift: {path}")
        shard_images = len(saved["records"]); shard_masks = int(saved["offsets"][-1])
        images += shard_images; masks += shard_masks
        excluded_empty_pairs += sum(
            item.get("reason") == "empty_mask"
            for record in saved["records"] for item in record["dropped_pairs"]
        )
        atomic_truncations += sum(
            item.get("reason") == "token_budget_atomic_suffix_truncation"
            for record in saved["records"] for item in record["dropped_pairs"]
        )
        shards.append({"path": str(path), "sha256": file_sha256(path), "images": shard_images, "masks": shard_masks})
    if images > len(local_indices):
        raise RuntimeError("resume cache contains more images than the frozen rank partition")
    local_indices = local_indices[images:]
    records = []; q_values = []; targets = []
    shard_index = len(existing_paths); started = time.time()
    with torch.no_grad():
        for batch in loader:
            cpu_masks = batch["masks_list"]
            gpu = move_batch(batch, device, torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**gpu, multiseg_cache_only=True)
            for i, sid in enumerate(batch["sample_ids"]):
                q = output["projected_seg_embeddings"][i].to(torch.bfloat16).cpu()
                k = int(batch["multiseg_pair_counts"][i])
                if len(q) != k or len(batch["multiseg_ref_indices"][i]) != k:
                    raise RuntimeError(f"SEG/ref/K mismatch for {sid}")
                target = packed_1024(None if k == 0 else cpu_masks[i].detach().cpu())
                if len(target) != k:
                    raise RuntimeError(f"target/K mismatch for {sid}")
                dropped = batch["multiseg_dropped_pairs"][i]
                records.append({
                    "sample_id": sid, "K": k,
                    "ref_indices": list(batch["multiseg_ref_indices"][i]),
                    "dropped_pairs": dropped,
                })
                q_values.append(q); targets.append(target); images += 1; masks += k
                excluded_empty_pairs += sum(x["reason"] == "empty_mask" for x in dropped)
                atomic_truncations += sum(x["reason"] == "token_budget_atomic_suffix_truncation" for x in dropped)
                if len(records) >= cli.images_per_shard:
                    shards.append(write_shard(split_dir, rank, shard_index, records, q_values, targets))
                    records, q_values, targets = [], [], []; shard_index += 1
            if images % 100 == 0:
                print(json.dumps({"split": cli.split, "rank": rank, "images": images, "masks": masks}), flush=True)
    if records:
        shards.append(write_shard(split_dir, rank, shard_index, records, q_values, targets))
    frozen_after = tensor_sha256(model.named_parameters())
    if frozen_before != frozen_after:
        raise RuntimeError("selected L2 mutated during cache extraction")
    rank_result = {
        "schema": "phase6c2_l3_cache_rank_v1", "status": "COMPLETE",
        "split": cli.split, "rank": rank, "world_size": world,
        "images": images, "masks": masks, "excluded_empty_pairs": excluded_empty_pairs,
        "atomic_truncations": atomic_truncations, "shards": shards,
        "L2_state_sha256_before": frozen_before, "L2_state_sha256_after": frozen_after,
        "seconds": time.time() - started,
        "firewall": {"internal_test_accessed": False, "external_accessed": False},
    }
    cli.audit_root.mkdir(parents=True, exist_ok=True)
    path = cli.audit_root / f"{cli.split}_rank{rank:02d}.json"
    path.write_text(json.dumps(rank_result, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        parts = [json.loads((cli.audit_root / f"{cli.split}_rank{i:02d}.json").read_text()) for i in range(world)]
        complete = {
            "schema": "phase6c2_l3_cache_complete_v1", "status": "COMPLETE",
            "split": cli.split, "world_size": world,
            "images": sum(x["images"] for x in parts), "masks": sum(x["masks"] for x in parts),
            "excluded_empty_pairs": sum(x["excluded_empty_pairs"] for x in parts),
            "atomic_truncations": sum(x["atomic_truncations"] for x in parts),
            "parts": parts, "L2_checkpoint_sha256": L2_SHA,
            "firewall": {"internal_test_accessed": False, "external_accessed": False},
        }
        (cli.audit_root / f"{cli.split}_complete.json").write_text(json.dumps(complete, indent=2) + "\n")
        print(json.dumps({key: complete[key] for key in ("status", "split", "images", "masks")}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
