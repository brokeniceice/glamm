#!/usr/bin/env python3
"""Phase 6G.11 zero-init side path, 2-GPU DDP variant.

Launch with torchrun --standalone --nproc_per_node=2.  Global batch 8 is
split into four samples per rank; validation is sharded and all-gathered.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.phase6g11_zero_init_side_path as base
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from scripts.phase6g7_staged_r1 import Store
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256
from tools.phase4f import Phase4FStore


def global_batch_split(order, rank, world_size, batch_size):
    """Yield (global_begin, local_ids, global_eligible) with equal local shards."""
    for begin in range(0, len(order), batch_size):
        block = order[begin:begin + batch_size]
        yield begin, block


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    base.seed_all()

    if (base.OUT / "summary.json").exists():
        if rank == 0:
            print(json.dumps({"status": "ALREADY_COMPLETE"}))
        dist.destroy_process_group()
        return

    rectifier, _ = base.load_a0_rectifier(device)
    recorder = base.TapRecorder(rectifier)
    fusion = base.load_fusion(device)
    projection = base.load_projection(device)
    sam = base.load_sam_runtime(base.hd.CFG, device)
    p4_train, p4_val = Phase4FStore(base.hd.CFG, "train"), Phase4FStore(base.hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_cache = load_c1_cache("train", p4_train.sample_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    train_ids = p4_train.sample_ids
    valid_mask = train_cache["valid"].bool()
    where = {sid: i for i, sid in enumerate(train_ids)}

    seed_state = random.getstate()
    random.seed(base.SEED)
    side = base.ZeroInitSide().to(device)
    side = DDP(side, device_ids=[local_rank], output_device=local_rank)
    parameters = list(side.parameters())
    total_steps = math.ceil(len(train_ids) / base.BATCH) * base.EPOCHS
    optimizer, scheduler = base.optimizer_and_scheduler(parameters, total_steps)

    # Reuse the exact single-GPU parity artifact when present; otherwise compute
    # a single-batch parity on rank 0 and a full validation replay on rank 0.
    parity_path = base.OUT / "init_parity.json"
    if parity_path.exists():
        init_parity = json.loads(parity_path.read_text())
    else:
        init_parity = {}
    if rank == 0:
        init_ids = train_ids[:base.BATCH]
        qseg = train_cache["q_seg"][:base.BATCH].to(device=device, dtype=torch.bfloat16)
        s64, _, _, sc, cc = base.phase4f_spatial_batch(p4_train, init_ids, device)
        evidence = base.fused_evidence(fusion, projection, fs_train, init_ids, device)
        valid = torch.ones(len(init_ids), 576, dtype=torch.bool, device=device)
        taps, _, out = recorder.run(s64, evidence, sc, cc, valid)
        side_value = base.side_residual(side.module, taps, out, len(init_ids))
        rectified = out["image_embeddings"] + side_value
        init_parity["max_abs_side"] = float(side_value.abs().max())
        init_parity["exact_bf16_parity"] = bool(torch.equal(rectified.to(torch.bfloat16), out["image_embeddings"]))
        base.dump(parity_path, init_parity)
    dist.barrier()

    history = []
    best = None
    root = base.OUT / "training"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        history_path = root / "training_curve.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())

    val_indices = list(range(len(dev["sample_ids"])))
    local_val_indices = val_indices[rank::world_size]
    local_dev = {
        "sample_ids": [dev["sample_ids"][i] for i in local_val_indices],
        "original_masks": [dev["original_masks"][i] for i in local_val_indices],
        "sam_geometries": [dev["sam_geometries"][i] for i in local_val_indices],
    }
    local_cache = {
        "valid": val_cache["valid"][local_val_indices],
        "q_seg": val_cache["q_seg"][local_val_indices],
    }

    for epoch in range(1, base.EPOCHS + 1):
        side.train()
        rectifier.eval()
        order = list(train_ids)
        random.Random(base.SEED + 1009 * epoch).shuffle(order)
        loss_sum, count = 0.0, 0
        for begin, block in global_batch_split(order, rank, world_size, base.BATCH):
            positions = torch.tensor([where[sid] for sid in block])
            eligible_positions = positions[valid_mask.index_select(0, positions)]
            if not len(eligible_positions):
                continue
            eligible_ids = [train_ids[i] for i in eligible_positions.tolist()]
            if len(eligible_ids) % world_size:
                eligible_ids = eligible_ids + [eligible_ids[-1]] * (world_size - len(eligible_ids) % world_size)
            shard = eligible_ids[rank::world_size]
            qseg = train_cache["q_seg"][
                torch.tensor([where[sid] for sid in shard])
            ].to(device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = base.train_step(
                rectifier, side, recorder, sam, fusion, projection,
                p4_train, fs_train, qseg, shard, device,
            )
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss["total"].detach()) * len(shard)
            count += len(shard)
        local_metrics, local_records = base.evaluate(
            rectifier, side.module, recorder, sam, fusion, projection,
            p4_val, fs_val, local_dev, local_cache, device,
        )
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_records)
        merged = {}
        for rows in gathered:
            for row in rows:
                merged[row["sample_id"]] = row
        records = [merged[sid] for sid in dev["sample_ids"]]
        metrics = summarize_extended(records)
        row = {"epoch": epoch, "seg_loss": loss_sum / max(1, count),
               "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "dev_g0_mean_f1": metrics["mean_foreground_f1"]}
        if rank == 0:
            history.append(row)
            base.dump(root / "training_curve.json", history)
            base.write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
            print(json.dumps({"stage": "SIDE_EPOCH", **row}), flush=True)
        if rank == 0 and (best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]):
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "side": {k: v.detach().cpu() for k, v in side.module.state_dict().items()}}
    dist.barrier()

    if rank == 0:
        torch.save({"schema": "phase6g11_side_selected_v1", "epoch": best["epoch"],
                    "side": best["side"]}, root / "selected_side.pt")
    dist.barrier()
    selected = torch.load(root / "selected_side.pt", map_location="cpu", weights_only=False)
    side.module.load_state_dict(selected["side"])

    _, local_records = base.evaluate(
        rectifier, side.module, recorder, sam, fusion, projection,
        p4_val, fs_val, local_dev, local_cache, device,
    )
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)
    merged = {}
    for rows in gathered:
        for row in rows:
            merged[row["sample_id"]] = row
    records = [merged[sid] for sid in dev["sample_ids"]]
    a0_records = base.rows(base.A0_VAL)
    paired = paired_statistics(
        torch.tensor([r["foreground_iou"] for r in records]),
        torch.tensor([r["foreground_iou"] for r in a0_records]),
    )
    if paired["mean_difference"] > 0 and paired["bootstrap_95_ci"][0] > 0:
        decision = "CONTROLLED_CORRECTION_SIDE_PATH_SUPPORTED"
    elif paired["mean_difference"] > 0:
        decision = "SIDE_PATH_SIGNAL_PRESENT_NOT_STABLE"
    elif paired["bootstrap_95_ci"][1] < 0:
        decision = "EXTRA_CORRECTION_CAPACITY_HARMS_SAM_ALIGNMENT"
    else:
        decision = "SUPPRESSED_FORENSIC_DIRECTIONS_NOT_TASK_USEFUL"

    if rank == 0:
        base.write_rows(root / "validation" / "selected.jsonl", records)
        diagnostics = base.mechanism_diagnostics(
            rectifier, side.module, recorder, sam, fusion, projection,
            p4_val, fs_val, dev, val_cache, device,
        )
        diagnostics["side_spectrum"] = base.side_spectrum(side.module)
        diagnostics["side_subspace"] = base.side_subspace_diagnostics(rectifier, side.module)
        weight = side.module.projection.weight.detach().float().cpu().reshape(256, 256)
        result = {
            "schema": "phase6g11_results_v1",
            "status": "COMPLETE_STOP",
            "execution": "2-GPU DDP",
            "decision": decision,
            "selected_epoch": best["epoch"],
            "A0_checkpoint": str(base.A0_CKPT.resolve()),
            "A0_checkpoint_sha256": file_sha256(base.A0_CKPT),
            "init_parity": init_parity,
            "A0": summarize_extended(a0_records),
            "A1": summarize_extended(records),
            "paired_A1_minus_A0": paired,
            "side_update_norm": float(weight.norm()),
            "side_weight_norm": float(weight.norm()),
            "side_bias_norm": float(side.module.projection.bias.detach().float().norm()),
            "mechanism_diagnostics": diagnostics,
            "firewall": {"utility": False, "joint_r1": False, "adapter": False, "fusion": False,
                         "main_translator_modified": False, "internal_test": False,
                         "official1000": False, "ood": False},
        }
        base.dump(base.OUT / "results.json", result)
        base.dump(base.OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                              "selected_epoch": best["epoch"], "firewall": result["firewall"]})
        base.render(result)
        print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
