#!/usr/bin/env python3
"""Phase 6G.12 A2 joint main+side training, 2-GPU DDP.

A0 is the existing Phase6G.10 A0 checkpoint and is not retrained here.
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
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.phase6g12_train_arm as base12
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from scripts.phase6g7_staged_r1 import Store
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4f import Phase4FStore

OUT = base12.OUT
ARM = "A2"
A0_VAL = ROOT / "outputs/phase6g10_post_attention_projection_bypass/arms/A0/validation/selected.jsonl"


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    base12.seed_all()

    root = OUT / "arms" / ARM
    if (root / "summary.json").exists():
        if rank == 0:
            print(json.dumps({"status": "ALREADY_COMPLETE"}))
        dist.destroy_process_group()
        return
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)

    fusion = base12.load_fusion(device)
    projection = base12.load_projection(device)
    sam = base12.load_sam_runtime(base12.hd.CFG, device)
    arms, _ = base12.matched_rectifiers(device)
    main_rectifier = arms["A0"]
    side = base12.ZeroInitSide().to(device)
    model = base12.JointRectifier(main_rectifier, side)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    parameters = list(model.parameters())

    p4_train, p4_val = Phase4FStore(base12.hd.CFG, "train"), Phase4FStore(base12.hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_cache = load_c1_cache("train", p4_train.sample_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    train_ids = p4_train.sample_ids
    valid_mask = train_cache["valid"].bool()
    where = {sid: i for i, sid in enumerate(train_ids)}

    if rank == 0:
        base12.dump(root / "initialization.json", {
            "arm": ARM,
            "main_hash": base12.tensor_state_sha256(main_rectifier.state_dict()),
            "side_hash": base12.tensor_state_sha256(side.state_dict()),
            "model_hash": base12.tensor_state_sha256(model.module.state_dict()),
            "execution": "2-GPU DDP",
        })
    dist.barrier()

    total_steps = math.ceil(len(train_ids) / base12.BATCH) * base12.EPOCHS
    optimizer, scheduler = base12.optimizer_and_scheduler(parameters, total_steps)
    history, best = [], None
    resume_path = root / "selected_checkpoint.pt"
    resume_flag = resume_path.exists() and (root / "training_curve.json").exists()
    if resume_flag:
        selected_existing = torch.load(resume_path, map_location="cpu", weights_only=False)
        history = json.loads((root / "training_curve.json").read_text())
        best = {"epoch": int(selected_existing["epoch"])}
        if rank == 0:
            print(json.dumps({"stage": "RESUME_SELECTED", "epoch": best["epoch"]}), flush=True)

    if rank == 0:
        init_ids = train_ids[:base12.BATCH]
        qseg = train_cache["q_seg"][:base12.BATCH].to(device=device, dtype=torch.bfloat16)
        s64, _, _, sc, cc = base12.phase4f_spatial_batch(p4_train, init_ids, device)
        evidence = base12.fused_evidence(fusion, projection, fs_train, init_ids, device)
        valid = torch.ones(len(init_ids), 576, dtype=torch.bool, device=device)
        _, rectified = base12.forward_model(model.module, s64, evidence, sc, cc, valid)
        _, a0_rectified = base12.forward_model(main_rectifier, s64, evidence, sc, cc, valid)
        base12.dump(root / "init_parity.json", {
            "max_abs_side": float(side.projection.weight.abs().max()),
            "exact_bf16_parity": bool(
                torch.equal(rectified.to(torch.bfloat16), a0_rectified.to(torch.bfloat16))
            ),
        })
    dist.barrier()

    # Validation shards are fixed once and gathered after each epoch.
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

    for epoch in (range(1, base12.EPOCHS + 1) if not resume_flag else []):
        model.train()
        order = list(train_ids)
        random.Random(base12.SEED + 1009 * epoch).shuffle(order)
        loss_sum, count = 0.0, 0
        for begin in range(0, len(order), base12.BATCH):
            block = order[begin:begin + base12.BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid_mask.index_select(0, positions)]
            if not len(eligible):
                continue
            ids = [train_ids[i] for i in eligible.tolist()]
            if len(ids) % world_size:
                ids = ids + [ids[-1]] * (world_size - len(ids) % world_size)
            shard = ids[rank::world_size]
            qseg = train_cache["q_seg"][
                torch.tensor([where[sid] for sid in shard])
            ].to(device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss = base12.train_step(model, sam, fusion, projection, p4_train, fs_train, qseg, shard, device)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss["total"].detach()) * len(shard)
            count += len(shard)
        _, local_records = base12.evaluate(
            model.module, sam, fusion, projection, p4_val, fs_val, local_dev, local_cache, device,
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
            base12.dump(root / "training_curve.json", history)
            base12.write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
            print(json.dumps({"stage": "RECT_EPOCH", **row}), flush=True)
        if rank == 0 and (best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]):
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "state": {k: v.detach().cpu() for k, v in model.module.state_dict().items()}}
    dist.barrier()

    if rank == 0 and not resume_flag:
        torch.save({"schema": "phase6g12_selected_v1", "arm": ARM, "epoch": best["epoch"],
                    "model": best["state"]}, root / "selected_checkpoint.pt")
    dist.barrier()
    selected = torch.load(root / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    model.module.load_state_dict(selected["model"])

    _, local_records = base12.evaluate(
        model.module, sam, fusion, projection, p4_val, fs_val, local_dev, local_cache, device,
    )
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)
    merged = {}
    for rows in gathered:
        for row in rows:
            merged[row["sample_id"]] = row
    records = [merged[sid] for sid in dev["sample_ids"]]
    a0_records = base12.rows(A0_VAL)
    paired = paired_statistics(
        torch.tensor([r["foreground_iou"] for r in records]),
        torch.tensor([r["foreground_iou"] for r in a0_records]),
    )
    if paired["mean_difference"] > 0 and paired["bootstrap_95_ci"][0] > 0:
        decision = "JOINT_MAIN_SIDE_RECTIFIER_SUPPORTED"
    elif paired["bootstrap_95_ci"][1] < 0:
        decision = "JOINT_RECTIFIER_COADAPTATION_HARMS_COMPLEMENTARITY"
    else:
        decision = "FROZEN_MAIN_SIDE_CORRECTION_PREFERRED"

    if rank == 0:
        base12.write_rows(root / "validation" / "selected.jsonl", records)
        diagnostics = base12.mechanism_diagnostics(
            main_rectifier, side, base12.TapRecorder(main_rectifier), sam, fusion, projection,
            p4_val, fs_val, dev, val_cache, device,
        )
        diagnostics["side_spectrum"] = base12.side_spectrum(side)
        diagnostics["side_subspace"] = base12.side_subspace_diagnostics(main_rectifier, side)
        weight = side.projection.weight.detach().float().cpu().reshape(256, 256)
        result = {
            "arm": ARM,
            "execution": "2-GPU DDP",
            "decision": decision,
            "selected_epoch": best["epoch"],
            "selected_metrics": summarize_extended(records),
            "paired_A2_minus_A0": paired,
            "side_update_norm": float(weight.norm()),
            "side_weight_norm": float(weight.norm()),
            "side_bias_norm": float(side.projection.bias.detach().float().norm()),
            "mechanism_diagnostics": diagnostics,
            "matrix_spectrum": {
                "main_composition": base12.spectrum(
                    main_rectifier.rectification.projection.weight.detach().float().cpu()
                    @ main_rectifier.rectification.cross_attention.out_proj.weight.detach().float().cpu()
                ),
                "side": base12.side_spectrum(side),
                "side_subspace": base12.side_subspace_diagnostics(main_rectifier, side),
            },
            "firewall": {"utility": False, "joint_r1": False, "fusion": False, "adapter": False,
                         "internal_test": False, "official1000": False, "ood": False},
        }
        base12.dump(root / "summary.json", result)
        base12.dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                          "selected_epoch": best["epoch"], "firewall": result["firewall"]})
        print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
