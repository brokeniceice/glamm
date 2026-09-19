#!/usr/bin/env python3
"""Phase 6G.13 Utility training on frozen main+side Rectifier, 2-GPU DDP."""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.phase6g13_utility_train as base13
from scripts import phase4g1q_conditional_utility as q, phase4hc_direct_utility_arms as hc, phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch, load_c1_cache
from scripts.phase6g7_staged_r1 import Store
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4f import Phase4FStore

OUT = base13.OUT


class UtilityWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, permutation=None):
        return hc.utility_forward(self.model, batch, permutation=permutation)


def train_step_ddp(ddp_utility, joint, sam, fusion, projection, head, p4_store, fs_store,
                   data, ids, positions, cache, cross, perm, device):
    s64, _, targets, sc, cc = base13.phase4f_spatial_batch(p4_store, ids, device)
    batch, _ = c1_language_batch(data, cache, positions, p4_store, sam, device)
    f24 = base13.fused_evidence(fusion, projection, fs_store, ids, device)
    zf24 = head(f24.float())
    batch["S64"] = s64
    batch["F24"] = f24
    batch["z_F24"] = zf24
    crossed_positions = cross.index_select(0, positions)
    crossed_ids = [data["sample_ids"][i] for i in crossed_positions.tolist()]
    cross_f24 = base13.fused_evidence(fusion, projection, fs_store, crossed_ids, device)
    crossed = dict(batch)
    crossed["F24"] = cross_f24
    crossed["z_F24"] = head(cross_f24.float())
    matched = ddp_utility(batch)
    cross_out = ddp_utility(crossed)
    shuffle_out = ddp_utility(batch, permutation=perm)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        rectified = joint(s64, f24, sc, cc, torch.ones(len(ids), 576, dtype=torch.bool, device=device))
    support = rectified["support"].reshape(len(ids), 1, 64, 64)
    gate = base13.gate_to_sam_grid(matched["U"], sc) * support.float()
    adapted = base13.gated_embedding(s64, rectified["image_embeddings"], gate)
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
    seg = base13.mask_loss(low, targets, hd.CFG)
    _, soft = q.target_delta(matched, data["target64"].index_select(0, positions).to(device))
    relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
    cr = hc.rank_loss(matched["U"], cross_out["U"], matched["support"])
    sr = hc.rank_loss(matched["U"], shuffle_out["U"], matched["support"])
    ranking = 0.5 * (cr + sr)
    total = seg["total"] + relative + ranking
    return total, {"seg": seg["total"], "relative": relative, "ranking": ranking, "cr": cr, "sr": sr}


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    base13.seed_all()
    if (OUT / "summary.json").exists():
        if rank == 0:
            print(json.dumps({"status": "ALREADY_COMPLETE"}))
        dist.destroy_process_group()
        return
    if rank == 0:
        (OUT / "validation").mkdir(parents=True, exist_ok=True)
        (OUT / "utility_gate").mkdir(parents=True, exist_ok=True)

    fusion = base13.load_fusion(device)
    projection = base13.load_projection(device)
    head = base13.load_head(device)
    joint, main, side = base13.load_frozen_rectifier(device)
    sam = base13.load_sam_runtime(hd.CFG, device)
    utility, _ = hc.load_utility("a2", device)
    wrapper = UtilityWrapper(utility)
    ddp_utility = DDP(wrapper, device_ids=[local_rank], output_device=local_rank)
    params = [p for p in ddp_utility.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 371803:
        raise RuntimeError("utility trainable parameter drift")
    optimizer = torch.optim.AdamW(params, lr=base13.LR, weight_decay=base13.WD)

    p4_train, p4_val = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_ids = p4_train.sample_ids
    train_cache = load_c1_cache("train", train_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    data = q.load_ids(train_ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    data["sample_ids"] = train_ids
    valid = train_cache["valid"].bool()
    where = {sid: i for i, sid in enumerate(train_ids)}
    cross = torch.tensor([(i + 1) % len(train_ids) for i in range(len(train_ids))])
    perm = torch.randperm(576, generator=torch.Generator().manual_seed(base13.SEED)).to(device)

    if rank == 0:
        base13.dump(OUT / "protocol.json", {
            "schema": "phase6g13_protocol_v1", "status": "FROZEN_BEFORE_FIRST_STEP",
            "execution": "2-GPU DDP", "A0_reference_mean_iou": 0.185546,
            "recipe": {"epochs": base13.EPOCHS, "batch": base13.BATCH, "lr": base13.LR,
                       "weight_decay": base13.WD, "grad_clip": base13.CLIP, "seed": base13.SEED,
                       "selector": "DEV G0 mean IoU", "loss": "seg+relative+ranking", "scheduler": None},
            "firewall": {"rectifier": False, "side": False, "adapter": False, "fusion": False,
                         "joint_r1": False, "internal_test": False, "official1000": False, "ood": False},
        })
    dist.barrier()

    val_indices = list(range(len(dev["sample_ids"])))
    local_val_indices = val_indices[rank::world_size]
    local_dev = {}
    total = len(dev["sample_ids"])
    for key, value in dev.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total:
            local_dev[key] = value[local_val_indices]
        elif isinstance(value, list) and len(value) == total:
            local_dev[key] = [value[i] for i in local_val_indices]
        else:
            local_dev[key] = value
    local_dev["sample_ids"] = [dev["sample_ids"][i] for i in local_val_indices]
    local_cache = {
        "sample_ids": local_dev["sample_ids"],
        "valid": val_cache["valid"][local_val_indices],
        "q_seg": val_cache["q_seg"][local_val_indices],
    }

    history, best = [], None
    for epoch in range(1, base13.EPOCHS + 1):
        ddp_utility.train()
        order = list(train_ids)
        random.Random(base13.SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float)
        updates = 0
        for begin in range(0, len(order), base13.BATCH):
            block = order[begin:begin + base13.BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid.index_select(0, positions)]
            if not len(eligible):
                continue
            if len(eligible) % world_size:
                eligible = torch.cat([eligible, eligible[-1:].repeat(world_size - len(eligible) % world_size)])
            shard = eligible[rank::world_size]
            ids = [train_ids[i] for i in shard.tolist()]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = train_step_ddp(ddp_utility, joint, sam, fusion, projection, head,
                                         p4_train, fs_train, data, ids, shard, train_cache, cross, perm, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, base13.CLIP)
            optimizer.step()
            updates += 1
            for key, value in parts.items():
                sums[key] += float(value.detach()) * len(ids)
            sums["samples"] += len(ids)
        _, local_records, local_gates = base13.evaluate(
            ddp_utility.module.model, joint, sam, fusion, projection, head,
            p4_val, fs_val, local_dev, local_cache, device,
        )
        gathered_records = [None for _ in range(world_size)]
        gathered_gates = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_records, local_records)
        dist.all_gather_object(gathered_gates, local_gates)
        merged_records, merged_gates = {}, {}
        for rows, gates in zip(gathered_records, gathered_gates):
            for row, gate in zip(rows, gates):
                merged_records[row["sample_id"]] = row
                merged_gates[row["sample_id"]] = gate
        records = [merged_records[sid] for sid in dev["sample_ids"]]
        gates = [merged_gates[sid] for sid in dev["sample_ids"]]
        metrics = summarize_extended(records)
        gate_by_sample = {r["sample_id"]: g for r, g in zip(records, gates)}
        row = {"epoch": epoch, "optimizer_updates": updates,
               "seg_loss": sums["seg"] / max(1, sums["samples"]),
               "relative_loss": sums["relative"] / max(1, sums["samples"]),
               "ranking_loss": sums["ranking"] / max(1, sums["samples"]),
               "total_loss": (sums["seg"] + sums["relative"] + sums["ranking"]) / max(1, sums["samples"]),
               "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "dev_g0_mean_f1": metrics["mean_foreground_f1"]}
        if rank == 0:
            history.append(row)
            base13.dump(OUT / "training_curve.json", history)
            base13.write_rows(OUT / "validation" / f"epoch_{epoch}.jsonl", records)
            base13.dump(OUT / "utility_gate" / f"epoch_{epoch}.json", gate_by_sample)
            print(json.dumps({"stage": "UTILITY_EPOCH", **row}), flush=True)
        if rank == 0 and (best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]):
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "model": {k: v.detach().cpu() for k, v in ddp_utility.module.state_dict().items()},
                    "gate": gate_by_sample, "records": records}
    dist.barrier()

    if rank == 0:
        torch.save({"schema": "phase6g13_utility_selected_v1", "epoch": best["epoch"],
                    "model": best["model"]}, OUT / "selected_utility.pt")
    dist.barrier()
    selected = torch.load(OUT / "selected_utility.pt", map_location="cpu", weights_only=False)
    ddp_utility.module.load_state_dict(selected["model"])

    _, local_records, local_gates = base13.evaluate(
        ddp_utility.module.model, joint, sam, fusion, projection, head,
        p4_val, fs_val, local_dev, local_cache, device,
    )
    gathered_records = [None for _ in range(world_size)]
    gathered_gates = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_records, local_records)
    dist.all_gather_object(gathered_gates, local_gates)
    merged_records, merged_gates = {}, {}
    for rows, gates in zip(gathered_records, gathered_gates):
        for row, gate in zip(rows, gates):
            merged_records[row["sample_id"]] = row
            merged_gates[row["sample_id"]] = gate
    records = [merged_records[sid] for sid in dev["sample_ids"]]
    gates = [merged_gates[sid] for sid in dev["sample_ids"]]
    a0_records, a1_records = base13.rows(base13.A0_VAL), base13.rows(base13.A1_VAL)
    paired = paired_statistics(
        torch.tensor([r["foreground_iou"] for r in records]),
        torch.tensor([r["foreground_iou"] for r in a1_records]),
    )
    if paired["mean_difference"] > 0 and paired["bootstrap_95_ci"][0] > 0:
        decision = "UTILITY_AMPLIFIES_COMPLEMENTARY_CORRECTION"
    elif paired["mean_difference"] > 0:
        decision = "UTILITY_SIGNAL_PRESENT_NOT_STABLE"
    elif paired["bootstrap_95_ci"][1] < 0:
        decision = "UTILITY_MISALIGNED_WITH_COMPLEMENTARY_CORRECTION"
    else:
        decision = "UTILITY_DOES_NOT_ADD_VALUE_TO_SIDE_CORRECTION"
    if rank == 0:
        base13.write_rows(OUT / "validation" / "selected.jsonl", records)
        diag = base13.diagnostics({r["sample_id"]: g for r, g in zip(records, gates)}, a0_records, a1_records)
        result = {"schema": "phase6g13_results_v1", "status": "COMPLETE_STOP", "execution": "2-GPU DDP",
                  "decision": decision, "selected_epoch": best["epoch"], "A0_reference": 0.185546,
                  "A1_utility": summarize_extended(records), "paired_A1_minus_A0": paired,
                  "utility_diagnostics": diag,
                  "seg_trigger_invariance": {"valid": int(val_cache["valid"].sum()),
                                             "n": len(val_cache["valid"]),
                                             "rate": float(val_cache["valid"].float().mean())},
                  "firewall": {"rectifier": False, "side": False, "adapter": False, "fusion": False,
                               "joint_r1": False, "internal_test": False, "official1000": False, "ood": False}}
        base13.dump(OUT / "results.json", result)
        base13.dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                           "selected_epoch": best["epoch"], "firewall": result["firewall"]})
        base13.render(result)
        print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
