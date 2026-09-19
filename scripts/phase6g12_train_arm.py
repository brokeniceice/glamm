#!/usr/bin/env python3
"""Phase 6G.12 full Rectifier training with complementary side correction.

A0 = main-only Rectifier.
A2 = main Rectifier + zero-initialized side projection, trained jointly.
A1 is the Phase6G.11 frozen-main reference and is not retrained here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, phase4f_spatial_batch
from scripts.phase6g7_staged_r1 import Store
from scripts.phase6g9_rectifier_internal_conversion_audit import TapRecorder
from scripts.phase6g10_train_arm import (
    EPOCHS,
    BATCH,
    load_fusion,
    load_projection,
    fused_evidence,
    matched_rectifiers,
    optimizer_and_scheduler,
    spectrum,
)
from scripts.phase6g11_zero_init_side_path import (
    ZeroInitSide,
    mechanism_diagnostics,
    side_spectrum,
    side_subspace_diagnostics,
)
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import inverse_sam_logits, metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_sam_runtime, mask_loss

OUT = ROOT / "outputs/phase6g12_full_rectifier_side_correction"
ARMS = ("A0", "A2")
SEED = 3407


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class JointRectifier(nn.Module):
    """Main Rectifier plus zero-init side projection, trained jointly."""

    def __init__(self, main, side):
        super().__init__()
        self.main = main
        self.side = side
        self._r2 = None
        self.main.rectification.cross_attention.out_proj.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._r2 = args[0]

    def forward(self, s64, evidence, sam_coordinates, clip_coordinates, valid):
        with torch.autocast(device_type=s64.device.type, dtype=torch.bfloat16):
            out = self.main(s64, evidence, sam_coordinates, clip_coordinates, valid)
        batch = s64.shape[0]
        r2 = self._r2.float().transpose(1, 2).reshape(batch, 256, 64, 64)
        support = out["support"].reshape(batch, 1, 64, 64).float()
        side = self.side(r2) * support
        rectified = out["image_embeddings"].float() + side
        return {
            "image_embeddings": rectified,
            "residual_main": out["residual"],
            "residual_side": side,
            "support": out["support"],
        }


def forward_model(model, s64, evidence, sc, cc, valid):
    if isinstance(model, JointRectifier):
        out = model(s64, evidence, sc, cc, valid)
        return out, out["image_embeddings"]
    with torch.autocast(device_type=s64.device.type, dtype=torch.bfloat16):
        out = model(s64, evidence, sc, cc, valid)
    return out, out["image_embeddings"].float()


def train_step(model, sam, fusion, projection, p4_store, fs_store, qseg, ids, device):
    s64, _, targets, sc, cc = phase4f_spatial_batch(p4_store, ids, device)
    evidence = fused_evidence(fusion, projection, fs_store, ids, device)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    _, rectified = forward_model(model, s64, evidence, sc, cc, valid)
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(qseg, rectified.to(torch.bfloat16))
    return mask_loss(low, targets, hd.CFG)


def evaluate(model, sam, fusion, projection, p4_store, fs_store, dev, cache, device):
    model.eval()
    records = []
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                records.append(invalid_record(sid, dev["original_masks"][i]))
                continue
            qseg = cache["q_seg"][i:i + 1].to(device=device, dtype=torch.bfloat16)
            s64, _, _, sc, cc = phase4f_spatial_batch(p4_store, [sid], device)
            evidence = fused_evidence(fusion, projection, fs_store, [sid], device)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            _, rectified = forward_model(model, s64, evidence, sc, cc, valid)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, rectified.to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]), dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "VAL", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    return summarize_extended(records), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_all()

    root = OUT / "arms" / args.arm
    if (root / "summary.json").exists():
        print(json.dumps({"arm": args.arm, "status": "ALREADY_COMPLETE"}))
        return
    root.mkdir(parents=True, exist_ok=True)

    fusion = load_fusion(device)
    projection = load_projection(device)
    sam = load_sam_runtime(hd.CFG, device)
    arms, _ = matched_rectifiers(device)
    main_rectifier = arms["A0"]
    side = ZeroInitSide().to(device)
    model = main_rectifier if args.arm == "A0" else JointRectifier(main_rectifier, side)
    trainable = [p for p in model.parameters() if p.requires_grad]

    p4_train, p4_val = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_cache = load_c1_cache("train", p4_train.sample_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    valid_mask = train_cache["valid"].bool()
    where = {sid: i for i, sid in enumerate(p4_train.sample_ids)}

    init_hash = tensor_state_sha256(model.state_dict())
    dump(root / "initialization.json", {
        "arm": args.arm,
        "main_hash": tensor_state_sha256(main_rectifier.state_dict()),
        "side_hash": tensor_state_sha256(side.state_dict()),
        "model_hash": init_hash,
    })
    if args.arm == "A2":
        init_ids = p4_train.sample_ids[:BATCH]
        qseg = train_cache["q_seg"][:BATCH].to(device=device, dtype=torch.bfloat16)
        s64, _, _, sc, cc = phase4f_spatial_batch(p4_train, init_ids, device)
        evidence = fused_evidence(fusion, projection, fs_train, init_ids, device)
        valid = torch.ones(len(init_ids), 576, dtype=torch.bool, device=device)
        _, rectified = forward_model(model, s64, evidence, sc, cc, valid)
        a0_out, _ = forward_model(main_rectifier, s64, evidence, sc, cc, valid)
        dump(root / "init_parity.json", {
            "max_abs_side": float(side.projection.weight.abs().max()),
            "exact_bf16_parity": bool(torch.equal(rectified.to(torch.bfloat16), a0_out["image_embeddings"].to(torch.bfloat16))),
        })

    total_steps = math.ceil(len(p4_train.sample_ids) / BATCH) * EPOCHS
    optimizer, scheduler = optimizer_and_scheduler(trainable, total_steps)
    history, best = [], None
    for epoch in range(1, EPOCHS + 1):
        began = time.time()
        model.train()
        order = list(p4_train.sample_ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        loss_sum, count = 0.0, 0
        for begin in range(0, len(order), BATCH):
            block = order[begin:begin + BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid_mask.index_select(0, positions)]
            if not len(eligible):
                continue
            ids = [p4_train.sample_ids[i] for i in eligible.tolist()]
            qseg = train_cache["q_seg"].index_select(0, eligible).to(device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(model, sam, fusion, projection, p4_train, fs_train, qseg, ids, device)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss["total"].detach()) * len(ids)
            count += len(ids)
        metrics, records = evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "seg_loss": loss_sum / max(1, count),
               "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "dev_g0_mean_f1": metrics["mean_foreground_f1"]}
        history.append(row)
        dump(root / "training_curve.json", history)
        payload = {"schema": "phase6g12_arm_last_v1", "arm": args.arm, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict()}
        torch.save(payload, root / "checkpoints" / f"epoch_{epoch}.pt")
        write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
        print(json.dumps({"stage": "RECT_EPOCH", "arm": args.arm, **row}), flush=True)
        if best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]:
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    torch.save({"schema": "phase6g12_selected_v1", "arm": args.arm, "epoch": best["epoch"],
                "model": best["state"]}, root / "selected_checkpoint.pt")
    metrics, records = evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    write_rows(root / "validation" / "selected.jsonl", records)
    dump(root / "selector.json", {"primary": "internal validation canonical G0 mean IoU",
                                  "selected_epoch": best["epoch"], "selected_metrics": metrics,
                                  "candidates": history})
    result = {
        "arm": args.arm,
        "selected_epoch": best["epoch"],
        "selected_metrics": metrics,
        "initialization": {"main_hash": tensor_state_sha256(main_rectifier.state_dict()),
                           "side_hash": tensor_state_sha256(side.state_dict()),
                           "model_hash": tensor_state_sha256(model.state_dict())},
        "matrix_spectrum": {
            "main_composition": spectrum(
                main_rectifier.rectification.projection.weight.detach().float().cpu()
                @ main_rectifier.rectification.cross_attention.out_proj.weight.detach().float().cpu()
            ),
            "side": "N/A" if args.arm == "A0" else side_spectrum(side),
            "side_subspace": "N/A" if args.arm == "A0" else side_subspace_diagnostics(main_rectifier, side),
        },
        "firewall": {"utility": False, "joint_r1": False, "fusion": False,
                     "adapter": False, "internal_test": False, "official1000": False, "ood": False},
    }
    if args.arm == "A2":
        result["mechanism_diagnostics"] = mechanism_diagnostics(
            main_rectifier, side, TapRecorder(main_rectifier), sam, fusion, projection,
            p4_val, fs_val, dev, val_cache, device,
        )
    dump(root / "summary.json", result)
    print(json.dumps({"arm": args.arm, "status": "COMPLETE", "selected_epoch": best["epoch"],
                      "mean_iou": metrics["mean_foreground_iou"]}), flush=True)


if __name__ == "__main__":
    main()
