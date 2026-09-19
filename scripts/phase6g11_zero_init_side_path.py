#!/usr/bin/env python3
"""Phase 6G.11 zero-initialized correction side path.

The Phase6G.10 A0 Rectifier is frozen.  Only a zero-initialized 1x1 Conv
side path is trained through the original Rectifier segmentation objective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    optimizer_and_scheduler,
    spectrum,
)
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_rectifier, load_sam_runtime, mask_loss

OUT = ROOT / "outputs/phase6g11_zero_init_correction_side_path"
BASE610 = ROOT / "outputs/phase6g10_post_attention_projection_bypass"
A0_CKPT = BASE610 / "arms/A0/selected_checkpoint.pt"
A0_VAL = BASE610 / "arms/A0/validation/selected.jsonl"
GAMMA_AUDIT = ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json"

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


def load_a0_rectifier(device):
    gamma = float(json.load(open(GAMMA_AUDIT))["selected_gamma"])
    model = load_rectifier(hd.CFG, gamma, device)
    payload = torch.load(A0_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


class ZeroInitSide(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, value):
        return self.projection(value)


def side_residual(side, taps, out, batch):
    support = out["support"].reshape(batch, 1, 64, 64).float()
    return side(taps["R2"]) * support


def train_step(rectifier, side, recorder, sam, fusion, projection, p4_store, fs_store, qseg, ids, device):
    s64, _, targets, sam_coordinates, clip_coordinates = phase4f_spatial_batch(p4_store, ids, device)
    evidence = fused_evidence(fusion, projection, fs_store, ids, device)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    taps, _, out = recorder.run(s64, evidence, sam_coordinates, clip_coordinates, valid)
    side_value = side_residual(side, taps, out, len(ids))
    rectified = out["image_embeddings"] + side_value
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(qseg, rectified.to(torch.bfloat16))
    return mask_loss(low, targets, hd.CFG), side_value, out


def evaluate(rectifier, side, recorder, sam, fusion, projection, p4_store, fs_store, dev, cache, device):
    rectifier.eval()
    side.eval()
    records = []
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                records.append(invalid_record(sid, dev["original_masks"][i]))
                continue
            qseg = cache["q_seg"][i:i + 1].to(device=device, dtype=torch.bfloat16)
            s64, _, _, sam_coordinates, clip_coordinates = phase4f_spatial_batch(p4_store, [sid], device)
            evidence = fused_evidence(fusion, projection, fs_store, [sid], device)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            taps, _, out = recorder.run(s64, evidence, sam_coordinates, clip_coordinates, valid)
            side_value = side_residual(side, taps, out, 1)
            rectified = out["image_embeddings"] + side_value
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, rectified.to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]), dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "VAL", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    return summarize_extended(records), records


def mechanism_diagnostics(rectifier, side, recorder, sam, fusion, projection, p4_store, fs_store, dev, cache, device, samples=256):
    valid_indices = [i for i, sid in enumerate(dev["sample_ids"]) if bool(cache["valid"][i])][:samples]
    norms = {"main": [], "side": [], "ratio": [], "cosine": []}
    rectifier.eval()
    side.eval()
    with torch.no_grad():
        for begin in range(0, len(valid_indices), BATCH):
            batch = valid_indices[begin:begin + BATCH]
            ids = [dev["sample_ids"][i] for i in batch]
            s64, _, _, sam_coordinates, clip_coordinates = phase4f_spatial_batch(p4_store, ids, device)
            evidence = fused_evidence(fusion, projection, fs_store, ids, device)
            valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
            taps, _, out = recorder.run(s64, evidence, sam_coordinates, clip_coordinates, valid)
            main = out["residual"].float()
            side_value = side_residual(side, taps, out, len(ids))
            for j in range(len(ids)):
                m = main[j].flatten()
                s = side_value[j].flatten()
                mn, sn = float(m.norm()), float(s.norm())
                norms["main"].append(mn)
                norms["side"].append(sn)
                norms["ratio"].append(sn / max(1e-12, mn))
                norms["cosine"].append(float(F.cosine_similarity(m, s, dim=0)) if mn > 0 and sn > 0 else 1.0)
    def agg(values):
        return {"mean": float(torch.tensor(values).mean()), "median": float(torch.tensor(values).median()),
                "std": float(torch.tensor(values).std()), "min": float(torch.tensor(values).min()),
                "max": float(torch.tensor(values).max())}
    result = {key: agg(value) for key, value in norms.items()}
    result["n"] = len(valid_indices)
    return result


def side_spectrum(side):
    weight = side.projection.weight.detach().float().cpu().reshape(256, 256)
    return spectrum(weight)


def side_subspace_diagnostics(rectifier, side):
    out_proj = rectifier.rectification.cross_attention.out_proj.weight.detach().float().cpu()
    projection = rectifier.rectification.projection.weight.detach().float().cpu()
    composition = projection @ out_proj
    _, _, right = torch.linalg.svd(composition.double(), full_matrices=False)
    weight = side.projection.weight.detach().float().cpu().reshape(256, 256).double()
    denom = weight.pow(2).sum().clamp_min(1e-30)
    top8 = right[:8].T
    bottom8 = right[-8:].T
    return {
        "side_input_energy_in_main_top8": float((weight @ top8).pow(2).sum() / denom),
        "side_input_energy_in_main_bottom8": float((weight @ bottom8).pow(2).sum() / denom),
        "side_input_energy_outside_main_top8": float(1.0 - (weight @ top8).pow(2).sum() / denom),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_all()

    if (OUT / "summary.json").exists():
        print(json.dumps({"status": "ALREADY_COMPLETE"}))
        return

    rectifier, a0_payload = load_a0_rectifier(device)
    recorder = TapRecorder(rectifier)
    fusion = load_fusion(device)
    projection = load_projection(device)
    sam = load_sam_runtime(hd.CFG, device)
    p4_train, p4_val = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_cache = load_c1_cache("train", p4_train.sample_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    seed_all()
    side = ZeroInitSide().to(device)

    # Zero-init parity: A1 must reproduce A0 exactly before any update.
    init_ids = p4_train.sample_ids[:BATCH]
    qseg = train_cache["q_seg"][:BATCH].to(device=device, dtype=torch.bfloat16)
    s64, _, _, sc, cc = phase4f_spatial_batch(p4_train, init_ids, device)
    evidence = fused_evidence(fusion, projection, fs_train, init_ids, device)
    valid = torch.ones(len(init_ids), 576, dtype=torch.bool, device=device)
    taps, _, out = recorder.run(s64, evidence, sc, cc, valid)
    side_value = side_residual(side, taps, out, len(init_ids))
    rectified = out["image_embeddings"] + side_value
    init_parity = {
        "max_abs_side": float(side_value.abs().max()),
        "exact_bf16_parity": bool(torch.equal(rectified.to(torch.bfloat16), out["image_embeddings"])),
    }
    _, init_records = evaluate(rectifier, side, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    a0_records = rows(A0_VAL)
    if [r["sample_id"] for r in init_records] != [r["sample_id"] for r in a0_records]:
        raise RuntimeError("A0 replay sample order drift")
    init_parity["validation_max_abs_iou_diff"] = max(
        abs(float(a["foreground_iou"]) - float(b["foreground_iou"])) for a, b in zip(init_records, a0_records)
    )
    init_parity["validation_exact_iou"] = all(
        abs(float(a["foreground_iou"]) - float(b["foreground_iou"])) == 0.0 for a, b in zip(init_records, a0_records)
    )
    dump(OUT / "init_parity.json", init_parity)

    parameters = list(side.parameters())
    total_steps = math.ceil(len(p4_train.sample_ids) / BATCH) * EPOCHS
    optimizer, scheduler = optimizer_and_scheduler(parameters, total_steps)
    history, best = [], None
    root = OUT / "training"
    root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        began = time.time()
        side.train()
        rectifier.eval()
        order = list(p4_train.sample_ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        where = {sid: i for i, sid in enumerate(p4_train.sample_ids)}
        valid_mask = train_cache["valid"].bool()
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
            loss, _, _ = train_step(rectifier, side, recorder, sam, fusion, projection, p4_train, fs_train, qseg, ids, device)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss["total"].detach()) * len(ids)
            count += len(ids)
        metrics, records = evaluate(rectifier, side, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "seg_loss": loss_sum / max(1, count),
               "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "dev_g0_mean_f1": metrics["mean_foreground_f1"]}
        history.append(row)
        dump(root / "training_curve.json", history)
        payload = {"schema": "phase6g11_side_last_v1", "epoch": epoch,
                   "side": side.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict()}
        torch.save(payload, root / f"epoch_{epoch}.pt")
        write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
        print(json.dumps({"stage": "SIDE_EPOCH", **row}), flush=True)
        if best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]:
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "side": {k: v.detach().cpu() for k, v in side.state_dict().items()}}

    side.load_state_dict(best["side"])
    metrics, records = evaluate(rectifier, side, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    write_rows(root / "validation" / "selected.jsonl", records)
    side_params = side.projection.weight.detach().float().cpu().reshape(256, 256)
    update_norm = float(side_params.norm())
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

    diagnostics = mechanism_diagnostics(rectifier, side, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    diagnostics["side_spectrum"] = side_spectrum(side)
    diagnostics["side_subspace"] = side_subspace_diagnostics(rectifier, side)
    result = {
        "schema": "phase6g11_results_v1",
        "status": "COMPLETE_STOP",
        "decision": decision,
        "selected_epoch": best["epoch"],
        "A0_checkpoint": str(A0_CKPT.resolve()),
        "A0_checkpoint_sha256": file_sha256(A0_CKPT),
        "init_parity": init_parity,
        "A0": summarize_extended(a0_records),
        "A1": metrics,
        "paired_A1_minus_A0": paired,
        "side_update_norm": update_norm,
        "side_weight_norm": float(side_params.norm()),
        "side_bias_norm": float(side.projection.bias.detach().float().norm()),
        "mechanism_diagnostics": diagnostics,
        "firewall": {"utility": False, "joint_r1": False, "adapter": False, "fusion": False,
                     "main_translator_modified": False, "internal_test": False,
                     "official1000": False, "ood": False},
    }
    dump(OUT / "results.json", result)
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                "selected_epoch": best["epoch"], "firewall": result["firewall"]})
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)


def render(result):
    lines = [
        "# Phase 6G.11 — Zero-Initialized Correction Side Path",
        "",
        "Status: **COMPLETE STOP**. Only the zero-initialized side path was trained; the Phase6G.10 A0 Rectifier and all other modules were frozen.",
        "",
        "## Rectifier objective",
        "",
        "| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |",
        "|---|---:|---:|---:|",
        f"| A0 frozen double projection | {result['A0']['mean_foreground_iou']:.6f} | {result['A0']['median_foreground_iou']:.6f} | {result['A0']['mean_foreground_f1']:.6f} |",
        f"| A1 A0 + zero-init side | {result['A1']['mean_foreground_iou']:.6f} | {result['A1']['median_foreground_iou']:.6f} | {result['A1']['mean_foreground_f1']:.6f} |",
        "",
        f"- paired A1 - A0: `{result['paired_A1_minus_A0']}`",
        f"- selected epoch: `{result['selected_epoch']}`",
        f"- side update norm: `{result['side_update_norm']:.6f}`",
        "",
        "## Mechanism diagnostics",
        "",
        f"- residual norms/cosines: `{result['mechanism_diagnostics']}`",
        f"- side spectrum: `{result['mechanism_diagnostics']['side_spectrum']}`",
        f"- side subspace: `{result['mechanism_diagnostics']['side_subspace']}`",
        "",
        f"```text\n{result['decision']}\n```",
        "",
        "No Utility, joint R1, internal test, Official1000, or OOD was accessed.",
    ]
    (ROOT / "docs/phase6g11_zero_init_correction_side_path.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
