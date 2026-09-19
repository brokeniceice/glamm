#!/usr/bin/env python3
"""Phase 6G.10 post-attention projection bypass ablation: one arm.

Trains only the Rectifier stage from matched initialization.  The frozen
evidence interface is the Phase6G.8B projection-only 1024->256 interface.
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

from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, phase4f_spatial_batch
from scripts.phase6g7_staged_r1 import Store
from scripts.phase6g9_rectifier_internal_conversion_audit import (
    TapRecorder,
    build_train_targets,
)
from tools.phase3c1 import binary_metrics, paired_statistics, probe_loss
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import inverse_sam_logits, metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_rectifier, load_sam_runtime, mask_loss

OUT = ROOT / "outputs/phase6g10_post_attention_projection_bypass"
PROJECTION_CKPT = ROOT / "outputs/phase6g8b_multilevel_evidence_interface_audit/interfaces/projection_256/selected_projection.pt"
FUSION_CKPT = ROOT / "outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt"
GAMMA_AUDIT = ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json"

SEED = 3407
EPOCHS, BATCH = 10, 8
LR, WEIGHT_DECAY, BETAS = 5e-5, 0.05, (0.9, 0.999)
GRAD_CLIP = 1.0
PROBE_EPOCHS, PROBE_BATCH, PROBE_LR, PROBE_WEIGHT_DECAY = 20, 8, 1e-3, 0.0
ARMS = ("A0", "A1", "A2")


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


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_projection(device):
    payload = torch.load(PROJECTION_CKPT, map_location="cpu", weights_only=False)
    model = nn.Conv2d(1024, 256, 1)
    state = {
        name.split("projection.", 1)[1]: value
        for name, value in payload["projection"].items()
        if name.startswith("projection.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_fusion(device):
    payload = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
    model = CrossLayerPatchAttention(1024, 8, 0.01)
    model.load_state_dict(payload["fusion"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def fused_evidence(fusion, projection, store, ids, device):
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        fused = fusion(*store.batch(ids, device))
        evidence = projection(fused)
    return evidence.detach()


def matched_rectifiers(device):
    gamma = float(json.load(open(GAMMA_AUDIT))["selected_gamma"])
    seed_all()
    base = load_rectifier(hd.CFG, gamma, device)
    a0 = copy.deepcopy(base)
    a1 = copy.deepcopy(base)
    a1.rectification.cross_attention.out_proj = nn.Identity()
    a2 = copy.deepcopy(a1)
    a2.rectification.projection = nn.Identity()
    return {"A0": a0, "A1": a1, "A2": a2}, gamma


def optimizer_and_scheduler(parameters, total_steps):
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
    warmup = max(1, round(total_steps * 0.05))

    def scale(step):
        if step < warmup:
            return float(step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def rectifier_train_step(model, sam, fusion, projection, p4_store, fs_store, qseg, ids, device):
    s64, _, targets, sam_coordinates, clip_coordinates = phase4f_spatial_batch(p4_store, ids, device)
    evidence = fused_evidence(fusion, projection, fs_store, ids, device)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        rv = model(s64, evidence, sam_coordinates, clip_coordinates, valid)
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(qseg, rv["image_embeddings"].to(torch.bfloat16))
    return mask_loss(low, targets, hd.CFG)


def rectifier_evaluate(model, sam, fusion, projection, p4_store, fs_store, dev, cache, device):
    model.eval()
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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                rv = model(s64, evidence, sam_coordinates, clip_coordinates, valid)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, rv["image_embeddings"].to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]), dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "RECT_VAL", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    return summarize_extended(records), records


def train_rectifier(arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device):
    root = OUT / "arms" / arm
    if (root / "summary.json").exists():
        print(json.dumps({"arm": arm, "status": "ALREADY_COMPLETE"}), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    train_ids = p4_train.sample_ids
    train_cache = load_c1_cache("train", train_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    valid = train_cache["valid"].bool()
    where = {sid: i for i, sid in enumerate(train_ids)}
    parameters = [p for p in model.parameters() if p.requires_grad]
    total_steps = math.ceil(len(train_ids) / BATCH) * EPOCHS
    optimizer, scheduler = optimizer_and_scheduler(parameters, total_steps)
    history_path, last_path = root / "training_curve.json", root / "last_state.pt"
    history = json.load(open(history_path)) if history_path.exists() else []
    start_epoch, updates = 1, 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, updates = int(state["epoch"]) + 1, int(state["updates"])

    for epoch in range(start_epoch, EPOCHS + 1):
        began = time.time()
        model.train()
        order = list(train_ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        loss_sum, count = 0.0, 0
        for begin in range(0, len(order), BATCH):
            block = order[begin:begin + BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid.index_select(0, positions)]
            if not len(eligible):
                continue
            ids = [train_ids[i] for i in eligible.tolist()]
            qseg = train_cache["q_seg"].index_select(0, eligible).to(device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss = rectifier_train_step(model, sam, fusion, projection, p4_train, fs_train, qseg, ids, device)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            updates += 1
            loss_sum += float(loss["total"].detach()) * len(ids)
            count += len(ids)
        metrics, records = rectifier_evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "seg_loss": loss_sum / max(1, count),
            "dev_g0_mean_iou": metrics["mean_foreground_iou"],
            "dev_g0_mean_f1": metrics["mean_foreground_f1"],
            "seconds": time.time() - began,
        }
        history.append(row)
        dump(history_path, history)
        payload = {
            "schema": "phase6g10_arm_last_v1",
            "arm": arm,
            "epoch": epoch,
            "updates": updates,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(payload, root / "checkpoints" / f"epoch_{epoch}.pt")
        torch.save(payload, last_path)
        write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
        print(json.dumps({"stage": "RECT_EPOCH", **row}), flush=True)

    best = max(history, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    checkpoint = root / "checkpoints" / f"epoch_{best['epoch']}.pt"
    selected = root / "selected_checkpoint.pt"
    selected.write_bytes(checkpoint.read_bytes())
    selected_state = torch.load(selected, map_location="cpu", weights_only=False)
    model.load_state_dict(selected_state["model"])
    metrics, records = rectifier_evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    write_rows(root / "validation" / "selected.jsonl", records)
    dump(root / "selector.json", {"primary": "internal validation canonical G0 mean IoU",
                                  "selected_epoch": best["epoch"], "selected_metrics": metrics,
                                  "candidates": history})
    return


def spectrum(matrix):
    if matrix is None:
        return "N/A"
    value = matrix.double()
    singular = torch.linalg.svdvals(value)
    squared = singular * singular
    total = squared.sum().clamp_min(1e-30)
    probability = squared / total
    return {
        "shape": list(value.shape),
        "rank": int((singular > 1e-6 * singular[0]).sum()),
        "effective_rank_shannon": float(torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum())),
        "effective_rank_participation": float((squared.sum() ** 2) / squared.pow(2).sum().clamp_min(1e-30)),
        "stable_rank": float(squared.sum() / (singular[0] ** 2).clamp_min(1e-30)),
        "condition_number": float(singular[0] / singular[-1].clamp_min(1e-12)),
        "singular_values_top10": [float(x) for x in singular[:10]],
    }


def matrix_spectrum(arm, model):
    if arm == "A0":
        out_proj = model.rectification.cross_attention.out_proj.weight.detach().float().cpu()
        projection = model.rectification.projection.weight.detach().float().cpu()
        return {"out_proj": spectrum(out_proj), "projection": spectrum(projection),
                "composition": spectrum(projection @ out_proj)}
    if arm == "A1":
        projection = model.rectification.projection.weight.detach().float().cpu()
        return {"out_proj": "N/A (identity)", "projection": spectrum(projection),
                "composition": spectrum(projection)}
    return {"out_proj": "N/A (identity)", "projection": "N/A (identity)", "composition": "N/A (identity)"}


def probe_validate(probes, recorder, sam, fusion, projection, p4_store, fs_store, dev, cache, device):
    probes.eval()
    records = {"R2": [None] * len(dev["sample_ids"]), "Delta_F": [None] * len(dev["sample_ids"])}
    with torch.no_grad():
        for begin in range(0, len(dev["sample_ids"]), PROBE_BATCH):
            raw_ix = torch.arange(begin, min(begin + PROBE_BATCH, len(dev["sample_ids"])))
            valid_ix = raw_ix[cache["valid"].index_select(0, raw_ix).bool()]
            for absolute in raw_ix.tolist():
                if bool(cache["valid"][absolute]):
                    continue
                sid = dev["sample_ids"][absolute]
                row = invalid_record(sid, dev["original_masks"][absolute])
                for key in records:
                    records[key][absolute] = dict(row)
            if not len(valid_ix):
                continue
            ids = [dev["sample_ids"][i] for i in valid_ix.tolist()]
            s64, _, _, sc, cc = phase4f_spatial_batch(p4_store, ids, device)
            evidence = fused_evidence(fusion, projection, fs_store, ids, device)
            valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
            taps, _, _ = recorder.run(s64, evidence, sc, cc, valid)
            logits = {"R2": probes["R2"](taps["R2"]), "Delta_F": probes["Delta_F"](taps["R6"])}
            for local, absolute in enumerate(valid_ix.tolist()):
                sid = ids[local]
                for key in records:
                    original = inverse_sam_logits(logits[key][local, 0][None, None], dev["sam_geometries"][absolute])
                    records[key][absolute] = {"sample_id": sid,
                                              **binary_metrics(original, dev["original_masks"][absolute].to(device)),
                                              "valid_g0": True}
    return records


def train_probes(arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device):
    root = OUT / "arms" / arm / "probe"
    if (root / "summary.json").exists():
        print(json.dumps({"arm": arm, "probe_status": "ALREADY_COMPLETE"}), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    model.eval().requires_grad_(False)
    recorder = TapRecorder(model)
    train_ids = p4_train.sample_ids
    targets = build_train_targets(train_ids)
    train_cache = load_c1_cache("train", train_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    seed_all()
    base = nn.Conv2d(256, 1, 1).to(device)
    probes = nn.ModuleDict({"R2": nn.Conv2d(256, 1, 1).to(device), "Delta_F": nn.Conv2d(256, 1, 1).to(device)})
    probes["R2"].load_state_dict(base.state_dict())
    probes["Delta_F"].load_state_dict(base.state_dict())
    optimizer = torch.optim.AdamW(probes.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    history, best = [], {}
    for epoch in range(1, PROBE_EPOCHS + 1):
        began = time.time()
        probes.train()
        order = torch.where(train_cache["valid"].bool())[0].tolist()
        random.Random(SEED + epoch).shuffle(order)
        total_loss, count = 0.0, 0
        for begin in range(0, len(order), PROBE_BATCH):
            ix = torch.tensor(order[begin:begin + PROBE_BATCH])
            ids = [train_ids[i] for i in ix.tolist()]
            s64, _, _, sc, cc = phase4f_spatial_batch(p4_train, ids, device)
            evidence = fused_evidence(fusion, projection, fs_train, ids, device)
            valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
            taps, _, _ = recorder.run(s64, evidence, sc, cc, valid)
            target = torch.stack([targets[s] for s in ids]).to(device)[:, None].float()
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(probes["R2"](taps["R2"]), target)["total"] + probe_loss(probes["Delta_F"](taps["R6"]), target)["total"]
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            count += len(ids)
        records = probe_validate(probes, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "train_total": total_loss / max(1, count),
               "R2": summarize_extended(records["R2"]), "Delta_F": summarize_extended(records["Delta_F"])}
        history.append(row)
        dump(root / "training_curve.json", history)
        for key in ("R2", "Delta_F"):
            score = (row[key]["mean_foreground_iou"], row[key]["mean_foreground_f1"])
            if key not in best or score > tuple(best[key]["score"]):
                best[key] = {"score": list(score), "epoch": epoch,
                             "state_dict": {k: v.detach().cpu() for k, v in probes[key].state_dict().items()}}
        print(json.dumps({"stage": "PROBE_EPOCH", "arm": arm, "epoch": epoch,
                          "R2": row["R2"]["mean_foreground_iou"],
                          "Delta_F": row["Delta_F"]["mean_foreground_iou"]}), flush=True)
    for key in ("R2", "Delta_F"):
        probes[key].load_state_dict(best[key]["state_dict"])
    torch.save({"schema": "phase6g10_probe_selected_v1", "arm": arm,
                "R2": best["R2"], "Delta_F": best["Delta_F"]}, root / "selected_probe.pt")
    records = probe_validate(probes, recorder, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    for key in ("R2", "Delta_F"):
        write_rows(root / "predictions" / f"{key}.jsonl", records[key])
    r2 = [row["foreground_iou"] for row in records["R2"]]
    delta = [row["foreground_iou"] for row in records["Delta_F"]]
    retention = paired_statistics(delta, r2)
    r2_metric = summarize_extended(records["R2"])
    delta_metric = summarize_extended(records["Delta_F"])
    dump(root / "summary.json", {
        "status": "COMPLETE",
        "arm": arm,
        "selected_epochs": {key: best[key]["epoch"] for key in ("R2", "Delta_F")},
        "R2": r2_metric,
        "Delta_F": delta_metric,
        "r2_to_delta_paired": retention,
        "r2_to_delta_ratio": delta_metric["mean_foreground_iou"] / max(1e-12, r2_metric["mean_foreground_iou"]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    fusion = load_fusion(device)
    projection = load_projection(device)
    sam = load_sam_runtime(hd.CFG, device)
    arms, gamma = matched_rectifiers(device)
    model = arms[args.arm]
    p4_train, p4_val = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    init_hashes = {"A0": tensor_state_sha256(arms["A0"].state_dict()),
                   "A1": tensor_state_sha256(arms["A1"].state_dict()),
                   "A2": tensor_state_sha256(arms["A2"].state_dict())}
    dump(OUT / "arms" / args.arm / "initialization.json",
         {"arm": args.arm, "gamma": gamma, "initial_hashes": init_hashes})
    train_rectifier(args.arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device)
    selected = torch.load(OUT / "arms" / args.arm / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    model.eval().requires_grad_(False)
    train_probes(args.arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device)
    arm_summary = {
        "status": "COMPLETE",
        "arm": args.arm,
        "gamma": gamma,
        "initial_hashes": init_hashes,
        "matrix_spectrum": matrix_spectrum(args.arm, model),
        "rectifier_selector": json.load(open(OUT / "arms" / args.arm / "selector.json")),
        "probe_summary": json.load(open(OUT / "arms" / args.arm / "probe" / "summary.json")),
        "firewall": {"utility": False, "joint_r1": False, "internal_test": False,
                     "official1000": False, "ood": False},
    }
    dump(OUT / "arms" / args.arm / "summary.json", arm_summary)
    print(json.dumps({"arm": args.arm, "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
