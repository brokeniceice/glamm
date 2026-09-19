#!/usr/bin/env python3
"""Phase 6G.13 Utility training on the frozen Phase6G.11 main+side Rectifier."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy import stats

from scripts import phase4g1q_conditional_utility as q, phase4hc_direct_utility_arms as hc, phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid, gated_embedding
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch, load_c1_cache, phase4f_spatial_batch
from scripts.phase6g7_staged_r1 import Store
from scripts.phase6g10_train_arm import load_fusion, load_projection, fused_evidence, matched_rectifiers, spectrum
from scripts.phase6g11_zero_init_side_path import ZeroInitSide
from scripts.phase6g12_train_arm import JointRectifier
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import inverse_sam_logits, metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_sam_runtime, mask_loss

OUT = ROOT / "outputs/phase6g13_utility_on_main_side_rectifier"
A0_MAIN = ROOT / "outputs/phase6g10_post_attention_projection_bypass/arms/A0/selected_checkpoint.pt"
A1_SIDE = ROOT / "outputs/phase6g11_zero_init_correction_side_path/training/selected_side.pt"
PROJECTION = ROOT / "outputs/phase6g8b_multilevel_evidence_interface_audit/interfaces/projection_256/selected_projection.pt"
A0_VAL = ROOT / "outputs/phase6g10_post_attention_projection_bypass/arms/A0/validation/selected.jsonl"
A1_VAL = ROOT / "outputs/phase6g11_zero_init_correction_side_path/training/validation/selected.jsonl"

SEED, EPOCHS, BATCH = 3407, 10, 8
LR, WD, CLIP = 1e-4, 1e-4, 1.0


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
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_head(device):
    payload = torch.load(PROJECTION, map_location="cpu", weights_only=False)
    head = nn.Conv2d(256, 1, 1, bias=True)
    state = {
        name.split("head.", 1)[1]: value
        for name, value in payload["training_head"].items()
        if name.startswith("head.")
    }
    head.load_state_dict(state, strict=True)
    return head.to(device).eval().requires_grad_(False)


def load_frozen_rectifier(device):
    payload = torch.load(A0_MAIN, map_location="cpu", weights_only=False)
    arms, _ = matched_rectifiers(device)
    main = arms["A0"]
    main.load_state_dict(payload["model"], strict=True)
    side_payload = torch.load(A1_SIDE, map_location="cpu", weights_only=False)
    side = ZeroInitSide().to(device)
    side.load_state_dict(side_payload["side"], strict=True)
    joint = JointRectifier(main, side).to(device).eval().requires_grad_(False)
    return joint, main, side


def parity_check(joint, fusion, projection, head, p4, fs, data, ids, device):
    s64, _, _, sc, cc = phase4f_spatial_batch(p4, ids, device)
    f24 = fused_evidence(fusion, projection, fs, ids, device)
    zf24 = head(f24.float())
    old = data["F24"][:len(ids)].to(device)
    oldz = data["z_F24"][:len(ids)].to(device)
    return {
        "f24_shape": list(f24.shape),
        "z_f24_shape": list(zf24.shape),
        "old_f24_shape": list(old.shape),
        "old_z_f24_shape": list(oldz.shape),
        "s64_max_abs_diff": float((s64.float() - data["S64"][:len(ids)].to(device).float()).abs().max()),
    }


def utility_batch(s64, qseg, zl, f24, zf24, geometries):
    return {
        "S64": s64,
        "q_seg": qseg,
        "z_L": zl,
        "F24": f24,
        "z_F24": zf24,
        "clip_geometries": geometries,
        "valid_g0": torch.ones(len(s64), dtype=torch.bool, device=s64.device),
        "forensic_present": torch.ones(len(s64), dtype=torch.bool, device=s64.device),
        "forensic_vacuous": torch.zeros(len(s64), dtype=torch.bool, device=s64.device),
        "forensic_off": torch.zeros(len(s64), dtype=torch.bool, device=s64.device),
    }


def train_step(model, joint, sam, fusion, projection, head, p4_store, fs_store, data, ids, positions, cache, cross, perm, device):
    s64, _, targets, sc, cc = phase4f_spatial_batch(p4_store, ids, device)
    batch, _ = c1_language_batch(data, cache, positions, p4_store, sam, device)
    f24 = fused_evidence(fusion, projection, fs_store, ids, device)
    zf24 = head(f24.float())
    batch["S64"] = s64
    batch["F24"] = f24
    batch["z_F24"] = zf24
    crossed_positions = cross.index_select(0, positions)
    crossed_ids = [data["sample_ids"][i] for i in crossed_positions.tolist()]
    cross_f24 = fused_evidence(fusion, projection, fs_store, crossed_ids, device)
    cross_zf24 = head(cross_f24.float())
    crossed = dict(batch)
    crossed["F24"] = cross_f24
    crossed["z_F24"] = cross_zf24
    matched = hc.utility_forward(model, batch)
    cross_out = hc.utility_forward(model, crossed)
    shuffle_out = hc.utility_forward(model, batch, permutation=perm)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        rectified = joint(s64, f24, sc, cc, torch.ones(len(ids), 576, dtype=torch.bool, device=device))
    support = rectified["support"].reshape(len(ids), 1, 64, 64)
    gate = gate_to_sam_grid(matched["U"], sc) * support.float()
    adapted = gated_embedding(s64, rectified["image_embeddings"], gate)
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
    seg = mask_loss(low, targets, hd.CFG)
    _, soft = q.target_delta(matched, data["target64"].index_select(0, positions).to(device))
    relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
    cr = hc.rank_loss(matched["U"], cross_out["U"], matched["support"])
    sr = hc.rank_loss(matched["U"], shuffle_out["U"], matched["support"])
    ranking = 0.5 * (cr + sr)
    total = seg["total"] + relative + ranking
    return total, {"seg": seg["total"], "relative": relative, "ranking": ranking, "cr": cr, "sr": sr}


def evaluate(model, joint, sam, fusion, projection, head, p4_store, fs_store, dev, cache, device):
    model.eval()
    records, gates = [], []
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                records.append(invalid_record(sid, dev["original_masks"][i]))
                gates.append(0.0)
                continue
            s64, _, _, sc, cc = phase4f_spatial_batch(p4_store, [sid], device)
            batch, _ = c1_language_batch(dev, cache, torch.tensor([i]), p4_store, sam, device)
            f24 = fused_evidence(fusion, projection, fs_store, [sid], device)
            zf24 = head(f24.float())
            batch["S64"] = s64
            batch["F24"] = f24
            batch["z_F24"] = zf24
            qseg = batch["q_seg"]
            out = hc.utility_forward(model, batch)
            rectified = joint(s64, f24, sc, cc, torch.ones(1, 576, dtype=torch.bool, device=device))
            support = rectified["support"].reshape(1, 1, 64, 64)
            gate = gate_to_sam_grid(out["U"], sc) * support.float()
            adapted = gated_embedding(s64, rectified["image_embeddings"], gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(qseg, adapted.to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]), dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
            gates.append(float(gate[support.bool()].mean()) if support.any() else 0.0)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "VAL", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    return summarize_extended(records), records, gates


def diagnostics(gate_values, a0_records, a1_records):
    a1 = {r["sample_id"]: float(r["foreground_iou"]) for r in a1_records}
    a0 = {r["sample_id"]: float(r["foreground_iou"]) for r in a0_records}
    ids = [r["sample_id"] for r in a1_records]
    benefit = np.asarray([a1[sid] - a0[sid] for sid in ids], dtype=np.float64)
    gate = np.asarray([gate_values[sid] for sid in ids], dtype=np.float64)
    help_mask = benefit > 0
    harm_mask = benefit < 0
    return {
        "n": len(ids),
        "gate": {"mean": float(gate.mean()), "median": float(np.median(gate)),
                 "q10": float(np.quantile(gate, 0.1)), "q25": float(np.quantile(gate, 0.25)),
                 "q75": float(np.quantile(gate, 0.75)), "q90": float(np.quantile(gate, 0.9))},
        "side_benefit": {"mean": float(benefit.mean()), "median": float(np.median(benefit)),
                         "positive_fraction": float(help_mask.mean())},
        "correlation": {
            "pearson": float(np.corrcoef(gate, benefit)[0, 1]) if gate.std() > 0 else None,
            "spearman": float(stats.spearmanr(gate, benefit).statistic) if gate.std() > 0 else None,
        },
        "side_help": {"n": int(help_mask.sum()), "gate_mean": float(gate[help_mask].mean()) if help_mask.any() else None,
                      "gate_median": float(np.median(gate[help_mask])) if help_mask.any() else None},
        "side_harm": {"n": int(harm_mask.sum()), "gate_mean": float(gate[harm_mask].mean()) if harm_mask.any() else None,
                      "gate_median": float(np.median(gate[harm_mask])) if harm_mask.any() else None},
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

    fusion = load_fusion(device)
    projection = load_projection(device)
    head = load_head(device)
    joint, main, side = load_frozen_rectifier(device)
    sam = load_sam_runtime(hd.CFG, device)
    utility, _ = hc.load_utility("a2", device)
    utility.train()
    for parameter in list(utility.language_source.parameters()) + list(utility.forensic_source.parameters()):
        parameter.requires_grad_(False)
    params = [p for p in utility.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 371803:
        raise RuntimeError("utility trainable parameter drift")
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

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
    perm = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    parity = parity_check(joint, fusion, projection, head, p4_train, fs_train, data, train_ids[:BATCH], device)
    dump(OUT / "protocol.json", {"schema": "phase6g13_protocol_v1", "status": "FROZEN_BEFORE_FIRST_STEP",
                                 "A0_reference_mean_iou": 0.185546, "input_parity": parity,
                                 "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD,
                                            "grad_clip": CLIP, "seed": SEED, "selector": "DEV G0 mean IoU",
                                            "loss": "seg+relative+ranking", "scheduler": None},
                                 "firewall": {"rectifier": False, "side": False, "adapter": False,
                                              "fusion": False, "joint_r1": False, "internal_test": False,
                                              "official1000": False, "ood": False}})

    history, best = [], None
    for epoch in range(1, EPOCHS + 1):
        began = time.time()
        utility.train()
        order = list(train_ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float)
        updates = 0
        for begin in range(0, len(order), BATCH):
            block = order[begin:begin + BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid.index_select(0, positions)]
            if not len(eligible):
                continue
            ids = [train_ids[i] for i in eligible.tolist()]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = train_step(utility, joint, sam, fusion, projection, head, p4_train, fs_train,
                                     data, ids, eligible, train_cache, cross, perm, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            updates += 1
            for key, value in parts.items():
                sums[key] += float(value.detach()) * len(ids)
            sums["samples"] += len(ids)
        metrics, records, gates = evaluate(utility, joint, sam, fusion, projection, head, p4_val, fs_val,
                                           dev, val_cache, device)
        gate_by_sample = {r["sample_id"]: g for r, g in zip(records, gates)}
        row = {"epoch": epoch, "seconds": time.time() - began, "optimizer_updates": updates,
               "seg_loss": sums["seg"] / max(1, sums["samples"]),
               "relative_loss": sums["relative"] / max(1, sums["samples"]),
               "ranking_loss": sums["ranking"] / max(1, sums["samples"]),
               "total_loss": (sums["seg"] + sums["relative"] + sums["ranking"]) / max(1, sums["samples"]),
               "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "dev_g0_mean_f1": metrics["mean_foreground_f1"]}
        history.append(row)
        dump(OUT / "training_curve.json", history)
        write_rows(OUT / "validation" / f"epoch_{epoch}.jsonl", records)
        dump(OUT / "utility_gate" / f"epoch_{epoch}.json", gate_by_sample)
        print(json.dumps({"stage": "UTILITY_EPOCH", **row}), flush=True)
        if best is None or (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]) > best["score"]:
            best = {"score": (row["dev_g0_mean_iou"], row["dev_g0_mean_f1"]), "epoch": epoch,
                    "model": {k: v.detach().cpu() for k, v in utility.state_dict().items()},
                    "gate": gate_by_sample, "records": records}
    utility.load_state_dict(best["model"])
    torch.save({"schema": "phase6g13_utility_selected_v1", "epoch": best["epoch"],
                "model": best["model"]}, OUT / "selected_utility.pt")
    write_rows(OUT / "validation" / "selected.jsonl", best["records"])
    a0_records, a1_records = rows(A0_VAL), rows(A1_VAL)
    paired = paired_statistics(
        torch.tensor([r["foreground_iou"] for r in best["records"]]),
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
    diag = diagnostics(best["gate"], a0_records, a1_records)
    result = {"schema": "phase6g13_results_v1", "status": "COMPLETE_STOP", "decision": decision,
              "selected_epoch": best["epoch"], "A0_reference": 0.185546,
              "A1_utility": summarize_extended(best["records"]),
              "paired_A1_minus_A0": paired, "utility_diagnostics": diag,
              "seg_trigger_invariance": {"valid": int(val_cache["valid"].sum()),
                                         "n": len(val_cache["valid"]),
                                         "rate": float(val_cache["valid"].float().mean())},
              "firewall": {"rectifier": False, "side": False, "adapter": False, "fusion": False,
                           "joint_r1": False, "internal_test": False, "official1000": False, "ood": False}}
    dump(OUT / "results.json", result)
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                "selected_epoch": best["epoch"], "firewall": result["firewall"]})
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)


def render(result):
    lines = [
        "# Phase 6G.13 — Utility Training on Frozen Main+Side Rectifier",
        "",
        "Status: **COMPLETE STOP**.",
        "",
        "| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |",
        "|---|---:|---:|---:|",
        f"| A0 frozen main+side (no Utility) | {result['A0_reference']:.6f} | - | - |",
        f"| A1 same Rectifier + trained Utility | {result['A1_utility']['mean_foreground_iou']:.6f} | "
        f"{result['A1_utility']['median_foreground_iou']:.6f} | {result['A1_utility']['mean_foreground_f1']:.6f} |",
        "",
        f"- paired A1 - A0: `{result['paired_A1_minus_A0']}`",
        f"- utility diagnostics: `{result['utility_diagnostics']}`",
        f"- SEG trigger invariance: `{result['seg_trigger_invariance']}`",
        "",
        f"```text\n{result['decision']}\n```",
        "",
        "No Rectifier/side/Adapter/fusion training, internal test, Official1000, or OOD was accessed.",
    ]
    (ROOT / "docs/phase6g13_utility_on_main_side_rectifier.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
