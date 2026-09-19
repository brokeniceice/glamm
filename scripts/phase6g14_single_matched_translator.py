#!/usr/bin/env python3
"""Phase 6G.14: factorized versus function-matched single affine translator."""
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

from scripts import phase6g10_train_arm as g10
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from tools.phase4e1 import tensor_state_sha256

OUT = ROOT / "outputs/phase6g14_single_matched_translator"
ARMS = ("A0", "A1")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def merge_affines(first: nn.Linear, second: nn.Linear) -> nn.Linear:
    """Return second(first(x)) as one exactly derived affine map."""
    if first.in_features != first.out_features or second.in_features != first.out_features:
        raise ValueError("Phase6G.14 expects matched square 256-D translators")
    merged = nn.Linear(first.in_features, second.out_features, bias=True)
    with torch.no_grad():
        merged.weight.copy_(second.weight.float() @ first.weight.float())
        merged.bias.copy_(second.weight.float() @ first.bias.float() + second.bias.float())
    return merged


def build_matched_arms(device):
    gamma = float(json.loads(g10.GAMMA_AUDIT.read_text())["selected_gamma"])
    g10.seed_all()
    base = g10.load_rectifier(g10.hd.CFG, gamma, device)
    a0 = copy.deepcopy(base)
    a1 = copy.deepcopy(base)
    single = merge_affines(
        a0.rectification.cross_attention.out_proj,
        a0.rectification.projection,
    ).to(device)
    a1.rectification.cross_attention.out_proj = nn.Identity()
    a1.rectification.projection = single
    return {"A0": a0, "A1": a1}, gamma


def step0_equivalence(a0, a1, device):
    gen = torch.Generator(device="cpu").manual_seed(g10.SEED + 1400)
    r2 = torch.randn(4, 4096, 256, generator=gen, dtype=torch.float32, device="cpu").to(device)
    with torch.no_grad():
        y0 = a0.rectification.projection(a0.rectification.cross_attention.out_proj(r2))
        y1 = a1.rectification.projection(a1.rectification.cross_attention.out_proj(r2))
        diff = (y0.float() - y1.float()).cpu()
        denom = y0.float().norm().clamp_min(1e-12)
        fp32 = {
            "max_abs_error": float(diff.abs().max()),
            "mean_abs_error": float(diff.abs().mean()),
            "relative_l2_error": float(diff.norm() / denom.cpu()),
        }
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z0 = a0.rectification.projection(a0.rectification.cross_attention.out_proj(r2))
            z1 = a1.rectification.projection(a1.rectification.cross_attention.out_proj(r2))
        bdiff = (z0.float() - z1.float()).cpu()
        bdenom = z0.float().norm().clamp_min(1e-12)
        bf16 = {
            "max_abs_error": float(bdiff.abs().max()),
            "mean_abs_error": float(bdiff.abs().mean()),
            "relative_l2_error": float(bdiff.norm() / bdenom.cpu()),
        }
    passed = fp32["max_abs_error"] <= 2e-6 and fp32["relative_l2_error"] <= 2e-6
    return {
        "schema": "phase6g14_step0_equivalence_v1",
        "definition": "A0 W2(W1 R2+b1)+b2 versus A1 W_single R2+b_single before common gamma/support",
        "fp32_primary_gate": fp32,
        "bf16_training_path_diagnostic": bf16,
        "tolerance": {"max_abs_error": 2e-6, "relative_l2_error": 2e-6},
        "passed": passed,
        "note": "BF16 is diagnostic only because factorized and merged evaluation have different rounding points.",
    }


def ownership(model):
    trainable = [{"name": n, "numel": p.numel(), "shape": list(p.shape)} for n, p in model.named_parameters() if p.requires_grad]
    frozen = [{"name": n, "numel": p.numel(), "shape": list(p.shape)} for n, p in model.named_parameters() if not p.requires_grad]
    return {
        "trainable_parameter_count": sum(x["numel"] for x in trainable),
        "frozen_parameter_count": sum(x["numel"] for x in frozen),
        "trainable": trainable,
        "frozen": frozen,
    }


def translator_norm(model, arm):
    with torch.no_grad():
        if arm == "A0":
            w1 = model.rectification.cross_attention.out_proj.weight.float()
            b1 = model.rectification.cross_attention.out_proj.bias.float()
            w2 = model.rectification.projection.weight.float()
            b2 = model.rectification.projection.bias.float()
            weight = w2 @ w1
            bias = w2 @ b1 + b2
        else:
            weight = model.rectification.projection.weight.float()
            bias = model.rectification.projection.bias.float()
        return {
            "effective_weight_frobenius": float(weight.norm()),
            "effective_bias_l2": float(bias.norm()),
            "gamma_main": float(model.rectification.gamma.detach()),
        }


def grad_norms(model):
    total = translator = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().square().sum())
        total += value
        if "cross_attention.out_proj" in name or "rectification.projection" in name:
            translator += value
    return math.sqrt(total), math.sqrt(translator)


def train_arm(arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device):
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
    total_steps = math.ceil(len(train_ids) / g10.BATCH) * g10.EPOCHS
    optimizer, scheduler = g10.optimizer_and_scheduler(parameters, total_steps)
    history_path, last_path = root / "training_curve.json", root / "last_state.pt"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    start_epoch, updates = 1, 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, updates = int(state["epoch"]) + 1, int(state["updates"])

    for epoch in range(start_epoch, g10.EPOCHS + 1):
        began = time.time()
        model.train()
        order = list(train_ids)
        random.Random(g10.SEED + 1009 * epoch).shuffle(order)
        loss_sum = grad_total_sum = grad_translator_sum = 0.0
        count = steps = 0
        for begin in range(0, len(order), g10.BATCH):
            block = order[begin:begin + g10.BATCH]
            positions = torch.tensor([where[sid] for sid in block])
            eligible = positions[valid.index_select(0, positions)]
            if not len(eligible):
                continue
            ids = [train_ids[i] for i in eligible.tolist()]
            qseg = train_cache["q_seg"].index_select(0, eligible).to(device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss = g10.rectifier_train_step(model, sam, fusion, projection, p4_train, fs_train, qseg, ids, device)
            loss["total"].backward()
            gn, tgn = grad_norms(model)
            torch.nn.utils.clip_grad_norm_(parameters, g10.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            updates += 1
            steps += 1
            loss_sum += float(loss["total"].detach()) * len(ids)
            grad_total_sum += gn
            grad_translator_sum += tgn
            count += len(ids)
            if updates == 1:
                dump(root / "first_update_runtime.json", {
                    "device": str(device),
                    "memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(device),
                    "loss": float(loss["total"].detach()),
                })
        metrics, records = g10.rectifier_evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "seg_loss": loss_sum / max(1, count),
            "dev_g0_mean_iou": metrics["mean_foreground_iou"],
            "dev_g0_mean_f1": metrics["mean_foreground_f1"],
            "gradient_norm_mean": grad_total_sum / max(1, steps),
            "translator_gradient_norm_mean": grad_translator_sum / max(1, steps),
            **translator_norm(model, arm),
            "seconds": time.time() - began,
        }
        history.append(row)
        dump(history_path, history)
        payload = {
            "schema": "phase6g14_arm_checkpoint_v1",
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
        g10.write_rows(root / "validation" / f"epoch_{epoch}.jsonl", records)
        print(json.dumps({"stage": "RECT_EPOCH", "arm": arm, **row}), flush=True)

    best = max(history, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    selected = root / "selected_checkpoint.pt"
    selected.write_bytes((root / "checkpoints" / f"epoch_{best['epoch']}.pt").read_bytes())
    selected_state = torch.load(selected, map_location="cpu", weights_only=False)
    model.load_state_dict(selected_state["model"])
    metrics, records = g10.rectifier_evaluate(model, sam, fusion, projection, p4_val, fs_val, dev, val_cache, device)
    g10.write_rows(root / "validation" / "selected.jsonl", records)
    dump(root / "selector.json", {
        "primary": "internal validation canonical G0 mean IoU; tie earlier epoch",
        "selected_epoch": best["epoch"], "selected_metrics": metrics, "candidates": history,
    })
    # Auxiliary probes are never selectors.
    old_out = g10.OUT
    try:
        g10.OUT = OUT
        g10.train_probes(arm, model, sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device)
    finally:
        g10.OUT = old_out
    dump(root / "summary.json", {
        "status": "COMPLETE",
        "arm": arm,
        "selector": json.loads((root / "selector.json").read_text()),
        "probe_summary": json.loads((root / "probe" / "summary.json").read_text()),
        "effective_map": g10.matrix_spectrum("A0" if arm == "A0" else "A1", model),
        "firewall": {"side": False, "utility": False, "nonlinear": False, "internal_test": False,
                     "official1000": False, "ood": False},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    arms, gamma = build_matched_arms(device)
    equivalence = step0_equivalence(arms["A0"], arms["A1"], device)
    dump(OUT / "step0_equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise RuntimeError(f"step-0 function equivalence failed: {equivalence}")
    init = {
        "schema": "phase6g14_initialization_v1",
        "gamma": gamma,
        "step0_equivalence": equivalence,
        "state_hashes": {key: tensor_state_sha256(value.state_dict()) for key, value in arms.items()},
        "ownership": {key: ownership(value) for key, value in arms.items()},
        "only_variable": "factorized double affine versus exactly merged single affine translator",
    }
    dump(OUT / "initialization_audit.json", init)
    fusion = g10.load_fusion(device)
    projection = g10.load_projection(device)
    sam = g10.load_sam_runtime(g10.hd.CFG, device)
    p4_train, p4_val = g10.Phase4FStore(g10.hd.CFG, "train"), g10.Phase4FStore(g10.hd.CFG, "val")
    fs_train, fs_val = g10.Store("train"), g10.Store("val")
    dev = g10.load_dev("g0")
    train_arm(args.arm, arms[args.arm], sam, fusion, projection, p4_train, fs_train, p4_val, fs_val, dev, device)
    print(json.dumps({"arm": args.arm, "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
