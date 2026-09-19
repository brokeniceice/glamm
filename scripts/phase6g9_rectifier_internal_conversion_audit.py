#!/usr/bin/env python3
"""Phase 6G.9 Rectifier internal conversion audit.

Read-only frozen forward with intermediate tap points.  No architecture,
Rectifier, Adapter, Utility, or R1 weights are modified.  Only fresh matched
1x1 Conv probes and diagnostic statistics are produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_forensic_adapter import CLIPSpatialArm
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, phase4f_spatial_batch
from scripts.phase6g0_multilevel_dense_clip import cache_paths, load_shard
from scripts.phase6g7_staged_r1 import Store, sources
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, probe_loss, summarize
from tools.phase4c_b import file_sha256, inverse_sam_logits
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, load_evidence_source, load_rectifier

OUT = ROOT / "outputs/phase6g9_rectifier_internal_conversion_audit"
DOC = ROOT / "docs/phase6g9_rectifier_internal_conversion_audit.md"
A0_CKPT = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
A1_CKPT = ROOT / "outputs/phase6g7_block11_17_staged_r1/joint/selected_checkpoint.pt"
GAMMA_AUDIT = ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json"

SEED, EPOCHS, BATCH = 3407, 20, 8
LR, WEIGHT_DECAY = 1e-3, 0.0
ARMS = ("A0", "A1")
STAGES = ("R0", "R1_k", "R1_v", "R2", "R3", "R4", "R5", "R6")
CLIP_STAGES = frozenset(("R0", "R1_k", "R1_v"))
PROBE_KEYS = tuple(f"{arm}__{stage}" for arm in ARMS for stage in STAGES)
GAIN_SEQUENCE = ("R0", "R1_v", "R2", "R3", "R4", "R5", "R6")


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


def sha_ids(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def invalid_record(sid, truth):
    truth = torch.as_tensor(truth).bool()
    total = int(truth.numel())
    fn = int(truth.sum())
    tn = total - fn
    bg_iou = tn / max(1, tn + fn)
    return {
        "sample_id": sid,
        "foreground_iou": 0.0,
        "foreground_f1": 0.0,
        "background_iou": float(bg_iou),
        "fg_bg_miou": float(bg_iou / 2.0),
        "tp": 0,
        "fp": 0,
        "fn": fn,
        "tn": tn,
        "valid_g0": False,
    }


def build_train_targets(train_ids):
    lookup = {}
    for path in cache_paths("late", "train"):
        shard = load_shard(path)
        for local, row in enumerate(shard["records"]):
            lookup[str(row["sample_id"])] = shard["targets"][local]
    if set(lookup) != set(train_ids) or len(lookup) != len(train_ids):
        raise RuntimeError("train CLIP target/sample ID drift")
    return lookup


def build_val_geometry(dev):
    lookup = {}
    for path in cache_paths("late", "val"):
        shard = load_shard(path)
        for local, row in enumerate(shard["records"]):
            lookup[str(row["sample_id"])] = row["geometry"]
    if set(lookup) != set(dev["sample_ids"]) or len(lookup) != len(dev["sample_ids"]):
        raise RuntimeError("validation CLIP geometry/sample ID drift")
    return lookup


def load_runtimes(device):
    scale = json.load(open(GAMMA_AUDIT))
    gamma = float(scale["selected_gamma"])
    a0_source = load_evidence_source(hd.CFG, "forensic_rect", device)
    fusion, a1_adapter, _, _ = sources(device)
    for module in (a0_source, fusion, a1_adapter):
        module.eval().requires_grad_(False)

    def rectifier(path):
        model = load_rectifier(hd.CFG, gamma, device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["rectifier_state"], strict=True)
        return model.to(device).eval().requires_grad_(False), payload

    a0_rectifier, a0_payload = rectifier(A0_CKPT)
    a1_rectifier, a1_payload = rectifier(A1_CKPT)
    return {
        "gamma": gamma,
        "a0_source": a0_source,
        "fusion": fusion,
        "a1_adapter": a1_adapter,
        "a0_rectifier": a0_rectifier,
        "a1_rectifier": a1_rectifier,
        "a0_payload": a0_payload,
        "a1_payload": a1_payload,
    }


class TapRecorder:
    """Read-only forward hooks for the actual CrossAttentiveSemanticRectification graph."""

    def __init__(self, rectifier):
        self.rectifier = rectifier
        self.store = {}
        handles = []
        ca = rectifier.rectification.cross_attention
        handles.append(ca.k_proj.register_forward_hook(lambda m, i, o: self.store.__setitem__("k", o.detach())))
        handles.append(ca.v_proj.register_forward_hook(lambda m, i, o: self.store.__setitem__("v", o.detach())))
        handles.append(ca.out_proj.register_forward_pre_hook(lambda m, a: self.store.__setitem__("r2", a[0].detach())))
        handles.append(ca.out_proj.register_forward_hook(lambda m, i, o: self.store.__setitem__("r3", o.detach())))
        handles.append(ca.register_forward_pre_hook(lambda m, a: self.store.__setitem__("rq", a[0].detach())))
        handles.append(ca.register_forward_hook(lambda m, i, o: self.store.__setitem__("attention", o[1].detach())))
        handles.append(
            rectifier.rectification.projection.register_forward_hook(
                lambda m, i, o: self.store.__setitem__("r4_proj", o.detach())
            )
        )
        self.handles = handles

    def clear(self):
        self.store.clear()

    def run(self, s64, evidence, sam_coordinates, evidence_coordinates, evidence_valid):
        self.clear()
        with torch.no_grad(), torch.autocast(device_type=s64.device.type, dtype=torch.bfloat16):
            out = self.rectifier(s64, evidence, sam_coordinates, evidence_coordinates, evidence_valid)
        batch = evidence.shape[0]
        gamma = self.rectifier.rectification.gamma.detach()
        taps = {
            "R0": evidence.detach().float(),
            "R1_k": self.store["k"].transpose(1, 2).reshape(batch, 256, 24, 24).float(),
            "R1_v": self.store["v"].transpose(1, 2).reshape(batch, 256, 24, 24).float(),
            "R2": self.store["r2"].transpose(1, 2).reshape(batch, 256, 64, 64).float(),
            "R3": self.store["r3"].transpose(1, 2).reshape(batch, 256, 64, 64).float(),
            "R4": (gamma * self.store["r4_proj"]).transpose(1, 2).reshape(batch, 256, 64, 64).float(),
            "R5": out["image_embeddings"].detach().float(),
            "R6": out["residual"].detach().float(),
        }
        diag = {
            "attention": self.store["attention"].detach().float(),
            "support": out["support"].detach(),
            "Rq": self.store["rq"].detach().float(),
        }
        return taps, diag, out


def evidence_batch(runtime, p4_store, fs_store, ids, device):
    s64, raw, _, sam_coordinates, clip_coordinates = phase4f_spatial_batch(p4_store, ids, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        a0 = runtime["a0_source"](raw, return_features=True)["F_forensic"]
        fused = runtime["fusion"](*fs_store.batch(ids, device))
        a1 = runtime["a1_adapter"](fused, return_features=True)["F_forensic"]
    return s64, sam_coordinates, clip_coordinates, a0.detach(), a1.detach()


def make_probes(device):
    seed_all()
    probes = nn.ModuleDict()
    for stage in STAGES:
        base = nn.Conv2d(256, 1, 1).to(device)
        for arm in ARMS:
            probe = nn.Conv2d(256, 1, 1).to(device)
            probe.load_state_dict(base.state_dict())
            probes[f"{arm}__{stage}"] = probe
    return probes


def probe_predictions(probes, taps, arm):
    return {stage: probes[f"{arm}__{stage}"](taps[stage]) for stage in STAGES}


def records_for_batch(probes, taps, arm, ids, dev, absolute, val_geometry, val_cache, device):
    logits = probe_predictions(probes, taps, arm)
    output = {stage: {} for stage in STAGES}
    for j, (sid, index) in enumerate(zip(ids, absolute.tolist())):
        truth = dev["original_masks"][index].to(device)
        for stage in STAGES:
            value = logits[stage][j, 0]
            if stage in CLIP_STAGES:
                original = inverse_logits(value[None, None], val_geometry[sid])
            else:
                original = inverse_sam_logits(value[None, None], dev["sam_geometries"][index])
            output[stage][sid] = {"sample_id": sid, **binary_metrics(original, truth), "valid_g0": True}
    return output


def validate(probes, runtime, recorders, p4_store, fs_store, dev, val_cache, val_geometry, device):
    n_total = len(dev["sample_ids"])
    records = {key: [None] * n_total for key in PROBE_KEYS}
    probes.eval()
    with torch.no_grad():
        for begin in range(0, n_total, BATCH):
            raw_ix = torch.arange(begin, min(begin + BATCH, n_total))
            valid_ix = raw_ix[val_cache["valid"].index_select(0, raw_ix).bool()]
            for absolute in raw_ix.tolist():
                if bool(val_cache["valid"][absolute]):
                    continue
                sid = dev["sample_ids"][absolute]
                row = invalid_record(sid, dev["original_masks"][absolute])
                for key in PROBE_KEYS:
                    records[key][absolute] = dict(row)
            if not len(valid_ix):
                continue
            ids = [dev["sample_ids"][i] for i in valid_ix.tolist()]
            s64, sc, cc, a0_evidence, a1_evidence = evidence_batch(runtime, p4_store, fs_store, ids, device)
            valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
            taps0, _, _ = recorders["A0"].run(s64, a0_evidence, sc, cc, valid)
            taps1, _, _ = recorders["A1"].run(s64, a1_evidence, sc, cc, valid)
            predicted0 = records_for_batch(probes, taps0, "A0", ids, dev, valid_ix, val_geometry, val_cache, device)
            predicted1 = records_for_batch(probes, taps1, "A1", ids, dev, valid_ix, val_geometry, val_cache, device)
            for local, absolute in enumerate(valid_ix.tolist()):
                sid = ids[local]
                for stage in STAGES:
                    records[f"A0__{stage}"][absolute] = predicted0[stage][sid]
                    records[f"A1__{stage}"][absolute] = predicted1[stage][sid]
            if (begin + len(raw_ix)) % 200 < BATCH:
                print(json.dumps({"stage": "VAL", "done": begin + len(raw_ix), "total": n_total}), flush=True)
    if any(any(value is None for value in values) for values in records.values()):
        raise RuntimeError("validation record coverage drift")
    return records


def stable_positive(paired):
    return paired["mean_difference"] > 0 and paired["bootstrap_95_ci"][0] > 0 and paired["wilcoxon_pvalue"] < 0.05


def stable_negative(paired):
    return paired["mean_difference"] < 0 and paired["bootstrap_95_ci"][1] < 0 and paired["wilcoxon_pvalue"] < 0.05


def stage_comparisons(records):
    out = {}
    for stage in STAGES:
        a0 = np.asarray([row["foreground_iou"] for row in records[f"A0__{stage}"]], dtype=np.float64)
        a1 = np.asarray([row["foreground_iou"] for row in records[f"A1__{stage}"]], dtype=np.float64)
        f0 = np.asarray([row["foreground_f1"] for row in records[f"A0__{stage}"]], dtype=np.float64)
        f1 = np.asarray([row["foreground_f1"] for row in records[f"A1__{stage}"]], dtype=np.float64)
        out[stage] = {
            "A0": summarize(records[f"A0__{stage}"]),
            "A1": summarize(records[f"A1__{stage}"]),
            "paired_iou": paired_statistics(a1, a0),
            "paired_f1": paired_statistics(f1, f0),
        }
    return out


def decide(comparisons):
    def stable(stage):
        return stable_positive(comparisons[stage]["paired_iou"])

    if not stable("R0"):
        return "NO_STABLE_RECTIFIER_INPUT_GAIN", None
    if not stable("R1_k") and not stable("R1_v"):
        return "KV_PROJECTION_IS_PRIMARY_RECTIFIER_BOTTLENECK", "R1"
    if not stable("R2"):
        return "CROSS_ATTENTION_IS_PRIMARY_RECTIFIER_BOTTLENECK", "R2"
    if not stable("R3"):
        return "POST_ATTENTION_PROJECTION_IS_PRIMARY_RECTIFIER_BOTTLENECK", "R3"
    if not stable("R4") or not stable("R5"):
        return "POST_ATTENTION_PROJECTION_IS_PRIMARY_RECTIFIER_BOTTLENECK", "R4/R5"
    if not stable("R6"):
        return "DELTA_HEAD_IS_PRIMARY_RECTIFIER_BOTTLENECK", "R6"
    return "RECTIFIER_GAIN_PRESERVED_TO_DELTA_F", None


def retention(comparisons):
    rows = []
    gains = {stage: comparisons[stage]["paired_iou"]["mean_difference"] for stage in GAIN_SEQUENCE}
    stable = {stage: stable_positive(comparisons[stage]["paired_iou"]) for stage in GAIN_SEQUENCE}
    for previous, current in zip(GAIN_SEQUENCE, GAIN_SEQUENCE[1:]):
        if stable[previous] and stable[current]:
            value = gains[current] / gains[previous] if abs(gains[previous]) > 1e-12 else None
            rows.append({"previous": previous, "current": current, "gain_previous": gains[previous],
                         "gain_current": gains[current], "retention": value, "interpretable": True})
        else:
            rows.append({"previous": previous, "current": current, "gain_previous": gains[previous],
                         "gain_current": gains[current], "retention": None, "interpretable": False})
    return rows


def feature_stats(value):
    x = value.float()
    l2 = x.pow(2).sum(1).sqrt().mean()
    channel_var = x.var(dim=(2, 3), unbiased=False).mean()
    spatial_var = x.var(dim=1, unbiased=False).mean()
    return float(l2), float(channel_var), float(spatial_var)


def cosine_map(a, b):
    a = a.float() if a.ndim == 4 else a.float()[None]
    b = b.float() if b.ndim == 4 else b.float()[None]
    if a.shape != b.shape:
        raise RuntimeError(f"cosine_map shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float(F.cosine_similarity(a, b, dim=1).mean())


def resize_like(a, b):
    if a.shape[-2:] == b.shape[-2:]:
        return a
    value = a if a.ndim == 4 else a[None]
    return F.interpolate(value, size=b.shape[-2:], mode="bilinear", align_corners=False)


def sample_diagnostics(taps, diag, s64, cc, gt_mask, device):
    stats = {"feature": {}, "pair": {}, "attention": {}}
    for stage, value in taps.items():
        l2, channel_var, spatial_var = feature_stats(value)
        stats["feature"][stage] = {"l2": l2, "channel_var": channel_var, "spatial_var": spatial_var}
    stats["pair"]["R0_R1_k"] = cosine_map(taps["R0"], taps["R1_k"])
    stats["pair"]["R0_R1_v"] = cosine_map(taps["R0"], taps["R1_v"])
    stats["pair"]["R1_v_R2"] = cosine_map(resize_like(taps["R1_v"], taps["R2"]), taps["R2"])
    stats["pair"]["R2_R3"] = cosine_map(taps["R2"], taps["R3"])
    stats["pair"]["R3_R4"] = cosine_map(taps["R3"], taps["R4"])
    stats["pair"]["R3_R6"] = cosine_map(taps["R3"], taps["R6"])
    rq = diag["Rq"].transpose(1, 2).reshape(1, 256, 64, 64)
    stats["pair"]["R2_Rq"] = cosine_map(taps["R2"], rq)
    stats["pair"]["R3_Rq"] = cosine_map(taps["R3"], rq)
    stats["pair"]["R6_S_base"] = cosine_map(taps["R6"], s64)
    stats["pair"]["R6_S_base_ratio"] = float(
        taps["R6"].float().pow(2).sum(1).sqrt().mean()
        / s64.float().pow(2).sum(1).sqrt().mean().clamp_min(1e-12)
    )

    attention = diag["attention"][0]
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
    max_weight = attention.max(-1).values
    effective = entropy.exp()
    x, y = cc[0, :, 0], cc[0, :, 1]
    mean_x = torch.einsum("hqk,k->hq", attention, x)
    mean_y = torch.einsum("hqk,k->hq", attention, y)
    var_x = torch.einsum("hqk,hqk->hq", attention, (x[None, None] - mean_x[..., None]).square())
    var_y = torch.einsum("hqk,hqk->hq", attention, (y[None, None] - mean_y[..., None]).square())
    spatial_variance = var_x + var_y
    mask = gt_mask.float()[None, None]
    grid = (cc[0] * 2 - 1)[None, :, None, :]
    gt_key = F.grid_sample(mask, grid, mode="nearest", align_corners=False)[0, 0, :, 0] > 0.5
    gt_mass = attention[..., gt_key].sum(-1) / attention.sum(-1).clamp_min(1e-12)
    stats["attention"] = {
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "max_weight_mean": float(max_weight.mean()),
        "effective_mean": float(effective.mean()),
        "spatial_variance_mean": float(spatial_variance.mean()),
        "gt_mass_mean": float(gt_mass.mean()),
        "non_gt_mass_mean": float(1.0 - gt_mass.mean()),
        "entropy_per_head": [float(v) for v in entropy.mean(1)],
        "max_weight_per_head": [float(v) for v in max_weight.mean(1)],
        "effective_per_head": [float(v) for v in effective.mean(1)],
        "spatial_variance_per_head": [float(v) for v in spatial_variance.mean(1)],
        "gt_mass_per_head": [float(v) for v in gt_mass.mean(1)],
    }
    return stats


def aggregate_diagnostics(values):
    keys = values[0].keys()
    result = {}
    for key in keys:
        array = np.asarray([row[key] for row in values], dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def diagnostics_pass(runtime, recorders, p4_store, fs_store, dev, val_cache, device):
    feature = {arm: {stage: [] for stage in STAGES} for arm in ARMS}
    pair = {arm: [] for arm in ARMS}
    attention = {arm: [] for arm in ARMS}
    for begin in range(0, len(dev["sample_ids"]), BATCH):
        raw_ix = torch.arange(begin, min(begin + BATCH, len(dev["sample_ids"])))
        valid_ix = raw_ix[val_cache["valid"].index_select(0, raw_ix).bool()]
        if not len(valid_ix):
            continue
        ids = [dev["sample_ids"][i] for i in valid_ix.tolist()]
        s64, sc, cc, a0_evidence, a1_evidence = evidence_batch(runtime, p4_store, fs_store, ids, device)
        valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
        taps0, diag0, _ = recorders["A0"].run(s64, a0_evidence, sc, cc, valid)
        taps1, diag1, _ = recorders["A1"].run(s64, a1_evidence, sc, cc, valid)
        for local, absolute in enumerate(valid_ix.tolist()):
            for arm, taps, diag in (("A0", taps0, diag0), ("A1", taps1, diag1)):
                current = {stage: value[local:local + 1] for stage, value in taps.items()}
                current_diag = {
                    "attention": diag["attention"][local:local + 1],
                    "support": diag["support"][local:local + 1],
                    "Rq": diag["Rq"][local:local + 1],
                }
                stats = sample_diagnostics(
                    current, current_diag, s64[local:local + 1], cc[local:local + 1],
                    dev["original_masks"][absolute].to(device), device,
                )
                for stage in STAGES:
                    feature[arm][stage].append(stats["feature"][stage])
                pair[arm].append(stats["pair"])
                attention[arm].append(stats["attention"])
        if (begin + len(raw_ix)) % 200 < BATCH:
            print(json.dumps({"stage": "DIAG", "done": begin + len(raw_ix), "total": len(dev["sample_ids"])}), flush=True)
    result = {"feature": {}, "pair": {}, "attention": {}}
    for arm in ARMS:
        result["feature"][arm] = {stage: aggregate_diagnostics(feature[arm][stage]) for stage in STAGES}
        result["pair"][arm] = aggregate_diagnostics(pair[arm])
        agg = {}
        for key in attention[arm][0]:
            array = np.asarray([row[key] for row in attention[arm]], dtype=np.float64)
            if key.endswith("_per_head"):
                agg[key] = {
                    "mean_per_head": [float(x) for x in array.mean(0)],
                    "median_per_head": [float(x) for x in np.median(array, axis=0)],
                    "std_per_head": [float(x) for x in array.std(0)],
                }
            else:
                agg[key] = {"mean": float(array.mean()), "median": float(np.median(array)),
                            "std": float(array.std()), "min": float(array.min()), "max": float(array.max())}
        result["attention"][arm] = agg
    return result


def render(result):
    labels = {
        "R0": "R0 evidence input",
        "R1_k": "R1 K projected",
        "R1_v": "R1 V projected",
        "R2": "R2 raw cross-attn output",
        "R3": "R3 output projected",
        "R4": "R4 residual pre-support",
        "R5": "R5 post-fusion (rectified)",
        "R6": "R6 Delta_F (residual)",
    }
    lines = [
        "# Phase 6G.9 — Rectifier Internal Conversion Audit",
        "",
        "Status: **COMPLETE STOP**. Only frozen forward, intermediate tensor export, fresh matched 1x1 Conv probes, and diagnostics were used. No architecture, Rectifier, Adapter, Utility, or R1 weights were modified.",
        "",
        "## Actual forward graph",
        "",
        "```text",
        "F_forensic [B,256,24,24]",
        "        ↓ flatten + forensic_norm + coordinate_sincos",
        "forensic K/V input [B,576,256]",
        "        ↓ k_proj / v_proj",
        "R1_k / R1_v [B,256,24,24]",
        "SAM semantic [B,256,64,64]",
        "        ↓ flatten + semantic_norm + coordinate_sincos",
        "SAM query [B,4096,256]",
        "        ↓ geometry-aware cross-attention (8 heads, 4096x576)",
        "R2 raw attention output [B,256,64,64]",
        "        ↓ out_proj",
        "R3 output projected [B,256,64,64]",
        "        ↓ projection + gamma + support mask",
        "R4 residual pre-support [B,256,64,64]",
        "        ↓ semantic + residual",
        "R5 post-fusion rectified [B,256,64,64]",
        "        ↓ support-masked residual",
        "R6 Delta_F [B,256,64,64]",
        "```",
        "",
        "The graph is taken from the actual `GeometryAwareSAMRectifier.forward`, `CrossAttentiveSemanticRectification.forward`, and `GeometryAwareCrossAttention.forward` implementations, not from documentation.",
        "",
        "## Core table (full 1106-sample validation)",
        "",
        "| Stage | A0 mean IoU | A1 mean IoU | A1-A0 Delta | IoU 95% CI | W/T/L | Wilcoxon p | A0 median IoU | A1 median IoU | A0 F1 | A1 F1 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        item = result["stage_comparisons"][stage]
        a0, a1, p = item["A0"], item["A1"], item["paired_iou"]
        lines.append(
            f"| {labels[stage]} | {a0['mean_foreground_iou']:.6f} | {a1['mean_foreground_iou']:.6f} | "
            f"{p['mean_difference']:+.6f} | [{p['bootstrap_95_ci'][0]:.6f}, {p['bootstrap_95_ci'][1]:.6f}] | "
            f"{p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} | "
            f"{a0['median_foreground_iou']:.6f} | {a1['median_foreground_iou']:.6f} | "
            f"{a0['mean_foreground_f1']:.6f} | {a1['mean_foreground_f1']:.6f} |"
        )
    lines += [
        "",
        "## Gain retention",
        "",
        "| Transition | Gain previous | Gain current | Retention | Interpretable |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["gain_retention"]:
        value = "not interpretable" if row["retention"] is None else f"{row['retention']:.4f}"
        lines.append(
            f"| {row['previous']} -> {row['current']} | {row['gain_previous']:+.6f} | "
            f"{row['gain_current']:+.6f} | {value} | {row['interpretable']} |"
        )
    lines += [
        "",
        "## Cross-attention diagnostics",
        "",
        "| Arm | Entropy mean | Max weight mean | Effective tokens mean | Spatial variance mean | GT-related mass mean | Non-GT mass mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = result["attention_diagnostics"][arm]
        lines.append(
            f"| {arm} | {item['entropy_mean']['mean']:.6f} | {item['max_weight_mean']['mean']:.6f} | "
            f"{item['effective_mean']['mean']:.6f} | {item['spatial_variance_mean']['mean']:.6f} | "
            f"{item['gt_mass_mean']['mean']:.6f} | {item['non_gt_mass_mean']['mean']:.6f} |"
        )
    lines += [
        "",
        "## Information transformation diagnostics",
        "",
        "Per-stage feature L2, channel variance, and spatial variance are stored in `transformation_diagnostics.json`. Key pairwise cosine and norm ratios:",
        "",
        "| Arm | R0-R1_v cos | R1_v-R2 cos | R2-R3 cos | R3-R6 cos | R6-S_base cos | ||R6||/||S_base|| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        pair = result["transformation_diagnostics"]["pair"][arm]
        ratio = result["transformation_diagnostics"]["pair"][arm]["R6_S_base_ratio"]
        lines.append(
            f"| {arm} | {pair['R0_R1_v']['mean']:.6f} | {pair['R1_v_R2']['mean']:.6f} | "
            f"{pair['R2_R3']['mean']:.6f} | {pair['R3_R6']['mean']:.6f} | "
            f"{pair['R6_S_base']['mean']:.6f} | {ratio['mean']:.6f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- decision: `{result['decision']}`",
        f"- first break: `{result['first_break']}`",
        f"- stable rule: positive requires mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05.",
        "",
        "```text",
        result["decision"],
        "```",
        "",
        "No internal test, Official1000, or OOD data were accessed.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


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

    runtime = load_runtimes(device)
    recorders = {arm: TapRecorder(runtime[f"{arm.lower()}_rectifier"]) for arm in ARMS}
    p4_train, p4_val = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    fs_train, fs_val = Store("train"), Store("val")
    dev = load_dev("g0")
    train_ids, val_ids = p4_train.sample_ids, dev["sample_ids"]
    if fs_train.ids != train_ids or fs_val.ids != val_ids:
        raise RuntimeError("multilevel source pairing drift")
    train_cache = load_c1_cache("train", train_ids)
    val_cache = load_c1_cache("val", val_ids)
    train_targets = build_train_targets(train_ids)
    val_geometry = build_val_geometry(dev)

    frozen_before = {
        "a0_source": tensor_state_sha256(runtime["a0_source"].state_dict()),
        "a1_fusion": tensor_state_sha256(runtime["fusion"].state_dict()),
        "a1_adapter": tensor_state_sha256(runtime["a1_adapter"].state_dict()),
        "a0_rectifier": tensor_state_sha256(runtime["a0_rectifier"].state_dict()),
        "a1_rectifier": tensor_state_sha256(runtime["a1_rectifier"].state_dict()),
    }
    protocol = {
        "schema": "phase6g9_protocol_v1",
        "status": "FROZEN_BEFORE_FIRST_STEP",
        "forward_graph": {
            "source_files": ["model/sam_forensic_rectifier.py", "model/tf_fdg.py"],
            "actual_modules": {
                "rectifier": "GeometryAwareSAMRectifier",
                "interaction": "CrossAttentiveSemanticRectification",
                "attention": "GeometryAwareCrossAttention",
                "q_proj": "GeometryAwareCrossAttention.q_proj",
                "k_proj": "GeometryAwareCrossAttention.k_proj",
                "v_proj": "GeometryAwareCrossAttention.v_proj",
                "out_proj": "GeometryAwareCrossAttention.out_proj",
                "residual_projection": "CrossAttentiveSemanticRectification.projection",
                "gamma": "CrossAttentiveSemanticRectification.gamma",
            },
            "tensor_flow": [
                {"stage": "semantic input", "shape": ["B", 256, 64, 64], "module": "GeometryAwareSAMRectifier.forward image_embeddings"},
                {"stage": "forensic input", "shape": ["B", 256, 24, 24], "module": "GeometryAwareSAMRectifier.forward evidence"},
                {"stage": "semantic tokens", "shape": ["B", 4096, 256], "module": "flatten + semantic_norm + coordinate_sincos"},
                {"stage": "forensic tokens", "shape": ["B", 576, 256], "module": "flatten + forensic_norm + coordinate_sincos"},
                {"stage": "K/V projection", "shape": ["B", 576, 256], "module": "k_proj / v_proj"},
                {"stage": "attention scores", "shape": ["B", 8, 4096, 576], "module": "GeometryAwareCrossAttention.forward"},
                {"stage": "raw attention output", "shape": ["B", 4096, 256], "module": "attention @ v before out_proj"},
                {"stage": "output projection", "shape": ["B", 4096, 256], "module": "out_proj"},
                {"stage": "residual projection", "shape": ["B", 4096, 256], "module": "projection + gamma + support mask"},
                {"stage": "rectified fusion", "shape": ["B", 4096, 256], "module": "semantic + residual"},
                {"stage": "Delta_F", "shape": ["B", 256, 64, 64], "module": "rectifier residual grid"},
            ],
        },
        "population": {
            "train": len(train_ids),
            "train_valid_seg": int(train_cache["valid"].sum()),
            "validation": len(val_ids),
            "validation_valid_seg": int(val_cache["valid"].sum()),
            "train_ids_sha256": sha_ids(train_ids),
            "validation_ids_sha256": sha_ids(val_ids),
        },
        "A0": {
            "definition": "current block22 selected new R1",
            "r1_checkpoint": str(A0_CKPT.resolve()),
            "r1_checkpoint_sha256": file_sha256(A0_CKPT),
            "evidence_source": str(Path(hd.CFG["evidence"]["forensic_checkpoint"]).resolve()),
        },
        "A1": {
            "definition": "block11+17 staged new R1",
            "r1_checkpoint": str(A1_CKPT.resolve()),
            "r1_checkpoint_sha256": file_sha256(A1_CKPT),
            "evidence_source": "Phase6G.2 selected fusion + Phase6G.3 A0 adapter",
        },
        "probe": {
            "architecture": "fresh Conv2d(256,1,1) per arm/stage; matched initialization for A0/A1",
            "epochs": EPOCHS,
            "batch": BATCH,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss": "sum over 16 probes of 2*BCE+0.5*Dice",
            "selector": "validation mean FG IoU; tie mean FG F1",
            "threshold": 0.0,
            "seed": SEED,
        },
        "geometry": "R0/R1 use legacy CLIP crop geometry; R2-R6 use SAM long-side-pad geometry",
        "frozen_before": frozen_before,
        "firewall": {
            "architecture_modification": False,
            "rectifier_retraining": False,
            "adapter_retraining": False,
            "utility_retraining": False,
            "r1_joint_training": False,
            "internal_test": False,
            "official1000": False,
            "ood": False,
        },
    }
    dump(OUT / "protocol.json", protocol)

    seed_all()
    probes = make_probes(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history_path, last_path, best_path = OUT / "training_curve.json", OUT / "last_state.pt", OUT / "best_states.pt"
    history = json.load(open(history_path)) if history_path.exists() else []
    start_epoch = 1
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(last["probes"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
    best = {}
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)["best"]

    for epoch in range(start_epoch, EPOCHS + 1):
        began = time.time()
        probes.train()
        order = torch.where(train_cache["valid"].bool())[0].tolist()
        random.Random(SEED + epoch).shuffle(order)
        total_loss, count = 0.0, 0
        for begin in range(0, len(order), BATCH):
            ix = torch.tensor(order[begin:begin + BATCH])
            ids = [train_ids[i] for i in ix.tolist()]
            s64, sc, cc, a0_evidence, a1_evidence = evidence_batch(runtime, p4_train, fs_train, ids, device)
            valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
            taps0, _, _ = recorders["A0"].run(s64, a0_evidence, sc, cc, valid)
            taps1, _, _ = recorders["A1"].run(s64, a1_evidence, sc, cc, valid)
            target = torch.stack([train_targets[s] for s in ids]).to(device)[:, None].float()
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for stage in STAGES:
                total = total + probe_loss(probes[f"A0__{stage}"](taps0[stage]), target)["total"]
                total = total + probe_loss(probes[f"A1__{stage}"](taps1[stage]), target)["total"]
            total.backward()
            optimizer.step()
            total_loss += float(total.detach()) * len(ids)
            count += len(ids)
        records = validate(probes, runtime, recorders, p4_val, fs_val, dev, val_cache, val_geometry, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "train_samples": count,
               "train_total": total_loss / max(1, count), "metrics": {}}
        for key in PROBE_KEYS:
            metric = summarize(records[key])
            row["metrics"][key] = metric
            score = (metric["mean_foreground_iou"], metric["mean_foreground_f1"])
            if key not in best or score > tuple(best[key]["score"]):
                best[key] = {"score": list(score), "epoch": epoch,
                             "state_dict": {k: v.detach().cpu() for k, v in probes[key].state_dict().items()}}
        history.append(row)
        dump(history_path, history)
        torch.save({"schema": "phase6g9_last_v1", "epoch": epoch,
                    "probes": probes.state_dict(), "optimizer": optimizer.state_dict()}, last_path)
        torch.save({"schema": "phase6g9_best_v1", "best": best}, best_path)
        print(json.dumps({"stage": "EPOCH", "epoch": epoch, "seconds": row["seconds"],
                          "metrics": {k: round(v["mean_foreground_iou"], 6) for k, v in row["metrics"].items()}}), flush=True)

    for key in PROBE_KEYS:
        probes[key].load_state_dict(best[key]["state_dict"])
    records = validate(probes, runtime, recorders, p4_val, fs_val, dev, val_cache, val_geometry, device)
    for key, values in records.items():
        write_rows(OUT / "predictions" / f"{key}.jsonl", values)
    comparisons = stage_comparisons(records)
    decision, first_break = decide(comparisons)
    gains = retention(comparisons)
    diagnostics = diagnostics_pass(runtime, recorders, p4_val, fs_val, dev, val_cache, device)

    frozen_after = {
        "a0_source": tensor_state_sha256(runtime["a0_source"].state_dict()),
        "a1_fusion": tensor_state_sha256(runtime["fusion"].state_dict()),
        "a1_adapter": tensor_state_sha256(runtime["a1_adapter"].state_dict()),
        "a0_rectifier": tensor_state_sha256(runtime["a0_rectifier"].state_dict()),
        "a1_rectifier": tensor_state_sha256(runtime["a1_rectifier"].state_dict()),
    }
    if frozen_before != frozen_after:
        raise RuntimeError("frozen model hash drift")
    result = {
        "schema": "phase6g9_results_v1",
        "status": "COMPLETE_STOP",
        "decision": decision,
        "first_break": first_break,
        "stage_comparisons": comparisons,
        "gain_retention": gains,
        "attention_diagnostics": diagnostics["attention"],
        "transformation_diagnostics": diagnostics,
        "selected_probe_epochs": {key: best[key]["epoch"] for key in PROBE_KEYS},
        "frozen_hashes": frozen_after,
        "firewall": protocol["firewall"],
    }
    dump(OUT / "results.json", result)
    dump(OUT / "attention_diagnostics.json", diagnostics["attention"])
    dump(OUT / "transformation_diagnostics.json", diagnostics)
    dump(OUT / "gain_retention.json", gains)
    dump(OUT / "summary.json", {"status": "COMPLETE_STOP", "decision": decision,
                                "first_break": first_break, "firewall": protocol["firewall"]})
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision, "first_break": first_break}), flush=True)


if __name__ == "__main__":
    main()
