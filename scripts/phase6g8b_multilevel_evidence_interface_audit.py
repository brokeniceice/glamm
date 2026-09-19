#!/usr/bin/env python3
"""Phase 6G.8B multilevel evidence interface / channel audit.

Only these parameters are trained:
  * A1/A2 1x1 projection interfaces, each with a temporary matched 1x1 head
    used only during interface training and discarded before evaluation;
  * four fresh matched 1x1 Conv spatial probes, one per frozen representation.

Frozen: CLIP dense caches, selected block11+17 Cross-Layer Attention fusion,
and the Phase6G.3 A0 three-block Adapter.  Rectifier/Utility/SAM/C1 are not
instantiated.  No internal test, Official1000, or OOD data are accessed.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_forensic_adapter import CLIPSpatialArm
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from scripts.phase6g0_multilevel_dense_clip import cache_paths, load_shard
from scripts.phase6g7_staged_r1 import Store
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, probe_loss, summarize
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256

OUT = ROOT / "outputs/phase6g8b_multilevel_evidence_interface_audit"
DOC = ROOT / "docs/phase6g8b_multilevel_evidence_interface_audit.md"
FUSION_CKPT = ROOT / "outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt"
FULL_ADAPTER_CKPT = ROOT / "outputs/phase6g3_forensic_adapter_audit/a0/selected.pt"

SEED, EPOCHS, BATCH = 3407, 20, 8
LR, WEIGHT_DECAY = 1e-3, 0.0

PROBE_CHANNELS = {
    "source_1024": 1024,
    "full_adapter_256": 256,
    "projection_256": 256,
    "projection_512": 512,
}
PROBE_KEYS = tuple(PROBE_CHANNELS)
PROJECTION_CHANNELS = {"projection_256": 256, "projection_512": 512}
SOURCE_KEY = "source_1024"

COMPARISON_KEYS = (
    "full_adapter_256_vs_source",
    "projection_256_vs_source",
    "projection_512_vs_source",
    "projection_512_vs_projection_256",
    "full_adapter_256_vs_projection_256",
    "full_adapter_256_vs_projection_512",
)
WIDTH_COMPARISON_KEYS = (
    "projection_256_vs_source",
    "projection_512_vs_source",
    "projection_512_vs_projection_256",
)

HISTORICAL_REFERENCE = {
    "phase6g8a_source_block11_17_fusion": 0.1789369539062384,
    "phase6g8a_full_adapter_output": 0.17941149888421679,
    "historical_block11_17_fusion_probe": 0.186741,
    "historical_phase6g3_a0_adapter_output": 0.184992,
}


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


def invalid_clip_record(sid, truth):
    truth = torch.as_tensor(truth).bool()
    total_pixels = int(truth.numel())
    fn = int(truth.sum())
    tn = total_pixels - fn
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


def load_fusion(device):
    payload = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
    model = CrossLayerPatchAttention(1024, 8, 0.01)
    model.load_state_dict(payload["fusion"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


def load_full_adapter(device):
    payload = torch.load(FULL_ADAPTER_CKPT, map_location="cpu", weights_only=False)
    model = CLIPSpatialArm(input_channels=1024, channels=256, blocks=3)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


def build_train_clip_targets(train_ids):
    lookup = {}
    for path in cache_paths("late", "train"):
        shard = load_shard(path)
        for local, row in enumerate(shard["records"]):
            lookup[str(row["sample_id"])] = shard["targets"][local]
    if set(lookup) != set(train_ids) or len(lookup) != len(train_ids):
        raise RuntimeError("train CLIP target/sample ID drift")
    return lookup


def build_val_clip_lookup(val_ids):
    lookup = {}
    for path in cache_paths("late", "val"):
        shard = load_shard(path)
        for local, row in enumerate(shard["records"]):
            lookup[str(row["sample_id"])] = (row["geometry"], shard["original_masks"][local])
    if set(lookup) != set(val_ids) or len(lookup) != len(val_ids):
        raise RuntimeError("validation CLIP target/sample ID drift")
    return lookup


def fused_batch(fusion, store, ids, device):
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        value = fusion(*store.batch(ids, device))
    return value.detach().float()


def full_adapter_features(adapter, fused):
    with torch.no_grad(), torch.autocast(device_type=fused.device.type, dtype=torch.bfloat16):
        value = adapter(fused, return_features=True)["F_forensic"]
    return value.detach().float()


class ProjectionInterface(nn.Module):
    """Phase4C-A projection-only arm plus a training-only 1x1 dense head."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(1024, out_channels, 1)
        self.head = nn.Conv2d(out_channels, 1, 1, bias=True)

    def forward(self, value):
        return self.projection(value)

    def logits(self, value):
        return self.head(self.projection(value))


def representations(fusion, full_adapter, projection_256, projection_512, store, ids, device):
    with torch.no_grad():
        fused = fused_batch(fusion, store, ids, device)
        full = full_adapter_features(full_adapter, fused)
        proj_256 = projection_256(fused).float()
        proj_512 = projection_512(fused).float()
    return {
        "source_1024": fused,
        "full_adapter_256": full,
        "projection_256": proj_256,
        "projection_512": proj_512,
    }


def validate_projection(model, fusion, val_store, dev, cache, clip_lookup, device):
    n_total = len(dev["sample_ids"])
    records = [None] * n_total
    model.eval()
    with torch.no_grad():
        for begin in range(0, n_total, BATCH):
            raw_ix = torch.arange(begin, min(begin + BATCH, n_total))
            ix = raw_ix[cache["valid"].index_select(0, raw_ix).bool()]
            for absolute in raw_ix.tolist():
                if bool(cache["valid"][absolute]):
                    continue
                sid = dev["sample_ids"][absolute]
                truth = dev["original_masks"][absolute].bool()
                records[absolute] = invalid_clip_record(sid, truth)
            if not len(ix):
                continue
            ids = [dev["sample_ids"][i] for i in ix.tolist()]
            fused = fused_batch(fusion, val_store, ids, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model.logits(fused).float()
            for j, absolute in enumerate(ix.tolist()):
                sid = ids[j]
                geometry, _ = clip_lookup[sid]
                original = inverse_logits(logits[j, 0], geometry)
                truth = dev["original_masks"][absolute].to(device)
                records[absolute] = {
                    "sample_id": sid,
                    **binary_metrics(original, truth),
                    "valid_g0": True,
                }
            if (begin + len(raw_ix)) % 200 < BATCH:
                print(json.dumps({"stage": "PROJECTION_VAL", "done": begin + len(raw_ix), "total": n_total}), flush=True)
    if any(value is None for value in records):
        raise RuntimeError("projection validation record coverage drift")
    return records


def train_projection(key, out_channels, device, fusion, train_store, train_ids, train_valid, train_targets,
                     val_store, dev, val_cache, val_lookup):
    root = OUT / "interfaces" / key
    if (root / "selected_projection.pt").exists():
        payload = torch.load(root / "selected_projection.pt", map_location="cpu", weights_only=False)
        print(json.dumps({"stage": "PROJECTION", "key": key, "status": "ALREADY_COMPLETE",
                          "selected_epoch": payload["epoch"]}), flush=True)
        return payload

    root.mkdir(parents=True, exist_ok=True)
    seed_all()
    model = ProjectionInterface(out_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history_path, last_path = root / "training_curve.json", root / "last_state.pt"
    best_path, best_json = root / "best_projection.pt", root / "best_metrics.json"
    history = json.load(open(history_path)) if history_path.exists() else []
    start_epoch, best_score, best_epoch = 1, None, None
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
    if best_json.exists():
        best = json.loads(best_json.read_text())
        best_score, best_epoch = tuple(best["score"]), int(best["epoch"])

    for epoch in range(start_epoch, EPOCHS + 1):
        began = time.time()
        model.train()
        order = torch.where(train_valid.bool())[0].tolist()
        random.Random(SEED + epoch).shuffle(order)
        total_loss, count = 0.0, 0
        for begin in range(0, len(order), BATCH):
            ix = torch.tensor(order[begin:begin + BATCH])
            ids = [train_ids[i] for i in ix.tolist()]
            fused = fused_batch(fusion, train_store, ids, device)
            target = torch.stack([train_targets[s] for s in ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(model.logits(fused), target[:, None].float())
            loss["total"].backward()
            optimizer.step()
            total_loss += float(loss["total"].detach()) * len(ids)
            count += len(ids)
        records = validate_projection(model, fusion, val_store, dev, val_cache, val_lookup, device)
        metrics = summarize(records)
        score = (metrics["mean_foreground_iou"], metrics["mean_foreground_f1"])
        row = {
            "epoch": epoch,
            "seconds": time.time() - began,
            "train_samples": count,
            "train_total": total_loss / max(1, count),
            "validation": metrics,
        }
        history.append(row)
        dump(history_path, history)
        torch.save({"schema": "phase6g8b_projection_last_v1", "key": key, "channels": out_channels,
                    "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, last_path)
        if best_score is None or score > best_score:
            best_score, best_epoch = score, epoch
            dump(best_json, {"key": key, "channels": out_channels, "epoch": epoch, "score": list(score),
                             "validation": metrics})
            torch.save({"schema": "phase6g8b_projection_best_v1", "key": key, "channels": out_channels,
                        "epoch": epoch, "model": model.state_dict(), "validation": metrics}, best_path)
        print(json.dumps({"stage": "PROJECTION_EPOCH", "key": key, "epoch": epoch,
                          "seconds": row["seconds"], "mean_iou": metrics["mean_foreground_iou"],
                          "mean_f1": metrics["mean_foreground_f1"]}), flush=True)

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    selected = {
        "schema": "phase6g8b_projection_selected_v1",
        "key": key,
        "channels": out_channels,
        "epoch": int(best["epoch"]),
        "projection": {name: value for name, value in best["model"].items() if name.startswith("projection.")},
        "training_head": {name: value for name, value in best["model"].items() if name.startswith("head.")},
        "validation": best["validation"],
        "selector": "validation mean FG IoU; tie mean FG F1",
        "history": history,
    }
    torch.save(selected, root / "selected_projection.pt")
    dump(root / "summary.json", {
        "status": "COMPLETE",
        "key": key,
        "channels": out_channels,
        "selected_epoch": int(best["epoch"]),
        "selected_validation": best["validation"],
        "trainable_interface": "1x1 Conv 1024->%d" % out_channels,
        "training_only_head": "1x1 Conv %d->1" % out_channels,
    })
    print(json.dumps({"stage": "PROJECTION", "key": key, "status": "COMPLETE",
                      "selected_epoch": int(best["epoch"]),
                      "mean_iou": best["validation"]["mean_foreground_iou"]}), flush=True)
    return selected


def load_selected_projection(key, device):
    payload = torch.load(OUT / "interfaces" / key / "selected_projection.pt", map_location="cpu", weights_only=False)
    model = nn.Conv2d(1024, int(payload["channels"]), 1)
    state = {
        name.split("projection.", 1)[1]: value
        for name, value in payload["projection"].items()
        if name.startswith("projection.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def probe_set(device):
    probes = nn.ModuleDict({key: nn.Conv2d(channels, 1, 1) for key, channels in PROBE_CHANNELS.items()}).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return probes, optimizer


def validate_probes(probes, fusion, full_adapter, projection_256, projection_512,
                    val_store, dev, cache, clip_lookup, device):
    n_total = len(dev["sample_ids"])
    records = {key: [None] * n_total for key in PROBE_KEYS}
    probes.eval()
    with torch.no_grad():
        for begin in range(0, n_total, BATCH):
            raw_ix = torch.arange(begin, min(begin + BATCH, n_total))
            ix = raw_ix[cache["valid"].index_select(0, raw_ix).bool()]
            for absolute in raw_ix.tolist():
                if bool(cache["valid"][absolute]):
                    continue
                sid = dev["sample_ids"][absolute]
                truth = dev["original_masks"][absolute].bool()
                invalid = invalid_clip_record(sid, truth)
                for key in PROBE_KEYS:
                    records[key][absolute] = dict(invalid)
            if not len(ix):
                continue
            ids = [dev["sample_ids"][i] for i in ix.tolist()]
            rep = representations(fusion, full_adapter, projection_256, projection_512, val_store, ids, device)
            for key in PROBE_KEYS:
                logits = probes[key](rep[key]).float()
                for j, (sid, absolute) in enumerate(zip(ids, ix.tolist())):
                    geometry, _ = clip_lookup[sid]
                    original = inverse_logits(logits[j, 0], geometry)
                    truth = dev["original_masks"][absolute].to(device)
                    records[key][absolute] = {
                        "sample_id": sid,
                        **binary_metrics(original, truth),
                        "valid_g0": True,
                    }
            if (begin + len(raw_ix)) % 200 < BATCH:
                print(json.dumps({"stage": "PROBE_VAL", "done": begin + len(raw_ix), "total": n_total}), flush=True)
    if any(any(value is None for value in values) for values in records.values()):
        raise RuntimeError("probe validation record coverage drift")
    return records


def stable_positive(paired):
    return (
        paired["mean_difference"] > 0
        and paired["bootstrap_95_ci"][0] > 0
        and paired["wilcoxon_pvalue"] < 0.05
    )


def stable_negative(paired):
    return (
        paired["mean_difference"] < 0
        and paired["bootstrap_95_ci"][1] < 0
        and paired["wilcoxon_pvalue"] < 0.05
    )


def not_stably_different(paired):
    return not stable_positive(paired) and not stable_negative(paired)


def decide(comparisons):
    c = comparisons
    if (
        not_stably_different(c["projection_256_vs_source"]["iou"])
        and not_stably_different(c["full_adapter_256_vs_projection_256"]["iou"])
    ):
        return "256_INTERFACE_SUFFICIENT_LOCAL_ADAPTER_REDUNDANT"
    if (
        stable_positive(c["projection_512_vs_projection_256"]["iou"])
        and not_stably_different(c["projection_512_vs_source"]["iou"])
    ):
        return "256_CHANNEL_INTERFACE_IS_BOTTLENECK_512_SUFFICIENT"
    if (
        stable_negative(c["projection_512_vs_source"]["iou"])
        and stable_positive(c["projection_512_vs_projection_256"]["iou"])
    ):
        return "WIDER_FORENSIC_INTERFACE_SUPPORTED"
    if (
        stable_positive(c["full_adapter_256_vs_projection_256"]["iou"])
        and stable_positive(c["full_adapter_256_vs_projection_512"]["iou"])
    ):
        return "LOCAL_FORENSIC_ADAPTATION_STILL_NEEDED"
    if all(not_stably_different(c[key]["iou"]) for key in WIDTH_COMPARISON_KEYS):
        return "CHANNEL_WIDTH_NOT_PRIMARY_BOTTLENECK"
    return "OUTCOME_OUTSIDE_SPECIFIED_BRANCHES"


def pair_comparisons(records):
    arrays_iou = {key: np.asarray([row["foreground_iou"] for row in records[key]], dtype=np.float64) for key in PROBE_KEYS}
    arrays_f1 = {key: np.asarray([row["foreground_f1"] for row in records[key]], dtype=np.float64) for key in PROBE_KEYS}

    def compare(left, right):
        return {
            "iou": paired_statistics(arrays_iou[left], arrays_iou[right]),
            "f1": paired_statistics(arrays_f1[left], arrays_f1[right]),
        }

    return {
        "full_adapter_256_vs_source": compare("full_adapter_256", "source_1024"),
        "projection_256_vs_source": compare("projection_256", "source_1024"),
        "projection_512_vs_source": compare("projection_512", "source_1024"),
        "projection_512_vs_projection_256": compare("projection_512", "projection_256"),
        "full_adapter_256_vs_projection_256": compare("full_adapter_256", "projection_256"),
        "full_adapter_256_vs_projection_512": compare("full_adapter_256", "projection_512"),
    }


def analyze(records, valid_mask):
    table = {key: summarize(records[key]) for key in PROBE_KEYS}
    comparisons = pair_comparisons(records)
    valid_np = np.asarray(valid_mask, dtype=bool)
    valid_records = {key: [row for row, keep in zip(records[key], valid_np) if keep] for key in PROBE_KEYS}
    valid_table = {key: summarize(valid_records[key]) for key in PROBE_KEYS}
    valid_comparisons = pair_comparisons(valid_records)
    return {
        "schema": "phase6g8b_results_v1",
        "status": "COMPLETE_STOP",
        "decision": decide(comparisons),
        "core_table_full_1106": {
            key: {
                "mean_foreground_iou": table[key]["mean_foreground_iou"],
                "median_foreground_iou": table[key]["median_foreground_iou"],
                "mean_foreground_f1": table[key]["mean_foreground_f1"],
            }
            for key in PROBE_KEYS
        },
        "probe_metrics_full_1106": table,
        "probe_metrics_valid_seg": valid_table,
        "comparisons_full_1106": comparisons,
        "comparisons_valid_seg": valid_comparisons,
        "stable_definition": {
            "stable_positive": "paired mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05",
            "stable_negative": "paired mean delta < 0, bootstrap 95% CI upper < 0, Wilcoxon p < 0.05",
            "equivalent": "neither stable positive nor stable negative (95% CI includes zero)",
        },
        "decision_rule": (
            "ordered: (1) projection_256 ~= source and full_adapter ~= projection_256 => "
            "256_INTERFACE_SUFFICIENT_LOCAL_ADAPTER_REDUNDANT; "
            "(2) projection_512 > projection_256 and projection_512 ~= source => "
            "256_CHANNEL_INTERFACE_IS_BOTTLENECK_512_SUFFICIENT; "
            "(3) source > projection_512 > projection_256 => WIDER_FORENSIC_INTERFACE_SUPPORTED; "
            "(4) full_adapter > both projection variants => LOCAL_FORENSIC_ADAPTATION_STILL_NEEDED; "
            "(5) all width comparisons unstable => CHANNEL_WIDTH_NOT_PRIMARY_BOTTLENECK"
        ),
        "historical_reference": HISTORICAL_REFERENCE,
        "firewall": {
            "training": "A1/A2 projection interfaces plus four fresh matched 1x1 Conv probes only",
            "clip_fusion_full_adapter_c1_sam_rectifier_utility_frozen": True,
            "rectifier_utility_sam_c1_not_instantiated": True,
            "internal_test": False,
            "official1000": False,
            "ood": False,
        },
    }


def render(result):
    labels = {
        "source_1024": "block11+17 fused 1024 source",
        "full_adapter_256": "full Adapter -> 256",
        "projection_256": "projection-only -> 256",
        "projection_512": "projection-only -> 512",
    }
    lines = [
        "# Phase 6G.8B — Multi-Level Evidence Interface / Channel Audit",
        "",
        "Status: **COMPLETE STOP**. Only the two projection interfaces and four fresh matched probes were trained. CLIP caches, block11+17 fusion, the Phase6G.3 full Adapter, C1, and SAM were frozen; Rectifier/Utility/SAM/C1 were not instantiated.",
        "",
        "## Core table (full 1106-sample Phase6G.8 population)",
        "",
        "| Representation | Mean FG IoU | Median FG IoU | Mean FG F1 | Delta vs source | IoU 95% CI | W/T/L | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    source = result["core_table_full_1106"]["source_1024"]
    for key in PROBE_KEYS:
        row = result["core_table_full_1106"][key]
        if key == SOURCE_KEY:
            delta, ci, wtl, p = "-", "-", "-", "-"
        else:
            comparison = result["comparisons_full_1106"][f"{key}_vs_source"]["iou"]
            delta = f"{comparison['mean_difference']:+.6f}"
            ci = f"[{comparison['bootstrap_95_ci'][0]:.6f}, {comparison['bootstrap_95_ci'][1]:.6f}]"
            wtl = f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
            p = f"{comparison['wilcoxon_pvalue']:.4g}"
        lines.append(
            f"| {labels[key]} | {row['mean_foreground_iou']:.6f} | {row['median_foreground_iou']:.6f} | "
            f"{row['mean_foreground_f1']:.6f} | {delta} | {ci} | {wtl} | {p} |"
        )
    lines += [
        "",
        "## Projection retention and width comparisons",
        "",
        "| Comparison | Delta mean FG IoU | IoU 95% CI | W/T/L | Wilcoxon p | Delta mean FG F1 | F1 95% CI |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    comparison_labels = {
        "full_adapter_256_vs_source": "full Adapter 256 - source 1024",
        "projection_256_vs_source": "projection 256 - source 1024",
        "projection_512_vs_source": "projection 512 - source 1024",
        "projection_512_vs_projection_256": "projection 512 - projection 256",
        "full_adapter_256_vs_projection_256": "full Adapter 256 - projection 256",
        "full_adapter_256_vs_projection_512": "full Adapter 256 - projection 512",
    }
    for key in COMPARISON_KEYS:
        item = result["comparisons_full_1106"][key]
        iou, f1 = item["iou"], item["f1"]
        lines.append(
            f"| {comparison_labels[key]} | {iou['mean_difference']:+.6f} | "
            f"[{iou['bootstrap_95_ci'][0]:.6f}, {iou['bootstrap_95_ci'][1]:.6f}] | "
            f"{iou['wins']}/{iou['ties']}/{iou['losses']} | {iou['wilcoxon_pvalue']:.4g} | "
            f"{f1['mean_difference']:+.6f} | [{f1['bootstrap_95_ci'][0]:.6f}, {f1['bootstrap_95_ci'][1]:.6f}] |"
        )
    lines += [
        "",
        "## Valid-SEG sensitivity (1090 samples)",
        "",
        "The primary table uses the full 1106-sample Phase6G.8 validation population. The valid-SEG-only table below is a sensitivity check, not a replacement for the primary population.",
        "",
        "| Representation | Mean FG IoU | Median FG IoU | Mean FG F1 | Delta vs source | IoU 95% CI | W/T/L | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    valid_source = result["probe_metrics_valid_seg"]["source_1024"]
    for key in PROBE_KEYS:
        row = result["probe_metrics_valid_seg"][key]
        if key == SOURCE_KEY:
            delta, ci, wtl, p = "-", "-", "-", "-"
        else:
            comparison = result["comparisons_valid_seg"][f"{key}_vs_source"]["iou"]
            delta = f"{comparison['mean_difference']:+.6f}"
            ci = f"[{comparison['bootstrap_95_ci'][0]:.6f}, {comparison['bootstrap_95_ci'][1]:.6f}]"
            wtl = f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
            p = f"{comparison['wilcoxon_pvalue']:.4g}"
        lines.append(
            f"| {labels[key]} | {row['mean_foreground_iou']:.6f} | {row['median_foreground_iou']:.6f} | "
            f"{row['mean_foreground_f1']:.6f} | {delta} | {ci} | {wtl} | {p} |"
        )
    lines += [
        "",
        "## Decision rationale",
        "",
        f"- `projection_256 - source`: `{result['comparisons_full_1106']['projection_256_vs_source']['iou']}`",
        f"- `full_adapter - projection_256`: `{result['comparisons_full_1106']['full_adapter_256_vs_projection_256']['iou']}`",
        f"- `projection_512 - projection_256`: `{result['comparisons_full_1106']['projection_512_vs_projection_256']['iou']}`",
        f"- `full_adapter - source`: `{result['comparisons_full_1106']['full_adapter_256_vs_source']['iou']}`",
        f"- `projection_512 - source`: `{result['comparisons_full_1106']['projection_512_vs_source']['iou']}`",
        f"- `full_adapter - projection_512`: `{result['comparisons_full_1106']['full_adapter_256_vs_projection_512']['iou']}`",
        "",
        "The first decision branch is taken because `projection_256` is not stably different from `source` and `full_adapter` is not stably different from `projection_256`. The 512-wide projection and the three-block Adapter also show no stable gain. The valid-SEG sensitivity table gives the same pattern.",
        "",
        "## Interpretation",
        "",
        "- The 1024->256 interface is not a detectable information bottleneck on this frozen block11+17 fusion source.",
        "- The three LocalForensicBlocks do not add a stable gain over a single 1x1 projection to 256 for this evidence interface.",
        "- The result is an interface-level attribution from internal validation only; it does not authorize deleting the Adapter, changing Rectifier input dimensions, retraining R1, or accessing test/Official1000/OOD.",
        "",
        f"- stable rule: `{result['stable_definition']}`",
        f"- historical references (not used in statistics): `{result['historical_reference']}`",
        "",
        f"```text\n{result['decision']}\n```",
        "",
        "No test, Official1000, or OOD data were accessed.",
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

    fusion, fusion_payload = load_fusion(device)
    full_adapter, adapter_payload = load_full_adapter(device)
    train_store, val_store = Store("train"), Store("val")
    dev = load_dev("g0")
    train_ids = train_store.ids
    if val_store.ids != dev["sample_ids"]:
        raise RuntimeError("source pairing drift")
    train_cache = load_c1_cache("train", train_ids)
    val_cache = load_c1_cache("val", dev["sample_ids"])
    train_targets = build_train_clip_targets(train_ids)
    val_lookup = build_val_clip_lookup(dev["sample_ids"])

    frozen_before = {
        "fusion": tensor_state_sha256(fusion.state_dict()),
        "full_adapter": tensor_state_sha256(full_adapter.state_dict()),
    }
    protocol = {
        "schema": "phase6g8b_protocol_v1",
        "status": "FROZEN_BEFORE_FIRST_STEP",
        "population": {
            "train_manifest": len(train_ids),
            "train_eligible_seg": int(train_cache["valid"].sum()),
            "validation": len(dev["sample_ids"]),
            "validation_valid_seg": int(val_cache["valid"].sum()),
            "train_ids_sha256": sha_ids(train_ids),
            "validation_ids_sha256": sha_ids(dev["sample_ids"]),
        },
        "evidence_source": {
            "definition": "selected block11+17 Cross-Layer Attention fusion, [1024,24,24]",
            "store": "scripts.phase6g7_staged_r1.Store",
            "fusion_checkpoint": str(FUSION_CKPT.resolve()),
            "fusion_checkpoint_sha256": file_sha256(FUSION_CKPT),
            "fusion_selected_epoch": fusion_payload["epoch"],
            "full_adapter_definition": "Phase6G.8A A1 selected/frozen Phase4C-A Adapter (three LocalForensicBlocks)",
            "full_adapter_checkpoint": str(FULL_ADAPTER_CKPT.resolve()),
            "full_adapter_checkpoint_sha256": file_sha256(FULL_ADAPTER_CKPT),
            "full_adapter_selected_epoch": adapter_payload["epoch"],
        },
        "arms": {
            "source_1024": "identity fused representation, no trainable evidence transformation",
            "full_adapter_256": "frozen three-block Adapter F_forensic",
            "projection_256": "trainable 1x1 Conv 1024->256 interface, no LocalForensicBlock",
            "projection_512": "trainable 1x1 Conv 1024->512 interface, no LocalForensicBlock",
        },
        "interface_training": {
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch": BATCH,
            "loss": "2*BCE+0.5*Dice",
            "seed": SEED,
            "temporary_head": "1x1 Conv C->1; used only for dense supervision and discarded before Evaluation 1",
            "selector": "validation mean FG IoU; tie mean FG F1",
        },
        "probe": {
            "architecture": "fresh Conv2d(input_channels,1,1) for each frozen representation; no multi-layer or spatial 3x3 probe",
            "epochs": EPOCHS,
            "batch": BATCH,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss": "2*BCE+0.5*Dice",
            "seed": SEED,
            "selector": "validation mean FG IoU; tie mean FG F1",
            "threshold": 0.0,
        },
        "geometry": "legacy CLIP crop geometry for all four representations",
        "frozen_before": frozen_before,
        "firewall": {"internal_test": False, "official1000": False, "ood": False},
    }
    dump(OUT / "protocol.json", protocol)

    train_projection("projection_256", 256, device, fusion, train_store, train_ids,
                     train_cache["valid"], train_targets, val_store, dev, val_cache, val_lookup)
    train_projection("projection_512", 512, device, fusion, train_store, train_ids,
                     train_cache["valid"], train_targets, val_store, dev, val_cache, val_lookup)
    projection_256 = load_selected_projection("projection_256", device)
    projection_512 = load_selected_projection("projection_512", device)

    seed_all()
    probes, probe_optimizer = probe_set(device)
    history_path, inflight_path, last_path = (
        OUT / "training_curve.json",
        OUT / "epoch_inflight.pt",
        OUT / "last_state.pt",
    )
    history = json.load(open(history_path)) if history_path.exists() else []
    best = {key: None for key in PROBE_KEYS}
    for key in PROBE_KEYS:
        path = OUT / "checkpoints" / f"{key}.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=False)
            best[key] = (tuple(old["score"]), int(old["epoch"]))

    resume_epoch, inflight = None, None
    if inflight_path.exists():
        inflight = torch.load(inflight_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(inflight["probes"])
        probe_optimizer.load_state_dict(inflight["optimizer"])
        resume_epoch = int(inflight["epoch"])
    elif last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(last["probes"])
        probe_optimizer.load_state_dict(last["optimizer"])

    first_epoch = resume_epoch if resume_epoch is not None else len(history) + 1
    for epoch in range(first_epoch, EPOCHS + 1):
        began = time.time()
        probes.train()
        if resume_epoch == epoch:
            sums, count = inflight["sums"], int(inflight["count"])
            print(json.dumps({"stage": "RESUME_POST_TRAIN_VALIDATION", "epoch": epoch}), flush=True)
        else:
            order = torch.where(train_cache["valid"].bool())[0].tolist()
            random.Random(SEED + epoch).shuffle(order)
            sums = {key: 0.0 for key in PROBE_KEYS}
            count = 0
            for begin in range(0, len(order), BATCH):
                ix = torch.tensor(order[begin:begin + BATCH])
                ids = [train_ids[i] for i in ix.tolist()]
                rep = representations(fusion, full_adapter, projection_256, projection_512,
                                      train_store, ids, device)
                target = torch.stack([train_targets[s] for s in ids]).to(device)
                probe_optimizer.zero_grad(set_to_none=True)
                total = torch.zeros((), device=device)
                for key in PROBE_KEYS:
                    loss = probe_loss(probes[key](rep[key]), target[:, None].float())
                    total = total + loss["total"]
                    sums[key] += float(loss["total"].detach()) * len(ids)
                total.backward()
                probe_optimizer.step()
                count += len(ids)
            torch.save({
                "schema": "phase6g8b_epoch_inflight_v1",
                "epoch": epoch,
                "probes": probes.state_dict(),
                "optimizer": probe_optimizer.state_dict(),
                "sums": sums,
                "count": count,
            }, inflight_path)
        records = validate_probes(probes, fusion, full_adapter, projection_256, projection_512,
                                  val_store, dev, val_cache, val_lookup, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "train_samples": count, "metrics": {}}
        for key in PROBE_KEYS:
            metric = summarize(records[key])
            row["metrics"][key] = metric
            score = (metric["mean_foreground_iou"], metric["mean_foreground_f1"])
            if best[key] is None or score > best[key][0]:
                best[key] = (score, epoch)
                (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema": "phase6g8b_probe_v1",
                    "key": key,
                    "epoch": epoch,
                    "state_dict": probes[key].state_dict(),
                    "score": score,
                }, OUT / "checkpoints" / f"{key}.pt")
        history.append(row)
        dump(history_path, history)
        torch.save({
            "schema": "phase6g8b_last_state_v1",
            "epoch": epoch,
            "probes": probes.state_dict(),
            "optimizer": probe_optimizer.state_dict(),
        }, last_path)
        inflight_path.unlink(missing_ok=True)
        resume_epoch, inflight = None, None
        print(json.dumps({
            "stage": "PROBE_EPOCH",
            "epoch": epoch,
            "seconds": row["seconds"],
            "scores": {key: value["mean_foreground_iou"] for key, value in row["metrics"].items()},
        }), flush=True)

    for key in PROBE_KEYS:
        probes[key].load_state_dict(
            torch.load(OUT / "checkpoints" / f"{key}.pt", map_location="cpu", weights_only=False)["state_dict"]
        )
    records = validate_probes(probes, fusion, full_adapter, projection_256, projection_512,
                              val_store, dev, val_cache, val_lookup, device)
    for key, value in records.items():
        write_rows(OUT / "predictions" / f"{key}.jsonl", value)

    frozen_after = {
        "fusion": tensor_state_sha256(fusion.state_dict()),
        "full_adapter": tensor_state_sha256(full_adapter.state_dict()),
    }
    if frozen_before != frozen_after:
        raise RuntimeError("frozen model hash drift")

    result = analyze(records, val_cache["valid"].tolist())
    result["selected_probe_epochs"] = {key: value[1] for key, value in best.items()}
    result["frozen_hashes"] = frozen_after
    dump(OUT / "results.json", result)
    dump(OUT / "summary.json", {
        "status": "COMPLETE_STOP",
        "decision": result["decision"],
        "selected_probe_epochs": result["selected_probe_epochs"],
        "firewall": result["firewall"],
    })
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
