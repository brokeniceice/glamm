#!/usr/bin/env python3
"""Phase 6G.8A matched Adapter input/output spatial probes.

Only matched 1x1 Conv probes are trained.  CLIP/fusion/Adapter/Utility/Rectifier/
SAM/C1 remain frozen, and Utility/Rectifier/SAM/C1 are not instantiated because
they are not needed to score the four evidence representations.
"""
from __future__ import annotations

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

from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, phase4f_spatial_batch
from scripts.phase6g0_multilevel_dense_clip import cache_paths, load_shard
from scripts.phase6g7_staged_r1 import Store, sources
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, probe_loss, summarize
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, load_evidence_source

OUT = ROOT / "outputs/phase6g8a_adapter_input_output_probe"
DOC = ROOT / "docs/phase6g8a_adapter_input_output_probe.md"
A0_CKPT = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
A1_CKPT = ROOT / "outputs/phase6g7_block11_17_staged_r1/joint/selected_checkpoint.pt"
A1_FUSION = ROOT / "outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt"
A1_ADAPTER = ROOT / "outputs/phase6g3_forensic_adapter_audit/a0/selected.pt"
A0_ADAPTER = Path(hd.CFG["evidence"]["forensic_checkpoint"])
SEED, EPOCHS, BATCH = 3407, 20, 8
ARMS = ("A0", "A1")
STAGES = ("source", "adapter_output")
PROBE_CHANNELS = {
    "A0__source": 1024,
    "A0__adapter_output": 256,
    "A1__source": 1024,
    "A1__adapter_output": 256,
}
PROBE_KEYS = tuple(PROBE_CHANNELS)
HISTORICAL_REFERENCE = {
    "historical_block22_source_probe": 0.143842,
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


def load_representers(device):
    a0_adapter = load_evidence_source(hd.CFG, "forensic_rect", device)
    fusion, a1_adapter, _, _ = sources(device)
    for model in (a0_adapter, fusion, a1_adapter):
        model.eval().requires_grad_(False)
    return a0_adapter, fusion, a1_adapter


def representations(a0_adapter, fusion, a1_adapter, spatial_store, fusion_store, ids, device):
    _, raw, _, _, _ = phase4f_spatial_batch(spatial_store, ids, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a0_out = a0_adapter(raw, return_features=True)["F_forensic"]
        fused = fusion(*fusion_store.batch(ids, device))
        a1_out = a1_adapter(fused, return_features=True)["F_forensic"]
    return {
        "A0__source": raw.detach().float(),
        "A0__adapter_output": a0_out.detach().float(),
        "A1__source": fused.detach().float(),
        "A1__adapter_output": a1_out.detach().float(),
    }


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


def probe_set(device):
    probes = nn.ModuleDict(
        {key: nn.Conv2d(channels, 1, 1) for key, channels in PROBE_CHANNELS.items()}
    ).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=1e-3, weight_decay=0.0)
    return probes, optimizer


def validate(probes, representers, spatial_store, fusion_store, dev, cache, clip_lookup, device):
    a0_adapter, fusion, a1_adapter = representers
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
            rep = representations(a0_adapter, fusion, a1_adapter, spatial_store, fusion_store, ids, device)
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
                print(json.dumps({"stage": "VAL", "done": begin + len(raw_ix), "total": n_total}), flush=True)
    if any(any(value is None for value in values) for values in records.values()):
        raise RuntimeError("validation record coverage drift")
    return records


def stable_positive(paired):
    return (
        paired["mean_difference"] > 0
        and paired["bootstrap_95_ci"][0] > 0
        and paired["wilcoxon_pvalue"] < 0.05
    )


def analyze(records, cache_valid):
    table = {key: summarize(records[key]) for key in PROBE_KEYS}
    arms = {}
    for arm in ARMS:
        src = records[f"{arm}__source"]
        out = records[f"{arm}__adapter_output"]
        src_iou = np.asarray([row["foreground_iou"] for row in src], dtype=np.float64)
        out_iou = np.asarray([row["foreground_iou"] for row in out], dtype=np.float64)
        src_f1 = np.asarray([row["foreground_f1"] for row in src], dtype=np.float64)
        out_f1 = np.asarray([row["foreground_f1"] for row in out], dtype=np.float64)
        delta_iou = out_iou - src_iou
        delta_f1 = out_f1 - src_f1
        arms[arm] = {
            "input": table[f"{arm}__source"],
            "output": table[f"{arm}__adapter_output"],
            "paired_iou": paired_statistics(out_iou, src_iou),
            "paired_f1": paired_statistics(out_f1, src_f1),
            "delta_iou_mean": float(delta_iou.mean()),
            "delta_iou_median": float(np.median(delta_iou)),
            "stable_positive_iou": stable_positive(paired_statistics(out_iou, src_iou)),
        }
    a0 = arms["A0"]
    a1 = arms["A1"]
    a0_iou, a1_iou = (
        np.asarray([row["foreground_iou"] for row in records["A0__source"]]),
        np.asarray([row["foreground_iou"] for row in records["A1__source"]]),
    )
    a0_out_iou = np.asarray([row["foreground_iou"] for row in records["A0__adapter_output"]])
    a1_out_iou = np.asarray([row["foreground_iou"] for row in records["A1__adapter_output"]])
    delta_a0 = a0_out_iou - a0_iou
    delta_a1 = a1_out_iou - a1_iou
    cross_delta = paired_statistics(delta_a1, delta_a0)
    a0_stable = stable_positive(a0["paired_iou"])
    a1_stable = stable_positive(a1["paired_iou"])
    if a0_stable and a1_stable:
        a1_weaker = (
            cross_delta["mean_difference"] < 0
            and cross_delta["bootstrap_95_ci"][1] < 0
            and cross_delta["wilcoxon_pvalue"] < 0.05
        )
        decision = (
            "ADAPTER_HAS_REDUCED_MARGINAL_VALUE_FOR_MULTILEVEL_FUSION"
            if a1_weaker
            else "ADAPTER_REMAINS_USEFUL_FOR_BOTH_SOURCES"
        )
    elif a0_stable and not a1_stable:
        decision = "ADAPTER_SPECIALIZATION_REDUNDANT_FOR_MULTILEVEL_FUSION"
    elif not a0_stable and not a1_stable:
        decision = "FULL_ADAPTER_SPECIALIZATION_NOT_NEEDED"
    else:
        decision = "ADAPTER_INPUT_OUTPUT_OUTCOME_OUTSIDE_SPECIFIED_BRANCHES"

    valid_np = np.asarray(cache_valid, dtype=bool)
    valid_arms = {}
    for arm in ARMS:
        src = [row for row, keep in zip(records[f"{arm}__source"], valid_np) if keep]
        out = [row for row, keep in zip(records[f"{arm}__adapter_output"], valid_np) if keep]
        src_iou = np.asarray([row["foreground_iou"] for row in src], dtype=np.float64)
        out_iou = np.asarray([row["foreground_iou"] for row in out], dtype=np.float64)
        src_f1 = np.asarray([row["foreground_f1"] for row in src], dtype=np.float64)
        out_f1 = np.asarray([row["foreground_f1"] for row in out], dtype=np.float64)
        valid_arms[arm] = {
            "input": summarize(src),
            "output": summarize(out),
            "paired_iou": paired_statistics(out_iou, src_iou),
            "paired_f1": paired_statistics(out_f1, src_f1),
        }

    return {
        "schema": "phase6g8a_results_v1",
        "status": "COMPLETE_STOP",
        "decision": decision,
        "core_table_full_1106": arms,
        "probe_metrics_full_1106": table,
        "probe_metrics_valid_seg": {key: summarize([r for r, k in zip(records[key], valid_np) if k]) for key in PROBE_KEYS},
        "adapter_input_output_valid_seg": valid_arms,
        "paired_delta_a1_minus_a0": cross_delta,
        "stable_positive_definition": "paired mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05",
        "decision_rule": (
            "A0 stable and A1 not stable => REDUNDANT; both stable and A1 delta significantly weaker "
            "than A0 => REDUCED_MARGINAL; both stable and not clearly weaker => REMAINS_USEFUL; "
            "neither stable => FULL_NOT_NEEDED"
        ),
        "historical_reference": HISTORICAL_REFERENCE,
        "firewall": {
            "training": "four fresh matched 1x1 Conv probes only",
            "clip_fusion_adapter_utility_rectifier_sam_c1_frozen": True,
            "internal_test": False,
            "official1000": False,
            "ood": False,
        },
    }


def render(result):
    lines = [
        "# Phase 6G.8A — Matched Adapter Input/Output Probe",
        "",
        "Status: **COMPLETE STOP**. Only four fresh matched 1x1 Conv probes were trained. CLIP, block11+17 fusion, Adapter, Rectifier, Utility, SAM, and C1 were frozen; Utility/Rectifier/SAM/C1 were not instantiated because they are not required to score these evidence representations.",
        "",
        "## Core table (full 1106-sample Phase6G.8 population)",
        "",
        "| Arm | Adapter input | Adapter output | Delta | IoU 95% CI | W/T/L | Wilcoxon p |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for arm in ARMS:
        x = result["core_table_full_1106"][arm]
        p = x["paired_iou"]
        lines.append(
            f"| {arm} | {x['input']['mean_foreground_iou']:.6f} | "
            f"{x['output']['mean_foreground_iou']:.6f} | {p['mean_difference']:+.6f} | "
            f"{p['bootstrap_95_ci']} | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} |"
        )
    lines += [
        "",
        "## Paired per-sample statistics",
        "",
    ]
    for arm in ARMS:
        x = result["core_table_full_1106"][arm]
        lines.append(f"- `{arm}` IoU: `{x['paired_iou']}`")
        lines.append(f"- `{arm}` F1: `{x['paired_f1']}`")
    lines += [
        "",
        f"- A1 adapter gain minus A0 adapter gain: `{result['paired_delta_a1_minus_a0']}`",
        f"- stable-positive rule: `{result['stable_positive_definition']}`",
        f"- historical references (not used in statistics): `{result['historical_reference']}`",
        "",
        f"```text\n{result['decision']}\n```",
        "",
        "No test, Official1000, or OOD data were accessed.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


def main():
    if (OUT / "summary.json").exists():
        print(json.dumps({"status": "ALREADY_COMPLETE"}))
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    a0_adapter, fusion, a1_adapter = load_representers(device)
    representers = (a0_adapter, fusion, a1_adapter)
    train_store, val_store = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    train_ids, dev = train_store.sample_ids, load_dev("g0")
    tc, vc = load_c1_cache("train", train_ids), load_c1_cache("val", dev["sample_ids"])
    fs_train, fs_val = Store("train"), Store("val")
    if fs_train.ids != train_ids or fs_val.ids != dev["sample_ids"]:
        raise RuntimeError("source pairing drift")

    frozen_before = {
        "a0_adapter": tensor_state_sha256(a0_adapter.state_dict()),
        "a1_fusion": tensor_state_sha256(fusion.state_dict()),
        "a1_adapter": tensor_state_sha256(a1_adapter.state_dict()),
    }
    train_clip_targets = build_train_clip_targets(train_ids)
    val_clip_lookup = build_val_clip_lookup(dev["sample_ids"])

    protocol = {
        "schema": "phase6g8a_protocol_v1",
        "status": "FROZEN_BEFORE_FIRST_STEP",
        "population": {
            "train_manifest": len(train_ids),
            "train_eligible_seg": int(tc["valid"].sum()),
            "validation": len(dev["sample_ids"]),
            "validation_valid_seg": int(vc["valid"].sum()),
            "train_ids_sha256": sha_ids(train_ids),
            "validation_ids_sha256": sha_ids(dev["sample_ids"]),
        },
        "A0": {
            "definition": "current block22 selected C1 new R1; input raw block22, output Phase4C-A forensic adapter",
            "R1_checkpoint": str(A0_CKPT.resolve()),
            "R1_checkpoint_sha256": file_sha256(A0_CKPT),
            "adapter_checkpoint": str(A0_ADAPTER.resolve()),
            "adapter_checkpoint_sha256": file_sha256(A0_ADAPTER),
        },
        "A1": {
            "definition": "block11+17 staged selected C1 new R1; input selected fusion, output Phase6G.3 A0 adapter",
            "R1_checkpoint": str(A1_CKPT.resolve()),
            "R1_checkpoint_sha256": file_sha256(A1_CKPT),
            "fusion_checkpoint": str(A1_FUSION.resolve()),
            "fusion_checkpoint_sha256": file_sha256(A1_FUSION),
            "adapter_checkpoint": str(A1_ADAPTER.resolve()),
            "adapter_checkpoint_sha256": file_sha256(A1_ADAPTER),
        },
        "representations": {
            "A0__source": "raw block22 dense feature (1024,24,24)",
            "A0__adapter_output": "Phase4C-A forensic adapter F_forensic (256,24,24)",
            "A1__source": "block11+17 Cross-Layer Attention fusion output (1024,24,24)",
            "A1__adapter_output": "Phase6G.3 A0 adapter F_forensic (256,24,24)",
        },
        "probe": {
            "architecture": "fresh Conv2d(input_channels,1,1) for each representation; no historical probe head reuse",
            "epochs": EPOCHS,
            "batch": BATCH,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss": "2*BCE+0.5*Dice",
            "selector": "validation mean FG IoU; tie mean FG F1",
            "threshold": 0.0,
            "seed": SEED,
        },
        "geometry": "legacy CLIP crop geometry for all four representations",
        "frozen_before": frozen_before,
        "firewall": {
            "internal_test": False,
            "official1000": False,
            "ood": False,
        },
    }
    dump(OUT / "protocol.json", protocol)

    probes, opt = probe_set(device)
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

    resume_epoch = None
    inflight = None
    if inflight_path.exists():
        inflight = torch.load(inflight_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(inflight["probes"])
        opt.load_state_dict(inflight["optimizer"])
        resume_epoch = int(inflight["epoch"])
    elif last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(last["probes"])
        opt.load_state_dict(last["optimizer"])

    first_epoch = resume_epoch if resume_epoch is not None else len(history) + 1
    for epoch in range(first_epoch, EPOCHS + 1):
        began = time.time()
        probes.train()
        if resume_epoch == epoch:
            sums, count = inflight["sums"], int(inflight["count"])
            print(json.dumps({"stage": "RESUME_POST_TRAIN_VALIDATION", "epoch": epoch}), flush=True)
        else:
            order = torch.where(tc["valid"].bool())[0].tolist()
            random.Random(SEED + epoch).shuffle(order)
            sums = {key: 0.0 for key in PROBE_KEYS}
            count = 0
            for begin in range(0, len(order), BATCH):
                ix = torch.tensor(order[begin : begin + BATCH])
                ids = [train_ids[i] for i in ix.tolist()]
                rep = representations(a0_adapter, fusion, a1_adapter, train_store, fs_train, ids, device)
                target = torch.stack([train_clip_targets[s] for s in ids]).to(device)
                opt.zero_grad(set_to_none=True)
                total = torch.zeros((), device=device)
                for key in PROBE_KEYS:
                    loss = probe_loss(probes[key](rep[key]), target[:, None].float())
                    total = total + loss["total"]
                    sums[key] += float(loss["total"].detach()) * len(ids)
                total.backward()
                opt.step()
                count += len(ids)
            torch.save(
                {
                    "schema": "phase6g8a_epoch_inflight_v1",
                    "epoch": epoch,
                    "probes": probes.state_dict(),
                    "optimizer": opt.state_dict(),
                    "sums": sums,
                    "count": count,
                },
                inflight_path,
            )
        rec = validate(probes, representers, val_store, fs_val, dev, vc, val_clip_lookup, device)
        row = {"epoch": epoch, "seconds": time.time() - began, "train_samples": count, "metrics": {}}
        for key in PROBE_KEYS:
            metric = summarize(rec[key])
            row["metrics"][key] = metric
            score = (metric["mean_foreground_iou"], metric["mean_foreground_f1"])
            if best[key] is None or score > best[key][0]:
                best[key] = (score, epoch)
                (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema": "phase6g8a_probe_v1",
                        "key": key,
                        "epoch": epoch,
                        "state_dict": probes[key].state_dict(),
                        "score": score,
                    },
                    OUT / "checkpoints" / f"{key}.pt",
                )
        history.append(row)
        dump(OUT / "training_curve.json", history)
        torch.save(
            {
                "schema": "phase6g8a_last_state_v1",
                "epoch": epoch,
                "probes": probes.state_dict(),
                "optimizer": opt.state_dict(),
            },
            last_path,
        )
        inflight_path.unlink(missing_ok=True)
        resume_epoch = None
        inflight = None
        print(
            json.dumps(
                {
                    "stage": "PROBE_EPOCH",
                    "epoch": epoch,
                    "seconds": row["seconds"],
                    "scores": {key: value["mean_foreground_iou"] for key, value in row["metrics"].items()},
                },
            ),
            flush=True,
        )

    for key in PROBE_KEYS:
        probes[key].load_state_dict(
            torch.load(OUT / "checkpoints" / f"{key}.pt", map_location="cpu", weights_only=False)["state_dict"]
        )
    records = validate(probes, representers, val_store, fs_val, dev, vc, val_clip_lookup, device)
    for key, value in records.items():
        write_rows(OUT / "predictions" / f"{key}.jsonl", value)

    frozen_after = {
        "a0_adapter": tensor_state_sha256(a0_adapter.state_dict()),
        "a1_fusion": tensor_state_sha256(fusion.state_dict()),
        "a1_adapter": tensor_state_sha256(a1_adapter.state_dict()),
    }
    if frozen_before != frozen_after:
        raise RuntimeError("frozen model hash drift")

    result = analyze(records, vc["valid"].tolist())
    result["selected_probe_epochs"] = {key: value[1] for key, value in best.items()}
    result["frozen_hashes"] = frozen_after
    dump(OUT / "results.json", result)
    dump(
        OUT / "summary.json",
        {
            "status": "COMPLETE_STOP",
            "decision": result["decision"],
            "selected_probe_epochs": result["selected_probe_epochs"],
            "firewall": result["firewall"],
        },
    )
    render(result)
    print(json.dumps({"status": "COMPLETE_STOP", "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
