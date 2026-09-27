#!/usr/bin/env python3
"""Matched C1-native staged R1 comparison: duplicated F24 vs frozen I-JEPA evidence."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4f_run as f4
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts import phase4g1q_conditional_utility as q
from scripts.phase4gf_formal_localization import full_inputs, load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, evidence_feature, invalid_record, load_evidence_source, load_rectifier, load_sam_runtime, mask_loss
from scripts.phase6i_ijepa_cache import OUT as JEPA_CACHE

ARM = os.environ.get("PHASE6I_ARM", "")
if ARM not in ("A0", "A1"):
    raise RuntimeError("set PHASE6I_ARM to A0 or A1")
OUT = ROOT / "outputs/phase6i_ijepa_r1" / ARM
CKPTS = Path("/data/yz/groundingLMM_official/checkpoints/phase6i_ijepa_r1") / ARM
C1_SHA = e2.C1_SHA
CFG = e2.CFG
SEED = e2.SEED


class JEPAStore:
    def __init__(self, split, expected_ids):
        self.locations = {}
        self.shards = []
        statuses = [json.loads((JEPA_CACHE / f"worker_{worker}_{split}.json").read_text()) for worker in (0, 1)]
        require(all(s["status"] == "COMPLETE" and s["split"] == split for s in statuses), "JEPA cache worker incomplete")
        require(statuses[0]["checkpoint_sha256"] == statuses[1]["checkpoint_sha256"], "JEPA source checkpoint drift")
        paths = sorted((JEPA_CACHE / split).glob("shard_*.pt"))
        sources = sorted((ROOT / "outputs/phase3c1_spatial_probe/cache/clip" / split).glob("shard_*.pt"))
        require([p.name for p in paths] == [p.name for p in sources], "JEPA cache shard set drift")
        for path, source in zip(paths, sources):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["schema"] == "phase6i_ijepa_448_clip_crop_v1", "JEPA cache schema drift")
            require(payload["source_path"] == str(source.resolve()), "JEPA cache source path drift")
            values = payload["features"]
            require(values.shape[1:] == (1280, 24, 24), "JEPA cache shape drift")
            j = len(self.shards)
            self.shards.append(values)
            for i, sid in enumerate(payload["sample_ids"]):
                require(sid not in self.locations, "duplicate JEPA sample ID")
                self.locations[sid] = (j, i)
        require(list(self.locations) == expected_ids, f"JEPA {split} population/order drift")

    def get(self, ids, device):
        return torch.stack([self.shards[j][i] for j, i in (self.locations[sid] for sid in ids)]).to(device, non_blocking=True)


class EvidenceAugmentedRectifier(nn.Module):
    """Same-sized auxiliary branch in both arms; input source is the only arm difference."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.norm = nn.LayerNorm(1280)
        self.projection = nn.Conv2d(1280, 256, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, s64, evidence, sam_coords, clip_coords, valid, auxiliary):
        if ARM == "A0":
            auxiliary = evidence.float().repeat(1, 5, 1, 1)
        elif auxiliary is None or auxiliary.shape != (evidence.shape[0], 1280, 24, 24):
            raise RuntimeError("missing/misaligned I-JEPA evidence")
        x = auxiliary.float().permute(0, 2, 3, 1)
        x = self.norm(x).permute(0, 3, 1, 2)
        supplement = self.projection(x)
        augmented = evidence.float() + self.alpha * supplement
        return self.base(s64, augmented.to(evidence.dtype), sam_coords, clip_coords, valid)


def make_rectifier(device):
    return EvidenceAugmentedRectifier(load_rectifier(CFG, c1_gamma(), device)).to(device)


def jepa_store(split, ids):
    return JEPAStore(split, ids) if ARM == "A1" else None


def auxiliary(jstore, ids, device):
    return jstore.get(ids, device) if jstore is not None else None


def require(test, message):
    if not test:
        raise RuntimeError(message)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def hash_state(module):
    return tensor_state_sha256(module.state_dict())


def c1_gamma():
    audit = json.loads((ROOT / "outputs/phase6e3_c1_native_staged/rectifier/c1_gamma_audit.json").read_text())
    require(audit["status"] == "PASS" and audit["uses_c1_validity"] and not audit["uses_p1_query"], "C1 gamma audit drift")
    return float(audit["selected_gamma"])


def load_inputs():
    status = json.loads((e2.BASE_OUT / "cache/status.json").read_text())
    require(status["status"] == "COMPLETE" and status["c1_sha256"] == C1_SHA, "C1 cache provenance drift")
    ids = train_ids()
    dev = load_dev("g0")
    train_store, val_store = Phase4FStore(CFG, "train"), Phase4FStore(CFG, "val")
    require(ids == train_store.sample_ids and dev["sample_ids"] == val_store.sample_ids, "spatial population/order drift")
    train_cache = e2.load_c1_cache("train", ids)
    val_cache = e2.load_c1_cache("val", dev["sample_ids"])
    require((len(ids), int(train_cache["valid"].sum()), len(dev["sample_ids"]), int(val_cache["valid"].sum())) == (8836, 8741, 1106, 1090), "C1 valid population drift")
    return ids, dev, train_store, val_store, train_cache, val_cache


def rect_metrics(rectifier, store, dev, cache, sam, source, jstore, device):
    records = []
    rectifier.eval()
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                row = invalid_record(sid, dev["original_masks"][i])
                row["valid_g0"] = False
                records.append(row)
                continue
            s64, raw, _, sc, cc = e2.phase4f_spatial_batch(store, [sid], device)
            evidence = evidence_feature(source, raw)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = rectifier(s64, evidence, sc, cc, valid, auxiliary(jstore, [sid], device))
            query = cache["q_seg"][i:i + 1].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(query, output["image_embeddings"].to(torch.bfloat16))
            logits = inverse_sam_logits(low, dev["sam_geometries"][i])
            row = metric_record(sid, logits, dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
    return summarize(records)


def phase4f_batch(store, ids, source, rectifier, jstore, device):
    s64, raw, targets, sc, cc = e2.phase4f_spatial_batch(store, ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
    valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = rectifier(s64, evidence, sc, cc, valid, auxiliary(jstore, ids, device))
    return s64, value["image_embeddings"], value["support"].reshape(len(ids), 1, 64, 64), targets, sc


def evaluate(model, rectifier, sam, source, store, dev, cache, jstore, device):
    records = []
    model.eval(); model.language_source.eval(); model.forensic_source.eval(); rectifier.eval()
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                row = invalid_record(sid, dev["original_masks"][i])
                row["valid_g0"] = False
                records.append(row)
                continue
            idx = torch.tensor([i])
            batch, _ = e2.c1_language_batch(dev, cache, idx, store, sam, device)
            s64, corrected, support, _, sc = phase4f_batch(store, [sid], source, rectifier, jstore, device)
            out = hc.utility_forward(model, batch)
            gate = hc.gate_to_sam_grid(out["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, corrected, gate)
            with torch.autocast("cuda", enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            logits = inverse_sam_logits(low, dev["sam_geometries"][i])
            row = metric_record(sid, logits, dev["original_masks"][i])
            row["valid_g0"] = True
            records.append(row)
    return summarize(records), records


def select(stage, history, key, payload_key):
    best = max(history, key=lambda row: (row[key], -row["epoch"]))
    path = CKPTS / stage / f"epoch_{best['epoch']}.pt"
    selected = OUT / stage / "selected_checkpoint.pt"
    selected.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, selected)
    from tools.phase4c_b import file_sha256
    info = {"status": "COMPLETE", "stage": stage, "primary": "internal DEV canonical G0 mean FG IoU",
            "selected_epoch": best["epoch"], "selected_metric": best[key], "selected_checkpoint": str(selected.resolve()),
            "selected_checkpoint_sha256": file_sha256(selected), "c1_sha256": C1_SHA,
            "candidates": [{"epoch": row["epoch"], "mean_fg_iou": row[key]} for row in history],
            "official1000_used_for_selection": False, "external_ood_used_for_selection": False,
            "checkpoint_payload": payload_key}
    dump(OUT / stage / "selector.json", info)
    return info


def stage1(device):
    stage = "rectifier"
    require(not (OUT / stage / "selector.json").exists(), "rectifier already selected")
    hc.seed_all()
    ids, dev, train_store, val_store, cache, val_cache = load_inputs()
    train_jepa, val_jepa = jepa_store("train", ids), jepa_store("val", dev["sample_ids"])
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    gamma = c1_gamma()
    require(gamma == 0.01, "C1 gamma drift")
    rectifier = make_rectifier(device)
    expected_rectifier = 329985 + 1280 * 256 + 256 + 2 * 1280 + 1
    require(sum(p.numel() for p in rectifier.parameters() if p.requires_grad) == expected_rectifier, "rectifier trainable scope drift")
    frozen = {"sam": hash_state(sam), "source": hash_state(source)}
    init = hash_state(rectifier)
    # Match Phase4F: AdamW, warmup+cosine, ten full traversals, epoch 0 eligible.
    optimizer, scheduler = f4.optimizer_scheduler(rectifier, 11050)
    history, updates = [], 0
    stage_ckpts = CKPTS / stage
    stage_ckpts.mkdir(parents=True, exist_ok=True)
    existing = sorted(stage_ckpts.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if existing:
        history = json.loads((OUT / stage / "history.json").read_text())
        epochs = [int(p.stem.split("_")[1]) for p in existing]
        require(epochs == list(range(epochs[-1] + 1)), "rectifier checkpoint epoch gap")
        require([row["epoch"] for row in history] == epochs, "rectifier history/checkpoint drift")
        zero = torch.load(existing[0], map_location="cpu", weights_only=False)
        require(zero["c1_sha256"] == C1_SHA and tensor_state_sha256(zero["rectifier_state"]) == init,
                "rectifier initialization drift on resume")
        last = epochs[-1]
        state = torch.load(existing[-1], map_location=device, weights_only=False)
        require(state["epoch"] == last and state["c1_sha256"] == C1_SHA, "rectifier resume checkpoint drift")
        rectifier.load_state_dict(state["rectifier_state"], strict=True)
        if last:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        updates = last * 1105
        require(history[-1]["updates"] == updates, "rectifier optimizer step drift")
        print(json.dumps({"stage": stage, "resume_from_complete_epoch": last,
                          "discarded_partial_epoch": last + 1}), flush=True)
    else:
        initial = rect_metrics(rectifier, val_store, dev, val_cache, sam, source, val_jepa, device)
        history.append({"epoch": 0, "updates": 0, "dev_g0_mean_iou": initial["mean_foreground_iou"]})
        dump(OUT / stage / "history.json", history)
        torch.save({"epoch": 0, "rectifier_state": rectifier.state_dict(), "c1_sha256": C1_SHA}, stage_ckpts / "epoch_0.pt")
        last = 0
    valid = cache["valid"].bool()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    for epoch in range(last + 1, 11):
        began = time.time()
        order = list(ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        rectifier.train()
        traversal = eligible = invalid = 0
        for start in range(0, len(order), 8):
            batch_ids = order[start:start + 8]
            positions = torch.tensor([id_to_i[sid] for sid in batch_ids])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(batch_ids); eligible += len(idx); invalid += len(batch_ids) - len(idx)
            if len(idx) == 0:
                continue
            active_ids = [ids[j] for j in idx.tolist()]
            s64, raw, target, sc, cc = e2.phase4f_spatial_batch(train_store, active_ids, device)
            with torch.no_grad():
                evidence = evidence_feature(source, raw)
            query = cache["q_seg"].index_select(0, idx).to(device=device, dtype=torch.bfloat16)
            token_valid = torch.ones(len(idx), 576, dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = rectifier(s64, evidence, sc, cc, token_valid, auxiliary(train_jepa, active_ids, device))
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(query, output["image_embeddings"].to(torch.bfloat16))
            loss = mask_loss(low, target, CFG)["total"]
            require(bool(torch.isfinite(loss)), "nonfinite rectifier loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(rectifier.parameters(), float(CFG["optimizer"]["gradient_clip_norm"]))
            require(bool(torch.isfinite(norm)), "nonfinite rectifier gradient")
            optimizer.step(); scheduler.step(); updates += 1
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": stage, "epoch": epoch, "update": updates, "loss": float(loss.detach())}), flush=True)
        require((traversal, eligible, invalid) == (8836, 8741, 95), "rectifier exposure drift")
        metrics = rect_metrics(rectifier, val_store, dev, val_cache, sam, source, val_jepa, device)
        row = {"epoch": epoch, "updates": updates, "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "sample_order_sha256": e2.ids_hash(order), "seconds": time.time() - began}
        history.append(row)
        dump(OUT / stage / "history.json", history)
        temp = stage_ckpts / f"epoch_{epoch}.pt.tmp"
        torch.save({"epoch": epoch, "rectifier_state": rectifier.state_dict(), "c1_sha256": C1_SHA,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, temp)
        os.replace(temp, stage_ckpts / f"epoch_{epoch}.pt")
        print(json.dumps({"stage": stage, **row}), flush=True)
    require(frozen == {"sam": hash_state(sam), "source": hash_state(source)}, "frozen backbone drift")
    selector = select(stage, history, "dev_g0_mean_iou", "rectifier_state")
    dump(OUT / stage / "provenance.json", {"status": "COMPLETE", "initial_state_sha256": init,
         "selected": selector, "frozen_hash_before_after": frozen, "c1_valid_train": 8741, "c1_valid_dev": 1090,
         "p1_trained_rectifier_loaded": False, "official1000_used": False})


def selected_state(stage, key):
    selector = json.loads((OUT / stage / "selector.json").read_text())
    from tools.phase4c_b import file_sha256
    require(selector["status"] == "COMPLETE" and selector["c1_sha256"] == C1_SHA, f"{stage} selector drift")
    path = Path(selector["selected_checkpoint"])
    require(file_sha256(path) == selector["selected_checkpoint_sha256"], f"{stage} selected hash drift")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["c1_sha256"] == C1_SHA and key in state, f"{stage} selected payload drift")
    return state[key], selector


def later_stage(stage, device):
    require(stage in ("utility", "joint"), "bad stage")
    require(not (OUT / stage / "selector.json").exists(), f"{stage} already selected")
    hc.seed_all()
    ids, dev, train_store, val_store, cache, val_cache = load_inputs()
    train_jepa, val_jepa = jepa_store("train", ids), jepa_store("val", dev["sample_ids"])
    model, _ = hc.load_utility("a2", device)
    random_utility_hash = tensor_state_sha256(hc.trainable_state(model))
    rectifier = make_rectifier(device)
    rect_state, rect_selector = selected_state("rectifier", "rectifier_state")
    rectifier.load_state_dict(rect_state, strict=True)
    if stage == "joint":
        utility_state, utility_selector = selected_state("utility", "utility_state")
        model.load_state_dict(utility_state, strict=True)
    else:
        utility_selector = None
        rectifier.requires_grad_(False)
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    frozen = {"sam": hash_state(sam), "source": hash_state(source), "heads": tensor_state_sha256(q.source_state(model))}
    if stage == "utility":
        frozen["rectifier"] = hash_state(rectifier)
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in rectifier.parameters() if p.requires_grad]
    expected_rectifier = 329985 + 1280 * 256 + 256 + 2 * 1280 + 1
    expected_count = 371803 if stage == "utility" else 371803 + expected_rectifier
    require(sum(p.numel() for p in params) == expected_count, f"{stage} trainable scope drift")
    optimizer = torch.optim.AdamW(params, lr=e2.LR, weight_decay=e2.WD)
    stage_ckpts = CKPTS / stage
    require(not list(stage_ckpts.glob("epoch_*.pt")), f"{stage} already started; refusing overwrite")
    stage_ckpts.mkdir(parents=True, exist_ok=True)
    valid = cache["valid"].bool()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    history, updates = [], 0
    for epoch in range(1, 11):
        began = time.time()
        order = list(ids)
        random.Random(SEED + 1009 * epoch).shuffle(order)
        model.train(); model.language_source.eval(); model.forensic_source.eval()
        rectifier.train() if stage == "joint" else rectifier.eval()
        sums = defaultdict(float)
        traversal = eligible = invalid = 0
        for begin in range(0, len(order), 8):
            all_ids = order[begin:begin + 8]
            positions = torch.tensor([id_to_i[sid] for sid in all_ids])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(all_ids); eligible += len(idx); invalid += len(all_ids) - len(idx)
            if len(idx) == 0:
                continue
            batch, _ = e2.c1_language_batch(data, cache, idx, train_store, sam, device)
            crossed = dict(batch)
            fi = cross.index_select(0, idx)
            crossed["F24"] = data["F24"].index_select(0, fi).to(device)
            crossed["z_F24"] = data["z_F24"].index_select(0, fi).to(device)
            active_ids = [ids[i] for i in idx.tolist()]
            s64, corrected, support, target, sc = phase4f_batch(train_store, active_ids, source, rectifier, train_jepa, device)
            optimizer.zero_grad(set_to_none=True)
            matched = hc.utility_forward(model, batch)
            crossed_out = hc.utility_forward(model, crossed)
            shuffled = hc.utility_forward(model, batch, permutation=permutation)
            gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, corrected, gate)
            with torch.autocast(device_type="cuda", enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            seg = mask_loss(low, target, CFG)["total"]
            _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]}, data["target64"].index_select(0, idx).to(device))
            relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
            cr = hc.rank_loss(matched["U"], crossed_out["U"], matched["support"])
            sr = hc.rank_loss(matched["U"], shuffled["U"], matched["support"])
            total = seg + relative + 0.5 * (cr + sr)
            require(bool(torch.isfinite(total)), f"nonfinite {stage} loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, e2.GRAD_CLIP)
            require(bool(torch.isfinite(norm)), f"nonfinite {stage} gradient")
            optimizer.step(); updates += 1
            sums["loss"] += float(total.detach()) * len(idx)
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": stage, "epoch": epoch, "update": updates, "loss": float(total.detach())}), flush=True)
        require((traversal, eligible, invalid) == (8836, 8741, 95), f"{stage} exposure drift")
        metrics, _ = evaluate(model, rectifier, sam, source, val_store, dev, val_cache, val_jepa, device)
        row = {"epoch": epoch, "updates": updates, "dev_g0_mean_iou": metrics["mean_foreground_iou"],
               "mean_train_loss": sums["loss"] / eligible, "sample_order_sha256": e2.ids_hash(order), "seconds": time.time() - began}
        history.append(row)
        dump(OUT / stage / "history.json", history)
        temp = stage_ckpts / f"epoch_{epoch}.pt.tmp"
        torch.save({"epoch": epoch, "utility_state": model.state_dict(), "rectifier_state": rectifier.state_dict(),
                    "c1_sha256": C1_SHA, "optimizer": optimizer.state_dict()}, temp)
        os.replace(temp, stage_ckpts / f"epoch_{epoch}.pt")
        print(json.dumps({"stage": stage, **row}), flush=True)
    after = {"sam": hash_state(sam), "source": hash_state(source), "heads": tensor_state_sha256(q.source_state(model))}
    if stage == "utility":
        after["rectifier"] = hash_state(rectifier)
    require(after == frozen, f"{stage} frozen state drift")
    selector = select(stage, history, "dev_g0_mean_iou", "utility_state+rectifier_state")
    dump(OUT / stage / "provenance.json", {"status": "COMPLETE", "selected": selector,
         "rectifier_input_selector_sha256": rect_selector["selected_checkpoint_sha256"],
         "utility_input_selector_sha256": utility_selector["selected_checkpoint_sha256"] if utility_selector else None,
         "random_utility_state_sha256": random_utility_hash, "trainable_parameters": expected_count,
         "frozen_hash_before": frozen, "frozen_hash_after": after,
         "p1_trained_utility_loaded": False, "p1_trained_rectifier_loaded": False,
         "official1000_used": False, "external_ood_used": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rectifier", "utility", "joint"))
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.stage == "rectifier":
        stage1(device)
    else:
        later_stage(args.stage, device)


if __name__ == "__main__":
    main()
