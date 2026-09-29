#!/usr/bin/env python3
"""C2-native Phase4F -> Phase4H-C A2 -> Phase4H-D R1, canonical G0."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4f_run as f4
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4g1q_conditional_utility as q
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts import phase6l0_r2_train as l0
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase6l0_r2_preflight import (CKPT as C2_CKPT, EXPECTED_SHA as C2_SHA,
                                           OUT as C2_CACHE, c2_native_sam_state,
                                           dump, load_native_runtime, require)
from model.pcerf import sam_lowres_to_original_normalized
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import (Phase4FStore, deterministic_order, evidence_feature,
                           invalid_record, load_evidence_source, load_rectifier, mask_loss)

OUT = ROOT / "outputs/phase6p0_c2_native_staged_r1"
CKPTS = Path("/data/yz/groundingLMM_official/checkpoints/phase6p0_c2_native_staged_r1")
CFG = e2.CFG
SEED, EPOCHS, BATCH = 3407, 10, 8
STAGES = ("rectifier", "utility", "joint")
TRAIN_VALID, DEV_VALID = 8682, 1084


def state_hash(module):
    return tensor_state_sha256(module.state_dict())


def frozen_hash(sam, source, model=None):
    result = {"c2_native_sam": state_hash(sam), "forensic_adapter": state_hash(source)}
    if model is not None:
        result["source_heads"] = tensor_state_sha256(q.source_state(model))
    return result


def load_population(split):
    ids = train_ids() if split == "train" else load_dev("g0")["sample_ids"]
    cache = l0.load_cache("train" if split == "train" else "val", ids)
    store = Phase4FStore(CFG, split)
    require(ids == cache["sample_ids"] == store.sample_ids, f"{split} canonical ID/order drift")
    require((len(ids), int(cache["valid_c2_g0"].sum())) ==
            ((8836, TRAIN_VALID) if split == "train" else (1106, DEV_VALID)),
            f"{split} C2 exactly-one-SEG population drift")
    return ids, cache, store


def forensic_data(ids, split):
    if split == "train":
        data = q.load_ids(ids, ("F24", "z_F24", "target64", "clip_geometries"))
    else:
        dev = load_dev("g0")
        data = {"sample_ids": dev["sample_ids"], "F24": dev["F24"],
                "z_F24": dev["z_F24"], "clip_geometries": dev["clip_geometries"]}
    require(data["sample_ids"] == ids, f"{split} frozen forensic ID/order drift")
    return data


def native_sam(device):
    state, provenance = c2_native_sam_state()
    sam = load_native_runtime(state, device)
    manifest = json.loads((C2_CACHE / "train.json").read_text())
    reference = json.loads((C2_CACHE / "preflight_subset.json").read_text())
    require(file_sha256(C2_CACHE / "c2_sam_runtime.pt") ==
            manifest["sam_runtime_sha256"] == reference["runtime_sha256"] and
            state_hash(sam) == reference["sam_state_sha256"],
            "C2-native SAM cache/runtime file or state mismatch")
    require(all(not p.requires_grad for p in sam.parameters()), "C2-native SAM not frozen")
    return sam, provenance


def language_batch(cache, forensic, indices, store, device):
    ids = [cache["sample_ids"][i] for i in indices]
    ix = torch.tensor(indices, dtype=torch.long)
    s64_raw, _, _, _, _ = e2.phase4f_spatial_batch(store, ids, device)
    s64_original = torch.cat([
        sam_lowres_to_original_normalized(s64_raw[j:j+1], store.geometries[sid],
                                          output_hw=(64, 64))
        for j, sid in enumerate(ids)
    ]).to(device, dtype=torch.bfloat16)
    low = cache["z_L"].index_select(0, ix).to(device, dtype=torch.bfloat16)
    z_original = torch.cat([
        sam_lowres_to_original_normalized(low[j:j+1], store.geometries[sid],
                                          output_hw=(256, 256))
        for j, sid in enumerate(ids)
    ]).to(device, dtype=torch.bfloat16)
    require([forensic["clip_geometries"][i] for i in indices] ==
            [geometry_for("clip", store.geometries[sid]["original_hw"]) for sid in ids],
            "C2/forensic CLIP geometry drift")
    return {"S64": s64_original, "q_seg": cache["q_seg"].index_select(0, ix).to(device, dtype=torch.bfloat16),
            "z_L": z_original,
            "F24": forensic["F24"].index_select(0, ix).to(device),
            "z_F24": forensic["z_F24"].index_select(0, ix).to(device),
            "clip_geometries": [forensic["clip_geometries"][i] for i in indices]}


def rectifier_forward(rectifier, source, store, ids, device):
    s64, raw, target, sc, cc = e2.phase4f_spatial_batch(store, ids, device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = rectifier(s64, evidence, sc, cc,
                           torch.ones(len(ids), 576, dtype=torch.bool, device=device))
    return s64, result["image_embeddings"], result["support"].reshape(len(ids), 1, 64, 64), target, sc


def full_forward(model, rectifier, sam, source, store, cache, forensic, indices, device,
                 permutation=None, with_loss=True):
    ids = [cache["sample_ids"][i] for i in indices]
    batch = language_batch(cache, forensic, indices, store, device)
    crossed = dict(batch)
    partner = torch.tensor([(i + 1) % len(cache["sample_ids"]) for i in indices])
    crossed["F24"] = forensic["F24"].index_select(0, partner).to(device)
    crossed["z_F24"] = forensic["z_F24"].index_select(0, partner).to(device)
    s64, corrected, support, target, sc = rectifier_forward(rectifier, source, store, ids, device)
    matched = hc.utility_forward(model, batch)
    gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
    adapted = hc.gated_embedding(s64, corrected, gate)
    with torch.autocast(device_type="cuda", enabled=False):
        low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
    if not with_loss:
        return low, None
    require(permutation is not None, "training permutation missing")
    cross = hc.utility_forward(model, crossed)
    shuffle = hc.utility_forward(model, batch, permutation=permutation)
    seg = mask_loss(low, target, CFG)["total"]
    ix = torch.tensor(indices, dtype=torch.long)
    _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]},
                             forensic["target64"].index_select(0, ix).to(device))
    relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
    rank_cross = hc.rank_loss(matched["U"], cross["U"], matched["support"])
    rank_shuffle = hc.rank_loss(matched["U"], shuffle["U"], matched["support"])
    rank = .5 * (rank_cross + rank_shuffle)
    total = seg + relative + rank
    require(bool(torch.isfinite(total)), "nonfinite full R1 loss")
    return low, {"total": total, "seg": seg, "relative": relative, "rank": rank,
                 "rank_cross": rank_cross, "rank_shuffle": rank_shuffle}


def metric_row(sid, low, store, valid):
    target = store.original_masks[sid]
    if valid:
        row = metric_record(sid, inverse_sam_logits(low, store.geometries[sid]), target)
        row["tn"] = int(target.numel()) - row["tp"] - row["fp"] - row["fn"]
    else:
        row = invalid_record(sid, target)
        row["tn"] = int(target.numel()) - row["fn"]
    row["valid_c2_g0"] = bool(valid)
    return row


def evaluate(stage, epoch, rectifier, sam, source, store, cache, device, model=None, forensic=None):
    rectifier.eval()
    if model is not None:
        model.eval()
        model.language_source.eval()
        model.forensic_source.eval()
    records = []
    with torch.no_grad():
        for i, sid in enumerate(cache["sample_ids"]):
            if not bool(cache["valid_c2_g0"][i]):
                records.append(metric_row(sid, None, store, False))
                continue
            if stage == "rectifier":
                _, corrected, _, _, _ = rectifier_forward(rectifier, source, store, [sid], device)
                qseg = cache["q_seg"][i:i+1].to(device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", enabled=False):
                    low = sam(qseg, corrected.to(torch.bfloat16))
            else:
                low, _ = full_forward(model, rectifier, sam, source, store, cache, forensic,
                                      [i], device, with_loss=False)
            records.append(metric_row(sid, low, store, True))
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "DEV", "arm": stage, "epoch": epoch,
                                  "done": i + 1, "total": len(cache["sample_ids"])}), flush=True)
    metrics = summarize(records)
    metrics.update({"valid_c2_g0": DEV_VALID, "invalid_c2_g0": 1106 - DEV_VALID,
                    **{k: sum(r[k] for r in records) for k in ("tp", "fp", "fn", "tn")}})
    location = OUT / stage
    dump(location / f"dev_epoch_{epoch:02d}.json", metrics)
    with (location / f"dev_epoch_{epoch:02d}_rows.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    return metrics


def gamma_audit(ids, cache, store, device):
    path = OUT / "preflight" / "gamma.json"
    if path.exists():
        value = json.loads(path.read_text())
        require(value["status"] == "PASS" and value["train_valid"] == TRAIN_VALID,
                "existing C2 gamma audit drift")
        return float(value["selected_gamma"])
    order = deterministic_order(ids, 0, seed=SEED)
    index = {sid: i for i, sid in enumerate(ids)}
    selected = [sid for sid in order if bool(cache["valid_c2_g0"][index[sid]])][:8]
    require(len(selected) == 8, "gamma audit needs eight C2-valid TRAIN images")
    s64, raw, _, sc, cc = e2.phase4f_spatial_batch(store, selected, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    with torch.no_grad():
        evidence = evidence_feature(source, raw)
    base = s64.float().flatten(2).transpose(1, 2)
    candidates = []
    for gamma in CFG["architecture"]["gamma_candidates"]:
        rectifier = load_rectifier(CFG, float(gamma), device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = rectifier(s64, evidence, sc, cc,
                               torch.ones(8, 576, dtype=torch.bool, device=device))
        residual = result["residual"].float().flatten(2).transpose(1, 2)
        ratio = residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-12)
        support = result["support"]
        median = float(ratio[support].median())
        changed = (result["image_embeddings"].to(torch.bfloat16) != s64).flatten(2).any(1)
        survival = float(changed[support].float().mean())
        passed = (float(CFG["architecture"]["residual_ratio_range"][0]) <= median <=
                  float(CFG["architecture"]["residual_ratio_range"][1]) and
                  survival >= float(CFG["architecture"]["minimum_bf16_token_survival"]))
        candidates.append({"gamma": float(gamma), "median_ratio": median,
                           "bf16_survival": survival, "pass": passed})
    passing = [row for row in candidates if row["pass"]]
    require(bool(passing), "no C2 TRAIN geometry gamma candidate passed")
    result = {"status": "PASS", "selected_gamma": passing[0]["gamma"],
              "sample_ids": selected, "train_valid": TRAIN_VALID,
              "uses_c2_train_validity": True, "uses_p1_query": False,
              "validation_or_test_used": False, "candidates": candidates}
    dump(path, result)
    return float(result["selected_gamma"])


def preflight(device):
    require(not (OUT / "preflight_gates.json").exists() and
            not any((OUT / x / "selector.json").exists() for x in ("utility", "joint")),
            "C2-native staged R1 Utility/joint already initialized")
    retained_rectifier = OUT / "rectifier" / "selector.json"
    rectifier_reuse_sha = None
    if retained_rectifier.exists():
        prior = json.loads(retained_rectifier.read_text())
        require(prior["status"] == "COMPLETE" and prior["stage"] == "rectifier" and
                prior["c2_sha256"] == C2_SHA and
                file_sha256(Path(prior["selected_checkpoint"])) ==
                prior["selected_checkpoint_sha256"],
                "retained geometry-independent Rectifier selector drift")
        rectifier_reuse_sha = prior["selected_checkpoint_sha256"]
    hc.seed_all()
    ids, train_cache, train_store = load_population("train")
    dev_ids, dev_cache, dev_store = load_population("val")
    sam, sam_provenance = native_sam(device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    initial_frozen = frozen_hash(sam, source)
    gamma = gamma_audit(ids, train_cache, train_store, device)
    rectifier = load_rectifier(CFG, gamma, device)
    model, _ = hc.load_utility("a2", device)
    require((sum(p.numel() for p in rectifier.parameters() if p.requires_grad),
             sum(p.numel() for p in model.parameters() if p.requires_grad)) ==
            (329985, 371803), "R1 trainable parameter scope drift")
    require(all(not p.requires_grad for p in source.parameters()) and
            all(not p.requires_grad for p in model.language_source.parameters()) and
            all(not p.requires_grad for p in model.forensic_source.parameters()),
            "frozen source requires_grad drift")
    source_heads = tensor_state_sha256(q.source_state(model))
    forensic = forensic_data(ids, "train")
    from tools.phase4f import evidence_feature as live_feature
    selected = [i for i in range(len(ids)) if bool(train_cache["valid_c2_g0"][i])][:BATCH]
    with torch.no_grad():
        errors = []
        for i in selected:
            raw = train_store._values(ids[i])[1][None].to(device, dtype=torch.bfloat16)
            live = live_feature(source, raw)
            errors.append(float((live.float() - forensic["F24"][i:i+1].to(device).float()).abs().max()))
    require(max(errors) == 0, "historical F24 cache differs from frozen adapter")
    dev_forensic_reference = load_dev("g0")
    dev_subset = [i for i in range(len(dev_ids)) if bool(dev_cache["valid_c2_g0"][i])][:BATCH]
    reconstructed = language_batch(dev_cache, dev_forensic_reference, dev_subset,
                                   dev_store, device)["S64"]
    require(torch.equal(reconstructed.cpu(),
                        dev_forensic_reference["S64"].index_select(
                            0, torch.tensor(dev_subset)).to(torch.bfloat16)),
            "R1 Utility original-normalized S64 geometry parity failed")
    # Full canonical DEV baseline parity, before touching any trainable state.
    reference = [json.loads(line) for line in
                 (ROOT / "outputs/phase6l0_r2/dev_epoch00_rows.jsonl").read_text().splitlines()]
    require(len(reference) == len(dev_ids) == 1106, "C2-G0 DEV reference population drift")
    with torch.no_grad():
        for i, sid in enumerate(dev_ids):
            valid = bool(dev_cache["valid_c2_g0"][i])
            if valid:
                s64, _, _, _, _ = e2.phase4f_spatial_batch(dev_store, [sid], device)
                query = dev_cache["q_seg"][i:i+1].to(device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", enabled=False):
                    low = sam(query, s64)
                require(torch.equal(low.cpu(), dev_cache["z_L"][i:i+1].cpu()),
                        f"C2-native SAM cached logit mismatch: {sid}")
            else:
                low = None
            require(metric_row(sid, low, dev_store, valid) == reference[i],
                    f"C2-G0 DEV row mismatch: {sid}")
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    rectifier.train()
    batch_ids = [ids[i] for i in selected]
    _, corrected, _, target, _ = rectifier_forward(rectifier, source, train_store, batch_ids, device)
    query = train_cache["q_seg"].index_select(0, torch.tensor(selected)).to(device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", enabled=False):
        low = sam(query, corrected.to(torch.bfloat16))
    rect_loss = mask_loss(low, target, CFG)["total"]
    rect_loss.backward()
    rect_grad = math.sqrt(sum(float(p.grad.float().square().sum()) for p in rectifier.parameters()
                              if p.grad is not None))
    require(math.isfinite(rect_grad) and rect_grad > 0, "Rectifier gradient routing failed")
    rectifier.zero_grad(set_to_none=True)
    model.train()
    _, losses = full_forward(model, rectifier, sam, source, train_store, train_cache,
                             forensic, selected, device, permutation)
    losses["total"].backward()
    utility_grad = math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters()
                                 if p.grad is not None))
    joint_rect_grad = math.sqrt(sum(float(p.grad.float().square().sum()) for p in rectifier.parameters()
                                    if p.grad is not None))
    require(all(math.isfinite(v) and v > 0 for v in (utility_grad, joint_rect_grad)),
            "Utility/joint gradient routing failed")
    require(all(p.grad is None for p in sam.parameters()) and
            all(p.grad is None for p in source.parameters()) and
            all(p.grad is None for p in model.language_source.parameters()) and
            all(p.grad is None for p in model.forensic_source.parameters()),
            "frozen module received gradient")
    require(initial_frozen == frozen_hash(sam, source) and
            source_heads == tensor_state_sha256(q.source_state(model)),
            "frozen source/SAM mutated in preflight")
    audit = {"status": "PASS", "C2_G0_full_DEV_exact": True,
             "C2_checkpoint_sha256": file_sha256(C2_CKPT),
             "C2_native_sam_state_sha256": initial_frozen["c2_native_sam"],
             "sam_provenance": sam_provenance,
             "adapter_checkpoint_sha256": file_sha256(Path(CFG["evidence"]["forensic_checkpoint"])),
             "adapter_state_sha256": initial_frozen["forensic_adapter"],
             "source_heads_state_sha256": source_heads,
             "c2_train_cache_manifest_sha256": file_sha256(C2_CACHE / "train.json"),
             "c2_dev_cache_manifest_sha256": file_sha256(C2_CACHE / "val.json"),
             "trainer_source_sha256": file_sha256(Path(__file__)),
             "gamma": gamma, "F24_live_cache_max_abs_error": max(errors),
             "R1_utility_S64_original_geometry_exact": True,
             "retained_rectifier_selected_checkpoint_sha256": rectifier_reuse_sha,
             "rectifier_grad_norm": rect_grad, "utility_grad_norm": utility_grad,
             "joint_rectifier_grad_norm": joint_rect_grad,
             "train_total": len(ids), "train_valid": TRAIN_VALID,
             "dev_total": len(dev_ids), "dev_valid": DEV_VALID,
             "trainable_rectifier": 329985, "trainable_utility": 371803,
             "selection": "per-stage canonical internal DEV Mean FG IoU, earlier epoch tie",
             "stages": list(STAGES), "official1000_accessed": False,
             "internal_test_accessed": False}
    dump(OUT / "preflight_gates.json", audit)
    print(json.dumps({"status": "PREFLIGHT_PASS", "gamma": gamma,
                      "C2_G0_full_DEV_exact": True}), flush=True)


def selected(stage, field):
    selector = json.loads((OUT / stage / "selector.json").read_text())
    require(selector["status"] == "COMPLETE" and selector["stage"] == stage,
            f"{stage} selector missing")
    path = Path(selector["selected_checkpoint"])
    require(file_sha256(path) == selector["selected_checkpoint_sha256"],
            f"{stage} checkpoint hash drift")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["stage"] == stage and state["c2_sha256"] == C2_SHA and field in state,
            f"{stage} checkpoint payload drift")
    return state[field], selector


def choose(stage, history):
    best = max(history, key=lambda row: (row["dev_mean_fg_iou"], -row["epoch"]))
    path = CKPTS / stage / f"epoch_{best['epoch']:02d}.pt"
    target = OUT / stage / "selected_checkpoint.pt"
    shutil.copy2(path, target)
    selector = {"status": "COMPLETE", "stage": stage, "selected_epoch": best["epoch"],
                "selected_checkpoint": str(target), "selected_checkpoint_sha256": file_sha256(target),
                "primary": "canonical internal DEV Mean FG IoU only", "tie_break": "earlier epoch",
                "selected_dev_mean_fg_iou": best["dev_mean_fg_iou"], "c2_sha256": C2_SHA,
                "official1000_used_for_selection": False}
    dump(OUT / stage / "selector.json", selector)
    return selector


def run_stage(stage, device):
    require(stage in STAGES, "invalid stage")
    gates = json.loads((OUT / "preflight_gates.json").read_text())
    require(gates["status"] == "PASS" and gates["C2_G0_full_DEV_exact"] and
            gates["trainer_source_sha256"] == file_sha256(Path(__file__)) and
            gates["C2_checkpoint_sha256"] == file_sha256(C2_CKPT) == C2_SHA and
            gates["c2_train_cache_manifest_sha256"] == file_sha256(C2_CACHE / "train.json") and
            gates["c2_dev_cache_manifest_sha256"] == file_sha256(C2_CACHE / "val.json"),
            "C2-native R1 preflight/source drift")
    require(not (OUT / stage / "selector.json").exists(), f"{stage} already selected")
    hc.seed_all()
    ids, train_cache, train_store = load_population("train")
    _, dev_cache, dev_store = load_population("val")
    sam, _ = native_sam(device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    require(frozen_hash(sam, source) == {"c2_native_sam": gates["C2_native_sam_state_sha256"],
                                        "forensic_adapter": gates["adapter_state_sha256"]},
            "frozen SAM/adapter changed after preflight")
    rectifier = load_rectifier(CFG, float(gates["gamma"]), device)
    model = None
    if stage != "rectifier":
        rect_state, rect_selector = selected("rectifier", "rectifier_state")
        rectifier.load_state_dict(rect_state, strict=True)
        model, _ = hc.load_utility("a2", device)
        require(tensor_state_sha256(q.source_state(model)) == gates["source_heads_state_sha256"],
                "frozen R1 source-head state drift")
        if stage == "joint":
            utility_state, utility_selector = selected("utility", "utility_state")
            model.load_state_dict(utility_state, strict=True)
        else:
            utility_selector = None
            rectifier.requires_grad_(False)
    else:
        rect_selector = utility_selector = None
    trainable = [p for p in (rectifier.parameters() if model is None else
                            list(rectifier.parameters()) + list(model.parameters())) if p.requires_grad]
    require(sum(p.numel() for p in trainable) ==
            (329985 if stage == "rectifier" else 371803 if stage == "utility" else 701788),
            f"{stage} trainable parameter drift")
    frozen_before = frozen_hash(sam, source, model)
    if stage == "utility":
        frozen_before["rectifier"] = state_hash(rectifier)
    initial = {"rectifier": state_hash(rectifier)}
    if model is not None:
        initial["utility"] = state_hash(model)
    dump(OUT / stage / "initialization.json", {
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP", "stage": stage,
        "initial_state_sha256": initial, "trainable_parameters": sum(p.numel() for p in trainable),
        "rectifier_input_selector_sha256": rect_selector["selected_checkpoint_sha256"] if rect_selector else None,
        "utility_input_selector_sha256": utility_selector["selected_checkpoint_sha256"] if utility_selector else None,
        "historical_P1_rectifier_or_utility_loaded": False, "c2_sha256": C2_SHA})
    forensic_train = forensic_data(ids, "train") if model is not None else None
    forensic_dev = forensic_data(dev_cache["sample_ids"], "val") if model is not None else None
    if stage == "rectifier":
        optimizer, scheduler = f4.optimizer_scheduler(rectifier, EPOCHS * 1105)
        metrics = evaluate(stage, 0, rectifier, sam, source, dev_store, dev_cache, device)
        history = [{"epoch": 0, "updates": 0, "dev_mean_fg_iou": metrics["mean_foreground_iou"],
                    "dev_global_fg_iou": metrics["global_foreground_iou"]}]
        ckpt0 = CKPTS / stage / "epoch_00.pt"
        ckpt0.parent.mkdir(parents=True, exist_ok=True)
        require(not ckpt0.exists(), "existing rectifier stage artifact")
        torch.save({"stage": stage, "epoch": 0, "rectifier_state": rectifier.state_dict(),
                    "c2_sha256": C2_SHA}, ckpt0)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=e2.LR, weight_decay=e2.WD)
        scheduler = None
        history = []
    ckpt_dir = CKPTS / stage
    require(not list(ckpt_dir.glob("epoch_0[1-9].pt")) and
            not (OUT / stage / "history.csv").exists(), f"existing {stage} training artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    index = {sid: i for i, sid in enumerate(ids)}
    valid = train_cache["valid_c2_g0"]
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    updates = 0
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        order = deterministic_order(ids, epoch, seed=SEED)
        rectifier.train() if stage in ("rectifier", "joint") else rectifier.eval()
        if model is not None:
            model.train()
            model.language_source.eval()
            model.forensic_source.eval()
        exposures = eligible = invalid = steps = clipped = 0
        losses = {key: 0.0 for key in ("total", "seg", "relative", "rank", "rank_cross", "rank_shuffle")}
        for offset in range(0, len(order), BATCH):
            candidates = [index[sid] for sid in order[offset:offset+BATCH]]
            chosen = [i for i in candidates if bool(valid[i])]
            exposures += len(candidates)
            eligible += len(chosen)
            invalid += len(candidates) - len(chosen)
            if not chosen:
                continue
            steps += 1
            optimizer.zero_grad(set_to_none=True)
            if stage == "rectifier":
                batch_ids = [ids[i] for i in chosen]
                _, corrected, _, target, _ = rectifier_forward(rectifier, source, train_store,
                                                                 batch_ids, device)
                query = train_cache["q_seg"].index_select(0, torch.tensor(chosen)).to(
                    device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", enabled=False):
                    low = sam(query, corrected.to(torch.bfloat16))
                seg = mask_loss(low, target, CFG)["total"]
                parts = {"total": seg, "seg": seg}
            else:
                _, parts = full_forward(model, rectifier, sam, source, train_store, train_cache,
                                        forensic_train, chosen, device, permutation)
            loss = parts["total"]
            require(bool(torch.isfinite(loss)), f"nonfinite {stage} loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, float(CFG["optimizer"]["gradient_clip_norm"])
                                                   if stage == "rectifier" else e2.GRAD_CLIP)
            require(bool(torch.isfinite(norm)), f"nonfinite {stage} gradient")
            clipped += int(float(norm) > (float(CFG["optimizer"]["gradient_clip_norm"])
                                          if stage == "rectifier" else e2.GRAD_CLIP))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            updates += 1
            for key, value in parts.items():
                losses[key] += float(value.detach()) * len(chosen)
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "arm": stage, "epoch": epoch,
                                  "update": updates, "loss": float(loss.detach())}), flush=True)
        require((exposures, eligible, invalid, steps) == (8836, TRAIN_VALID, 154, 1105),
                f"{stage} TRAIN exposure/update drift")
        require(all(bool(torch.isfinite(p).all()) for p in trainable),
                f"{stage} nonfinite trained parameter")
        metrics = evaluate(stage, epoch, rectifier, sam, source, dev_store, dev_cache,
                           device, model, forensic_dev)
        row = {"epoch": epoch, "updates": updates, "exposures": exposures,
               "eligible": eligible, "invalid": invalid, "steps": steps,
               "dev_mean_fg_iou": metrics["mean_foreground_iou"],
               "dev_mean_fg_f1": metrics["mean_foreground_f1"],
               "dev_global_fg_iou": metrics["global_foreground_iou"],
               "dev_global_fg_f1": metrics["global_foreground_f1"],
               "train_loss": losses["total"] / eligible,
               "seg_loss": losses["seg"] / eligible,
               "relative_loss": losses["relative"] / eligible,
               "rank_loss": losses["rank"] / eligible,
               "rank_cross_loss": losses["rank_cross"] / eligible,
               "rank_shuffle_loss": losses["rank_shuffle"] / eligible,
               "clip_frequency": clipped / steps,
               "sample_order_sha256": e2.ids_hash(order), "seconds": time.time() - started}
        history.append(row)
        path = OUT / stage / "history.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        ckpt = ckpt_dir / f"epoch_{epoch:02d}.pt"
        tmp = ckpt.with_suffix(".pt.tmp")
        payload = {"stage": stage, "epoch": epoch, "updates": updates,
                   "rectifier_state": {k: v.detach().cpu() for k, v in rectifier.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "c2_sha256": C2_SHA,
                   "validation": metrics, "initialization": initial}
        if model is not None:
            payload["utility_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, tmp)
        os.replace(tmp, ckpt)
        require(frozen_hash(sam, source, model) ==
                {k: v for k, v in frozen_before.items() if k != "rectifier"} and
                (stage != "utility" or state_hash(rectifier) == frozen_before["rectifier"]),
                f"{stage} frozen runtime/source drift")
        print(json.dumps({"stage": "EPOCH_COMPLETE", "arm": stage, **row}), flush=True)
    selector = choose(stage, history)
    dump(OUT / stage / "provenance.json", {
        "status": "COMPLETE", "stage": stage, "selected": selector,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_hash_before_after": frozen_before, "initial_state_sha256": initial,
        "epochs": EPOCHS, "updates": updates, "valid_train": TRAIN_VALID,
        "valid_dev": DEV_VALID, "official1000_used_for_selection": False})
    print(json.dumps({"stage": "SELECTED", "arm": stage,
                      "epoch": selector["selected_epoch"],
                      "dev_mean_fg_iou": selector["selected_dev_mean_fg_iou"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight",) + STAGES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    if args.stage == "preflight":
        preflight(device)
    else:
        run_stage(args.stage, device)


if __name__ == "__main__":
    main()
