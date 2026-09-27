#!/usr/bin/env python3
"""Phase6H.1: exact Phase6E.2 main replay, then C-only spatial auxiliary loss."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record, paired
from tools.phase4e1 import summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, invalid_record, load_evidence_source, load_sam_runtime, mask_loss

OUT = ROOT / "outputs/phase6h"
ARMS = {"A0": "A0_phase6e2_repro", "A1": "A1_c_spatial_supervision"}
LAMBDA = 0.1
SEED, EPOCHS, BATCH, LR, WD, CLIP = e2.SEED, e2.EPOCHS, e2.BATCH, e2.LR, e2.WD, e2.GRAD_CLIP
HISTORICAL = ROOT / "outputs/phase6e2_c1_specific_r1/selector.json"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text())


def manifest(model, rectifier, aux=None):
    parts = [("utility", model), ("rectifier", rectifier)]
    if aux is not None:
        parts.append(("aux_head", aux))
    return [dict(name=f"{kind}.{name}" if kind != "aux_head" else f"aux_head.{name}",
                 shape=list(param.shape), numel=param.numel(), requires_grad=True, module=kind)
            for kind, module in parts for name, param in module.named_parameters() if param.requires_grad]


def initial_model(device, with_aux: bool):
    hc.seed_all()
    model, rectifier, rectifier_path = hd.load_common(device)
    rectifier.requires_grad_(True)
    original_hashes = {"utility": tensor_state_sha256(model.state_dict()),
                       "rectifier": tensor_state_sha256(rectifier.state_dict())}
    aux = None
    if with_aux:
        # Auxiliary initialization must not shift any original model RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 6001)
            aux = nn.Conv2d(256, 1, kernel_size=1, bias=True)
        aux = aux.to(device)
    return model, rectifier, aux, original_hashes, rectifier_path


def preflight():
    if OUT.exists() and any(OUT.glob("A*/selected_checkpoint.pt")):
        raise RuntimeError("Phase6H training already has a selected checkpoint")
    a0 = initial_model(torch.device("cpu"), False)
    a1 = initial_model(torch.device("cpu"), True)
    h0, h1 = a0[3], a1[3]
    if h0 != h1:
        raise RuntimeError("A0/A1 original initial state mismatch")
    old = read(ROOT / "outputs/phase6e2_c1_specific_r1/initialization_provenance.json")
    expected = {"utility": old["new_r1_initialization"]["utility_state_sha256"],
                "rectifier": old["new_r1_initialization"]["rectifier_state_sha256"]}
    if h0 != expected:
        raise RuntimeError(f"original Phase6E.2 initial state mismatch: {h0} vs {expected}")
    m0, m1 = manifest(*a0[:2]), manifest(*a1[:3])
    d0, d1 = {x["name"]: x for x in m0}, {x["name"]: x for x in m1}
    added = set(d1) - set(d0)
    if d0 != {k: d1[k] for k in d0} or added != {"aux_head.weight", "aux_head.bias"}:
        raise RuntimeError(f"trainable parameter diff drift: {added}")
    if sum(d1[x]["numel"] for x in added) != 257:
        raise RuntimeError("auxiliary head must contain exactly 257 parameters")
    if sum(x["numel"] for x in m0 if x["module"] == "utility") != 371803:
        raise RuntimeError("Utility trainable count drift")
    if sum(x["numel"] for x in m0 if x["module"] == "rectifier") != 329985:
        raise RuntimeError("Rectifier trainable count drift")
    cache_status = read(e2.BASE_OUT / "cache/status.json")
    if cache_status["status"] != "COMPLETE" or cache_status["c1_sha256"] != e2.C1_SHA:
        raise RuntimeError("C1 cache incomplete/drifted")
    assert cache_status["splits"]["train"]["n"] == 8836
    assert cache_status["splits"]["val"]["n"] == 1106
    assert cache_status["splits"]["val"]["valid_exactly_one_seg"] == 1090
    c1 = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
    if file_sha256(c1) != e2.C1_SHA:
        raise RuntimeError("C1 checkpoint drift")
    if file_sha256(a0[4]) != old["new_r1_initialization"]["rectifier_source_sha256"]:
        raise RuntimeError("Phase4F rectifier source drift")
    if file_sha256(hd.A2_SELECTED) != old["new_r1_initialization"]["utility_source_sha256"]:
        raise RuntimeError("Phase4H-C A2 source drift")
    dump(OUT / "trainable_params_A0.json", m0)
    dump(OUT / "trainable_params_A1.json", m1)
    dump(OUT / "initialization_hashes.json", {"status": "PASS", "A0": h0, "A1": h1,
         "Phase6E2_initial_expected": expected, "extra_trainable_A1": sorted(added),
         "extra_parameter_count": 257, "c1_sha256": e2.C1_SHA,
         "utility_source_sha256": file_sha256(hd.A2_SELECTED),
         "rectifier_source_sha256": file_sha256(a0[4]),
         "cache_splits": {k: {t: v for t, v in x.items() if t != "shards"}
                          for k, x in cache_status["splits"].items()},
         "historical_selector": {k: read(HISTORICAL)[k] for k in
              ("selected_epoch", "selected_dev_g0", "selected_checkpoint_sha256")}})
    print(json.dumps({"stage": "PREFLIGHT", "status": "PASS", "A0_A1_original_hashes_equal": True,
                      "extra_A1": sorted(added), "gpu_1_2_not_used": True}), flush=True)


def aux_loss(logits, targets):
    # Exactly the Phase6E.2 interpolation and soft-Dice definition; unit weights.
    cfg = {"loss": {"mask_bce_weight": 1.0, "mask_dice_weight": 1.0}}
    return mask_loss(logits, targets, cfg)


def evaluate_a1(model, rectifier, aux, sam, source, store, dev, cache, device):
    records, aux_records = [], []
    model.eval(); model.language_source.eval(); model.forensic_source.eval()
    rectifier.eval(); aux.eval()
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                row = invalid_record(sid, dev["original_masks"][i]); row["valid_g0"] = False
                records.append(row)
                aux_records.append(dict(row))
                continue
            idx = torch.tensor([i])
            batch, _ = e2.c1_language_batch(dev, cache, idx, store, sam, device)
            s64, p4f, support, _, sc = e2.phase4f_batch(store, [sid], source, rectifier, device)
            out = hc.utility_forward(model, batch)
            gate = hc.gate_to_sam_grid(out["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, p4f, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]),
                                dev["original_masks"][i]); row["valid_g0"] = True
            records.append(row)
            correction = p4f.float() - s64.float()
            logits = aux(correction)
            aux_row = metric_record(sid, inverse_sam_logits(logits, dev["sam_geometries"][i]),
                                    dev["original_masks"][i]); aux_row["valid_g0"] = True
            aux_records.append(aux_row)
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "INTERNAL_VAL", "arm": "A1", "done": i + 1}), flush=True)
    return summarize(records), records, summarize(aux_records), aux_records


def write_records(path, records, cache):
    dump(path, [dict(r, valid_seg=bool(cache["valid"][i]), seg_count=int(cache["seg_count"][i]))
                for i, r in enumerate(records)])


def loss_scale_check(model, rectifier, aux, sam, source, train_store, train_cache, data, ids, device):
    order = list(ids); random.Random(SEED + 1009).shuffle(order)
    valid = train_cache["valid"].bool(); id_to_i = {sid: i for i, sid in enumerate(ids)}
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    entries = []
    model.train(); model.language_source.eval(); model.forensic_source.eval(); rectifier.train(); aux.train()
    for begin in range(0, len(order), BATCH):
        positions = torch.tensor([id_to_i[sid] for sid in order[begin:begin + BATCH]])
        idx = positions[valid.index_select(0, positions)]
        if not len(idx): continue
        batch, _ = e2.c1_language_batch(data, train_cache, idx, train_store, sam, device)
        fi = cross.index_select(0, idx); crossed = dict(batch)
        crossed["F24"] = data["F24"].index_select(0, fi).to(device)
        crossed["z_F24"] = data["z_F24"].index_select(0, fi).to(device)
        s64, p4f, support, targets, sc = e2.phase4f_batch(
            train_store, [ids[i] for i in idx.tolist()], source, rectifier, device)
        matched = hc.utility_forward(model, batch)
        crossed_out = hc.utility_forward(model, crossed)
        shuffled = hc.utility_forward(model, batch, permutation=permutation)
        gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
        adapted = hc.gated_embedding(s64, p4f, gate)
        with torch.autocast(device_type=device.type, enabled=False):
            low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
        seg = mask_loss(low, targets, e2.CFG)
        _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]},
                                  data["target64"].index_select(0, idx).to(device))
        relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
        cr = hc.rank_loss(matched["U"], crossed_out["U"], matched["support"])
        sr = hc.rank_loss(matched["U"], shuffled["U"], matched["support"])
        ranking = .5 * (cr + sr); original = seg["total"] + relative + ranking
        correction = p4f.float() - s64.float()
        if correction.shape != (len(idx), 256, 64, 64) or not correction.requires_grad:
            raise RuntimeError("true injected C shape/gradient drift")
        al = aux_loss(aux(correction), targets)
        values = {"L_seg": float(seg["total"].detach()), "L_relative": float(relative.detach()),
                  "L_ranking": float(ranking.detach()), "L_original_R1": float(original.detach()),
                  "L_aux_BCE": float(al["bce"].detach()), "L_aux_Dice": float(al["dice"].detach()),
                  "L_aux": float(al["total"].detach()), "lambda_aux_times_L_aux": float((LAMBDA * al["total"]).detach()),
                  "L_total": float((original + LAMBDA * al["total"]).detach())}
        values["aux_to_original_ratio"] = values["lambda_aux_times_L_aux"] / max(values["L_original_R1"], 1e-12)
        entries.append(values)
        if len(entries) == 3: break
    if len(entries) != 3:
        raise RuntimeError("not enough valid sanity-check batches")
    if max(x["aux_to_original_ratio"] for x in entries) >= 2.0:
        raise RuntimeError("lambda=0.1 auxiliary loss overwhelms original task; stop before A1 training")
    return entries


def train(arm: str):
    if read(OUT / "initialization_hashes.json")["status"] != "PASS":
        raise RuntimeError("preflight required")
    root = OUT / ARMS[arm]
    if root.exists() and any(p.name != "logs" for p in root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output {root}")
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    model, rectifier, aux, hashes, rectifier_path = initial_model(device, arm == "A1")
    if hashes != read(OUT / "initialization_hashes.json")["A0"]:
        raise RuntimeError("runtime original initialization mismatch")
    status = read(e2.BASE_OUT / "cache/status.json")
    if status["status"] != "COMPLETE" or status["c1_sha256"] != e2.C1_SHA:
        raise RuntimeError("C1 cache status drift")
    train_store, val_store = Phase4FStore(e2.CFG, "train"), Phase4FStore(e2.CFG, "val")
    ids = train_ids(); dev = load_dev("g0")
    tc, vc = e2.load_c1_cache("train", ids), e2.load_c1_cache("val", dev["sample_ids"])
    if ids != train_store.sample_ids or dev["sample_ids"] != val_store.sample_ids:
        raise RuntimeError("Phase4F spatial population/order drift")
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    valid = tc["valid"].bool(); id_to_i = {sid: i for i, sid in enumerate(ids)}
    sam = load_sam_runtime(e2.CFG, device)
    source = load_evidence_source(e2.CFG, "forensic_rect", device)
    frozen_before = {"sam": tensor_state_sha256(sam.state_dict()),
                     "source": tensor_state_sha256(source.state_dict()),
                     "heads": tensor_state_sha256(q.source_state(model))}
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in rectifier.parameters() if p.requires_grad]
    if aux is not None: params += list(aux.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    cross = torch.tensor([(i + 1) % len(ids) for i in range(len(ids))])
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    dump(root / "config_snapshot/protocol.json", {
        "arm": arm, "phase6e2_config": hd.config_contract(), "original_trainer": str(e2.__file__),
        "c1_sha256": e2.C1_SHA, "initial_hashes": hashes, "rectifier_source": str(rectifier_path),
        "train_count": len(ids), "train_valid": int(valid.sum()), "val_count": len(dev["sample_ids"]),
        "val_valid": int(vc["valid"].sum()), "train_sample_ids_sha256": e2.ids_hash(ids),
        "val_sample_ids_sha256": e2.ids_hash(dev["sample_ids"]), "lambda_aux": LAMBDA if aux else None,
        "physical_gpu": 1 if arm == "A0" else 2})
    if aux is not None:
        entries = loss_scale_check(model, rectifier, aux, sam, source, train_store, tc, data, ids, device)
        dump(root / "loss_scale_check.json", {"status": "PASS", "lambda_aux": LAMBDA,
             "rule": "stop if lambda_aux*L_aux / L_original_R1 >= 2 on any of first three valid batches",
             "batches": entries})
        print(json.dumps({"stage": "LOSS_SCALE", "status": "PASS", "ratios":
                          [x["aux_to_original_ratio"] for x in entries]}), flush=True)
    history, updates = [], 0
    for epoch in range(1, EPOCHS + 1):
        began = time.time(); order = list(ids); random.Random(SEED + 1009 * epoch).shuffle(order)
        sums = defaultdict(float); traversal = eligible = invalid = 0
        model.train(); model.language_source.eval(); model.forensic_source.eval(); rectifier.train()
        if aux is not None: aux.train()
        for begin in range(0, len(order), BATCH):
            positions = torch.tensor([id_to_i[sid] for sid in order[begin:begin + BATCH]])
            idx = positions[valid.index_select(0, positions)]
            traversal += len(order[begin:begin + BATCH]); eligible += len(idx)
            invalid += len(order[begin:begin + BATCH]) - len(idx)
            if not len(idx): continue
            batch, _ = e2.c1_language_batch(data, tc, idx, train_store, sam, device)
            fi = cross.index_select(0, idx); crossed = dict(batch)
            crossed["F24"] = data["F24"].index_select(0, fi).to(device)
            crossed["z_F24"] = data["z_F24"].index_select(0, fi).to(device)
            batch_ids = [ids[i] for i in idx.tolist()]
            s64, p4f, support, targets, sc = e2.phase4f_batch(train_store, batch_ids, source, rectifier, device)
            optimizer.zero_grad(set_to_none=True)
            matched = hc.utility_forward(model, batch)
            crossed_out = hc.utility_forward(model, crossed)
            shuffled = hc.utility_forward(model, batch, permutation=permutation)
            gate = hc.gate_to_sam_grid(matched["U"], sc) * support.float()
            adapted = hc.gated_embedding(s64, p4f, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(batch["q_seg"], adapted.to(torch.bfloat16))
            seg = mask_loss(low, targets, e2.CFG)
            _, soft = q.target_delta({"p_L": matched["p_L"], "p_F": matched["p_F"]},
                                      data["target64"].index_select(0, idx).to(device))
            relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
            cr = hc.rank_loss(matched["U"], crossed_out["U"], matched["support"])
            sr = hc.rank_loss(matched["U"], shuffled["U"], matched["support"])
            ranking = .5 * (cr + sr); original = seg["total"] + relative + ranking
            total = original
            if aux is not None:
                correction = p4f.float() - s64.float()
                if correction.shape != (len(idx), 256, 64, 64) or not correction.requires_grad:
                    raise RuntimeError("C shape/gradient drift")
                al = aux_loss(aux(correction), targets)
                total = original + LAMBDA * al["total"]
                for key, val in (("aux_bce_loss", al["bce"]), ("aux_dice_loss", al["dice"]),
                                 ("aux_loss", al["total"]), ("weighted_aux_loss", LAMBDA * al["total"])):
                    sums[key] += float(val.detach()) * len(idx)
            if not torch.isfinite(total): raise RuntimeError("nonfinite loss")
            total.backward(); norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
            if not torch.isfinite(norm): raise RuntimeError("nonfinite gradient")
            optimizer.step(); updates += 1; n = len(idx)
            for key, value in (("seg_loss", seg["total"]), ("relative_loss", relative),
                               ("ranking_loss", ranking), ("cross_rank_loss", cr),
                               ("shuffle_rank_loss", sr), ("original_loss", original),
                               ("total_loss", total)):
                sums[key] += float(value.detach()) * n
            if updates <= 3 or updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "arm": arm, "epoch": epoch,
                                  "update": updates, "loss": float(total.detach()),
                                  "original_loss": float(original.detach()),
                                  "grad_norm": float(norm)}), flush=True)
        if aux is None:
            val_metric, val_records = e2.evaluate(model, rectifier, sam, source, val_store, dev, vc, device)
            aux_metric = aux_records = None
        else:
            val_metric, val_records, aux_metric, aux_records = evaluate_a1(
                model, rectifier, aux, sam, source, val_store, dev, vc, device)
        row = {"epoch": epoch, "optimizer_updates": updates,
               **{k: sums[k] / eligible for k in ("seg_loss", "relative_loss", "ranking_loss",
                   "cross_rank_loss", "shuffle_rank_loss", "original_loss", "total_loss")},
               "traversal_exposures": traversal, "optimization_eligible_exposures": eligible,
               "invalid_g0_exposures": invalid, "dev_g0_mean_iou": val_metric["mean_foreground_iou"],
               "dev_g0_mean_f1": val_metric["mean_foreground_f1"],
               "dev_g0_global_iou": val_metric["global_foreground_iou"],
               "dev_g0_global_f1": val_metric["global_foreground_f1"],
               "sample_order_sha256": e2.ids_hash(order), "seconds": time.time() - began}
        if aux is not None:
            row.update({k: sums[k] / eligible for k in
                        ("aux_bce_loss", "aux_dice_loss", "aux_loss", "weighted_aux_loss")})
            row.update({"aux_dev_mean_iou": aux_metric["mean_foreground_iou"],
                        "aux_dev_mean_f1": aux_metric["mean_foreground_f1"],
                        "aux_dev_global_iou": aux_metric["global_foreground_iou"],
                        "aux_dev_global_f1": aux_metric["global_foreground_f1"]})
        history.append(row); dump(root / "per_epoch_metrics.json", history)
        ckpt = {"schema": "phase6h1_checkpoint_v1", "arm": arm, "epoch": epoch,
                "optimizer_updates": updates,
                "utility_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "rectifier_state": {k: v.detach().cpu() for k, v in rectifier.state_dict().items()},
                "optimizer": optimizer.state_dict(), "validation_g0": val_metric,
                "initial_hashes": hashes, "c1_sha256": e2.C1_SHA}
        if aux is not None:
            ckpt["aux_state"] = {k: v.detach().cpu() for k, v in aux.state_dict().items()}
            ckpt["aux_validation_g0"] = aux_metric
        path = root / f"checkpoints/epoch_{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp"); torch.save(ckpt, tmp); tmp.replace(path)
        # Persist per-epoch records so selection/finalization does not rerun or change the evaluator.
        write_records(root / f"validation/epoch_{epoch}_final.json", val_records, vc)
        if aux_records is not None:
            write_records(root / f"validation/epoch_{epoch}_aux.json", aux_records, vc)
        print(json.dumps({"stage": "EPOCH_COMPLETE", "arm": arm, **row}), flush=True)
    best = max(history, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    chosen = root / f"checkpoints/epoch_{best['epoch']}.pt"
    shutil.copy2(chosen, root / "selected_checkpoint.pt")
    selected_rows = read(root / f"validation/epoch_{best['epoch']}_final.json")
    dump(root / ("val_per_sample.json" if aux is None else "final_val_per_sample.json"), selected_rows)
    if aux is not None:
        dump(root / "aux_val_per_sample.json", read(root / f"validation/epoch_{best['epoch']}_aux.json"))
    frozen_after = {"sam": tensor_state_sha256(sam.state_dict()),
                    "source": tensor_state_sha256(source.state_dict()),
                    "heads": tensor_state_sha256(q.source_state(model))}
    if frozen_after != frozen_before:
        raise RuntimeError("frozen runtime/source mutated")
    historical = read(HISTORICAL)
    delta = best["dev_g0_mean_iou"] - historical["selected_dev_g0"]
    repro_gate = "PASS" if .195 <= best["dev_g0_mean_iou"] < .210 else "STOP_INVESTIGATE"
    summary = {"status": "COMPLETE", "arm": arm, "selected_epoch": best["epoch"],
               "selected_final_metrics": torch.load(chosen, map_location="cpu", weights_only=False)["validation_g0"],
               "selected_aux_metrics": aux_metric if aux is not None and best["epoch"] == EPOCHS else
                   torch.load(chosen, map_location="cpu", weights_only=False).get("aux_validation_g0"),
               "selected_checkpoint_sha256": file_sha256(root / "selected_checkpoint.pt"),
               "historical_anchor": historical["selected_dev_g0"],
               "difference_from_historical": delta,
               "reproduction_gate": repro_gate if arm == "A0" else "NOT_APPLICABLE",
               "optimizer_updates": updates, "initial_hashes": hashes,
               "frozen_hash_before": frozen_before, "frozen_hash_after": frozen_after,
               "population": {"train": len(ids), "train_valid": int(valid.sum()),
                              "validation": len(dev["sample_ids"]),
                              "validation_valid": int(vc["valid"].sum())},
               "physical_gpu": 1 if arm == "A0" else 2}
    dump(root / "summary.json", summary)
    print(json.dumps({"stage": "TRAIN_COMPLETE", "arm": arm, "selected_epoch": best["epoch"],
                      "selected_mean_iou": best["dev_g0_mean_iou"], "reproduction_gate":
                      summary["reproduction_gate"]}), flush=True)


def finalize():
    a0 = read(OUT / ARMS["A0"] / "summary.json")
    a1 = read(OUT / ARMS["A1"] / "summary.json")
    if a0["reproduction_gate"] != "PASS": raise RuntimeError("A0 reproduction did not pass")
    r0 = read(OUT / ARMS["A0"] / "val_per_sample.json")
    r1 = read(OUT / ARMS["A1"] / "final_val_per_sample.json")
    if len(r0) != len(r1) or len(r0) != 1106:
        raise RuntimeError("validation count mismatch")
    if [(x["sample_id"],x["valid_seg"],x["seg_count"]) for x in r0] != [
            (x["sample_id"],x["valid_seg"],x["seg_count"]) for x in r1]:
        raise RuntimeError("A0/A1 sample identity/SEG validity mismatch")
    stats = paired(r1, r0, repeats=10000, seed=SEED)
    joined = [dict(sample_id=x["sample_id"],valid_seg=x["valid_seg"],seg_count=x["seg_count"],
                   A0_fg_iou=x["foreground_iou"],A1_fg_iou=y["foreground_iou"],
                   delta_fg_iou=y["foreground_iou"]-x["foreground_iou"],
                   A0_fg_f1=x["foreground_f1"],A1_fg_f1=y["foreground_f1"],
                   delta_fg_f1=y["foreground_f1"]-x["foreground_f1"])
              for x,y in zip(r0,r1)]
    dest = OUT / "paired_A0_A1"
    dump(dest / "paired_metrics.json", joined)
    dump(dest / "bootstrap.json", stats)
    m0, m1 = a0["selected_final_metrics"], a1["selected_final_metrics"]
    gain = stats["foreground_iou"]["mean_difference"]
    ci_low = stats["foreground_iou"]["bootstrap_95_ci"][0]
    global_iou_delta = m1["global_foreground_iou"]-m0["global_foreground_iou"]
    global_f1_delta = m1["global_foreground_f1"]-m0["global_foreground_f1"]
    # Any opposite-direction global decrease is marked for review rather than silently passing.
    tradeoff = gain > 0 and (global_iou_delta < 0 or global_f1_delta < 0)
    decision = "PASS" if gain >= .010 and ci_low > 0 and not tradeoff else "STOP"
    summary = {"decision": decision, "A0": m0, "A1": m1,
               "A1_aux": a1["selected_aux_metrics"], "paired": stats,
               "delta_global_iou": global_iou_delta, "delta_global_f1": global_f1_delta,
               "tradeoff": tradeoff, "primary_mean_gain_at_least_0_010": gain >= .010,
               "primary_bootstrap_lower_positive": ci_low > 0,
               "seg_trigger_rate": sum(x["valid_seg"] for x in r0)/len(r0),
               "population": len(r0), "valid_seg": sum(x["valid_seg"] for x in r0)}
    dump(dest / "summary.json", summary)
    print(json.dumps({"stage": "FINAL", "decision": decision,
                      "mean_gain": gain, "ci_low": ci_low, "tradeoff": tradeoff}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "train", "finalize"])
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.stage == "preflight": preflight()
    elif args.stage == "train":
        if not args.arm: ap.error("--arm required")
        train(args.arm)
    else: finalize()


if __name__ == "__main__": main()
