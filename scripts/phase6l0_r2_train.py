#!/usr/bin/env python3
"""Single-arm, C2-native R2 preflight and 10-epoch internal-DEV training."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.c2_r2_localizer import C2R2Localizer
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6l0_r2_preflight import EXPECTED_SHA, OUT, load_native_runtime, c2_native_sam_state, dump, require
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import summarize, tensor_state_sha256, sam_coordinates, clip_coordinates
from tools.phase4f import Phase4FStore, deterministic_order, invalid_record, mask_loss

CFG_PATH = ROOT / "configs/phase6l0_c2_r2.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
P4F_CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
CKPTS = Path(CFG["experiment"]["checkpoint_root"])
SEED, BATCH, EPOCHS = 3407, 8, 10


def load_cache(split: str, expected: list[str]) -> dict:
    manifest = json.loads((OUT / f"{split}.json").read_text())
    require(manifest["status"] == "COMPLETE" and manifest["n"] == len(expected) and
            manifest["c2_sha256"] == EXPECTED_SHA, f"{split} R2 cache incomplete or C2 drift")
    import hashlib
    require(manifest["ids_sha256"] == hashlib.sha256("\n".join(expected).encode()).hexdigest(),
            f"{split} R2 ordered IDs drift")
    for source in ("sam", "clip"):
        reference = manifest["spatial_references"][source]
        require(file_sha256(Path(reference["complete_path"])) == reference["complete_sha256"],
                f"{split} {source} spatial manifest SHA drift")
        for spec in reference["shards"]:
            require(file_sha256(Path(spec["path"])) == spec["sha256"],
                    f"{split} {source} spatial shard SHA drift")
    values, ids = [], []
    for spec in manifest["shards"]:
        path = Path(spec["path"])
        require(file_sha256(path) == spec["sha256"], f"{split} R2 shard SHA drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["schema"] == "phase6l0_r2_cache_shard_v1" and payload["c2_sha256"] == EXPECTED_SHA and
                payload["sam_runtime_sha256"] == manifest["sam_runtime_sha256"], "R2 shard provenance drift")
        ids.extend(payload["sample_ids"])
        values.append(payload)
    require(ids == expected, f"{split} R2 shard order drift")
    keys = ("q_seg", "A", "E", "r_prime", "z_L", "valid_c2_g0", "seg_count")
    cache = {"sample_ids": ids, **{key: torch.cat([v[key] for v in values]) for key in keys}}
    n = len(ids)
    require(cache["q_seg"].shape == (n, 256) and cache["A"].shape == (n, 8, 24, 24) and
            cache["E"].shape == (n, 512, 24, 24) and cache["r_prime"].shape == (n, 4096) and
            cache["z_L"].shape == (n, 1, 256, 256), "R2 tensor contract drift")
    valid = cache["valid_c2_g0"]
    require(torch.equal(valid, cache["seg_count"].eq(1)) and int(valid.sum()) == manifest["valid_c2_g0"] and
            bool(torch.isfinite(cache["q_seg"][valid]).all() and torch.isnan(cache["q_seg"][~valid]).all() and
                 torch.isfinite(cache["A"]).all() and torch.isfinite(cache["E"]).all() and
                 torch.isfinite(cache["r_prime"]).all() and torch.isfinite(cache["z_L"]).all()),
            "R2 validity/finite contract failed")
    require(float((cache["A"].flatten(2).sum(-1)-1).abs().max()) < 1e-5, "A head mass drift")
    return cache


def spatial_batch(store: Phase4FStore, cache: dict, indices: list[int], device: torch.device) -> dict:
    ids = [cache["sample_ids"][i] for i in indices]
    source = [store._values(sid) for sid in ids]  # deliberately never read store.q (P1)
    s64 = torch.stack([x[0] for x in source]).to(device=device, dtype=torch.bfloat16)
    raw = torch.stack([x[1] for x in source]).to(device=device, dtype=torch.bfloat16)
    target = torch.stack([x[2] for x in source]).to(device=device)
    index = torch.tensor(indices, dtype=torch.long)
    return {"sample_ids": ids, "s64": s64, "raw_clip_grid": raw, "target": target,
            "attention_map": cache["A"].index_select(0, index).to(device=device),
            "evidence_map": cache["E"].index_select(0, index).to(device=device),
            "r_prime": cache["r_prime"].index_select(0, index).to(device=device),
            "q_seg": cache["q_seg"].index_select(0, index).to(device=device),
            "z_l": cache["z_L"].index_select(0, index).to(device=device),
            "clip_geometry": [geometry_for("clip", store.geometries[sid]["original_hw"]) for sid in ids]}


def forward(model: C2R2Localizer, sam, batch: dict, *, loss: bool = True):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(raw_clip_grid=batch["raw_clip_grid"], attention_map=batch["attention_map"],
                       evidence_map=batch["evidence_map"], r_prime=batch["r_prime"],
                       q_seg=batch["q_seg"], s64=batch["s64"], z_l=batch["z_l"],
                       clip_geometry=batch["clip_geometry"])
    # Frozen SAM is intentionally OUTSIDE no_grad: its operations carry the
    # derivative from the final mask loss back into S_adapt and R2.
    with torch.autocast(device_type="cuda", enabled=False):
        low = sam(batch["q_seg"].to(torch.bfloat16), output["S_adapt"].to(torch.bfloat16))
    losses = mask_loss(low, batch["target"], P4F_CFG) if loss else None
    return output, low, losses


def grad_norm(module) -> float:
    squares = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def gradient_modules(model):
    return {"EvidenceConsolidator": model.evidence, "GeometryAwareBridge": model.bridge,
            "GlobalConditioner": model.conditioner,
            **{f"ResidualBlock{i+1}": block for i, block in enumerate(model.adapter.blocks)},
            "W_out": model.adapter.output}


def preflight(model, sam, store, cache, device, sam_hash, c2_sha):
    require(c2_sha == EXPECTED_SHA and file_sha256(ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt") == c2_sha,
            "G1 C2 checkpoint drift")
    subset = json.loads((OUT / "preflight_subset.json").read_text())
    classification_parity = json.loads((ROOT / "outputs/phase6k0_c2_after/cache/full_capture_parity.json").read_text())
    require(subset["status"] == "PASS" and subset["capture_exact"] and subset.get("classification_logits_exact") and
            subset["R2_step0_SAM_exact"] and
            subset["sam_state_sha256"] == sam_hash and classification_parity["status"] == "PASS" and
            classification_parity["c2_sha256"] == c2_sha and
            all(classification_parity[key] for key in ("fused_token_exact", "classification_logits_exact",
                                                        "q_seg_exact", "mask_exact", "generated_token_ids_exact")),
            "G2/G3/G5 subset and historical classification parity incomplete")
    require(all(not p.requires_grad for p in sam.parameters()), "G7 SAM parameter requires_grad drift")
    valid_idx = int(cache["valid_c2_g0"].nonzero()[0])
    batch = spatial_batch(store, cache, [valid_idx], device)
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    output, low, losses = forward(model, sam, batch)
    require(torch.count_nonzero(output["deltaS"]) == 0 and torch.equal(output["S_adapt"], batch["s64"]),
            "G5 R2 step0 identity failed")
    require(torch.equal(low.detach().cpu(), batch["z_l"].cpu()), "G5 R2 step0 versus cached C2-G0 low logits failed")
    require(torch.isfinite(losses["total"]), "G8 first loss nonfinite")
    losses["total"].backward()
    first = grad_norm(model.adapter.output)
    require(first > 0 and all(p.grad is None for p in sam.parameters()), "G6 first backward firewall failed")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output, low, losses = forward(model, sam, batch)
    require(torch.isfinite(losses["total"]) and all(torch.isfinite(v).all() for v in output.values()),
            "G8 second forward nonfinite")
    losses["total"].backward()
    gradients = {key: grad_norm(module) for key, module in gradient_modules(model).items()}
    require(all(math.isfinite(value) and value > 0 for value in gradients.values()) and
            all(p.grad is None for p in sam.parameters()), f"G6 second backward gradient failure: {gradients}")
    require(tensor_state_sha256(sam.state_dict()) == sam_hash, "G7 SAM state changed")
    model.load_state_dict(original, strict=True)
    model.zero_grad(set_to_none=True)
    require(torch.count_nonzero(model.adapter.output.weight) == 0 and
            torch.count_nonzero(model.adapter.output.bias) == 0, "R2 fresh zero head restore failed")
    dump(OUT.parent / "preflight_gates.json", {
        "status": "PASS", "G1_C2_CHECKPOINT_IDENTITY": "PASS", "G2_C2_CAPTURE_PARITY": "PASS",
        "G3_A_E_CONTEXT_RECONSTRUCTION": "PASS", "G4_CACHE_ID_ORDER_GEOMETRY": "PASS",
        "G5_R2_STEP0_C2_G0_IDENTITY": "PASS", "G6_GRADIENT_ROUTING": "PASS",
        "G7_FROZEN_C2_SAM_INTEGRITY": "PASS", "G8_FINITE_FORWARD_BACKWARD": "PASS",
        "first_W_out_grad_norm": first, "second_gradient_norms": gradients,
        "sam_state_sha256": sam_hash, "c2_checkpoint_sha256": c2_sha,
        "r2_model_source_sha256": file_sha256(ROOT / "model/c2_r2_localizer.py"),
        "training_config_sha256": file_sha256(CFG_PATH),
        "train_valid_c2_g0": int(cache["valid_c2_g0"].sum()),
        "R2_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})


def _record(sid, low, store):
    target = store.original_masks[sid]
    row = metric_record(sid, inverse_sam_logits(low, store.geometries[sid]), target)
    row["tn"] = int(target.numel()) - row["tp"] - row["fp"] - row["fn"]
    row["valid_c2_g0"] = True
    return row


def _invalid(sid, store):
    target = store.original_masks[sid]
    row = invalid_record(sid, target)
    row["tn"] = int(target.numel()) - row["fn"]
    row["valid_c2_g0"] = False
    return row


def _diagnostic(output, s64):
    delta, adapted = output["deltaS"].float(), output["S_adapt"].float()
    support = output["support64"].expand_as(delta)
    inside = delta.abs().masked_select(support)
    outside = delta.abs().masked_select(~support)
    offsets = output["bridge_offsets"].float().square().sum(dim=3).sqrt()
    attn = output["bridge_attention"].float()
    valid = output["bridge_valid"]
    entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=2)
    return {"delta_ratio": float(delta.norm() / s64.float().norm().clamp_min(1e-12)),
            "cosine_S64_Sadapt": float(torch.nn.functional.cosine_similarity(s64.float().flatten(1), adapted.flatten(1)).mean()),
            "inside_support_mean_abs_delta": float(inside.mean()) if inside.numel() else 0.,
            "outside_support_mean_abs_delta": float(outside.mean()) if outside.numel() else 0.,
            "offset_mean": float(offsets.mean()), "offset_median": float(offsets.median()),
            "offset_p95": float(torch.quantile(offsets.flatten(), .95)), "offset_max": float(offsets.max()),
            "offset_boundary_fraction": float((output["bridge_offsets"].float().abs() >= 3.99).float().mean()),
            "attention_entropy": float(entropy.mean()), "attention_max_mass": float(attn.max(dim=2).values.mean()),
            "invalid_sample_fraction": float((~valid).float().mean()),
            "all_invalid_query_fraction": float((~valid.any(dim=2)).float().mean())}


def summarize_diagnostics(rows):
    return {key: {"mean": float(np.mean([r[key] for r in rows])),
                  "median": float(np.median([r[key] for r in rows])),
                  "p95": float(np.quantile([r[key] for r in rows], .95)),
                  "max": float(np.max([r[key] for r in rows]))}
            for key in rows[0]} if rows else {}


def evaluate(model, sam, store, cache, device, epoch):
    model.eval()
    records, diagnostics = [], []
    with torch.no_grad():
        for i, sid in enumerate(cache["sample_ids"]):
            if not bool(cache["valid_c2_g0"][i]):
                records.append(_invalid(sid, store))
                continue
            batch = spatial_batch(store, cache, [i], device)
            if epoch == 0:
                with torch.autocast(device_type="cuda", enabled=False):
                    low = sam(batch["q_seg"].to(torch.bfloat16), batch["s64"].to(torch.bfloat16))
                require(torch.equal(low.cpu(), batch["z_l"].cpu()), "epoch0 C2-G0 cached z_L parity failed")
            else:
                output, low, _ = forward(model, sam, batch, loss=False)
                diagnostics.append(_diagnostic(output, batch["s64"]))
            records.append(_record(sid, low, store))
            if (i+1) % 100 == 0:
                print(json.dumps({"stage": "DEV", "epoch": epoch, "done": i+1, "total": len(cache["sample_ids"])}), flush=True)
    summary = summarize(records)
    summary.update({"tp": sum(r["tp"] for r in records), "fp": sum(r["fp"] for r in records),
                    "fn": sum(r["fn"] for r in records), "tn": sum(r["tn"] for r in records),
                    "valid_c2_g0": int(cache["valid_c2_g0"].sum()),
                    "invalid_c2_g0": int((~cache["valid_c2_g0"]).sum())})
    dump(OUT.parent / f"dev_epoch{epoch:02d}_summary.json", summary)
    with (OUT.parent / f"dev_epoch{epoch:02d}_rows.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    if diagnostics:
        dump(OUT.parent / f"diagnostics_epoch{epoch:02d}.json", summarize_diagnostics(diagnostics))
    return summary


def main(device):
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_expected, dev_data = train_ids(), load_dev("g0")
    dev_expected = dev_data["sample_ids"]
    train_cache = load_cache("train", train_expected)
    dev_cache = load_cache("val", dev_expected)
    train_store, dev_store = Phase4FStore(P4F_CFG, "train"), Phase4FStore(P4F_CFG, "val")
    require(train_store.sample_ids == train_expected and dev_store.sample_ids == dev_expected,
            "canonical spatial population drift")
    for store in (train_store, dev_store):
        for sid in store.sample_ids:
            sam_geo = store.geometries[sid]
            require(sam_geo == geometry_for("sam", sam_geo["original_hw"]) and
                    torch.equal(store.sam_coords[sid], sam_coordinates(sam_geo, grid=64)) and
                    torch.equal(store.clip_coords[sid], clip_coordinates(
                        geometry_for("clip", sam_geo["original_hw"]), grid=24)),
                    f"SAM/CLIP geometry or support mapping drift: {sid}")
    state, _ = c2_native_sam_state()
    sam = load_native_runtime(state, device)
    sam_hash = tensor_state_sha256(sam.state_dict())
    require(sam_hash == json.loads((OUT / "preflight_subset.json").read_text())["sam_state_sha256"],
            "C2-native SAM state drift")
    model = C2R2Localizer().to(device)
    require(all(p.requires_grad for p in model.parameters()), "R2 trainable scope drift")
    preflight(model, sam, train_store, train_cache, device, sam_hash, EXPECTED_SHA)
    baseline = evaluate(model, sam, dev_store, dev_cache, device, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    CKPTS.mkdir(parents=True, exist_ok=True)
    history, best_epoch, best_iou, total_updates = [], None, -math.inf, 0
    id_to_index = {sid: i for i, sid in enumerate(train_expected)}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ordered = deterministic_order(train_expected, epoch, seed=SEED)
        epoch_losses, epoch_grads, clips = [], {name: [] for name in gradient_modules(model)}, 0
        eligible = 0
        for start in range(0, len(ordered), BATCH):
            indices = [id_to_index[sid] for sid in ordered[start:start+BATCH]]
            indices = [i for i in indices if bool(train_cache["valid_c2_g0"][i])]
            if not indices:
                continue
            eligible += len(indices)
            batch = spatial_batch(train_store, train_cache, indices, device)
            optimizer.zero_grad(set_to_none=True)
            output, _low, losses = forward(model, sam, batch)
            require(torch.isfinite(losses["total"]) and all(torch.isfinite(v).all() for v in output.values()),
                    f"epoch {epoch} step {start//BATCH} nonfinite forward")
            losses["total"].backward()
            norms = {key: grad_norm(module) for key, module in gradient_modules(model).items()}
            require(all(math.isfinite(value) for value in norms.values()) and
                    all(p.grad is None for p in sam.parameters()), "frozen SAM gradient or nonfinite R2 gradient")
            for key, value in norms.items():
                epoch_grads[key].append(value)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require(torch.isfinite(total_norm), "nonfinite clipped gradient")
            clips += float(total_norm > 1.0)
            optimizer.step()
            epoch_losses.append(float(losses["total"].detach()))
            total_updates += 1
            if total_updates % 100 == 0:
                print(json.dumps({"stage": "TRAIN", "epoch": epoch, "updates": total_updates,
                                  "batch_loss": epoch_losses[-1]}), flush=True)
        require(eligible == int(train_cache["valid_c2_g0"].sum()), "TRAIN valid eligibility traversal drift")
        require(all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite R2 parameter")
        require(tensor_state_sha256(sam.state_dict()) == sam_hash and
                file_sha256(ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt") == EXPECTED_SHA,
                "G7 frozen C2/SAM state integrity failure")
        metrics = evaluate(model, sam, dev_store, dev_cache, device, epoch)
        epoch_path = CKPTS / f"epoch_{epoch:02d}.pt"
        temporary = epoch_path.with_suffix(".pt.tmp")
        torch.save({"schema": "phase6l0_c2_r2_epoch_v1", "epoch": epoch,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(), "updates": total_updates,
                    "c2_sha256": EXPECTED_SHA, "sam_state_sha256": sam_hash,
                    "r2_model_source_sha256": file_sha256(ROOT / "model/c2_r2_localizer.py"),
                    "training_config_sha256": file_sha256(CFG_PATH),
                    "train_cache_sha256": file_sha256(OUT / "train.json"),
                    "dev_cache_sha256": file_sha256(OUT / "val.json"),
                    "dev_metrics": metrics}, temporary)
        os.replace(temporary, epoch_path)
        row = {"epoch": epoch, "updates": total_updates, "eligible_train": eligible,
               "train_loss": float(np.mean(epoch_losses)), "dev_mean_fg_iou": metrics["mean_foreground_iou"],
               "dev_mean_fg_f1": metrics["mean_foreground_f1"],
               "dev_global_fg_iou": metrics["global_foreground_iou"],
               "dev_global_fg_f1": metrics["global_foreground_f1"],
               "gradient_clip_frequency": clips/len(epoch_losses),
               **{f"grad_{key}": float(np.mean(values)) for key, values in epoch_grads.items()},
               "checkpoint": str(epoch_path), "checkpoint_sha256": file_sha256(epoch_path)}
        history.append(row)
        with (OUT.parent / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerows(history)
        dump(OUT.parent / "training_status.json", {"status": "RUNNING", "epoch": epoch,
                                                    "optimizer_updates": total_updates,
                                                    "c2_sha256": EXPECTED_SHA, "sam_state_sha256": sam_hash})
        if metrics["mean_foreground_iou"] > best_iou:
            best_iou, best_epoch = metrics["mean_foreground_iou"], epoch
        print(json.dumps({"stage": "EPOCH_COMPLETE", "epoch": epoch, "dev_mean_fg_iou": best_iou,
                          "best_epoch": best_epoch}), flush=True)
    selected = CKPTS / f"epoch_{best_epoch:02d}.pt"
    destination = OUT.parent / "selected_checkpoint.pt"
    shutil.copy2(selected, destination)
    dump(OUT.parent / "selector.json", {"status": "COMPLETE", "selected_epoch": best_epoch,
                                         "selector": "internal_DEV_mean_FG_IoU_earliest_tie",
                                         "selected_dev_mean_fg_iou": best_iou,
                                         "selected_checkpoint": str(destination),
                                         "selected_checkpoint_sha256": file_sha256(destination),
                                         "epoch0": baseline, "optimizer_updates": total_updates,
                                         "c2_sha256": EXPECTED_SHA, "sam_state_sha256": sam_hash})
    dump(OUT.parent / "training_status.json", {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV",
                                                "selected_epoch": best_epoch,
                                                "optimizer_updates": total_updates})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(torch.device(args.device))
