#!/usr/bin/env python3
"""Phase 6G.19: matched Structured Forensic Context, C versus S_rect."""
from __future__ import annotations

import argparse, json, os, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.csculf import ForensicContext64, build_comparison_features
from model.pcerf import evidence_to_dirichlet, ecolaf_discount
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import phase6g10_train_arm as g10
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch, load_c1_cache, phase4f_spatial_batch
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import BATCH, CLIP, EPOCHS, LR, SEED, WD, gate_stats, load_collapsed, rect_forward, rows, seed_all, write_rows
from scripts.phase4ha_utility_gated_rectification import gate_to_sam_grid, gated_embedding
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import invalid_record, mask_loss

OUT = ROOT / "outputs/phase6g19_r1_c_vs_srect"
DOC = ROOT / "docs/phase6g19_r1_c_vs_srect.md"
ARMS = ("A1_C", "A2_Srect")


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


class StructuredForensicContext64(nn.Module):
    """Evidence and proposal are encoded separately, then merged before CMX."""
    def __init__(self):
        super().__init__()
        self.evidence = ForensicContext64(64)
        self.x = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())

    def forward(self, f64, z_f64, support64, x64):
        support = support64.detach().float()
        evidence = self.evidence(f64, z_f64, support)
        proposal = self.x(x64.detach().float()) * support
        return self.merge(torch.cat((evidence, proposal), 1)) * support


def init_model(device):
    seed_all()
    model, _ = hc.load_utility("a2", device)
    model.forensic_context = StructuredForensicContext64().to(device)
    model.language_source.eval().requires_grad_(False)
    model.forensic_source.eval().requires_grad_(False)
    return model


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v for n, v in model.state_dict().items() if n in names}


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def structured_forward(model, batch, x):
    s64, query, zl = batch["S64"], batch["q_seg"], batch["z_L"]
    f24, zf, geometries = batch["F24"], batch["z_F24"], batch["clip_geometries"]
    with torch.autocast(device_type=s64.device.type, dtype=torch.bfloat16):
        evidence_l = model.language_source(s64.detach(), query.detach(), zl.detach()) / model.temperature_l
        evidence_f = model.forensic_source(f24.detach(), zf.detach()) / model.temperature_f
        opinion_l, opinion_f = evidence_to_dirichlet(evidence_l), evidence_to_dirichlet(evidence_f)
        mass_l = opinion_l["masses"]
        mass_f, support = model._map_forensic(opinion_f["masses"], geometries, output_hw=(64, 64), vacuous=True)
        aligned_f, mapped_support = model._map_forensic(f24.detach(), geometries, output_hw=(64, 64), vacuous=False)
        aligned_z, _ = model._map_forensic(zf.detach(), geometries, output_hw=(64, 64), vacuous=False)
        if not torch.equal(support, mapped_support):
            raise RuntimeError("forensic geometry/support drift")
        p_l = opinion_l["posterior"]
        p_f = mass_f[:, :-1] + mass_f[:, -1:] / 2.0
        l64 = model.language_context(s64, query, zl)
        fctx = model.forensic_context(aligned_f, aligned_z, support, x)
        lr, fr, rectification = model.rectification(l64, fctx, support)
        lr, fr, exchange = model.exchange(lr, fr, support)
        conflict = ecolaf_discount(torch.stack((mass_l, mass_f), 2), classes=2)[1]
        comparison, parts = build_comparison_features(lr, fr, p_l, p_f, conflict, support)
        hidden = model.utility_head.net[:-1](comparison)
        logit = model.utility_head.net[-1](hidden)
        utility = logit.sigmoid() * support.float()
    return {"utility_logit": logit, "U": utility, "support": support, "p_L": p_l, "p_F": p_f,
            "Fctx64": fctx, "utility_hidden": hidden, "rectification": rectification,
            "exchange": exchange, "parts": parts}


def context(data, cache, positions, p4, sam, s64, f24, zf24):
    batch, _ = c1_language_batch(data, cache, positions, p4, sam, s64.device)
    batch["S64"] = s64; batch["F24"] = f24; batch["z_F24"] = zf24
    return batch


def choose_x(arm, adapted, correction):
    return correction if arm == "A1_C" else adapted


def train_step(model, arm, collapsed, sam, fusion, projection, head, p4, fs, data, ids, pos, cache, cross, perm, device):
    s64, _, targets, sc, cc = phase4f_spatial_batch(p4, ids, device)
    f = g10.fused_evidence(fusion, projection, fs, ids, device); z = head(f.float())
    batch = context(data, cache, pos, p4, sam, s64, f, z)
    cross_pos = cross.index_select(0, pos); cross_ids = [data["sample_ids"][i] for i in cross_pos.tolist()]
    fc = g10.fused_evidence(fusion, projection, fs, cross_ids, device); zc = head(fc.float())
    crossed = dict(batch); crossed["F24"] = fc; crossed["z_F24"] = zc
    fp = f.flatten(2)[:, :, perm].reshape_as(f); zp = z.flatten(2)[:, :, perm].reshape_as(z)
    shuffled = dict(batch); shuffled["F24"] = fp; shuffled["z_F24"] = zp

    _, srect, correction, support = rect_forward(collapsed, s64, f, sc, cc)
    _, srect_cross, correction_cross, support_cross = rect_forward(collapsed, s64, fc, sc, cc)
    _, srect_shuffle, correction_shuffle, support_shuffle = rect_forward(collapsed, s64, fp, sc, cc)
    if not torch.equal(support, support_cross) or not torch.equal(support, support_shuffle):
        raise RuntimeError("rectifier support drift across matched negatives")

    matched = structured_forward(model, batch, choose_x(arm, srect, correction))
    cross_out = structured_forward(model, crossed, choose_x(arm, srect_cross, correction_cross))
    shuffle_out = structured_forward(model, shuffled, choose_x(arm, srect_shuffle, correction_shuffle))
    gate = gate_to_sam_grid(matched["U"], sc) * support
    # Both arms always gate the actual positive correction through srect.
    final = gated_embedding(s64, srect, gate)
    with torch.autocast(device_type=device.type, enabled=False):
        low = sam(batch["q_seg"], final.to(torch.bfloat16))
    seg = mask_loss(low, targets, hd.CFG)
    _, soft = q.target_delta(matched, data["target64"].index_select(0, pos).to(device))
    relative = q.image_balanced_loss(matched["utility_logit"], soft, matched["support"])
    ranking = .5 * (hc.rank_loss(matched["U"], cross_out["U"], matched["support"]) +
                      hc.rank_loss(matched["U"], shuffle_out["U"], matched["support"]))
    return seg["total"] + relative + ranking, {"seg": seg["total"], "relative": relative, "ranking": ranking}


def evaluate(model, arm, collapsed, sam, fusion, projection, head, p4, fs, dev, cache, device):
    model.eval(); records = []; pixels = []; per_image = []
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                records.append(invalid_record(sid, dev["original_masks"][i])); per_image.append(0.0); continue
            s64, _, _, sc, cc = phase4f_spatial_batch(p4, [sid], device)
            f = g10.fused_evidence(fusion, projection, fs, [sid], device); z = head(f.float())
            batch = context(dev, cache, torch.tensor([i]), p4, sam, s64, f, z)
            _, srect, correction, support = rect_forward(collapsed, s64, f, sc, cc)
            out = structured_forward(model, batch, choose_x(arm, srect, correction))
            gate = gate_to_sam_grid(out["U"], sc) * support
            final = gated_embedding(s64, srect, gate)
            with torch.autocast(device_type=device.type, enabled=False):
                low = sam(batch["q_seg"], final.to(torch.bfloat16))
            records.append(metric_record(sid, inverse_sam_logits(low, dev["sam_geometries"][i]), dev["original_masks"][i]))
            values = gate[support.bool()].float().cpu().numpy(); pixels.extend(values.tolist()); per_image.append(float(values.mean()))
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "VAL", "arm": arm, "done": i + 1}), flush=True)
    return summarize_extended(records), records, {"pixels": gate_stats(pixels), "per_image": gate_stats(per_image)}


def pair(a1, a2):
    if [r["sample_id"] for r in a1] != [r["sample_id"] for r in a2]:
        raise RuntimeError("paired sample order drift")
    x = torch.tensor([r["foreground_iou"] for r in a1]); y = torch.tensor([r["foreground_iou"] for r in a2])
    out = paired_statistics(x, y); delta = (x-y).numpy()
    out["wins_ties_losses"] = [int((delta > 1e-12).sum()), int((abs(delta) <= 1e-12).sum()), int((delta < -1e-12).sum())]
    out["wilcoxon_pvalue"] = float(stats.wilcoxon(delta).pvalue) if np.any(delta) else 1.0
    f1x = torch.tensor([r["foreground_f1"] for r in a1]); f1y = torch.tensor([r["foreground_f1"] for r in a2])
    out["f1"] = paired_statistics(f1x, f1y)
    return out


def train(arm, device):
    root = OUT / arm
    if (root / "summary.json").exists():
        print(json.dumps({"arm": arm, "status": "ALREADY_COMPLETE"}), flush=True); return
    model = init_model(device); _, collapsed, _ = load_collapsed(device)
    fusion = g10.load_fusion(device); projection = g10.load_projection(device); head = load_head(device)
    sam = g10.load_sam_runtime(hd.CFG, device)
    p4t, p4v = g10.Phase4FStore(hd.CFG, "train"), g10.Phase4FStore(hd.CFG, "val")
    fst, fsv = g10.Store("train"), g10.Store("val"); dev = g10.load_dev("g0")
    ids = p4t.sample_ids; tc = load_c1_cache("train", ids); vc = load_c1_cache("val", dev["sample_ids"])
    data = q.load_ids(ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries")); data["sample_ids"] = ids
    valid = tc["valid"].bool(); where = {sid: i for i, sid in enumerate(ids)}
    cross = torch.tensor([(i+1) % len(ids) for i in range(len(ids))]); perm = torch.randperm(576, generator=torch.Generator().manual_seed(SEED)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]; optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    frozen = {"collapsed": tensor_state_sha256(collapsed.state_dict()), "sam": tensor_state_sha256(sam.state_dict()),
              "fusion": tensor_state_sha256(fusion.state_dict()), "projection": tensor_state_sha256(projection.state_dict()),
              "head": tensor_state_sha256(head.state_dict()), "source_heads": tensor_state_sha256(q.source_state(model))}
    init_hash = tensor_state_sha256(trainable_state(model)); root.mkdir(parents=True, exist_ok=True)
    dump(root / "protocol.json", {"status": "FROZEN_BEFORE_FIRST_STEP", "arm": arm, "X": "C" if arm == "A1_C" else "S_rect",
          "final_injection": "S64 + U_F * C for both arms", "trainable_parameters": count_trainable(model),
          "initial_trainable_sha256": init_hash, "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD,
          "seed": SEED, "loss": "seg+relative+ranking", "selector": "internal validation G0 mean FG IoU; tie earlier"},
          "firewall": {"test": False, "official1000": False, "ood": False, "new_objective": False}})
    history = []; updates = 0; start = 1
    checkpoints = sorted((root / "checkpoints").glob("epoch_*.pt")) if (root / "checkpoints").exists() else []
    if checkpoints:
        last = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1])); state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start = state["epoch"] + 1
        updates = state["updates"]; history = json.loads((root / "training_curve.json").read_text())
    for epoch in range(start, EPOCHS + 1):
        began = time.time(); model.train(); model.language_source.eval(); model.forensic_source.eval()
        order = list(ids); random.Random(SEED + 1009 * epoch).shuffle(order); sums = defaultdict(float); n = 0
        for begin in range(0, len(order), BATCH):
            positions = torch.tensor([where[s] for s in order[begin:begin+BATCH]])
            positions = positions[valid.index_select(0, positions)]
            if not len(positions): continue
            batch_ids = [ids[i] for i in positions.tolist()]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = train_step(model, arm, collapsed, sam, fusion, projection, head, p4t, fst, data, batch_ids, positions, tc, cross, perm, device)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, CLIP); optimizer.step(); updates += 1; n += len(batch_ids)
            for key, value in parts.items(): sums[key] += float(value.detach()) * len(batch_ids)
        metrics, records, gates = evaluate(model, arm, collapsed, sam, fusion, projection, head, p4v, fsv, dev, vc, device)
        row = {"epoch": epoch, "updates": updates, "seg_loss": sums["seg"]/n, "relative_loss": sums["relative"]/n,
               "ranking_loss": sums["ranking"]/n, "total_loss": sum(sums.values())/n,
               "dev_g0_mean_iou": metrics["mean_foreground_iou"], "dev_g0_mean_f1": metrics["mean_foreground_f1"],
               "seconds": time.time()-began}
        history.append(row); dump(root / "training_curve.json", history); write_rows(root / f"validation/epoch_{epoch}.jsonl", records)
        dump(root / f"gate/epoch_{epoch}.json", gates); (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "updates": updates, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, root / f"checkpoints/epoch_{epoch}.pt")
        print(json.dumps({"stage": "G19_EPOCH", "arm": arm, **row}), flush=True)
    best = max(history, key=lambda x: (x["dev_g0_mean_iou"], -x["epoch"]))
    selected = root / "selected_checkpoint.pt"; shutil.copy2(root / f"checkpoints/epoch_{best['epoch']}.pt", selected)
    model.load_state_dict(torch.load(selected, map_location="cpu", weights_only=False)["model"])
    metrics, records, gates = evaluate(model, arm, collapsed, sam, fusion, projection, head, p4v, fsv, dev, vc, device)
    write_rows(root / "validation/selected.jsonl", records)
    after = {"collapsed": tensor_state_sha256(collapsed.state_dict()), "sam": tensor_state_sha256(sam.state_dict()),
             "fusion": tensor_state_sha256(fusion.state_dict()), "projection": tensor_state_sha256(projection.state_dict()),
             "head": tensor_state_sha256(head.state_dict()), "source_heads": tensor_state_sha256(q.source_state(model))}
    if after != frozen: raise RuntimeError("frozen parameter hash drift")
    dump(root / "summary.json", {"status": "COMPLETE", "arm": arm, "selected_epoch": best["epoch"],
          "selected_checkpoint_sha256": file_sha256(selected), "metrics": metrics, "gate_stats": gates,
          "trainable_parameters": count_trainable(model), "initial_trainable_sha256": init_hash})


def finalize():
    s1 = json.loads((OUT / "A1_C/summary.json").read_text()); s2 = json.loads((OUT / "A2_Srect/summary.json").read_text())
    if s1["trainable_parameters"] != s2["trainable_parameters"] or s1["initial_trainable_sha256"] != s2["initial_trainable_sha256"]:
        raise RuntimeError("matched capacity/initialization contract failed")
    r1 = rows(OUT / "A1_C/validation/selected.jsonl"); r2 = rows(OUT / "A2_Srect/validation/selected.jsonl")
    paired = pair(r1, r2); lo, hi = paired["bootstrap_95_ci"]
    decision = "UPDATE_PROPOSAL_C_PREFERRED" if paired["mean_difference"] > 0 and lo > 0 else (
        "RECTIFIED_STATE_SRECT_PREFERRED" if paired["mean_difference"] < 0 and hi < 0 else "C_AND_SRECT_NO_STABLE_DIFFERENCE")
    result = {"status": "COMPLETE_STOP", "decision": decision, "primary_comparison": "A1_C minus A2_Srect",
              "A1_C": s1, "A2_Srect": s2, "paired": paired,
              "firewall": {"test": False, "official1000": False, "ood": False}}
    dump(OUT / "results.json", result)
    DOC.write_text("\n".join(["# Phase 6G.19 — R1 Structured Forensic Context: C vs Srect", "",
        "Status: **COMPLETE STOP**.", "", "## Matched contract", "",
        f"Trainable parameters per arm: `{s1['trainable_parameters']}`. Initial trainable hash exact match: `{s1['initial_trainable_sha256'] == s2['initial_trainable_sha256']}`.",
        "Both arms use the same frozen C1, CLIP block11+17 fusion, projection, collapsed Translator/Rectifier, gamma, support, SAM and evidential heads. Final injection is always `S64 + U_F*C`.",
        "", "## Results", "", f"A1 C: `{json.dumps(s1, ensure_ascii=False)}`", "", f"A2 Srect: `{json.dumps(s2, ensure_ascii=False)}`",
        "", "## Paired A1-A2", "", f"`{json.dumps(paired, ensure_ascii=False)}`", "", "## Decision", "", f"```text\n{decision}\n```",
        "", "No zero-X arm, late logit residual, modulation, new objective, test, Official1000, or OOD was used."]) + "\n")
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=ARMS); parser.add_argument("--device", default="cuda:1"); parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize: finalize(); return
    if not args.arm: parser.error("--arm is required unless --finalize")
    device = torch.device(args.device); torch.cuda.set_device(device); train(args.arm, device)


if __name__ == "__main__":
    main()
