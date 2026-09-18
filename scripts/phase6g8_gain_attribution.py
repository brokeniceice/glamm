#!/usr/bin/env python3
"""Phase 6G.8 frozen evidence-to-correction gain attribution."""
from __future__ import annotations

import hashlib, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, c1_language_batch, phase4f_spatial_batch
from scripts.phase6g0_multilevel_dense_clip import cache_paths, load_shard
from scripts.phase6g7_staged_r1 import Store, sources
from tools.phase3c1 import binary_metrics, inverse_logits, paired_statistics, probe_loss, summarize
from tools.phase4c_b import file_sha256, inverse_sam_logits
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, load_evidence_source, load_rectifier, load_sam_runtime

OUT = ROOT / "outputs/phase6g8_gain_attribution"
DOC = ROOT / "docs/phase6g8_evidence_to_correction_gain_attribution.md"
A0_CKPT = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
A1_CKPT = ROOT / "outputs/phase6g7_block11_17_staged_r1/joint/selected_checkpoint.pt"
SEED, EPOCHS, BATCH = 3407, 20, 8
STAGES = ("F_forensic", "Delta_F", "U_Delta_F", "S_adapt")
ARMS = ("A0", "A1")


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values))


def sha_ids(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def load_models(device):
    # A0 is the selected C1-specific current block22 new R1.
    a0_source = load_evidence_source(hd.CFG, "forensic_rect", device)
    a0_u, _ = hc.load_utility("a2", device)
    scale = json.load(open(ROOT / "outputs/phase4f_language_preserving_rectification/preflight/geometry_scale_audit.json"))
    a0_r = load_rectifier(hd.CFG, float(scale["selected_gamma"]), device)
    x0 = torch.load(A0_CKPT, map_location="cpu", weights_only=False)
    a0_u.load_state_dict(x0["utility_state"]); a0_r.load_state_dict(x0["rectifier_state"])

    # A1 uses the selected block11+17 fusion/adapter and staged new R1.
    fusion, a1_source, _, _ = sources(device)
    a1_u, _ = hc.load_utility("a2", device)
    a1_r = load_rectifier(hd.CFG, float(scale["selected_gamma"]), device)
    x1 = torch.load(A1_CKPT, map_location="cpu", weights_only=False)
    a1_u.load_state_dict(x1["utility_state"]); a1_r.load_state_dict(x1["rectifier_state"])
    models = {"A0": (a0_u, a0_r, a0_source), "A1": (a1_u, a1_r, a1_source)}
    for u, r, source in models.values():
        u.eval().requires_grad_(False); u.language_source.eval(); u.forensic_source.eval()
        r.eval().requires_grad_(False); source.eval().requires_grad_(False)
    fusion.eval().requires_grad_(False)
    return models, fusion, x0, x1


def a0_evidence(source, raw):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = source(raw, return_features=True)
    return value["F_forensic"], value["logits"]


def a1_evidence(fusion, source, store, ids, device):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = source(fusion(*store.batch(ids, device)), return_features=True)
    return value["F_forensic"], value["logits"]


def representations(models, fusion, fusion_store, spatial_store, data, c1cache, indices, sam, device):
    ids = [c1cache["sample_ids"][i] for i in indices.tolist()]
    s64, raw, targets, sc, cc = phase4f_spatial_batch(spatial_store, ids, device)
    batch, _ = c1_language_batch(data, c1cache, indices, spatial_store, sam, device)
    result = {}
    for arm in ARMS:
        u, r, source = models[arm]
        f24, z24 = a0_evidence(source, raw) if arm == "A0" else a1_evidence(fusion, source, fusion_store, ids, device)
        ubatch = dict(batch); ubatch["F24"] = f24; ubatch["z_F24"] = z24
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rv = r(s64, f24, sc, cc, torch.ones(len(ids), 576, dtype=torch.bool, device=device))
            uv = hc.utility_forward(u, ubatch)
        delta = rv["image_embeddings"].float() - s64.float()
        gate = hc.gate_to_sam_grid(uv["U"], sc) * rv["support"].reshape(len(ids), 1, 64, 64).float()
        injected = gate.float() * delta
        sadapt = s64.float() + injected
        result[arm] = {"F_forensic": f24.float(), "Delta_F": delta,
                       "U_Delta_F": injected, "S_adapt": sadapt, "U_F": gate.float(),
                       "support": rv["support"].reshape(len(ids), 1, 64, 64)}
    return result, targets


def probe_set(device):
    probes = nn.ModuleDict({f"{a}__{s}": nn.Conv2d(256, 1, 1) for a in ARMS for s in STAGES}).to(device)
    return probes, torch.optim.AdamW(probes.parameters(), lr=1e-3, weight_decay=0.)


def train_target(stage, clip_target, sam_target):
    return clip_target[:, None].float() if stage == "F_forensic" else F.interpolate(
        sam_target[:, None].float(), (64, 64), mode="nearest")


def predict(probes, models, fusion, fusion_store, spatial_store, data, cache, sam, dev, device,
            save_maps=False):
    n_total = len(dev["sample_ids"])
    records = {f"{a}__{s}": [None] * n_total for a in ARMS for s in STAGES}
    utility = {a: [None] * n_total for a in ARMS}; maps = []
    clip_by_id = {}
    for p in cache_paths("late", "val"):
        x = load_shard(p)
        for i, row in enumerate(x["records"]):
            clip_by_id[row["sample_id"]] = (row["geometry"], x["original_masks"][i])
    with torch.no_grad():
        for begin in range(0, len(dev["sample_ids"]), BATCH):
            raw_ix = torch.arange(begin, min(begin + BATCH, len(dev["sample_ids"])))
            ix = raw_ix[cache["valid"].index_select(0, raw_ix).bool()]
            for absolute in raw_ix.tolist():
                if bool(cache["valid"][absolute]): continue
                sid = dev["sample_ids"][absolute]; truth = dev["original_masks"][absolute].bool()
                total_pixels = int(truth.numel()); fn = int(truth.sum()); tn = total_pixels - fn
                bg_iou = tn / max(1, tn + fn)
                invalid = {"sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
                           "background_iou": float(bg_iou), "fg_bg_miou": float(bg_iou / 2.0),
                           "tp": 0, "fp": 0, "fn": fn, "tn": tn, "valid_g0": False}
                for key in records: records[key][absolute] = dict(invalid)
                for arm in ARMS: utility[arm][absolute] = {"sample_id": sid, "valid_g0": False}
            if not len(ix): continue
            ids = [dev["sample_ids"][i] for i in ix.tolist()]
            rep, _ = representations(models, fusion, fusion_store, spatial_store, data, cache, ix, sam, device)
            batch_maps = {a: {} for a in ARMS}
            for arm in ARMS:
                gate = rep[arm]["U_F"]
                for j, (sid, absolute) in enumerate(zip(ids, ix.tolist())):
                    g = gate[j, 0][rep[arm]["support"][j, 0]].float()
                    utility[arm][absolute] = {"sample_id": sid, "valid_g0": True, "mean": float(g.mean()),
                                              "median": float(g.median()), "p5": float(torch.quantile(g, .05)),
                                              "p95": float(torch.quantile(g, .95)), "max": float(g.max())}
                for stage in STAGES:
                    logits = probes[f"{arm}__{stage}"](rep[arm][stage]).float()
                    batch_maps[arm][stage] = logits.detach().cpu()
                    for j, (sid, absolute) in enumerate(zip(ids, ix.tolist())):
                        if stage == "F_forensic":
                            geom, truth = clip_by_id[sid]; original = inverse_logits(logits[j, 0], geom)
                        else:
                            truth = dev["original_masks"][absolute]; original = inverse_sam_logits(logits[j:j+1], dev["sam_geometries"][absolute])
                        records[f"{arm}__{stage}"][absolute] = {"sample_id": sid, **binary_metrics(original, truth.to(device)), "valid_g0": True}
            if save_maps:
                for j, (sid, absolute) in enumerate(zip(ids, ix.tolist())):
                    maps.append({"sample_id": sid, "valid_seg": True,
                                 "A0_U": rep["A0"]["U_F"][j].half().cpu(), "A1_U": rep["A1"]["U_F"][j].half().cpu(),
                                 "A0_Delta_logits": batch_maps["A0"]["Delta_F"][j].half(),
                                 "A1_Delta_logits": batch_maps["A1"]["Delta_F"][j].half()})
            if (begin + len(raw_ix)) % 200 < BATCH:
                print(json.dumps({"stage": "VAL", "done": begin + len(raw_ix), "total": len(dev["sample_ids"])}), flush=True)
    if any(any(x is None for x in values) for values in records.values()): raise RuntimeError("validation record coverage drift")
    return records, utility, maps


def subset_records(values, valid):
    return [x for x, keep in zip(values, valid) if bool(keep)]


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()), "median": float(np.median(x)),
            "p5": float(np.quantile(x, .05)), "p25": float(np.quantile(x, .25)),
            "p75": float(np.quantile(x, .75)), "p95": float(np.quantile(x, .95)),
            "min": float(x.min()), "max": float(x.max())}


def corr(x, y, kind="spearman"):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return None
    return float(stats.spearmanr(x, y).statistic if kind == "spearman" else np.corrcoef(x, y)[0, 1])


def analyze(records, utility, maps, valid):
    table, paired, valid_table = {}, {}, {}
    for stage in STAGES:
        a, b = records[f"A0__{stage}"], records[f"A1__{stage}"]
        ai = np.asarray([x["foreground_iou"] for x in a]); bi = np.asarray([x["foreground_iou"] for x in b])
        af = np.asarray([x["foreground_f1"] for x in a]); bf = np.asarray([x["foreground_f1"] for x in b])
        table[stage] = {"A0": summarize(a), "A1": summarize(b)}
        paired[stage] = {"iou": paired_statistics(bi, ai), "f1": paired_statistics(bf, af)}
        av, bv = subset_records(a, valid), subset_records(b, valid)
        valid_table[stage] = {"A0": summarize(av), "A1": summarize(bv)}
    final = json.load(open(ROOT / "outputs/phase6g7_block11_17_staged_r1/summary.json"))["comparison"]
    gains = {s: paired[s]["iou"]["mean_difference"] for s in STAGES}
    def ratio(num, den):
        return {"value": num / den, "interpretable": True} if den > 0 else {"value": None, "interpretable": False}
    retention = {"rectifier": ratio(gains["Delta_F"], gains["F_forensic"]),
                 "utility": ratio(gains["U_Delta_F"], gains["Delta_F"]),
                 "sam_space": ratio(gains["S_adapt"], gains["U_Delta_F"])}
    valid_np = np.asarray(valid, dtype=bool)
    ud = {}
    for arm in ARMS:
        vals = [x["mean"] for x, keep in zip(utility[arm], valid_np) if keep]
        meds = [x["median"] for x, keep in zip(utility[arm], valid_np) if keep]
        ud[arm] = {"per_sample_mean_distribution": distribution(vals), "per_sample_median_distribution": distribution(meds)}
    iou = {f"{a}_{s}": np.asarray([x["foreground_iou"] for x in records[f"{a}__{s}"]]) for a in ARMS for s in STAGES}
    gate0 = np.asarray([x.get("mean", np.nan) for x in utility["A0"]]); gate1 = np.asarray([x.get("mean", np.nan) for x in utility["A1"]])
    gain_f = iou["A1_F_forensic"] - iou["A0_F_forensic"]
    gain_d = iou["A1_Delta_F"] - iou["A0_Delta_F"]
    advantage = gain_d > .05
    utility_attr = {"A1_gate_minus_A0_paired": paired_statistics(gate1[valid_np], gate0[valid_np]),
                    "A1_gate_on_delta_advantage_locations_proxy": {
                        "definition": "samples with A1-A0 Delta_F probe IoU > 0.05",
                        "n": int((advantage & valid_np).sum()),
                        "A1_mean_gate": float(gate1[advantage & valid_np].mean()) if (advantage & valid_np).any() else None,
                        "A1_mean_gate_elsewhere": float(gate1[(~advantage) & valid_np].mean())},
                    "correlations_valid": {
                        "gain_F_vs_A1_gate_spearman": corr(gain_f[valid_np], gate1[valid_np]),
                        "gain_Delta_vs_A1_gate_spearman": corr(gain_d[valid_np], gate1[valid_np]),
                        "gain_Delta_vs_A1_gate_pearson": corr(gain_d[valid_np], gate1[valid_np], "pearson")}}
    # Pixel-level proxy: where the A1 Delta probe fixes A0's thresholded error, compare A1 gate magnitude.
    rescued, other = [], []
    for m in maps:
        # Gate and Delta logits already share the exact SAM 64x64 geometry.
        a0 = m["A0_Delta_logits"].float()[0] > 0; a1 = m["A1_Delta_logits"].float()[0] > 0
        # Disagreement favoring A1 cannot be labeled without loading GT again; use A1-positive/A0-negative
        # as a transparent spatial suppression proxy and keep it separate from IoU-based attribution.
        mask = a1 & ~a0; g = m["A1_U"].float()[0]
        if mask.any(): rescued.extend(g[mask].tolist())
        if (~mask).any(): other.extend(g[~mask].tolist())
    utility_attr["position_proxy"] = {"definition": "A1 Delta probe positive and A0 negative at aligned 64x64 locations",
                                       "n_positions": len(rescued), "A1_U_mean": float(np.mean(rescued)) if rescued else None,
                                       "other_positions_A1_U_mean": float(np.mean(other)) if other else None}
    # First clear relative attenuation; thresholds are descriptive and declared, not a trained gate.
    seq = [gains[s] for s in STAGES]
    if gains["F_forensic"] > 0 and gains["Delta_F"] < .5 * gains["F_forensic"]:
        decision = "RECTIFIER_IS_PRIMARY_CONVERSION_BOTTLENECK"
    elif gains["Delta_F"] > 0 and gains["U_Delta_F"] < .5 * gains["Delta_F"]:
        decision = "UTILITY_IS_PRIMARY_SUPPRESSION_BOTTLENECK"
    elif gains["U_Delta_F"] > 0 and gains["S_adapt"] < .5 * gains["U_Delta_F"]:
        decision = "SAM_SPACE_OR_DECODER_IS_PRIMARY_BOTTLENECK"
    else:
        decision = "GAIN_DISSIPATES_GRADUALLY_ACROSS_R1"
    return {"schema": "phase6g8_results_v1", "status": "COMPLETE_STOP", "decision": decision,
            "probe_metrics_full_1106": table, "probe_metrics_valid_seg_1090": valid_table,
            "paired_full_1106": paired, "final_mask": final, "gain_retention": retention,
            "utility_distributions_valid_seg": ud, "utility_attribution": utility_attr,
            "decision_rule": "first stage whose positive gain falls below 50% of preceding stable positive gain; otherwise gradual",
            "firewall": {"training": "linear probes only", "adapter_rectifier_utility_sam_frozen": True,
                         "internal_test": False, "official1000": False, "ood": False}}


def render(result):
    lines = ["# Phase 6G.8 — Evidence-to-Correction Gain Attribution", "",
             "Status: **COMPLETE STOP**. A0/A1 Adapter, Rectifier, Utility, SAM, C1 and CLIP-side sources were frozen. Only matched 1x1 spatial probes were trained on internal TRAIN and selected on internal validation.", "",
             "## Core gain-retention table", "", "| Stage | block22 A0 | block11+17 A1 | Delta | IoU 95% CI | W/T/L | Wilcoxon p |",
             "|---|---:|---:|---:|---|---|---:|"]
    labels = {"F_forensic":"F_forensic", "Delta_F":"Delta_F", "U_Delta_F":"U_F * Delta_F", "S_adapt":"S_adapt"}
    for s in STAGES:
        m, p = result["probe_metrics_full_1106"][s], result["paired_full_1106"][s]["iou"]
        lines.append(f"| {labels[s]} | {m['A0']['mean_foreground_iou']:.6f} | {m['A1']['mean_foreground_iou']:.6f} | {p['mean_difference']:+.6f} | {p['bootstrap_95_ci']} | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} |")
    f = result["final_mask"]; p = f["paired"]["iou"]
    lines.append(f"| final mask | {f['A0']['mean_fg_iou']:.6f} | {f['A1']['mean_fg_iou']:.6f} | {p['mean_delta']:+.6f} | {p['bootstrap_95_ci']} | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} |")
    lines += ["", "## Retention and Utility attribution", "", f"- Gain retention: `{result['gain_retention']}`",
              f"- Utility distributions (valid SEG only): `{result['utility_distributions_valid_seg']}`",
              f"- Utility attribution: `{result['utility_attribution']}`", "",
              "The 1106-row table preserves the historical comparison population. Utility interpretation is restricted to the 1090 deployable exactly-one-SEG samples; the corresponding probe metrics are saved separately in `results.json`.", "",
              f"```text\n{result['decision']}\n```", "", "No test, Official1000, or OOD data were accessed."]
    DOC.write_text("\n".join(lines) + "\n")


def main():
    if (OUT / "summary.json").exists():
        print(json.dumps({"status": "ALREADY_COMPLETE"})); return
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    models, fusion, x0, x1 = load_models(device)
    train_store, val_store = Phase4FStore(hd.CFG, "train"), Phase4FStore(hd.CFG, "val")
    train_ids, dev = train_store.sample_ids, load_dev("g0")
    tc, vc = load_c1_cache("train", train_ids), load_c1_cache("val", dev["sample_ids"])
    fs_train, fs_val = Store("train"), Store("val")
    if fs_train.ids != train_ids or fs_val.ids != dev["sample_ids"]: raise RuntimeError("source pairing drift")
    train_data = q.load_ids(train_ids, ("valid_g0", "S64", "q_seg", "z_L", "F24", "z_F24", "target64", "clip_geometries"))
    sam = load_sam_runtime(hd.CFG, device)
    frozen_before = {"sam": tensor_state_sha256(sam.state_dict()), "fusion": tensor_state_sha256(fusion.state_dict()),
                     **{f"{a}_{k}": tensor_state_sha256(v.state_dict()) for a, triple in models.items() for k, v in zip(("utility","rectifier","adapter"), triple)}}
    protocol = {"schema": "phase6g8_protocol_v1", "status": "FROZEN_BEFORE_FIRST_STEP",
                "population": {"train_manifest": len(train_ids), "train_eligible_seg": int(tc["valid"].sum()),
                               "validation": len(dev["sample_ids"]), "validation_valid_seg": int(vc["valid"].sum()),
                               "train_ids_sha256": sha_ids(train_ids), "validation_ids_sha256": sha_ids(dev["sample_ids"])},
                "A0": {"definition": "current block22 selected C1 new R1", "checkpoint": str(A0_CKPT.resolve()), "sha256": file_sha256(A0_CKPT), "epoch": x0["epoch"]},
                "A1": {"definition": "block11+17 staged selected C1 new R1", "checkpoint": str(A1_CKPT.resolve()), "sha256": file_sha256(A1_CKPT), "epoch": x1["epoch"]},
                "representations": list(STAGES), "probe": {"architecture": "independent Conv2d(256,1,1)", "epochs": EPOCHS,
                "batch": BATCH, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0., "loss": "2*BCE+0.5*Dice",
                "selector": "validation mean FG IoU; tie mean FG F1", "seed": SEED},
                "geometry": {"F_forensic": "legacy CLIP crop geometry", "other_stages": "SAM ResizeLongest1024+pad geometry"},
                "firewall": {"internal_test": False, "official1000": False, "ood": False}}
    dump(OUT / "protocol.json", protocol)
    probes, opt = probe_set(device)
    history_path, inflight_path, last_path = OUT/"training_curve.json", OUT/"epoch_inflight.pt", OUT/"last_state.pt"
    history = json.load(open(history_path)) if history_path.exists() else []
    best = {k: None for k in probes}
    for key in probes:
        p = OUT/"checkpoints"/f"{key}.pt"
        if p.exists():
            old = torch.load(p, map_location="cpu", weights_only=False); best[key] = (tuple(old["score"]), int(old["epoch"]))
    resume_epoch = None; inflight = None
    if inflight_path.exists():
        inflight = torch.load(inflight_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(inflight["probes"]); opt.load_state_dict(inflight["optimizer"]); resume_epoch = int(inflight["epoch"])
    elif last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        probes.load_state_dict(last["probes"]); opt.load_state_dict(last["optimizer"])
    first_epoch = resume_epoch if resume_epoch is not None else len(history) + 1
    for epoch in range(first_epoch, EPOCHS + 1):
        began = time.time(); probes.train()
        if resume_epoch == epoch:
            sums, count = inflight["sums"], int(inflight["count"])
            print(json.dumps({"stage":"RESUME_POST_TRAIN_VALIDATION", "epoch":epoch}), flush=True)
        else:
            order = torch.where(tc["valid"].bool())[0].tolist(); random.Random(SEED + epoch).shuffle(order)
            sums = {k: 0. for k in probes}; count = 0
            for begin in range(0, len(order), BATCH):
                ix = torch.tensor(order[begin:begin+BATCH]); ids = [train_ids[i] for i in ix.tolist()]
                rep, sam_target = representations(models, fusion, fs_train, train_store, train_data, tc, ix, sam, device)
                # Store exposes only features; build a small target lookup lazily
                # from the existing, exactly paired Phase6G.0 block17 shards.
                if not hasattr(main, "clip_train_targets"):
                    lookup = {}
                    for p in cache_paths("late", "train"):
                        z = load_shard(p)
                        for j, row in enumerate(z["records"]): lookup[row["sample_id"]] = z["targets"][j]
                    main.clip_train_targets = lookup
                clip_target = torch.stack([main.clip_train_targets[s] for s in ids]).to(device)
                opt.zero_grad(set_to_none=True); total = torch.zeros((), device=device)
                for arm in ARMS:
                    for stage in STAGES:
                        key = f"{arm}__{stage}"; loss = probe_loss(probes[key](rep[arm][stage].detach()), train_target(stage, clip_target, sam_target))
                        total = total + loss["total"]; sums[key] += float(loss["total"].detach()) * len(ids)
                total.backward(); opt.step(); count += len(ids)
            torch.save({"schema":"phase6g8_epoch_inflight_v1", "epoch":epoch, "probes":probes.state_dict(),
                        "optimizer":opt.state_dict(), "sums":sums, "count":count}, inflight_path)
        rec, _, _ = predict(probes, models, fusion, fs_val, val_store, dev, vc, sam, dev, device)
        row = {"epoch": epoch, "seconds": time.time()-began, "train_samples": count, "metrics": {}}
        for key in probes:
            met = summarize(rec[key]); row["metrics"][key] = met; score = (met["mean_foreground_iou"], met["mean_foreground_f1"])
            if best[key] is None or score > best[key][0]:
                best[key] = (score, epoch); (OUT/"checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save({"schema":"phase6g8_probe_v1", "key":key, "epoch":epoch, "state_dict":probes[key].state_dict(), "score":score}, OUT/"checkpoints"/f"{key}.pt")
        history.append(row); dump(OUT/"training_curve.json", history)
        torch.save({"schema":"phase6g8_last_state_v1", "epoch":epoch, "probes":probes.state_dict(),
                    "optimizer":opt.state_dict()}, last_path)
        inflight_path.unlink(missing_ok=True); resume_epoch = None; inflight = None
        print(json.dumps({"stage":"PROBE_EPOCH", "epoch":epoch, "seconds":row["seconds"], "scores":{k:v["mean_foreground_iou"] for k,v in row["metrics"].items()}}), flush=True)
    for key in probes:
        probes[key].load_state_dict(torch.load(OUT/"checkpoints"/f"{key}.pt", map_location="cpu", weights_only=False)["state_dict"])
    records, utility, maps = predict(probes, models, fusion, fs_val, val_store, dev, vc, sam, dev, device, True)
    for key, value in records.items(): write_rows(OUT/"predictions"/f"{key}.jsonl", value)
    dump(OUT/"utility_per_sample.json", utility); torch.save(maps, OUT/"aligned_utility_delta_maps.pt")
    frozen_after = {"sam": tensor_state_sha256(sam.state_dict()), "fusion": tensor_state_sha256(fusion.state_dict()),
                    **{f"{a}_{k}": tensor_state_sha256(v.state_dict()) for a, triple in models.items() for k, v in zip(("utility","rectifier","adapter"), triple)}}
    if frozen_before != frozen_after: raise RuntimeError("frozen model hash drift")
    result = analyze(records, utility, maps, vc["valid"].tolist())
    result["selected_probe_epochs"] = {k:v[1] for k,v in best.items()}; result["frozen_hashes"] = frozen_after
    dump(OUT/"results.json", result); dump(OUT/"summary.json", {"status":"COMPLETE_STOP", "decision":result["decision"],
         "gain_retention":result["gain_retention"], "selected_probe_epochs":result["selected_probe_epochs"], "firewall":result["firewall"]})
    render(result); print(json.dumps({"status":"COMPLETE_STOP", "decision":result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
