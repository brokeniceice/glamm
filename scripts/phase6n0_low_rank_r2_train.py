#!/usr/bin/env python3
"""Phase6N0 fixed-architecture preflight and ten-epoch C2-native training."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from model.c2_low_rank_evidence_r2 import C2LowRankEvidenceR2
from model.c2_r2_localizer import resample_clip_to_sam_lattice
from model.pcerf import (ForensicEvidentialHead, LanguageEvidentialHead,
                         evidence_to_dirichlet, resample_clip_to_original_normalized)
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6l0_r2_train as l0
from scripts.phase4g1p_freeze_preflight import G1C_CKPT, G1C_TEMPERATURES
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase6l0_r2_preflight import (CKPT as C2_CKPT, EXPECTED_SHA as C2_SHA,
                                            OUT as C2_CACHE, c2_native_sam_state,
                                            dump, load_native_runtime, require)
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import sam_coordinates, summarize, tensor_state_sha256
from tools.phase4f import Phase4FStore, deterministic_order, invalid_record, mask_loss

OUT = ROOT / "outputs/phase6n0_low_rank_evidence_r2"
CKPTS = Path("/data/yz/groundingLMM_official/checkpoints/phase6n0_low_rank_evidence_r2")
DOC = ROOT / "docs/phase6n0_low_rank_evidence_r2.md"
CFG_PATH = ROOT / "configs/phase6n0_low_rank_evidence_r2.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
P4F_CFG = l0.P4F_CFG
BASIS_PATH = ROOT / "outputs/phase6m0_r1_intervention/train_basis.pt"
SEED, BATCH, EPOCHS = 3407, 8, 10
PERM = torch.randperm(576, generator=torch.Generator().manual_seed(SEED))


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_basis():
    m0 = json.loads((ROOT / "outputs/phase6m0_r1_intervention/summary.json").read_text())
    require(m0["status"] == "COMPLETE_STOP" and m0["train_valid"] == 8741 and m0["dev_valid"] == 1090,
            "Phase6M0 basis source incomplete")
    actual = file_sha256(BASIS_PATH)
    require(actual == m0["artifacts_sha256"]["train_basis.pt"], "Phase6M0 train basis file SHA drift")
    data = torch.load(BASIS_PATH, map_location="cpu", weights_only=False)
    basis = data["delta"]["eigenvectors"][:, :2].float().contiguous()
    gram = basis.T @ basis
    require(basis.shape == (256, 2) and bool(torch.isfinite(basis).all()) and
            float((gram-torch.eye(2)).abs().max()) < 1e-5, "Phase6M0 basis orthonormality drift")
    rank = json.loads((ROOT / "outputs/phase6m0_r1_intervention/train_rank.json").read_text())
    require(rank["delta"]["image_balanced"]["rank"]["99"] == 2,
            "Phase6M0 two-dimensional TRAIN basis provenance drift")
    return basis, {"path": str(BASIS_PATH), "file_sha256": actual, "tensor_sha256": tensor_sha(basis),
                   "gram_max_abs_error": float((gram-torch.eye(2)).abs().max()),
                   "source_population": "Phase6M0 TRAIN valid C1, uncentered delta second moment"}


class FrozenHeads:
    def __init__(self, device: torch.device):
        target_manifest=json.loads((ROOT/"outputs/phase4g1p/utility_target_manifest.json").read_text())
        require(file_sha256(G1C_CKPT)==target_manifest["source_checkpoint_sha256"] and
                file_sha256(G1C_TEMPERATURES)==target_manifest["temperature_checkpoint_sha256"],
                "G1 historical frozen source-head checkpoint hash drift")
        source = torch.load(G1C_CKPT, map_location="cpu", weights_only=False)
        temps = torch.load(G1C_TEMPERATURES, map_location="cpu", weights_only=False)
        self.language = LanguageEvidentialHead().to(device)
        self.forensic = ForensicEvidentialHead().to(device)
        self.language.load_state_dict(source["language_head"], strict=True)
        self.forensic.load_state_dict(source["forensic_head"], strict=True)
        self.language.eval().requires_grad_(False)
        self.forensic.eval().requires_grad_(False)
        self.t_l = float(temps["T_L"])
        self.t_f = float(temps["T_F"])
        self.hashes = {"g1c_checkpoint_sha256": file_sha256(G1C_CKPT),
                       "temperature_checkpoint_sha256": file_sha256(G1C_TEMPERATURES),
                       "language_state_sha256": tensor_state_sha256(self.language.state_dict()),
                       "forensic_state_sha256": tensor_state_sha256(self.forensic.state_dict())}

    @torch.no_grad()
    def target(self, batch: dict, z_f24: torch.Tensor, target64: torch.Tensor,
               clip_geometries: list, sam_coordinates64: torch.Tensor):
        with torch.autocast(device_type=batch["s64"].device.type, dtype=torch.bfloat16):
            e_l = self.language(batch["s64"], batch["q_seg"], batch["z_l"]) / self.t_l
            e_f = self.forensic(batch["f24"], z_f24) / self.t_f
        p_l = evidence_to_dirichlet(e_l)["posterior"]
        masses = evidence_to_dirichlet(e_f)["masses"]
        mapped = []
        for j, geometry in enumerate(clip_geometries):
            value, _ = resample_clip_to_original_normalized(masses[j:j+1], geometry,
                                                               output_hw=(64, 64), vacuous=True)
            mapped.append(value)
        mapped = torch.cat(mapped)
        p_f = mapped[:, :-1] + mapped[:, -1:] / 2.0
        _, soft = q.target_delta({"p_L": p_l, "p_F": p_f}, target64)
        # Historical target lives on original-normalized 64 cells. Authority
        # lives on padded SAM cells; use the historical audited alignment.
        aligned = hc.gate_to_sam_grid(soft, sam_coordinates64)
        return aligned, {"p_L": p_l, "p_F": p_f, "soft_original": soft}

    def verify(self):
        require(tensor_state_sha256(self.language.state_dict()) == self.hashes["language_state_sha256"] and
                tensor_state_sha256(self.forensic.state_dict()) == self.hashes["forensic_state_sha256"],
                "frozen evidential source head mutated")


def load_population(split: str):
    expected = train_ids() if split == "train" else load_dev("g0")["sample_ids"]
    cache = l0.load_cache(split if split == "train" else "val", expected)
    store = Phase4FStore(P4F_CFG, "train" if split == "train" else "val")
    require(expected == cache["sample_ids"] == store.sample_ids,
            f"{split} canonical ID/order drift")
    require(len(expected) == (8836 if split == "train" else 1106) and
            int(cache["valid_c2_g0"].sum()) == (8682 if split == "train" else 1084),
            f"{split} C2-valid population drift")
    if split == "train":
        source = q.load_ids(expected, ("F24", "z_F24", "target64", "clip_geometries"))
        require(source["sample_ids"] == expected, "frozen F24 TRAIN order drift")
    else:
        dev = load_dev("g0")
        source = {"sample_ids": expected, "F24": dev["F24"],
                  "z_F24": dev["z_F24"], "clip_geometries": dev["clip_geometries"]}
    return expected, cache, store, source


def batch_for(indices: list[int], cache: dict, store: Phase4FStore, source: dict,
              device: torch.device, *, targets: bool = True):
    batch = l0.spatial_batch(store, cache, indices, device)
    index = torch.tensor(indices, dtype=torch.long)
    batch["f24"] = source["F24"].index_select(0, index).to(device)
    batch["z_f24"] = source["z_F24"].index_select(0, index).to(device)
    batch["clip_geometries_source"] = [source["clip_geometries"][i] for i in indices]
    batch["sam_coordinates64"] = torch.stack([store.sam_coords[sid] for sid in batch["sample_ids"]]).to(device)
    if targets:
        batch["target64"] = source["target64"].index_select(0, index).to(device)
    require(batch["clip_geometry"] == batch["clip_geometries_source"], "forensic/C2 geometry drift")
    return batch


def change_evidence(batch: dict, source: dict, cache: dict, indices: list[int],
                    device: torch.device, kind: str):
    result = dict(batch)
    if kind == "cross":
        partner = [(i+1) % len(cache["sample_ids"]) for i in indices]
        ix = torch.tensor(partner, dtype=torch.long)
        result["f24"] = source["F24"].index_select(0, ix).to(device)
        result["attention_map"] = cache["A"].index_select(0, ix).to(device)
        result["evidence_map"] = cache["E"].index_select(0, ix).to(device)
        result["r_prime"] = cache["r_prime"].index_select(0, ix).to(device)
        result["evidence_partner_ids"] = [cache["sample_ids"][j] for j in partner]
    elif kind == "shuffle":
        perm = PERM.to(device)
        for field in ("f24", "attention_map", "evidence_map"):
            value = batch[field]
            result[field] = value.flatten(2)[:, :, perm].reshape_as(value)
        result["evidence_partner_ids"] = list(batch["sample_ids"])
    else:
        raise ValueError(kind)
    return result


def model_kwargs(batch: dict):
    return {"f24": batch["f24"], "attention_map": batch["attention_map"],
            "evidence_map": batch["evidence_map"], "r_prime": batch["r_prime"],
            "q_seg": batch["q_seg"], "s64": batch["s64"], "z_l": batch["z_l"],
            "clip_geometry": batch["clip_geometry"]}


def forward(model, sam, batch, *, with_sam: bool):
    with torch.autocast(device_type=batch["s64"].device.type, dtype=torch.bfloat16):
        result = model(**model_kwargs(batch))
    if not with_sam:
        return result, None, None
    # SAM parameters are frozen, but its operations propagate gradients into
    # S_adapt. Do not put this call under torch.no_grad during training.
    with torch.autocast(device_type=batch["s64"].device.type, enabled=False):
        low = sam(batch["q_seg"].to(torch.bfloat16), result["S_adapt"].to(torch.bfloat16))
    return result, low, mask_loss(low, batch["target"], P4F_CFG)


def total_loss(model, sam, heads, batch, cross, shuffle):
    matched, low, seg = forward(model, sam, batch, with_sam=True)
    crossed, _, _ = forward(model, sam, cross, with_sam=False)
    shuffled, _, _ = forward(model, sam, shuffle, with_sam=False)
    soft, posterior = heads.target(batch, batch["z_f24"], batch["target64"],
                                   batch["clip_geometries_source"], batch["sam_coordinates64"])
    relative = q.image_balanced_loss(matched["authority_logit"], soft, matched["support64"])
    cr = hc.rank_loss(matched["g"], crossed["g"], matched["support64"])
    sr = hc.rank_loss(matched["g"], shuffled["g"], matched["support64"])
    rank = .5 * (cr + sr)
    total = seg["total"] + relative + rank
    require(bool(torch.isfinite(total)) and all(bool(torch.isfinite(v).all()) for v in
            (matched["a"], matched["g"], matched["deltaS"], crossed["g"], shuffled["g"], soft)),
            "nonfinite forward/loss")
    return total, {"seg": seg["total"], "relative": relative, "rank": rank,
                   "cross_rank": cr, "shuffle_rank": sr}, (matched, crossed, shuffled)


def trainable_modules(model):
    return {"SemanticContext": model.semantic, "ForensicContext": model.forensic,
            "SourceAwareMixer": model.mixer, "CoefficientHead": model.coefficient,
            "AuthorityHead": model.authority}


def grad_norm(module) -> float:
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def frozen_hashes(sam, heads, model):
    return {"sam": tensor_state_sha256(sam.state_dict()),
            "language_head": tensor_state_sha256(heads.language.state_dict()),
            "forensic_head": tensor_state_sha256(heads.forensic.state_dict()),
            "basis": tensor_sha(model.basis)}


def record(sid, low, store):
    target = store.original_masks[sid]
    row = metric_record(sid, inverse_sam_logits(low, store.geometries[sid]), target)
    row["tn"] = int(target.numel())-row["tp"]-row["fp"]-row["fn"]
    row["valid_c2_g0"] = True
    return row


def invalid(sid, store):
    target = store.original_masks[sid]
    row = invalid_record(sid, target)
    row["tn"] = int(target.numel())-row["fn"]
    row["valid_c2_g0"] = False
    return row


def summarize_values(values):
    x = np.asarray(values, dtype=np.float64)
    require(len(x) > 0 and np.isfinite(x).all(), "diagnostic population empty/nonfinite")
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


@torch.no_grad()
def image_diagnostic(matched, crossed, shuffled, s64):
    support = matched["support64"]
    mask = support[:, 0].bool()
    coeff = matched["a"].float()
    a1, a2 = coeff[:, 0][mask].abs(), coeff[:, 1][mask].abs()
    gate = matched["g"].float()[:, 0][mask]
    gc = crossed["g"].float()[:, 0][mask]
    gs = shuffled["g"].float()[:, 0][mask]
    delta, base = matched["deltaS"].float(), s64.float()
    active = mask[:, None].expand_as(delta)
    outside = delta.masked_select(~active)
    require(not outside.numel() or float(outside.abs().max()) == 0.0,
            "intervention outside forensic support")
    dnorm = torch.linalg.vector_norm(delta, dim=1)[mask]
    snorm = torch.linalg.vector_norm(base, dim=1)[mask].clamp_min(1e-8)
    ratio = dnorm / snorm
    cosine = torch.nn.functional.cosine_similarity(base.flatten(1),
                                                     matched["S_adapt"].float().flatten(1), dim=1)
    # Orthonormal frozen basis makes these exact component energies.
    energy1 = (gate*a1).square().sum()
    energy2 = (gate*a2).square().sum()
    covariance = torch.cov(torch.stack((coeff[:, 0][mask], coeff[:, 1][mask])))
    return {"a1_abs_mean": float(a1.mean()), "a1_abs_p50": float(torch.quantile(a1,.5)),
            "a1_abs_p90": float(torch.quantile(a1,.9)), "a1_abs_p95": float(torch.quantile(a1,.95)),
            "a2_abs_mean": float(a2.mean()), "a2_abs_p50": float(torch.quantile(a2,.5)),
            "a2_abs_p90": float(torch.quantile(a2,.9)), "a2_abs_p95": float(torch.quantile(a2,.95)),
            "a2_a1_magnitude_ratio": float(a2.mean()/a1.mean().clamp_min(1e-8)),
            "delta_ratio_mean": float(ratio.mean()), "cos_S64_Sadapt": float(cosine.mean()),
            "inside_support_mean_abs_delta": float(delta.masked_select(active).abs().mean()),
            "outside_support_max_abs_delta": float(outside.abs().max()) if outside.numel() else 0.0,
            "b1_energy_fraction": float(energy1/(energy1+energy2).clamp_min(1e-8)),
            "b2_energy_fraction": float(energy2/(energy1+energy2).clamp_min(1e-8)),
            "coefficient_covariance": covariance.cpu().tolist(),
            "g_match": float(gate.mean()), "g_cross": float(gc.mean()),
            "g_shuffle": float(gs.mean()),
            "g_match_minus_cross": float((gate-gc).mean()),
            "g_match_minus_shuffle": float((gate-gs).mean()),
            "support_pixels": int(mask.sum())}


def evaluate(model, sam, store, cache, source, device, epoch: int, *, parity: bool = False):
    model.eval()
    reference = None
    if parity:
        reference = [json.loads(line) for line in
                     (ROOT / "outputs/phase6l0_r2/dev_epoch00_rows.jsonl").read_text().splitlines()]
        require(len(reference) == len(cache["sample_ids"]) == 1106,
                "G4 C2-G0 reference population drift")
    rows, diagnostics = [], []
    with torch.no_grad():
        for i, sid in enumerate(cache["sample_ids"]):
            valid = bool(cache["valid_c2_g0"][i])
            if parity:
                require(reference[i]["sample_id"] == sid and
                        reference[i]["valid_c2_g0"] == valid and
                        valid == bool(cache["seg_count"][i] == 1),
                        f"G4 DEV ID/order/validity drift: {sid}")
            if not valid:
                rows.append(invalid(sid,store)); continue
            batch = batch_for([i],cache,store,source,device,targets=False)
            matched, low, _ = forward(model,sam,batch,with_sam=True)
            if parity:
                require(torch.count_nonzero(matched["a"]) == 0 and
                        torch.count_nonzero(matched["deltaS"]) == 0 and
                        torch.equal(matched["S_adapt"],batch["s64"]),
                        f"G4 DEV step0 embedding mismatch: {sid}")
                with torch.autocast(device_type="cuda",enabled=False):
                    direct = sam(batch["q_seg"].to(torch.bfloat16),batch["s64"].to(torch.bfloat16))
                require(torch.equal(low.cpu(),direct.cpu()) and
                        torch.equal(low.cpu(),batch["z_l"].cpu()),
                        f"G4 DEV step0 SAM logit mismatch: {sid}")
                require(torch.equal(inverse_sam_logits(low,store.geometries[sid]).gt(0),
                                    inverse_sam_logits(batch["z_l"].to(device),store.geometries[sid]).gt(0)),
                        f"G4 DEV step0 binary mask mismatch: {sid}")
            else:
                cross = change_evidence(batch,source,cache,[i],device,"cross")
                shuffle = change_evidence(batch,source,cache,[i],device,"shuffle")
                crossed,_,_ = forward(model,sam,cross,with_sam=False)
                shuffled,_,_ = forward(model,sam,shuffle,with_sam=False)
                diagnostics.append(image_diagnostic(matched,crossed,shuffled,batch["s64"]))
            rows.append(record(sid,low,store))
            if (i+1)%100==0:
                print(json.dumps({"stage":"DEV","epoch":epoch,"done":i+1,
                                  "total":len(cache["sample_ids"])}),flush=True)
    metrics=summarize(rows)
    metrics.update({"tp":sum(r["tp"] for r in rows),"fp":sum(r["fp"] for r in rows),
                    "fn":sum(r["fn"] for r in rows),"tn":sum(r["tn"] for r in rows),
                    "valid_c2_g0":int(cache["valid_c2_g0"].sum()),
                    "invalid_c2_g0":int((~cache["valid_c2_g0"]).sum())})
    if parity:
        require(rows == reference and
                metrics == json.loads((ROOT/"outputs/phase6l0_r2/dev_epoch00_summary.json").read_text()),
                "G4 full DEV C2-G0 per-sample/summary mismatch")
        dump(OUT/"preflight/epoch0_full_dev_parity.json",{"status":"PASS","n":len(rows),
            "valid":metrics["valid_c2_g0"],"invalid":metrics["invalid_c2_g0"],
            "reference_rows_sha256":file_sha256(ROOT/"outputs/phase6l0_r2/dev_epoch00_rows.jsonl")})
    dump(OUT/f"dev_epoch{epoch:02d}_summary.json",metrics)
    path=OUT/f"dev_epoch{epoch:02d}_rows.jsonl"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as handle:
        for row in rows:handle.write(json.dumps(row)+"\n")
    if diagnostics:
        summary={key:summarize_values([d[key] for d in diagnostics]) for key in diagnostics[0]
                 if key != "coefficient_covariance"}
        cov=np.asarray([d["coefficient_covariance"] for d in diagnostics],dtype=np.float64)
        summary["coefficient_covariance_mean"]=cov.mean(0).tolist()
        summary["basis_energy_fraction_pooled"]={
            "b1":float(np.mean([d["b1_energy_fraction"] for d in diagnostics])),
            "b2":float(np.mean([d["b2_energy_fraction"] for d in diagnostics]))}
        dump(OUT/f"diagnostics_epoch{epoch:02d}.json",summary)
    return metrics


def preflight(device: torch.device):
    require(not (OUT/"selector.json").exists() and not (OUT/"training_status.json").exists(),
            "Phase6N0 formal outputs already exist")
    seed_all()
    basis,basis_meta=load_basis()
    model=C2LowRankEvidenceR2(basis).to(device)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(trainable>0 and trainable<4_080_000 and not model.basis.requires_grad,
            "trainable scope or frozen basis drift")
    initial_hash=tensor_state_sha256(model.state_dict())
    require(file_sha256(C2_CKPT)==C2_SHA,"G1 C2 checkpoint hash drift")
    c2_sam_state,sam_provenance=c2_native_sam_state()
    sam=load_native_runtime(c2_sam_state,device)
    heads=FrozenHeads(device)
    require(all(not p.requires_grad for p in sam.parameters()) and
            all(not p.requires_grad for p in heads.language.parameters()) and
            all(not p.requires_grad for p in heads.forensic.parameters()),
            "G1/G7 frozen module requires_grad drift")
    adapter_path=Path(P4F_CFG["evidence"]["forensic_checkpoint"])
    adapter_sha=file_sha256(adapter_path)
    require(adapter_sha==P4F_CFG["evidence"]["forensic_checkpoint_sha256"],
            "G1 frozen forensic adapter checkpoint drift")
    source_manifest=ROOT/"outputs/phase4g1/g1c/source_cache_manifest.json"
    manifest=json.loads(source_manifest.read_text())
    require(manifest["status"]=="COMPLETE" and manifest["population_n"]==8836,
            "G1 historical F24 source cache incomplete")
    frozen_before=frozen_hashes(sam,heads,model)
    ids,cache,store,source=load_population("train")
    dev_ids,dev_cache,dev_store,dev_source=load_population("val")
    require(ids_sha(ids)==manifest["population_ids_sha256"],
            "G1 historical F24 ordered population SHA drift")
    # A fixed, non-selected audit subset checks that F24 from the immutable
    # historical cache equals a live frozen forensic adapter forward.
    from tools.phase4f import evidence_feature, load_evidence_source
    adapter=load_evidence_source(P4F_CFG,"forensic_rect",device)
    adapter_state=tensor_state_sha256(adapter.state_dict())
    subset=[int(i) for i in cache["valid_c2_g0"].nonzero().flatten()[:BATCH]]
    require(len(subset)==BATCH,"G8 no complete valid TRAIN batch")
    errors=[]
    with torch.no_grad():
        for i in subset:
            sid=ids[i]
            raw=store._values(sid)[1][None].to(device=device,dtype=torch.bfloat16)
            live=evidence_feature(adapter,raw)
            cached=source["F24"][i:i+1].to(device)
            errors.append(float((live.float()-cached.float()).abs().max()))
    require(max(errors)==0.0,"G1 cached F24 differs from frozen forensic adapter")
    require(tensor_state_sha256(adapter.state_dict())==adapter_state,
            "G7 frozen adapter changed during parity")
    del adapter
    batch=batch_for(subset,cache,store,source,device)
    f64,support=resample_clip_to_sam_lattice(batch["f24"],batch["clip_geometry"])
    a64,sa=resample_clip_to_sam_lattice(batch["attention_map"],batch["clip_geometry"])
    e64,se=resample_clip_to_sam_lattice(batch["evidence_map"],batch["clip_geometry"])
    require(torch.equal(support,sa) and torch.equal(support,se) and
            bool(support.any()) and bool((~support).any()) and
            float(f64.masked_select(~support.expand_as(f64)).abs().max())==0 and
            float(a64.masked_select(~support.expand_as(a64)).abs().max())==0 and
            float(e64.masked_select(~support.expand_as(e64)).abs().max())==0,
            "G3 geometry support parity/zero outside support failed")
    # The first fixed sample explicitly checks crop geometry against canonical
    # Phase4F coordinates, including SAM padding.
    for sid,geo in zip(batch["sample_ids"],batch["clip_geometry"]):
        require(geo==geometry_for("clip",store.geometries[sid]["original_hw"]) and
                torch.equal(store.sam_coords[sid],sam_coordinates(store.geometries[sid],grid=64)),
                "G3 canonical CLIP/SAM geometry drift")
    dump(OUT/"preflight/geometry_support.json",{"status":"PASS","sample_ids":batch["sample_ids"],
        "support_pixel_counts":support.flatten(1).sum(1).tolist(),"cached_f24_live_max_abs_error":max(errors),
        "mapping_source_sha256":file_sha256(ROOT/"model/c2_r2_localizer.py")})
    # Exact full-DEV identity is checked before any preflight optimizer update.
    baseline=evaluate(model,sam,dev_store,dev_cache,dev_source,device,0,parity=True)
    cross=change_evidence(batch,source,cache,subset,device,"cross")
    shuffle=change_evidence(batch,source,cache,subset,device,"shuffle")
    require(all(torch.equal(batch[key],cross[key]) and torch.equal(batch[key],shuffle[key])
                for key in ("s64","q_seg","z_l")) and
            cross["evidence_partner_ids"]==[ids[(i+1)%len(ids)] for i in subset] and
            torch.equal(cross["f24"],source["F24"].index_select(0,torch.tensor([(i+1)%len(ids) for i in subset])).to(device)) and
            torch.equal(shuffle["f24"],batch["f24"].flatten(2)[:,:,PERM.to(device)].reshape_as(batch["f24"])) and
            torch.equal(batch["r_prime"],shuffle["r_prime"]) and
            batch["clip_geometry"]==cross["clip_geometry"]==shuffle["clip_geometry"],
            "G5 match/cross/shuffle historical protocol drift")
    dump(OUT/"preflight/counterfactual_protocol.json",{"status":"PASS","cross":"canonical cyclic next ID",
        "shuffle":"same fixed CPU seed3407 permutation of F24/A/E 576 patches; r_prime unchanged",
        "cross_forensic_fields":["F24","A","E","r_prime"],
        "semantic_fields_fixed":["S64","q_seg","z_L"],
        "permutation_sha256":tensor_sha(PERM),"sample_ids":batch["sample_ids"],
        "partner_ids":cross["evidence_partner_ids"]})
    # Two temporary optimizer steps prove route availability, then strict
    # restore of the original step-zero model before any formal training.
    original=copy.deepcopy(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    gradients=[]; losses=[]
    model.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss,parts,_=total_loss(model,sam,heads,batch,cross,shuffle)
        loss.backward()
        norms={name:grad_norm(module) for name,module in trainable_modules(model).items()}
        require(all(math.isfinite(v) and v>0 for v in norms.values()) and
                all(p.grad is None for p in sam.parameters()) and
                all(p.grad is None for p in heads.language.parameters()) and
                all(p.grad is None for p in heads.forensic.parameters()) and
                model.basis.grad is None,"G6 two-backward gradient routing failed")
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "G8 nonfinite full-batch gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        gradients.append(norms)
        losses.append({k:float(v.detach()) for k,v in parts.items()})
    model.load_state_dict(original,strict=True)
    model.zero_grad(set_to_none=True)
    require(tensor_state_sha256(model.state_dict())==initial_hash and
            torch.count_nonzero(model.coefficient.output.weight)==0 and
            torch.count_nonzero(model.coefficient.output.bias)==0,
            "preflight optimizer update leaked into formal initialization")
    heads.verify()
    require(frozen_hashes(sam,heads,model)==frozen_before and
            file_sha256(adapter_path)==adapter_sha and file_sha256(C2_CKPT)==C2_SHA,
            "G7 frozen source/basis/SAM integrity failure")
    record={"status":"PASS","G1_SOURCE_CHECKPOINT_HASHES":"PASS",
            "G2_PHASE6M0_BASIS_HASH_AND_ORTHONORMALITY":"PASS",
            "G3_GEOMETRY_SUPPORT_PARITY":"PASS","G4_STEP0_FULL_DEV_EXACT_C2_G0_PARITY":"PASS",
            "G5_MATCH_CROSS_SHUFFLE_PROTOCOL_PARITY":"PASS",
            "G6_TWO_BACKWARD_GRADIENT_ROUTING":"PASS",
            "G7_FROZEN_C2_SAM_ADAPTER_INTEGRITY":"PASS","G8_FINITE_FULL_BATCH":"PASS",
            "basis":basis_meta,"c2_checkpoint_sha256":C2_SHA,
            "c2_sam_state_sha256":frozen_before["sam"],"adapter_checkpoint_sha256":adapter_sha,
            "adapter_state_sha256":adapter_state,"source_heads":heads.hashes,
            "frozen_state_hashes":frozen_before,"initial_model_state_sha256":initial_hash,
            "trainable_parameters":trainable,"train_total":len(ids),"train_valid":int(cache["valid_c2_g0"].sum()),
            "dev_total":len(dev_ids),"dev_valid":int(dev_cache["valid_c2_g0"].sum()),
            "train_c2_cache_index_sha256":file_sha256(C2_CACHE/"train.json"),
            "dev_c2_cache_index_sha256":file_sha256(C2_CACHE/"val.json"),
            "historical_f24_manifest_sha256":file_sha256(source_manifest),
            "model_source_sha256":file_sha256(ROOT/"model/c2_low_rank_evidence_r2.py"),
            "trainer_source_sha256":file_sha256(Path(__file__)),
            "config_sha256":file_sha256(CFG_PATH),
            "full_dev_epoch0_mean_iou":baseline["mean_foreground_iou"],
            "two_backward_grad_norms":gradients,"two_backward_loss_parts":losses}
    dump(OUT/"preflight_gates.json",record)
    print(json.dumps({"status":"PREFLIGHT_PASS","trainable_parameters":trainable,
                      "full_dev_epoch0_mean_iou":baseline["mean_foreground_iou"]}),flush=True)


def training(device: torch.device, *, resume: bool = False):
    gates=json.loads((OUT/"preflight_gates.json").read_text())
    require(gates["status"]=="PASS" and all(gates[k]=="PASS" for k in gates if k.startswith("G")),
            "Phase6N0 formal training forbidden: preflight incomplete")
    require(file_sha256(ROOT/"model/c2_low_rank_evidence_r2.py")==gates["model_source_sha256"] and
            file_sha256(Path(__file__))==gates["trainer_source_sha256"] and
            file_sha256(CFG_PATH)==gates["config_sha256"] and
            file_sha256(BASIS_PATH)==gates["basis"]["file_sha256"] and
            file_sha256(C2_CACHE/"train.json")==gates["train_c2_cache_index_sha256"] and
            file_sha256(C2_CACHE/"val.json")==gates["dev_c2_cache_index_sha256"],
            "frozen implementation/cache/basis changed after preflight")
    require(not (OUT/"selector.json").exists() and not (OUT/"training_status.json").exists(),
            "formal training already selected/completed")
    seed_all()
    basis,meta=load_basis()
    model=C2LowRankEvidenceR2(basis).to(device)
    require(tensor_state_sha256(model.state_dict())==gates["initial_model_state_sha256"] and
            sum(p.numel() for p in model.parameters() if p.requires_grad)==gates["trainable_parameters"],
            "fresh model initialization/parameter count differs from preflight")
    sam_state,_=c2_native_sam_state()
    sam=load_native_runtime(sam_state,device)
    heads=FrozenHeads(device)
    frozen=frozen_hashes(sam,heads,model)
    require(frozen==gates["frozen_state_hashes"],"fresh frozen state differs from preflight")
    train_ids_,train_cache,train_store,train_source=load_population("train")
    _,dev_cache,dev_store,dev_source=load_population("val")
    id_to_index={sid:i for i,sid in enumerate(train_ids_)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    CKPTS.mkdir(parents=True,exist_ok=True)
    history=[];updates=0;start_epoch=1
    progress_path=OUT/"training_progress.json"
    history_path=OUT/"training_history.csv"
    if resume:
        require(progress_path.exists() and history_path.exists(),"no resumable Phase6N0 progress")
        progress=json.loads(progress_path.read_text())
        require(progress["status"]=="EPOCH_COMPLETE" and 1<=progress["epoch"]<EPOCHS,
                "resume requires a completed intermediate epoch")
        checkpoint=CKPTS/f"epoch_{progress['epoch']:02d}.pt"
        require(file_sha256(checkpoint)==progress["checkpoint_sha256"],"resume checkpoint drift")
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        require(state["epoch"]==progress["epoch"] and state["model_source_sha256"]==gates["model_source_sha256"] and
                state["basis_sha256"]==gates["basis"]["tensor_sha256"] and
                state["c2_sha256"]==C2_SHA,"resume metadata drift")
        model.load_state_dict(state["model_state"],strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        updates=int(state["updates"]);start_epoch=int(state["epoch"])+1
        with history_path.open() as handle:history=list(csv.DictReader(handle))
        require(len(history)==start_epoch-1 and [int(r["epoch"]) for r in history]==list(range(1,start_epoch)),
                "resume history drift")
    else:
        require(not progress_path.exists() and not history_path.exists() and not list(CKPTS.glob("epoch_*.pt")),
                "existing Phase6N0 training artifacts; use explicit --resume after audit")
    for epoch in range(start_epoch,EPOCHS+1):
        model.train()
        order=deterministic_order(train_ids_,epoch,seed=SEED)
        aggregates={key:0.0 for key in ("total","seg","relative","rank","cross_rank","shuffle_rank")}
        gradients={key:[] for key in trainable_modules(model)}
        eligible=invalid_count=clips=steps=0
        for start in range(0,len(order),BATCH):
            positions=[id_to_index[sid] for sid in order[start:start+BATCH]]
            chosen=[i for i in positions if bool(train_cache["valid_c2_g0"][i])]
            invalid_count+=len(positions)-len(chosen)
            if not chosen:continue
            n=len(chosen);eligible+=n;steps+=1
            batch=batch_for(chosen,train_cache,train_store,train_source,device)
            cross=change_evidence(batch,train_source,train_cache,chosen,device,"cross")
            shuffle=change_evidence(batch,train_source,train_cache,chosen,device,"shuffle")
            optimizer.zero_grad(set_to_none=True)
            loss,parts,_=total_loss(model,sam,heads,batch,cross,shuffle)
            loss.backward()
            norms={name:grad_norm(module) for name,module in trainable_modules(model).items()}
            require(all(math.isfinite(v) for v in norms.values()) and
                    all(p.grad is None for p in sam.parameters()) and
                    all(p.grad is None for p in heads.language.parameters()) and
                    all(p.grad is None for p in heads.forensic.parameters()) and model.basis.grad is None,
                    "frozen source gradient or nonfinite trainable gradient")
            for key,value in norms.items():gradients[key].append(value)
            total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(bool(torch.isfinite(total_norm)),"nonfinite gradient clip norm")
            clips+=int(float(total_norm)>1.0)
            optimizer.step();updates+=1
            aggregates["total"]+=float(loss.detach())*n
            for key,value in parts.items():aggregates[key]+=float(value.detach())*n
            if updates%100==0:
                print(json.dumps({"stage":"TRAIN","epoch":epoch,"updates":updates,
                                  "batch_loss":float(loss.detach()),"eligible":eligible}),flush=True)
        require((eligible,invalid_count)==(8682,154),
                f"epoch {epoch} canonical TRAIN eligibility drift: {eligible}/{invalid_count}")
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"nonfinite trained parameter")
        heads.verify()
        require(frozen_hashes(sam,heads,model)==frozen and
                file_sha256(C2_CKPT)==C2_SHA and
                file_sha256(Path(P4F_CFG["evidence"]["forensic_checkpoint"]))==gates["adapter_checkpoint_sha256"],
                "G7 frozen source integrity changed during epoch")
        metrics=evaluate(model,sam,dev_store,dev_cache,dev_source,device,epoch)
        ckpt=CKPTS/f"epoch_{epoch:02d}.pt"
        tmp=ckpt.with_suffix(".pt.tmp")
        torch.save({"schema":"phase6n0_low_rank_evidence_r2_epoch_v1","epoch":epoch,"updates":updates,
                    "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "optimizer_state":optimizer.state_dict(),"validation":metrics,
                    "model_source_sha256":gates["model_source_sha256"],
                    "trainer_source_sha256":gates["trainer_source_sha256"],
                    "basis_sha256":gates["basis"]["tensor_sha256"],
                    "c2_sha256":C2_SHA,"frozen_state_hashes":frozen},tmp)
        os.replace(tmp,ckpt)
        row={"epoch":epoch,"updates":updates,"eligible":eligible,"invalid":invalid_count,
             "steps":steps,"sample_order_sha256":ids_sha(order),
             **{key+"_loss":aggregates[key]/eligible for key in aggregates},
             "grad_clip_frequency":clips/steps,
             **{key+"_grad_norm_mean":float(np.mean(values)) for key,values in gradients.items()},
             "dev_mean_fg_iou":metrics["mean_foreground_iou"],
             "dev_mean_fg_f1":metrics["mean_foreground_f1"],
             "dev_global_fg_iou":metrics["global_foreground_iou"],
             "dev_global_fg_f1":metrics["global_foreground_f1"],
             "checkpoint":str(ckpt),"checkpoint_sha256":file_sha256(ckpt)}
        history.append(row)
        with history_path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        dump(progress_path,{"status":"EPOCH_COMPLETE","epoch":epoch,"updates":updates,
                            "checkpoint_sha256":row["checkpoint_sha256"],
                            "dev_mean_fg_iou":row["dev_mean_fg_iou"]})
        print(json.dumps({"stage":"EPOCH_COMPLETE","epoch":epoch,"updates":updates,
                          "dev_mean_fg_iou":row["dev_mean_fg_iou"]}),flush=True)
    best=max(history,key=lambda r:(float(r["dev_mean_fg_iou"]),-int(r["epoch"])))
    best_epoch=int(best["epoch"])
    selected=OUT/"selected_checkpoint.pt";selected.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(CKPTS/f"epoch_{best_epoch:02d}.pt",selected)
    dump(OUT/"selector.json",{"status":"COMPLETE","selected_epoch":best_epoch,
        "selected_checkpoint":str(selected),"selected_checkpoint_sha256":file_sha256(selected),
        "primary":"canonical internal DEV Mean FG IoU only","tie_break":"earlier epoch",
        "selected_dev_mean_fg_iou":float(best["dev_mean_fg_iou"]),"optimizer_updates":updates,
        "c2_sha256":C2_SHA,"basis_sha256":meta["tensor_sha256"]})
    dump(OUT/"training_status.json",{"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV",
        "epochs":EPOCHS,"optimizer_updates":updates,"selected_epoch":best_epoch,
        "historical_sources_untouched":True})
    print(json.dumps({"status":"TRAINING_COMPLETE_STOP_AFTER_INTERNAL_DEV",
                      "selected_epoch":best_epoch,"updates":updates}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","train"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    require(not args.resume or args.stage=="train","--resume only valid for training")
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    if args.stage=="preflight":preflight(device)
    else:training(device,resume=args.resume)


if __name__=="__main__":main()
