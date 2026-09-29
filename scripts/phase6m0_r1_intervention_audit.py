#!/usr/bin/env python3
"""Phase6M0: read-only, checkpoint-frozen R1 intervention audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase4g1q_conditional_utility as q
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6e2_c1_specific_r1_train as e2
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore, load_evidence_source, load_sam_runtime

OUT = ROOT / "outputs/phase6m0_r1_intervention"
CKPT = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
EXPECTED_SHA = "c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603"
SEED = 3407
EPS = 1e-8
KS = (1, 2, 4, 8, 16, 32, 64, 128)
PCTS = (10, 25, 50, 75, 90, 95, 99)


def dump(name: str, obj) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def require(cond, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def describe(x, pcts=PCTS):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "max": float(x.max()), "percentiles": {str(p): float(np.percentile(x, p)) for p in pcts}}


def weighted_quantiles(values: list[np.ndarray], pcts=PCTS):
    # Each image has total weight one; every supported pixel within it is equal.
    x = np.concatenate(values)
    w = np.concatenate([np.full(len(v), 1.0 / len(v), dtype=np.float64) for v in values])
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / len(values)
    return {str(p): float(x[min(np.searchsorted(cdf, p / 100), len(x)-1)]) for p in pcts}


def spectrum(moment: torch.Tensor, mean: torch.Tensor | None = None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    m = moment.clone() if mean is None else moment - torch.outer(mean, mean)
    m = (m + m.T) * .5
    require(bool(torch.isfinite(m).all()), "nonfinite second moment")
    ev, vec = torch.linalg.eigh(m)
    negative = float(ev[ev < 0].abs().sum())
    minimum = float(ev.min())
    ev = ev.flip(0).clamp_min(0)
    vec = vec.flip(1)
    total = ev.sum()
    require(float(total) > 0, "zero eigen energy")
    cumulative = ev.cumsum(0) / total
    ranks = {str(p): int(torch.searchsorted(cumulative, p / 100).item() + 1) for p in (50,75,80,90,95,99)}
    info = {"rank": ranks, "eigenvalues": ev.tolist(), "minimum_eigenvalue": minimum,
            "negative_mass": negative, "trace": float(m.trace()),
            "eigenvalue_sum": float(ev.sum()), "trace_relative_error": float(abs(m.trace()-ev.sum())/total)}
    require(info["trace_relative_error"] < 1e-5, "eigen trace mismatch")
    require(negative / float(total) < 1e-5, "large negative eigenvalue mass")
    return info, ev, vec


class Moments:
    def __init__(self):
        self.sum = torch.zeros(256, dtype=torch.float64)
        self.second = torch.zeros((256,256), dtype=torch.float64)
        self.pooled_sum = torch.zeros(256, dtype=torch.float64)
        self.pooled_second = torch.zeros((256,256), dtype=torch.float64)
        self.n_images = self.n_pixels = 0

    def add(self, x: torch.Tensor):
        if not x.numel():
            return
        x = x.detach().to(device="cpu", dtype=torch.float64)
        require(bool(torch.isfinite(x).all()), "nonfinite vectors")
        s, ss = x.sum(0), x.T @ x
        self.sum += s / len(x)
        self.second += ss / len(x)
        self.pooled_sum += s
        self.pooled_second += ss
        self.n_images += 1
        self.n_pixels += len(x)

    def finish(self):
        require(self.n_images > 0, "no supported images")
        a, b = self.sum / self.n_images, self.second / self.n_images
        c, d = self.pooled_sum / self.n_pixels, self.pooled_second / self.n_pixels
        ib, ev, vec = spectrum(b)
        centered, cev, cvec = spectrum(b, a)
        pooled, _, _ = spectrum(d)
        return ({"image_balanced": ib, "pixel_pooled": pooled, "centered": centered,
                 "n_images": self.n_images, "n_pixels": self.n_pixels},
                {"mean": a, "eigenvectors": vec, "eigenvalues": ev,
                 "centered_eigenvectors": cvec, "centered_eigenvalues": cev})


class Runtime:
    def __init__(self, device: torch.device):
        require(file_sha256(CKPT) == EXPECTED_SHA, "R1 selected checkpoint SHA drift")
        self.device = device
        self.state = torch.load(CKPT, map_location="cpu", weights_only=False)
        require((self.state.get("epoch"), self.state.get("optimizer_updates")) == (8,8840), "selected epoch/update drift")
        require(self.state.get("c1_sha256") == e2.C1_SHA, "checkpoint C1 SHA drift")
        self.model, _ = hc.load_utility("a2", device)
        self.model.load_state_dict(self.state["utility_state"], strict=True)
        self.model.eval().requires_grad_(False)
        self.rectifier, _ = hc.load_phase4f(device)
        self.rectifier.load_state_dict(self.state["rectifier_state"], strict=True)
        self.rectifier.eval().requires_grad_(False)
        self.sam = load_sam_runtime(e2.CFG, device)
        self.source = load_evidence_source(e2.CFG, "forensic_rect", device)
        history = json.loads((e2.BASE_OUT / "initialization_provenance.json").read_text())
        frozen = {"sam": tensor_state_sha256(self.sam.state_dict()),
                  "source": tensor_state_sha256(self.source.state_dict()),
                  "heads": tensor_state_sha256(q.source_state(self.model))}
        require(frozen == history["frozen_hash_before"] == history["frozen_hash_after"], "frozen SAM/adapter/head state drift")
        self.hashes = {"r1_checkpoint": EXPECTED_SHA, "c1_checkpoint": e2.C1_SHA,
                       "utility_state": tensor_state_sha256(self.model.state_dict()),
                       "rectifier_state": tensor_state_sha256(self.rectifier.state_dict()), **frozen}
        for key, state_key in (("utility_state", "utility_state"), ("rectifier_state", "rectifier_state")):
            require(self.hashes[key] == tensor_state_sha256(self.state[state_key]), f"{key} failed exact restore")

    def population(self, split: str):
        ids = train_ids() if split == "train" else load_dev("g0")["sample_ids"]
        store = Phase4FStore(e2.CFG, "train" if split == "train" else "val")
        require(ids == store.sample_ids, f"{split} spatial ID drift")
        cache = e2.load_c1_cache(split if split == "train" else "val", ids)
        valid = cache["valid"].bool()
        require(int(valid.sum()) == (8741 if split == "train" else 1090), f"{split} valid count drift")
        data = (q.load_ids(ids, ("S64", "q_seg", "z_L", "F24", "z_F24", "clip_geometries")) if split == "train"
                else load_dev("g0"))
        require(data["sample_ids"] == ids, "shared data ID drift")
        return ids, store, cache, data

    @torch.no_grad()
    def forward(self, ids, store, cache, data, index: int):
        idx = torch.tensor([index])
        batch, _ = e2.c1_language_batch(data, cache, idx, store, self.sam, self.device)
        sid = ids[index]
        s64, rect, support, _, sc = e2.phase4f_batch(store, [sid], self.source, self.rectifier, self.device)
        u = hc.utility_forward(self.model, batch)
        gate = hc.gate_to_sam_grid(u["U"], sc) * support.float()
        adapted = hc.gated_embedding(s64, rect, gate)
        return {"batch": batch, "s64": s64, "rect": rect, "support": support.bool(),
                "sc": sc, "u": u, "gate": gate, "adapted": adapted,
                "c": rect.float()-s64.float(), "delta": adapted.float()-s64.float()}


def gate(rt: Runtime):
    ids, store, cache, data = rt.population("val")
    indices = [i for i in (0, 127, 254, 381, 508, 635, 762, 889, 1016, 1105) if bool(cache["valid"][i])]
    require(len(indices) >= 8, "DEV replay subset too small")
    errors = {k: 0.0 for k in ("q_seg", "F24", "z_F24", "S64", "C_identity", "intervention", "repeat_rectifier", "repeat_utility", "repeat_logits")}
    with torch.no_grad():
        for i in indices:
            one = rt.forward(ids, store, cache, data, i)
            two = rt.forward(ids, store, cache, data, i)
            expected_q = cache["q_seg"][i:i+1].to(rt.device, dtype=torch.bfloat16)
            errors["q_seg"] = max(errors["q_seg"], float((one["batch"]["q_seg"].float()-expected_q.float()).abs().max()))
            for key in ("F24", "z_F24", "S64"):
                expected = data[key][i:i+1].to(rt.device)
                errors[key] = max(errors[key], float((one["batch"][key].float()-expected.float()).abs().max()))
            errors["C_identity"] = max(errors["C_identity"], float((one["c"]-(one["rect"].float()-one["s64"].float())).abs().max()))
            errors["intervention"] = max(errors["intervention"], float((one["delta"]-one["gate"]*one["c"]).abs().max()))
            errors["repeat_rectifier"] = max(errors["repeat_rectifier"], float((one["rect"].float()-two["rect"].float()).abs().max()))
            errors["repeat_utility"] = max(errors["repeat_utility"], float((one["u"]["U"].float()-two["u"]["U"].float()).abs().max()))
            with torch.autocast(device_type=rt.device.type, enabled=False):
                a = rt.sam(one["batch"]["q_seg"], one["adapted"].to(torch.bfloat16))
                b = rt.sam(two["batch"]["q_seg"], two["adapted"].to(torch.bfloat16))
            errors["repeat_logits"] = max(errors["repeat_logits"], float((a.float()-b.float()).abs().max()))
    require(max(errors.values()) == 0.0 or (errors["intervention"] <= 0.015625 and
            max(v for k,v in errors.items() if k != "intervention") == 0.0), f"exact replay gate failed: {errors}")
    result = {"status": "PASS", "fixed_dev_indices": indices, "sample_ids": [ids[i] for i in indices],
              "max_abs_error": errors, "intervention_rounding_tolerance": 0.015625,
              "historical_intermediate_tensors_available": False,
              "scope": "strict selected checkpoint/state restore, authoritative forward, input identity, repeatability, algebra; historical intermediate snapshots were not persisted"}
    dump("diagnostics/exact_replay_gate.json", result)
    return result


def vectorize(out):
    mask = out["support"][0,0]
    require(bool(mask.any()), "sample has no supported locations")
    s = out["s64"][0].float().permute(1,2,0)[mask].cpu()
    c = out["c"][0].permute(1,2,0)[mask].cpu()
    d = out["delta"][0].permute(1,2,0)[mask].cpu()
    u = out["gate"][0,0][mask].float().cpu()
    sn, cn, dn = s.norm(dim=1), c.norm(dim=1), d.norm(dim=1)
    r = dn / sn.clamp_min(EPS)
    return s,c,d,u,sn,cn,dn,r


def train_pass(rt: Runtime):
    ids, store, cache, data = rt.population("train")
    moments = {name: Moments() for name in ("C", "delta", "C_direction", "delta_direction", "delta_half_a", "delta_half_b")}
    ratios, per_image, per_image_k = [], [], []
    per_pixel = {key: [] for key in ("r_C", "delta_over_C", "U64", "cos_C_S64", "cos_delta_S64")}
    image_cos, n = [], 0
    for i,sid in enumerate(ids):
        if not bool(cache["valid"][i]): continue
        with torch.no_grad(): out = rt.forward(ids,store,cache,data,i)
        s,c,d,u,sn,cn,dn,r = vectorize(out)
        for key,x in (("C",c),("delta",d)):
            moments[key].add(x)
            norms = x.norm(dim=1)
            positive = norms > EPS
            moments[key+"_direction"].add(x[positive]/norms[positive,None])
        moments["delta_half_a" if int(hashlib.sha256(sid.encode()).hexdigest(),16)%2 == 0 else "delta_half_b"].add(d)
        ratios.append(r.numpy())
        per_image.append({"sample_id": sid, "support_pixels": int(len(r)), "r": describe(r.numpy(), (50,90,95)),
                          "cos_S64_Sadapt": float(torch.nn.functional.cosine_similarity(s.flatten()[None], (s+d).flatten()[None]).item())})
        per_pixel["r_C"].append((cn/sn.clamp_min(EPS)).numpy())
        per_pixel["delta_over_C"].append((dn/cn.clamp_min(EPS)).numpy())
        per_pixel["U64"].append(u.numpy())
        per_pixel["cos_C_S64"].append(torch.nn.functional.cosine_similarity(c,s,dim=1,eps=EPS).numpy())
        per_pixel["cos_delta_S64"].append(torch.nn.functional.cosine_similarity(d,s,dim=1,eps=EPS).numpy())
        image_cos.append(per_image[-1]["cos_S64_Sadapt"])
        # Per-image dimensionality is computed on its own supported vectors.
        m = (d.double().T @ d.double()) / len(d)
        ev = torch.linalg.eigvalsh((m+m.T)*.5).flip(0).clamp_min(0)
        cum = ev.cumsum(0)/ev.sum().clamp_min(EPS)
        per_image_k.append({"sample_id": sid, "K90": int(torch.searchsorted(cum,.9).item()+1),
                            "K95": int(torch.searchsorted(cum,.95).item()+1)})
        n += 1
        if n % 200 == 0: print(json.dumps({"stage":"TRAIN_MOMENTS","done":n,"total":int(cache["valid"].sum())}),flush=True)
    require(n == 8741, "TRAIN traversal incomplete")
    rank, basis = {}, {}
    for key,item in moments.items(): rank[key],basis[key] = item.finish()
    stability = {}
    for k in (4,8,16,32):
        va,vb = basis["delta_half_a"]["eigenvectors"][:,:k],basis["delta_half_b"]["eigenvectors"][:,:k]
        cos = torch.linalg.svdvals(va.T @ vb).clamp(0,1)
        stability[str(k)] = {"mean_canonical_cosine":float(cos.mean()), "minimum_canonical_cosine":float(cos.min()),
                             "principal_angles_degrees":torch.rad2deg(torch.arccos(cos)).tolist()}
    rank["half_split_stability_delta"] = stability
    rank["per_image_delta"] = {key:describe([row[key] for row in per_image_k],(25,75,90,95)) for key in ("K90","K95")}
    rank["half_assignment"] = "SHA256(sample_id) integer parity"
    magnitude = {"r": {"image_balanced":weighted_quantiles(ratios), "pixel_pooled":describe(np.concatenate(ratios)),
                       "per_image":{key:describe([row["r"][key] if key != "max" else row["r"]["max"] for row in per_image])
                                    for key in ("mean","median","max")}},
                 "cos_S64_Sadapt_per_image":describe(image_cos), "support_pixels":int(sum(map(len,ratios))),
                 "valid_images":n}
    for key,arrays in per_pixel.items():
        magnitude[key] = {"image_balanced":weighted_quantiles(arrays),"pixel_pooled":describe(np.concatenate(arrays))}
    magnitude["U64"]["fraction_lt_0_1_pixel"] = float(np.mean(np.concatenate(per_pixel["U64"])<.1))
    magnitude["U64"]["fraction_gt_0_9_pixel"] = float(np.mean(np.concatenate(per_pixel["U64"])>.9))
    magnitude["U64"]["fraction_lt_0_1_image"] = float(np.mean([np.mean(x<.1) for x in per_pixel["U64"]]))
    magnitude["U64"]["fraction_gt_0_9_image"] = float(np.mean([np.mean(x>.9) for x in per_pixel["U64"]]))
    thresholds = {"top75":magnitude["r"]["image_balanced"]["25"],
                  "top50":magnitude["r"]["image_balanced"]["50"],
                  "top25":magnitude["r"]["image_balanced"]["75"]}
    rank["active_thresholds_train_image_balanced"] = thresholds
    dump("train_rank.json",rank)
    dump("train_magnitude.json",magnitude)
    temp = OUT / "train_basis.pt.tmp"
    torch.save({key:{field:value for field,value in row.items()} for key,row in basis.items()},temp)
    os.replace(temp,OUT/"train_basis.pt")
    dump("diagnostics/train_image_dimensionality.json",{"rows":per_image_k})
    dump("diagnostics/train_image_magnitude.json",{"rows":per_image})
    return rank,magnitude


def provenance(rt: Runtime):
    cache_status = json.loads((e2.BASE_OUT/"cache/status.json").read_text())
    require(cache_status["status"] == "COMPLETE", "C1 cache incomplete")
    population = {}
    for split in ("train","val"):
        ids,store,cache,_ = rt.population(split)
        index = e2.BASE_OUT/"cache"/("train.json" if split=="train" else "val.json")
        population[split] = {"n":len(ids),"valid":int(cache["valid"].sum()),
                             "invalid":int((~cache["valid"].bool()).sum()),"ordered_ids_sha256":ids_sha(ids),
                             "cache_index_path":str(index),"cache_index_sha256":file_sha256(index),
                             "cache_shard_sha256":[file_sha256(p) for p in sorted((e2.CACHE/split).glob("shard_*.pt"))]}
    dump("protocol.json",{"status":"DIAGNOSTIC_ONLY","no_training":True,"no_new_architecture":True,
                          "no_checkpoint_selection":True,"seed":SEED,"primary_population":"TRAIN valid C1 and Rectifier support",
                          "rank":"uncentered image-balanced FP64 CPU second moment","mismatch_A":"Utility only",
                          "mismatch_B":"Utility and Rectifier","forbidden_splits":["internal_test","Official1000","OOD"]})
    dump("provenance.json",{"checkpoint_lineage":"Phase6E.2 C1-conditioned R1 initialized from A2 epoch3 and Phase4F epoch9",
                            "phase6e3_distinct":True,"source_hashes":rt.hashes,"population":population,
                            "implementation_sha256":file_sha256(Path(__file__)),"seed":SEED})
    return population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","train"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda:0")
    rt=Runtime(device)
    if args.stage=="preflight":
        provenance(rt)
        gate(rt)
        print(json.dumps({"status":"PREFLIGHT_PASS"}),flush=True)
    else:
        require((OUT/"diagnostics/exact_replay_gate.json").exists(),"preflight gate absent")
        require(json.loads((OUT/"diagnostics/exact_replay_gate.json").read_text())["status"]=="PASS","preflight failed")
        train_pass(rt)
        print(json.dumps({"status":"TRAIN_MOMENTS_COMPLETE"}),flush=True)


if __name__ == "__main__": main()
