#!/usr/bin/env python3
"""Phase 6F.1 frozen C1/new-R1 internal-validation coverage attribution."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from tools.phase4c_b import file_sha256, inverse_sam_logits
from tools.phase4e1 import sam_coordinates
from tools.phase4f import Phase4FStore, evidence_feature, load_evidence_source, load_sam_runtime
from tools.phase3c1 import geometry_for

OUT = ROOT / "outputs/phase6f1_coverage_attribution"
DOC = ROOT / "docs/phase6f1_c1_newr1_coverage_attribution.md"
C1 = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"
SELECTOR = ROOT / "outputs/phase6e2_c1_specific_r1/selector.json"
NEW_R1 = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
INIT = ROOT / "outputs/phase6e2_c1_specific_r1/initialization_provenance.json"
SEED = 3407


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def region_masks(mask: torch.Tensor, geometry: dict) -> dict[str, torch.Tensor]:
    h, w = mask.shape
    rh, rw = map(float, geometry["resized_hw"])
    top, left, bottom, right = map(float, geometry["crop_box_yxyx"])
    y = (torch.arange(h, dtype=torch.float64) + .5) / h
    x = (torch.arange(w, dtype=torch.float64) + .5) / w
    crop_y = (y >= top / rh) & (y < bottom / rh)
    crop_x = (x >= left / rw) & (x < right / rw)
    crop = crop_y[:, None] & crop_x[None, :]
    # Current semantic_support is the convex hull of the first/last 14x14 patch centers.
    hull_y = (y >= (top + 7.) / rh) & (y <= (top + 329.) / rh)
    hull_x = (x >= (left + 7.) / rw) & (x <= (left + 329.) / rw)
    hull = hull_y[:, None] & hull_x[None, :]
    return {"hull": hull, "boundary": crop & ~hull, "outside": ~crop, "crop": crop}


def binary_scores(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float, int, int, int]:
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    den = tp + fp + fn
    iou = tp / den if den else 1.0
    fden = 2 * tp + fp + fn
    f1 = 2 * tp / fden if fden else 1.0
    return iou, f1, tp, fp, fn


def bootstrap_mean(values: list[float], repeats: int = 10000) -> list[float]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a): return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    means = np.empty(repeats)
    for start in range(0, repeats, 500):
        n = min(500, repeats - start)
        means[start:start+n] = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def summarize_group(rows: list[dict]) -> dict:
    di = [r["delta_iou"] for r in rows]; df = [r["delta_f1"] for r in rows]
    result = {
        "n": len(rows),
        "c1_mean_iou": float(np.mean([r["c1_iou"] for r in rows])) if rows else None,
        "c1_median_iou": float(np.median([r["c1_iou"] for r in rows])) if rows else None,
        "new_r1_mean_iou": float(np.mean([r["new_r1_iou"] for r in rows])) if rows else None,
        "new_r1_median_iou": float(np.median([r["new_r1_iou"] for r in rows])) if rows else None,
        "mean_delta_iou": float(np.mean(di)) if rows else None,
        "delta_iou_bootstrap_95_ci": bootstrap_mean(di) if rows else None,
        "mean_delta_f1": float(np.mean(df)) if rows else None,
        "delta_f1_bootstrap_95_ci": bootstrap_mean(df) if rows else None,
    }
    for model in ("c1", "new_r1"):
        for region in ("hull", "boundary", "outside"):
            result[f"{model}_fn_{region}"] = int(sum(r[f"{model}_fn_{region}"] for r in rows))
    return result


def comparison_bootstrap(left: list[float], right: list[float]) -> dict:
    a, b = np.asarray(left), np.asarray(right)
    observed = float(a.mean() - b.mean())
    rng = np.random.default_rng(SEED); values = np.empty(10000)
    for i in range(10000):
        values[i] = a[rng.integers(len(a), size=len(a))].mean() - b[rng.integers(len(b), size=len(b))].mean()
    return {"difference_left_minus_right": observed,
            "bootstrap_95_ci": [float(x) for x in np.quantile(values, [.025, .975])]}


def density(rows: list[dict], model: str, region: str, error: str = "fn") -> float:
    num = sum(r[f"{model}_{error}_{region}"] for r in rows)
    den = sum(r[f"{region}_gt_pixels"] if error == "fn" else r[f"{region}_background_pixels"] for r in rows)
    return num / den if den else float("nan")


def main() -> None:
    if OUT.exists() and (OUT / "summary.json").exists():
        raise RuntimeError("Phase6F.1 already complete; refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    selector = json.loads(SELECTOR.read_text()); init = json.loads(INIT.read_text())
    selected_sha = file_sha256(NEW_R1)
    if selector["selected_checkpoint_sha256"] != selected_sha:
        raise RuntimeError("selected new-R1 SHA drift")
    if init["selected_checkpoint_sha256"] != selected_sha or init["selected_epoch"] != selector["selected_epoch"]:
        raise RuntimeError("new-R1 selector/provenance drift")
    if file_sha256(C1) != C1_SHA: raise RuntimeError("C1 SHA drift")
    protocol = {
        "schema": "phase6f1_protocol_v1", "status": "FROZEN_BEFORE_INFERENCE",
        "scope": "internal validation Fake 1106 only", "baseline": "C1", "candidate": "C1+selected new R1",
        "canonical_g0": True, "mask_threshold": "logit > 0", "optimizer_count": 0,
        "bins": ["1.0", "[0.9,1.0)", "[0.5,0.9)", "(0,0.5)", "0"],
        "firewall": {"internal_test": False, "official1000": False, "external_ood": False,
                     "training": False, "checkpoint_selection": False, "threshold_tuning": False},
    }
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "checkpoint_provenance.json", {
        "c1": {"path": str(C1.resolve()), "sha256": C1_SHA},
        "new_r1": {"path": str(NEW_R1.resolve()), "sha256": selected_sha,
                   "selected_epoch_from_selector": selector["selected_epoch"], "selector": str(SELECTOR.resolve()),
                   "source_provenance": init["new_r1_initialization"]},
        "forensic_adapter": {"path": hd.CFG["evidence"]["forensic_checkpoint"],
                             "sha256": hd.CFG["evidence"]["forensic_checkpoint_sha256"]},
    })
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    store = Phase4FStore(hd.CFG, "val"); ids = store.sample_ids
    if len(ids) != 1106 or len(set(ids)) != 1106: raise RuntimeError("validation population drift")
    cache = load_c1_cache("val", ids)
    state = torch.load(NEW_R1, map_location="cpu", weights_only=False)
    utility, rectifier, _ = hd.load_common(device)
    utility.load_state_dict(state["utility_state"], strict=True); rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
    sam = load_sam_runtime(hd.CFG, device); source = load_evidence_source(hd.CFG, "forensic_rect", device)
    modules = {"utility": utility, "rectifier": rectifier, "sam": sam, "forensic_adapter": source}
    if any(p.requires_grad for m in modules.values() for p in m.parameters()): raise RuntimeError("freeze audit failed")
    records_path = OUT / "per_sample_attribution.jsonl"
    existing = []
    if records_path.exists():
        existing = [json.loads(x) for x in records_path.open() if x.strip()]
        if [x["sample_id"] for x in existing] != ids[:len(existing)]: raise RuntimeError("resume prefix drift")
    parity = {"checked_valid": 0, "rectifier_residual_outside_support_max_abs": 0.,
              "gate_outside_support_max_abs": 0., "decoded_outside_hull_difference_pixels": 0,
              "decoded_total_difference_pixels": 0}
    for pos in range(len(existing), len(ids)):
        sid = ids[pos]; truth = store.original_masks[sid].bool(); h, w = truth.shape
        geom = store.geometries[sid]; clip_geom = geometry_for("clip", (h, w))
        regions = region_masks(truth, clip_geom); gt_n = int(truth.sum())
        valid = bool(cache["valid"][pos])
        if not valid:
            c1_pred = torch.zeros_like(truth); new_pred = torch.zeros_like(truth)
        else:
            s64, raw_clip, _, sc, cc, _ = store.batch([sid], device)
            qseg = cache["q_seg"][pos:pos+1].to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=False): low0 = sam(qseg, s64)
                base_logits = inverse_sam_logits(low0, geom)
                z_l = sam_lowres_to_original_normalized(low0, geom, output_hw=(256,256)).to(torch.bfloat16)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    forensic_out = source(raw_clip, return_features=True)
                forensic = forensic_out["F_forensic"].detach()
                zf = forensic_out["logits"].detach().to(torch.bfloat16)
                batch = {"S64": s64, "q_seg": qseg, "z_L": z_l, "F24": forensic,
                         "z_F24": zf, "clip_geometries": [clip_geom]}
                u = hc.utility_forward(utility, batch)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    rect = rectifier(s64, forensic, sc, cc, torch.ones(1,576,dtype=torch.bool,device=device))
                support = rect["support"].reshape(1,1,64,64)
                gate = ha.gate_to_sam_grid(u["U"], sc) * support.float()
                residual_flat = rect["residual"].flatten(2).transpose(1,2)
                outside = ~rect["support"]
                rv = float(residual_flat[outside].abs().max()) if outside.any() else 0.
                gv = float(gate.flatten(2).transpose(1,2)[outside].abs().max()) if outside.any() else 0.
                parity["checked_valid"] += 1
                parity["rectifier_residual_outside_support_max_abs"] = max(parity["rectifier_residual_outside_support_max_abs"],rv)
                parity["gate_outside_support_max_abs"] = max(parity["gate_outside_support_max_abs"],gv)
                adapted = ha.gated_embedding(s64, rect["image_embeddings"], gate)
                with torch.autocast(device_type=device.type, enabled=False): low1 = sam(qseg, adapted.to(torch.bfloat16))
                new_logits = inverse_sam_logits(low1, geom)
            c1_pred = base_logits.gt(0).cpu(); new_pred = new_logits.gt(0).cpu()
        c1_iou,c1_f1,_,_,_ = binary_scores(c1_pred, truth)
        nr_iou,nr_f1,_,_,_ = binary_scores(new_pred, truth)
        diff = c1_pred ^ new_pred
        parity["decoded_outside_hull_difference_pixels"] += int((diff & ~regions["hull"]).sum())
        parity["decoded_total_difference_pixels"] += int(diff.sum())
        rec = {"sample_id":sid,"width":w,"height":h,"aspect_ratio":max(w,h)/min(w,h),
               "valid_g0":valid,"gt_pixel_count":gt_n,
               "exact_crop_gt_coverage":float((truth&regions["crop"]).sum()/gt_n),
               "current_hull_gt_coverage":float((truth&regions["hull"]).sum()/gt_n),
               "boundary_recoverable_gt_fraction":float((truth&regions["boundary"]).sum()/gt_n),
               "outside_crop_gt_fraction":float((truth&regions["outside"]).sum()/gt_n),
               "gt_area_fraction":gt_n/(h*w),"c1_iou":c1_iou,"c1_f1":c1_f1,
               "new_r1_iou":nr_iou,"new_r1_f1":nr_f1,"delta_iou":nr_iou-c1_iou,"delta_f1":nr_f1-c1_f1}
        for name, region in (("hull",regions["hull"]),("boundary",regions["boundary"]),("outside",regions["outside"])):
            rec[f"{name}_gt_pixels"] = int((truth&region).sum()); rec[f"{name}_background_pixels"] = int((~truth&region).sum())
            for model,pred in (("c1",c1_pred),("new_r1",new_pred)):
                rec[f"{model}_fn_{name}"] = int((truth&~pred&region).sum())
                rec[f"{model}_fp_{name}"] = int((~truth&pred&region).sum())
            rec[f"recovered_fn_{name}"] = int((truth&~c1_pred&new_pred&region).sum())
            rec[f"new_fn_{name}"] = int((truth&c1_pred&~new_pred&region).sum())
            rec[f"new_fp_{name}"] = int((~truth&~c1_pred&new_pred&region).sum())
        append(records_path, rec)
        if (pos+1)%20==0: print(json.dumps({"stage":"PHASE6F1_FROZEN_INFERENCE","done":pos+1,"total":1106}),flush=True)
    rows = [json.loads(x) for x in records_path.open() if x.strip()]
    if [r["sample_id"] for r in rows] != ids: raise RuntimeError("completed order drift")
    if parity["rectifier_residual_outside_support_max_abs"] != 0 or parity["gate_outside_support_max_abs"] != 0:
        raise RuntimeError(f"embedding support parity failed: {parity}")
    def covbin(v):
        if np.isclose(v,1): return "1.0"
        if v>=.9:return "[0.9,1.0)"
        if v>=.5:return "[0.5,0.9)"
        if v>0:return "(0,0.5)"
        return "0"
    bins={k:[] for k in protocol["bins"]}
    for r in rows: bins[covbin(r["exact_crop_gt_coverage"])].append(r)
    coverage_bins={k:summarize_group(v) for k,v in bins.items()}
    aspect={"aspect_ratio_lt_1.2":summarize_group([r for r in rows if r["aspect_ratio"]<1.2]),
            "aspect_ratio_ge_1.2":summarize_group([r for r in rows if r["aspect_ratio"]>=1.2])}
    exact=np.array([r["exact_crop_gt_coverage"] for r in rows]); hull=np.array([r["current_hull_gt_coverage"] for r in rows]); delta=np.array([r["delta_iou"] for r in rows])
    spear={"exact_crop_coverage_vs_delta_iou":dict(zip(("rho","pvalue"),map(float,stats.spearmanr(exact,delta)))),
           "current_hull_coverage_vs_delta_iou":dict(zip(("rho","pvalue"),map(float,stats.spearmanr(hull,delta))))}
    full=[r for r in rows if np.isclose(r["exact_crop_gt_coverage"],1)]; low=[r for r in rows if r["exact_crop_gt_coverage"]<.9]
    group_compare=comparison_bootstrap([r["delta_iou"] for r in low],[r["delta_iou"] for r in full])
    X=np.column_stack([np.ones(len(rows)),exact,[r["gt_area_fraction"] for r in rows],[r["aspect_ratio"] for r in rows]])
    coef=np.linalg.lstsq(X,delta,rcond=None)[0]; pred=X@coef
    adjusted={"formula":"delta_iou ~ exact_crop_coverage + gt_area_fraction + aspect_ratio",
              "coefficients":dict(zip(("intercept","coverage","gt_area_fraction","aspect_ratio"),map(float,coef))),
              "r_squared":float(1-((delta-pred)**2).sum()/((delta-delta.mean())**2).sum())}
    affected=[r for r in rows if r["boundary_recoverable_gt_fraction"]>0]
    unaffected=[r for r in rows if r["boundary_recoverable_gt_fraction"]==0]
    boundary={"affected_n":len(affected),"unaffected_n":len(unaffected),
              "mean_boundary_recoverable_gt_fraction_all":float(np.mean([r["boundary_recoverable_gt_fraction"] for r in rows])),
              "maximum_theoretical_additional_gt_access_pixels":int(sum(r["boundary_gt_pixels"] for r in rows)),
              "maximum_theoretical_additional_gt_access_fraction":float(sum(r["boundary_gt_pixels"] for r in rows)/sum(r["gt_pixel_count"] for r in rows)),
              "affected_vs_unaffected_gain":comparison_bootstrap([r["delta_iou"] for r in affected],[r["delta_iou"] for r in unaffected]),
              "fn_density":{m:{reg:density(rows,m,reg) for reg in ("hull","boundary","outside")} for m in ("c1","new_r1")},
              "fp_density":{m:{reg:density(rows,m,reg,"fp") for reg in ("hull","boundary","outside")} for m in ("c1","new_r1")}}
    statistics={"spearman":spear,"coverage_lt_0.9_minus_coverage_eq_1_gain":group_compare,
                "adjusted_ols_exploratory":adjusted,"overall":summarize_group(rows),"aspect_groups":aspect,
                "architecture_parity":parity}
    dump(OUT/"coverage_bins.json",coverage_bins); dump(OUT/"support_boundary.json",boundary); dump(OUT/"statistics.json",statistics)
    # Conservative preregistered interpretation: require a significant weaker low-coverage gain after the adjusted coefficient has the expected sign.
    acquisition = group_compare["bootstrap_95_ci"][1] < 0 and adjusted["coefficients"]["coverage"] > 0
    boundary_supported = boundary["affected_vs_unaffected_gain"]["bootstrap_95_ci"][1] < 0 and boundary["fn_density"]["new_r1"]["boundary"] > boundary["fn_density"]["new_r1"]["hull"]
    if acquisition: verdict="COVERAGE_BOTTLENECK_SUPPORTED"
    elif boundary_supported: verdict="SUPPORT_BOUNDARY_BOTTLENECK_SUPPORTED"
    else: verdict="COVERAGE_NOT_PRIMARY"
    summary={"schema":"phase6f1_summary_v1","status":"COMPLETE_STOP","verdict":verdict,
             "support_boundary_effect_present":"YES" if boundary_supported else "NO",
             "population":len(rows),"sample_ids_sha256":ids_sha(ids),"optimizer_count":0,
             "artifacts":{"per_sample":str(records_path.resolve()),"coverage_bins":str((OUT/'coverage_bins.json').resolve()),
                          "support_boundary":str((OUT/'support_boundary.json').resolve()),"statistics":str((OUT/'statistics.json').resolve())},
             "firewall":protocol["firewall"]}
    dump(OUT/"summary.json",summary)
    render_doc(summary,coverage_bins,boundary,statistics)
    protocol["status"]="COMPLETE_STOP"; dump(OUT/"protocol.json",protocol)


def render_doc(summary,bins,boundary,st):
    o=st["overall"]; rows=[]
    for k,v in bins.items():
        rows.append(f"| {k} | {v['n']} | {v['c1_mean_iou']:.6f} | {v['c1_median_iou']:.6f} | {v['new_r1_mean_iou']:.6f} | {v['new_r1_median_iou']:.6f} | {v['mean_delta_iou']:.6f} | {v['delta_iou_bootstrap_95_ci']} | {v['mean_delta_f1']:.6f} | {v['delta_f1_bootstrap_95_ci']} | {v['new_r1_fn_hull']} | {v['new_r1_fn_boundary']} | {v['new_r1_fn_outside']} |")
    text=f"""# Phase 6F.1 — C1 + new R1 Forensic Coverage Attribution Diagnostic

## Protocol

Read-only internal-validation Fake diagnostic (`n=1106`). C1 and selected new R1 were frozen; optimizer count was 0. No training, checkpoint selection, threshold tuning, internal test, Official1000, or external OOD access occurred. Both arms reused the identical cached C1 canonical-G0 trajectory, original-space inverse geometry, and `mask logit > 0`; the only model-path difference was selected new R1 disabled/enabled.

## Overall

- C1 mean IoU: `{o['c1_mean_iou']:.6f}`
- C1 + new R1 mean IoU: `{o['new_r1_mean_iou']:.6f}`
- mean delta IoU: `{o['mean_delta_iou']:.6f}`, bootstrap CI `{o['delta_iou_bootstrap_95_ci']}`
- mean delta F1: `{o['mean_delta_f1']:.6f}`, bootstrap CI `{o['delta_f1_bootstrap_95_ci']}`

## Exact-crop coverage bins

| coverage | n | C1 mean IoU | C1 median | new R1 mean IoU | new R1 median | delta IoU | IoU CI | delta F1 | F1 CI | newR1 FN hull | FN boundary | FN outside |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
{chr(10).join(rows)}

## Attribution

- Spearman exact crop coverage vs delta IoU: `{st['spearman']['exact_crop_coverage_vs_delta_iou']}`
- Spearman current hull coverage vs delta IoU: `{st['spearman']['current_hull_coverage_vs_delta_iou']}`
- gain difference, coverage<0.9 minus coverage=1.0: `{st['coverage_lt_0.9_minus_coverage_eq_1_gain']}`
- adjusted exploratory OLS: `{st['adjusted_ols_exploratory']}`
- aspect groups: `{st['aspect_groups']}`

## Support boundary

- boundary-affected images: `{boundary['affected_n']}`; unaffected: `{boundary['unaffected_n']}`
- maximum GT access added by expanding center hull to exact crop footprint: `{boundary['maximum_theoretical_additional_gt_access_pixels']}` pixels, `{boundary['maximum_theoretical_additional_gt_access_fraction']:.6f}` of aggregate GT
- affected-minus-unaffected new-R1 gain: `{boundary['affected_vs_unaffected_gain']}`
- FN densities: `{boundary['fn_density']}`
- FP densities: `{boundary['fp_density']}`

Embedding-level assertions passed: rectifier residual and utility gate were exactly zero outside semantic support. The decoded mask was not forced to match outside support; observed decoded outside-hull changes are recorded in `statistics.json`.

## Decision

`{summary['verdict']}`

`SUPPORT_BOUNDARY_EFFECT_PRESENT = {summary['support_boundary_effect_present']}`

This result is an attribution diagnostic, not evidence that a full-FOV method improves performance. Phase 6F.1 stops here; no multi-tile or geometry correction was implemented.
"""
    DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text(text)


if __name__ == "__main__": main()
