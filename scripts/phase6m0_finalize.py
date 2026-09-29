#!/usr/bin/env python3
"""Finalize Phase6M0 only after all frozen diagnostics are complete."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.phase6m0_r1_intervention_audit import OUT, describe, dump, require
from scripts import phase6e2_c1_specific_r1_train as e2
from tools.phase4c_b import file_sha256

DOC=ROOT/"docs/phase6m0_r1_intervention_manifold.md"

def load(name):return json.loads((OUT/name).read_text())
def f(x):return f"{x:.6f}"
def ranks(row):
    r=row["image_balanced"]["rank"]
    return f"| {r['90']} | {r['95']} | {r['99']} |"

def main():
    needed=("protocol.json","provenance.json","train_rank.json","train_magnitude.json",
            "train_counterfactual.json","train_basis.pt","dev_basis_transfer.json",
            "diagnostics/exact_replay_gate.json","diagnostics/train_active_rank.json",
            "diagnostics/train_reconstruction.json","diagnostics/train_counterfactual_rows.json")
    for name in needed:require((OUT/name).exists(),f"missing {name}")
    proto,prov,rank,mag,cf,dev,gate=(load(name) for name in ("protocol.json","provenance.json","train_rank.json",
        "train_magnitude.json","train_counterfactual.json","dev_basis_transfer.json","diagnostics/exact_replay_gate.json"))
    cache_status=json.loads((e2.BASE_OUT/"cache/status.json").read_text())
    c1_path=Path(cache_status["model_meta"]["checkpoint"])
    sam_path=Path(e2.CFG["experiment"]["runtime_root"])/"p1_sam_runtime.pt"
    adapter_path=Path(e2.CFG["evidence"]["forensic_checkpoint"])
    source_files={"c1_checkpoint":{"path":str(c1_path),"sha256":file_sha256(c1_path)},
                  "r1_checkpoint":{"path":str(e2.SELECTED),"sha256":file_sha256(e2.SELECTED)},
                  "sam_runtime":{"path":str(sam_path),"sha256":file_sha256(sam_path)},
                  "forensic_adapter":{"path":str(adapter_path),"sha256":file_sha256(adapter_path)}}
    require(source_files["c1_checkpoint"]["sha256"]==e2.C1_SHA and
            source_files["r1_checkpoint"]["sha256"]==prov["source_hashes"]["r1_checkpoint"] and
            source_files["forensic_adapter"]["sha256"]==e2.CFG["evidence"]["forensic_checkpoint_sha256"],"source file SHA drift")
    dump("source_hashes.json",{"files":source_files,"loaded_state_hashes":prov["source_hashes"]})
    image_mag=load("diagnostics/train_image_magnitude.json")["rows"]
    mag["r"]["per_image"]["P90"]=describe([row["r"]["percentiles"]["90"] for row in image_mag])
    mag["r"]["per_image"]["P95"]=describe([row["r"]["percentiles"]["95"] for row in image_mag])
    dump("train_magnitude.json",mag)
    require(gate["status"]=="PASS" and dev["status"]=="PASS","replay or DEV gate failed")
    require(prov["population"]["train"]["valid"]==cf["n"]==8741,"TRAIN population discrepancy")
    require(prov["population"]["val"]["valid"]==1090,"DEV population discrepancy")
    active=load("diagnostics/train_active_rank.json")
    recon=load("diagnostics/train_reconstruction.json")
    mr=mag["r"]["image_balanced"]
    out=["# Phase6M0 — R1 Intervention Manifold Audit","",
         "Status: **COMPLETE STOP**. Diagnostic only; no training, architecture change, checkpoint selection, Official1000, or OOD run.","",
         "## Provenance and replay","",
         f"Checkpoint: `outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt` (SHA256 `{prov['source_hashes']['r1_checkpoint']}`).",
         "This is Phase6E.2 C1-conditioned R1, initialized from Phase4H-C A2 epoch 3 and Phase4F epoch 9. Phase6E.3 is a distinct from-C1 staged R1.",
         f"TRAIN: {prov['population']['train']['n']} total, {prov['population']['train']['valid']} valid; DEV: {prov['population']['val']['n']} total, {prov['population']['val']['valid']} valid.",
         f"Fixed-subset exact replay gate: PASS; max intervention algebra error {gate['max_abs_error']['intervention']} from bfloat16 rounding; repeated Rectifier, Utility and SAM logits exact.",
         "Historical intermediate tensor snapshots were not saved, so the gate verifies checkpoint/state hashes, authoritative inputs/forward, within-run exact repeatability and intervention algebra. The complete DEV replay is additionally checked against the historical selected mean IoU.","",
         "## Q1. Channel subspace rank","",
         "Primary: uncentered, image-balanced second moment on C1-valid images and Rectifier support pixels; each image contributes equally. C is Rectifier proposal; ΔS is the actual post-gate SAM intervention.","",
         "| Tensor | K90 | K95 | K99 |","|---|---:|---:|---:|",
         f"| C uncentered {ranks(rank['C'])}",f"| ΔS uncentered {ranks(rank['delta'])}",
         f"| C direction {ranks(rank['C_direction'])}",f"| ΔS direction {ranks(rank['delta_direction'])}",
         f"| ΔS centered | {rank['delta']['centered']['rank']['90']} | {rank['delta']['centered']['rank']['95']} | {rank['delta']['centered']['rank']['99']} |","",
         f"ΔS uncentered top-one energy share = {rank['delta']['image_balanced']['eigenvalues'][0]/sum(rank['delta']['image_balanced']['eigenvalues']):.4f}; centered top-one variation share = {rank['delta']['centered']['eigenvalues'][0]/sum(rank['delta']['centered']['eigenvalues']):.4f}. The centered result shows the concentration is not solely a shared mean vector.",
         "Pixel-pooled ΔS ranks: K90/K95/K99 = "+"/".join(str(rank['delta']['pixel_pooled']['rank'][p]) for p in ('90','95','99'))+".",
         "TRAIN active thresholds (image-balanced r P25/P50/P75) = "+" / ".join(f"{k}: {f(v)}" for k,v in rank['active_thresholds_train_image_balanced'].items())+".",
         "Active-only ΔS K95 = "+" / ".join(f"{k}: {v['delta']['image_balanced']['rank']['95']}" for k,v in active.items())+".",
         "Per-image ΔS K90/K95 medians = "+f"{rank['per_image_delta']['K90']['median']} / {rank['per_image_delta']['K95']['median']}.",
         "Half-split mean canonical cosines (K4/8/16/32) = "+" / ".join(f"{rank['half_split_stability_delta'][str(k)]['mean_canonical_cosine']:.4f}" for k in (4,8,16,32))+".",
         "TRAIN ΔS relative reconstruction errors (image-balanced mean, K1/2/4/8/16/32/64/128) = "+" / ".join(f"{recon['delta'][str(k)]['mean']:.4f}" for k in (1,2,4,8,16,32,64,128))+".",
         "Frozen TRAIN ΔS basis → DEV mean projection energy at TRAIN K90/K95/K99 = "+" / ".join(f"{dev['projection_energy_image_balanced'][p]['mean']:.4f}" for p in ('90','95','99'))+".","",
         "## Q2. Intervention magnitude","",
         "r = ||ΔS||₂ / (||S64||₂ + ε), on the same model support. Image-balanced TRAIN quantiles:","",
         "| P50 | P75 | P90 | P95 | P99 | max |","|---:|---:|---:|---:|---:|---:|",
         "| "+" | ".join(f(mr[p]) for p in ('50','75','90','95','99'))+f" | {f(mag['r']['pixel_pooled']['max'])} |","",
         f"C/S64 median (image-balanced) = {f(mag['r_C']['image_balanced']['50'])}; ΔS/C median = {f(mag['delta_over_C']['image_balanced']['50'])}.",
         f"cos(S64,Sadapt), per-image mean = {f(mag['cos_S64_Sadapt_per_image']['mean'])}.","",
         "## Q3. Evidence counterfactuals","",
         "A holds the matched Rectifier C fixed and changes only Utility; this matches the historical Utility ranking semantics. B changes both Rectifier and Utility evidence; it is a new diagnostic, not the historical training contract.","",
         "TRAIN IoU/F1 use the Phase4F SAM-canvas target after removing padding and mapping to the original-normalized 256 grid. The frozen SAM logits use the same geometry normalization path. DEV replay IoU below uses original-image masks.","",
         "| Condition | mean U | mean intervention ratio | mean ΔIoU vs SAM | mean mask change |","|---|---:|---:|---:|---:|"]
    for key,label,part in (("matched","matched",cf["A_historical_utility_semantics"]),
                           ("cross_utility","cross Utility",cf["A_historical_utility_semantics"]),
                           ("shuffle_utility","shuffle Utility",cf["A_historical_utility_semantics"]),
                           ("cross_full","cross full",cf["B_full_evidence_mismatch_new_diagnostic"]),
                           ("shuffle_full","shuffle full",cf["B_full_evidence_mismatch_new_diagnostic"])):
        r=part[key]
        out.append(f"| {label} | {f(r['mean_u']['mean'])} | {f(r['intervention_ratio']['mean'])} | {f(r['delta_iou']['mean'])} | {f(r['mask_change']['mean'])} |")
    out += ["", "Projection energy onto the frozen matched ΔS TRAIN K95 basis: "+
            " / ".join(f"{name}: {cf[part][name]['projection_energy_K95']['mean']:.4f}" for name,part in
                         (("matched","A_historical_utility_semantics"),("cross_utility","A_historical_utility_semantics"),
                          ("shuffle_utility","A_historical_utility_semantics"),("cross_full","B_full_evidence_mismatch_new_diagnostic"),
                          ("shuffle_full","B_full_evidence_mismatch_new_diagnostic")))+"."]
    out += ["","Paired A comparisons (matched minus mismatch):","",
            "| Comparison | ΔU mean [95% CI] | Δr mean [95% CI] | ΔIoU mean [95% CI] | ΔIoU W/T/L | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|"]
    for key,label in (("cross_utility","cross Utility"),("shuffle_utility","shuffle Utility")):
        r=cf["pairwise"]["matched_minus_"+key]
        fmt=lambda x:f"{f(x['mean'])} [{f(x['bootstrap_95_ci'][0])}, {f(x['bootstrap_95_ci'][1])}]"
        d=r["delta_iou"]
        out.append(f"| {label} | {fmt(r['mean_u'])} | {fmt(r['intervention_ratio'])} | {fmt(d)} | {d['wins']}/{d['ties']}/{d['losses']} | {d['wilcoxon_p_two_sided']:.3g} |")
    out += ["","## Interpretation","",
            "Q1: The deployed Phase6E.2 R1 intervention is concentrated in a low-dimensional channel subspace on internal TRAIN. This survives centered, active-only, image-balanced/pixel-pooled and fixed half-split checks, and the frozen basis transfers to DEV. A low-dimensional channel basis alone does not establish that a new model can reproduce the spatial coefficients or the final SAM masks.","",
            "Q2: The actual R1 intervention is large relative to S64 on model support; it is not a small perturbation in this measured norm. This magnitude distribution is descriptive, not a chosen R2 residual budget.","",
            "Q3: Under the historical Utility-only mismatch contract, matched forensic context produces a larger gate and a larger mean TRAIN segmentation benefit than deterministic cross/shuffle evidence. Paired intervals exclude zero, while the per-image gate-difference/benefit-difference correlations are weak. Full evidence mismatch is a separate diagnostic. Similar matched-basis projection energy across conditions means out-of-basis movement is not the dominant contrast captured by this basis test; amplitude and spatial coefficients require separate interpretation.","",
            f"Historical selected DEV mean IoU = {dev['historical_selected_dev_metric']:.9f}; replay = {dev['replayed_dev_metric']['mean_foreground_iou']:.9f}; absolute drift = {dev['mean_iou_absolute_drift']:.9f}.",
            "","The rank and magnitude findings describe this deployed R1 on internal TRAIN. They do not establish an intrinsic SAM dimension or a chosen R2 hyperparameter. All hypothesis failures remain in the diagnostic JSON files.",""]
    DOC.parent.mkdir(parents=True,exist_ok=True)
    temp=DOC.with_suffix(".md.tmp");temp.write_text("\n".join(out));os.replace(temp,DOC)
    dump("summary.json",{"status":"COMPLETE_STOP","checkpoint_sha256":prov["source_hashes"]["r1_checkpoint"],
                         "train_valid":8741,"dev_valid":1090,"report":str(DOC),
                         "artifacts_sha256":{name:file_sha256(OUT/name) for name in (*needed,"source_hashes.json")},
                         "report_sha256":file_sha256(DOC),"no_training":True,"no_new_r2":True})
    print(json.dumps({"status":"COMPLETE_STOP","report":str(DOC)}),flush=True)

if __name__=="__main__":main()
