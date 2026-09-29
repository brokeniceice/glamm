#!/usr/bin/env python3
"""Phase6N0 final internal-DEV-only comparisons and fixed-protocol report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from model.c2_low_rank_evidence_r2 import C2LowRankEvidenceR2
from scripts.phase6l0_r2_finalize import global_bootstrap, rows
from scripts.phase6l0_r2_preflight import dump, require
from scripts.phase6n0_low_rank_r2_train import BASIS_PATH, CFG_PATH, OUT, load_basis
from tools.phase3c1 import paired_statistics
from tools.phase4c_b import file_sha256

L0=ROOT/"outputs/phase6l0_r2"
E2=ROOT/"outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
DOC=ROOT/"docs/phase6n0_low_rank_evidence_r2.md"
E2_SHA="c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603"


def paired(left,right):
    require(len(left)==len(right)==1106 and
            [x["sample_id"] for x in left]==[x["sample_id"] for x in right] and
            [x["valid_c2_g0"] for x in left]==[x["valid_c2_g0"] for x in right],
            "paired DEV ID/order/validity drift")
    return {"paired":{key:paired_statistics([x[key] for x in left],[x[key] for x in right],
                                           repeats=2000,seed=3407)
                      for key in ("foreground_iou","foreground_f1")},
            "global_paired":global_bootstrap(left,right)}


def main():
    gates=json.loads((OUT/"preflight_gates.json").read_text())
    status=json.loads((OUT/"training_status.json").read_text())
    selector=json.loads((OUT/"selector.json").read_text())
    require(gates["status"]=="PASS" and all(gates[k]=="PASS" for k in gates if k.startswith("G")) and
            status["status"]=="COMPLETE_STOP_AFTER_INTERNAL_DEV" and status["epochs"]==10 and
            selector["status"]=="COMPLETE","Phase6N0 gates or ten-epoch training incomplete")
    require(file_sha256(ROOT/"model/c2_low_rank_evidence_r2.py")==gates["model_source_sha256"] and
            file_sha256(ROOT/"scripts/phase6n0_low_rank_r2_train.py")==gates["trainer_source_sha256"] and
            file_sha256(CFG_PATH)==gates["config_sha256"],"implementation drift after preflight")
    history=list(csv.DictReader((OUT/"training_history.csv").open()))
    require(len(history)==10 and [int(x["epoch"]) for x in history]==list(range(1,11)) and
            all(int(x["eligible"])==8682 and int(x["invalid"])==154 for x in history),
            "formal TRAIN traversal incomplete")
    epoch=int(selector["selected_epoch"])
    best=max(history,key=lambda r:(float(r["dev_mean_fg_iou"]),-int(r["epoch"])))
    require(epoch==int(best["epoch"]) and
            float(selector["selected_dev_mean_fg_iou"])==float(best["dev_mean_fg_iou"]),
            "selector did not use DEV Mean FG IoU with earlier tie")
    checkpoint=Path(selector["selected_checkpoint"])
    require(file_sha256(checkpoint)==selector["selected_checkpoint_sha256"]==best["checkpoint_sha256"],
            "selected checkpoint file SHA drift")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    basis,basis_meta=load_basis()
    model=C2LowRankEvidenceR2(basis)
    model.load_state_dict(state["model_state"],strict=True)
    require(state["epoch"]==epoch and state["basis_sha256"]==basis_meta["tensor_sha256"] and
            sum(p.numel() for p in model.parameters() if p.requires_grad)==gates["trainable_parameters"] and
            torch.equal(model.basis,basis),"selected checkpoint basis/architecture drift")
    l0_final=json.loads((L0/"final_internal_dev.json").read_text())
    require(l0_final["status"]=="COMPLETE_STOP_AFTER_INTERNAL_DEV" and
            file_sha256(L0/"selected_checkpoint.pt")==l0_final["selected_checkpoint_sha256"],
            "R2-v0 frozen baseline checkpoint drift")
    baseline=rows(L0/"dev_epoch00_rows.jsonl")
    v0=rows(L0/f"dev_epoch{l0_final['selected_epoch']:02d}_rows.jsonl")
    selected=rows(OUT/f"dev_epoch{epoch:02d}_rows.jsonl")
    require(baseline==rows(OUT/"dev_epoch00_rows.jsonl"),"Phase6N0 epoch0 C2-G0 baseline drift")
    vs_c2,vs_v0=paired(selected,baseline),paired(selected,v0)
    metrics={"C2_G0":json.loads((OUT/"dev_epoch00_summary.json").read_text()),
             "R2_v0":l0_final["metrics"]["selected"],
             "Phase6N0_R2":json.loads((OUT/f"dev_epoch{epoch:02d}_summary.json").read_text())}
    require(metrics["Phase6N0_R2"]["n"]==metrics["C2_G0"]["n"]==1106 and
            metrics["Phase6N0_R2"]["valid_c2_g0"]==1084,"selected DEV population drift")
    diagnostics={str(i):json.loads((OUT/f"diagnostics_epoch{i:02d}.json").read_text()) for i in range(1,11)}
    chosen=diagnostics[str(epoch)]
    require(chosen["outside_support_max_abs_delta"]["max"]==0.0 and
            chosen["b1_energy_fraction"]["mean"]+chosen["b2_energy_fraction"]["mean"]>0.999,
            "selected basis/support diagnostic integrity failure")
    require(file_sha256(E2)==E2_SHA,"historical E2 R1 checkpoint drift")
    e2=torch.load(E2,map_location="cpu",weights_only=False)
    historical={"label":"Phase6E.2 C1-conditioned R1 contextual reference only",
                "checkpoint_sha256":E2_SHA,"epoch":e2["epoch"],
                "metrics":e2["validation_g0"],"matched_causal_comparison":False,
                "reason":"C1/C2 backbone and SEG-valid population differ"}
    require(abs(historical["metrics"]["mean_foreground_iou"]-0.20273210126852226)<1e-12 and
            abs(historical["metrics"]["global_foreground_iou"]-0.2293094676544747)<1e-12,
            "historical R1 metric anchor drift")
    result={"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV","selected_epoch":epoch,
            "selected_checkpoint_sha256":selector["selected_checkpoint_sha256"],
            "optimizer_updates":selector["optimizer_updates"],"trainable_parameters":gates["trainable_parameters"],
            "basis":basis_meta,"metrics":metrics,
            "comparisons":{"Phase6N0_minus_C2_G0":vs_c2,"Phase6N0_minus_R2_v0":vs_v0},
            "diagnostics_selected":chosen,"diagnostics_all_epochs":diagnostics,
            "historical_Phase6E2_R1":historical,
            "firewall":{"internal_test":False,"official1000":False,"localization_ood":False,
                        "architecture_search":False}}
    dump(OUT/"final_internal_dev.json",result)
    f=lambda x:f"{float(x):.6f}"
    lines=["# Phase6N0 — Low-Rank Evidence-Authorized R2","",
           "Status: **COMPLETE STOP AFTER INTERNAL DEV**.","",
           "## Frozen protocol","",
           f"Frozen Phase6M0 TRAIN ΔS uncentered top-two basis: file SHA256 `{basis_meta['file_sha256']}`; tensor SHA256 `{basis_meta['tensor_sha256']}`. Basis orthonormality max error {basis_meta['gram_max_abs_error']:.2e}. No DEV basis fitting or basis rotation.","",
           f"Preflight G1–G8: PASS. Full canonical DEV epoch0 C2-G0 exact per-sample parity: PASS. Trainable parameters: {gates['trainable_parameters']:,}. Ten epochs and {selector['optimizer_updates']} optimizer updates; selected epoch {epoch} by DEV Mean FG IoU only.","",
           "## Canonical internal DEV","",
           "| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |",
           "|---|---:|---:|---:|---:|"]
    for name,key in (("C2-G0","C2_G0"),("R2-v0 selected","R2_v0"),
                     (f"Phase6N0 selected epoch {epoch}","Phase6N0_R2")):
        m=metrics[key]
        lines.append(f"| {name} | {f(m['mean_foreground_iou'])} | {f(m['mean_foreground_f1'])} | {f(m['global_foreground_iou'])} | {f(m['global_foreground_f1'])} |")
    lines += ["","| Paired contrast | Mean IoU Δ | 95% CI | W/T/L | Wilcoxon p | Global IoU Δ [95% CI] |",
              "|---|---:|---|---:|---:|---|"]
    for label,comp in (("N0 − C2-G0",vs_c2),("N0 − R2-v0",vs_v0)):
        p=comp["paired"]["foreground_iou"]
        g=comp["global_paired"]["iou"]
        lines.append(f"| {label} | {f(p['mean_difference'])} | [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.3g} | {f(g['difference'])} [{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}] |")
    h=historical["metrics"]
    lines += ["","Mean/global F1, confusion counts and complete paired statistics are saved in `final_internal_dev.json`. The selected epoch was chosen on this DEV population, so the intervals are descriptive for selection, not independent confirmation.","",
              "## Mechanism diagnostics","",
              f"Selected |a1|/|a2| means: {f(chosen['a1_abs_mean']['mean'])}/{f(chosen['a2_abs_mean']['mean'])}; a2/a1 magnitude ratio: {f(chosen['a2_a1_magnitude_ratio']['mean'])}. Per-image basis intervention energy fractions b1/b2: {f(chosen['b1_energy_fraction']['mean'])}/{f(chosen['b2_energy_fraction']['mean'])}. Coefficient covariance mean: `{chosen['coefficient_covariance_mean']}`.","",
              f"Intervention ||ΔS||/||S64|| mean: {f(chosen['delta_ratio_mean']['mean'])}; cosine(S64,Sadapt) mean: {f(chosen['cos_S64_Sadapt']['mean'])}; outside-support max |ΔS|: {f(chosen['outside_support_max_abs_delta']['max'])}.","",
              f"Authority g matched/cross/shuffle means: {f(chosen['g_match']['mean'])}/{f(chosen['g_cross']['mean'])}/{f(chosen['g_shuffle']['mean'])}; matched−cross/shuffle: {f(chosen['g_match_minus_cross']['mean'])}/{f(chosen['g_match_minus_shuffle']['mean'])}. Cross uses the next canonical ID for F24/A/E/r_prime with current geometry; shuffle uses one fixed seed-3407 576-patch permutation for F24/A/E and keeps r_prime.","",
              "Each epoch's coefficient quantiles, basis use, authority differences, gradient norms and clipping frequency are retained in the diagnostic JSON files and training history. None was a selector.","",
              "## Required questions","",
              f"Q1 — C2-G0: selected Phase6N0 {'is higher' if metrics['Phase6N0_R2']['mean_foreground_iou']>metrics['C2_G0']['mean_foreground_iou'] else 'is not higher'} in DEV Mean FG IoU; paired mean Δ = {f(vs_c2['paired']['foreground_iou']['mean_difference'])}, CI [{f(vs_c2['paired']['foreground_iou']['bootstrap_95_ci'][0])}, {f(vs_c2['paired']['foreground_iou']['bootstrap_95_ci'][1])}].","",
              f"Q2 — R2-v0: Phase6N0 is numerically higher in DEV Mean FG IoU by {f(vs_v0['paired']['foreground_iou']['mean_difference'])}, but the paired 95% CI [{f(vs_v0['paired']['foreground_iou']['bootstrap_95_ci'][0])}, {f(vs_v0['paired']['foreground_iou']['bootstrap_95_ci'][1])}] includes zero and Wilcoxon p = {vs_v0['paired']['foreground_iou']['wilcoxon_pvalue']:.3g}. This DEV result does not establish a stable Mean IoU improvement over v0.","",
              f"Q3 — historical C1 R1: Phase6E.2 mean/global IoU = {f(h['mean_foreground_iou'])}/{f(h['global_foreground_iou'])}; Phase6N0 mean/global IoU = {f(metrics['Phase6N0_R2']['mean_foreground_iou'])}/{f(metrics['Phase6N0_R2']['global_foreground_iou'])}. N0 is numerically higher in mean by {f(metrics['Phase6N0_R2']['mean_foreground_iou']-h['mean_foreground_iou'])} and lower in global by {f(metrics['Phase6N0_R2']['global_foreground_iou']-h['global_foreground_iou'])}. Different backbones and validity populations make this contextual, not a matched causal contrast.","",
              f"Q4 — second basis direction: measured mean b2 energy fraction = {f(chosen['b2_energy_fraction']['mean'])}, with |a2|/|a1| mean ratio = {f(chosen['a2_a1_magnitude_ratio']['mean'])}. No binary mechanism threshold was predeclared; these are the observed usage measures.","",
              f"Q5 — matched authority: mean g_match−g_cross = {f(chosen['g_match_minus_cross']['mean'])}, g_match−g_shuffle = {f(chosen['g_match_minus_shuffle']['mean'])}; both {'positive' if chosen['g_match_minus_cross']['mean']>0 and chosen['g_match_minus_shuffle']['mean']>0 else 'not both positive'} on selected DEV. This describes the fixed mismatch protocol, not calibrated confidence.","",
              "No K sweep, extra arm, internal test, Official1000, localization OOD or follow-on R2 was run."]
    DOC.write_text("\n".join(lines)+"\n")
    dump(OUT/"summary.json",{"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV",
        "selected_epoch":epoch,"selected_checkpoint_sha256":selector["selected_checkpoint_sha256"],
        "final_result_sha256":file_sha256(OUT/"final_internal_dev.json"),
        "report":str(DOC),"report_sha256":file_sha256(DOC)})
    print(json.dumps({"status":"COMPLETE_STOP_AFTER_INTERNAL_DEV","selected_epoch":epoch}),flush=True)


if __name__=="__main__":main()
