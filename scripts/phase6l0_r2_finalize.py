#!/usr/bin/env python3
"""Paired internal-DEV-only R2 finalization; no sealed dataset access."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase6l0_r2_preflight import OUT, dump, require
from model.c2_r2_localizer import C2R2Localizer
from tools.phase3c1 import paired_statistics
from tools.phase4c_b import file_sha256

RESULTS = OUT.parent


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def global_bootstrap(selected: list[dict], baseline: list[dict], *, repeats=2000, seed=3407) -> dict:
    rng = np.random.default_rng(seed)
    n = len(selected)
    indices = rng.integers(0, n, size=(repeats, n))
    counts = {}
    for name, data in (("selected", selected), ("baseline", baseline)):
        counts[name] = {key: np.asarray([row[key] for row in data], dtype=np.float64)
                        for key in ("tp", "fp", "fn")}
    def metric(which, kind, idx):
        tp = counts[which]["tp"][idx].sum(axis=-1)
        fp = counts[which]["fp"][idx].sum(axis=-1)
        fn = counts[which]["fn"][idx].sum(axis=-1)
        return tp / np.maximum(1, tp+fp+fn) if kind == "iou" else 2*tp / np.maximum(1, 2*tp+fp+fn)
    return {kind: {"difference": float(metric("selected", kind, np.arange(n)) - metric("baseline", kind, np.arange(n))),
                   "bootstrap_95_ci": [float(v) for v in np.quantile(
                       metric("selected", kind, indices) - metric("baseline", kind, indices), [.025, .975])]}
            for kind in ("iou", "f1")}


def main() -> None:
    selector = json.loads((RESULTS / "selector.json").read_text())
    status = json.loads((RESULTS / "training_status.json").read_text())
    require(selector["status"] == "COMPLETE" and status["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV",
            "formal R2 training is incomplete")
    selected_path = Path(selector["selected_checkpoint"])
    require(file_sha256(selected_path) == selector["selected_checkpoint_sha256"], "selected checkpoint SHA drift")
    epoch = int(selector["selected_epoch"])
    baseline = rows(RESULTS / "dev_epoch00_rows.jsonl")
    selected = rows(RESULTS / f"dev_epoch{epoch:02d}_rows.jsonl")
    require(len(baseline) == len(selected) == 1106 and
            [row["sample_id"] for row in baseline] == [row["sample_id"] for row in selected],
            "internal DEV paired population/order drift")
    metrics = {"baseline": json.loads((RESULTS / "dev_epoch00_summary.json").read_text()),
               "selected": json.loads((RESULTS / f"dev_epoch{epoch:02d}_summary.json").read_text())}
    paired = {key: paired_statistics([row[key] for row in selected], [row[key] for row in baseline],
                                    repeats=2000, seed=3407)
              for key in ("foreground_iou", "foreground_f1")}
    global_paired = global_bootstrap(selected, baseline)
    history = list(csv.DictReader((RESULTS / "training_history.csv").open()))
    require(len(history) == 10, "training history lacks ten epochs")
    gates = json.loads((RESULTS / "preflight_gates.json").read_text())
    require(gates["status"] == "PASS", "preflight gates not complete")
    trainable_count = sum(p.numel() for p in C2R2Localizer().parameters())
    require(trainable_count == gates["R2_trainable_parameters"], "R2 parameter count drift")
    native = torch.load(OUT / "c2_sam_runtime.pt", map_location="cpu", weights_only=False)
    sam_count = sum(t.numel() for state in native["state"].values() for t in state.values())
    diagnostics = json.loads((RESULTS / f"diagnostics_epoch{epoch:02d}.json").read_text())
    result = {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV", "selected_epoch": epoch,
              "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
              "optimizer_updates": selector["optimizer_updates"],
              "metrics": metrics, "paired": paired, "global_paired": global_paired,
              "trainable_R2_parameters": trainable_count, "frozen_C2_SAM_runtime_parameters": sam_count,
              "diagnostics_selected": diagnostics,
              "firewall": {"internal_test": False, "official1000": False, "localization_ood": False}}
    dump(RESULTS / "final_internal_dev.json", result)

    def fmt(x): return f"{float(x):.6f}"
    lines = ["# Phase6L0 — C2-native R2 Localization", "", "Status: **COMPLETE STOP AFTER INTERNAL DEV**.", "",
             "## Architecture and provenance", "",
             "R2 uses a new EvidenceConsolidator, geometry-aware local deformable bridge, global q_seg/r_prime conditioner, and SAM residual adapter. It loads no R1 Rectifier or Utility state. The bridge aligns CLIP evidence to the padded S64 lattice. Support is an input feature; deltaS is not multiplied by support. The final residual projection starts at zero, so epoch0 is C2-G0 exactly.", "",
             f"C2 checkpoint SHA256: `{selector['c2_sha256']}`. Frozen SAM state SHA256: `{selector['sam_state_sha256']}`. Selected R2 checkpoint SHA256: `{selector['selected_checkpoint_sha256']}`.", "",
             "Tensor contracts: raw CLIP `[B,1024,24,24]`; A `[B,8,24,24]`; E `[B,512,24,24]`; r_prime `[B,4096]`; q_seg `[B,256]`; S64/F64/R64/deltaS `[B,256,64,64]`; z_L `[B,1,256,256]`; final low-res SAM logits `[B,1,256,256]`.", "",
             f"R2 trainable parameters: {trainable_count:,}. Frozen C2 model: 8,025,250,866 parameters (from the formal C2 model loader audit); the lightweight frozen C2 SAM prompt/mask runtime contains {sam_count:,} of those parameters. C2, RINE, CLIP, LLM, projectors and SAM are frozen during R2 training.", "",
             "## Population and training", "",
             f"TRAIN {json.loads((OUT/'train.json').read_text())['n']} total; C2-valid {json.loads((OUT/'train.json').read_text())['valid_c2_g0']}; invalid {json.loads((OUT/'train.json').read_text())['invalid_c2_g0']}. DEV {metrics['baseline']['n']} total; C2-valid {metrics['baseline']['valid_c2_g0']}; invalid {metrics['baseline']['invalid_c2_g0']}. Invalid DEV images are retained and scored as zero.", "",
             f"Ten joint epochs, AdamW lr 1e-4, weight decay 1e-4, batch 8, seed 3407, no scheduler or accumulation; {selector['optimizer_updates']} optimizer updates. The only loss is 2×BCEWithLogits + 0.5×soft Dice on the frozen C2 SAM final mask. Epoch {epoch} was selected by DEV Mean FG IoU, with earlier epoch as tie break.", "",
             "| Epoch | Train loss | DEV Mean IoU | Mean F1 | Global IoU | Global F1 |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in history:
        lines.append(f"| {row['epoch']} | {fmt(row['train_loss'])} | {fmt(row['dev_mean_fg_iou'])} | {fmt(row['dev_mean_fg_f1'])} | {fmt(row['dev_global_fg_iou'])} | {fmt(row['dev_global_fg_f1'])} |")
    chosen_history = history[epoch-1]
    lines += ["", "## Selected versus epoch0 C2-G0", "",
              "| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | TP | FP | FN | TN |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, key in (("C2-G0 epoch0", "baseline"), (f"C2 + R2 epoch{epoch}", "selected")):
        m = metrics[key]
        lines.append(f"| {name} | {fmt(m['mean_foreground_iou'])} | {fmt(m['mean_foreground_f1'])} | {fmt(m['global_foreground_iou'])} | {fmt(m['global_foreground_f1'])} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
    lines += ["", "Paired selected-minus-epoch0 statistics on the same 1,106 DEV IDs (2,000 bootstrap resamples, seed 3407):", "",
              "| Metric | Mean delta | Median delta | 95% CI | W/T/L | Wilcoxon p |",
              "| --- | ---: | ---: | --- | --- | ---: |"]
    for label, key in (("FG IoU", "foreground_iou"), ("FG F1", "foreground_f1")):
        p = paired[key]
        lines.append(f"| {label} | {fmt(p['mean_difference'])} | {fmt(p['median_difference'])} | [{fmt(p['bootstrap_95_ci'][0])}, {fmt(p['bootstrap_95_ci'][1])}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} |")
    iou_paired = paired["foreground_iou"]
    if iou_paired["mean_difference"] > 0 and iou_paired["bootstrap_95_ci"][0] > 0:
        conclusion = "The internal DEV paired result supports conversion of C2-derived forensic-semantic spatial evidence into SAM-compatible residual adaptation through the jointly trained R2 branch. This does not identify A/E or bridge attention as causal pixel attribution."
    elif iou_paired["mean_difference"] > 0:
        conclusion = "The selected R2 has a positive DEV Mean FG IoU point difference, but its paired bootstrap interval includes zero; this does not establish a stable improvement."
    else:
        conclusion = "The selected R2 did not improve DEV Mean FG IoU over epoch0 C2-G0. No further architecture change or additional arm was started."
    lines += ["", f"Paired bootstrap global IoU delta: {fmt(global_paired['iou']['difference'])}, 95% CI [{fmt(global_paired['iou']['bootstrap_95_ci'][0])}, {fmt(global_paired['iou']['bootstrap_95_ci'][1])}]. Global F1 delta: {fmt(global_paired['f1']['difference'])}, 95% CI [{fmt(global_paired['f1']['bootstrap_95_ci'][0])}, {fmt(global_paired['f1']['bootstrap_95_ci'][1])}].", "",
              "## Mechanism diagnostics", "",
              "Diagnostics are recorded for each epoch in `outputs/phase6l0_r2/diagnostics_epoch*.json`: residual ratio and support behavior, bridge offsets and attention, invalid sampling, and gradients in `training_history.csv`. These were not selectors.", "",
              f"Selected epoch per-image residual ratio mean/median/P95/max: {fmt(diagnostics['delta_ratio']['mean'])}/{fmt(diagnostics['delta_ratio']['median'])}/{fmt(diagnostics['delta_ratio']['p95'])}/{fmt(diagnostics['delta_ratio']['max'])}; cosine(S64,S_adapt): {fmt(diagnostics['cosine_S64_Sadapt']['mean'])}; mean |deltaS| inside/outside support: {fmt(diagnostics['inside_support_mean_abs_delta']['mean'])} / {fmt(diagnostics['outside_support_mean_abs_delta']['mean'])}. Offset mean/median/P95/max (averaged per-image statistics): {fmt(diagnostics['offset_mean']['mean'])}/{fmt(diagnostics['offset_median']['mean'])}/{fmt(diagnostics['offset_p95']['mean'])}/{fmt(diagnostics['offset_max']['mean'])}; boundary fraction {fmt(diagnostics['offset_boundary_fraction']['mean'])}; attention entropy {fmt(diagnostics['attention_entropy']['mean'])}; maximum mass {fmt(diagnostics['attention_max_mass']['mean'])}; invalid sampled-point fraction {fmt(diagnostics['invalid_sample_fraction']['mean'])}; all-invalid query fraction {fmt(diagnostics['all_invalid_query_fraction']['mean'])}.", "",
              f"Selected epoch gradient means: Evidence {fmt(chosen_history['grad_EvidenceConsolidator'])}, Bridge {fmt(chosen_history['grad_GeometryAwareBridge'])}, GlobalConditioner {fmt(chosen_history['grad_GlobalConditioner'])}, blocks 1/2/3 {fmt(chosen_history['grad_ResidualBlock1'])}/{fmt(chosen_history['grad_ResidualBlock2'])}/{fmt(chosen_history['grad_ResidualBlock3'])}, W_out {fmt(chosen_history['grad_W_out'])}; clipping frequency {fmt(chosen_history['gradient_clip_frequency'])}. Finite loss and parameters were enforced; frozen SAM and C2 checkpoint hashes were checked after each epoch.", "",
              "## Conclusion", "", conclusion, "",
              "## Stop boundary", "", "No internal test, Official1000, localization OOD, R2.1, or architecture search was run. The conclusion is limited to canonical internal DEV."]
    (ROOT / "docs/phase6l0_r2.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "selected_epoch": epoch}), flush=True)


if __name__ == "__main__": main()
