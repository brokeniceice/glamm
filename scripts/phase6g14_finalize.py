#!/usr/bin/env python3
"""Finalize Phase6G.14 after both fresh matched arms complete."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6g10_train_arm as g10
from scripts.phase6g14_single_matched_translator import OUT, build_matched_arms, dump
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache, phase4f_spatial_batch
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def effective_map(model, arm):
    if arm == "A0":
        w1 = model.rectification.cross_attention.out_proj.weight.detach().double().cpu()
        b1 = model.rectification.cross_attention.out_proj.bias.detach().double().cpu()
        w2 = model.rectification.projection.weight.detach().double().cpu()
        b2 = model.rectification.projection.bias.detach().double().cpu()
        return w2 @ w1, w2 @ b1 + b2
    return (model.rectification.projection.weight.detach().double().cpu(),
            model.rectification.projection.bias.detach().double().cpu())


def correction_distribution(models, fusion, projection, p4_val, fs_val, dev, device):
    cache = load_c1_cache("val", dev["sample_ids"])
    recorders = {key: g10.TapRecorder(model.eval().requires_grad_(False)) for key, model in models.items()}
    values = []
    with torch.no_grad():
        for i, sid in enumerate(dev["sample_ids"]):
            if not bool(cache["valid"][i]):
                continue
            s64, _, _, sc, cc = phase4f_spatial_batch(p4_val, [sid], device)
            evidence = g10.fused_evidence(fusion, projection, fs_val, [sid], device)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            taps = {}
            for arm, recorder in recorders.items():
                taps[arm], _, _ = recorder.run(s64, evidence, sc, cc, valid)
            r2a, r2b = taps["A0"]["R2"].float(), taps["A1"]["R2"].float()
            da, db = taps["A0"]["R6"].float(), taps["A1"]["R6"].float()
            values.append({
                "sample_id": sid,
                "r2_cosine": float(F.cosine_similarity(r2a.flatten(), r2b.flatten(), dim=0)),
                "r2_A1_over_A0_norm": float(r2b.norm() / r2a.norm().clamp_min(1e-12)),
                "delta_cosine": float(F.cosine_similarity(da.flatten(), db.flatten(), dim=0)),
                "delta_A1_over_A0_norm": float(db.norm() / da.norm().clamp_min(1e-12)),
                "delta_relative_difference": float((db-da).norm() / da.norm().clamp_min(1e-12)),
            })
            if (i + 1) % 200 == 0:
                print(json.dumps({"stage": "EFFECTIVE_OUTPUT", "done": i + 1, "total": len(dev["sample_ids"])}), flush=True)
    def summary(key):
        x = torch.tensor([row[key] for row in values], dtype=torch.float64)
        return {"n": len(x), "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
                "median": float(x.median()), "p5": float(torch.quantile(x, .05)),
                "p95": float(torch.quantile(x, .95))}
    return values, {key: summary(key) for key in values[0] if key != "sample_id"}


def main():
    for arm in ("A0", "A1"):
        if not (OUT / "arms" / arm / "summary.json").exists():
            raise RuntimeError(f"{arm} is incomplete")
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    models, _ = build_matched_arms(device)
    for arm, model in models.items():
        payload = torch.load(OUT / "arms" / arm / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)

    rec = {arm: rows(OUT / "arms" / arm / "validation" / "selected.jsonl") for arm in ("A0", "A1")}
    if [x["sample_id"] for x in rec["A0"]] != [x["sample_id"] for x in rec["A1"]]:
        raise RuntimeError("A0/A1 selected validation order drift")
    metrics = {arm: summarize_extended(value) for arm, value in rec.items()}
    paired = {
        key: paired_statistics([x[key] for x in rec["A1"]], [x[key] for x in rec["A0"]])
        for key in ("foreground_iou", "foreground_f1")
    }
    w0, b0 = effective_map(models["A0"], "A0")
    w1, b1 = effective_map(models["A1"], "A1")
    matrix = {
        "A0": {"weight": g10.spectrum(w0), "bias_l2": float(b0.norm()), "weight_frobenius": float(w0.norm())},
        "A1": {"weight": g10.spectrum(w1), "bias_l2": float(b1.norm()), "weight_frobenius": float(w1.norm())},
        "comparison": {
            "weight_cosine": float(F.cosine_similarity(w0.flatten(), w1.flatten(), dim=0)),
            "normalized_matrix_similarity": float((w0/w0.norm() * (w1/w1.norm())).sum()),
            "weight_difference_frobenius": float((w1-w0).norm()),
            "relative_weight_difference": float((w1-w0).norm()/w0.norm().clamp_min(1e-12)),
            "bias_cosine": float(F.cosine_similarity(b0, b1, dim=0)),
            "bias_difference_l2": float((b1-b0).norm()),
        },
    }
    fusion = g10.load_fusion(device)
    projection = g10.load_projection(device)
    p4_val, fs_val = g10.Phase4FStore(g10.hd.CFG, "val"), g10.Store("val")
    dev = g10.load_dev("g0")
    output_rows, output_summary = correction_distribution(models, fusion, projection, p4_val, fs_val, dev, device)
    g10.write_rows(OUT / "effective_output_per_sample.jsonl", output_rows)

    d = paired["foreground_iou"]
    delta = float(d["mean_difference"])
    lo, hi = d["bootstrap_95_ci"]
    if delta > 0 and lo > 0:
        decision = "DIRECT_SINGLE_TRANSLATOR_OPTIMIZATION_SUPPORTED"
    elif delta < 0 and hi < 0:
        decision = "FACTORIZED_TRANSLATOR_OPTIMIZATION_SUPPORTED"
    else:
        decision = "SINGLE_TRANSLATOR_SUFFICIENT"
    result = {
        "schema": "phase6g14_final_v1", "status": "COMPLETE_STOP", "decision": decision,
        "metrics": metrics, "paired_A1_minus_A0": paired, "effective_map": matrix,
        "effective_output": output_summary,
        "historical_references": {"phase6g10_A0": .1812439115316707, "phase6g11_main_side": .18554553828922224},
        "selector_rule": "each fresh arm independently selected by DEV G0 mean FG IoU, tie earlier; probes diagnostic only",
        "firewall": {"side_training": False, "utility": False, "nonlinear": False,
                     "internal_test": False, "official1000": False, "ood": False},
    }
    dump(OUT / "results.json", result)
    a0, a1 = metrics["A0"], metrics["A1"]
    f1 = paired["foreground_f1"]
    init = json.loads((OUT / "initialization_audit.json").read_text())
    curves = {arm: json.loads((OUT / "arms" / arm / "training_curve.json").read_text()) for arm in ("A0", "A1")}
    report = f"""# Phase 6G.14 — Single Matched SAM-Compatible Translator

## 1. Exact scientific question

Under frozen block11+17 evidence, C1, and SAM, does an exactly function-matched single affine translator optimize as well as or better than the factorized two-affine translator?

## 2. A0/A1 architecture

- A0: `R2 -> out_proj -> rectification.projection -> gamma -> support -> Delta`.
- A1: `R2 -> single_projection -> gamma -> support -> Delta`.
- Q/K/V, geometry, evidence, C1, SAM, data, seed, optimizer, schedule, precision and selector are matched. No side, Utility, or nonlinearity.

## 3. Function-equivalent initialization derivation

`W_single=W2@W1`, `b_single=W2@b1+b2`, with common `gamma=0.01`.

## 4. Step-0 equivalence numerical check

```json
{json.dumps(init['step0_equivalence'], ensure_ascii=False, indent=2)}
```

The FP32 analytic/function gate passed. BF16 differences are reported, not hidden: factorized and direct paths have different rounding points.

## 5. Parameter / trainable ownership audit

- A0 trainable: {init['ownership']['A0']['trainable_parameter_count']:,}
- A1 trainable: {init['ownership']['A1']['trainable_parameter_count']:,}
- Complete names/shapes are stored in `initialization_audit.json`.

## 6. Training recipe

Phase6G.10 matched recipe: seed 3407, batch 8, 10 epochs, AdamW lr 5e-5, weight decay .05, betas (.9,.999), 5% warmup+cosine, gradient clip 1.0, BF16 forward, canonical internal TRAIN/DEV, independent DEV mean-IoU selection with earlier-epoch tie break.

## 7. DEV results

| Arm | selected epoch | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| A0 fresh factorized | {json.loads((OUT/'arms/A0/selector.json').read_text())['selected_epoch']} | {a0['mean_foreground_iou']:.6f} | {a0['median_foreground_iou']:.6f} | {a0['mean_foreground_f1']:.6f} | {a0['global_foreground_iou']:.6f} | {a0['global_foreground_f1']:.6f} |
| A1 fresh single | {json.loads((OUT/'arms/A1/selector.json').read_text())['selected_epoch']} | {a1['mean_foreground_iou']:.6f} | {a1['median_foreground_iou']:.6f} | {a1['mean_foreground_f1']:.6f} | {a1['global_foreground_iou']:.6f} | {a1['global_foreground_f1']:.6f} |

## 8. Paired statistics

- IoU A1-A0: delta={d['mean_difference']:+.6f}, CI={d['bootstrap_95_ci']}, W/T/L={d['wins']}/{d['ties']}/{d['losses']}, Wilcoxon p={d['wilcoxon_pvalue']:.6g}.
- F1 A1-A0: delta={f1['mean_difference']:+.6f}, CI={f1['bootstrap_95_ci']}, W/T/L={f1['wins']}/{f1['ties']}/{f1['losses']}, Wilcoxon p={f1['wilcoxon_pvalue']:.6g}.

## 9. Training dynamics

Full per-epoch loss, DEV IoU, total/translator gradient norm, effective-map norm and gamma trajectory are in each arm's `training_curve.json`. A0 rows={len(curves['A0'])}; A1 rows={len(curves['A1'])}.

## 10. Effective-map analysis

```json
{json.dumps(matrix, ensure_ascii=False, indent=2)}
```

Actual DEV R2/Delta distribution comparison:

```json
{json.dumps(output_summary, ensure_ascii=False, indent=2)}
```

## 11. Auxiliary R2/Delta diagnostics

Stored under `arms/A0/probe` and `arms/A1/probe`. These probes were trained only after final-localization checkpoint selection and never selected an arm/checkpoint.

## 12. Historical G10/G11 references

| Reference | mean FG IoU | Role |
|---|---:|---|
| historical Phase6G.10 A0 | 0.181244 | non-primary reference |
| historical Phase6G.11 frozen-main+side | 0.185546 | secondary, not fresh paired control |

## 13. Supported interpretation

Decision: **{decision}**. The primary claim is about optimization geometry/implicit regularization of equivalent affine function classes, including precision and AdamW parameterization effects.

## 14. Unsupported interpretation

This experiment cannot claim different nonlinear expressive capacity, cannot select from probes, cannot attribute any result to Utility/side routing, and cannot treat historical references as fresh paired controls.

## 15. Decision

`{decision}`

Status: COMPLETE STOP. No nonlinear translator experiment was started.
"""
    (ROOT / "docs/phase6g14_single_matched_translator.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE_STOP", "decision": decision}), flush=True)


if __name__ == "__main__":
    main()
