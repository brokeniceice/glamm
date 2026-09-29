#!/usr/bin/env python3
"""Finalize only the canonical internal DEV comparison for Phase6L1."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.c2_r2_localizer_v1 import C2R2LocalizerV1
from scripts.phase6l0_r2_finalize import global_bootstrap, rows
from scripts.phase6l0_r2_preflight import OUT as L0_CACHE, dump, require
from scripts.phase6l1_r2_v1_train import load_cache, spatial_batch
from scripts.phase4gf_formal_localization import load_dev
from tools.phase3c1 import paired_statistics
from tools.phase4c_b import file_sha256
from tools.phase4f import Phase4FStore
from scripts.phase6l1_r2_v1_train import P4F_CFG

RESULTS = ROOT / "outputs/phase6l1_r2_v1"
L0 = ROOT / "outputs/phase6l0_r2"
C1_NATIVE = ROOT / "outputs/phase6e3_c1_native_staged/joint"


def compare(left: list[dict], right: list[dict]) -> dict:
    require(len(left) == len(right) == 1106 and
            [x["sample_id"] for x in left] == [x["sample_id"] for x in right] and
            [x["valid_c2_g0"] for x in left] == [x["valid_c2_g0"] for x in right],
            "paired DEV ID/order/validity mismatch")
    return {"paired": {key: paired_statistics([x[key] for x in left], [x[key] for x in right],
                                                 repeats=2000, seed=3407)
                       for key in ("foreground_iou", "foreground_f1")},
            "global_paired": global_bootstrap(left, right)}


def complete_pair_distance_minima(history: list[dict]) -> None:
    """Replay each frozen epoch to recover the protocol's global minimum.

    Training records the mean/quantiles of per-image minima. Their minimum
    cannot be reconstructed from those summaries, so audit it directly from
    the saved checkpoints and immutable DEV tensors before finalization.
    """
    existing_path = RESULTS / "pair_min_replay.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if (existing.get("status") == "PASS" and len(existing.get("epochs", [])) == 10 and
                all(int(row["epoch"]) == int(saved["epoch"]) and
                    row["checkpoint_sha256"] == saved["checkpoint_sha256"] and
                    file_sha256(RESULTS / f"diagnostics_epoch{int(row['epoch']):02d}.json") ==
                    saved["diagnostic_sha256"]
                    for row, saved in zip(history, existing["epochs"]))):
            return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dev_data = load_dev("g0")
    cache = load_cache("val", dev_data["sample_ids"])
    store = Phase4FStore(P4F_CFG, "val")
    require(store.sample_ids == cache["sample_ids"], "pair-distance replay DEV order drift")
    model = C2R2LocalizerV1().to(device).eval()
    audit = {"status": "PASS", "source": "all ten immutable DEV checkpoints", "epochs": []}
    for row in history:
        epoch = int(row["epoch"])
        checkpoint_path = Path(row["checkpoint"])
        require(file_sha256(checkpoint_path) == row["checkpoint_sha256"],
                f"epoch {epoch} checkpoint drift during pair-distance replay")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        per_image_min = []
        with torch.no_grad():
            for i in cache["valid_c2_g0"].nonzero().flatten().tolist():
                batch = spatial_batch(store, cache, [i], device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(raw_clip_grid=batch["raw_clip_grid"],
                                   attention_map=batch["attention_map"],
                                   evidence_map=batch["evidence_map"],
                                   r_prime=batch["r_prime"], q_seg=batch["q_seg"],
                                   s64=batch["s64"], z_l=batch["z_l"],
                                   clip_geometry=batch["clip_geometry"])
                offsets = output["bridge_offsets"].float()
                per_image_min.append(min(float(torch.linalg.vector_norm(
                    offsets[:, :, a] - offsets[:, :, b], dim=2).min())
                    for a in range(4) for b in range(a + 1, 4)))
        require(len(per_image_min) == int(cache["valid_c2_g0"].sum()),
                "pair-distance replay valid population drift")
        path = RESULTS / f"diagnostics_epoch{epoch:02d}.json"
        diagnostic = json.loads(path.read_text())
        replay_mean = sum(per_image_min) / len(per_image_min)
        require(abs(replay_mean - diagnostic["pair_distance_min"]["mean"]) < 1e-5,
                f"epoch {epoch} pair-distance replay differs from original evaluation")
        minimum = min(per_image_min)
        diagnostic["pair_distance_min"]["min"] = minimum
        dump(path, diagnostic)
        audit["epochs"].append({"epoch": epoch, "checkpoint_sha256": row["checkpoint_sha256"],
                                 "valid_n": len(per_image_min), "per_image_min_mean": replay_mean,
                                 "global_minimum": minimum,
                                 "diagnostic_sha256": file_sha256(path)})
        print(json.dumps({"stage": "PAIR_MIN_REPLAY", "epoch": epoch,
                          "global_minimum": minimum}), flush=True)
    dump(RESULTS / "pair_min_replay.json", audit)


def main() -> None:
    selector = json.loads((RESULTS / "selector.json").read_text())
    status = json.loads((RESULTS / "training_status.json").read_text())
    gates = json.loads((RESULTS / "preflight_gates.json").read_text())
    require(selector["status"] == "COMPLETE" and status["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV" and
            gates["status"] == "PASS" and all(gates[k] == "PASS" for k in gates if k.startswith("G")),
            "Phase6L1 training or hard gates incomplete")
    epoch = int(selector["selected_epoch"])
    require(1 <= epoch <= 10 and file_sha256(Path(selector["selected_checkpoint"])) ==
            selector["selected_checkpoint_sha256"], "selected v1 checkpoint identity drift")
    history = list(csv.DictReader((RESULTS / "training_history.csv").open()))
    require(len(history) == 10 and [int(row["epoch"]) for row in history] == list(range(1, 11)),
            "Phase6L1 ten-epoch history incomplete")
    require(sum(p.numel() for p in C2R2LocalizerV1().parameters()) == gates["R2_trainable_parameters"],
            "v1 parameter-count drift")
    complete_pair_distance_minima(history)

    l0_result = json.loads((L0 / "final_internal_dev.json").read_text())
    require(l0_result["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV" and
            l0_result["selected_checkpoint_sha256"] ==
            "19ec58057b4a8283c6dc666a0ce901c0ce97f86fee8b03e63fcebfd26861b3c5" and
            file_sha256(L0 / "selected_checkpoint.pt") == l0_result["selected_checkpoint_sha256"],
            "Phase6L0 frozen historical baseline drift")
    baseline = rows(RESULTS / "dev_epoch00_rows.jsonl")
    selected = rows(RESULTS / f"dev_epoch{epoch:02d}_rows.jsonl")
    v0 = rows(L0 / "dev_epoch10_rows.jsonl")
    require(baseline == rows(L0 / "dev_epoch00_rows.jsonl"), "v1 epoch0 differs from L0 C2-G0")
    vs_c2 = compare(selected, baseline)
    vs_v0 = compare(selected, v0)
    metrics = {"C2_G0": json.loads((RESULTS / "dev_epoch00_summary.json").read_text()),
               "R2_v0": l0_result["metrics"]["selected"],
               "R2_v1": json.loads((RESULTS / f"dev_epoch{epoch:02d}_summary.json").read_text())}
    diagnostics = json.loads((RESULTS / f"diagnostics_epoch{epoch:02d}.json").read_text())
    v0_diagnostics = l0_result["diagnostics_selected"]
    # The predeclared 0.05-cell minimum remains the strict mechanism claim
    # gate. Also report prevalence so a rare boundary collision is not called
    # a complete bridge collapse.
    four_point_fraction = diagnostics["unique_count_fraction_4"]["mean"]
    one_point_fraction = diagnostics["unique_count_fraction_1"]["mean"]
    all_queries_distinct = diagnostics["pair_distance_min"]["min"] > 0.05
    bridge_not_collapsed = (four_point_fraction > 0.5 and
                            four_point_fraction > one_point_fraction and
                            diagnostics["attention_std_valid_mean"]["mean"] > 0)
    residual_smaller = diagnostics["delta_ratio"]["mean"] < v0_diagnostics["delta_ratio"]["mean"]
    improvement = (vs_v0["paired"]["foreground_iou"]["mean_difference"] > 0 and
                   vs_v0["paired"]["foreground_iou"]["bootstrap_95_ci"][0] > 0)
    if improvement and all_queries_distinct and residual_smaller:
        conclusion = ("The joint R2-v1 revision improves canonical internal-DEV localization, "
                      "retains distinct multi-point sampling at every query, and has a smaller effective SAM residual than R2-v0.")
    elif improvement and not bridge_not_collapsed:
        conclusion = ("R2-v1 improves internal-DEV localization, but the intended multi-point "
                      "deformable mechanism is not retained after training.")
    elif improvement and not all_queries_distinct:
        conclusion = ("R2-v1 improves internal-DEV localization and usually retains distinct sampling, "
                      "but the predeclared every-query diversity criterion is not met. The joint mechanism claim is withheld.")
    elif improvement and not residual_smaller:
        conclusion = ("R2-v1 improves internal-DEV localization, but the residual-scale hypothesis "
                      "is not supported by the selected effective residual norm.")
    elif not improvement and all_queries_distinct and residual_smaller:
        conclusion = ("The intended bridge and residual behavior is better realized, but it does not "
                      "establish improved final localization over R2-v0 on internal DEV.")
    else:
        conclusion = ("The joint R2-v1 revision does not establish improved final localization "
                      "over R2-v0 on canonical internal DEV.")

    c1_selector = json.loads((C1_NATIVE / "selector.json").read_text())
    c1_checkpoint = torch.load(C1_NATIVE / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    require(c1_selector["status"] == "COMPLETE" and
            file_sha256(C1_NATIVE / "selected_checkpoint.pt") == c1_selector["selected_checkpoint_sha256"] and
            c1_checkpoint["epoch"] == c1_selector["selected_epoch"] and
            c1_checkpoint["c1_sha256"] == c1_selector["c1_sha256"],
            "historical C1-native staged R1 selector drift")
    historical = {"label": "C1-native staged R1 Phase6E.3 selector, context only",
                  "epoch": c1_selector["selected_epoch"],
                  "checkpoint_sha256": c1_selector["selected_checkpoint_sha256"],
                  "metrics": {"mean_foreground_iou": c1_selector["selected_metric"]},
                  "matched_single_variable_comparison": False,
                  "reason": "C1/C2 backbones and q_seg-valid populations differ"}
    result = {"status": "COMPLETE_STOP_AFTER_INTERNAL_DEV", "selected_epoch": epoch,
              "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
              "optimizer_updates": selector["optimizer_updates"], "metrics": metrics,
              "comparisons": {"R2_v1_minus_C2_G0": vs_c2, "R2_v1_minus_R2_v0": vs_v0},
              "diagnostics_selected": diagnostics,
              "mechanism": {"four_points_distinct_at_all_selected_queries": all_queries_distinct,
                            "multi_point_sampling_retained_on_majority_of_queries": bridge_not_collapsed,
                            "four_point_unique_fraction": four_point_fraction,
                            "one_point_unique_fraction": one_point_fraction,
                            "effective_residual_mean_ratio_below_v0": residual_smaller,
                            "v1_minus_v0_paired_mean_iou_ci_above_zero": improvement,
                            "conclusion": conclusion},
              "historical_C1_R1": historical,
              "firewall": {"internal_test": False, "official1000": False, "localization_ood": False}}
    dump(RESULTS / "final_internal_dev.json", result)

    def f(x): return f"{float(x):.6f}"
    lines = ["# Phase6L1 — R2-v1 joint revision", "",
             "Status: **COMPLETE STOP AFTER INTERNAL DEV**.", "",
             "## Protocol", "",
             "R2-v1 changes only the bridge offset initialization and adds an unconstrained per-channel LayerScale after the zero-initialized residual head. LayerScale is a residual-scale hypothesis, not a guaranteed norm bound. Frozen C2/SAM, immutable Phase6L0 cache, geometry, loss, optimizer, seed, ten epochs, and DEV Mean FG IoU selector were retained.", "",
             f"Preflight G1–G7: PASS. Epoch0 exact per-sample C2-G0 parity: PASS. Selected epoch: {epoch}; updates: {selector['optimizer_updates']}. Selected checkpoint SHA256: `{selector['selected_checkpoint_sha256']}`.", "",
             "## Canonical internal DEV results", "",
             "| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, key in (("C2-G0 epoch0", "C2_G0"), ("R2-v0 selected", "R2_v0"),
                      (f"R2-v1 selected epoch {epoch}", "R2_v1")):
        m = metrics[key]
        lines.append(f"| {name} | {f(m['mean_foreground_iou'])} | {f(m['mean_foreground_f1'])} | {f(m['global_foreground_iou'])} | {f(m['global_foreground_f1'])} |")
    lines += ["", "| Paired contrast | Mean IoU delta | Median delta | 95% CI | W/T/L | Wilcoxon p | Global IoU delta [95% CI] |",
              "| --- | ---: | ---: | --- | --- | ---: | --- |"]
    for label, value in (("v1 − C2-G0", vs_c2), ("v1 − v0", vs_v0)):
        p = value["paired"]["foreground_iou"]
        g = value["global_paired"]["iou"]
        lines.append(f"| {label} | {f(p['mean_difference'])} | {f(p['median_difference'])} | [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['wilcoxon_pvalue']:.4g} | {f(g['difference'])} [{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}] |")
    lines += ["", "Mean/global F1 paired statistics and full confusion counts are in `outputs/phase6l1_r2_v1/final_internal_dev.json`. Both v0 and v1 selected epochs were chosen using this DEV population, so these intervals do not represent independent held-out confirmation.", "",
              "## Mechanism and stability", "",
              f"Selected v1: minimum pair distance {f(diagnostics['pair_distance_min']['min'])} SAM cells; four-point unique fraction {f(diagnostics['unique_count_fraction_4']['mean'])}; valid-query attention standard deviation {f(diagnostics['attention_std_valid_mean']['mean'])}.", "",
              f"Effective residual ratio v0 → v1: {f(v0_diagnostics['delta_ratio']['mean'])} → {f(diagnostics['delta_ratio']['mean'])}. Cosine(S64,S_adapt) v0 → v1: {f(v0_diagnostics['cosine_S64_Sadapt']['mean'])} → {f(diagnostics['cosine_S64_Sadapt']['mean'])}. Raw residual ratio v1: {f(diagnostics['delta_raw_ratio']['mean'])}; alpha mean/min/max: {f(diagnostics['alpha_mean']['mean'])}/{f(diagnostics['alpha_min']['mean'])}/{f(diagnostics['alpha_max']['mean'])}.", "",
              "Gradient clipping frequency is reported in `training_history.csv`; its value alone cannot prove residual stabilization because LayerScale changes gradient units. Point-specific radii, per-head drift, support behavior, offset/attention distributions, and all epochs are preserved in diagnostics files.", "",
              "## Historical C1-native R1 context", "",
              f"Phase6E.3 C1-native staged R1 epoch-{historical['epoch']} internal DEV Mean FG IoU: {f(historical['metrics']['mean_foreground_iou'])}. R2-v1 {'exceeds' if metrics['R2_v1']['mean_foreground_iou'] > historical['metrics']['mean_foreground_iou'] else 'does not exceed'} it. The Phase6E.3 selected selector does not preserve a DEV global-IoU result, so none is inferred here. C1 and C2 use different backbones and valid populations; this is contextual, not a matched causal comparison.", "",
              "## Conclusion", "", conclusion, "",
              "No separate gain attribution to the two joint revisions is possible. No internal test, Official1000, localization OOD, hyperparameter search, or additional R2 arm was run."]
    (ROOT / "docs/phase6l1_r2_v1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "selected_epoch": epoch}), flush=True)


if __name__ == "__main__":
    main()
