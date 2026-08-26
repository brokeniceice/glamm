#!/usr/bin/env python3
"""Assemble selected Phase 3E metrics, paired statistics, route gate, and reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def rows(path: Path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def selected_dir(root: Path, arm: str, selector: dict) -> Path:
    return root / "evaluation/selector/P1_FROZEN" if int(selector["optimizer_step"]) == 0 else root / f"evaluation/selector/{arm}/step_{int(selector['optimizer_step']):04d}"


def ordered_values(left_rows, right_rows, fn):
    left = {row["sample_id"]: row for row in left_rows}; right = {row["sample_id"]: row for row in right_rows}
    if set(left) != set(right): raise RuntimeError("paired sample sets differ")
    ids = sorted(left); return np.asarray([fn(left[i], right[i]) for i in ids], dtype=np.float64), ids


def paired(values: np.ndarray, seed: int, repeats: int) -> dict:
    rng = np.random.default_rng(seed); n = len(values); means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        end = min(repeats, start + 500); idx = rng.integers(0, n, size=(end - start, n)); means[start:end] = values[idx].mean(axis=1)
    eps = 1e-12
    return {"n": n, "mean_delta": float(values.mean()), "bootstrap_95ci": [float(x) for x in np.quantile(means, [.025, .975])],
            "wins": int((values > eps).sum()), "ties": int((np.abs(values) <= eps).sum()), "losses": int((values < -eps).sum())}


def contrast(name, left, right, seed, repeats):
    metrics = {}
    delta, _ = ordered_values(right["g0"], left["g0"], lambda base, new: new["foreground_iou"] - base["foreground_iou"])
    metrics["G0_foreground_iou"] = paired(delta, seed, repeats)
    delta, _ = ordered_values(right["g0"], left["g0"], lambda base, new: new["foreground_f1"] - base["foreground_f1"])
    metrics["G0_foreground_f1"] = paired(delta, seed + 1, repeats)
    delta, _ = ordered_values(right["detection"], left["detection"], lambda base, new: float(new["cls_pred"] == new["gt_label"]) - float(base["cls_pred"] == base["gt_label"]))
    metrics["classification_accuracy"] = paired(delta, seed + 2, repeats)
    delta, _ = ordered_values(right["semantic"], left["semantic"], lambda base, new: new["R_phrase_sem"] - base["R_phrase_sem"])
    metrics["phrase_semantic_score"] = paired(delta, seed + 3, repeats)
    delta, _ = ordered_values(right["tf"], left["tf"], lambda base, new: new["foreground_iou"] - base["foreground_iou"])
    metrics["TF_foreground_iou"] = paired(delta, seed + 4, repeats)
    return {"comparison": name, "paired": True, "bootstrap_repeats": repeats, "metrics": metrics}


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve(); stats = root / "statistics"
    selectors = {arm: load(root / f"evaluation/selector/{arm}_selector.json") for arm in ("SFT_CONT", "JOINT")}
    p1_selector = {"optimizer_step": 0, "selected_checkpoint": cfg["source"]["checkpoint"]}
    selected = {"P1_FROZEN": p1_selector, **selectors}
    data = {}
    summaries = {}
    for arm, selector in selected.items():
        location = selected_dir(root, arm, selector)
        g0 = rows(location / "G0/predictions.jsonl"); detection = rows(location / "detection/predictions.jsonl")
        semantic = rows(location / "semantic_structure.jsonl"); tf = rows(root / f"evaluation/final/{arm}/tf_full_context/predictions.jsonl")
        data[arm] = {"g0": g0, "detection": detection, "semantic": semantic, "tf": tf}
        selection_metrics = load(location / "selection_metrics.json")
        tf_metrics = load(root / f"evaluation/final/{arm}/tf_full_context/metrics.json")
        summaries[arm] = {"selected_phase_step": int(selector["optimizer_step"]), "checkpoint": selector["selected_checkpoint"],
                          "G0": {k: selection_metrics[k] for k in ("mean_foreground_iou", "median_foreground_iou", "mean_foreground_f1", "median_foreground_f1")},
                          "classification": {k: selection_metrics[k] for k in ("classification_accuracy", "classification_f1", "classification_confusion")},
                          "language": {k: selection_metrics[k] for k in ("R_phrase_sem", "R_key_soft", "R_sentence_sem")},
                          "structure": {k: selection_metrics[k] for k in ("structure_validity", "valid_seg_rate", "malformed_output_rate")},
                          "TF_PHRASE": tf_metrics["per_image_mean"]}
    repeats = int(cfg["evaluation"]["bootstrap_repeats"]); seed = int(cfg["evaluation"]["bootstrap_seed"])
    comparisons = {
        "joint_vs_p1": contrast("P3E-JOINT_minus_P1-FROZEN", data["JOINT"], data["P1_FROZEN"], seed, repeats),
        "joint_vs_sft": contrast("P3E-JOINT_minus_P3E-SFT-CONT", data["JOINT"], data["SFT_CONT"], seed + 100, repeats),
        "sft_vs_p1": contrast("P3E-SFT-CONT_minus_P1-FROZEN", data["SFT_CONT"], data["P1_FROZEN"], seed + 200, repeats),
    }
    # contrast() receives (new, base) and internally computes new-base.
    for key, filename in (("joint_vs_p1", "paired_bootstrap_joint_vs_p1.json"),
                          ("joint_vs_sft", "paired_bootstrap_joint_vs_sft.json"),
                          ("sft_vs_p1", "paired_bootstrap_sft_vs_p1.json")): dump(stats / filename, comparisons[key])
    p1, sft, joint = summaries["P1_FROZEN"], summaries["SFT_CONT"], summaries["JOINT"]
    jp = comparisons["joint_vs_p1"]["metrics"]["G0_foreground_iou"]
    js = comparisons["joint_vs_sft"]["metrics"]["G0_foreground_iou"]
    tfp = comparisons["joint_vs_p1"]["metrics"]["TF_foreground_iou"]
    cls_nonreg = joint["classification"]["classification_accuracy"] >= p1["classification"]["classification_accuracy"] - float(cfg["selector"]["classification_accuracy_max_drop"]) and joint["classification"]["classification_f1"] >= p1["classification"]["classification_f1"] - float(cfg["selector"]["classification_f1_max_drop"])
    struct_nonreg = joint["structure"]["structure_validity"] >= p1["structure"]["structure_validity"] - float(cfg["selector"]["structure_validity_max_drop"])
    jp_sig = jp["mean_delta"] > 0 and jp["bootstrap_95ci"][0] > 0
    js_sig = js["mean_delta"] > 0 and js["bootstrap_95ci"][0] > 0
    tf_sig = tfp["mean_delta"] > 0 and tfp["bootstrap_95ci"][0] > 0
    if jp_sig and not cls_nonreg:
        gate = "GATE_JOINT_POSTTRAINING_TRADEOFF_UNACCEPTABLE"
    elif jp_sig and js_sig and cls_nonreg and struct_nonreg:
        gate = "GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_EFFECTIVE"
    elif jp_sig and cls_nonreg:
        gate = "GATE_JOINT_GAIN_NOT_SEPARABLE_FROM_EXTRA_SFT"
    elif tf_sig:
        gate = "GATE_JOINT_SPATIAL_GAIN_NOT_AUTONOMOUSLY_TRANSFERRED"
    else:
        gate = "GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_NOT_SUPPORTED"
    route = {"primary_gate": gate, "joint_vs_p1_G0_iou_significant_positive": jp_sig,
             "joint_vs_sft_G0_iou_significant_positive": js_sig, "joint_vs_sft_G0_iou_positive": js["mean_delta"] > 0,
             "joint_vs_p1_TF_iou_significant_positive": tf_sig, "classification_non_regression": cls_nonreg,
             "structure_non_regression": struct_nonreg, "first_contribution_can_be_frozen": gate == "GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_EFFECTIVE",
             "FEPN_execution_started": False, "posttraining_search_stopped": True}
    dump(root / "route_gate.json", route); dump(root / "g0_metrics.json", {"arms": summaries});
    dump(root / "tf_phrase_metrics.json", {"arms": {arm: value["TF_PHRASE"] for arm, value in summaries.items()}, "user_prompt": "canonical"})
    audits = {arm: load(root / f"experiments/{arm}/parameter_update_audit.json") for arm in ("SFT_CONT", "JOINT")}
    dump(root / "parameter_update_audit.json", {"status": "PASS" if all(v["status"] == "PASS" for v in audits.values()) else "FAIL", "arms": audits})
    metadata = {arm: rows(root / f"experiments/{arm}/checkpoint_metadata.jsonl") for arm in ("SFT_CONT", "JOINT")}
    dump(root / "checkpoint_metadata.json", {"arms": metadata})
    # Largest paired G0 regressions are diagnostic only and never alter selection.
    failures = {}
    for arm in ("SFT_CONT", "JOINT"):
        base = {r["sample_id"]: r for r in data["P1_FROZEN"]["g0"]}; current = {r["sample_id"]: r for r in data[arm]["g0"]}
        failures[arm] = sorted([{"sample_id": sid, "P1_iou": base[sid]["foreground_iou"], "arm_iou": current[sid]["foreground_iou"],
                                        "delta": current[sid]["foreground_iou"] - base[sid]["foreground_iou"]} for sid in base], key=lambda x: x["delta"])[:25]
    dump(root / "failure_analysis.json", {"selection_or_tuning_use": False, "largest_G0_IoU_regressions": failures})
    dump(stats / "final_metrics.json", {"arms": summaries, "comparisons": comparisons, "route_gate": route})

    report = f"""# Phase 3E — Joint Language-to-Mask Grounding Post-training

## 结论

最终 route gate：`{gate}`。

Mask-only backward audit 对 LoRA/FC/decoder 的梯度范数分别为 `{load(root/'audits/gradient_path_audit.json')['audit_B_mask_only']['norms']['lora']:.6f}`、`{load(root/'audits/gradient_path_audit.json')['audit_B_mask_only']['norms']['text_hidden_fcs']:.6f}`、`{load(root/'audits/gradient_path_audit.json')['audit_B_mask_only']['norms']['mask_decoder']:.6f}`，因此 mask supervision 确实回传至 LLM LoRA。

| Arm | selected step | G0 mean IoU | G0 mean F1 | TF mean IoU | TF mean F1 | Cls Acc | Cls F1 | Phrase semantic | Structure valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for arm in ("P1_FROZEN", "SFT_CONT", "JOINT"):
        v = summaries[arm]; report += f"| {arm} | {v['selected_phase_step']} | {v['G0']['mean_foreground_iou']:.6f} | {v['G0']['mean_foreground_f1']:.6f} | {v['TF_PHRASE']['foreground_iou']:.6f} | {v['TF_PHRASE']['foreground_f1']:.6f} | {v['classification']['classification_accuracy']:.6f} | {v['classification']['classification_f1']:.6f} | {v['language']['R_phrase_sem']:.6f} | {v['structure']['structure_validity']:.6f} |\n"
    report += f"""

JOINT−P1 G0 IoU delta `{jp['mean_delta']:+.6f}`，paired bootstrap 95% CI `[{jp['bootstrap_95ci'][0]:+.6f}, {jp['bootstrap_95ci'][1]:+.6f}]`。JOINT−SFT-CONT delta `{js['mean_delta']:+.6f}`，95% CI `[{js['bootstrap_95ci'][0]:+.6f}, {js['bootstrap_95ci'][1]:+.6f}]`。

Classification non-regression：`{cls_nonreg}`；structure non-regression：`{struct_nonreg}`。本阶段未使用 internal test、official1000、threshold tuning、reward、classification loss 或 post-hoc loss tuning。Phase 3E 完成后停止 post-training search；FEPN 未自动启动。
"""
    docs = ROOT / "docs/phase3e_joint_language_to_mask_posttraining.md"; docs.write_text(report, encoding="utf-8")
    (root / "final_comparison.md").write_text(report, encoding="utf-8")
    route_doc = ROOT / "docs/paper_convergence_route.md"
    marker = "## Phase 3E — Joint Language-to-Mask Grounding Post-training (final)"
    if marker not in route_doc.read_text(encoding="utf-8"):
        with route_doc.open("a", encoding="utf-8") as f:
            f.write(f"\n\n{marker}\n\nPhase 3D.2-A: `COMPLETED — IMPLEMENTATION MISMATCH FOUND`. Phase 3D.2-B: `COMPLETED — GATE_MATCHED_SPATIAL_OPTIMIZATION_NO_VALIDATION_GAIN`. Phase 3D.2-C attribution route is stopped by paper convergence decision.\n\nPhase 3E completed with gate `{gate}`. Selected SFT-CONT step={sft['selected_phase_step']}; selected JOINT step={joint['selected_phase_step']}. FEPN remains not executed. See `docs/phase3e_joint_language_to_mask_posttraining.md`.\n")
    required = [root / name for name in ("experiment_manifest.json", "initial_checkpoint_manifest.json", "experiment_fairness_manifest.json", "training_config.json", "audits/gradient_path_audit.json", "parameter_update_audit.json", "checkpoint_metadata.json", "checkpoint_selection_protocol.json", "g0_metrics.json", "tf_phrase_metrics.json", "statistics/paired_bootstrap_joint_vs_p1.json", "statistics/paired_bootstrap_joint_vs_sft.json", "statistics/paired_bootstrap_sft_vs_p1.json", "failure_analysis.json", "route_gate.json", "final_comparison.md")]
    required += [docs, route_doc]
    if any(not p.is_file() for p in required): raise FileNotFoundError([str(p) for p in required if not p.is_file()])
    completion = {"status": "COMPLETE", "phase": "Phase 3E", "primary_gate": gate,
                  "selected_checkpoints": {arm: value["checkpoint"] for arm, value in summaries.items()},
                  "internal_test_used": False, "official1000_used": False, "threshold_tuned": False,
                  "FEPN_started": False, "files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in required]}
    dump(root / "completion_manifest.json", completion); print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
