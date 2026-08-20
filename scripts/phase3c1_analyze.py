#!/usr/bin/env python3
"""Aggregate Phase 3C.1 validation evidence and freeze the route gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2a_final_evaluate import file_sha256
from tools.phase3c1 import SOURCES, paired_statistics


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def significant(stat):
    return float(stat["bootstrap_95_ci"][0]) > 0


def stable(stat):
    return float(stat["median_difference"]) > 0 and int(stat["wins"]) > int(stat["losses"])


def main():
    config = yaml.safe_load((ROOT / "configs/phase3c1_spatial_probe.yaml").read_text())
    out = (ROOT / config["experiment"]["output_root"]).resolve()
    persistent = set(json.loads((out / "audit/persistent476_ids.json").read_text())["sample_ids"])
    audits, geometries, hashes = {}, {}, {}
    training, selected, val_metrics, persistent_metrics, controls = {}, {}, {}, {}, {}
    normal = {}
    available = []
    for source in SOURCES:
        audit_path = out / "audit" / f"{source}.json"
        metrics_path = out / "validation" / source / "metrics.json"
        if not audit_path.exists() or not metrics_path.exists():
            continue
        available.append(source)
        audit = json.loads(audit_path.read_text()); audits[source] = audit
        geometries[source] = {
            "preprocessing": audit["preprocessing"], "example": audit["geometry_example"],
            "feature_shape": audit["feature_shape"], "feature_dtype": audit["feature_dtype"],
        }
        hashes[source] = {
            "before": audit["parameter_hash_before"], "after": audit["parameter_hash_after"],
            "exact": audit["parameter_hash_exact"], "regression": audit["regression"],
        }
        training[source] = json.loads((out / "probes" / source / "training_manifest.json").read_text())
        ckpt = out / "probes" / source / "selected.pt"
        selected[source] = {"path": str(ckpt), "sha256": file_sha256(ckpt),
                            "epoch": training[source]["selected_epoch"]}
        metrics = json.loads(metrics_path.read_text())
        val_metrics[source] = metrics["all_val_fake"]
        persistent_metrics[source] = metrics["persistent476"]
        controls[source] = json.loads((out / "statistics" / source / "negative_controls.json").read_text())
        normal[source] = rows(out / "validation" / source / "normal_predictions.jsonl")
    if set(available) != set(SOURCES):
        raise RuntimeError(f"analysis requires all sources; available={available}")
    provenance = {
        "phase": "Phase 3C.1 — Frozen Spatial Evidence Probe",
        "repository": str(ROOT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "device": "cuda:1", "seed": 3407,
        "P1_checkpoint": config["base_checkpoint"],
        "route_selection_inputs": ["internal val Fake 1106", "fixed Phase3C.0 persistent 476"],
        "prohibited_route_inputs": ["internal test", "official1000", "RAISE", "LOKI"],
        "probe_only_training": True,
        "forbidden_training_or_modification": [
            "P1/GLaMM", "LLM/LoRA", "CLIP", "SAM image/prompt/mask modules",
            "NPR/SRM/FOCAL backbones", "fusion", "GRPO",
        ],
    }
    dump(out / "provenance.json", provenance)
    dump(out / "feature_source_audit.json", audits)
    dump(out / "geometry_audit.json", geometries)
    dump(out / "frozen_backbone_hashes.json", hashes)
    dump(out / "probe_training_manifest.json", training)
    dump(out / "selected_checkpoints.json", selected)
    dump(out / "val_metrics.json", val_metrics)
    dump(out / "persistent476_metrics.json", persistent_metrics)
    dump(out / "negative_control_statistics.json", controls)

    by_source = {
        source: {row["sample_id"]: row for row in normal[source] if row["sample_id"] in persistent}
        for source in SOURCES
    }
    comparisons = [(expert, base) for expert in ("npr", "srm", "focal") for base in ("sam", "clip")]
    cross = {}
    ordered_ids = [row["sample_id"] for row in normal["sam"] if row["sample_id"] in persistent]
    if len(ordered_ids) != 476:
        raise RuntimeError(f"persistent comparison population mismatch: {len(ordered_ids)}")
    for left, right in comparisons:
        stat = paired_statistics(
            [by_source[left][sid]["foreground_iou"] for sid in ordered_ids],
            [by_source[right][sid]["foreground_iou"] for sid in ordered_ids],
        )
        stat.update({"left": left, "right": right, "population": "persistent476",
                     "decodability_comparison_not_architecture_matched_causal_test": True,
                     "outlier_stability": stable(stat)})
        cross[f"{left}_vs_{right}"] = stat
    dump(out / "cross_source_statistics.json", cross)

    self_pass = {}
    for source in SOURCES:
        source_stats = controls[source]
        self_pass[source] = all(
            significant(source_stats[name]["persistent476"])
            and stable(source_stats[name]["persistent476"])
            for name in ("spatial_shuffle", "sample_shuffle", "global_broadcast")
        )
    existing = [source for source in ("sam", "clip") if source in available]
    best_existing = max(existing, key=lambda value: persistent_metrics[value]["mean_foreground_iou"])
    complementary = []
    for source in ("npr", "srm", "focal"):
        stat = cross[f"{source}_vs_{best_existing}"]
        if self_pass[source] and significant(stat) and stable(stat):
            complementary.append(source)
    mixed = []
    for source in ("npr", "srm", "focal"):
        stat = cross[f"{source}_vs_{best_existing}"]
        balanced = stat["wins"] / stat["n"] >= 0.30 and stat["losses"] / stat["n"] >= 0.30
        if self_pass[source] and self_pass[best_existing] and not significant(stat) and balanced:
            mixed.append(source)
    forensic_signal = [source for source in ("npr", "srm", "focal") if self_pass[source]]
    existing_signal = [source for source in existing if self_pass[source]]
    if complementary:
        gate = "GATE_FORENSIC_SPATIAL_COMPLEMENTARITY_SUPPORTED"
    elif mixed:
        gate = "GATE_MIXED_SPATIAL_EVIDENCE"
    elif forensic_signal:
        gate = "GATE_SPATIAL_SIGNAL_PRESENT_NOT_FORENSIC_SPECIFIC"
    elif existing_signal:
        gate = "GATE_EXISTING_GLAMM_SPATIAL_SIGNAL_UNDERUSED"
    else:
        gate = "GATE_SPATIAL_PREFLIGHT_NOT_SUPPORTED"
    if complementary:
        confirmatory_sources = list(dict.fromkeys([best_existing, *complementary]))
    elif mixed:
        confirmatory_sources = list(dict.fromkeys([best_existing, *mixed]))
    elif forensic_signal:
        confirmatory_sources = list(dict.fromkeys([best_existing, *forensic_signal]))
    elif existing_signal:
        confirmatory_sources = [best_existing]
    else:
        confirmatory_sources = []
    route = {
        "status": "FROZEN_ON_INTERNAL_VAL_FAKE", "route_selection_population": "internal val Fake only",
        "persistent_population": 476, "selected_gate": gate, "best_existing_glamm_source": best_existing,
        "self_negative_control_pass": self_pass, "existing_spatial_signal": existing_signal,
        "forensic_spatial_signal": forensic_signal, "complementary_sources": complementary,
        "mixed_sources": mixed,
        "confirmatory_sources_authorized_after_gate_freeze": confirmatory_sources,
        "significance_rule": "paired IoU bootstrap 95% CI lower bound > 0",
        "outlier_rule": "median paired delta > 0 and wins > losses",
        "confirmatory_not_run_before_gate_freeze": True,
        "authorization_boundary": (
            "diagnostic gate only; do not modify GLaMM, add fusion/decoder, tune P1/B1, or start GRPO"
        ),
    }
    dump(out / "route_gate.json", route)
    report = f"""# Phase 3C.1 — Frozen Spatial Evidence Probe

## 冻结协议

本阶段仅在 frozen spatial feature 上训练 `Conv2d(C,1,1,bias=True)`。P1、LLM/LoRA、CLIP、SAM、NPR、SRM、FOCAL 均保持 `eval + no_grad`，所有 backbone 参数哈希前后 exact。route selection 只使用 internal val Fake 1106；persistent-476 membership 完全继承 Phase 3C.0。

## Validation 结果

| Source | all-val mean FG IoU | persistent mean FG IoU | persistent median | persistent mean F1 |
|---|---:|---:|---:|---:|
"""
    for source in SOURCES:
        report += (
            f"| {source.upper()} | {val_metrics[source]['mean_foreground_iou']:.6f} | "
            f"{persistent_metrics[source]['mean_foreground_iou']:.6f} | "
            f"{persistent_metrics[source]['median_foreground_iou']:.6f} | "
            f"{persistent_metrics[source]['mean_foreground_f1']:.6f} |\n"
        )
    report += f"""

## Route gate

`{gate}`

- best existing GLaMM spatial probe：`{best_existing}`。
- 通过自身三项 negative controls：{', '.join(k for k,v in self_pass.items() if v) or 'none'}。
- forensic complementary sources：{', '.join(complementary) or 'none'}。

Linear probe 优于 shuffle controls 只说明 feature 中存在可线性解码的空间信息；forensic probe 表现更好也不等于接入 GLaMM 后必然改善 G0。跨 source 是 decodability comparison，不是严格 architecture-matched causal test。

## 停止边界

Phase 3C.1 route gate 冻结后停止。本报告不授权自动新增 NPR/SRM/FOCAL fusion、decoder、GLaMM architecture 修改、P1/B1 tuning 或 GRPO。
"""
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "final_report.md").write_text(report, encoding="utf-8")
    (out / "reports/final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(route, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
