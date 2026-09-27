#!/usr/bin/env python3
"""Frozen A1 selected-checkpoint Official1000 evaluation, using the 6E.2 inference path."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6e2_official_finalize as official
from tools.phase4c_b import file_sha256
from tools.phase4e1 import compare, summarize

A1 = ROOT / "outputs/phase6h/A1_c_spatial_supervision"
OUT = ROOT / "outputs/phase6h/A1_official1000"
RECORDS = OUT / "predictions.jsonl"
BASELINE = ROOT / "outputs/phase6e2_c1_specific_r1/official1000_new_r1_records.jsonl"
DOC = ROOT / "docs/phase6h1_a1_official1000.md"


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def preflight() -> tuple[dict, list[dict]]:
    summary = json.loads((A1 / "summary.json").read_text())
    curve = json.loads((A1 / "per_epoch_metrics.json").read_text())
    checkpoint = A1 / "selected_checkpoint.pt"
    if summary["status"] != "COMPLETE" or len(curve) != 10:
        raise RuntimeError("A1 training or 10-epoch DEV selection incomplete")
    best = max(curve, key=lambda row: (row["dev_g0_mean_iou"], -row["epoch"]))
    if summary["selected_epoch"] != best["epoch"]:
        raise RuntimeError("A1 selector drift")
    if file_sha256(checkpoint) != summary["selected_checkpoint_sha256"]:
        raise RuntimeError("A1 selected checkpoint SHA256 drift")
    if summary["frozen_hash_before"] != summary["frozen_hash_after"]:
        raise RuntimeError("frozen model/source hash drift")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state["arm"] != "A1" or state["epoch"] != best["epoch"] or
            state["c1_sha256"] != official.C1_SHA or "aux_state" not in state):
        raise RuntimeError("A1 checkpoint provenance drift")
    if file_sha256(official.C1_CKPT) != official.C1_SHA:
        raise RuntimeError("C1 checkpoint SHA256 drift")
    baseline = official.rows(BASELINE)
    c1_rows = official.rows(official.E1_OUT / "c1_records.jsonl")
    expected_ids = [row["sample_id"] for row in baseline]
    if (len(expected_ids) != 1000 or len(set(expected_ids)) != 1000 or
            expected_ids != [row["sample_id"] for row in c1_rows]):
        raise RuntimeError("historical Official1000 baseline/C1 sample identities drift")
    if RECORDS.exists():
        existing = official.rows(RECORDS)
        if len(existing) > 1000 or [r["sample_id"] for r in existing] != expected_ids[:len(existing)]:
            raise RuntimeError("A1 Official1000 resume prefix drift")
        if not (OUT / "protocol.json").exists():
            raise RuntimeError("existing predictions lack frozen protocol")
        prior = json.loads((OUT / "protocol.json").read_text())
        if prior["checkpoint_sha256"] != summary["selected_checkpoint_sha256"]:
            raise RuntimeError("existing predictions use another checkpoint")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "phase6h1_a1_official1000_v1", "status": "FROZEN_BEFORE_INFERENCE",
        "authorization": "user explicitly requested A1 Official1000 after A1 completed",
        "arm": "A1", "selected_epoch": best["epoch"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": summary["selected_checkpoint_sha256"],
        "c1_sha256": official.C1_SHA,
        "inference_code": str(Path(official.__file__).resolve()),
        "population": 1000, "baseline_sample_ids_sha256": ids_hash(expected_ids),
        "baseline_records_sha256": file_sha256(BASELINE),
        "target": "all reference mask union", "mode": "canonical G0", "threshold_logit": 0.0,
        "seg_policy": "no [SEG] -> zero; multiple masks -> logit max union",
        "selection": "internal DEV selected checkpoint only; no Official1000 tuning/reselection",
        "comparison_limit": "historical 6E.2 main is a reference, not the unfinished matched A0 reproduction",
    }
    if (OUT / "protocol.json").exists():
        previous = json.loads((OUT / "protocol.json").read_text())
        if {k: v for k, v in previous.items() if k != "status"} != {k: v for k, v in protocol.items() if k != "status"}:
            raise RuntimeError("frozen Official1000 protocol drift")
    else:
        official.dump(OUT / "protocol.json", protocol)
    official.dump(OUT / "selector.json", {
        "status": "COMPLETE_SELECTED_FROZEN", "arm": "A1",
        "selected_epoch": best["epoch"],
        "selected_checkpoint_sha256": summary["selected_checkpoint_sha256"],
        "official1000_used": False, "internal_test_used": False})
    return summary, baseline


def evaluate(summary: dict, baseline: list[dict]) -> None:
    expected_ids = [row["sample_id"] for row in baseline]
    original_fake_dataset = official.matrix.fake_dataset

    def checked_fake_dataset(tokenizer, population):
        if population != "official1000":
            raise RuntimeError(f"unexpected population: {population}")
        full, indices = original_fake_dataset(tokenizer, population)
        actual_ids = [full.rows[i]["sample_id"] for i in indices]
        if actual_ids != expected_ids:
            raise RuntimeError("Official1000 manifest order/identity differs from frozen baseline")
        return full, indices

    official.matrix.fake_dataset = checked_fake_dataset
    official.OUT = OUT
    official.SELECTED = A1 / "selected_checkpoint.pt"
    official.RECORDS = RECORDS
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    official.evaluate(device)


def finalize(summary: dict, baseline: list[dict]) -> None:
    current = official.rows(RECORDS)
    expected_ids = [row["sample_id"] for row in baseline]
    if len(current) != 1000 or [row["sample_id"] for row in current] != expected_ids:
        raise RuntimeError("A1 Official1000 incomplete or unpaired")
    metrics = summarize(current)
    metrics["seg_trigger_rate"] = sum(bool(row["valid_q_seg"]) for row in current) / 1000
    base_metrics = summarize(baseline)
    base_metrics["seg_trigger_rate"] = sum(bool(row["valid_q_seg"]) for row in baseline) / 1000
    paired = compare(current, baseline, seed=3407)
    a0_audit_path = ROOT / "outputs/phase6h/A0_reproduction_audit.json"
    matched_path = ROOT / "outputs/phase6h/paired_A0_A1/summary.json"
    a0_audit = json.loads(a0_audit_path.read_text()) if a0_audit_path.exists() else None
    matched = json.loads(matched_path.read_text()) if matched_path.exists() else None
    result = {
        "schema": "phase6h1_a1_official1000_result_v1", "status": "COMPLETE",
        "checkpoint_sha256": summary["selected_checkpoint_sha256"],
        "selected_epoch": summary["selected_epoch"],
        "A1": metrics, "historical_phase6e2_reference": base_metrics,
        "paired_A1_minus_historical_phase6e2": paired,
        "sample_ids_sha256": ids_hash(expected_ids),
        "records_sha256": file_sha256(RECORDS),
        "comparison_limit": "Phase6H.1 A0 reproduction was not used as the Official1000 baseline; historical 6E.2 is contextual only.",
        "internal_A0_reproduction_status": None if a0_audit is None else a0_audit["status"],
        "internal_A1_vs_A0_decision": None if matched is None else matched["decision"],
    }
    official.dump(OUT / "results.json", result)
    lines = ["# Phase6H.1 A1 — Official1000", "",
             "用户在 A1 完成后授权对冻结的 A1 selected checkpoint 做一次 Official1000 canonical G0 测试。使用 Phase6E.2 原始 Official1000 推理路径；不按官方结果选择 epoch、调整 λ 或重训。", "",
             f"A1 选中 epoch **{summary['selected_epoch']}**；checkpoint SHA256 `{summary['selected_checkpoint_sha256']}`。1000 张 Fake 的 manifest 顺序与历史记录严格相等。", "",
             "| 指标 | A1 | 历史 6E.2 参考 | 差值 |", "|---|---:|---:|---:|",]
    for key, label in (("mean_foreground_iou", "mean FG IoU"), ("mean_foreground_f1", "mean FG F1"),
                       ("global_foreground_iou", "global FG IoU"), ("global_foreground_f1", "global FG F1"),
                       ("seg_trigger_rate", "SEG trigger rate")):
        lines.append(f"| {label} | {metrics[key]:.6f} | {base_metrics[key]:.6f} | {metrics[key]-base_metrics[key]:+.6f} |")
    iou, f1 = paired["foreground_iou"], paired["foreground_f1"]
    internal_note = ("A0 完整复现已通过，Phase6H.1 内部严格配对 A1−A0 mean IoU "
                     f"{matched['paired']['foreground_iou']['mean_difference']:+.6f}，"
                     f"95% CI [{matched['paired']['foreground_iou']['bootstrap_95_ci'][0]:+.6f}, "
                     f"{matched['paired']['foreground_iou']['bootstrap_95_ci'][1]:+.6f}]，"
                     f"预注册判定 **{matched['decision']}**。"
                     if a0_audit is not None and a0_audit["status"] == "PASS" and matched is not None
                     else "A0 匹配复现仍待完成；官方历史 6E.2 仅作背景参考。")
    lines += ["",
              f"严格配对 1000 张：IoU 平均差 {iou['mean_difference']:+.6f}，bootstrap 95% CI "
              f"[{iou['bootstrap_95_ci'][0]:+.6f}, {iou['bootstrap_95_ci'][1]:+.6f}]，"
              f"胜/平/负 {iou['wins']}/{iou['ties']}/{iou['losses']}；"
              f"F1 平均差 {f1['mean_difference']:+.6f}，95% CI "
              f"[{f1['bootstrap_95_ci'][0]:+.6f}, {f1['bootstrap_95_ci'][1]:+.6f}]。"
              "两个均值差的区间均跨 0；global 指标下降。", "",
              internal_note, "",
              "历史 6E.2 官方行只作背景参考；A0 未单独跑 Official1000，不能把 A1 与历史 6E.2 的官方差值直接当作 A1−A0 的因果效应。", "",
              "完整逐样本记录、冻结协议及配对统计见 `outputs/phase6h/A1_official1000/`。没有访问 internal test，也没有进入 Phase6H.2。", ""]
    DOC.write_text("\n".join(lines))
    official.dump(OUT / "status.json", {"status": "COMPLETE", "records": 1000,
                                        "checkpoint_sha256": summary["selected_checkpoint_sha256"]})
    print(json.dumps({"status": "COMPLETE", "mean_foreground_iou": metrics["mean_foreground_iou"],
                      "report": str(DOC)}), flush=True)


def main():
    try:
        summary, baseline = preflight()
        official.dump(OUT / "status.json", {"status": "RUNNING", "pid": os.getpid(),
                                            "checkpoint_sha256": summary["selected_checkpoint_sha256"]})
        evaluate(summary, baseline)
        finalize(summary, baseline)
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        official.dump(OUT / "status.json", {"status": "FAILED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
