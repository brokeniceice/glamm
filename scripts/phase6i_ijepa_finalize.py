#!/usr/bin/env python3
"""Internal DEV only: freeze selected checkpoints, replay and pair A1 against A0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6i_ijepa_staged as trial
from scripts import phase4hc_direct_utility_arms as hc
from scripts.phase4gf_formal_localization import load_dev
from tools.phase4c_b import file_sha256
from tools.phase4e1 import compare
from tools.phase4f import Phase4FStore, load_evidence_source, load_sam_runtime

BASE = trial.ROOT / "outputs/phase6i_ijepa_r1"


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def replay():
    arm = trial.ARM
    out = BASE / arm
    dest = out / "internal_dev_records.jsonl"
    if dest.exists():
        raise RuntimeError("internal DEV replay already exists")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dev = load_dev("g0")
    ids = dev["sample_ids"]
    store = Phase4FStore(trial.CFG, "val")
    if store.sample_ids != ids:
        raise RuntimeError("internal DEV population drift")
    cache = trial.e2.load_c1_cache("val", ids)
    jstore = trial.jepa_store("val", ids)
    model, _ = hc.load_utility("a2", device)
    rectifier = trial.make_rectifier(device)
    utility_state, selector = trial.selected_state("joint", "utility_state")
    rectifier_state, _ = trial.selected_state("joint", "rectifier_state")
    model.load_state_dict(utility_state, strict=True)
    rectifier.load_state_dict(rectifier_state, strict=True)
    sam = load_sam_runtime(trial.CFG, device)
    source = load_evidence_source(trial.CFG, "forensic_rect", device)
    metrics, records = trial.evaluate(model, rectifier, sam, source, store, dev, cache, jstore, device)
    temp = dest.with_suffix(".tmp")
    with temp.open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, dest)
    trial.dump(out / "internal_dev_summary.json", {
        "status": "COMPLETE", "arm": arm, "population": "internal DEV canonical G0", "metrics": metrics,
        "records": str(dest), "records_sha256": file_sha256(dest),
        "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
        "official1000_accessed": False, "external_ood_accessed": False,
    })


def paired():
    summaries = {arm: json.loads((BASE / arm / "internal_dev_summary.json").read_text()) for arm in ("A0", "A1")}
    for arm, summary in summaries.items():
        if summary["status"] != "COMPLETE" or file_sha256(Path(summary["records"])) != summary["records_sha256"]:
            raise RuntimeError(f"{arm} internal DEV replay incomplete")
    a0, a1 = (rows(BASE / arm / "internal_dev_records.jsonl") for arm in ("A0", "A1"))
    dev = load_dev("g0")
    ids = dev["sample_ids"]
    if [x["sample_id"] for x in a0] != ids or [x["sample_id"] for x in a1] != ids:
        raise RuntimeError("paired DEV sample ID/order mismatch")
    initials = [json.loads((BASE / arm / "rectifier/provenance.json").read_text())["initial_state_sha256"] for arm in ("A0", "A1")]
    if initials[0] != initials[1]:
        raise RuntimeError("A0/A1 Rectifier initialization mismatch")
    epoch0 = [json.loads((BASE / arm / "rectifier/history.json").read_text())[0]["dev_g0_mean_iou"] for arm in ("A0", "A1")]
    if abs(epoch0[0] - epoch0[1]) > 1e-10:
        raise RuntimeError("A0/A1 zero-gate initial final-mask mismatch")
    stats = compare(a1, a0)
    small_indices = [i for i, mask in enumerate(dev["original_masks"])
                     if float(mask.float().mean()) <= 0.05]
    low_coverage_indices = []
    for i, geometry in enumerate(dev["clip_geometries"]):
        rh, rw = geometry["resized_hw"]
        top, left, bottom, right = geometry["crop_box_yxyx"]
        coverage = ((bottom - top) * (right - left)) / (rh * rw)
        if coverage < 0.80:
            low_coverage_indices.append(i)
    subgroups = {}
    for name, indices in (("gt_area_at_most_5_percent", small_indices),
                          ("clip_crop_coverage_below_80_percent", low_coverage_indices)):
        subgroups[name] = {"n": len(indices), "paired": compare([a1[i] for i in indices], [a0[i] for i in indices])
                           if indices else None}
    mean = stats["foreground_iou"]
    global_delta = summaries["A1"]["metrics"]["global_foreground_iou"] - summaries["A0"]["metrics"]["global_foreground_iou"]
    passed = mean["mean_difference"] >= 0.010 and mean["bootstrap_95_ci"][0] > 0 and global_delta >= -0.005
    result = {"status": "COMPLETE_STOP", "comparison": "A1_IJEPA_minus_A0_duplicate_F24",
              "population": "internal DEV canonical G0", "n": len(ids),
              "initial_rectifier_state_sha256": initials[0], "epoch0_mean_fg_iou": epoch0[0],
              "A0": summaries["A0"]["metrics"],
              "A1": summaries["A1"]["metrics"], "paired": stats,
              "secondary_subgroups": subgroups,
              "global_foreground_iou_difference": global_delta,
              "preregistered_gate": {"mean_iou_difference_min": 0.010,
                                     "paired_mean_iou_ci_lower_gt_zero": True,
                                     "global_iou_difference_min": -0.005},
              "gate_passed": passed, "official1000_accessed": False,
              "external_ood_accessed": False, "classification_changed": False}
    trial.dump(BASE / "internal_dev_paired.json", result)
    label = "PASS: 允许另行设计后续评测" if passed else "STOP: 内部证据不足，不升级"
    doc = trial.ROOT / "docs/phase6i_ijepa_r1_internal_results.md"
    doc.write_text(
        "# Phase6I I-JEPA 附加证据：internal DEV 结果\n\n"
        "状态：**COMPLETE STOP**。本轮未访问 Official1000、internal test 或外部 OOD。\n\n"
        "| 模型 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {arm} | {summary['metrics']['n']} | {summary['metrics']['mean_foreground_iou']:.6f} | "
            f"{summary['metrics']['mean_foreground_f1']:.6f} | {summary['metrics']['global_foreground_iou']:.6f} | "
            f"{summary['metrics']['global_foreground_f1']:.6f} |"
            for arm, summary in summaries.items()
        ) + "\n\n"
        f"A1−A0 mean FG IoU = **{mean['mean_difference']:+.6f}**，逐图 bootstrap 95% CI "
        f"`[{mean['bootstrap_95_ci'][0]:+.6f}, {mean['bootstrap_95_ci'][1]:+.6f}]`；"
        f"胜/平/负 = {mean['wins']}/{mean['ties']}/{mean['losses']}。"
        f"Global FG IoU 差 = **{global_delta:+.6f}**。\n\n"
        f"预注册判定：**{label}**。门槛为 mean IoU 差 ≥ +0.010、配对 CI 下界 > 0、"
        "global IoU 差 ≥ −0.005，三者须同时满足。\n\n"
        "实验结构与数据边界见 [协议](phase6i_ijepa_r1_internal_protocol.md)；"
        "小面积掩码（GT ≤ 5%）与低裁剪覆盖（< 80%）仅作预定次级分组，数值见 JSON，不参与主门槛。"
        "逐图结果和哈希见 [`internal_dev_paired.json`](../outputs/phase6i_ijepa_r1/internal_dev_paired.json)。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("replay", "paired"))
    args = p.parse_args()
    replay() if args.mode == "replay" else paired()


if __name__ == "__main__":
    main()
