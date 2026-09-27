#!/usr/bin/env python3
"""Fail-closed scoped evaluation of the validation-selected C2 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

BASE = ROOT / "outputs/phase6j0_c2"
OUT = BASE / "final_evaluation"
STATUS = OUT / "status.json"
CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
CFG = ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml"
OFFICIAL_MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest"
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"
REPORT = ROOT / "docs/final_evaluation_report.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(stage, command, status):
    if status is not None:
        status["stage"] = stage
        status["updated_at_utc"] = now()
        write(STATUS, status)
    log_path = OUT / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}).returncode
    require(code == 0, f"{stage} failed with exit code {code}; see {log_path}")


def fmt(value):
    return "-" if value is None else f"{float(value):.6f}"


def classification_row(label, value):
    n = int(value.get("n", value.get("num_samples", 0)))
    tn, fp = int(value["tn"]), int(value["fp"])
    tnr = tn / max(1, tn + fp)
    real_only = int(value.get("fake", n - tn - fp)) == 0
    return (f"| {label} | {n} | {fmt(value['accuracy'])} | {fmt(None if real_only else value['f1'])} | "
            f"{fmt(None if real_only else value.get('roc_auc'))} | {fmt(None if real_only else value['fake_recall'])} | "
            f"{fmt(tnr)} | {fmt(1 - tnr)} |")


def localization_row(label, condition, n, mean_iou, mean_f1, global_iou, global_f1):
    return (f"| {label} | {condition} | {n} | {fmt(mean_iou)} | {fmt(mean_f1)} | "
            f"{fmt(global_iou)} | {fmt(global_f1)} |")


def append_report(result):
    checkpoint_sha = result["selected_checkpoint_sha256"]
    internal = result["internal_test"]["modes"]["detection"]["classification_head"]
    internal = {**internal, "n": result["internal_test"]["samples"]}
    ood = result["classification_ood"]["datasets"]
    official = result["official1000"]["modes"]["G0"]
    lines = [
        "<!-- PHASE6J0_C2_BEGIN -->",
        "\n## C2 Pre-LN cross attention｜本轮阶段性评测\n",
        f"选模 checkpoint：`{checkpoint_sha}`（epoch {result['selected_epoch']}，step {result['selected_step']}）。"
        "C2 是 RINE query / CLIP patch K,V 的单层交叉注意力，分类使用 raw H2 阈值 0.5；"
        "从第 3,500 步的完整 checkpoint 恢复，仍用每卡 batch 10、双卡全局 batch 20。"
        "C2 未接入 R1 定位矫正器；历史 C1 为单卡训练，以下差异不是严格 matched 单变量因果效应。\n",
        "### 分类｜内部测试集\n",
        "| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        classification_row("C2-raw", internal),
        "\n### 分类｜外部 OOD\n",
    ]
    for name, title in (("aigi_holmes", "AIGI-Holmes TestSet"),
                        ("genimage", "GenImage held-out"),
                        ("loki", "LOKI 分类"), ("raise998", "RAISE998")):
        lines.extend([f"#### {title}\n", "| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|",
                      classification_row("C2-raw", ood[name]["classification_head"]), ""])
    lines.extend([
        "RAISE998 只有 Real，F1、ROC-AUC、Fake recall 不作为双类指标解释。"
        "AIGI-Holmes 的既有 internal-TRAIN exact 重叠审计仍适用，未删除重叠样本。\n",
        "### 定位｜SynthScars Official1000\n",
        "| 模型 | 条件 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
        localization_row("C2-raw", "G0", official["num_gt_fake"],
                         official["per_image_mean"]["foreground_iou"],
                         official["per_image_mean"]["foreground_f1"],
                         official["global_pixel"]["foreground_iou"],
                         official["global_pixel"]["foreground_f1"]),
    ])
    lines.extend([
        "本轮只评测内部测试集、外部 OOD 分类和 Official1000 定位；C2 外部 OOD 定位留待接入 R1 后评测。"
        "逐图预测、manifest 哈希、checkpoint 身份和分类 GenImage 分生成器指标见 "
        "[C2 阶段性评测汇总](../outputs/phase6j0_c2/final_evaluation/results.json)。"
        "Official1000 在内部 validation 选模之后运行；外部 OOD 未用于选模或阈值调整。",
        "<!-- PHASE6J0_C2_END -->",
    ])
    original = REPORT.read_text()
    require("<!-- PHASE6J0_C2_BEGIN -->" not in original,
            "C2 report section already exists; refusing duplicate append")
    original_lines = original.splitlines()
    for index, line in enumerate(original_lines):
        if line.startswith("更新："):
            original_lines[index] = f"更新：{now()[:10]}。数值保留六位小数；`-` 表示缺少可报告结果或指标不适用。每个数据集单独一表。"
            break
    original = "\n".join(original_lines) + "\n"
    temporary = REPORT.with_suffix(".md.tmp")
    temporary.write_text(original.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    os.replace(temporary, REPORT)


def main():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    step, epoch = int(state["optimizer_step"]), int(state["epoch"])
    del state
    training = read_json(BASE / "training/training_state.json")
    require(training["optimizer_step"] == 5000 and training["epoch"] == 10,
            "C2 training must complete before final evaluation")
    official_manifest = OFFICIAL_MANIFEST / "test_combined.jsonl"
    require(official_manifest.is_file(), "Official1000 manifest missing")
    protocol = {"schema": "phase6j0_c2_scoped_evaluation_protocol_v1", "status": "FROZEN",
                "selected_checkpoint": str(CKPT.resolve()), "selected_checkpoint_sha256": sha(CKPT),
                "selected_epoch": epoch, "selected_step": step,
                "selector": "minimum internal-validation total loss",
                "scope": ["internal_test", "classification_ood", "official1000_G0"],
                "external_localization_ood": "DEFERRED_FOR_C2_PLUS_R1",
                "classification_threshold": 0.5, "mask_logit_threshold": 0.0,
                "official1000": {"manifest": str(official_manifest.resolve()),
                                 "sha256": sha(official_manifest), "expected_n": 1000,
                                 "condition": "G0"},
                "no_test_or_ood_model_selection": True}
    OUT.mkdir(parents=True, exist_ok=True)
    existing = OUT / "protocol.json"
    if existing.exists():
        require(read_json(existing) == protocol, "C2 final-evaluation protocol drift")
    else:
        write(existing, protocol)
    status = {"status": "RUNNING", "started_at_utc": now(), "stage": "PREFLIGHT"}
    write(STATUS, status)
    try:
        run("OFFICIAL1000_G0", [PY, "scripts/phase6j0_c2_evaluate.py", "--config", str(CFG),
                               "--checkpoint", str(CKPT), "--output-dir", str(OUT / "official1000"),
                               "--manifest-dir", str(OFFICIAL_MANIFEST), "--split", "test", "--device", "cuda:0",
                               "--modes", "G0", "--expected-step", str(step),
                               "--expected-epoch", str(epoch), "--generation-batch-size", "1",
                               "--skip-spatial-save"], status)
        official = read_json(OUT / "official1000/summary.json")
        require(official["samples"] == 1000 and official["modes"]["G0"]["num_gt_fake"] == 1000,
                "Official1000 full-N check failed")
        official_ids = [str(row["sample_id"]) for row in read_jsonl(official_manifest)]
        prediction_ids = [str(row["sample_id"]) for row in
                          read_jsonl(OUT / "official1000/G0/predictions.jsonl")]
        require(prediction_ids == official_ids, "Official1000 ID/order check failed")
        require(official["checkpoint_file_sha256"] == protocol["selected_checkpoint_sha256"],
                "Official1000 checkpoint drift")
        run("CLASSIFICATION_OOD", [PY, "scripts/phase6j0_c2_full_ood.py", "--mode", "supervisor",
                                   "--batch-size", "8"], status)
        ood = read_json(OUT / "classification_ood/results.json")
        require(ood["status"] == "COMPLETE" and
                all(ood["datasets"][name]["classification_head"]["n"] == count
                    for name, count in (("aigi_holmes", 99999), ("genimage", 100000),
                                        ("loki", 2217), ("raise998", 998))),
                "Classification OOD full-N check failed")

        internal_test = read_json(BASE / "evaluation/test/summary.json")
        require(internal_test["samples"] == 2208, "Internal test population drift")
        require(internal_test["checkpoint_file_sha256"] == protocol["selected_checkpoint_sha256"],
                "Internal checkpoint drift")
        result = {"schema": "phase6j0_c2_scoped_evaluation_v1", "status": "COMPLETE",
                  "selected_checkpoint_sha256": protocol["selected_checkpoint_sha256"],
                  "selected_step": step, "selected_epoch": epoch,
                  "strictly_matched_to_historical_C1": False,
                  "internal_test": internal_test,
                  "official1000": official, "classification_ood": ood,
                  "external_localization_ood": "DEFERRED_FOR_C2_PLUS_R1", "protocol": protocol}
        write(OUT / "results.json", result)
        append_report(result)
        status.update(status="COMPLETE", stage="REPORT_APPENDED", completed_at_utc=now(),
                      result=str((OUT / "results.json").resolve()), report=str(REPORT.resolve()))
    except BaseException as exc:
        status.update(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                      updated_at_utc=now())
        raise
    finally:
        write(STATUS, status)


if __name__ == "__main__":
    main()
