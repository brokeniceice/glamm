#!/usr/bin/env python3
"""Detached two-GPU classification evaluation of public LE + aligned Stage-2 head."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
MODEL_ROOT = ROOT / "checkpoints/legion_public_le_stage2_aligned"
MODEL = MODEL_ROOT / "final_model"
IDENTITY = MODEL_ROOT / "final_model_identity.json"
OUT = ROOT / "outputs/legion_public_le_stage2_evaluation"
WORKER = ROOT / "scripts/legion_retrained_match_classification.py"
STATUS = OUT / "status.json"
MODEL_NAME = "legion_public_le_stage2_aligned"
MODEL_SHA = "274aafcc01a81cf1848ecf99db65578c42b2b758b9a9417eb625e8d007cc658b"
MANIFESTS = {
    "internal": ("datasets/Internal2208/manifests/eval_manifest.jsonl", 2208, "fbc2d422c45fc871e06ed82d7b4c8fab88f1ed9e83d523a56591c93173ddd174"),
    "aigi_holmes": ("datasets/AIGI-Holmes/manifests/eval_manifest.jsonl", 99999, "380325bc0cbba1e80044d0d2dbb40ff2f788865288bb47402c1d4b7967431982"),
    "genimage": ("datasets/GenImage/manifests/eval_manifest.jsonl", 100000, "f7844320a3d358e62cb709a13fc1b10f10786a59c400febd26bd4f02091cbcf7"),
    "loki": ("datasets/LOKI/manifests/classification_eval_manifest.jsonl", 2217, "9376271a34603ff0ebcb7b21d9fc9a9219d7e28e4d350ffff6f8612d2a953c17"),
    "raise998": ("datasets/RAISE/manifests/eval_manifest.jsonl", 998, "d792f5e684abc42cbddd8466019576a2b6ee9fd53c7fcbe5354322a94caa4e49"),
}
JOBS = [("gpu1_internal_genimage", 1, ["internal", "genimage"]),
        ("gpu2_aigi_loki_raise", 2, ["aigi_holmes", "loki", "raise998"])]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def verify_model() -> dict:
    if not MODEL.is_dir() or not IDENTITY.is_file():
        raise FileNotFoundError("Completed Stage-2 model or identity record is missing")
    train = json.loads((MODEL_ROOT / "status.json").read_text())
    if train["status"] != "COMPLETE":
        raise RuntimeError("Stage-2 training is not complete")
    identity = json.loads(IDENTITY.read_text())
    if identity["canonical_sha256"] != MODEL_SHA or identity["root"] != str(MODEL.resolve()):
        raise RuntimeError("Final model identity drift")
    files = []
    for expected in identity["files"]:
        item = MODEL / expected["path"]
        if not item.is_file() or item.stat().st_size != expected["bytes"] or sha(item) != expected["sha256"]:
            raise RuntimeError(f"Final model file drift: {item}")
        files.append(expected)
    canonical = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if canonical != MODEL_SHA:
        raise RuntimeError("Final model canonical hash drift")
    index = json.loads((MODEL / "pytorch_model.bin.index.json").read_text())
    if len([key for key in index["weight_map"] if key.startswith("prediction_head.")]) != 4:
        raise RuntimeError("Stage-2 classification head missing from final model")
    return identity


def verify_manifests() -> dict:
    result = {}
    for name, (relative, expected_n, expected_sha) in MANIFESTS.items():
        path = ROOT / relative
        if sha(path) != expected_sha:
            raise RuntimeError(f"Frozen manifest hash drift: {name}")
        seen = set()
        labels = {0: 0, 1: 0}
        count = 0
        for row in rows(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise RuntimeError(f"Duplicate sample ID in {name}: {sample_id}")
            seen.add(sample_id)
            labels[1 if row["label"] == "Fake" else 0] += 1
            count += 1
        if count != expected_n:
            raise RuntimeError(f"Frozen manifest count drift: {name}")
        result[name] = {"path": str(path.resolve()), "sha256": expected_sha,
                        "n": count, "real": labels[0], "fake": labels[1]}
    return result


def launch(name: str, gpu: int, datasets: list[str]) -> tuple[subprocess.Popen, object]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    })
    command = [str(PYTHON), str(WORKER), "--datasets", *datasets, "--device", "cuda:0",
               "--batch-size", "32", "--model-dir", str(MODEL), "--identity", str(IDENTITY),
               "--output-root", str(OUT / "classification"), "--model-name", MODEL_NAME]
    path = OUT / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("a", buffering=1)
    log.write(f"[{now()}] COMMAND {json.dumps(command)}\n")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def verify_dataset(name: str, manifest: dict) -> dict:
    dest = OUT / "classification" / name
    result = json.loads((dest / "results.json").read_text())
    if (result["status"] != "COMPLETE" or result["model"] != MODEL_NAME
            or result["dataset"] != name
            or result["manifest_sha256"] != manifest["sha256"]
            or result["checkpoint"]["canonical_sha256"] != MODEL_SHA
            or result["checkpoint"]["path"] != str(MODEL.resolve())):
        raise RuntimeError(f"Result provenance mismatch: {name}")
    predictions = dest / "predictions.jsonl"
    source = rows(Path(manifest["path"]))
    observed = rows(predictions)
    labels = []
    scores = []
    for index, (expected, actual) in enumerate(zip(source, observed)):
        gt = 1 if expected["label"] == "Fake" else 0
        if actual["sample_id"] != expected["sample_id"] or actual["gt"] != gt:
            raise RuntimeError(f"Prediction order/label mismatch: {name} row {index}")
        score = float(actual["prob_fake"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"Invalid probability: {name} row {index}")
        if actual["pred"] != ("fake" if score >= 0.5 else "real"):
            raise RuntimeError(f"Prediction threshold mismatch: {name} row {index}")
        labels.append(gt)
        scores.append(score)
    if len(labels) != manifest["n"]:
        raise RuntimeError(f"Prediction count mismatch: {name}")
    try:
        next(source)
        raise RuntimeError(f"Extra manifest rows: {name}")
    except StopIteration:
        pass
    try:
        next(observed)
        raise RuntimeError(f"Extra prediction rows: {name}")
    except StopIteration:
        pass
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int8)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    m = result["metrics"]
    expected_metrics = {"n": len(y), "real": int((y == 0).sum()),
                        "fake": int((y == 1).sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if any(m[key] != value for key, value in expected_metrics.items()):
        raise RuntimeError(f"Confusion matrix drift: {name}")
    values = {"accuracy": (tp + tn) / len(y),
              "precision": tp / max(1, tp + fp),
              "recall": tp / max(1, tp + fn),
              "specificity_tnr": tn / max(1, tn + fp),
              "fpr": fp / max(1, tn + fp),
              "f1": 2 * tp / max(1, 2 * tp + fp + fn),
              "brier": float(np.mean((p - y) ** 2))}
    if len(set(labels)) == 2:
        values["roc_auc"] = float(roc_auc_score(y, p))
        values["auprc"] = float(average_precision_score(y, p))
    if any(not math.isclose(m[key], value, rel_tol=1e-10, abs_tol=1e-10)
           for key, value in values.items()):
        raise RuntimeError(f"Metric recomputation drift: {name}")
    if name == "genimage" and len(m["per_source"]) != 8:
        raise RuntimeError("GenImage generator breakdown incomplete")
    return result


def render(results: dict, protocol: dict) -> str:
    order = ["internal", "aigi_holmes", "genimage", "loki", "raise998"]
    names = {"internal": "Internal2208", "aigi_holmes": "AIGI-Holmes TestSet",
             "genimage": "GenImage held-out", "loki": "LOKI classification", "raise998": "RAISE998"}
    lines = ["# Public LEGION-LE + aligned Stage-2：分类测试与外部 OOD", "",
             "状态：**COMPLETE**。本轮只评测冻结的第二阶段分类头；公开 LE 的定位/解释权重不变。",
             "", "| 数据集 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in order:
        m = results[name]["metrics"]
        number = lambda value: "-" if value is None else f"{value:.6f}"
        real_only = name == "raise998"
        lines.append("| " + " | ".join([
            names[name], str(m["n"]), number(m["accuracy"]),
            "-" if real_only else number(m["f1"]),
            "-" if real_only else number(m.get("roc_auc")),
            "-" if real_only else number(m["recall"]),
            number(m["specificity_tnr"]), number(m["fpr"]),
        ]) + " |")
    lines += ["", "RAISE998 只有 Real；F1、ROC-AUC 与 Fake recall 不作双类解释。GenImage 的八个生成器分组指标见结果 JSON。", "",
              "## 评测口径", "",
              "- 模型：作者公开 `legion_LE` 第一阶段权重 + 本项目按 `legion-retrained` 第二阶段配方训练的分类头；冻结 checkpoint SHA256：`" + MODEL_SHA + "`。",
              "- 两张卡并行推理：卡 1 跑 Internal2208、GenImage；卡 2 跑 AIGI-Holmes、LOKI、RAISE998。分类阈值固定为 Fake probability `>= 0.5`，Real=1/Fake=0 为模型内部标签，汇总中 Fake 为正类。", 
              "- 所有结果已核对冻结 manifest SHA256、样本数、唯一性、逐样本 ID 顺序、GT 标签、预测阈值和重新计算的总体指标；没有用测试集或外部 OOD 调阈值、改权重或选模。",
              "- 冻结泄漏审计为 **BLOCKED_OVERLAP**，继续评测 override 为 **ACTIVE**。AIGI-Holmes 与 internal-TRAIN 有 779 个 exact SHA256 重叠，保留在完整 TestSet 中。外部清单合计 793 对 pHash 近重叠，AIGI-Holmes 与 GenImage 之间另有 35 对 pHash 近重叠。",
              "", "## 复核入口", "",
              "- [汇总 JSON](../outputs/legion_public_le_stage2_evaluation/results.json)",
              "- [冻结评测协议](../outputs/legion_public_le_stage2_evaluation/protocol.json)",
              "- [训练记录](legion_public_le_stage2_aligned.md)",
              "- [最终模型](../checkpoints/legion_public_le_stage2_aligned/final_model)", ""]
    return "\n".join(lines)


def main() -> None:
    old = json.loads(STATUS.read_text()) if STATUS.exists() else None
    if old and old.get("status") == "COMPLETE":
        return
    if not PYTHON.is_file() or not WORKER.is_file():
        raise FileNotFoundError("Classification runtime missing")
    verify_model()
    manifests = verify_manifests()
    protocol = {"schema": "legion_public_le_stage2_evaluation_v1", "frozen_at_utc": now(),
                "model_name": MODEL_NAME, "checkpoint": str(MODEL.resolve()),
                "checkpoint_sha256": MODEL_SHA, "manifests": manifests,
                "jobs": [{"name": n, "physical_gpu": g, "datasets": d} for n, g, d in JOBS],
                "batch_size": 32, "threshold_prob_fake": 0.5,
                "train_or_selection_on_test_or_ood": False,
                "leakage_audit": "BLOCKED_OVERLAP; override ACTIVE; AIGI-Holmes exact overlap 779"}
    save(OUT / "protocol.json", protocol)
    status = {"status": "RUNNING", "stage": "classification", "started_at_utc": now(),
              "jobs": {}}
    save(STATUS, status)
    active = {}
    try:
        for name, gpu, datasets in JOBS:
            process, log = launch(name, gpu, datasets)
            active[name] = (process, log)
            status["jobs"][name] = {"pid": process.pid, "physical_gpu": gpu,
                                     "datasets": datasets, "status": "RUNNING"}
        save(STATUS, status)
        while active:
            time.sleep(15)
            for name, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[name]
                status["jobs"][name]["status"] = "COMPLETE" if code == 0 else "FAILED"
                status["jobs"][name]["exit_code"] = code
                status["updated_at_utc"] = now()
                save(STATUS, status)
                if code:
                    raise RuntimeError(f"Worker {name} failed with exit code {code}")
        results = {name: verify_dataset(name, manifest) for name, manifest in manifests.items()}
        save(OUT / "results.json", {"schema": "legion_public_le_stage2_evaluation_results_v1",
                                    "status": "COMPLETE", "model": MODEL_NAME,
                                    "checkpoint_sha256": MODEL_SHA,
                                    "generated_at_utc": now(), "classification": results})
        doc = ROOT / "docs/legion_public_le_stage2_evaluation.md"
        temp_doc = doc.with_suffix(".md.tmp")
        temp_doc.write_text(render(results, protocol))
        os.replace(temp_doc, doc)
        status.update(status="COMPLETE", stage="complete", completed_at_utc=now(),
                      results=str((OUT / "results.json").resolve()), report=str(doc.resolve()))
        save(STATUS, status)
    except BaseException as exc:
        for process, log in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            log.close()
        status.update(status="FAILED", failed_at_utc=now(),
                      exception=repr(exc), traceback=traceback.format_exc())
        save(STATUS, status)
        raise


if __name__ == "__main__":
    main()
