#!/usr/bin/env python3
"""Run frozen C2-raw G1 OOD on GPUs 1/2 and finalize the original report."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import final_eval_localization_supervisor as frozen

OUT = ROOT / "outputs/phase6j0_c2/final_evaluation/localization"
REPORT = ROOT / "docs/final_evaluation_report.md"
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"
ORDER = ("loki", "xaigd", "pal4vst")
HEADINGS = {
    "loki": "### LOKI229（687 个官方区域框在 229 张 Fake 图像上取 union；空 GT 0）",
    "xaigd": "### X-AIGD labeled_test（官方人类感知伪影 polygon union；空 GT 247）",
    "pal4vst": "### PAL4VST test（官方像素伪影 mask；空 GT 313）",
}


def status(state, **extra):
    frozen.atomic_json(OUT / "supervisor_status.json", {
        "status": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra})


def start(dataset, gpu):
    log = OUT / "logs" / f"{dataset}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", buffering=1)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    process = subprocess.Popen([PY, "-u", str(ROOT / "scripts/phase6j0_c2_localization_ood.py"),
                                "--dataset", dataset, "--device", f"cuda:{gpu}"],
                               cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process, handle


def check(dataset, process, handle):
    code = process.wait()
    handle.close()
    if code:
        raise RuntimeError(f"{dataset} worker exited {code}; inspect {OUT / 'logs' / (dataset + '.log')}")


def finalize():
    results = {}
    for dataset in ORDER:
        rows, manifest = frozen.manifest_info(dataset)
        result = frozen.read_json(OUT / dataset / "results.json")
        predictions = frozen.rows(OUT / dataset / "predictions.jsonl")
        if (result["status"] != "COMPLETE" or result["model"] != "C2-raw" or
                result["inference_condition"] != "G1" or result["manifest"] != manifest or
                len(predictions) != len(rows) or
                [x["sample_id"] for x in predictions] != [str(x["sample_id"]) for x in rows] or
                result["prediction_sha256"] != frozen.sha256_file(OUT / dataset / "predictions.jsonl") or
                result["metrics"] != frozen.summarize(predictions)):
            raise RuntimeError(f"{dataset} population, metric, or result integrity drift")
        cache_dir = OUT / dataset / "r1_replay_cache"
        for ordinal, prediction in enumerate(predictions):
            cache_path = cache_dir / f"{ordinal:06d}.pt"
            if (prediction["replay_cache_path"] != str(cache_path.resolve()) or
                    not cache_path.is_file() or
                    frozen.sha256_file(cache_path) != prediction["replay_cache_sha256"]):
                raise RuntimeError(f"{dataset} replay cache missing or corrupt at {ordinal}")
        results[dataset] = result
    lines = REPORT.read_text().splitlines()
    for dataset in ORDER:
        heading = HEADINGS[dataset]
        if lines.count(heading) != 1:
            raise RuntimeError(f"report heading drift: {heading}")
        start = lines.index(heading)
        separator = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("|---")), None)
        if separator is None:
            raise RuntimeError(f"report table missing: {dataset}")
        end = separator + 1
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        metrics = results[dataset]["metrics"]
        row = (f"| C2-raw | G1 | {metrics['n']} | {metrics['mean_foreground_iou']:.6f} | "
               f"{metrics['mean_foreground_f1']:.6f} | {metrics['global_foreground_iou']:.6f} | "
               f"{metrics['global_foreground_f1']:.6f} |")
        existing = [i for i in range(separator + 1, end) if lines[i].startswith("| C2-raw |")]
        if len(existing) > 1:
            raise RuntimeError(f"duplicate C2-raw rows: {dataset}")
        if existing:
            lines[existing[0]] = row
        else:
            lines.insert(end, row)
    old = "C2 外部定位 OOD 留待接入 R1 后评测。"
    new = ("C2-raw 外部定位 OOD 的 G1 逐样本结果和可供后续 C2 接 R1 复用的冻结特征缓存见"
           "[C2 定位 OOD 汇总](../outputs/phase6j0_c2/final_evaluation/localization/summary.json)。")
    content = "\n".join(lines) + "\n"
    if old in content:
        content = content.replace(old, new)
    elif new not in content:
        raise RuntimeError("C2 report source note drift")
    tmp = REPORT.with_suffix(".md.tmp")
    tmp.write_text(content)
    os.replace(tmp, REPORT)
    frozen.atomic_json(OUT / "summary.json", {
        "schema": "phase6j0_c2_raw_localization_ood_summary_v1", "status": "COMPLETE",
        "datasets": results, "report": str(REPORT.resolve()),
        "cache_policy": "Per-image frozen C2 G1 q_seg/S64_raw/F24/z_F24 with geometry and SHA256; no R1 inference performed"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status("RUNNING", stage="loki+xaigd", gpu_assignment={"loki": 1, "xaigd": 2, "pal4vst": 1})
    loki, loki_log = start("loki", 1)
    xaigd, xaigd_log = start("xaigd", 2)
    try:
        check("loki", loki, loki_log)
        status("RUNNING", stage="pal4vst+xaigd", gpu_assignment={"xaigd": 2, "pal4vst": 1})
        pal, pal_log = start("pal4vst", 1)
        check("pal4vst", pal, pal_log)
        check("xaigd", xaigd, xaigd_log)
        status("RUNNING", stage="finalize")
        finalize()
        status("COMPLETE", summary=str((OUT / "summary.json").resolve()))
    except BaseException as exc:
        status("FAILED", error=str(exc))
        raise


if __name__ == "__main__":
    main()
