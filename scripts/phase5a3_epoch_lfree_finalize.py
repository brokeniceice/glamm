#!/usr/bin/env python3
"""Finalize the three Stage-1 epoch L-FREE validation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a3_stage1_epoch_lfree_validation"
MANIFEST = OUT / "manifests/internal_val_fake.jsonl"
REPORT = ROOT / "docs/phase5a3_stage1_epoch_lfree_validation.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    iou = np.asarray([row["foreground_iou"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["foreground_f1"] for row in rows], dtype=np.float64)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "n": len(rows), "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)), "mean_foreground_f1": float(f1.mean()),
        "global_foreground_iou": tp / (tp + fp + fn),
        "global_foreground_f1": 2 * tp / (2 * tp + fp + fn),
        "valid_seg_and_mask": sum(row["status"] == "OK" for row in rows),
        "status_counts": {name: sum(row["status"] == name for row in rows) for name in sorted({r["status"] for r in rows})},
    }


def paired(a: list[dict], b: list[dict]) -> dict:
    av = np.asarray([row["foreground_iou"] for row in a])
    bv = np.asarray([row["foreground_iou"] for row in b])
    delta = bv - av
    rng = np.random.default_rng(3407)
    boots = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10000)])
    nz = delta[np.abs(delta) > 1e-12]
    stat, p = wilcoxon(nz) if len(nz) else (0.0, 1.0)
    return {
        "mean_iou_delta_later_minus_earlier": float(delta.mean()),
        "bootstrap_95_ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
        "wins_ties_losses": [int((delta > 1e-12).sum()), int((np.abs(delta) <= 1e-12).sum()), int((delta < -1e-12).sum())],
        "wilcoxon_p": float(p), "wilcoxon_statistic": float(stat),
    }


def main() -> None:
    manifest = load_jsonl(MANIFEST)
    ids = [row["sample_id"] for row in manifest]
    epochs: dict[str, dict] = {}
    epoch_rows: dict[str, list[dict]] = {}
    for epoch in (1, 2, 3):
        shard_root = OUT / f"epoch{epoch}/original/shards"
        rows = []
        for path in sorted(shard_root.glob("*.predictions.jsonl")):
            rows.extend(load_jsonl(path))
        by_id = {row["sample_id"]: row for row in rows}
        if len(rows) != 1106 or len(by_id) != 1106 or set(by_id) != set(ids):
            raise RuntimeError(f"epoch {epoch} incomplete or identity drift: {len(rows)}/{len(by_id)}")
        ordered = [by_id[sample_id] for sample_id in ids]
        epoch_rows[str(epoch)] = ordered
        epochs[str(epoch)] = summarize(ordered)
    results = {
        "schema": "phase5a3_stage1_epoch_lfree_validation_results_v1", "status": "COMPLETE",
        "population": {"n": 1106, "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST)},
        "protocol": "official image-only L-FREE; free generation; [SEG]->SAM; logit>0; multi-mask union; full-N failures=zero",
        "teacher_forced_val_total_loss": {"1": 1.065668, "2": 1.026487, "3": 1.018036},
        "epochs": epochs,
        "paired": {
            "epoch2_minus_epoch1": paired(epoch_rows["1"], epoch_rows["2"]),
            "epoch3_minus_epoch2": paired(epoch_rows["2"], epoch_rows["3"]),
            "epoch3_minus_epoch1": paired(epoch_rows["1"], epoch_rows["3"]),
        },
        "decision": "STOP_AFTER_DIAGNOSTIC_NO_AUTOMATIC_TRAINING_EXTENSION",
    }
    result_path = OUT / "results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Phase 5A-3 — Stage-1 Epoch L-FREE Validation Diagnostic", "",
        "本阶段只在冻结 internal-validation Fake 1,106 张 unique images 上比较现有 Epoch 1/2/3；没有继续训练。", "",
        "固定协议：官方 image-only L-FREE、free generation、`[SEG]→SAM`、mask logit `>0`、multiple masks union；无 `[SEG]`/空 mask 在 full-N 中计零。GT 是每图全部官方 refs polygon 的原分辨率 pixel union。", "",
        "| Epoch | TF val total loss | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in (1, 2, 3):
        value = epochs[str(epoch)]
        lines.append(f"| {epoch} | {results['teacher_forced_val_total_loss'][str(epoch)]:.6f} | {value['n']} | {value['mean_foreground_iou']:.6f} | {value['median_foreground_iou']:.6f} | {value['mean_foreground_f1']:.6f} | {value['global_foreground_iou']:.6f} | {value['global_foreground_f1']:.6f} | {value['valid_seg_and_mask']}/{value['n']} |")
    lines += ["", "## Paired changes", ""]
    for name, value in results["paired"].items():
        lo, hi = value["bootstrap_95_ci"]
        w, t, l = value["wins_ties_losses"]
        lines.append(f"- {name}: mean FG IoU delta `{value['mean_iou_delta_later_minus_earlier']:+.6f}`, 95% CI `[{lo:+.6f}, {hi:+.6f}]`, W/T/L `{w}/{t}/{l}`, Wilcoxon p=`{value['wilcoxon_p']:.8g}`.")
    lines += ["", "本结果仅用于判断是否值得授权延长 Stage-1 训练；本阶段不会自动继续训练或选择新 checkpoint。", "", f"Machine result: `{result_path}`", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "results": str(result_path), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
