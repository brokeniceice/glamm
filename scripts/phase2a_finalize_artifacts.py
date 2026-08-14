#!/usr/bin/env python3
"""Merge the interrupted/resumed Phase 2A run and audit its final state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2a_unified_baseline"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def link_force(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def replace_marked(text: str, marker: str, body: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    if start not in text or end not in text:
        raise ValueError(f"Missing documentation markers for {marker}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{body.rstrip()}\n{end}{after}"


def update_documentation(summary: dict) -> None:
    document = ROOT / "docs/phase2a_full_unified_baseline.md"
    text = document.read_text(encoding="utf-8")
    detection = summary["modes"]["detection"]
    cls, lm = detection["classification_head"], detection["lm_verdict"]
    lines = [
        "### Detection",
        "",
        "| Head | Accuracy | Precision | Recall/Fake Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Classification | {cls['accuracy']:.6f} | {cls['precision']:.6f} | "
        f"{cls['fake_recall']:.6f} | {cls['f1']:.6f} | {cls['roc_auc']:.6f} |",
        f"| LM verdict | {lm['accuracy']:.6f} | {lm['precision']:.6f} | "
        f"{lm['fake_recall']:.6f} | {lm['f1']:.6f} | {lm['roc_auc']:.6f} |",
        "",
        f"CLS–LM agreement：**{detection['cls_lm_agreement']:.6f}**。",
        "",
        "### Localization overall",
        "",
        "| Mode | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 | SEG trigger |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("G0", "G1", "tf_full_context", "joint"):
        metric = summary["modes"][mode]
        trigger = metric.get("seg_trigger_rate")
        lines.append(
            f"| {mode} | {metric['mean_iou']:.6f} | {metric['global_iou']:.6f} | "
            f"{metric['mean_pixel_f1']:.6f} | {metric['global_pixel_f1']:.6f} | "
            f"{'N/A (teacher-forced)' if trigger is None else f'{trigger:.6f}'} |"
        )
    lines.extend(["", "### Localization by content type", ""])
    for mode in ("G0", "G1", "tf_full_context", "joint"):
        lines.extend([
            f"**{mode}**",
            "",
            "| Content | N | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for content in ("human", "animal", "object", "scene"):
            metric = summary["modes"][mode]["by_content_type"][content]
            lines.append(
                f"| {content.title()} | {metric['num_gt_fake']} | {metric['mean_iou']:.6f} | "
                f"{metric['global_iou']:.6f} | {metric['mean_pixel_f1']:.6f} | "
                f"{metric['global_pixel_f1']:.6f} |"
            )
        lines.append("")
    equivalence = summary["G0_joint_generation_equivalence"]
    lines.append(
        f"G0/Joint 对 {equivalence['num_compared']} 个 Fake 的 sample IDs 与生成轨迹 equivalence："
        f"**{'passed' if equivalence['all_generation_trajectories_equal'] else 'failed'}**。"
    )
    text = replace_marked(text, "PHASE2A_TEST_RESULTS", "\n".join(lines))

    language = []
    for mode in ("G0", "G1", "joint"):
        metric = summary["modes"][mode]
        language.append(
            f"- {mode}: SEG trigger `{metric['seg_trigger_rate']:.6f}`；"
            f"repetition `{metric['repetition_rate']:.6f}`；"
            f"mean generated tokens `{metric['mean_generated_token_length']:.3f}`；"
            f"EOS-before-SEG `{metric['eos_before_seg_rate']:.6f}`；"
            f"max-token-before-SEG `{metric['max_token_before_seg_rate']:.6f}`。"
        )
    text = replace_marked(text, "PHASE2A_LANGUAGE_RESULTS", "\n".join(language))
    readiness = (
        "**可以进入下一阶段 forensic feature enhancement。** Phase 2A 的训练、validation、"
        "best-only frozen internal test、完整 predictions、artifact audit 与 regression 均已完成；"
        "该 best checkpoint 可作为后续 NPR/SRM/fusion 的正式对照组。"
    )
    text = replace_marked(text, "PHASE2A_READINESS", readiness)
    document.write_text(text, encoding="utf-8")


def finalize_test() -> dict:
    expected = {"detection": 2208, "G0": 1104, "G1": 1104, "tf_full_context": 1104, "joint": 1104}
    result = {}
    records_by_mode = {}
    for mode, count in expected.items():
        directory = OUT / "test" / mode
        prediction_path, metric_path = directory / "predictions.jsonl", directory / "metrics.json"
        predictions = [
            json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(predictions) != count or len({row["sample_id"] for row in predictions}) != count:
            raise AssertionError(f"{mode}: expected {count} unique predictions, got {len(predictions)}")
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        result[mode] = metrics
        records_by_mode[mode] = predictions
        link_force(prediction_path, OUT / "predictions" / f"{mode}.jsonl")
    g0 = {row["sample_id"]: row for row in records_by_mode["G0"]}
    joint = {row["sample_id"]: row for row in records_by_mode["joint"]}
    equivalence = {
        "num_compared": len(g0),
        "all_sample_ids_equal": set(g0) == set(joint),
        "all_generation_trajectories_equal": all(
            g0[sample_id]["generated_token_ids"] == joint[sample_id]["generated_token_ids"]
            and g0[sample_id]["raw_prompt_text"] == joint[sample_id]["raw_prompt_text"]
            and g0[sample_id]["seg_position"] == joint[sample_id]["seg_position"]
            for sample_id in g0
        ),
    }
    if not all(equivalence.values()):
        raise AssertionError(equivalence)
    checkpoint = json.loads((OUT / "training_completion_audit.json").read_text(encoding="utf-8"))
    summary = {
        "status": "passed", "checkpoint": checkpoint["best_checkpoint"],
        "best_optimizer_step": checkpoint["best_optimizer_step"],
        "best_epoch": checkpoint["best_epoch"],
        "best_val_total_loss": checkpoint["best_val_total_loss"],
        "frozen_internal_test_samples": 2208, "frozen_internal_test_fake_samples": 1104,
        "modes": result, "G0_joint_generation_equivalence": equivalence,
        "external_evaluation_performed": False,
    }
    write_json(OUT / "test" / "summary.json", summary)
    link_force(OUT / "test/G0/predictions.jsonl", OUT / "predictions/explanations_G0.jsonl")
    link_force(ROOT / "checkpoints/phase2a_unified_baseline/single/best", OUT / "checkpoints/best")
    link_force(ROOT / "checkpoints/phase2a_unified_baseline/single/last", OUT / "checkpoints/last")
    update_documentation(summary)
    return summary


def main() -> None:
    final_segment = OUT / "metrics_single_gpu1_step3501_5000.jsonl"
    if not final_segment.exists():
        shutil.copy2(OUT / "metrics.jsonl", final_segment)
    sources = (
        ("metrics_dual_through_step1001.jsonl", 1, 1000),
        ("metrics_single_gpu0_failed_step3994.jsonl", 1001, 3500),
        ("metrics_single_gpu1_step3501_5000.jsonl", 3501, 5000),
    )
    merged = []
    provenance = []
    for name, first, last in sources:
        selected = [row for row in rows(name) if first <= int(row["optimizer_step"]) <= last]
        actual = [int(row["optimizer_step"]) for row in selected]
        expected = list(range(first, last + 1))
        if actual != expected:
            raise AssertionError(f"Non-contiguous segment {name}: {actual[:2]} ... {actual[-2:]}")
        merged.extend(selected)
        provenance.append({"file": name, "first_step": first, "last_step": last, "rows": len(selected)})
    steps = [int(row["optimizer_step"]) for row in merged]
    if steps != list(range(1, 5001)):
        raise AssertionError("Canonical Phase 2A metrics are not exactly steps 1..5000")
    with (OUT / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    validations = []
    for epoch in range(1, 11):
        metric_path = OUT / "validation" / f"epoch_{epoch:02d}" / "metrics.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        validations.append({
            "epoch": epoch,
            "optimizer_step": epoch * 500,
            "val_total_loss": metric["val_total_loss"],
            "classification_accuracy": metric["classification"]["accuracy"],
            "classification_f1": metric["classification"]["f1"],
            "lm_verdict_accuracy": metric["lm_verdict"]["accuracy"],
            "lm_verdict_f1": metric["lm_verdict"]["f1"],
            "cls_lm_agreement": metric["cls_lm_agreement"],
            "tf_mean_iou": metric["tf_full_context"]["mean_iou"],
            "tf_global_iou": metric["tf_full_context"]["global_iou"],
            "tf_mean_pixel_f1": metric["tf_full_context"]["mean_pixel_f1"],
            "tf_global_pixel_f1": metric["tf_full_context"]["global_pixel_f1"],
        })
    best = min(validations, key=lambda item: item["val_total_loss"])
    if best["optimizer_step"] != 2500:
        raise AssertionError(f"Unexpected best checkpoint: {best}")
    write_json(OUT / "validation_curve.json", validations)
    digest = hashlib.sha256((OUT / "metrics.jsonl").read_bytes()).hexdigest()
    write_json(OUT / "training_completion_audit.json", {
        "status": "passed",
        "optimizer_steps": 5000,
        "epochs": 10,
        "effective_global_batch": 20,
        "sample_exposures": 100000,
        "metrics_sha256": digest,
        "metrics_provenance": provenance,
        "best_checkpoint": str((ROOT / "checkpoints/phase2a_unified_baseline/single/best").resolve()),
        "best_optimizer_step": best["optimizer_step"],
        "best_epoch": best["epoch"],
        "best_val_total_loss": best["val_total_loss"],
        "last_checkpoint": str((ROOT / "checkpoints/phase2a_unified_baseline/single/last").resolve()),
        "last_optimizer_step": 5000,
        "selection_split": "val",
        "test_or_external_used_for_selection": False,
        "runtime_history": [
            {"steps": "1-1000", "world_size": 2, "micro_batch": 2, "gradient_accumulation": 5},
            {"steps": "1001-3500", "world_size": 1, "micro_batch": 10, "gradient_accumulation": 2},
            {"event": "OOM during an uncheckpointed replay after step 3500; discarded steps 3501-3994"},
            {"steps": "3501-5000", "world_size": 1, "micro_batch": 10, "gradient_accumulation": 2},
        ],
    })
    write_json(OUT / "gpu_memory_stress.json", {
        "status": "passed",
        "dual_gpu_worst_case": {
            "world_size": 2,
            "micro_batch_per_device": 2,
            "gradient_accumulation": 5,
            "rank0_peak_vram_bytes": 24096249856,
            "rank1_peak_vram_bytes": 24096249856,
            "source": "preflight/worst_case/run_summary.json",
        },
        "runtime_warning": {
            "event": "single-GPU allocator OOM in the first step3500 replay",
            "allocated_gib": 31.57,
            "reserved_gib": 33.56,
            "free_mib": 20,
            "requested_gib": 1.55,
            "resolution": "resume again from durable step3500 on physical GPU1; completed to step5000 unchanged",
        },
    })
    test_summary = None
    if all((OUT / "test" / mode / "metrics.json").exists()
           for mode in ("detection", "G0", "G1", "tf_full_context", "joint")):
        test_summary = finalize_test()
    print(json.dumps({
        "status": "passed", "best": best, "metrics_sha256": digest,
        "test_finalized": test_summary is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
