#!/usr/bin/env python3
"""Direct batch=1 internal-test robustness supplement for Phase 3C.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics import _localization_record
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import file_sha256
from scripts.phase2c_forensic_fusion import binary_metrics
from scripts.phase3a_evaluate import preserve_spatial_prediction, summarize_localization
from scripts.phase3c2_p3 import (
    CFG,
    CKPT,
    OUT,
    corrupted_sample,
    dataset_for,
    load_expert,
    load_glamm,
    load_p3,
)
from tools.phase3c2 import CONDITIONS, mcnemar_exact


INTERNAL = OUT / "robustness_internal"
FROZEN_ORIGINAL_G0 = ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal/G0/predictions.jsonl"
SECTION_MARKER = "## 6. Internal test 五条件鲁棒性补充"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--condition", choices=CONDITIONS, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--max-samples", type=int, default=None)
    sub.add_parser("analyze")
    sub.add_parser("finalize")
    return parser.parse_args(argv)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(condition: str, device_name: str, max_samples: int | None) -> None:
    device = torch.device(device_name)
    model, tokenizer, checkpoint_meta, model_cfg = load_glamm("p1", device)
    expert, expert_path = load_expert(device)
    p3, p3_checkpoint = load_p3(device)
    transform = build_npr_transform(load_size=256, crop_size=224, training=False)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(model_cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = dataset_for(tokenizer, model_cfg, "test")
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    root = INTERNAL / condition
    cls_path, g0_path = root / "classification.jsonl", root / "P1_G0.jsonl"
    cls_done = {row["sample_id"] for row in read_jsonl(cls_path)}
    g0_done = {row["sample_id"] for row in read_jsonl(g0_path)}
    captured = []
    hook = model.classification_head.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    try:
        for index in range(limit):
            row = dataset.rows[index]
            sample_id = row["sample_id"]
            is_fake = int(row["class_label"]) == 1
            need_cls = sample_id not in cls_done
            need_g0 = condition != "original" and is_fake and sample_id not in g0_done
            if not need_cls and not need_g0:
                continue
            sample, image, image_hash = corrupted_sample(dataset, index, condition)
            if need_cls:
                captured.clear()
                batch = backend._batch(sample, "")
                batch["grounding_enc_images"] = None
                with torch.no_grad():
                    output = model.model_forward(**batch)
                if len(captured) != 1:
                    raise RuntimeError("P1 classification hidden-state hook mismatch")
                p1_logits = output["cls_logits"].detach().float()
                expert_input = transform(Image.fromarray(image, mode="RGB"))[None].to(device)
                with torch.no_grad():
                    branches = expert.extract_branch_features(expert_input)
                    fused = p3(
                        p1_logits, captured[0], npr=branches["npr"], srm=branches["srm_gated"],
                    )
                append_jsonl(cls_path, {
                    "sample_id": sample_id, "gt": int(sample["cls_label"]),
                    "condition": condition, "corrupted_rgb_sha256": image_hash,
                    "P1_prob_fake": float(p1_logits.softmax(-1)[0, 1]),
                    "P1_pred": int(p1_logits.argmax(-1)[0]),
                    "P3_prob_fake": float(fused.logits.softmax(-1)[0, 1]),
                    "P3_pred": int(fused.logits.argmax(-1)[0]),
                })
                cls_done.add(sample_id)
            if need_g0:
                generated = backend.generate_localization_batch(
                    [sample], provide_gt_fake=False, generation_mode="unified_fake_generate"
                )[0]
                record = _localization_record(
                    sample, generated, "unified_fake_generate", uses_gt_authenticity=False,
                    uses_gt_explanation=False, classification_gate=False,
                )
                record = preserve_spatial_prediction(
                    root, "G0", sample, generated, record, save_spatial=False,
                )
                record["condition"] = condition
                record["corrupted_rgb_sha256"] = image_hash
                append_jsonl(g0_path, record)
                g0_done.add(sample_id)
            if (index + 1) % 10 == 0:
                print(
                    f"internal robust {condition}: {index + 1}/{limit} "
                    f"classification={len(cls_done)} G0={len(g0_done)}",
                    flush=True,
                )
    finally:
        hook.remove()
    expected_cls = limit
    expected_g0 = sum(int(dataset.rows[i]["class_label"]) == 1 for i in range(limit))
    actual_g0 = expected_g0 if condition == "original" else len(g0_done)
    if len(cls_done) != expected_cls or actual_g0 != expected_g0:
        raise AssertionError(f"incomplete {condition}: cls={len(cls_done)}/{expected_cls}, G0={actual_g0}/{expected_g0}")
    write_json(root / "provenance.json", {
        "population": "internal_test", "condition": condition,
        "classification_samples": expected_cls, "P1_G0_fake_samples": expected_g0,
        "batch_size": 1, "P1_checkpoint": checkpoint_meta,
        "P3_checkpoint": str(CKPT / "best.pt"), "P3_checkpoint_sha256": file_sha256(CKPT / "best.pt"),
        "P3_selected_step": int(p3_checkpoint["step"]),
        "expert_checkpoint": str(expert_path), "expert_checkpoint_sha256": CFG["expert"]["sha256"],
        "P1_G0_source": str(FROZEN_ORIGINAL_G0 if condition == "original" else g0_path),
        "original_P1_G0_reused": condition == "original",
        "perturbation": CFG["robustness"]["conditions"][condition],
    })


def bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        end = min(start + 500, repeats)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analyze() -> dict:
    result = {
        "population": "frozen internal test: Real 1104 + Fake 1104",
        "classification_batch_size": 1,
        "P1_G0_population": "internal test Fake 1104",
        "P3_canonical_G0_shared_with_P1": True,
        "conditions": {},
    }
    original_g0 = {row["sample_id"]: row for row in read_jsonl(FROZEN_ORIGINAL_G0)}
    if len(original_g0) != 1104:
        raise AssertionError("frozen original P1 G0 is incomplete")
    for condition_index, condition in enumerate(CONDITIONS):
        root = INTERNAL / condition
        cls = read_jsonl(root / "classification.jsonl")
        if len(cls) != 2208 or len({row["sample_id"] for row in cls}) != 2208:
            raise AssertionError(f"incomplete internal classification: {condition}")
        labels = np.asarray([row["gt"] for row in cls])
        p1_prob = np.asarray([row["P1_prob_fake"] for row in cls])
        p3_prob = np.asarray([row["P3_prob_fake"] for row in cls])
        p1_pred, p3_pred = p1_prob >= 0.5, p3_prob >= 0.5
        g0 = original_g0 if condition == "original" else {
            row["sample_id"]: row for row in read_jsonl(root / "P1_G0.jsonl")
        }
        if set(g0) != set(original_g0):
            raise AssertionError(f"internal G0 sample-set mismatch: {condition}")
        ordered_ids = list(original_g0)
        deltas = np.asarray([
            float(g0[sample_id]["foreground_iou"]) - float(original_g0[sample_id]["foreground_iou"])
            for sample_id in ordered_ids
        ])
        p1_metrics = binary_metrics(labels, p1_prob)
        p3_metrics = binary_metrics(labels, p3_prob)
        result["conditions"][condition] = {
            "P1_classification": p1_metrics,
            "P3_classification": p3_metrics,
            "P3_minus_P1": {
                key: p3_metrics[key] - p1_metrics[key]
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "auprc", "brier", "ece_15")
            },
            "paired_mcnemar": mcnemar_exact(p1_pred, p3_pred, labels),
            "P1_canonical_G0": summarize_localization(list(g0.values()), autoregressive=True),
            "P3_canonical_G0_exactly_shared": True,
            "P1_G0_delta_vs_original": {
                "mean_foreground_iou": float(deltas.mean()),
                "bootstrap_95ci": bootstrap_mean_ci(
                    deltas, int(CFG["experiment"]["seed"]) + condition_index,
                ),
            },
        }
    write_json(INTERNAL / "summary.json", result)
    return result


def report_section(summary: dict) -> str:
    rows = []
    for condition in CONDITIONS:
        value = summary["conditions"][condition]
        p1, p3 = value["P1_classification"], value["P3_classification"]
        g0 = value["P1_canonical_G0"]["per_image_mean"]
        paired = value["paired_mcnemar"]
        rows.append(
            f"| {condition} | {p1['accuracy']:.6f} | {p3['accuracy']:.6f} | "
            f"{p3['accuracy']-p1['accuracy']:+.6f} | {p1['f1']:.6f} | {p3['f1']:.6f} | "
            f"{paired['net_corrected']:+d} | {paired['exact_two_sided_p']:.6g} | "
            f"{g0['foreground_iou']:.6f} | "
            f"{value['P1_G0_delta_vs_original']['mean_foreground_iou']:+.6f} |"
        )
    return f"""

{SECTION_MARKER}

本补充只使用冻结 internal test（Real 1104 + Fake 1104）。P1/P3 classification 均采用 direct batch=1；P3 仍为 classification-only，因此 canonical G0 只运行P1，P3与P1逐样本完全共享。Original P1 G0直接复用Phase 3A冻结的1104张Fake结果，其余四个扰动重新推理。JPEG与Gaussian定义和official1000鲁棒性实验完全一致。

| 条件 | P1 Acc | P3 Acc | P3−P1 Acc | P1 F1 | P3 F1 | 净纠正 | McNemar p | P1 G0 FG IoU | G0 Δ vs Original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

这里的internal direct batch=1结果取代此前仅用于P3训练/快速比较的batch=8 cache绝对值；cache仍用于冻结训练输入与selector，但不作为最终部署式分类数值。P3 canonical G0与P1相等是结构约束，不代表P3获得定位增益。完整配对统计与每个扰动相对Original的bootstrap 95% CI见 `outputs/phase3c2_p3_robustness/robustness_internal/summary.json`。
"""


def finalize() -> None:
    summary_path = INTERNAL / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else analyze()
    section = report_section(summary)
    docs = ROOT / "docs/phase3c2_p3_robustness.md"
    current = docs.read_text(encoding="utf-8")
    if SECTION_MARKER in current:
        current = current.split(SECTION_MARKER, 1)[0].rstrip()
    updated = current.rstrip() + section
    docs.write_text(updated.rstrip() + "\n", encoding="utf-8")
    mirror = OUT / "reports/phase3c2_p3_robustness.md"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(updated.rstrip() + "\n", encoding="utf-8")
    completion_path = OUT / "completion_manifest.json"
    completion = json.loads(completion_path.read_text())
    report_hash = file_sha256(docs)
    for entry in completion.get("files", []):
        if Path(entry["path"]).resolve() == docs.resolve():
            entry["sha256"] = report_hash
    completion.update({
        "status": "COMPLETE_WITH_INTERNAL_ROBUSTNESS",
        "internal_robustness_complete": True,
        "internal_robustness_summary": str(summary_path),
        "report_sha256_after_internal_append": report_hash,
        "P1_checkpoint_sha256_after": file_sha256((ROOT / CFG["base_checkpoint"]["path"]).resolve()),
        "P3_checkpoint_sha256_after": file_sha256(CKPT / "best.pt"),
    })
    if completion["P1_checkpoint_sha256_after"] != CFG["base_checkpoint"]["sha256"]:
        raise AssertionError("P1 checkpoint changed during internal robustness")
    selected = json.loads((OUT / "training/selection.json").read_text())
    if completion["P3_checkpoint_sha256_after"] != selected["checkpoint_sha256"]:
        raise AssertionError("P3 checkpoint changed during internal robustness")
    write_json(completion_path, completion)
    write_json(INTERNAL / "completion_manifest.json", {
        "status": "COMPLETE", "classification_samples_per_condition": 2208,
        "P1_G0_fake_samples_per_condition": 1104,
        "conditions": list(CONDITIONS), "classification_batch_size": 1,
        "P3_canonical_G0_shared_with_P1": True,
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
        "report": str(docs), "report_sha256": file_sha256(docs),
        "P1_primary_unchanged": True, "P3_checkpoint_unchanged": True,
    })


def main(argv=None):
    cli = parse_args(argv)
    if cli.command == "evaluate":
        evaluate(cli.condition, cli.device, cli.max_samples)
    elif cli.command == "analyze":
        analyze()
    elif cli.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
