#!/usr/bin/env python3
"""Finalize the user-stopped Phase 3E as a validation-only scientific report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3e_joint_language_mask_posttraining"
STATS = OUT / "statistics/validation_final_comparisons.json"
REPORT = ROOT / "docs/phase3e_joint_language_to_mask_posttraining.md"
OUTPUT_REPORT = OUT / "final_validation_report.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired(left, right, key: str, seed: int, repeats: int = 10000):
    a = {row["sample_id"]: row for row in left}; b = {row["sample_id"]: row for row in right}
    if set(a) != set(b): raise RuntimeError("paired sample sets differ")
    ids = sorted(a); delta = np.asarray([b[i][key] - a[i][key] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed); means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        end = min(repeats, start + 500)
        index = rng.integers(0, len(delta), size=(end - start, len(delta)))
        means[start:end] = delta[index].mean(axis=1)
    eps = 1e-12
    return {
        "n": len(delta), "metric": key, "mean_delta": float(delta.mean()),
        "bootstrap_95ci": [float(x) for x in np.quantile(means, [.025, .975])],
        "wins": int((delta > eps).sum()), "ties": int((np.abs(delta) <= eps).sum()),
        "losses": int((delta < -eps).sum()), "bootstrap_repeats": repeats, "seed": seed,
    }


def main() -> None:
    p1_sel = load(OUT / "evaluation/selector/P1_FROZEN/selection_metrics.json")
    sft_selector = load(OUT / "evaluation/selector/SFT_CONT_selector.json")
    joint_selector = load(OUT / "evaluation/selector/JOINT_selector.json")
    sft_sel = sft_selector["selected_metrics"]
    joint_trained = next(row for row in joint_selector["candidates"] if row["optimizer_step"] == 300)
    triplet = load(OUT / "tf_phrase_triplet/triplet_metrics.json")
    tf = triplet["metrics"]

    g0_rows = {
        "P1": rows(OUT / "evaluation/selector/P1_FROZEN/G0/predictions.jsonl"),
        "SFT": rows(OUT / "evaluation/selector/SFT_CONT/step_0900/G0/predictions.jsonl"),
        "JOINT_TRAINED": rows(OUT / "evaluation/selector/JOINT/step_0300/G0/predictions.jsonl"),
    }
    tf_rows = {
        "P1": rows(OUT / "evaluation/final/P1_FROZEN/tf_full_context/predictions.jsonl"),
        "SFT": rows(OUT / "evaluation/final/SFT_CONT/tf_full_context/predictions.jsonl"),
        "JOINT_TRAINED": rows(OUT / "evaluation/final/JOINT_TRAINED_STEP_0300/tf_full_context/predictions.jsonl"),
    }
    comparisons = {}
    pairs = (("sft_minus_p1", "P1", "SFT"), ("joint_trained_minus_p1", "P1", "JOINT_TRAINED"),
             ("joint_trained_minus_sft", "SFT", "JOINT_TRAINED"))
    for index, (name, base, new) in enumerate(pairs):
        comparisons[name] = {
            "G0_foreground_iou": paired(g0_rows[base], g0_rows[new], "foreground_iou", 3407 + 10 * index),
            "G0_foreground_f1": paired(g0_rows[base], g0_rows[new], "foreground_f1", 3408 + 10 * index),
            "TF_foreground_iou": paired(tf_rows[base], tf_rows[new], "foreground_iou", 3409 + 10 * index),
            "TF_foreground_f1": paired(tf_rows[base], tf_rows[new], "foreground_f1", 3410 + 10 * index),
        }
    result = {
        "status": "COMPLETE_VALIDATION_ONLY", "phase_execution_status": "STOPPED_BY_USER",
        "primary_gate": "GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_NOT_SUPPORTED_ON_VALIDATION",
        "population": {"split": "validation", "all_samples_for_detection": 2212, "fake_samples_for_G0_TF": 1106},
        "protocol": {"detection_prompt": "canonical", "G0_prompt": "canonical", "TF_prompt": "canonical",
                     "generation_batch_size": 1, "mask_logit_threshold": 0.0, "threshold_tuned": False},
        "selected": {"SFT_CONT_step": 900, "JOINT_formal_step": 0, "JOINT_fallback_to_P1": True,
                     "JOINT_trained_diagnostic_step": 300},
        "comparisons": comparisons, "internal_test_used": False, "official1000_used": False,
        "heldout_claim_allowed": False,
    }
    dump(STATS, result)

    def ci(name, mode):
        value = comparisons[name][mode]
        return f"{value['mean_delta']:+.6f} [{value['bootstrap_95ci'][0]:+.6f}, {value['bootstrap_95ci'][1]:+.6f}]"

    p1_tf = tf["P1_FROZEN"]["per_image_mean"]
    sft_tf = tf["SFT_CONT_SELECTED_STEP_0900"]["per_image_mean"]
    joint_tf = tf["JOINT_TRAINED_BEST_STEP_0300"]["per_image_mean"]
    text = f"""# Phase 3E — Joint Language-to-Mask Grounding Post-training

## 最终状态与结论

本阶段由用户在 selector 完成后终止，随后仅授权完成三模型 canonical TF-PHRASE validation 诊断。最终 validation route gate 为：

`GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_NOT_SUPPORTED_ON_VALIDATION`

JOINT 的十个非零 checkpoint 均未超过 P1，正式 selector 回退 P1 step 0。SFT-CONT 选择 step 900，出现很小的 G0 validation 增益，但 paired bootstrap CI 跨 0；其 canonical TF-PHRASE 则出现很小但 CI 不跨 0 的下降。当前联合 mask objective 因此没有显示 autonomous G0 或 teacher-forced spatial gain。

## 协议

- 数据：固定 validation；Detection 2212 张，G0/TF-PHRASE 为其中 1106 张 Fake。
- Detection、G0、TF-PHRASE User prompt 均显式 canonical；G0/Detection direct batch=1。
- mask threshold 固定为 0.0；未调 threshold。
- 未使用 internal test 或 official1000；本报告不支持 held-out 泛化结论。
- mask target 是官方 polygons 派生的 per-image all-ref union，不是 pseudo mask。

## Selector 结果

| Arm | checkpoint role | G0 mean IoU | G0 mean F1 | Cls Acc | Phrase semantic | Structure valid |
|---|---|---:|---:|---:|---:|---:|
| P1 | frozen step 0 | {p1_sel['mean_foreground_iou']:.6f} | {p1_sel['mean_foreground_f1']:.6f} | {p1_sel['classification_accuracy']:.6f} | {p1_sel['R_phrase_sem']:.6f} | {p1_sel['structure_validity']:.6f} |
| SFT-CONT | selected step 900 | {sft_sel['mean_foreground_iou']:.6f} | {sft_sel['mean_foreground_f1']:.6f} | {sft_sel['classification_accuracy']:.6f} | {sft_sel['R_phrase_sem']:.6f} | {sft_sel['structure_validity']:.6f} |
| JOINT | best trained step 300, diagnostic | {joint_trained['mean_foreground_iou']:.6f} | {joint_trained['mean_foreground_f1']:.6f} | {joint_trained['classification_accuracy']:.6f} | {joint_trained['R_phrase_sem']:.6f} | {joint_trained['structure_validity']:.6f} |
| JOINT formal | fallback P1 step 0 | {p1_sel['mean_foreground_iou']:.6f} | {p1_sel['mean_foreground_f1']:.6f} | {p1_sel['classification_accuracy']:.6f} | {p1_sel['R_phrase_sem']:.6f} | {p1_sel['structure_validity']:.6f} |

G0 IoU paired delta（new−base，95% bootstrap CI）：SFT−P1 `{ci('sft_minus_p1','G0_foreground_iou')}`；trained JOINT−P1 `{ci('joint_trained_minus_p1','G0_foreground_iou')}`；trained JOINT−SFT `{ci('joint_trained_minus_sft','G0_foreground_iou')}`。SFT checkpoint 是在同一 validation 上按 G0 选择，故其 CI 是 selection-conditioned 描述，不能作为独立确认性显著性。

## Canonical TF-PHRASE

| Arm | checkpoint role | Mean IoU | Mean F1 | Global IoU | Global F1 |
|---|---|---:|---:|---:|---:|
| P1 | frozen baseline | {p1_tf['foreground_iou']:.6f} | {p1_tf['foreground_f1']:.6f} | {tf['P1_FROZEN']['global_pixel']['foreground_iou']:.6f} | {tf['P1_FROZEN']['global_pixel']['foreground_f1']:.6f} |
| SFT-CONT | selected step 900 | {sft_tf['foreground_iou']:.6f} | {sft_tf['foreground_f1']:.6f} | {tf['SFT_CONT_SELECTED_STEP_0900']['global_pixel']['foreground_iou']:.6f} | {tf['SFT_CONT_SELECTED_STEP_0900']['global_pixel']['foreground_f1']:.6f} |
| JOINT | trained step 300 diagnostic | {joint_tf['foreground_iou']:.6f} | {joint_tf['foreground_f1']:.6f} | {tf['JOINT_TRAINED_BEST_STEP_0300']['global_pixel']['foreground_iou']:.6f} | {tf['JOINT_TRAINED_BEST_STEP_0300']['global_pixel']['foreground_f1']:.6f} |

TF IoU paired delta：SFT−P1 `{ci('sft_minus_p1','TF_foreground_iou')}`；trained JOINT−P1 `{ci('joint_trained_minus_p1','TF_foreground_iou')}`；trained JOINT−SFT `{ci('joint_trained_minus_sft','TF_foreground_iou')}`。

## 机制解释与边界

训练公平性审计 PASS；两臂各 1000 optimizer steps、4000 image exposures，样本顺序、标签、text-token 分母和 LoRA LR 完全一致。mask-only backward 对 LoRA、text_hidden_fcs、mask_decoder 的梯度均非零，因此不能把负结果解释成“mask loss 没有进入模型”。

允许的结论是：在 P1 初始化、当前 4000-exposure budget、既定 loss 和可训练模块下，联合 mask post-training 没有产生 validation G0 或 TF-PHRASE 增益。结果更符合当前优化目标/接口未能改善 autonomous language-to-mask grounding，而不是实现断路。

不能声称 mask decoder 是唯一瓶颈、所有 direct spatial supervision 都无效，或 SFT-CONT 已获得 held-out 泛化提升。若论文需要最终有效性结论，应冻结本报告中的 checkpoint 与协议后，再一次性解封 held-out test；本阶段没有执行该步骤。
"""
    REPORT.write_text(text, encoding="utf-8")
    OUTPUT_REPORT.write_text(text, encoding="utf-8")

    route = ROOT / "docs/paper_convergence_route.md"
    marker = "## Phase 3E — validation-only stopped final report"
    if marker not in route.read_text(encoding="utf-8"):
        with route.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n{marker}\n\nPhase 3E selector completed before user termination. SFT-CONT selected step 900; JOINT formally fell back to P1 step 0. A separately labeled trained-JOINT step-300 canonical TF-PHRASE diagnostic was completed on validation. Final validation gate: `GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_NOT_SUPPORTED_ON_VALIDATION`. No internal test, official1000, threshold tuning, FEPN, or further post-training search was executed. See `docs/phase3e_joint_language_to_mask_posttraining.md`.\n")

    manifest = {
        "status": "STOPPED_WITH_VALIDATION_REPORT_COMPLETE", "phase": "Phase 3E",
        "primary_gate": result["primary_gate"], "heldout_evaluation_complete": False,
        "internal_test_used": False, "official1000_used": False, "threshold_tuned": False,
        "formal_selected_checkpoints": {"SFT_CONT": sft_selector["selected_checkpoint"],
                                        "JOINT": joint_selector["selected_checkpoint"]},
        "diagnostic_joint_checkpoint": joint_trained["checkpoint"],
        "files": [{"path": str(path.resolve()), "sha256": sha256(path)}
                  for path in (STATS, REPORT, OUTPUT_REPORT, route, OUT / "tf_phrase_triplet/triplet_metrics.json")],
    }
    dump(OUT / "stopped_phase_report_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
