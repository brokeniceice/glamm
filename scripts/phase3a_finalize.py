#!/usr/bin/env python3
"""Build Phase 3A paired analyses, phrase audit, qualitative cases, and Chinese report."""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a_phrase_grounding"
C0_OFFICIAL = ROOT / "outputs/phase2b_legion_parity/official1000_raw"
C0_INTERNAL = ROOT / "outputs/phase2a_unified_baseline/test"
P1_OFFICIAL = OUT / "evaluation/official1000"
P1_INTERNAL = OUT / "evaluation/internal"
OFFICIAL_MANIFEST = (
    ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def overlap(reference: str, prediction: str | None) -> dict:
    ref, pred = Counter(tokens(reference)), Counter(tokens(prediction))
    common = sum((ref & pred).values())
    precision = common / sum(pred.values()) if pred else 0.0
    recall = common / sum(ref.values()) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "normalized_token_precision": precision,
        "normalized_token_recall": recall,
        "normalized_token_f1": f1,
        "normalized_exact_match": bool(ref == pred and ref),
        "extra_token_count": max(0, sum(pred.values()) - common),
        "missing_token_count": max(0, sum(ref.values()) - common),
    }


def paired_bootstrap(differences: np.ndarray, seed: int = 3407, repeats: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        width = min(500, repeats - start)
        indices = rng.integers(0, differences.size, size=(width, differences.size))
        means[start:start + width] = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "bootstrap_repeats": repeats,
        "bootstrap_95ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def add_c0_background(row: dict) -> dict:
    width, height = Image.open(row["image_path"]).size
    tn = width * height - int(row["tp"]) - int(row["fp"]) - int(row["fn"])
    fg_iou = float(row["image_iou"])
    bg_den = tn + int(row["fp"]) + int(row["fn"])
    bg_iou = tn / bg_den if bg_den else 1.0
    return {
        **row, "tn": tn, "foreground_iou": fg_iou,
        "foreground_f1": float(row["image_pixel_f1"]),
        "background_iou": bg_iou, "fg_bg_miou": (fg_iou + bg_iou) / 2,
    }


def paired_analysis(c0_rows: list[dict], p1_rows: list[dict]) -> tuple[dict, list[dict]]:
    c0 = {row["sample_id"]: add_c0_background(row) for row in c0_rows}
    p1 = {row["sample_id"]: row for row in p1_rows}
    if set(c0) != set(p1):
        raise ValueError(f"C0/P1 sample mismatch: {len(c0)} vs {len(p1)}")
    rows = []
    for sample_id in c0:
        a, b = c0[sample_id], p1[sample_id]
        rows.append({
            "sample_id": sample_id, "image_path": b["image_path"],
            "c0_iou": a["foreground_iou"], "p1_iou": b["foreground_iou"],
            "iou_difference": b["foreground_iou"] - a["foreground_iou"],
            "c0_f1": a["foreground_f1"], "p1_f1": b["foreground_f1"],
            "f1_difference": b["foreground_f1"] - a["foreground_f1"],
            "c0_fg_bg_miou": a["fg_bg_miou"], "p1_fg_bg_miou": b["fg_bg_miou"],
            "c0_generated_text": a.get("generated_text"),
            "p1_generated_text": b.get("decoded_text"),
            "p1_generated_phrase": b.get("generated_localization_phrase"),
            "p1_binary_mask_path": b.get("binary_mask_path"),
        })
    result = {"comparison_scope": "PAIRED_EVALUATION_UNPAIRED_TRAINING"}
    for metric in ("iou", "f1"):
        diff = np.array([row[f"{metric}_difference"] for row in rows])
        wins, ties, losses = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
        wilcoxon = stats.wilcoxon(diff) if np.any(diff) else None
        result[metric] = {
            **paired_bootstrap(diff), "wins": wins, "ties": ties, "losses": losses,
            "wilcoxon_statistic": None if wilcoxon is None else float(wilcoxon.statistic),
            "wilcoxon_pvalue": None if wilcoxon is None else float(wilcoxon.pvalue),
            "c0_mean": float(np.mean([row[f"c0_{metric}"] for row in rows])),
            "p1_mean": float(np.mean([row[f"p1_{metric}"] for row in rows])),
        }
    result["fg_bg_miou"] = {
        "c0_mean": float(np.mean([r["c0_fg_bg_miou"] for r in rows])),
        "p1_mean": float(np.mean([r["p1_fg_bg_miou"] for r in rows])),
        **paired_bootstrap(np.array([r["p1_fg_bg_miou"] - r["c0_fg_bg_miou"] for r in rows])),
    }
    return result, rows


def phrase_analysis(p1_rows: list[dict], manifest_rows: list[dict]) -> tuple[dict, list[dict]]:
    manifests = {row["sample_id"]: row for row in manifest_rows}
    results = []
    for row in p1_rows:
        manifest = manifests[row["sample_id"]]
        raw = [" ".join(str(ref.get("phrase") or "").split()) for ref in manifest.get("refs") or []]
        phrases = list(dict.fromkeys(raw))
        reference = "; ".join(phrases)
        predicted = row.get("generated_localization_phrase")
        metrics = overlap(reference, predicted)
        present = bool(row.get("phrase_parse", {}).get("target_field_present"))
        if not present or not predicted:
            category = "missing_target"
        elif metrics["normalized_exact_match"]:
            category = "exact_target"
        elif metrics["normalized_token_f1"] > 0:
            category = "partial_target"
        else:
            category = "extra_or_hallucinated_target"
        results.append({
            "sample_id": row["sample_id"], "gt_authoritative_phrases": phrases,
            "normalized_gt_phrase": reference, "generated_phrase": predicted,
            "target_presence": present, "category": category,
            "foreground_iou": row["foreground_iou"], **metrics,
        })
    f1s = np.array([row["normalized_token_f1"] for row in results])
    ious = np.array([row["foreground_iou"] for row in results])
    correlation = stats.spearmanr(f1s, ious)
    categories = Counter(row["category"] for row in results)
    exact = [r["foreground_iou"] for r in results if r["normalized_exact_match"]]
    nonexact = [r["foreground_iou"] for r in results if not r["normalized_exact_match"]]
    metrics = {
        "num_samples": len(results), "category_counts": dict(categories),
        "target_presence_rate": sum(r["target_presence"] for r in results) / len(results),
        "normalized_exact_match_rate": sum(r["normalized_exact_match"] for r in results) / len(results),
        "mean_normalized_token_f1": float(f1s.mean()),
        "spearman_phrase_f1_vs_iou": {
            "rho": float(correlation.statistic), "pvalue": float(correlation.pvalue),
        },
        "mean_iou_exact_phrase": float(np.mean(exact)) if exact else None,
        "mean_iou_nonexact_phrase": float(np.mean(nonexact)) if nonexact else None,
        "authoritative_metric": "deterministic normalized token overlap; no LLM evaluator",
    }
    return metrics, results


def severe_analysis(c0_g0: list[dict], c0_tf: list[dict], paired_rows: list[dict]) -> tuple[dict, list[dict]]:
    g0 = {r["sample_id"]: r for r in c0_g0}
    tf = {r["sample_id"]: r for r in c0_tf}
    severe_ids = {sid for sid in g0 if tf[sid]["image_iou"] >= 0.70 and g0[sid]["image_iou"] <= 0.30}
    selected = [{**row, "historical_tf_iou": tf[row["sample_id"]]["image_iou"]}
                for row in paired_rows if row["sample_id"] in severe_ids]
    c0_mean = float(np.mean([r["c0_iou"] for r in selected]))
    p1_mean = float(np.mean([r["p1_iou"] for r in selected]))
    p1_tf = {r["sample_id"]: r for r in load_jsonl(P1_OFFICIAL / "tf_full_context/predictions.jsonl")}
    p1_severe = [r for r in selected if p1_tf[r["sample_id"]]["foreground_iou"] >= .70 and r["p1_iou"] <= .30]
    metrics = {
        "definition": "historical TF IoU >= 0.70 and historical G0 IoU <= 0.30",
        "historical_severe_count": len(selected),
        "historical_severe_proportion": len(selected) / len(paired_rows),
        "mean_g0_iou_on_historical_severe": {"c0": c0_mean, "p1": p1_mean, "difference": p1_mean - c0_mean},
        "p1_severe_count_under_same_thresholds": len(p1_severe),
        "failure_recovery_count": len(selected) - len(p1_severe),
    }
    return metrics, selected


def classification_analysis() -> dict:
    c0 = {r["sample_id"]: r for r in load_jsonl(C0_INTERNAL / "detection/predictions.jsonl")}
    p1 = {r["sample_id"]: r for r in load_jsonl(P1_INTERNAL / "detection/predictions.jsonl")}
    common = sorted(set(c0) & set(p1))
    result = {"num_paired": len(common), "mcnemar": {}}
    for field in ("cls_pred", "lm_verdict_pred"):
        c0_correct = np.array([c0[s][field] == c0[s]["gt_label"] for s in common])
        p1_correct = np.array([p1[s][field] == p1[s]["gt_label"] for s in common])
        b = int((c0_correct & ~p1_correct).sum())
        c = int((~c0_correct & p1_correct).sum())
        pvalue = float(stats.binomtest(min(b, c), b + c, .5).pvalue) if b + c else 1.0
        result["mcnemar"][field] = {
            "c0_accuracy": float(c0_correct.mean()), "p1_accuracy": float(p1_correct.mean()),
            "c0_only_correct": b, "p1_only_correct": c, "exact_pvalue": pvalue,
        }
    result["p1_metrics"] = load_json(P1_INTERNAL / "detection/metrics.json")
    result["c0_metrics"] = load_json(C0_INTERNAL / "detection/metrics.json")
    return result


def make_qualitative(paired_rows: list[dict]) -> None:
    asset_dir = OUT / "qualitative/assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(paired_rows, key=lambda r: r["iou_difference"], reverse=True)[:20]
    selected += sorted(paired_rows, key=lambda r: r["iou_difference"])[:20]
    cards = []
    for index, row in enumerate(selected):
        image = Image.open(row["image_path"]).convert("RGB")
        binary = torch.load(row["p1_binary_mask_path"], map_location="cpu").numpy().astype(bool)
        overlay = np.asarray(image).copy()
        overlay[binary] = (0.45 * overlay[binary] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        path = asset_dir / f"{index:03d}.jpg"
        Image.fromarray(overlay).save(path, quality=90)
        kind = "P1 rescue" if row["iou_difference"] >= 0 else "P1 regression"
        cards.append(f'''<article><h2>{kind}: {html.escape(row['sample_id'])}</h2>
<img src="assets/{path.name}" width="512"><p>C0 IoU={row['c0_iou']:.4f}; P1 IoU={row['p1_iou']:.4f}; Δ={row['iou_difference']:+.4f}</p>
<p>P1 phrase: {html.escape(str(row['p1_generated_phrase']))}</p>
<details><summary>C0 generation</summary><pre>{html.escape(str(row['c0_generated_text']))}</pre></details>
<details><summary>P1 generation</summary><pre>{html.escape(str(row['p1_generated_text']))}</pre></details></article>''')
    (OUT / "qualitative/index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Phase 3A</title>" + "\n".join(cards),
        encoding="utf-8",
    )


def fmt(value) -> str:
    return "NA" if value is None else f"{value:.6f}"


def main():
    c0_g0 = load_jsonl(C0_OFFICIAL / "G0/predictions.jsonl")
    c0_tf = load_jsonl(C0_OFFICIAL / "tf_full_context/predictions.jsonl")
    p1_g0 = load_jsonl(P1_OFFICIAL / "G0/predictions.jsonl")
    manifest = load_jsonl(OFFICIAL_MANIFEST)
    paired, paired_rows = paired_analysis(c0_g0, p1_g0)
    phrase_metrics, phrase_rows = phrase_analysis(p1_g0, manifest)
    severe_metrics, severe_rows = severe_analysis(c0_g0, c0_tf, paired_rows)
    classification = classification_analysis()
    dump(OUT / "predictions/c0/official1000_reference.json", {
        "source": str(C0_OFFICIAL), "raw_spatial_masks_available": False,
        "reason": "Phase 2B historical evaluation predates Phase 3A raw-spatial contract",
    })
    dump(OUT / "predictions/p1/official1000_reference.json", {"source": str(P1_OFFICIAL)})
    dump(OUT / "phrase_analysis/generated_phrase_metrics.json", phrase_metrics)
    dump_jsonl(OUT / "phrase_analysis/per_sample_phrase_analysis.jsonl", phrase_rows)
    dump(OUT / "severe_analysis/severe_metrics.json", severe_metrics)
    dump_jsonl(OUT / "severe_analysis/per_sample_severe.jsonl", severe_rows)
    dump(OUT / "classification/classification_metrics.json", classification)
    dump(OUT / "classification/paired_tests.json", classification["mcnemar"])
    dump(OUT / "statistics/paired_official1000.json", paired)
    dump_jsonl(OUT / "statistics/per_sample_paired.jsonl", paired_rows)
    make_qualitative(paired_rows)

    p1_official = load_json(P1_OFFICIAL / "summary.json")
    p1_internal = load_json(P1_INTERNAL / "summary.json")
    c0_official_metrics = load_json(C0_OFFICIAL / "G0/metrics.json")
    p1_g0_metrics = p1_official["modes"]["G0"]
    p1_tf_metrics = p1_official["modes"]["tf_full_context"]
    p1_phrase_metrics = p1_official["modes"]["phrase_only"]
    selector = load_json(OUT / "selection/p1_selector.json")
    audit = load_json(OUT / "audit/phrase_annotation_audit.json")
    config_diff = load_json(OUT / "configs/config_diff.json")
    report = f"""# Phase 3A：Phrase-Aligned 自回归定位增强

## 1. 执行摘要

Phase 3A 只改变 Fake 语言目标：在完整 forensic explanation 与唯一 `[SEG]` 之间加入来自 `refs.phrase` 的 `Target regions:` 字段。模型结构、union mask、损失、学习率、训练步数、selector、G0 prompt、generation config 与阈值均未改变。用户明确要求不重训 C0，因此训练层面的结论是 **UNPAIRED_TRAINING_COMPARISON**；历史 Phase 2A step 2500 同时作为 C0 与历史参考，不能把全部差异严格归因为 phrase。

正式 P1 checkpoint：step {selector['optimizer_step']} / epoch {selector['epoch']}，selector=`min validation total loss`，official test 在 selector 冻结后才运行。

## 2. 继承的 Phase 2D.1 证据与假设

Phase 2D.1 已确认 G0 与 TF 使用共同的 full-sequence localization downstream，差异已编码在 `[SEG]` predictor representation 中。Phase 3A 检验：显式学习在 `[SEG]` 前生成与 union mask 对应的 authoritative phrase，能否把 TF 条件下的定位能力迁移到真实自由生成 G0。

## 3. 数据与 phrase 审计

仍使用 Phase 2A 的统一 Real/Fake train/val/internal-test split；official1000 仅作冻结后的最终测试。审计原始数据：train Fake={audit['splits']['train']['fake_images']}，val Fake={audit['splits']['val']['fake_images']}，internal test Fake={audit['splits']['test']['fake_images']}；空 phrase 均为 0。多 phrase 按 annotation 顺序、空白归一化、精确重复去重、分号连接；mask 始终沿用原 union mask，不生成 proxy phrase 或新 mask。

## 4. C0/P1 协议与精确差异

C0 是历史 Phase 2A step-2500，不重训。P1 单卡、micro batch=10、gradient accumulation=2、effective global batch=20、5000 optimizer steps、seed=3407、LR=3e-4、linear warmup=100、BF16、ZeRO-2。`max_length=1536` 保持不变；仅对极少数 P1 超长样本保护完整 `Target regions ... [SEG]`，历史 C0 逻辑未改。机器可读 diff 状态：`{config_diff.get('status', 'PASS')}`。

## 5. Checkpoint 选择与训练纪律

P1 在完整 5000 steps 后按冻结的最小 validation total loss 选择 step {selector['optimizer_step']}。official test 未用于训练、继续训练、checkpoint 选择、prompt/template/threshold/generation 调参。

## 6. Internal evaluation

P1 internal G0 mean foreground IoU={fmt(p1_internal['modes']['G0']['per_image_mean']['foreground_iou'])}；TF-full={fmt(p1_internal['modes']['tf_full_context']['per_image_mean']['foreground_iou'])}；phrase-only oracle={fmt(p1_internal['modes']['phrase_only']['per_image_mean']['foreground_iou'])}。G0 是 deployable 主指标，后两者仅为 oracle diagnostic。

## 7. Official1000 G0 主结果

Historical/C0 G0 mean foreground IoU={fmt(c0_official_metrics['mean_iou'])}、mean foreground F1={fmt(c0_official_metrics['mean_pixel_f1'])}。P1 G0 mean foreground IoU={fmt(p1_g0_metrics['per_image_mean']['foreground_iou'])}、mean foreground F1={fmt(p1_g0_metrics['per_image_mean']['foreground_f1'])}、mean fg/bg mIoU={fmt(p1_g0_metrics['per_image_mean']['fg_bg_miou'])}；global-pixel foreground IoU={fmt(p1_g0_metrics['global_pixel']['foreground_iou'])}、foreground F1={fmt(p1_g0_metrics['global_pixel']['foreground_f1'])}、fg/bg mIoU={fmt(p1_g0_metrics['global_pixel']['fg_bg_miou'])}。固定阈值始终为 mask logit > 0，无 sweep。

## 8. Paired effect 与统计

Official1000 同图 paired G0 IoU 差={paired['iou']['mean_difference']:+.6f}，95% bootstrap CI=[{paired['iou']['bootstrap_95ci'][0]:+.6f}, {paired['iou']['bootstrap_95ci'][1]:+.6f}]，win/tie/loss={paired['iou']['wins']}/{paired['iou']['ties']}/{paired['iou']['losses']}，Wilcoxon p={paired['iou']['wilcoxon_pvalue']:.6g}。这是真实 paired evaluation effect，但训练 run 未 paired，因果等级受限。

## 9. Generated phrase 质量与 mask 关联

target presence={phrase_metrics['target_presence_rate']:.4%}，normalized exact match={phrase_metrics['normalized_exact_match_rate']:.4%}，mean token F1={phrase_metrics['mean_normalized_token_f1']:.6f}；phrase-F1 与 G0-IoU Spearman rho={phrase_metrics['spearman_phrase_f1_vs_iou']['rho']:.6f}（p={phrase_metrics['spearman_phrase_f1_vs_iou']['pvalue']:.6g}）。这里使用确定性归一化 token overlap，不使用 LLM evaluator。

## 10. Severe failure 恢复

历史 severe 定义固定为 TF IoU>=0.70 且 G0 IoU<=0.30。历史 severe={severe_metrics['historical_severe_count']}；其 mean G0 IoU 从 {severe_metrics['mean_g0_iou_on_historical_severe']['c0']:.6f} 变为 {severe_metrics['mean_g0_iou_on_historical_severe']['p1']:.6f}；P1 在相同阈值下 severe={severe_metrics['p1_severe_count_under_same_thresholds']}。

## 11. G0 与 oracle diagnostics

P1 official G0={fmt(p1_g0_metrics['per_image_mean']['foreground_iou'])}，TF-full={fmt(p1_tf_metrics['per_image_mean']['foreground_iou'])}，phrase-only={fmt(p1_phrase_metrics['per_image_mean']['foreground_iou'])}。TF-G0 gap={p1_tf_metrics['per_image_mean']['foreground_iou'] - p1_g0_metrics['per_image_mean']['foreground_iou']:.6f}。不能用 oracle 指标替代 G0 成功判断。

## 12. Classification non-regression 与 explanation 审计

Internal paired CLS accuracy：C0={classification['mcnemar']['cls_pred']['c0_accuracy']:.6f}，P1={classification['mcnemar']['cls_pred']['p1_accuracy']:.6f}，McNemar exact p={classification['mcnemar']['cls_pred']['exact_pvalue']:.6g}。LM verdict accuracy：C0={classification['mcnemar']['lm_verdict_pred']['c0_accuracy']:.6f}，P1={classification['mcnemar']['lm_verdict_pred']['p1_accuracy']:.6f}。原始 generation 全量保留，qualitative 页面同时列出 C0/P1 explanation、P1 phrase 与 P1 mask overlay，用于检查 explanation 丢失、重复字段与 hallucination。

## 13. CERTAIN

- 数据/template/config 审计通过；phrase 来自 authoritative `refs.phrase`，mask identity 不变。
- P1 在冻结 selector 和固定 G0/threshold 下的直接测量值、逐图 raw mask/token/phrase 产物与 paired evaluation effect 如上。
- official test 未用于训练或 checkpoint 选择。

## 14. SUPPORTED BUT NON-CAUSAL

- P1 与历史 C0 的差异支持 phrase-aligned protocol 的有效性判断，但因用户决定不重训 paired C0，训练随机性无法完全排除。
- phrase 质量与 mask IoU 的相关性属于机制关联，不是单样本因果证明。

## 15. UNRESOLVED 与 Phase 3B 建议

- 未完成相同初始化、相同时间窗的 C0/P1 paired retraining，因此严格 phrase-only causal gain 未识别。
- C0 历史 official 产物没有 raw spatial logits/mask（Phase 3A 规范建立前生成），其空间产物无法事后补造；P1 已完整保存。
- 不自动启动 Phase 3B。是否进入下一阶段应依据 G0 effect、severe recovery、classification non-regression 与残余 TF-G0 gap共同决定。

## 16. Artifact inventory 与测试

- selector：`outputs/phase3a_phrase_grounding/selection/`
- internal/official evaluation：`outputs/phase3a_phrase_grounding/evaluation/`
- raw spatial predictions：各 mode 的 `spatial/`
- phrase/severe/statistics/classification：对应子目录
- qualitative：`outputs/phase3a_phrase_grounding/qualitative/index.html`
- 正式中文报告：`docs/phase3a_phrase_grounding.md`
- 回归测试结果记录于最终 manifest；正式评测 schema 包含 logits、binary mask、tokens、phrase、TP/FP/FN/TN。

## 17. Experiment discipline

`official_test_used_for_training: false`  
`official_test_used_for_checkpoint_selection: false`  
`proxy_phrase_labels_created: false`  
`threshold_sweep_performed: false`  
`forensic_fusion_used: false`  
`multi_seg_training_used: false`
"""
    report_path = ROOT / "docs/phase3a_phrase_grounding.md"
    report_path.write_text(report, encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    phase_manifest = load_json(manifest_path)
    phase_manifest.update({
        "stage": "completed", "official_test_started": True,
        "official_test_completed": True, "report": str(report_path),
    })
    dump(manifest_path, phase_manifest)
    print(report_path)


if __name__ == "__main__":
    main()
