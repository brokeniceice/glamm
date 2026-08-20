#!/usr/bin/env python3
"""Finalize paired Phase 3B statistics, diagnostics, qualitative cases and Chinese report."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3b_replay import phrase_overlap, replay_eligibility

OUT = ROOT / "outputs/phase3b_generated_replay"
ARMS = ("b0_gold_replay", "b1_generated_replay")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def dump_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for value in values: h.write(json.dumps(value, ensure_ascii=False) + "\n")


def paired(differences, *, seed=3407, repeats=10000):
    x = np.asarray(differences, dtype=np.float64); rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for start in range(0, repeats, 500):
        n = min(500, repeats - start); idx = rng.integers(0, x.size, size=(n, x.size)); boot[start:start+n] = x[idx].mean(1)
    nonzero = x[np.abs(x) > 1e-12]; test = stats.wilcoxon(nonzero) if nonzero.size else None
    return {"n": int(x.size), "mean_difference": float(x.mean()), "median_difference": float(np.median(x)),
            "bootstrap_repeats": repeats, "bootstrap_seed": seed,
            "bootstrap_95ci": [float(v) for v in np.quantile(boot, (.025, .975))],
            "wins": int((x > 1e-12).sum()), "ties": int((np.abs(x) <= 1e-12).sum()), "losses": int((x < -1e-12).sum()),
            "wilcoxon_statistic": None if test is None else float(test.statistic),
            "wilcoxon_pvalue": None if test is None else float(test.pvalue)}


def prediction_maps(split, mode="G0"):
    return {arm: {r["sample_id"]: r for r in rows(OUT / f"evaluation/{split}/{arm}/{mode}/predictions.jsonl")} for arm in ARMS}


def generation_summary(predictions, manifest):
    values = []
    for row in predictions.values():
        reference = UnifiedForensicsDataset.authoritative_localization_field(manifest[row["sample_id"]])["normalized_training_phrase"]
        phrase = phrase_overlap(reference, row.get("generated_localization_phrase"))
        values.append((row, phrase))
    lengths = [len(row.get("generated_token_ids") or []) for row, _ in values]
    return {"n": len(values), "seg_trigger_rate": sum(bool(r["seg_triggered"] and r["has_pred_mask"]) for r, _ in values)/len(values),
            "target_presence_rate": sum(r.get("generated_localization_phrase") is not None for r, _ in values)/len(values),
            "normalized_exact_rate": sum(p["normalized_exact_match"] for _, p in values)/len(values),
            "mean_token_f1": sum(p["normalized_token_f1"] for _, p in values)/len(values),
            "generation_length_mean": float(np.mean(lengths)), "generation_length_median": float(np.median(lengths)),
            "generation_length_max": max(lengths)}


def subgroup_analysis(split, b0, b1, manifest, p1_path, severe_ids):
    p1 = {r["sample_id"]: r for r in rows(p1_path)}
    lengths = np.array([len(r.get("generated_token_ids") or []) for r in p1.values()]); median = float(np.median(lengths))
    groups = {name: [] for name in ("phrase_high_quality", "phrase_low_quality", "phrase_present", "phrase_absent",
                                    "generation_short", "generation_long", "replay_eligible", "replay_ineligible",
                                    "historical_severe", "historical_nonsevere")}
    per_sample = []
    for sid in sorted(set(b0) & set(b1) & set(p1)):
        baseline = p1[sid]; generated_ids = baseline.get("generated_token_ids") or []
        eligibility = replay_eligibility(generated_ids, 32004)["replay_eligible"]
        reference = UnifiedForensicsDataset.authoritative_localization_field(manifest[sid])["normalized_training_phrase"]
        quality = phrase_overlap(reference, baseline.get("generated_localization_phrase"))["normalized_token_f1"]
        length = len(generated_ids); difference = b1[sid]["foreground_iou"] - b0[sid]["foreground_iou"]
        labels = ["phrase_high_quality" if quality >= .8 else "phrase_low_quality",
                  "phrase_present" if baseline.get("generated_localization_phrase") is not None else "phrase_absent",
                  "generation_short" if length <= median else "generation_long",
                  "replay_eligible" if eligibility else "replay_ineligible",
                  "historical_severe" if sid in severe_ids else "historical_nonsevere"]
        for label in labels: groups[label].append(difference)
        per_sample.append({"sample_id": sid, "split": split, "frozen_p1_phrase_token_f1": quality,
                           "frozen_p1_generation_length": length, "frozen_p1_replay_eligible": eligibility,
                           "historical_severe": sid in severe_ids, "b1_minus_b0_foreground_iou": difference})
    summary = {name: (paired(values) if values else {"n": 0}) for name, values in groups.items()}
    summary["frozen_p1_generation_length_median_split_threshold"] = median
    return summary, per_sample


def main():
    selectors = {arm: load(OUT / f"selection/{arm}_selector.json") for arm in ARMS}
    manifests = {
        "internal": {r["sample_id"]: r for r in rows(ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl")},
        "official1000": {r["sample_id"]: r for r in rows(ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl")},
    }
    p1_paths = {"internal": ROOT / "outputs/phase3a_phrase_grounding/evaluation/internal/G0/predictions.jsonl",
                "official1000": ROOT / "outputs/phase3a_phrase_grounding/evaluation/official1000/G0/predictions.jsonl"}
    severe_rows = rows(ROOT / "outputs/phase3a1_paired_control/severe/historical_severe_per_sample.jsonl")
    severe_ids = {r["sample_id"] for r in severe_rows}
    all_stats = {}; generation = {}; subgroup = {}; subgroup_rows = []
    for split in ("internal", "official1000"):
        maps = prediction_maps(split)
        common = sorted(set(maps[ARMS[0]]) & set(maps[ARMS[1]]))
        metrics = {}
        for metric in ("foreground_iou", "foreground_f1", "fg_bg_miou"):
            metrics[metric] = paired([maps[ARMS[1]][sid][metric] - maps[ARMS[0]][sid][metric] for sid in common])
        all_stats[split] = metrics
        generation[split] = {arm: generation_summary(maps[arm], manifests[split]) for arm in ARMS}
        sg, per = subgroup_analysis(split, maps[ARMS[0]], maps[ARMS[1]], manifests[split], p1_paths[split], severe_ids)
        subgroup[split] = sg; subgroup_rows.extend(per)
    dump(OUT / "statistics/paired_g0.json", all_stats)
    dump(OUT / "statistics/bootstrap.json", {s: {m: v["bootstrap_95ci"] for m,v in x.items()} for s,x in all_stats.items()})
    dump(OUT / "statistics/wilcoxon.json", {s: {m: {"statistic":v["wilcoxon_statistic"],"pvalue":v["wilcoxon_pvalue"]} for m,v in x.items()} for s,x in all_stats.items()})
    dump(OUT / "statistics/win_tie_loss.json", {s: {m: {k:v[k] for k in ("wins","ties","losses")} for m,v in x.items()} for s,x in all_stats.items()})
    dump(OUT / "generation_diagnostics/comparison.json", generation)
    dump(OUT / "statistics/replay_subgroups.json", subgroup); dump_rows(OUT / "statistics/replay_subgroups_per_sample.jsonl", subgroup_rows)
    classification = {}
    for split in ("internal", "official1000"):
        classification[split] = {arm: load(OUT / f"evaluation/{split}/{arm}/detection/metrics.json") for arm in ARMS}
    b0_acc = classification["internal"][ARMS[0]]["classification_head"]["accuracy"]
    b1_acc = classification["internal"][ARMS[1]]["classification_head"]["accuracy"]
    nonreg = {"internal_b1_minus_b0_cls_accuracy": b1_acc-b0_acc, "non_regression_threshold": -.005,
              "passes": b1_acc-b0_acc >= -.005}
    dump(OUT / "statistics/classification_non_regression.json", {"metrics":classification, **nonreg})
    severe = {}
    official = prediction_maps("official1000")
    p1 = {r["sample_id"]: r for r in rows(p1_paths["official1000"])}
    fixed = sorted(severe_ids & set(official[ARMS[0]]) & set(official[ARMS[1]]))
    severe["num_fixed_historical_severe"] = len(fixed)
    severe["mean_foreground_iou"] = {"p1_historical": float(np.mean([p1[s]["foreground_iou"] for s in fixed])),
                                     **{arm: float(np.mean([official[arm][s]["foreground_iou"] for s in fixed])) for arm in ARMS}}
    severe["b1_minus_b0"] = paired([official[ARMS[1]][s]["foreground_iou"]-official[ARMS[0]][s]["foreground_iou"] for s in fixed])
    dump(OUT / "severe119/fixed_historical_severe119.json", severe)
    tf_gap = {}
    for split in ("internal", "official1000"):
        tf_gap[split] = {}
        for arm in ARMS:
            g0 = load(OUT / f"evaluation/{split}/{arm}/G0/metrics.json")["per_image_mean"]["foreground_iou"]
            tf = load(OUT / f"evaluation/{split}/{arm}/tf_full_context/metrics.json")["per_image_mean"]["foreground_iou"]
            tf_gap[split][arm] = {"g0":g0,"tf_phrase":tf,"tf_minus_g0_gap":tf-g0}
        tf_gap[split]["delta_gap_b1_minus_b0"] = tf_gap[split][ARMS[1]]["tf_minus_g0_gap"]-tf_gap[split][ARMS[0]]["tf_minus_g0_gap"]
    dump(OUT / "statistics/tf_g0_gap.json", tf_gap)
    official_iou = all_stats["official1000"]["foreground_iou"]; internal_iou = all_stats["internal"]["foreground_iou"]
    no_collapse = generation["official1000"][ARMS[1]]["seg_trigger_rate"] >= generation["official1000"][ARMS[0]]["seg_trigger_rate"]-.01
    if official_iou["bootstrap_95ci"][1] < 0 or not nonreg["passes"]:
        gate = "GENERATED_REPLAY_HARMFUL"
    elif official_iou["bootstrap_95ci"][0] > 0 and internal_iou["mean_difference"] > 0 and no_collapse:
        gate = "GENERATED_REPLAY_STRONGLY_SUPPORTED"
    elif official_iou["mean_difference"] > 0 and internal_iou["mean_difference"] >= 0 and nonreg["passes"] and no_collapse:
        gate = "GENERATED_REPLAY_WEAKLY_SUPPORTED"
    else:
        gate = "GENERATED_REPLAY_NOT_SUPPORTED"
    dump(OUT / "statistics/final_gate.json", {"gate":gate,"classification_non_regression":nonreg["passes"],
                                               "no_generation_collapse":no_collapse,
                                               "official_iou":official_iou,"internal_iou":internal_iou})
    per_official = [{"sample_id":sid,"image_path":official[ARMS[0]][sid]["image_path"],
                     "b0_iou":official[ARMS[0]][sid]["foreground_iou"],"b1_iou":official[ARMS[1]][sid]["foreground_iou"],
                     "difference":official[ARMS[1]][sid]["foreground_iou"]-official[ARMS[0]][sid]["foreground_iou"]}
                    for sid in sorted(official[ARMS[0]])]
    per_official.sort(key=lambda x:x["difference"], reverse=True)
    dump_rows(OUT / "qualitative/rescue_cases.jsonl", per_official[:20]); dump_rows(OUT / "qualitative/regression_cases.jsonl", per_official[-20:])
    cache = load(OUT / "cache/cache_stats.json")
    val = {arm: selectors[arm]["selected_metrics"] for arm in ARMS}
    def metric(split,arm): return load(OUT / f"evaluation/{split}/{arm}/G0/metrics.json")["per_image_mean"]
    report = f"""# Phase 3B：固定策略 Generated-Context Replay 的配对实验

## 1. P1 source checkpoint / hash

两臂均从冻结 P1 step 3500 / epoch 7 继续：`{selectors[ARMS[0]]['selected_metrics'].get('source_checkpoint','/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt')}`；P1 SHA256=`fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。两臂使用 fresh、相同的 AdamW/scheduler 状态，模型可训练权重 zero-step hash 另见 audit。

## 2. Replay cache eligibility

内部 train Fake={cache['num_train_fake']}，eligible={cache['num_replay_eligible']}（{cache['replay_eligible_rate']:.4%}）；唯一资格规则是恰有一个、且前方存在 predictor token 的 `[SEG]`。NoSEG 不补写、不纠正、不进入辅助分支。cache SHA256=`{cache['cache_sha256']}`。

## 3. B0 selected step

B0 选择 step {selectors[ARMS[0]]['optimizer_step']} / epoch {selectors[ARMS[0]]['epoch']}，checkpoint SHA256=`{selectors[ARMS[0]]['checkpoint_sha256']}`。

## 4. B1 selected step

B1 选择 step {selectors[ARMS[1]]['optimizer_step']} / epoch {selectors[ARMS[1]]['epoch']}，checkpoint SHA256=`{selectors[ARMS[1]]['checkpoint_sha256']}`。

## 5. Validation selector values

| Arm | val Fake G0 mean FG IoU | mean FG F1 |
|---|---:|---:|
| B0 gold replay | {val[ARMS[0]]['val_fake_g0_mean_foreground_iou']:.6f} | {val[ARMS[0]]['val_fake_g0_mean_foreground_f1']:.6f} |
| B1 generated replay | {val[ARMS[1]]['val_fake_g0_mean_foreground_iou']:.6f} | {val[ARMS[1]]['val_fake_g0_mean_foreground_f1']:.6f} |

Selector 仅使用 internal val Fake G0；internal test、official1000 和 TF 均未参与选模。

## 6. Internal test B0/B1 G0

| Arm | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| B0 | {metric('internal',ARMS[0])['foreground_iou']:.6f} | {metric('internal',ARMS[0])['foreground_f1']:.6f} | {metric('internal',ARMS[0])['fg_bg_miou']:.6f} |
| B1 | {metric('internal',ARMS[1])['foreground_iou']:.6f} | {metric('internal',ARMS[1])['foreground_f1']:.6f} | {metric('internal',ARMS[1])['fg_bg_miou']:.6f} |

## 7. Official1000 B0/B1 G0

| Arm | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| B0 | {metric('official1000',ARMS[0])['foreground_iou']:.6f} | {metric('official1000',ARMS[0])['foreground_f1']:.6f} | {metric('official1000',ARMS[0])['fg_bg_miou']:.6f} |
| B1 | {metric('official1000',ARMS[1])['foreground_iou']:.6f} | {metric('official1000',ARMS[1])['foreground_f1']:.6f} | {metric('official1000',ARMS[1])['fg_bg_miou']:.6f} |

## 8. Paired B1−B0 statistics

Official FG IoU mean={official_iou['mean_difference']:+.6f}，median={official_iou['median_difference']:+.6f}，bootstrap 95% CI=[{official_iou['bootstrap_95ci'][0]:+.6f}, {official_iou['bootstrap_95ci'][1]:+.6f}]，win/tie/loss={official_iou['wins']}/{official_iou['ties']}/{official_iou['losses']}，Wilcoxon p={official_iou['wilcoxon_pvalue']:.6g}。Internal mean={internal_iou['mean_difference']:+.6f}，95% CI=[{internal_iou['bootstrap_95ci'][0]:+.6f}, {internal_iou['bootstrap_95ci'][1]:+.6f}]。

## 9. Classification non-regression

Internal CLS accuracy B1−B0={nonreg['internal_b1_minus_b0_cls_accuracy']:+.6f}，预注册容忍下限 −0.005，pass={nonreg['passes']}。完整 CLS/LM/F1/agreement 见 `statistics/classification_non_regression.json`。

## 10. SEG / phrase / generation length

完整 internal 与 official1000 的 `[SEG]` trigger、target presence、exact、token F1、长度对照见 `generation_diagnostics/comparison.json`；no-generation-collapse={no_collapse}。

## 11. B0/B1 TF-PHRASE

Official TF-PHRASE mean FG IoU：B0={tf_gap['official1000'][ARMS[0]]['tf_phrase']:.6f}，B1={tf_gap['official1000'][ARMS[1]]['tf_phrase']:.6f}。TF 是 oracle diagnostic，不替代 G0。

## 12. TF−G0 gap / delta-gap

Official gap：B0={tf_gap['official1000'][ARMS[0]]['tf_minus_g0_gap']:+.6f}，B1={tf_gap['official1000'][ARMS[1]]['tf_minus_g0_gap']:+.6f}，delta-gap={tf_gap['official1000']['delta_gap_b1_minus_b0']:+.6f}。负 delta-gap 仅作为 exposure-gap 缩小的 secondary evidence。

## 13. Fixed severe119

固定 historical severe-119（实际匹配 {severe['num_fixed_historical_severe']}）：P1/B0/B1 mean FG IoU={severe['mean_foreground_iou']['p1_historical']:.6f}/{severe['mean_foreground_iou'][ARMS[0]]:.6f}/{severe['mean_foreground_iou'][ARMS[1]]:.6f}；B1−B0 mean={severe['b1_minus_b0']['mean_difference']:+.6f}，95% CI={severe['b1_minus_b0']['bootstrap_95ci']}。

## 14. Replay subgroup analysis

按冻结 P1 在对应 eval 样本上的生成属性进行 post-hoc 分组，结果见 `statistics/replay_subgroups.json`。这些结果全部是 exploratory，未用于 selector 或调参；重点比较 `phrase_low_quality` 且 `replay_eligible` 子群，但不可将训练 cache 属性错误地逐样本套到 test。

## 15. Artifact paths

完整 artifact 位于 `outputs/phase3b_generated_replay/` 的 cache、audit、training、selection、evaluation、statistics、severe119、generation_diagnostics、qualitative、reports。checkpoint 实体位于 `/data/yz/groundingLMM_official/checkpoints/phase3b_generated_replay/`。

## 16. Regression tests

测试结果写入 `reports/regression_tests.txt`。核心 Phase 3B 契约与全部既有 regression 均须通过后才标记 COMPLETE。

## 17. Final gate

**`{gate}`**。

本阶段未使用 NPR/SRM、FOCAL、SAM 架构修改、新 decoder、新 phrase/mask、mask refinement、threshold sweep、动态 replay、on-policy generation、scheduled sampling、consistency loss、TF distillation 或 test-set 选模。Phase 3C 未启动。
"""
    report_path = OUT / "reports/phase3b_generated_replay.md"; report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8"); shutil.copy2(report_path, ROOT / "docs/phase3b_generated_replay.md")
    dump(OUT / "manifest.json", {"stage":"ANALYSIS_COMPLETE_REGRESSION_PENDING","gate":gate,
                                  "phase3c_started":False,"npr_srm_used":False,"focal_used":False,
                                  "official_test_used_for_selection":False,"threshold_sweep":False})
    print(gate)


if __name__ == "__main__": main()
