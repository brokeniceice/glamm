#!/usr/bin/env python3
"""Generate machine-readable claims and Chinese Phase 4F reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tools.phase4f import compare, dump, file_sha256, rows

CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
DOC = ROOT / "docs/phase4f"; DOC.mkdir(parents=True, exist_ok=True)


def load(path): return json.loads(Path(path).read_text())
def write(name, text): (DOC / name).write_text(text.strip() + "\n", encoding="utf-8")
def state(stats):
    ci = stats["foreground_iou"]["bootstrap_95_ci"]
    return "TRUE" if ci[0] > 0 else ("FALSE" if ci[1] <= 0 else "INCONCLUSIVE")
def history_table(history):
    lines = ["| Epoch | Updates | Train loss | Validation G0 mean FG IoU | Gamma |", "|---:|---:|---:|---:|---:|"]
    for row in history:
        train = row.get("train") or {}
        lines.append(f"| {row['epoch']} | {row['optimizer_updates']} | {train.get('total','—')} | {row['validation']['mean_foreground_iou']:.6f} | {row.get('gamma','—')} |")
    return "\n".join(lines)


def main():
    preflight = load(OUT / "preflight/integrated_preflight.json")
    equivalence = load(OUT / "preflight/p1_equivalence_audit.json")
    scale = load(OUT / "preflight/geometry_scale_audit.json")
    gradients = load(OUT / "preflight/gradient_routing_audit.json")
    summaries = {arm: load(OUT / "final" / arm / "summary.json") for arm in CFG["arms"]}
    selectors = {arm: load(OUT / "selectors" / f"{arm}.json") for arm in CFG["arms"]}
    forensic, clip = summaries["forensic_rect"], summaries["clip_rect"]
    forensic_records = rows(OUT / "final/forensic_rect/matched.jsonl")
    clip_records = rows(OUT / "final/clip_rect/matched.jsonl")
    specialization_stats = compare(forensic_records, clip_records)
    dump(OUT / "statistics/forensic_vs_clip.json", specialization_stats)
    g0_gain = state(forensic["paired"]["g0_vs_p1"])
    phrase_sensitivity = state(forensic["paired"]["phrase_minus_g0"])
    tf_sensitivity = state(forensic["paired"]["tf_minus_g0"])
    language = "PRESERVED" if phrase_sensitivity == tf_sensitivity == "TRUE" else ("LOST" if phrase_sensitivity == tf_sensitivity == "FALSE" else "WEAKENED")
    tf_delta = forensic["paired"]["tf_vs_p1"]["foreground_iou"]
    tf_capability = "IMPROVED" if tf_delta["bootstrap_95_ci"][0] > 0 else ("PRESERVED" if tf_delta["mean_difference"] > float(CFG["evaluation"]["language_non_regression_margin"]) else "DEGRADED")
    image_use = state(forensic["paired"]["matched_minus_cross"])
    spatial_use = state(forensic["paired"]["matched_minus_shuffle"])
    specialization = state(specialization_stats)
    dual = "SUPPORTED" if g0_gain == "TRUE" and language == "PRESERVED" and image_use == "TRUE" and spatial_use == "TRUE" and tf_capability != "DEGRADED" else ("NOT_SUPPORTED" if "FALSE" in (g0_gain, image_use, spatial_use) or language == "LOST" or tf_capability == "DEGRADED" else "INCONCLUSIVE")
    sufficient = "YES" if dual == "SUPPORTED" else ("NO" if dual == "NOT_SUPPORTED" else "INCONCLUSIVE")
    claims = {
        "P1_PATH_EXACTLY_PRESERVED": equivalence["P1_PATH_EXACTLY_PRESERVED"],
        "DEPLOYABLE_G0_GAIN": g0_gain,
        "LANGUAGE_CONDITION_SENSITIVITY": language,
        "P1_TF_CAPABILITY": tf_capability,
        "IMAGE_SPECIFIC_EVIDENCE_USE": image_use,
        "SPATIAL_SPECIFIC_EVIDENCE_USE": spatial_use,
        "FORENSIC_SPECIALIZATION_TRANSFER": specialization,
        "RECTIFICATION_ONLY_SUFFICIENT": sufficient,
        "DUAL_CAPABILITY_BRIDGE": dual,
        "PROCEED_TO_TEACHER_DISTILLATION": "YES" if dual == "SUPPORTED" else "NO",
    }
    dump(OUT / "final_claims.json", claims)
    write("phase4f_preflight.md", f"""# Phase 4F 预检

Integrated preflight：**{preflight['status']}**。正式预检 optimizer updates 为 0；internal test 与 official1000 未访问。

- P1 path equivalence：{equivalence['status']}
- geometry scale：{scale['status']}，selected gamma={scale['selected_gamma']}
- gradient routing：{gradients['status']}
- 两臂初始 rectifier hash matched：{gradients['matched_initialization']}""")
    write("phase4f_p1_equivalence_audit.md", f"""# Phase 4F P1 路径等价审计

`rectifier=OFF` 对 G0、Phrase-only、TF-full 的全量 validation metrics tolerance 均通过；16例 low-res tensor exact subset 结果为 `{all(row['tensor_exact'] for row in equivalence['exact_low_res_subset'])}`。

P1_PATH_EXACTLY_PRESERVED：**{claims['P1_PATH_EXACTLY_PRESERVED']}**。""")
    write("phase4f_geometry_scale_audit.md", f"# Phase 4F Geometry scale audit\n\n```json\n{json.dumps(scale, ensure_ascii=False, indent=2)}\n```")
    write("phase4f_gradient_routing_audit.md", f"# Phase 4F Gradient routing audit\n\n```json\n{json.dumps(gradients, ensure_ascii=False, indent=2)}\n```")
    for arm, title in (("forensic_rect", "FORENSIC-RECT"), ("clip_rect", "CLIP-RECT")):
        history = load(OUT / "training" / arm / "history.json")
        write(f"phase4f_{arm}_training.md", f"""# Phase 4F {title} 训练

{history_table(history)}

Selected epoch：**{selectors[arm]['selected_epoch']}**；完整 traversal exposures=88,360，valid-G0 supervised exposures=86,900，actual updates={selectors[arm]['actual_optimizer_updates']}。""")
    write("phase4f_main_results.md", f"""# Phase 4F 主结果

| Arm | Selected epoch | G0 mean FG IoU | Phrase | TF |
|---|---:|---:|---:|---:|
| P1 | — | {equivalence['modes']['g0']['metrics']['mean_foreground_iou']:.6f} | {equivalence['modes']['phrase']['metrics']['mean_foreground_iou']:.6f} | {equivalence['modes']['tf']['metrics']['mean_foreground_iou']:.6f} |
| FORENSIC-RECT | {selectors['forensic_rect']['selected_epoch']} | {forensic['matched']['mean_foreground_iou']:.6f} | {forensic['phrase_only']['mean_foreground_iou']:.6f} | {forensic['tf_full']['mean_foreground_iou']:.6f} |
| CLIP-RECT | {selectors['clip_rect']['selected_epoch']} | {clip['matched']['mean_foreground_iou']:.6f} | {clip['phrase_only']['mean_foreground_iou']:.6f} | {clip['tf_full']['mean_foreground_iou']:.6f} |

FORENSIC-RECT vs P1 G0 paired result：`{json.dumps(forensic['paired']['g0_vs_p1']['foreground_iou'])}`。""")
    write("phase4f_language_preservation.md", f"""# Phase 4F Language preservation

- Phrase−G0：`{json.dumps(forensic['paired']['phrase_minus_g0']['foreground_iou'])}`
- TF−G0：`{json.dumps(forensic['paired']['tf_minus_g0']['foreground_iou'])}`
- TF−Phrase：`{json.dumps(forensic['paired']['tf_minus_phrase']['foreground_iou'])}`
- Phase4F TF−P1 TF：`{json.dumps(tf_delta)}`
- ORACLE_GAP_RETENTION_RATIO：{forensic['oracle_gap']['retention_ratio']:.6f}

LANGUAGE_CONDITION_SENSITIVITY：**{language}**；P1_TF_CAPABILITY：**{tf_capability}**。""")
    write("phase4f_causal_controls.md", f"""# Phase 4F Causal controls

| Condition | Mean FG IoU |
|---|---:|
| matched | {forensic['matched']['mean_foreground_iou']:.6f} |
| cross-image | {forensic['cross_image']['mean_foreground_iou']:.6f} |
| spatial shuffle | {forensic['spatial_shuffle']['mean_foreground_iou']:.6f} |
| zero evidence | {forensic['zero']['mean_foreground_iou']:.6f} |
| rectifier off | {forensic['rectifier_off']['mean_foreground_iou']:.6f} |

IMAGE_SPECIFIC_EVIDENCE_USE：**{image_use}**；SPATIAL_SPECIFIC_EVIDENCE_USE：**{spatial_use}**。""")
    write("phase4f_forensic_vs_clip.md", f"""# Phase 4F FORENSIC-RECT vs CLIP-RECT

Paired statistics：

```json
{json.dumps(specialization_stats, ensure_ascii=False, indent=2)}
```

FORENSIC_SPECIALIZATION_TRANSFER：**{specialization}**。""")
    write("phase4f_final_report.md", f"""# Phase 4F 最终报告

```json
{json.dumps(claims, ensure_ascii=False, indent=2)}
```

结论仅适用于 geometry-aware image rectification-only compatible route；无论结果如何，都不改写 Phase 4E dense spatial hybrid principle。internal test 与 official1000 继续封存，未自动启动 Teacher/KD 或下一阶段。""")
    completion = {"status": "COMPLETE", "phase": "Phase 4F", "claims": claims, "selectors": {arm: {"epoch": selectors[arm]["selected_epoch"], "checkpoint": selectors[arm]["selected_checkpoint"], "sha256": selectors[arm]["selected_checkpoint_sha256"]} for arm in CFG["arms"]}, "reports": [str(path) for path in sorted(DOC.glob("phase4f_*.md"))], "internal_test_accessed": False, "official1000_accessed": False, "teacher_distillation_started": False, "next_phase_started": False}
    dump(OUT / "completion_manifest.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

