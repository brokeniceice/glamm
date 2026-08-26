#!/usr/bin/env python3
"""Finalize Phase 4B-G after selected oracle diagnostics (and optional matched control)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.phase4b import dump, file_sha256


def load(path): return json.loads(Path(path).read_text())


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4b_global_fepn_injection.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]
    route=load(out/"route_gate.json"); only=load(out/"selector_proj_only.json"); lora=load(out/"selector_proj_lora.json")
    p1_diag=load(out/"evaluation/final/P1_FROZEN/summary.json"); lora_diag=load(out/"evaluation/final/PROJ-LORA/summary.json")
    phrase_only={"P1":p1_diag["modes"]["phrase_only"],"PROJ-LORA":lora_diag["modes"]["phrase_only"]}
    tf_full={"P1":p1_diag["modes"]["tf_full_context"],"PROJ-LORA":lora_diag["modes"]["tf_full_context"]}
    dump(out/"phrase_only_metrics.json",phrase_only); dump(out/"tf_full_metrics.json",tf_full)
    matched_required=bool(route["matched_sft_triggered"]); matched_done=(out/"paired_bootstrap_proj_lora_vs_sft.json").is_file()
    if matched_required and not matched_done:
        dump(out/"completion_manifest.json",{"status":"MATCHED_SFT_REQUIRED","matched_sft_triggered":True,
          "phase_complete":False,"internal_test_used":False,"official1000_used":False})
        print(json.dumps({"status":"MATCHED_SFT_REQUIRED"})); return
    final_gate=route["gate"]
    if matched_required:
        comparison=load(out/"paired_bootstrap_proj_lora_vs_sft.json")
        evidence_specific=comparison["metrics"]["foreground_iou"]["ci95"][0]>0
        final_gate="GATE_GLOBAL_FORENSIC_EVIDENCE_EFFECTIVE" if evidence_specific else "GATE_GLOBAL_FEPN_GAIN_NOT_SEPARABLE_FROM_CONTINUATION"
    else: evidence_specific=False
    route.update({"status":"COMPLETE","gate":final_gate,"matched_sft_completed":matched_done,
                  "evidence_specific_gain_supported":evidence_specific})
    dump(out/"route_gate.json",route)
    g0=load(out/"g0_metrics.json"); detection=load(out/"detection_metrics.json"); phrase=load(out/"phrase_metrics.json")
    bootstrap=load(out/"paired_bootstrap_proj_lora_vs_p1.json"); io=bootstrap["metrics"]["foreground_iou"]
    report=f"""# Phase 4B-G — Global Forensic Evidence Injection into P1

## Outcome

Final gate: `{final_gate}`.

Phase 4A 成功学习了强 global authenticity representation，但 dense evidence 显著弱于 matched CLIP，因此本阶段只授权 global interface，不把 dense FEPN 接入 SAM/grounding path。

## Frozen interface

FEPN 使用 epoch 6，特征是 `dense_features.mean((2,3))` 的 128D classification-MLP input；它不是 scalar logit 或 P(fake)，GAP 后没有额外 activation/normalization。固定 projector 为 `128→256→4×4096`，共 4,243,712 参数。四个 continuous tokens 位于 576 个原视觉 embedding 后、canonical text 前，真实进入 autonomous generation 初始上下文与 KV cache；tokenizer、raw token IDs、embedding table 和 LM head 均未修改。

## Selected checkpoints

- PROJ-ONLY: step {only['optimizer_step']}.
- PROJ-LORA: step {lora['optimizer_step']}.
- P1 remains the original step 3500 / epoch 7 baseline.

## Validation results

| arm | Detection Acc | Detection F1 | G0 mean FG IoU | G0 mean FG F1 | phrase semantic | structure | SEG-valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | {detection['P1']['classification_accuracy']:.6f} | {detection['P1']['classification_f1']:.6f} | {g0['P1']['mean_foreground_iou']:.6f} | {g0['P1']['mean_foreground_f1']:.6f} | {phrase['P1']['R_phrase_sem']:.6f} | {phrase['P1']['structure_validity']:.6f} | {phrase['P1']['valid_seg_rate']:.6f} |
| PROJ-ONLY | {detection['PROJ-ONLY']['classification_accuracy']:.6f} | {detection['PROJ-ONLY']['classification_f1']:.6f} | {g0['PROJ-ONLY']['mean_foreground_iou']:.6f} | {g0['PROJ-ONLY']['mean_foreground_f1']:.6f} | {phrase['PROJ-ONLY']['R_phrase_sem']:.6f} | {phrase['PROJ-ONLY']['structure_validity']:.6f} | {phrase['PROJ-ONLY']['valid_seg_rate']:.6f} |
| PROJ-LORA | {detection['PROJ-LORA']['classification_accuracy']:.6f} | {detection['PROJ-LORA']['classification_f1']:.6f} | {g0['PROJ-LORA']['mean_foreground_iou']:.6f} | {g0['PROJ-LORA']['mean_foreground_f1']:.6f} | {phrase['PROJ-LORA']['R_phrase_sem']:.6f} | {phrase['PROJ-LORA']['structure_validity']:.6f} | {phrase['PROJ-LORA']['valid_seg_rate']:.6f} |

PROJ-LORA vs P1 paired G0 IoU delta is {io['delta']:+.6f}, 95% CI [{io['ci95'][0]:+.6f}, {io['ci95'][1]:+.6f}]. Phrase-Only and TF-Full are recorded in `phrase_only_metrics.json` and `tf_full_metrics.json`; they are diagnostics rather than selector inputs.

## Attribution and boundaries

PROJ-ONLY answers whether original P1 can consume a mapped representation without language-path adaptation; PROJ-LORA tests whether LoRA is needed. Classification, phrase, structure and G0 are reported separately so a binary authenticity shortcut is not mislabeled as rich forensic reasoning. Matched SFT trigger: {matched_required}; completed: {matched_done}. Evidence-specific attribution supported: {evidence_specific}.

All selector and statistics use internal validation, canonical prompt, direct batch=1 and fixed mask-logit threshold 0. Internal test and official1000 remained sealed. No K/depth/LR sweep, FEPN fine-tuning, dense integration, SAM fusion or threshold tuning was performed.

## Route decision

The next FEPN decision must follow `{final_gate}`. This phase does not support claims that dense FEPN is established, that the global feature precisely localizes artifacts, or that FEPN resolves the full localization bottleneck. Per the hard stop, no Phase 4C or further variant starts automatically.
"""
    (out/"final_report.md").write_text(report,encoding="utf-8")
    docs=ROOT/"docs/phase4b_global_fepn_injection.md"; docs.write_text(report,encoding="utf-8")
    living=ROOT/"docs/fepn_design_route.md"
    living_text=living.read_text(encoding="utf-8")
    marker="## Version 0.4 — Phase 4B-G outcome"
    if marker not in living_text:
        living.write_text(living_text+f"\n\n{marker}\n\n- Final gate: `{final_gate}`.\n- Selected PROJ-ONLY step: {only['optimizer_step']}; selected PROJ-LORA step: {lora['optimizer_step']}.\n- PROJ-LORA vs P1 G0 mean FG IoU delta: {io['delta']:+.6f}, 95% CI [{io['ci95'][0]:+.6f}, {io['ci95'][1]:+.6f}].\n- Matched SFT trigger: {matched_required}; evidence-specific attribution supported: {evidence_specific}.\n- Internal test and official1000 remained sealed. The Phase 4B-G hard stop is active; the next route requires new explicit authorization.\n",encoding="utf-8")
    paper=ROOT/"docs/paper_convergence_route.md"; paper_text=paper.read_text(encoding="utf-8")
    paper_marker="### Phase 4B-G terminal outcome"
    if paper_marker not in paper_text:
        paper.write_text(paper_text+f"\n\n{paper_marker}\n\nFinal gate: `{final_gate}`. Selected PROJ-ONLY/PROJ-LORA steps were {only['optimizer_step']}/{lora['optimizer_step']}. PROJ-LORA vs P1 validation canonical G0 mean FG IoU delta was `{io['delta']:+.6f}`, paired-bootstrap 95% CI `[{io['ci95'][0]:+.6f}, {io['ci95'][1]:+.6f}]`. Matched SFT trigger: {matched_required}; evidence-specific attribution supported: {evidence_specific}. No internal test, official1000, threshold tuning, dense FEPN integration, or post-Phase-4B-G experiment was run.\n",encoding="utf-8")
    required=["experiment_manifest.json","initial_checkpoint_manifest.json","fepn_checkpoint_manifest.json",
      "fepn_global_feature_spec.json","fepn_global_feature_cache_manifest.json","interface_architecture.json","projector_config.json",
      "training_config.json","preflight_feature_audit.json","preflight_sequence_integrity.json","checkpoint_metrics_proj_only.json",
      "checkpoint_metrics_proj_lora.json","selector_proj_only.json","selector_proj_lora.json","detection_metrics.json","g0_metrics.json",
      "phrase_metrics.json","phrase_only_metrics.json","tf_full_metrics.json","paired_bootstrap_proj_lora_vs_p1.json",
      "paired_bootstrap_proj_lora_vs_proj_only.json","qualitative_analysis.md","route_gate.json","final_report.md"]
    missing=[name for name in required if not (out/name).is_file()]
    completion={"status":"COMPLETE" if not missing else "INCOMPLETE","completed_utc":datetime.now(timezone.utc).isoformat(),
      "phase":"Phase 4B-G","final_gate":final_gate,"selected_steps":{"PROJ-ONLY":only["optimizer_step"],"PROJ-LORA":lora["optimizer_step"]},
      "matched_sft_triggered":matched_required,"matched_sft_completed":matched_done,"missing":missing,
      "artifacts":{name:{"path":str((out/name).resolve()),"sha256":file_sha256(out/name)} for name in required if (out/name).is_file()},
      "internal_test_used":False,"official1000_used":False,"hard_stop":True}
    dump(out/"completion_manifest.json",completion); print(json.dumps(completion,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
