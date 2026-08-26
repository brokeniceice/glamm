#!/usr/bin/env python3
"""Real-model post-training output-invariance regression for Phase 2C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from scripts.phase2c_forensic_fusion import OUTPUT_ROOT, load_config, load_fusion, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def tensor_diff(left, right):
    if left is None or right is None:
        return {"equal": left is None and right is None, "max_abs": None}
    return {"equal": bool(torch.equal(left, right)), "max_abs": float((left.float() - right.float()).abs().max())}


def snapshot(backend, sample):
    detection_batch = backend._batch(sample, "")
    detection_batch["grounding_enc_images"] = None
    lm_logits = []
    hook = backend.model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: lm_logits.append(output.detach().cpu())
    )
    try:
        with torch.no_grad(): detection = backend.model.model_forward(**detection_batch)
    finally:
        hook.remove()
    g0 = backend.generate_localization(sample, provide_gt_fake=False, generation_mode="unified_fake_generate")
    g1 = backend.generate_localization(sample, provide_gt_fake=True,
                                       generation_mode="unified_prompt_gt_fake_prefix")
    tf = backend.teacher_forced_localization(
        sample, context="full", user_prompt="canonical"
    )
    return {
        "lm_token_logits": lm_logits[0],
        "base_logits": detection["cls_logits"].detach().cpu(),
        "h_cls_unavailable_note": "Fusion feature is read from frozen cache; no model parameter is modified.",
        "g0_token_ids": g0["generated_token_ids"], "g0_mask": None if g0["pred_mask"] is None else g0["pred_mask"].cpu(),
        "g1_token_ids": g1["generated_token_ids"], "g1_mask": None if g1["pred_mask"] is None else g1["pred_mask"].cpu(),
        "tf_mask": None if tf["pred_mask"] is None else tf["pred_mask"].cpu(),
    }


def main(argv=None):
    cli = parse_args(argv)
    device = torch.device(cli.device)
    phase2a = load_config("configs/phase2a_unified_baseline_full.yaml")
    model, tokenizer, _ = load_model(
        phase2a, REPO_ROOT / phase2a["checkpoint"]["output_root"] / "best/checkpoint/mp_rank_00_model_states.pt",
        device, expected_step=2500, expected_epoch=5,
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / phase2a["data"]["manifest_dir"], tokenizer, phase2a["model"]["vision_tower"],
        split="test", datasets_root=REPO_ROOT / phase2a["data"]["datasets_root"],
        synthscars_root=REPO_ROOT / phase2a["data"]["synthscars_root"], image_size=1024,
    )
    sample = next(dataset[index] for index, row in enumerate(dataset.rows) if row["class_label"] == 1)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    before = snapshot(backend, sample)
    fusion, _ = load_fusion(OUTPUT_ROOT / "B3_npr_srm", device)
    cache = torch.load(OUTPUT_ROOT / "feature_cache/test.pt", map_location="cpu")
    index = cache["sample_ids"].index(sample["sample_id"])
    with torch.no_grad():
        fused = fusion(cache["base_logits"][index:index+1].to(device),
                       cache["h_cls"][index:index+1].to(device),
                       npr=cache["npr"][index:index+1].to(device),
                       srm=cache["srm"][index:index+1].to(device))
    after = snapshot(backend, sample)
    report = {
        "sample_id": sample["sample_id"],
        "classification_logits_allowed_to_change": not torch.equal(fused.logits.cpu(), before["base_logits"]),
        "lm_token_logits": tensor_diff(before["lm_token_logits"], after["lm_token_logits"]),
        "g0_generated_token_ids_equal": before["g0_token_ids"] == after["g0_token_ids"],
        "g0_seg_position_before": before["g0_token_ids"].index(model.seg_token_idx),
        "g0_seg_position_after": after["g0_token_ids"].index(model.seg_token_idx),
        "g0_mask_logits": tensor_diff(before["g0_mask"], after["g0_mask"]),
        "g1_generated_token_ids_equal": before["g1_token_ids"] == after["g1_token_ids"],
        "g1_mask_logits": tensor_diff(before["g1_mask"], after["g1_mask"]),
        "tf_mask_logits": tensor_diff(before["tf_mask"], after["tf_mask"]),
    }
    required = [report["lm_token_logits"]["equal"], report["g0_generated_token_ids_equal"],
                report["g0_mask_logits"]["equal"], report["g1_generated_token_ids_equal"],
                report["g1_mask_logits"]["equal"], report["tf_mask_logits"]["equal"],
                report["g0_seg_position_before"] == report["g0_seg_position_after"]]
    report["passed"] = all(required)
    if not report["passed"]: raise AssertionError(report)
    write_json(OUTPUT_ROOT / "error_analysis/real_model_output_invariance.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
