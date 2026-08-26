#!/usr/bin/env python3
"""Prove language/classification behavior is exactly invariant on frozen audit samples."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.phase3d0 import parse_structure


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--selected-step", required=True, type=int)
    parser.add_argument("--selected-epoch", required=True, type=int)
    cli = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs/phase3d2_direct_spatial_path.yaml").read_text())
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device("cuda:0"); torch.cuda.set_device(device)

    # Tokenizer/dataset construction is identical for both checkpoints.
    model, tokenizer, _ = load_model(
        model_cfg, Path(cfg["source"]["checkpoint"]), device,
        expected_step=int(cfg["source"]["optimizer_step"]), expected_epoch=int(cfg["source"]["epoch"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"], split="val",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    count = int(cfg["audit"]["deterministic_sample_count"]); per_class = count // 2
    selected = []
    for label in (0, 1):
        values = [i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == label]
        values.sort(key=lambda i: hashlib.sha256(f"3407:phase3d2-audit:{dataset.rows[i]['sample_id']}".encode()).hexdigest())
        selected.extend(values[:per_class])

    def snapshot(active_model, active_tokenizer):
        backend = GLaMMForensicsBackend(active_model, active_tokenizer, device=device, dtype=torch.bfloat16,
                                        use_mm_start_end=True, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]))
        output = []
        for index in selected:
            sample = dataset[index]
            detection_batch = backend._batch(sample, "")
            detection_batch["grounding_enc_images"] = None
            with torch.no_grad():
                detection = active_model.model_forward(**detection_batch)
            generated = backend.generate_localization(
                sample, provide_gt_fake=False, generation_mode="unified_fake_generate"
            )
            tokens = [int(value) for value in generated.get("generated_token_ids") or []]
            parsed = parse_phrase_aligned_generation(generated.get("generated_text") or "")
            structure = parse_structure(parsed, tokens, active_model.seg_token_idx)
            verdict = next((value for value in tokens if value in (active_model.real_token_idx, active_model.fake_token_idx)), None)
            output.append({
                "sample_id": sample["sample_id"], "gt_class": int(sample["cls_label"]),
                "generated_token_ids": tokens, "generated_text": generated.get("generated_text"),
                "verdict_token": verdict, "target_regions_phrase": parsed.get("target_region"),
                "seg_occurrence": tokens.count(active_model.seg_token_idx),
                "seg_position": next((i for i, value in enumerate(tokens) if value == active_model.seg_token_idx), None),
                "classification_logits": detection["cls_logits"][0].detach().float().cpu().tolist(),
                "classification_prediction": int(detection["cls_pred"][0].detach().cpu()),
                "structure_validity": bool(structure["structural_validity"]),
            })
        return output

    p1 = snapshot(model, tokenizer)
    del model; gc.collect(); torch.cuda.empty_cache()
    spatial, tokenizer2, _ = load_model(
        model_cfg, Path(cli.selected_checkpoint), device,
        expected_step=cli.selected_step, expected_epoch=cli.selected_epoch,
    )
    adapted = snapshot(spatial, tokenizer2)
    fields = (
        "generated_token_ids", "generated_text", "verdict_token", "target_regions_phrase",
        "seg_occurrence", "seg_position", "classification_logits",
        "classification_prediction", "structure_validity",
    )
    comparisons = []
    for before, after in zip(p1, adapted):
        exact = {field: before[field] == after[field] for field in fields}
        comparisons.append({"sample_id": before["sample_id"], "exact": exact, "all_required_exact": all(exact.values())})
    passed = all(row["all_required_exact"] for row in comparisons)
    result = {
        "status": "PASS" if passed else "FAIL", "sample_count": len(comparisons),
        "sample_ids": [row["sample_id"] for row in comparisons], "required_fields": list(fields),
        "comparisons": comparisons, "P1_records": p1, "P3D2_records": adapted,
        "mask_outputs_allowed_to_change": True,
    }
    dump(ROOT / cfg["experiment"]["output_root"] / "behavioral_invariance_audit.json", result)
    if not passed:
        raise RuntimeError("behavioral invariance failed")
    print(json.dumps({"status": "PASS", "samples": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
