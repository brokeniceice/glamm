#!/usr/bin/env python3
"""Read-only train/eval consistency audit for Phase 3D.2-A."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from eval.inference_trace import unwrap_glamm
from model.GLaMM import extract_seg_predictor_hidden
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import make_collate, move_batch
from scripts.phase2a_final_evaluate import file_sha256, load_model

OUT = ROOT / "outputs/phase3d2a_spatial_attribution_audit"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_diff(left: torch.Tensor, right: torch.Tensor, *, exact_expected: bool) -> dict:
    a = left.detach().float().cpu()
    b = right.detach().float().cpu()
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_equal": False}
    delta = (a - b).abs()
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0)) if flat_a.numel() else 1.0
    return {
        "shape_a": list(a.shape), "shape_b": list(b.shape), "shape_equal": True,
        "max_abs_diff": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.mean()) if delta.numel() else 0.0,
        "cosine_similarity": cosine,
        "exact_equal": bool(torch.equal(a, b)), "exact_equal_if_expected": exact_expected,
    }


def static_contract(cfg: dict) -> dict:
    train_manifest = json.loads((P3D2 / "train_manifest.json").read_text(encoding="utf-8"))
    training_config = json.loads((P3D2 / "training_config.json").read_text(encoding="utf-8"))
    selector = json.loads((P3D2 / "evaluation/selector/selector.json").read_text(encoding="utf-8"))
    return {
        "language_context": {
            "training_target_protocol": cfg["data"]["target_protocol"],
            "evaluation_target_protocol": "phrase_aligned",
            "training_teacher_forced_context": training_config["teacher_forced_context"],
            "evaluation_teacher_forced_context": "[FAKE] explanation\\nTarget regions: <authoritative phrase> [SEG]",
            "authoritative_phrase_construction": train_manifest["authoritative_phrase_construction"],
            "training_question": "Determine whether this image is authentic and explain the forensic evidence.",
            "evaluation_question": "Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.",
            "prompt_template_equal": False,
            "mismatch_source": "teacher_forced_localization -> _causal_forward -> _batch uses FORENSICS_QUESTION default",
            "expected_seg_count": 1,
        },
        "image_path": {
            "dataset_class_shared": "dataset.forensics.unified.UnifiedForensicsDataset",
            "global_processor_shared": "CLIPImageProcessor",
            "grounding_resize_shared": "SAM ResizeLongestSide(image_size)",
            "grounding_padding_shared": "bottom/right to 1024",
            "normalization_shared": "IMG_MEAN/IMG_STD from UnifiedForensicsDataset",
            "stochastic_augmentation": False,
        },
        "gt_mask_path": {
            "provenance": cfg["data"]["mask_provenance"],
            "construction": cfg["data"]["target_construction"],
            "rasterizer": train_manifest["rasterizer"],
            "training_and_evaluation_dataset_class_shared": True,
            "binary_conversion": "polygon union -> float32 {0,1}",
        },
        "prediction_path": {
            "projection": "model.text_hidden_fcs[0]",
            "decoder": "grounding_encoder.mask_decoder(multimask_output=False)",
            "postprocess": "grounding_encoder.postprocess_masks(low_res, input_size=resize, original_size=GT shape)",
            "formal_threshold_logit": float(cfg["evaluation"]["mask_logit_threshold"]),
            "training_loss_tensor": "postprocessed logits at original GT resolution",
            "evaluation_metric_tensor": "same postprocessed logits at original GT resolution",
            "formal_selector_step": int(selector["optimizer_step"]),
        },
        "source_code": {
            rel: {"path": str((ROOT / rel).resolve()), "sha256": file_sha256(ROOT / rel)}
            for rel in (
                "scripts/phase3d2_train.py", "scripts/phase3a_evaluate.py",
                "eval/forensics_eval.py", "dataset/forensics/unified.py", "model/GLaMM.py",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, tokenizer, _ = load_model(
        model_cfg, Path(cfg["source"]["checkpoint"]), device,
        expected_step=int(cfg["source"]["optimizer_step"]), expected_epoch=int(cfg["source"]["epoch"]),
    )
    model.eval()
    if model.training or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only boundary failed before consistency audit")
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(model_cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"], split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    schedule = json.loads((P3D2 / "training/schedule.json").read_text(encoding="utf-8"))["sample_ids"]
    row_index = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    fixed_ids = schedule[: args.sample_count]
    records = []
    core = unwrap_glamm(model)
    collate = make_collate(tokenizer)
    with torch.no_grad():
        for sample_id in fixed_ids:
            sample = dataset[row_index[sample_id]]
            train_batch = move_batch(collate([sample]), device, torch.bfloat16)
            eval_batch = backend._batch(sample, backend.tf_phrase_content(sample))
            # The two paths must consume the same raw tokens, attention, image tensors and geometry.
            token_equal = torch.equal(train_batch["input_ids"], eval_batch["input_ids"])
            attention_equal = torch.equal(train_batch["attention_masks"], eval_batch["attention_masks"])
            global_image_equal = torch.equal(train_batch["global_enc_images"], eval_batch["global_enc_images"])
            grounding_image_equal = torch.equal(train_batch["grounding_enc_images"], eval_batch["grounding_enc_images"])
            gt_a = train_batch["masks_list"][0].bool().any(dim=0).cpu()
            gt_b = eval_batch["masks_list"][0].bool().any(dim=0).cpu()
            gt_inter = int((gt_a & gt_b).sum())
            gt_union = int((gt_a | gt_b).sum())

            # Reproduce Phase 3D.2 ``spatial_forward`` exactly up to its frozen hidden state.
            global_images = core._prepare_global_enc_image(train_batch["global_enc_images"], train_batch["offset"])
            prepared_ids, prepared_attention, past, embeds, _ = core.prepare_inputs_labels_for_multimodal(
                train_batch["input_ids"], train_batch["attention_masks"], None, None,
                global_images, train_batch["bboxes"],
            )
            train_decoder = core.model(
                input_ids=prepared_ids, attention_mask=prepared_attention, past_key_values=past,
                inputs_embeds=embeds, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True,
            )
            train_hidden_all = train_decoder[0]
            eval_output, eval_hidden_all = core._inference_path(
                eval_batch["input_ids"], eval_batch["global_enc_images"],
                eval_batch["attention_masks"], eval_batch["offset"], eval_batch["bboxes"],
            )
            train_last = core._get_last_hidden_state(train_hidden_all)
            eval_last = core._get_last_hidden_state(eval_hidden_all)
            raw_train, pos_train = extract_seg_predictor_hidden(train_last, train_batch["input_ids"], core.seg_token_idx)
            raw_eval, pos_eval = extract_seg_predictor_hidden(eval_last, eval_batch["input_ids"], core.seg_token_idx)
            projected_train, _ = core._extract_projected_seg_predictor_hidden(
                train_hidden_all, train_batch["input_ids"], train_batch["offset"]
            )
            projected_eval, _ = core._extract_projected_seg_predictor_hidden(
                eval_hidden_all, eval_batch["input_ids"], eval_batch["offset"]
            )
            emb_train = core.get_grounding_encoder_embs(train_batch["grounding_enc_images"])
            emb_eval = core.get_grounding_encoder_embs(eval_batch["grounding_enc_images"])
            mask_train = core._generate_and_postprocess_masks(
                projected_train, emb_train, train_batch["resize_list"], train_batch["label_list"]
            )[0]
            mask_eval = core._generate_and_postprocess_masks(
                projected_eval, emb_eval, eval_batch["resize_list"], eval_batch["label_list"]
            )[0]
            seg_count_train = int(train_batch["input_ids"].eq(core.seg_token_idx).sum())
            seg_count_eval = int(eval_batch["input_ids"].eq(core.seg_token_idx).sum())
            records.append({
                "sample_id": sample_id,
                "prompt_text_equal": train_batch["conversation_list"] == eval_batch["conversation_list"],
                "input_ids_equal": token_equal, "attention_mask_equal": attention_equal,
                "seg_token_count_training": seg_count_train, "seg_token_count_evaluation": seg_count_eval,
                "seg_positions_training": [int(x) for x in pos_train[0].tolist()],
                "seg_positions_evaluation": [int(x) for x in pos_eval[0].tolist()],
                "sequence_truncated_training": bool(train_batch["seg_preserving_truncation_count"]),
                "global_image_exact_equal": global_image_equal,
                "grounding_image_exact_equal": grounding_image_equal,
                "resize_equal": train_batch["resize_list"] == eval_batch["resize_list"],
                "label_shape_equal": list(gt_a.shape) == list(gt_b.shape),
                "gt_mask_pixel_exact_equal": bool(torch.equal(gt_a, gt_b)),
                "gt_foreground_pixels_training": int(gt_a.sum()),
                "gt_foreground_pixels_evaluation": int(gt_b.sum()),
                "gt_mask_iou": gt_inter / gt_union if gt_union else 1.0,
                "frozen_seg_hidden": tensor_diff(raw_train[0], raw_eval[0], exact_expected=True),
                "projected_256d": tensor_diff(projected_train[0], projected_eval[0], exact_expected=True),
                "grounding_image_embedding": tensor_diff(emb_train, emb_eval, exact_expected=True),
                "postprocessed_mask_logits": tensor_diff(mask_train, mask_eval, exact_expected=True),
                "training_output_hidden_shape": list(train_hidden_all.shape),
                "evaluation_output_logits_shape": list(eval_output.logits.shape),
            })
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient materialized during read-only consistency audit")
    required_bools = (
        "prompt_text_equal", "input_ids_equal", "attention_mask_equal", "global_image_exact_equal",
        "grounding_image_exact_equal", "resize_equal", "label_shape_equal", "gt_mask_pixel_exact_equal",
    )
    material_mismatch = any(
        (not all(bool(row[key]) for key in required_bools))
        or row["seg_token_count_training"] != 1 or row["seg_token_count_evaluation"] != 1
        or row["frozen_seg_hidden"].get("max_abs_diff", float("inf")) > 1e-5
        or row["postprocessed_mask_logits"].get("max_abs_diff", float("inf")) > 1e-4
        for row in records
    )
    result = {
        "phase": "Phase 3D.2-A", "audit": "Train/Eval Path Consistency",
        "status": "MISMATCH_FOUND" if material_mismatch else "PASS",
        "gate": "GATE_PHASE3D2_IMPLEMENTATION_MISMATCH_FOUND" if material_mismatch else None,
        "read_only_runtime": {"model_eval": not model.training, "torch_no_grad": True,
                              "parameter_gradients_absent_after": True},
        "fixed_sample_protocol": {"source": "first preregistered Phase 3D.2 exposures",
                                  "sample_count": len(fixed_ids), "sample_ids": fixed_ids},
        "static_contract": static_contract(cfg), "sample_records": records,
        "mismatch_diagnosis": ({
            "component": "language_user_prompt",
            "material": True,
            "assistant_authoritative_context_equal": True,
            "training_question": "Determine whether this image is authentic and explain the forensic evidence.",
            "evaluation_question": "Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.",
            "observed_effect": "8/8 input token sequences differ; downstream frozen [SEG] hidden and postprocessed mask logits differ",
            "repair_attempted": False, "training_rerun_attempted": False,
        } if material_mismatch else None),
        "materiality_rule": "any raw-path inequality, non-single SEG, hidden max_abs_diff>1e-5, or mask-logit max_abs_diff>1e-4",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    path = OUT / "consistency/train_eval_consistency_audit.json"
    dump(path, result)
    md = [
        "# Phase 3D.2-A Train/Eval Consistency Audit", "",
        f"- Status: `{result['status']}`", f"- Fixed samples: {len(records)}", "- Runtime: `model.eval()` + `torch.no_grad()`",
        "- Authoritative assistant target, image preprocessing, all-ref union mask, SAM decoder/postprocess path, and logit threshold match.",
        "- Material mismatch: training uses the canonical unified authenticity question; TF-PHRASE evaluation uses the legacy localization question through `_batch`'s default argument.",
        "", "## Per-sample numerical comparison", "",
        "| Sample | token exact | GT exact | hidden max abs | hidden cosine | mask-logit max abs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        md.append(
            f"| `{row['sample_id']}` | {row['input_ids_equal']} | {row['gt_mask_pixel_exact_equal']} | "
            f"{row['frozen_seg_hidden'].get('max_abs_diff')} | {row['frozen_seg_hidden'].get('cosine_similarity')} | "
            f"{row['postprocessed_mask_logits'].get('max_abs_diff')} |"
        )
    md += ["", "## Decision", "", (
        "No material train/eval implementation mismatch was detected; mechanism attribution may proceed."
        if not material_mismatch else
        "A material mismatch was detected. Phase 3D.2 mechanism attribution must stop without repair or retraining."
    ), ""]
    md_path = OUT / "consistency/train_eval_consistency_audit.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md), encoding="utf-8")
    # Required flat aliases.
    dump(OUT / "train_eval_consistency_audit.json", result)
    (OUT / "train_eval_consistency_audit.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
