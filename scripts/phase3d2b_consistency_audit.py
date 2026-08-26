#!/usr/bin/env python3
"""Verify exact Phase 3D.2 training/matched-evaluation equivalence on 8 samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import unwrap_glamm
from model.GLaMM import extract_seg_predictor_hidden
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import make_collate, move_batch
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3d2a_consistency_audit import tensor_diff

OUT = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_masks(core, projected, image_embeddings, resize_list, label_list):
    native, postprocessed = [], []
    for index, pred_embedding in enumerate(projected):
        sparse, dense = core.model.grounding_encoder.prompt_encoder(
            points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
        )
        sparse = sparse.to(pred_embedding.dtype)
        low_res, _ = core.model.grounding_encoder.mask_decoder(
            image_embeddings=image_embeddings[index].unsqueeze(0),
            image_pe=core.model.grounding_encoder.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False,
        )
        native.append(low_res[:, 0])
        postprocessed.append(core.model.grounding_encoder.postprocess_masks(
            low_res, input_size=resize_list[index], original_size=label_list[index].shape
        )[:, 0])
    return native, postprocessed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(args.device); torch.cuda.set_device(device)
    model, tokenizer, _ = load_model(
        model_cfg, Path(cfg["source"]["checkpoint"]), device,
        expected_step=int(cfg["source"]["optimizer_step"]), expected_epoch=int(cfg["source"]["epoch"]),
    )
    model.eval()
    if model.training or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only boundary failed before matched consistency audit")
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(model_cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"], split="train",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    fixed_ids = json.loads((P3D2 / "training/schedule.json").read_text(encoding="utf-8"))["sample_ids"][:8]
    row_index = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    core, collate, records = unwrap_glamm(model), make_collate(tokenizer), []
    with torch.no_grad():
        for sample_id in fixed_ids:
            sample = dataset[row_index[sample_id]]
            training = move_batch(collate([sample]), device, torch.bfloat16)
            matched = backend._batch(
                sample, backend.tf_phrase_content(sample), question=UNIFIED_FORENSICS_QUESTION
            )
            global_images = core._prepare_global_enc_image(training["global_enc_images"], training["offset"])
            prepared_ids, prepared_attention, past, embeds, _ = core.prepare_inputs_labels_for_multimodal(
                training["input_ids"], training["attention_masks"], None, None,
                global_images, training["bboxes"],
            )
            train_hidden = core.model(
                input_ids=prepared_ids, attention_mask=prepared_attention, past_key_values=past,
                inputs_embeds=embeds, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True,
            )[0]
            _, matched_hidden_all = core._inference_path(
                matched["input_ids"], matched["global_enc_images"], matched["attention_masks"],
                matched["offset"], matched["bboxes"],
            )
            matched_hidden = core._get_last_hidden_state(matched_hidden_all)
            raw_train, pos_train = extract_seg_predictor_hidden(train_hidden, training["input_ids"], core.seg_token_idx)
            raw_matched, pos_matched = extract_seg_predictor_hidden(matched_hidden, matched["input_ids"], core.seg_token_idx)
            proj_train, _ = core._extract_projected_seg_predictor_hidden(train_hidden, training["input_ids"], training["offset"])
            proj_matched, _ = core._extract_projected_seg_predictor_hidden(matched_hidden_all, matched["input_ids"], matched["offset"])
            image_train = core.get_grounding_encoder_embs(training["grounding_enc_images"])
            image_matched = core.get_grounding_encoder_embs(matched["grounding_enc_images"])
            native_train, post_train = decode_masks(core, proj_train, image_train, training["resize_list"], training["label_list"])
            native_matched, post_matched = decode_masks(core, proj_matched, image_matched, matched["resize_list"], matched["label_list"])
            gt_train = training["masks_list"][0].bool().cpu()
            gt_matched = matched["masks_list"][0].bool().cpu()
            records.append({
                "sample_id": sample_id,
                "prompt_text_exact_equal": training["conversation_list"] == matched["conversation_list"],
                "raw_input_tokens_exact_equal": bool(torch.equal(training["input_ids"], matched["input_ids"])),
                "valid_token_sequence_exact_equal": bool(torch.equal(
                    training["input_ids"][training["attention_masks"]], matched["input_ids"][matched["attention_masks"]]
                )),
                "attention_mask_exact_equal": bool(torch.equal(training["attention_masks"], matched["attention_masks"])),
                "seg_count_training": int(training["input_ids"].eq(core.seg_token_idx).sum()),
                "seg_count_matched": int(matched["input_ids"].eq(core.seg_token_idx).sum()),
                "seg_positions_training": [int(x) for x in pos_train[0].tolist()],
                "seg_positions_matched": [int(x) for x in pos_matched[0].tolist()],
                "gt_mask_pixel_exact_equal": bool(torch.equal(gt_train, gt_matched)),
                "global_image_exact_equal": bool(torch.equal(training["global_enc_images"], matched["global_enc_images"])),
                "grounding_image_exact_equal": bool(torch.equal(training["grounding_enc_images"], matched["grounding_enc_images"])),
                "seg_hidden": tensor_diff(raw_train[0], raw_matched[0], exact_expected=True),
                "text_hidden_fcs_output": tensor_diff(proj_train[0], proj_matched[0], exact_expected=True),
                "native_mask_logits": tensor_diff(native_train[0], native_matched[0], exact_expected=True),
                "postprocessed_mask_logits": tensor_diff(post_train[0], post_matched[0], exact_expected=True),
            })
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient materialized during matched consistency audit")
    mismatch = any(
        not row["valid_token_sequence_exact_equal"] or row["seg_count_training"] != row["seg_count_matched"]
        or row["seg_positions_training"] != row["seg_positions_matched"]
        or not row["gt_mask_pixel_exact_equal"]
        or row["seg_hidden"].get("max_abs_diff", float("inf")) > 1e-5
        or row["text_hidden_fcs_output"].get("max_abs_diff", float("inf")) > 1e-5
        or row["native_mask_logits"].get("max_abs_diff", float("inf")) > 1e-4
        or row["postprocessed_mask_logits"].get("max_abs_diff", float("inf")) > 1e-4
        for row in records
    )
    result = {
        "status": "MISMATCH_FOUND" if mismatch else "PASS",
        "gate": "GATE_MATCHED_EVALUATOR_STILL_INCONSISTENT" if mismatch else None,
        "phase": "Phase 3D.2-B", "sample_count": len(records), "sample_ids": fixed_ids,
        "runtime": {"model_eval": not model.training, "torch_no_grad": True, "gradients_absent_after": True},
        "training_user_prompt": UNIFIED_FORENSICS_QUESTION,
        "matched_evaluation_user_prompt": UNIFIED_FORENSICS_QUESTION,
        "records": records,
        "materiality_thresholds": {"hidden_max_abs": 1e-5, "mask_logit_max_abs": 1e-4},
    }
    dump(OUT / "consistency/matched_protocol_consistency_audit.json", result)
    dump(OUT / "matched_protocol_consistency_audit.json", result)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
