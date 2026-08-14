#!/usr/bin/env python3
"""Frozen selected-subset trace, replay, and intervention runner for Phase 2D.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION, UnifiedForensicsDataset
from eval.controlled_intervention import (
    decode_predictor_hidden, decode_projected_embedding, decode_sam_prompt, validate_intervention,
)
from eval.exact_replay import replay_stepwise_predictor_hidden
from eval.forensics import compute_binary_mask_metrics
from eval.forensics_eval import GLaMMForensicsBackend
from eval.inference_trace import (
    tensor_descriptor, tensor_sha256, trace_full_forward, trace_full_forward_batch, unwrap_glamm,
)
from eval.representation_metrics import binary_mask_iou, recovery, tensor_distance
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model

CHECKPOINT_SHA = "07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072"
HISTORICAL_IOU_TOLERANCE = 5e-4


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return tensor_descriptor(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def save_trace(path: Path, trace: dict, *, sample_id: str, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = {key: value.detach().cpu() if torch.is_tensor(value) else value
                 for key, value in trace.items() if key != "image_embedding"}
    persisted["sample_id"] = sample_id
    persisted["mode"] = mode
    persisted["image_embedding_descriptor"] = tensor_descriptor(trace["image_embedding"])
    torch.save(persisted, path)
    return {"sample_id": sample_id, "mode": mode, "file": str(path), "file_sha256": file_sha256(path),
            "tensors": {key: tensor_descriptor(value) for key, value in trace.items() if torch.is_tensor(value)}}


def mask_metrics(mask_logits, sample):
    return compute_binary_mask_metrics(mask_logits, sample["masks"])


def module_hash(module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        tensor = value.detach().contiguous().cpu().view(torch.uint8)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_trace(a, b):
    keys = ("llm_predictor_hidden", "projected_embedding", "sparse_prompt_embedding",
            "dense_prompt_embedding", "low_res_mask_logits", "postprocessed_mask_logits")
    result = {key: tensor_distance(a[key], b[key]) for key in keys}
    result["binary_mask_exact"] = bool(torch.equal(a["binary_mask"], b["binary_mask"]))
    result["binary_mask_iou"] = binary_mask_iou(a["binary_mask"], b["binary_mask"])
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a_unified_baseline_full.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt")
    parser.add_argument("--selection", default="outputs/phase2d1_trace_replay/selection/selected_samples.jsonl")
    parser.add_argument("--output-dir", default="outputs/phase2d1_trace_replay")
    parser.add_argument("--manifest-dir", default="outputs/phase2b_legion_parity/split_audit/official1000_manifest")
    parser.add_argument("--synthscars-root", default="/data/yz/myLISA_storage/AIGC/SynthScars")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stepwise-limit", type=int, default=17)
    parser.add_argument("--invariance-limit", type=int, default=3)
    return parser.parse_args()


def main():
    cli = parse_args()
    output_root = (ROOT / cli.output_dir).resolve()
    selected = read_jsonl((ROOT / cli.selection).resolve())
    if cli.max_samples is not None:
        selected = selected[:cli.max_samples]
    completed_path = output_root / "controlled_modes/per_sample_results.jsonl"
    completed = {row["sample_id"] for row in read_jsonl(completed_path)} if completed_path.exists() else set()
    config = yaml.safe_load((ROOT / cli.config).read_text(encoding="utf-8"))
    checkpoint = (ROOT / cli.checkpoint).resolve()
    if file_sha256(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Frozen checkpoint SHA256 mismatch")
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    model, tokenizer, checkpoint_meta = load_model(config, checkpoint, device, expected_step=2500, expected_epoch=5)
    core = unwrap_glamm(model)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(config["evaluation"]["max_new_tokens"]))
    dataset = UnifiedForensicsDataset(
        (ROOT / cli.manifest_dir).resolve(), tokenizer, config["model"]["vision_tower"], split="test",
        datasets_root=config["data"]["datasets_root"], synthscars_root=cli.synthscars_root,
        image_size=int(config["model"]["image_size"]),
    )
    indices = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    historical_g0 = {row["sample_id"]: row for row in read_jsonl(
        ROOT / "outputs/phase2b_legion_parity/official1000_raw/G0/predictions.jsonl")}
    historical_tf = {row["sample_id"]: row for row in read_jsonl(
        ROOT / "outputs/phase2b_legion_parity/official1000_raw/tf_full_context/predictions.jsonl")}
    trace_manifest_path = output_root / "traces/trace_manifest.jsonl"
    trace_manifest_done = {(row["sample_id"], row["mode"]) for row in read_jsonl(trace_manifest_path)} if trace_manifest_path.exists() else set()
    checkpoint_before = file_sha256(checkpoint)
    module_hashes_before = {"text_projection": module_hash(core.model.text_hidden_fcs),
                            "sam_prompt_encoder": module_hash(core.model.grounding_encoder.prompt_encoder),
                            "sam_mask_decoder": module_hash(core.model.grounding_encoder.mask_decoder)}
    replay_rows, invariance_rows = [], []
    processed_count = 0
    for selection_index, selected_row in enumerate(selected):
        sample_id = selected_row["sample_id"]
        if sample_id in completed:
            continue
        dataset_index = indices[sample_id]
        sample = dataset[dataset_index]
        original_size = tuple(sample["masks"].shape[-2:])
        g0_batch = backend._batch(sample, "", question=CANONICAL_UNIFIED_QUESTION, continue_assistant=False)
        stored = historical_g0[sample_id]
        prompt_ids = torch.tensor(stored["prompt_token_ids"], device=device, dtype=torch.long).unsqueeze(0)
        if not torch.equal(prompt_ids, g0_batch["input_ids"]):
            raise RuntimeError(f"Canonical prompt token mismatch for {sample_id}")
        full_g0_ids = torch.tensor(stored["prompt_token_ids"] + stored["generated_token_ids"],
                                   device=device, dtype=torch.long).unsqueeze(0)
        group_start = (dataset_index // 8) * 8
        group_indices = list(range(group_start, min(group_start + 8, len(dataset))))
        group_samples = [dataset[index] for index in group_indices]
        group_batch = backend._batch_many(
            group_samples, "", question=CANONICAL_UNIFIED_QUESTION, continue_assistant=False
        )
        group_sequences = []
        for group_sample in group_samples:
            history = historical_g0[group_sample["sample_id"]]
            group_sequences.append(history["prompt_token_ids"] + history["generated_token_ids"])
        max_length = max(map(len, group_sequences))
        pad_id = int(core.config.pad_token_id)
        group_ids = torch.tensor(
            [ids + [pad_id] * (max_length - len(ids)) for ids in group_sequences],
            device=device, dtype=torch.long,
        )
        group_sizes = [tuple(group_sample["masks"].shape[-2:]) for group_sample in group_samples]
        group_traces = trace_full_forward_batch(
            model, group_batch, group_ids, original_sizes=group_sizes
        )
        local_index = dataset_index - group_start
        g0_trace = group_traces[local_index]
        g0_metrics = mask_metrics(g0_trace["postprocessed_mask_logits"], sample)
        explanation = " ".join(str(sample["manifest_row"]["explanation"]).split())
        tf_batch = backend._batch(sample, f"[FAKE] {explanation} [SEG]")
        tf_trace = trace_full_forward(model, tf_batch, tf_batch["input_ids"], original_size=original_size)
        tf_metrics = mask_metrics(tf_trace["postprocessed_mask_logits"], sample)
        historical_errors = {"G0_iou_abs_error": abs(g0_metrics["image_iou"] - float(stored["image_iou"])),
            "TF_iou_abs_error": abs(tf_metrics["image_iou"] - float(historical_tf[sample_id]["image_iou"]))}
        reproducible = max(historical_errors.values()) <= HISTORICAL_IOU_TOLERANCE
        row = {"sample_id": sample_id, "subsets": selected_row["subsets"], "taxonomy": selected_row["taxonomy"],
            "trace_forward_reproducible": reproducible, "historical_errors": historical_errors,
            "historical_iou_tolerance": HISTORICAL_IOU_TOLERANCE,
            "historical_G0_batch_size": 8,
            "G0": {"iou": g0_metrics["image_iou"], "mask_metrics": g0_metrics, "token_length": int(full_g0_ids.shape[1]),
                   "seg_position": g0_trace["seg_token_position_raw"], "predictor_position": g0_trace["predictor_position_raw"]},
            "TF": {"iou": tf_metrics["image_iou"], "mask_metrics": tf_metrics, "token_length": int(tf_batch["input_ids"].shape[1]),
                   "seg_position": tf_trace["seg_token_position_raw"], "predictor_position": tf_trace["predictor_position_raw"]},
            "G0_text": stored["generated_text"], "TF_text": explanation}
        for mode, trace in (("g0", g0_trace), ("tf", tf_trace)):
            trace_path = output_root / "traces" / mode / f"{sample_id.replace(':','__')}.pt"
            entry = save_trace(trace_path, trace, sample_id=sample_id, mode=mode)
            if (sample_id, mode) not in trace_manifest_done:
                append_jsonl(trace_manifest_path, entry)
                trace_manifest_done.add((sample_id, mode))
        if not reproducible:
            row["status"] = "TRACE_FORWARD_REPRODUCIBILITY_MISMATCH"
            row["interventions_executed"] = False
            append_jsonl(completed_path, row)
            processed_count += 1
            print(f"phase2d1 progress={processed_count}/{len(selected)} sample={sample_id} status={row['status']}", flush=True)
            torch.cuda.empty_cache()
            continue
        if g0_trace["predictor_position_raw"] is None:
            row["status"] = "G0_SEG_PREDICTOR_UNAVAILABLE"
            row["interventions_executed"] = False
            append_jsonl(completed_path, row)
            processed_count += 1
            print(f"phase2d1 progress={processed_count}/{len(selected)} sample={sample_id} status={row['status']}", flush=True)
            torch.cuda.empty_cache()
            continue

        phrases = "; ".join(str(ref.get("phrase") or "").strip() for ref in sample["manifest_row"].get("refs", []) if ref.get("phrase"))
        h3_batch = backend._batch(sample, f"[FAKE] {phrases} [SEG]")
        h3_trace = trace_full_forward(model, h3_batch, h3_batch["input_ids"], original_size=original_size)
        h3_metrics = mask_metrics(h3_trace["postprocessed_mask_logits"], sample)

        integrities = {mode: validate_intervention(
            g0_trace, tf_trace, base_sample_id=sample_id, donor_sample_id=sample_id,
            intervention=mode, checkpoint_sha256=CHECKPOINT_SHA, donor_checkpoint_sha256=CHECKPOINT_SHA,
        ) for mode in ("H2a_predictor_hidden_swap", "H2b_projection_swap", "H2c_sam_prompt_swap")}
        if any(value["status"] != "VALID" for value in integrities.values()):
            raise RuntimeError(f"Intervention integrity failure for {sample_id}: {integrities}")
        resize = tuple(g0_batch["resize_list"][0])
        h2a = decode_predictor_hidden(model, g0_trace, tf_trace["llm_predictor_hidden"],
                                     resize=resize, original_size=original_size)
        h2b = decode_projected_embedding(model, g0_trace, tf_trace["projected_embedding"],
                                         resize=resize, original_size=original_size)
        h2c = decode_sam_prompt(model, g0_trace, sparse=tf_trace["sparse_prompt_embedding"],
                                dense=tf_trace["dense_prompt_embedding"], resize=resize,
                                original_size=original_size)
        h2_metrics = {name: mask_metrics(trace["postprocessed_mask_logits"], sample)
                      for name, trace in (("H2a", h2a), ("H2b", h2b), ("H2c", h2c))}
        representations = {"G0_vs_TF_hidden": tensor_distance(g0_trace["llm_predictor_hidden"], tf_trace["llm_predictor_hidden"]),
            "G0_vs_TF_projection": tensor_distance(g0_trace["projected_embedding"], tf_trace["projected_embedding"]),
            "G0_vs_TF_sparse_prompt": tensor_distance(g0_trace["sparse_prompt_embedding"], tf_trace["sparse_prompt_embedding"]),
            "G0_vs_TF_dense_prompt": tensor_distance(g0_trace["dense_prompt_embedding"], tf_trace["dense_prompt_embedding"]),
            "G0_vs_TF_mask_logits": tensor_distance(g0_trace["postprocessed_mask_logits"], tf_trace["postprocessed_mask_logits"]),
            "G0_vs_TF_binary_mask_iou": binary_mask_iou(g0_trace["binary_mask"], tf_trace["binary_mask"])}
        row.update({"H1a_oracle_context": {"iou": tf_metrics["image_iou"], "status": "MULTI_FACTOR_ORACLE_CONTEXT_DIAGNOSTIC",
                **recovery(tf_metrics["image_iou"], g0_metrics["image_iou"], tf_metrics["image_iou"])},
            "H2a": {"iou": h2_metrics["H2a"]["image_iou"], **recovery(h2_metrics["H2a"]["image_iou"],g0_metrics["image_iou"],tf_metrics["image_iou"])},
            "H2b": {"iou": h2_metrics["H2b"]["image_iou"], **recovery(h2_metrics["H2b"]["image_iou"],g0_metrics["image_iou"],tf_metrics["image_iou"])},
            "H2c": {"iou": h2_metrics["H2c"]["image_iou"], **recovery(h2_metrics["H2c"]["image_iou"],g0_metrics["image_iou"],tf_metrics["image_iou"])},
            "H3_phrase_only": {"iou": h3_metrics["image_iou"], "status": "MULTI_FACTOR_DIAGNOSTIC_ONLY",
                **recovery(h3_metrics["image_iou"],g0_metrics["image_iou"],tf_metrics["image_iou"])},
            "representations": representations, "intervention_integrity": integrities,
            "phrases": phrases, "interventions_executed": True})
        row["status"] = "COMPLETE"

        trace_path = output_root / "traces/h3" / f"{sample_id.replace(':','__')}.pt"
        entry = save_trace(trace_path, h3_trace, sample_id=sample_id, mode="h3")
        if (sample_id, "h3") not in trace_manifest_done:
            append_jsonl(trace_manifest_path, entry)
            trace_manifest_done.add((sample_id, "h3"))
        intervention_path = output_root / "traces/h2" / f"{sample_id.replace(':','__')}.pt"
        intervention_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_id": sample_id, "H2a": {k:v.detach().cpu() for k,v in h2a.items()},
                    "H2b": {k:v.detach().cpu() for k,v in h2b.items()},
                    "H2c": {k:v.detach().cpu() for k,v in h2c.items()}}, intervention_path)

        if selection_index < cli.stepwise_limit:
            stepwise = replay_stepwise_predictor_hidden(
                model, g0_batch, full_g0_ids, prompt_length=len(stored["prompt_token_ids"])
            )
            stepwise_mask = decode_predictor_hidden(
                model, g0_trace, stepwise["llm_predictor_hidden"], resize=resize, original_size=original_size
            )
            replay_row = {"sample_id": sample_id,
                "full_forward_vs_stepwise_hidden": tensor_distance(g0_trace["llm_predictor_hidden"], stepwise["llm_predictor_hidden"]),
                "full_forward_vs_stepwise_projection": tensor_distance(g0_trace["projected_embedding"], stepwise_mask["projected_embedding"]),
                "full_forward_vs_stepwise_mask_logits": tensor_distance(g0_trace["postprocessed_mask_logits"], stepwise_mask["postprocessed_mask_logits"]),
                "full_forward_vs_stepwise_binary_mask_iou": binary_mask_iou(g0_trace["binary_mask"],stepwise_mask["binary_mask"]),
                "stepwise_mask_metrics": mask_metrics(stepwise_mask["postprocessed_mask_logits"], sample),
                "stepwise": json_safe(stepwise)}
            append_jsonl(output_root / "replay/per_sample_replay.jsonl", replay_row)
            replay_rows.append(replay_row)

        if selection_index < cli.invariance_limit:
            with torch.no_grad():
                sequences, original_masks, _, _ = core.evaluate(
                    group_batch["global_enc_images"], group_batch["grounding_enc_images"], group_batch["input_ids"],
                    group_batch["resize_list"], group_sizes, max_tokens_new=int(config["evaluation"]["max_new_tokens"]),
                    bboxes=group_batch["bboxes"], force_cls_token=False, return_generation_details=True,
                )
            traced_generated_group = trace_full_forward_batch(
                model, group_batch, sequences, original_sizes=group_sizes
            )
            traced_generated = traced_generated_group[local_index]
            with torch.no_grad():
                tf_original = core.model_forward(**tf_batch)
            tf_original_mask = tf_original["pred_masks"][0]
            invariance = {"sample_id": sample_id,
                "current_original_tokens_identical_to_traced_tokens": bool(torch.equal(
                    sequences[local_index:local_index + 1], traced_generated["input_ids"])),
                "current_original_tokens_match_historical": bool(torch.equal(sequences, group_ids)),
                "current_original_seg_position_identical_to_trace": int(
                    sequences[local_index].eq(core.seg_token_idx).nonzero()[0]
                ) == traced_generated["seg_token_position_raw"],
                "mask_logits_max_abs": float((original_masks[local_index]-traced_generated["postprocessed_mask_logits"]).abs().max()),
                "binary_mask_identical": bool(torch.equal(original_masks[local_index].gt(0),traced_generated["binary_mask"])),
                "TF_seg_position_identical": int(tf_batch["input_ids"][0].eq(core.seg_token_idx).nonzero()[0]) == tf_trace["seg_token_position_raw"],
                "TF_mask_logits_max_abs": float((tf_original_mask-tf_trace["postprocessed_mask_logits"]).abs().max()),
                "TF_binary_mask_identical": bool(torch.equal(tf_original_mask.gt(0),tf_trace["binary_mask"]))}
            invariance["status"] = "PASS" if (invariance["current_original_tokens_identical_to_traced_tokens"]
                and invariance["current_original_seg_position_identical_to_trace"]
                and invariance["mask_logits_max_abs"] == 0 and invariance["binary_mask_identical"]
                and invariance["TF_seg_position_identical"] and invariance["TF_mask_logits_max_abs"] == 0
                and invariance["TF_binary_mask_identical"]) else "TRACE_NOT_INVARIANT"
            append_jsonl(output_root / "replay/trace_invariance_samples.jsonl", invariance)
            invariance_rows.append(invariance)
        append_jsonl(completed_path, row)
        processed_count += 1
        print(f"phase2d1 progress={processed_count}/{len(selected)} sample={sample_id} status={row['status']}", flush=True)
        torch.cuda.empty_cache()

    checkpoint_after = file_sha256(checkpoint)
    module_hashes_after = {"text_projection": module_hash(core.model.text_hidden_fcs),
                           "sam_prompt_encoder": module_hash(core.model.grounding_encoder.prompt_encoder),
                           "sam_mask_decoder": module_hash(core.model.grounding_encoder.mask_decoder)}
    discipline = {"checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after,
                  "checkpoint_unchanged": checkpoint_before == checkpoint_after,
                  "runtime_module_hashes_before": module_hashes_before, "runtime_module_hashes_after": module_hashes_after,
                  "runtime_modules_unchanged": module_hashes_before == module_hashes_after,
                  "training_started": False, "model_weights_modified": False}
    path = output_root / "audit/no_weight_mutation.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(discipline,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"processed":processed_count,"checkpoint":checkpoint_meta,"discipline":discipline},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
