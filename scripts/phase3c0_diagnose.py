#!/usr/bin/env python3
"""Read-only, resumable four-condition Phase 3C.0 P1 diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import MASK_LOGIT_THRESHOLD, compute_binary_mask_metrics, compute_empty_prediction_metrics
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import trace_full_forward
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from tools.phase3b_replay import phrase_overlap
from tools.phase3c0 import insert_phrase_tokens, repair_phrase_tokens, sha256_ints


EXPECTED_CHECKPOINT_SHA256 = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"
ROUTE_THRESHOLDS = {
    "g0_failure_iou_max": 0.30,
    "language_recoverable_tf_iou_min": 0.70,
    "language_nontrivial_clipped_recovery_min": 0.10,
    "persistent_min_count": 50,
    "persistent_min_fraction_of_g0_failures": 0.10,
    "usable_seg_fraction_min": 0.80,
    "bootstrap_repeats": 10000,
    "lexical_bins_exploratory": {"high_min_f1": 0.75, "medium_min_f1": 0.35},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3a_p1.yaml")
    parser.add_argument("--selector", default="outputs/phase3a_phrase_grounding/selection/p1_selector.json")
    parser.add_argument("--output-dir", default="outputs/phase3c0_residual_diagnosis")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Fake-only cap for preflight; omitted means all frozen val Fake samples")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-model-hash", action="store_true", help="Preflight only")
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_stem(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def model_state_sha256(model) -> str:
    """Hash names, tensor metadata and every model-state byte without modifying state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        flat = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, flat.numel(), 16 * 1024 * 1024):
            digest.update(flat[start:start + 16 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def spatial_metrics(logits: torch.Tensor, gt: torch.Tensor) -> dict:
    gt = gt.bool().any(dim=0).cpu() if gt.ndim == 3 else gt.bool().cpu()
    if logits.numel():
        metrics = compute_binary_mask_metrics(logits.float().cpu(), gt)
        binary = logits.float().cpu().amax(dim=0).gt(MASK_LOGIT_THRESHOLD)
    else:
        metrics = compute_empty_prediction_metrics(gt)
        binary = torch.zeros_like(gt)
    tn = int((~binary & ~gt).sum())
    bg_den = tn + int(metrics["fp"]) + int(metrics["fn"])
    bg_iou = tn / bg_den if bg_den else 1.0
    return {
        "foreground_iou": float(metrics["image_iou"]),
        "foreground_f1": float(metrics["image_pixel_f1"]),
        "background_iou": float(bg_iou),
        "fg_bg_miou": float((metrics["image_iou"] + bg_iou) / 2),
        "tp": int(metrics["tp"]), "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]), "tn": tn,
    }


def save_trace(root: Path, condition: str, stem: str, trace: dict, gt: torch.Tensor) -> dict:
    rep_path = root / "representations" / condition / f"{stem}.pt"
    logits_path = root / condition / "spatial" / f"{stem}.logits.pt"
    binary_path = root / condition / "spatial" / f"{stem}.binary.pt"
    gt_path = root / "g0" / "gt" / f"{stem}.pt"
    for path in (rep_path, logits_path, binary_path, gt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    hidden = trace["llm_predictor_hidden"].detach().cpu().reshape(-1, trace["llm_predictor_hidden"].shape[-1])
    projected = trace["projected_embedding"].detach().cpu().reshape(-1, trace["projected_embedding"].shape[-1])
    logits = trace["postprocessed_mask_logits"].detach().cpu()
    binary = trace["binary_mask"].detach().cpu()
    torch.save({"predictor_hidden_4096d": hidden, "projected_embedding_256d": projected}, rep_path)
    torch.save(logits.to(torch.bfloat16), logits_path)
    torch.save(binary.bool(), binary_path)
    if not gt_path.exists():
        torch.save(gt.detach().cpu().bool(), gt_path)
    return {
        "representation_path": str(rep_path.resolve()),
        "mask_logits_path": str(logits_path.resolve()),
        "binary_mask_path": str(binary_path.resolve()),
        "gt_mask_path": str(gt_path.resolve()),
        "predictor_hidden_shape": list(hidden.shape),
        "projected_embedding_shape": list(projected.shape),
        "seg_position": trace["seg_token_position_raw"],
        "predictor_position": trace["predictor_position_raw"],
        **spatial_metrics(logits, gt),
    }


def representation_pair(left: torch.Tensor, right: torch.Tensor) -> dict:
    if not left.numel() or not right.numel():
        return {"cosine": None, "relative_l2": None}
    a, b = left.float().reshape(-1), right.float().reshape(-1)
    return {
        "cosine": float(F.cosine_similarity(a, b, dim=0)),
        "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)),
    }


def trace_fixed(backend, sample: dict, assistant_content: str) -> tuple[dict, torch.Tensor]:
    batch = backend._batch(sample, assistant_content, question=UNIFIED_FORENSICS_QUESTION)
    input_ids = batch["input_ids"]
    trace = trace_full_forward(
        backend.model, batch, input_ids, original_size=tuple(batch["label_list"][0].shape)
    )
    return trace, input_ids


def main(argv=None):
    cli = parse_args(argv)
    random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cli.seed)
    output = (ROOT / cli.output_dir).resolve()
    prediction_path = output / "paired/paired_conditions.jsonl"
    if cli.reset and output.exists():
        if output == ROOT or ROOT not in output.parents:
            raise RuntimeError(f"Refusing unsafe reset target: {output}")
        import shutil
        shutil.rmtree(output)

    config_path = (ROOT / cli.config).resolve()
    selector_path = (ROOT / cli.selector).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(selector["selected_checkpoint"]).resolve()
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256 or checkpoint_sha != selector["checkpoint_sha256"]:
        raise RuntimeError(f"Frozen P1 checkpoint SHA mismatch: {checkpoint_sha}")
    if int(selector["optimizer_step"]) != 3500 or int(selector["epoch"]) != 7:
        raise RuntimeError("P1 selector is not frozen step 3500 / epoch 7")

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    model, tokenizer, checkpoint = load_model(
        config, checkpoint_path, device, expected_step=3500, expected_epoch=7
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("No-training contract violated after model load")
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    manifest_dir = (ROOT / config["data"]["manifest_dir"]).resolve()
    dataset = UnifiedForensicsDataset(
        manifest_dir, tokenizer, config["model"]["vision_tower"], split="val",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"],
        image_size=int(config["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    fake_indices = [index for index, row in enumerate(dataset.rows) if int(row["class_label"]) == 1]
    all_fake_ids = [str(dataset.rows[index]["sample_id"]) for index in fake_indices]
    if len(dataset) != 2212 or len(fake_indices) != 1106:
        raise RuntimeError(f"Frozen val population mismatch: total={len(dataset)}, Fake={len(fake_indices)}")
    if cli.max_samples is not None:
        fake_indices = fake_indices[:cli.max_samples]

    model_hash_before = None if cli.skip_model_hash else model_state_sha256(model)
    provenance = {
        "phase": "Phase 3C.0 — Residual Localization Bottleneck Diagnosis",
        "status": "RUNNING", "read_only_no_training": True,
        "checkpoint": checkpoint, "selector_path": str(selector_path),
        "checkpoint_file_sha256": checkpoint_sha,
        "config_path": str(config_path), "manifest_dir": str(manifest_dir),
        "split": "val", "population": "Fake only", "val_total": len(dataset),
        "val_fake_total": len(all_fake_ids), "val_fake_ordered_ids_sha256": ordered_ids_sha256(all_fake_ids),
        "requested_samples": len(fake_indices), "seed": cli.seed,
        "device": str(device), "dtype": "torch.bfloat16", "model_eval": not model.training,
        "all_requires_grad_false": not any(p.requires_grad for p in model.parameters()),
        "max_new_tokens": backend.max_new_tokens, "generation": {"do_sample": False, "num_beams": 1},
        "mask_threshold": MASK_LOGIT_THRESHOLD,
        "route_selection_inputs": ["internal_validation_Fake"],
        "prohibited_route_inputs": ["internal_test", "official1000", "RAISE", "LOKI"],
        "route_thresholds_preregistered_before_results": ROUTE_THRESHOLDS,
        "historical_p1_val_g0_artifact": "HISTORICAL_P1_VAL_G0_ARTIFACT_UNAVAILABLE",
        "model_state_sha256_before": model_hash_before,
    }
    dump(output / "provenance.json", provenance)
    completed = {row["sample_id"] for row in rows(prediction_path)}
    seg_id = int(model.seg_token_idx)

    for ordinal, index in enumerate(fake_indices, start=1):
        sample = dataset[index]
        sid = str(sample["sample_id"])
        if sid in completed:
            continue
        stem = stable_stem(sid)
        gt = torch.as_tensor(sample["masks"]).bool()
        phrase = backend.authoritative_phrase(sample)

        # A: one canonical free-generation trajectory, followed by a fixed no-cache trace.
        generated = backend.generate_localization(
            sample, provide_gt_fake=False, generation_mode="unified_fake_generate"
        )
        prompt_batch = backend._batch(sample, "", question=UNIFIED_FORENSICS_QUESTION)
        prompt_ids = prompt_batch["input_ids"]
        if generated["prompt_token_ids"] != [int(x) for x in prompt_ids[0].detach().cpu().tolist()]:
            raise RuntimeError(f"Canonical prompt identity mismatch for {sid}")
        generated_ids = [int(x) for x in generated["generated_token_ids"]]
        full_a_ids = torch.cat((prompt_ids, torch.tensor([generated_ids], device=device)), dim=1)
        trace_a = trace_full_forward(
            model, prompt_batch, full_a_ids, original_size=tuple(prompt_batch["label_list"][0].shape)
        )
        a = save_trace(output, "g0", stem, trace_a, gt)
        parsed = parse_phrase_aligned_generation(generated.get("generated_text") or "")
        seg_positions = [i for i, token in enumerate(generated_ids) if token == seg_id]
        canonical_logits = generated.get("pred_mask")
        trace_logits = trace_a["postprocessed_mask_logits"].detach().float().cpu()
        if canonical_logits is None:
            canonical_binary_equal = not trace_logits.numel()
            canonical_max_abs = None
        else:
            canonical = torch.as_tensor(canonical_logits).detach().float().cpu()
            canonical_binary_equal = bool(torch.equal(canonical.gt(0), trace_logits.gt(0)))
            canonical_max_abs = float((canonical - trace_logits).abs().max())
        a.update({
            "generated_token_ids": generated_ids,
            "generated_token_ids_sha256": sha256_ints(generated_ids),
            "full_input_token_ids_sha256": sha256_ints(full_a_ids[0].detach().cpu().tolist()),
            "generated_text": generated.get("generated_text"),
            "verdict": parsed["verdict"], "target_field_present": parsed["target_field_present"],
            "generated_phrase_raw": parsed["target_region"],
            "generated_phrase_normalized": " ".join(str(parsed["target_region"] or "").split()),
            "generated_explanation": parsed["explanation"], "phrase_parse": parsed,
            "seg_count": len(seg_positions), "seg_positions_generated": seg_positions,
            "generation_length": len(generated_ids),
            "trace_vs_canonical_binary_exact": canonical_binary_equal,
            "trace_vs_canonical_mask_max_abs": canonical_max_abs,
        })

        # B: exact token-span replacement, or separately flagged exploratory insertion.
        b = None; insertion = None; repair_audit = None
        usable_seg = len(seg_positions) == 1 and seg_positions[0] > 0
        reliable_target_field = (
            parsed["target_field_present"] and bool(parsed["target_region"])
            and "REPEATED_TARGET_FIELD" not in parsed["parse_status"]
        )
        if usable_seg and reliable_target_field:
            repair_audit = repair_phrase_tokens(generated_ids, tokenizer, seg_id, phrase)
            repaired_ids = repair_audit.pop("generated_token_ids")
            full_b_ids = torch.cat((prompt_ids, torch.tensor([repaired_ids], device=device)), dim=1)
            trace_b = trace_full_forward(
                model, prompt_batch, full_b_ids, original_size=tuple(prompt_batch["label_list"][0].shape)
            )
            b = save_trace(output, "phrase_repair", stem, trace_b, gt)
            b.update({"assistant_token_ids": repaired_ids,
                      "assistant_token_ids_sha256": sha256_ints(repaired_ids),
                      "diagnostic_label": "CONTROLLED_MULTI_FACTOR_DIAGNOSTIC", **repair_audit})
        elif usable_seg and not parsed["target_field_present"]:
            insertion_audit = insert_phrase_tokens(generated_ids, tokenizer, seg_id, phrase)
            insertion_ids = insertion_audit.pop("generated_token_ids")
            full_i_ids = torch.cat((prompt_ids, torch.tensor([insertion_ids], device=device)), dim=1)
            trace_i = trace_full_forward(
                model, prompt_batch, full_i_ids, original_size=tuple(prompt_batch["label_list"][0].shape)
            )
            insertion = save_trace(output, "phrase_insertion", stem, trace_i, gt)
            insertion.update({"assistant_token_ids": insertion_ids, "exploratory_only": True, **insertion_audit})

        # C/D use the exact shared builders consumed by the existing Phase 3A backend methods.
        c_content = backend.phrase_only_content(sample)
        trace_c, ids_c = trace_fixed(backend, sample, c_content)
        c = save_trace(output, "phrase_only", stem, trace_c, gt)
        c.update({"assistant_content": c_content, "full_input_token_ids_sha256": sha256_ints(ids_c[0].cpu().tolist()),
                  "oracle_diagnostic": True})
        d_content = backend.tf_phrase_content(sample)
        trace_d, ids_d = trace_fixed(backend, sample, d_content)
        d = save_trace(output, "tf_phrase", stem, trace_d, gt)
        d.update({"assistant_content": d_content, "full_input_token_ids_sha256": sha256_ints(ids_d[0].cpu().tolist()),
                  "oracle_diagnostic": True})

        overlap = phrase_overlap(phrase, parsed["target_region"])
        overlap.update({
            "phrase_presence": bool(parsed["target_region"]),
            "generated_phrase_length": len(str(parsed["target_region"] or "").split()),
            "authoritative_phrase_length": len(phrase.split()),
        })
        rep = {
            "hidden_A_D": representation_pair(trace_a["llm_predictor_hidden"], trace_d["llm_predictor_hidden"]),
            "projected_A_D": representation_pair(trace_a["projected_embedding"], trace_d["projected_embedding"]),
            "hidden_C_D": representation_pair(trace_c["llm_predictor_hidden"], trace_d["llm_predictor_hidden"]),
            "projected_C_D": representation_pair(trace_c["projected_embedding"], trace_d["projected_embedding"]),
        }
        if b is not None:
            rep.update({
                "hidden_B_D": representation_pair(trace_b["llm_predictor_hidden"], trace_d["llm_predictor_hidden"]),
                "projected_B_D": representation_pair(trace_b["projected_embedding"], trace_d["projected_embedding"]),
            })
        record = {
            "sample_id": sid, "image_path": sample["image_path"], "source": sample["source"],
            "content_category": sample.get("content_category"), "authoritative_phrase": phrase,
            "authoritative_explanation": " ".join(str(sample["manifest_row"].get("explanation") or "").split()),
            "g0_seg_status": "USABLE" if usable_seg else "G0_SEG_UNAVAILABLE",
            "reliable_single_target_field": reliable_target_field,
            "primary_phrase_repair_eligible": b is not None,
            "phrase_insertion_exploratory": insertion is not None,
            "A": a, "B": b, "C": c, "D": d, "phrase_insertion": insertion,
            "phrase_quality": overlap, "representation_distances": rep,
        }
        append(prediction_path, record)
        completed.add(sid)
        print(f"phase3c0 {ordinal}/{len(fake_indices)} sid={sid} repair={b is not None}", flush=True)

    final_rows = rows(prediction_path)
    requested_ids = {str(dataset.rows[index]["sample_id"]) for index in fake_indices}
    present_ids = {row["sample_id"] for row in final_rows}
    if not requested_ids.issubset(present_ids):
        raise RuntimeError(f"Incomplete Phase3C0 output: missing {len(requested_ids - present_ids)}")
    model_hash_after = None if cli.skip_model_hash else model_state_sha256(model)
    if model_hash_before != model_hash_after:
        raise RuntimeError("Model state changed during read-only Phase 3C.0")
    frozen_hash = {
        "checkpoint_file_sha256_before": checkpoint_sha,
        "checkpoint_file_sha256_after": file_sha256(checkpoint_path),
        "model_state_sha256_before": model_hash_before, "model_state_sha256_after": model_hash_after,
        "exact_checkpoint_identity": checkpoint_sha == file_sha256(checkpoint_path),
        "exact_model_state_identity": model_hash_before == model_hash_after,
        "model_hash_skipped_preflight_only": cli.skip_model_hash,
    }
    dump(output / "frozen_model_hash.json", frozen_hash)
    provenance.update({"status": "EVALUATION_COMPLETE", "completed_records": len(requested_ids),
                       "model_state_sha256_after": model_hash_after})
    dump(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
