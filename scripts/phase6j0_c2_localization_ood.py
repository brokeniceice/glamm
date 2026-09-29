#!/usr/bin/env python3
"""C2 localization on the frozen LOKI, X-AIGD, and PAL4VST final manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/phase6j0_c2/final_evaluation/localization"
CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
CFG = ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml"

from eval.forensics import _localization_record
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts import final_eval_localization_supervisor as frozen
from scripts.phase3a1_loki_g0 import LokiLocalizationDataset
from scripts.phase3a_evaluate import preserve_spatial_prediction
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase3c1_cache import clip_grid
from tools.phase3c1 import geometry_for
from tools.phase4f import load_evidence_source

EXPECTED_C2_SHA = "4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce"


def generated_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def save_replay_cache(path, *, dataset, ordinal, sample_id, checkpoint_sha256,
                      adapter_sha256, target_hw, output, model, source, device):
    """Save frozen C2 trajectory and spatial inputs for a possible later R1 replay."""
    q_values = output["projected_seg_embeddings"]
    q = (torch.empty((0, 256), dtype=torch.bfloat16) if q_values is None
         else q_values.detach().cpu().to(torch.bfloat16))
    if q.ndim != 2 or q.shape[1] != 256:
        raise RuntimeError(f"invalid projected [SEG] states: {tuple(q.shape)}")
    if int(q.shape[0]) != output["generated_token_ids"].count(model.seg_token_idx):
        raise RuntimeError("projected [SEG] count differs from generated trajectory")
    raw_s64 = f24 = z_f24 = None
    if len(q):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw_s64 = model.get_grounding_encoder_embs(
                output["_grounding_image"][None].to(device, dtype=torch.bfloat16))
            tokens, _ = model.get_model().get_vision_tower()(
                output["_global_image"][None].to(device, dtype=torch.bfloat16))
            evidence = source(clip_grid(tokens), return_features=True)
        raw_s64 = raw_s64[0].detach().cpu().to(torch.bfloat16)
        f24 = evidence["F_forensic"][0].detach().cpu().to(torch.bfloat16)
        z_f24 = evidence["logits"][0].detach().cpu().to(torch.bfloat16)
        if raw_s64.shape != (256, 64, 64) or f24.shape != (256, 24, 24) or z_f24.shape != (1, 24, 24):
            raise RuntimeError(f"replay spatial shape drift: {raw_s64.shape}, {f24.shape}, {z_f24.shape}")
        if not all(torch.isfinite(t.float()).all().item() for t in (q, raw_s64, f24, z_f24)):
            raise RuntimeError("nonfinite replay feature")
    payload = {
        "schema": "c2_raw_g1_r1_replay_cache_v1", "dataset": dataset,
        "ordinal": ordinal, "sample_id": sample_id,
        "c2_checkpoint_sha256": checkpoint_sha256,
        "forensic_adapter_checkpoint_sha256": adapter_sha256,
        "generation_mode": output["generation_mode"],
        "prompt_sha256": output["prompt_sha256"],
        "generated_token_ids": output["generated_token_ids"],
        "generated_token_ids_sha256": generated_sha(output["generated_token_ids"]),
        "seg_count": int(q.shape[0]), "q_seg": q,
        "S64_raw": raw_s64, "F24": f24, "z_F24": z_f24,
        "sam_geometry": geometry_for("sam", target_hw),
        "clip_geometry": geometry_for("clip", target_hw),
        "spatial_present_iff_seg": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return frozen.sha256_file(path), int(q.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("loki", "xaigd", "pal4vst"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    frozen_rows, manifest = frozen.manifest_info(args.dataset)
    config = yaml.safe_load(CFG.read_text())
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    step, epoch = int(state["optimizer_step"]), int(state["epoch"])
    del state
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_c2_model(config, CKPT, device, expected_step=step, expected_epoch=epoch)
    if meta["checkpoint_sha256"] != EXPECTED_C2_SHA or (step, epoch) != (3500, 7):
        raise RuntimeError("selected C2 checkpoint provenance drift")
    model.eval().requires_grad_(False)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    evidence_cfg = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
    source = load_evidence_source(evidence_cfg, "forensic_rect", device)
    adapter_sha = evidence_cfg["evidence"]["forensic_checkpoint_sha256"]
    if args.dataset == "loki":
        dataset = LokiLocalizationDataset(ROOT / "datasets/LOKI/legion_localization/manifest.jsonl",
                                          config["model"]["vision_tower"], int(config["model"]["image_size"]))
    else:
        dataset = frozen.ExternalArtifactDataset(args.dataset, tokenizer,
                                                  config["model"]["vision_tower"],
                                                  int(config["model"]["image_size"]))
    expected_ids = [str(row["sample_id"]) for row in frozen_rows]
    if [str(row["sample_id"]) for row in dataset.rows] != expected_ids:
        raise RuntimeError("C2 localization dataset identity/order differs from frozen final manifest")
    if args.dataset == "loki" and any(
        row["image_path"] != official["image_path"] or
        row["mask_path"] != official["derived_union_mask_path"]
        for row, official in zip(dataset.rows, frozen_rows)
    ):
        raise RuntimeError("LOKI image or GT mask path differs from frozen final manifest")
    dest = OUT / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    predictions = dest / "predictions.jsonl"
    previous = frozen.rows(predictions)
    if [str(row["sample_id"]) for row in previous] != expected_ids[:len(previous)]:
        raise RuntimeError("C2 localization resume prefix drift")
    cache_dir = dest / "r1_replay_cache"
    for ordinal, row in enumerate(previous):
        cache_path = cache_dir / f"{ordinal:06d}.pt"
        if (row.get("replay_cache_sha256") is None or not cache_path.is_file() or
                frozen.sha256_file(cache_path) != row["replay_cache_sha256"]):
            raise RuntimeError(f"C2 R1 replay cache resume drift: {cache_path}")
    status = dest / "worker_status.json"
    frozen.atomic_json(status, {"status": "RUNNING", "dataset": args.dataset,
                                "checkpoint_sha256": meta["checkpoint_sha256"],
                                "manifest": manifest, "device": str(device),
                                "completed_prefix": len(previous)})
    try:
        for ordinal in range(len(previous), len(dataset)):
            sample = dataset[ordinal]
            output = backend.generate_localization(
                sample, provide_gt_fake=True, generation_mode="unified_prompt_gt_fake_prefix"
            )
            # The frozen image encoders are replayed once here; later R1 OOD
            # evaluation can consume these exact inputs without C2 generation.
            cache_output = {**output, "_grounding_image": sample["grounding_enc_image"],
                            "_global_image": sample["global_enc_image"]}
            cache_sha, seg_count = save_replay_cache(
                cache_dir / f"{ordinal:06d}.pt", dataset=args.dataset,
                ordinal=ordinal, sample_id=expected_ids[ordinal],
                checkpoint_sha256=meta["checkpoint_sha256"],
                adapter_sha256=adapter_sha,
                target_hw=tuple(sample["masks"].shape[-2:]),
                output=cache_output, model=model, source=source, device=device)
            record = _localization_record(
                sample, output, "G1", uses_gt_authenticity=True,
                uses_gt_explanation=False, classification_gate=False,
            )
            record = preserve_spatial_prediction(dest, "G1", sample, output, record,
                                                 save_spatial=False)
            target = sample["masks"].bool().any(0)
            gt_pixels = int(target.sum())
            record.update({"sample_id": expected_ids[ordinal], "ordinal": ordinal,
                           "status": "OK" if record["seg_triggered"] and record["has_pred_mask"] else "EMPTY_OR_NO_SEG",
                           "gt_foreground_pixels": gt_pixels, "gt_is_empty": gt_pixels == 0,
                           "generator/source": frozen_rows[ordinal].get("generator/source"),
                           "artifact_categories": frozen_rows[ordinal].get("artifact_categories"),
                           "seg_count": seg_count, "valid_q_seg": seg_count > 0,
                           "replay_cache_path": str((cache_dir / f"{ordinal:06d}.pt").resolve()),
                           "replay_cache_sha256": cache_sha,
                           "generated_token_ids_sha256": generated_sha(output["generated_token_ids"])})
            compact = frozen.compact_record(record)
            for key in ("seg_count", "replay_cache_path", "replay_cache_sha256",
                        "generated_token_ids_sha256"):
                compact[key] = record[key]
            frozen.append_jsonl(predictions, frozen.enforce_failure_policy(compact))
            if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(dataset):
                print(json.dumps({"dataset": args.dataset, "done": ordinal + 1,
                                  "total": len(dataset)}), flush=True)
        records = frozen.rows(predictions)
        if len(records) != len(expected_ids) or [str(row["sample_id"]) for row in records] != expected_ids:
            raise RuntimeError("C2 localization output incomplete or out of order")
        result = {"schema": "phase6j0_c2_localization_ood_v1", "status": "COMPLETE",
                  "model": "C2-raw", "dataset": args.dataset, "inference_condition": "G1",
                  "checkpoint": meta, "manifest": manifest, "gt_type": frozen.GT_TYPES[args.dataset],
                  "failure_policy": frozen.FAILURE_POLICY_VERSION,
                  "metrics": frozen.summarize(records),
                  "prediction_file": str(predictions.resolve()),
                  "prediction_sha256": frozen.sha256_file(predictions),
                  "r1_replay_cache": {"schema": "c2_raw_g1_r1_replay_cache_v1",
                                      "directory": str(cache_dir.resolve()),
                                      "count": len(records),
                                      "adapter_checkpoint_sha256": adapter_sha,
                                      "valid_q_seg": sum(bool(row["valid_q_seg"]) for row in records)}}
        frozen.atomic_json(dest / "results.json", result)
        frozen.atomic_json(status, {"status": "COMPLETE", "dataset": args.dataset,
                                    "count": len(records), "result": str(dest / "results.json")})
    except BaseException as exc:
        frozen.atomic_json(status, {"status": "FAILED", "dataset": args.dataset,
                                    "error": str(exc), "completed_prefix": len(frozen.rows(predictions))})
        raise


if __name__ == "__main__":
    main()
