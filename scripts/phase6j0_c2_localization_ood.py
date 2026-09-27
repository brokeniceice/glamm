#!/usr/bin/env python3
"""C2 localization on the frozen LOKI, X-AIGD, and PAL4VST final manifests."""

from __future__ import annotations

import argparse
import json
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
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
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
                           "artifact_categories": frozen_rows[ordinal].get("artifact_categories")})
            frozen.append_jsonl(predictions, frozen.enforce_failure_policy(frozen.compact_record(record)))
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
                  "prediction_sha256": frozen.sha256_file(predictions)}
        frozen.atomic_json(dest / "results.json", result)
        frozen.atomic_json(status, {"status": "COMPLETE", "dataset": args.dataset,
                                    "count": len(records), "result": str(dest / "results.json")})
    except BaseException as exc:
        frozen.atomic_json(status, {"status": "FAILED", "dataset": args.dataset,
                                    "error": str(exc), "completed_prefix": len(frozen.rows(predictions))})
        raise


if __name__ == "__main__":
    main()
