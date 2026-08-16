#!/usr/bin/env python3
"""Frozen, resumable Phase 3A evaluation with spatial prediction preservation."""

from __future__ import annotations

import argparse
import hashlib
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

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import (
    MASK_LOGIT_THRESHOLD,
    _localization_record,
    evaluate_detection,
    summarize_detection_records,
)
from eval.forensics_eval import GLaMMForensicsBackend
from eval.phase3a_metrics import parse_phrase_aligned_generation, validate_prediction_record
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model


MODES = ("detection", "G0", "tf_full_context", "phrase_only")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3a_p1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--synthscars-root", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-spatial-save", action="store_true",
                        help="Keep exact metrics but omit logits/binary tensors (validation selector only).")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_stem(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _union_logits(pred_mask: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor:
    if pred_mask is None:
        return torch.empty((0, *shape), dtype=torch.float32)
    logits = torch.as_tensor(pred_mask).detach().float().cpu()
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or tuple(logits.shape[-2:]) != shape:
        raise ValueError(f"Unexpected prediction shape {tuple(logits.shape)} for target {shape}")
    return logits


def preserve_spatial_prediction(
    root: Path, mode: str, sample: dict, output: dict, record: dict, *, save_spatial: bool = True,
) -> dict:
    gt = torch.as_tensor(sample["masks"]).bool().any(dim=0).cpu()
    logits = _union_logits(output.get("pred_mask"), tuple(gt.shape))
    union_logits = logits.amax(dim=0) if logits.shape[0] else torch.full(gt.shape, -torch.inf)
    binary = union_logits > MASK_LOGIT_THRESHOLD
    stem = stable_stem(str(sample["sample_id"]))
    logits_path = binary_path = None
    if save_spatial:
        tensor_dir = root / mode / "spatial"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        logits_path = tensor_dir / f"{stem}.logits.pt"
        binary_path = tensor_dir / f"{stem}.binary.pt"
        torch.save(logits.to(torch.bfloat16), logits_path)
        torch.save(binary, binary_path)
    tn = int((~binary & ~gt).sum().item())
    fg_iou = float(record["image_iou"])
    fg_f1 = float(record["image_pixel_f1"])
    bg_den = tn + int(record["fp"]) + int(record["fn"])
    bg_iou = tn / bg_den if bg_den else 1.0
    record.update({
        "decoded_text": output.get("generated_text"),
        "generated_localization_phrase": None,
        "mask_logits_path": None if logits_path is None else str(logits_path.resolve()),
        "binary_mask_path": None if binary_path is None else str(binary_path.resolve()),
        "tn": tn,
        "foreground_iou": fg_iou,
        "foreground_f1": fg_f1,
        "background_iou": bg_iou,
        "fg_bg_miou": (fg_iou + bg_iou) / 2,
        "mask_logit_threshold": MASK_LOGIT_THRESHOLD,
    })
    if mode == "G0":
        parsed = parse_phrase_aligned_generation(output.get("generated_text") or "")
        record["generated_localization_phrase"] = parsed["target_region"]
        record["phrase_parse"] = parsed
    else:
        record["generated_token_ids"] = output.get("generated_token_ids", [])
        record["seg_position"] = output.get("seg_position")
    validate_prediction_record(record)
    return record


def summarize_localization(records: list[dict], *, autoregressive: bool) -> dict:
    count = len(records)
    totals = {key: sum(int(row[key]) for row in records) for key in ("tp", "fp", "fn", "tn")}
    fg_den = totals["tp"] + totals["fp"] + totals["fn"]
    bg_den = totals["tn"] + totals["fp"] + totals["fn"]
    f1_den = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    result = {
        "num_gt_fake": count,
        "mask_logit_threshold": MASK_LOGIT_THRESHOLD,
        "per_image_mean": {
            "foreground_iou": sum(r["foreground_iou"] for r in records) / count if count else None,
            "foreground_f1": sum(r["foreground_f1"] for r in records) / count if count else None,
            "fg_bg_miou": sum(r["fg_bg_miou"] for r in records) / count if count else None,
        },
        "global_pixel": {
            "foreground_iou": totals["tp"] / fg_den if fg_den else (1.0 if count else None),
            "foreground_f1": 2 * totals["tp"] / f1_den if f1_den else (1.0 if count else None),
            "background_iou": totals["tn"] / bg_den if bg_den else (1.0 if count else None),
        },
        "confusion_pixels": totals,
    }
    fg = result["global_pixel"]["foreground_iou"]
    bg = result["global_pixel"]["background_iou"]
    result["global_pixel"]["fg_bg_miou"] = None if fg is None else (fg + bg) / 2
    if autoregressive:
        triggered = sum(bool(r["seg_triggered"] and r["has_pred_mask"]) for r in records)
        result["seg_trigger_rate"] = triggered / count if count else None
        result["phrase_parse_success_rate"] = (
            sum(bool(r.get("phrase_parse", {}).get("parse_success")) for r in records) / count
            if count else None
        )
    return result


def main(argv=None):
    cli = parse_args(argv)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    config_path = (ROOT / cli.config).resolve()
    checkpoint_path = (ROOT / cli.checkpoint).resolve()
    output_root = (ROOT / cli.output_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    model, tokenizer, checkpoint = load_model(
        config, checkpoint_path, device,
        expected_step=cli.expected_step, expected_epoch=cli.expected_epoch,
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    manifest_dir = (
        Path(cli.manifest_dir).resolve()
        if cli.manifest_dir else (ROOT / config["data"]["manifest_dir"]).resolve()
    )
    dataset = UnifiedForensicsDataset(
        manifest_dir, tokenizer, config["model"]["vision_tower"], split="test",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=cli.synthscars_root or config["data"]["synthscars_root"],
        image_size=int(config["model"]["image_size"]),
        target_protocol=config["forensics"].get("target_protocol", "historical"),
    )
    limit = len(dataset) if cli.max_samples is None else min(len(dataset), cli.max_samples)
    requested = set(cli.modes)
    if cli.reset:
        for mode in requested:
            path = output_root / mode / "predictions.jsonl"
            if path.exists():
                path.unlink()
    completed = {
        mode: {row["sample_id"] for row in read_jsonl(output_root / mode / "predictions.jsonl")}
        for mode in requested
    }

    if "G0" in requested:
        fake_indices = [
            i for i in range(limit)
            if int(dataset.rows[i]["class_label"]) == 1 and dataset.rows[i]["sample_id"] not in completed["G0"]
        ]
        batch_size = max(1, cli.generation_batch_size)
        for start in range(0, len(fake_indices), batch_size):
            samples = [dataset[i] for i in fake_indices[start:start + batch_size]]
            outputs = backend.generate_localization_batch(
                samples, provide_gt_fake=False, generation_mode="unified_fake_generate"
            )
            for sample, output in zip(samples, outputs):
                record = _localization_record(
                    sample, output, "unified_fake_generate", uses_gt_authenticity=False,
                    uses_gt_explanation=False, classification_gate=False,
                )
                record = preserve_spatial_prediction(
                    output_root, "G0", sample, output, record,
                    save_spatial=not cli.skip_spatial_save,
                )
                append_jsonl(output_root / "G0" / "predictions.jsonl", record)
            print(f"phase3a-G0 {min(start + batch_size, len(fake_indices))}/{len(fake_indices)}", flush=True)

    for index in range(limit):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        if "detection" in requested and sample_id not in completed["detection"]:
            records, _ = evaluate_detection([sample], backend)
            append_jsonl(output_root / "detection" / "predictions.jsonl", records[0])
        if not (int(sample["cls_label"]) == 1 and bool(sample["seg_valid"])):
            continue
        if "tf_full_context" in requested and sample_id not in completed["tf_full_context"]:
            output = backend.teacher_forced_localization(sample, context="full")
            record = _localization_record(
                sample, output, "tf_full_context", uses_gt_authenticity=True,
                uses_gt_explanation=True, classification_gate=False,
            )
            record = preserve_spatial_prediction(
                output_root, "tf_full_context", sample, output, record,
                save_spatial=not cli.skip_spatial_save,
            )
            append_jsonl(output_root / "tf_full_context" / "predictions.jsonl", record)
        if "phrase_only" in requested and sample_id not in completed["phrase_only"]:
            output = backend.phrase_only_localization(sample)
            record = _localization_record(
                sample, output, "phrase_only", uses_gt_authenticity=True,
                uses_gt_explanation=False, classification_gate=False,
            )
            record["uses_gt_localization_phrase"] = True
            record["oracle_phrase"] = output["oracle_phrase"]
            record["oracle_template"] = output["oracle_template"]
            record = preserve_spatial_prediction(
                output_root, "phrase_only", sample, output, record,
                save_spatial=not cli.skip_spatial_save,
            )
            append_jsonl(output_root / "phrase_only" / "predictions.jsonl", record)
        if (index + 1) % 25 == 0:
            print(f"phase3a-diagnostics {index + 1}/{limit}", flush=True)

    summary = {
        "checkpoint": checkpoint,
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "config": str(config_path),
        "manifest_dir": str(manifest_dir),
        "test_samples": limit,
        "seed": cli.seed,
        "generation": {
            "do_sample": False, "num_beams": 1,
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
            "batch_size": cli.generation_batch_size,
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "modes": {},
    }
    for mode in cli.modes:
        records = read_jsonl(output_root / mode / "predictions.jsonl")
        metrics = (
            summarize_detection_records(records) if mode == "detection"
            else summarize_localization(records, autoregressive=mode == "G0")
        )
        dump_json(output_root / mode / "metrics.json", metrics)
        summary["modes"][mode] = metrics
    dump_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
