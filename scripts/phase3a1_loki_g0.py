#!/usr/bin/env python3
"""Frozen new-C0/P1 G0/G1 evaluation on LEGION's 229-image LOKI scope."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    CANONICAL_UNIFIED_QUESTION,
    UnifiedForensicsDataset,
)
from eval.forensics import _localization_record
from eval.forensics_eval import GLaMMForensicsBackend
from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase3a_evaluate import (
    append_jsonl,
    dump_json,
    preserve_spatial_prediction,
    read_jsonl,
    summarize_localization,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("G0", "G1"), default="G0")
    parser.add_argument("--manifest", default="datasets/LOKI/legion_localization/manifest.jsonl")
    parser.add_argument(
        "--output-root",
        default="outputs/phase3a1_paired_control/evaluation/external_g0/loki",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args(argv)


class LokiLocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: Path, global_image_encoder: str, image_size: int):
        self.manifest_path = manifest_path.resolve()
        self.rows = [
            json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.rows) != 229:
            raise ValueError(f"Expected 229 LOKI localization rows, got {len(self.rows)}")
        self.global_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.transform = ResizeLongestSide(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"OpenCV failed to decode {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise OSError(f"Invalid mask {mask_path}: {None if mask is None else mask.shape}")
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        resized = self.transform.apply_image(image)
        descriptions = [region["description"] for region in row["regions"] if region["description"]]
        explanation = " ".join(descriptions)
        manifest_row = {
            **row,
            "forensics_domain": "fake",
            "class_label": 1,
            "explanation": explanation,
        }
        return {
            "image_path": str(image_path),
            "global_enc_image": self.global_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0],
            "grounding_enc_image": UnifiedForensicsDataset.grounding_enc_processor(
                torch.from_numpy(resized).permute(2, 0, 1).contiguous()
            ),
            "bboxes": None,
            "conversations": [],
            "masks": mask,
            "label": torch.full(
                (height, width), UnifiedForensicsDataset.IGNORE_LABEL, dtype=torch.long
            ),
            "resize": resized.shape[:2],
            "questions": [CANONICAL_UNIFIED_QUESTION],
            "sampled_classes": ["synthetic artifact"],
            "cls_label": 1,
            "seg_valid": True,
            "sample_id": row["sample_id"],
            "source": "LOKI",
            "content_category": None,
            "manifest_row": manifest_row,
            "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256,
            "target_protocol": "external_loki_g0_g1",
        }


def main(argv=None):
    cli = parse_args(argv)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    config_path = (ROOT / cli.config).resolve()
    checkpoint_path = (ROOT / cli.checkpoint).resolve()
    manifest_path = (ROOT / cli.manifest).resolve()
    output_dir = (ROOT / cli.output_root / cli.model_name).resolve()
    prediction_path = output_dir / cli.mode / "predictions.jsonl"
    if cli.reset and prediction_path.exists():
        prediction_path.unlink()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, checkpoint = load_model(
        config, checkpoint_path, device,
        expected_step=cli.expected_step, expected_epoch=cli.expected_epoch,
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    dataset = LokiLocalizationDataset(
        manifest_path, config["model"]["vision_tower"], int(config["model"]["image_size"])
    )
    existing = read_jsonl(prediction_path)
    completed = {row["sample_id"] for row in existing}
    pending = [index for index, row in enumerate(dataset.rows) if row["sample_id"] not in completed]
    batch_size = max(1, cli.generation_batch_size)
    is_g1 = cli.mode == "G1"
    generation_mode = "unified_prompt_gt_fake_prefix" if is_g1 else "unified_fake_generate"
    for start in range(0, len(pending), batch_size):
        samples = [dataset[index] for index in pending[start:start + batch_size]]
        outputs = backend.generate_localization_batch(
            samples, provide_gt_fake=is_g1, generation_mode=generation_mode
        )
        for sample, output in zip(samples, outputs):
            record = _localization_record(
                sample, output, generation_mode, uses_gt_authenticity=is_g1,
                uses_gt_explanation=False, classification_gate=False,
            )
            record["gt_mask_semantics"] = "LOKI regional bbox union, matching LEGION Table 2"
            record = preserve_spatial_prediction(output_dir, cli.mode, sample, output, record)
            append_jsonl(prediction_path, record)
        print(
            f"loki-{cli.mode}-{cli.model_name} "
            f"{min(start + batch_size, len(pending))}/{len(pending)}",
            flush=True,
        )

    records = read_jsonl(prediction_path)
    expected_ids = [row["sample_id"] for row in dataset.rows]
    indexed = {row["sample_id"]: row for row in records}
    if len(indexed) != 229 or set(indexed) != set(expected_ids):
        raise RuntimeError(f"LOKI scope mismatch: got {len(indexed)}, expected 229")
    ordered = [indexed[sample_id] for sample_id in expected_ids]
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    metrics = summarize_localization(ordered, autoregressive=True)
    metrics["legion_table2_metric_aligned"] = {
        "mIoU_percent": 100.0 * metrics["global_pixel"]["fg_bg_miou"],
        "foreground_F1_percent": 100.0 * metrics["global_pixel"]["foreground_f1"],
        "aggregation": "dataset-global pixel confusion, then foreground/background IoU mean",
        "gt_semantics": "union of LOKI regional bounding boxes rasterized as filled rectangles",
        "scope": "229 fully synthetic LOKI images with valid regional annotations",
        "protocol_boundary": (
            "G1 supplies a structural GT [FAKE] continuation prefix; LEGION uses a direct "
            "artifact-localization instruction, so inference prompts are not identical."
            if is_g1 else
            "G0 freely generates the authenticity verdict; LEGION uses a direct "
            "artifact-localization instruction, so inference prompts are not aligned."
        ),
    }
    dump_json(output_dir / cli.mode / "metrics.json", metrics)
    summary = {
        "model_name": cli.model_name,
        "checkpoint": checkpoint,
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "config": str(config_path),
        "manifest": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "num_images": len(dataset),
        "seed": cli.seed,
        "selection_used_loki": False,
        "generation": {
            "protocol": (
                "G1 canonical unified prompt with structural GT [FAKE] continuation prefix"
                if is_g1 else "G0 canonical unified free generation"
            ),
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
            "batch_size": batch_size,
        },
        "metrics": metrics,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    summary_path = output_dir / "summary.json" if cli.mode == "G0" else output_dir / cli.mode / "summary.json"
    dump_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
