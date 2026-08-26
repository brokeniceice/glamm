#!/usr/bin/env python3
"""Bounded best-checkpoint inference and Phase 2B diagnostic panels."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase1d_training_policy import configure_args
from scripts.phase2a_final_evaluate import load_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def binary_mask(logits, shape) -> np.ndarray:
    if logits is None:
        return np.zeros(shape, dtype=bool)
    value = torch.as_tensor(logits).detach().float().cpu()
    if value.ndim == 3:
        value = value.amax(dim=0)
    return value.numpy() > 0


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mask.shape != base.shape[:2]:
        raise ValueError(f"mask/image mismatch: {mask.shape}/{base.shape[:2]}")
    result = base.copy()
    result[mask] = result[mask] * 0.45 + np.asarray(color) * 0.55
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=ROOT / "outputs/phase2b_legion_parity/gap_analysis/selected_cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase2b_legion_parity/visualizations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cases", type=int, default=80)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/phase2a_unified_baseline_full.yaml").read_text())
    device = torch.device(args.device); torch.cuda.set_device(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    checkpoint = ROOT / "checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt"
    model, tokenizer, meta = load_model(config, checkpoint, device, expected_step=2500, expected_epoch=5)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / config["data"]["manifest_dir"], tokenizer, config["model"]["vision_tower"], split="test",
        datasets_root=config["data"]["datasets_root"], synthscars_root=config["data"]["synthscars_root"],
        image_size=int(config["model"]["image_size"]),
    )
    index_by_id = {row["sample_id"]: i for i, row in enumerate(dataset.rows)}
    selected = read_jsonl(args.selected)[:args.max_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "bounded_inference.jsonl"
    completed = {r["sample_id"] for r in read_jsonl(records_path)} if records_path.exists() else set()
    font = ImageFont.load_default()
    for ordinal, selection in enumerate(selected, 1):
        sample_id = selection["sample_id"]
        panel_path = args.output / selection["quadrant"] / f"{sample_id.replace(':', '_')}.jpg"
        if sample_id in completed and panel_path.exists():
            continue
        sample = dataset[index_by_id[sample_id]]
        g0 = backend.generate_localization(sample, provide_gt_fake=False, generation_mode="unified_fake_generate")
        g1 = backend.generate_localization(sample, provide_gt_fake=True, generation_mode="unified_prompt_gt_fake_prefix")
        tf = backend.teacher_forced_localization(
            sample, context="full", user_prompt="canonical"
        )
        image = Image.open(sample["image_path"]).convert("RGB")
        height, width = image.height, image.width
        gt_mask = torch.as_tensor(sample["masks"]).bool().any(dim=0).numpy()
        masks = {
            "GT": gt_mask,
            "TF": binary_mask(tf.get("pred_mask"), (height, width)),
            "G0": binary_mask(g0.get("pred_mask"), (height, width)),
            "G1": binary_mask(g1.get("pred_mask"), (height, width)),
        }
        display_width = 320
        display_height = max(1, round(height * display_width / width))
        tiles = [("Original", image)] + [
            (name, overlay(image, masks[name], (255, 40, 40) if name == "GT" else (40, 220, 80)))
            for name in ("GT", "TF", "G0", "G1")
        ]
        text_lines = [
            f"{selection['quadrant']} | {sample_id}",
            f"content={selection['content']} area={selection['gt_area_ratio']:.4f} refs={selection['artifact_count']} components={selection['connected_components']}",
            f"IoU TF={selection['TF_IoU']:.4f} G0={selection['G0_IoU']:.4f} G1={selection['G1_IoU']:.4f}",
            "GT: " + selection["gt_explanation"],
            "G0: " + g0.get("generated_explanation", ""),
        ]
        # PIL's bundled default bitmap font is latin-1 only in the pinned
        # environment. Panels retain full Unicode in JSON; rendering uses a
        # deterministic ASCII replacement solely to avoid a visualization crash.
        rendered_lines = [value.encode("ascii", "replace").decode("ascii") for value in text_lines]
        wrapped = [line for value in rendered_lines for line in textwrap.wrap(value, width=190) or [""]]
        text_height = 18 * len(wrapped) + 16
        canvas = Image.new("RGB", (display_width * 5, display_height + text_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (name, tile) in enumerate(tiles):
            resized = tile.resize((display_width, display_height), Image.Resampling.BILINEAR)
            canvas.paste(resized, (index * display_width, 0))
            draw.rectangle((index * display_width, 0, index * display_width + 70, 18), fill="white")
            draw.text((index * display_width + 3, 2), name, fill="black", font=font)
        y = display_height + 6
        for line in wrapped:
            draw.text((6, y), line, fill="black", font=font); y += 18
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(panel_path, quality=90)
        raw_dir = panel_path.with_suffix("")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in masks.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(raw_dir / f"{name}.png")
        record = {
            "sample_id": sample_id, "quadrant": selection["quadrant"], "panel": str(panel_path),
            "checkpoint": meta, "gt_explanation": selection["gt_explanation"],
            "G0_generated_explanation": g0.get("generated_explanation"),
            "G1_generated_explanation": g1.get("generated_explanation"),
            "G0_seg_triggered": g0.get("seg_triggered"), "G1_seg_triggered": g1.get("seg_triggered"),
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed.add(sample_id)
        print(f"phase2b-visualization {ordinal}/{len(selected)} {sample_id}", flush=True)


if __name__ == "__main__":
    main()
