#!/usr/bin/env python3
"""Online-vs-cache parity check on 32 deterministic random Phase 2C samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import load_model
from scripts.phase2c_forensic_fusion import (
    OUTPUT_ROOT, load_config, load_expert, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def comparison(cached, online):
    cached = cached.float()
    online = online.float()
    cosine = torch.nn.functional.cosine_similarity(cached.flatten(1), online.flatten(1), dim=1)
    return {
        "max_abs": float((cached - online).abs().max()),
        "mean_abs": float((cached - online).abs().mean()),
        "min_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
    }


def main(argv=None):
    cli = parse_args(argv)
    config = load_config("configs/phase2c_npr_srm.yaml")
    phase2a_config = load_config("configs/phase2a_unified_baseline_full.yaml")
    cache = torch.load(OUTPUT_ROOT / "feature_cache/train.pt", map_location="cpu")
    rng = random.Random(3407)
    indices = sorted(rng.sample(range(len(cache["labels"])), cli.samples))
    device = torch.device(cli.device)
    model, tokenizer, _ = load_model(
        phase2a_config, REPO_ROOT / config["base_checkpoint"]["path"], device,
        expected_step=2500, expected_epoch=5,
    )
    expert, _ = load_expert(config, device)
    transform = build_npr_transform(training=False)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=1)
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer,
        "openai/clip-vit-large-patch14-336", split="train",
        datasets_root=REPO_ROOT / config["data"]["datasets_root"],
        synthscars_root=REPO_ROOT / config["data"]["synthscars_root"], image_size=1024,
    )
    online = {key: [] for key in ("base_logits", "lm_logits", "h_cls", "npr", "srm")}
    captured = []
    hook = model.classification_head.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    try:
        for start in range(0, len(indices), cli.batch_size):
            selected = indices[start:start + cli.batch_size]
            samples = [dataset[index] for index in selected]
            batch = backend._batch_many(samples, "")
            batch["grounding_enc_images"] = None
            captured.clear()
            with torch.no_grad(): output = model.model_forward(**batch)
            tensors = []
            for sample in samples:
                with Image.open(sample["image_path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            with torch.no_grad(): branches = expert.extract_branch_features(torch.stack(tensors).to(device))
            online["base_logits"].append(output["cls_logits"].float().cpu())
            online["lm_logits"].append(output["lm_verdict_logits"].float().cpu())
            online["h_cls"].append(captured[0].float().cpu())
            online["npr"].append(branches["npr"].float().cpu())
            online["srm"].append(branches["srm_gated"].float().cpu())
    finally:
        hook.remove()
    selected_cache = torch.tensor(indices)
    report = {
        "seed": 3407, "samples": len(indices), "indices": indices,
        "sample_ids": [cache["sample_ids"][index] for index in indices],
        "comparisons": {
            key: comparison(cache[key][selected_cache], torch.cat(online[key])) for key in online
        },
        "tolerance": {"float32_max_abs": 0.0, "float16_feature_max_abs": 0.01,
                      "min_cosine_similarity": 0.9999},
    }
    for key in ("base_logits", "lm_logits"):
        if report["comparisons"][key]["max_abs"] != 0.0:
            raise AssertionError(f"{key} online parity failed: {report['comparisons'][key]}")
    for key in ("h_cls", "npr", "srm"):
        values = report["comparisons"][key]
        if values["max_abs"] > 0.01 or values["min_cosine_similarity"] < 0.9999:
            raise AssertionError(f"{key} online parity failed: {values}")
    report["passed"] = True
    write_json(OUTPUT_ROOT / "feature_cache/online_parity_32.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
