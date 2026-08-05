#!/usr/bin/env python3
"""Frozen-CLIP content labeling for SynthScars and real candidate images.

The script is shard-based and resumable. It never updates model parameters.
Two deterministic image views are averaged for the final embedding, while
view-level predictions are retained for uncertainty auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_SNAPSHOT = Path(
    "/data/yz/myLISA_storage/checkpoints/.hf_cache/hub/"
    "models--openai--clip-vit-large-patch14-336/"
    "snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
)
DEFAULT_TOKENIZER_SNAPSHOT = Path(
    "/home/yz/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/"
    "snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/data/content_labels_clip_v1.json")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_SNAPSHOT)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/real_images_unified_v1/candidate_train_manifest.jsonl",
    )
    parser.add_argument(
        "--synth-manifest",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/synthscars_image_grouped_v1/train_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--overwrite-shards", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: Mapping[str, Any], model_path: Path, tokenizer_path: Path) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload)
    digest.update(str(model_path.resolve()).encode("utf-8"))
    digest.update(str(tokenizer_path.resolve()).encode("utf-8"))
    return digest.hexdigest()


def build_items(
    real_manifest: Path,
    synth_manifest: Path,
    datasets_root: Path,
    limit_per_source: int,
) -> List[Dict[str, Any]]:
    datasets_root = datasets_root.expanduser().resolve()
    real_rows = load_jsonl(real_manifest)
    synth_rows = load_jsonl(synth_manifest)
    items = []
    for row in real_rows:
        items.append({
            "sample_id": row["sample_id"],
            "domain": "real_candidate",
            "source": row["source"],
            "image_path": str(datasets_root / row["image_relpath"]),
            "source_content_category": row.get("content_category", "unknown"),
            "content_sha256": row.get("content_sha256"),
        })
    for row in synth_rows:
        items.append({
            "sample_id": row["sample_id"],
            "domain": "synthscars_fake",
            "source": "SynthScars",
            "image_path": str(datasets_root / "SynthScars" / row["image_relpath"]),
            "source_content_category": "unknown",
            "content_sha256": None,
        })
    if limit_per_source > 0:
        limited, counts = [], Counter()
        for item in items:
            if counts[item["source"]] >= limit_per_source:
                continue
            limited.append(item)
            counts[item["source"]] += 1
        items = limited
    for index, item in enumerate(items):
        item["embedding_index"] = index
    return items


def full_fit_view(image: Image.Image, fill: Sequence[int]) -> Image.Image:
    side = max(image.size)
    return ImageOps.pad(image, (side, side), method=Image.Resampling.BICUBIC, color=tuple(fill), centering=(0.5, 0.5))


def load_two_views(path: str, fill: Sequence[int]) -> tuple[Image.Image, Image.Image]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return image, full_fit_view(image, fill)


@torch.inference_mode()
def build_class_prototypes(
    model: CLIPModel,
    processor: CLIPProcessor,
    categories: Sequence[str],
    prompts: Mapping[str, Sequence[str]],
    device: torch.device,
) -> torch.Tensor:
    prototypes = []
    for category in categories:
        inputs = processor(text=list(prompts[category]), return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        prototype = torch.nn.functional.normalize(features.mean(dim=0), dim=-1)
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)


def probabilities_from_scores(scores: np.ndarray, temperature: float = 0.02) -> np.ndarray:
    logits = scores / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


@torch.inference_mode()
def classify_batch(
    paths: Sequence[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    prototypes: torch.Tensor,
    device: torch.device,
    fill: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_views, fit_views = [], []
    for path in paths:
        center, fit = load_two_views(path, fill)
        center_views.append(center)
        fit_views.append(fit)
    all_views = center_views + fit_views
    pixel_values = processor(images=all_views, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=model.dtype)
    features = model.get_image_features(pixel_values=pixel_values)
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    center_features, fit_features = features.chunk(2, dim=0)
    mean_features = torch.nn.functional.normalize(center_features + fit_features, dim=-1)
    center_scores = center_features @ prototypes.T
    fit_scores = fit_features @ prototypes.T
    mean_scores = (center_scores + fit_scores) / 2
    return (
        mean_features.cpu().numpy().astype(np.float16),
        mean_scores.cpu().numpy().astype(np.float32),
        torch.stack((center_scores, fit_scores), dim=1).cpu().numpy().astype(np.float32),
    )


def shard_paths(output_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"part-{shard_index:05d}"
    return output_dir / "embedding_shards" / f"{stem}.npy", output_dir / "prediction_shards" / f"{stem}.jsonl"


def valid_existing_shard(embedding_path: Path, prediction_path: Path, expected_count: int) -> bool:
    if not embedding_path.is_file() or not prediction_path.is_file():
        return False
    try:
        embeddings = np.load(embedding_path, mmap_mode="r")
        with prediction_path.open("r", encoding="utf-8") as handle:
            prediction_count = sum(1 for line in handle if line.strip())
        return embeddings.shape == (expected_count, 768) and prediction_count == expected_count
    except Exception:
        return False


def predictions_for_batch(
    items: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    view_scores: np.ndarray,
    categories: Sequence[str],
) -> List[Dict[str, Any]]:
    probabilities = probabilities_from_scores(scores)
    records = []
    for item, sample_scores, sample_probabilities, sample_view_scores in zip(items, scores, probabilities, view_scores):
        order = np.argsort(sample_scores)[::-1]
        predicted_index, second_index = int(order[0]), int(order[1])
        view_predictions = np.argmax(sample_view_scores, axis=1)
        records.append({
            **item,
            "predicted_category": categories[predicted_index],
            "scores": {category: float(value) for category, value in zip(categories, sample_scores)},
            "probabilities": {category: float(value) for category, value in zip(categories, sample_probabilities)},
            "top1_top2_margin": float(sample_scores[predicted_index] - sample_scores[second_index]),
            "view_predictions": [categories[int(value)] for value in view_predictions],
            "view_agreement": bool(view_predictions[0] == view_predictions[1]),
        })
    return records


def consolidate(output_dir: Path, shard_count: int, item_count: int) -> None:
    all_predictions = []
    all_embeddings = np.empty((item_count, 768), dtype=np.float16)
    cursor = 0
    for shard_index in range(shard_count):
        embedding_path, prediction_path = shard_paths(output_dir, shard_index)
        embeddings = np.load(embedding_path)
        predictions = load_jsonl(prediction_path)
        if len(embeddings) != len(predictions):
            raise RuntimeError(f"Shard length mismatch: {shard_index}")
        all_embeddings[cursor : cursor + len(embeddings)] = embeddings
        all_predictions.extend(predictions)
        cursor += len(embeddings)
    if cursor != item_count:
        raise RuntimeError(f"Expected {item_count} consolidated items, got {cursor}")
    np.save(output_dir / "embeddings.float16.npy", all_embeddings)
    atomic_write_jsonl(output_dir / "predictions.jsonl", all_predictions)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    model_path = args.model_path.expanduser().resolve()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local CLIP snapshot not found: {model_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Local CLIP tokenizer snapshot not found: {tokenizer_path}")
    items = build_items(
        args.real_manifest,
        args.synth_manifest,
        args.datasets_root,
        args.limit_per_source,
    )
    if not items:
        raise RuntimeError("No content-labeling items were loaded")
    missing = [item["image_path"] for item in items if not Path(item["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input images are missing; first: {missing[0]}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config, model_path, tokenizer_path)
    run_config = {
        **config,
        "config_sha256": sha256_file(args.config),
        "run_fingerprint": fingerprint,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "model_weight_sha256": sha256_file(model_path / "pytorch_model.bin"),
        "device": args.device,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "item_count": len(items),
        "source_counts": dict(Counter(item["source"] for item in items)),
        "real_manifest_sha256": sha256_file(args.real_manifest),
        "synth_manifest_sha256": sha256_file(args.synth_manifest),
    }
    existing_config_path = output_dir / "run_config.json"
    if existing_config_path.is_file():
        existing = load_json(existing_config_path)
        if existing.get("run_fingerprint") != fingerprint or existing.get("item_count") != len(items):
            raise RuntimeError("Output directory contains a different run configuration")
    atomic_write_json(existing_config_path, run_config)
    atomic_write_jsonl(output_dir / "input_index.jsonl", items)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    image_processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    model.requires_grad_(False).eval().to(device)
    if device.type == "cuda":
        model.half()
    categories = list(config["categories"])
    prototypes = build_class_prototypes(model, processor, categories, config["prompts"], device)
    image_mean = processor.image_processor.image_mean
    fill = [int(round(channel * 255)) for channel in image_mean]

    shard_count = math.ceil(len(items) / args.shard_size)
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(len(items), start + args.shard_size)
        shard_items = items[start:end]
        embedding_path, prediction_path = shard_paths(output_dir, shard_index)
        if not args.overwrite_shards and valid_existing_shard(embedding_path, prediction_path, len(shard_items)):
            print(f"shard {shard_index + 1}/{shard_count}: existing", flush=True)
            continue
        shard_embeddings, shard_predictions = [], []
        for batch_start in range(0, len(shard_items), args.batch_size):
            batch_items = shard_items[batch_start : batch_start + args.batch_size]
            embeddings, scores, view_scores = classify_batch(
                [item["image_path"] for item in batch_items],
                model,
                processor,
                prototypes,
                device,
                fill,
            )
            shard_embeddings.append(embeddings)
            shard_predictions.extend(predictions_for_batch(batch_items, scores, view_scores, categories))
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_embedding = embedding_path.with_suffix(".npy.part")
        with temporary_embedding.open("wb") as handle:
            np.save(handle, np.concatenate(shard_embeddings, axis=0))
        os.replace(temporary_embedding, embedding_path)
        atomic_write_jsonl(prediction_path, shard_predictions)
        print(f"shard {shard_index + 1}/{shard_count}: wrote {len(shard_items)}", flush=True)

    consolidate(output_dir, shard_count, len(items))
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
