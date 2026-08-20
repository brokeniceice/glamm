#!/usr/bin/env python3
"""Audit and cache frozen spatial sources for Phase 3C.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from model.llava import conversation as conversation_lib
from npr_expert.focal_extractor import FrozenFOCALViT
from npr_expert.official_npr_srm import OfficialNPRSRM
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import file_sha256, load_model
from tools.phase3c1 import (
    PREPROCESS_VERSIONS, SOURCES, geometry_for, ordered_ids_sha256, sha256_text,
    tensor_sha256, transform_mask,
)
from model.SAM.utils.transforms import ResizeLongestSide


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "cache"))
    parser.add_argument("--config", default="configs/phase3c1_spatial_probe.yaml")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "official1000"), default="val")
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def resolved(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def setup_cache_link(config) -> tuple[Path, Path]:
    output = resolved(config["experiment"]["output_root"])
    storage = resolved(config["experiment"]["cache_storage_root"])
    output.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)
    link = output / "cache"
    if link.is_symlink():
        if link.resolve() != storage:
            raise RuntimeError(f"cache symlink points to unexpected target: {link.resolve()}")
    elif link.exists():
        if link.resolve() != storage:
            raise RuntimeError(f"refusing to replace existing cache path: {link}")
    else:
        os.symlink(storage, link, target_is_directory=True)
    return output, storage


def fake_rows(config, split: str) -> list[dict]:
    path = resolved(config["data"]["manifest_dir"]) / (
        "official_synthscars_test.jsonl" if split == "official1000" else f"{split}_combined.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    fake = [row for row in rows if row.get("forensics_domain") == "fake" and int(row["class_label"]) == 1]
    expected = 1000 if split == "official1000" else int(config["data"][f"{split}_fake"])
    if len(fake) != expected or len({str(row["sample_id"]) for row in fake}) != expected:
        raise RuntimeError(f"frozen {split} Fake population mismatch: {len(fake)} != {expected}")
    return fake


def image_path(config, row: dict) -> Path:
    candidate = Path(str(row.get("image_path") or "")).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    path = resolved(config["data"]["synthscars_root"]) / str(row["image_relpath"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def union_mask(row: dict, height: int, width: int) -> torch.Tensor:
    return UnifiedForensicsDataset._fake_union_mask(row, height, width).bool()


def module_hash(module: torch.nn.Module) -> str:
    return tensor_sha256([(name, value) for name, value in module.state_dict().items()])


def load_p1(config, device):
    phase3a = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text(encoding="utf-8"))
    checkpoint = resolved(config["base_checkpoint"]["path"])
    digest = file_sha256(checkpoint)
    if digest != config["base_checkpoint"]["sha256"]:
        raise RuntimeError(f"P1 SHA256 mismatch: {digest}")
    model, tokenizer, meta = load_model(
        phase3a, checkpoint, device,
        expected_step=int(config["base_checkpoint"]["optimizer_step"]),
        expected_epoch=int(config["base_checkpoint"]["epoch"]),
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("P1 frozen/eval contract violated")
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    return phase3a, model, tokenizer, meta


def load_p1_dataset(config, phase3a, tokenizer, split):
    return UnifiedForensicsDataset(
        resolved(config["data"]["manifest_dir"]), tokenizer, phase3a["model"]["vision_tower"],
        split=split, datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=1024,
        target_protocol="phrase_aligned",
    )


def standalone_p1_sample(dataset, row, path):
    """Build only the two frozen vision inputs for non-internal manifests."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV failed to decode {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    global_image = dataset.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    resized = ResizeLongestSide(1024).apply_image(image)
    grounding = UnifiedForensicsDataset.grounding_enc_processor(
        torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    )
    mask = union_mask(row, image.shape[0], image.shape[1])
    return {"sample_id": row["sample_id"], "global_enc_image": global_image,
            "grounding_enc_image": grounding, "masks": mask}


def source_dtype(config, source):
    return getattr(torch, str(config["cache"]["storage_dtype"][source]))


def clip_grid(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise RuntimeError(f"CLIP patch tensor is not [B,N,C]: {tuple(tokens.shape)}")
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"CLIP_SPATIAL_UNAVAILABLE: token count {tokens.shape[1]} is not a square grid")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)


def audit(config, source: str, device: torch.device, split: str):
    output, _ = setup_cache_link(config)
    rows = fake_rows(config, split)
    row = rows[0]
    path = image_path(config, row)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_hw = [rgb.height, rgb.width]
    parameter_before = parameter_after = None
    regression = {}
    if source in {"sam", "clip"}:
        phase3a, model, tokenizer, checkpoint = load_p1(config, device)
        dataset_split = "val" if split == "official1000" else split
        dataset = load_p1_dataset(config, phase3a, tokenizer, dataset_split)
        if split == "official1000":
            sample = standalone_p1_sample(dataset, row, path)
        else:
            index = next(i for i, value in enumerate(dataset.rows) if value["sample_id"] == row["sample_id"])
            sample = dataset[index]
        module = (
            model.model.model.grounding_encoder.image_encoder if source == "sam"
            else model.get_model().get_vision_tower()
        )
        parameter_before = module_hash(module)
        with torch.no_grad():
            if source == "sam":
                feature = model.get_grounding_encoder_embs(
                    sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                )
                source_module = "model.model.grounding_encoder.image_encoder"
            else:
                tokens, _ = module(
                    sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                )
                feature = clip_grid(tokens)
                source_module = "model.get_model().get_vision_tower().vision_tower hidden patch tokens"
        parameter_after = module_hash(module)
        regression = {
            "P1_G0_code_path_modified": False,
            "P1_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "P1_all_requires_grad_false": not any(p.requires_grad for p in model.parameters()),
            "P1_eval": not model.training,
        }
    elif source in {"npr", "srm"}:
        checkpoint_path = resolved(config["expert"]["npr_srm_checkpoint"])
        expert = OfficialNPRSRM.from_checkpoint(checkpoint_path, freeze=True).to(device).eval()
        parameter_before = module_hash(expert)
        transform = build_npr_transform(load_size=256, crop_size=224, training=False)
        with Image.open(path) as image:
            inputs = transform(image.convert("RGB"))[None].to(device)
        with torch.no_grad():
            old_logits = expert(inputs)
            old_npr = expert.extract_npr_features(inputs)
            old_srm = expert.extract_srm_features(inputs)
            npr_spatial = expert.extract_npr_spatial_features(inputs)
            srm_spatial = expert.extract_srm_spatial_features(inputs)
            new_npr = expert.avgpool(npr_spatial).flatten(1).float()
            new_srm = expert.srm_pool(srm_spatial).flatten(1).float()
            rebuilt_logits = expert.fc1(new_npr + expert.srm_gate.float() * new_srm)
            feature = npr_spatial if source == "npr" else srm_spatial
        parameter_after = module_hash(expert)
        regression = {
            "historical_logits_exact": bool(torch.equal(old_logits, rebuilt_logits)),
            "npr_pooled_exact": bool(torch.equal(old_npr, new_npr)),
            "srm_pooled_exact": bool(torch.equal(old_srm, new_srm)),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        }
        source_module = (
            "OfficialNPRSRM.layer2(layer1(maxpool(relu(bn1(conv1(NPR))))))"
            if source == "npr" else "OfficialNPRSRM.srm_stem(srm_residual_features(images))"
        )
    else:
        weights = resolved(config["expert"]["focal_weights"])
        focal = FrozenFOCALViT(weights, micro_batch_size=1).to(device).eval()
        parameter_before = module_hash(focal.image_encoder)
        transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        with Image.open(path) as image:
            inputs = transform(image.convert("RGB"))[None]
        with torch.no_grad():
            old_pooled = focal(inputs)
            feature = focal.forward_spatial(inputs)
            rebuilt = torch.cat([feature.mean((2, 3)), feature.amax((2, 3))], dim=1)
        parameter_after = module_hash(focal.image_encoder)
        regression = {
            "historical_pooled_512d_exact": bool(torch.equal(old_pooled, rebuilt)),
            "weights": str(weights), "weights_sha256": file_sha256(weights),
            "FP32_extraction": feature.dtype == torch.float32,
        }
        source_module = "FrozenFOCALViT.image_encoder"
    geometry = geometry_for(source, original_hw)
    target = transform_mask(union_mask(row, *original_hw), geometry)
    payload = {
        "source": source, "source_module": source_module, "feature_shape": list(feature.shape),
        "feature_dtype": str(feature.dtype), "preprocessing": PREPROCESS_VERSIONS[source],
        "preprocessing_sha256": sha256_text(PREPROCESS_VERSIONS[source]),
        "geometry_example": geometry, "target_shape": list(target.shape),
        "parameter_hash_before": parameter_before, "parameter_hash_after": parameter_after,
        "parameter_hash_exact": parameter_before == parameter_after,
        "model_eval": True, "no_grad": True, "regression": regression,
    }
    failures = [key for key, value in regression.items() if key.endswith("_exact") and not value]
    if parameter_before != parameter_after or failures:
        payload["status"] = "ABORT_EXISTING_FORWARD_CHANGED"
        dump(output / "audit" / f"{source}.json", payload)
        raise RuntimeError(payload["status"] + f": {failures}")
    payload["status"] = "PASS"
    dump(output / "audit" / f"{source}.json", payload)
    return payload


def cache(config, source: str, split: str, device: torch.device, max_samples, force):
    output, storage = setup_cache_link(config)
    audit_path = output / "audit" / f"{source}.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["status"] != "PASS":
        audit(config, source, device, split)
    rows = fake_rows(config, split)
    if max_samples is not None:
        rows = rows[:max_samples]
    destination = storage / source / split
    destination.mkdir(parents=True, exist_ok=True)
    complete_path = destination / "complete.json"
    if complete_path.exists() and not force:
        existing = json.loads(complete_path.read_text())
        if int(existing["samples"]) == len(rows):
            print(f"complete cache exists: {complete_path}", flush=True)
            return
    shard_size = int(config["cache"]["shard_size"])
    storage_dtype = source_dtype(config, source)
    phase3a = model = tokenizer = dataset = expert = focal = None
    if source in {"sam", "clip"}:
        phase3a, model, tokenizer, checkpoint_meta = load_p1(config, device)
        dataset_split = "val" if split == "official1000" else split
        dataset = load_p1_dataset(config, phase3a, tokenizer, dataset_split)
        fake_indices = [] if split == "official1000" else [
            i for i, row in enumerate(dataset.rows) if row.get("forensics_domain") == "fake"
        ][:len(rows)]
        module = model.model.model.grounding_encoder.image_encoder if source == "sam" else model.get_model().get_vision_tower()
        source_checkpoint = checkpoint_meta["checkpoint"]
        source_checkpoint_sha = checkpoint_meta["checkpoint_sha256"]
        source_parameter_hash = module_hash(module)
    elif source in {"npr", "srm"}:
        checkpoint_path = resolved(config["expert"]["npr_srm_checkpoint"])
        expert = OfficialNPRSRM.from_checkpoint(checkpoint_path, freeze=True).to(device).eval()
        transform = build_npr_transform(load_size=256, crop_size=224, training=False)
        source_checkpoint, source_checkpoint_sha = str(checkpoint_path), file_sha256(checkpoint_path)
        source_parameter_hash = module_hash(expert)
    else:
        weights = resolved(config["expert"]["focal_weights"])
        focal = FrozenFOCALViT(weights, micro_batch_size=1).to(device).eval()
        transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        source_checkpoint, source_checkpoint_sha = str(weights), file_sha256(weights)
        source_parameter_hash = module_hash(focal.image_encoder)
    started = time.time()
    feature_shapes, cache_error = set(), {"max_abs_error": 0.0, "sum_abs": 0.0, "count": 0, "cosine": []}
    for start in range(0, len(rows), shard_size):
        end = min(start + shard_size, len(rows))
        shard_path = destination / f"shard_{start:06d}_{end:06d}.pt"
        if shard_path.exists() and not force:
            print(f"skip {source} {split} [{start},{end})", flush=True)
            continue
        features, targets, originals, records = [], [], [], []
        for ordinal in range(start, end):
            row = rows[ordinal]
            path = image_path(config, row)
            if source in {"sam", "clip"}:
                sample = (
                    standalone_p1_sample(dataset, row, path)
                    if split == "official1000" else dataset[fake_indices[ordinal]]
                )
                if sample["sample_id"] != row["sample_id"]:
                    raise RuntimeError("manifest/dataset Fake order mismatch")
                original = torch.as_tensor(sample["masks"]).bool()
                original_hw = list(original.shape[-2:])
                with torch.no_grad():
                    if source == "sam":
                        live = model.get_grounding_encoder_embs(
                            sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                        )
                    else:
                        tokens, _ = model.get_model().get_vision_tower()(
                            sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                        )
                        live = clip_grid(tokens)
            else:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    original_hw = [rgb.height, rgb.width]
                    original = union_mask(row, *original_hw)
                    inputs = transform(rgb)[None]
                with torch.no_grad():
                    if source == "npr":
                        live = expert.extract_npr_spatial_features(inputs.to(device))
                    elif source == "srm":
                        live = expert.extract_srm_spatial_features(inputs.to(device))
                    else:
                        live = focal.forward_spatial(inputs)
            if live.shape[0] != 1 or live.ndim != 4:
                raise RuntimeError(f"invalid {source} feature shape: {tuple(live.shape)}")
            geometry = geometry_for(source, original_hw)
            target = transform_mask(original, geometry)
            cached = live.detach().to(dtype=storage_dtype).cpu()[0]
            feature_shapes.add(tuple(cached.shape))
            if len(cache_error["cosine"]) < 16:
                reference, restored = live.detach().float().cpu().flatten(), cached.float().flatten()
                error = (reference - restored).abs()
                cache_error["max_abs_error"] = max(cache_error["max_abs_error"], float(error.max()))
                cache_error["sum_abs"] += float(error.sum())
                cache_error["count"] += error.numel()
                cache_error["cosine"].append(float(F.cosine_similarity(reference, restored, dim=0)))
            features.append(cached)
            targets.append(target.cpu())
            if split != "train":
                originals.append(original.any(dim=0).cpu())
            records.append({
                "sample_id": str(row["sample_id"]), "image_path": str(path),
                "source": source, "feature_shape": list(cached.shape),
                "feature_dtype": str(cached.dtype), "preprocessing_version": PREPROCESS_VERSIONS[source],
                "preprocessing_sha256": sha256_text(PREPROCESS_VERSIONS[source]),
                "source_checkpoint": source_checkpoint, "source_checkpoint_sha256": source_checkpoint_sha,
                "source_parameter_hash": source_parameter_hash, "original_image_size": original_hw,
                "geometry": geometry, "manifest_ordinal": ordinal,
            })
        payload = {
            "schema": "phase3c1_spatial_cache_v1", "source": source, "split": split,
            "start": start, "end": end, "features": torch.stack(features),
            "targets": torch.stack(targets), "records": records,
        }
        if split != "train":
            payload["original_masks"] = originals
        temporary = shard_path.with_name(shard_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        torch.save(payload, temporary)
        os.replace(temporary, shard_path)
        print(f"cache {source} {split}: {end}/{len(rows)} {tuple(payload['features'].shape)}", flush=True)
    if len(feature_shapes) != 1:
        raise RuntimeError(f"non-uniform {source} feature shapes: {sorted(feature_shapes)}")
    complete = {
        "status": "COMPLETE", "source": source, "split": split, "samples": len(rows),
        "sample_ids_sha256": ordered_ids_sha256([str(row["sample_id"]) for row in rows]),
        "feature_shape": list(next(iter(feature_shapes))), "storage_dtype": str(storage_dtype),
        "shard_size": shard_size, "shards": (len(rows) + shard_size - 1) // shard_size,
        "preprocessing_version": PREPROCESS_VERSIONS[source],
        "preprocessing_sha256": sha256_text(PREPROCESS_VERSIONS[source]),
        "source_checkpoint": source_checkpoint, "source_checkpoint_sha256": source_checkpoint_sha,
        "source_parameter_hash_before": source_parameter_hash,
        "source_parameter_hash_after": (
            module_hash(module) if source in {"sam", "clip"} else
            module_hash(expert) if source in {"npr", "srm"} else module_hash(focal.image_encoder)
        ),
        "cache_precision_audit_subset_n": len(cache_error["cosine"]),
        "cache_precision_audit": {
            "max_abs_error": cache_error["max_abs_error"],
            "mean_abs_error": cache_error["sum_abs"] / max(1, cache_error["count"]),
            "mean_cosine_similarity": float(np.mean(cache_error["cosine"])),
        },
        "seconds": time.time() - started,
    }
    complete["source_parameter_hash_exact"] = (
        complete["source_parameter_hash_before"] == complete["source_parameter_hash_after"]
    )
    if not complete["source_parameter_hash_exact"]:
        complete["status"] = "ABORT_FROZEN_BACKBONE_CHANGED"
    dump(complete_path, complete)
    if complete["status"].startswith("ABORT"):
        raise RuntimeError(complete["status"])


def main(argv=None):
    cli = parse_args(argv)
    config = load_config(cli.config)
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(3407)
    if cli.command == "audit":
        print(json.dumps(audit(config, cli.source, device, cli.split), ensure_ascii=False, indent=2))
    else:
        cache(config, cli.source, cli.split, device, cli.max_samples, cli.force)


if __name__ == "__main__":
    main()
