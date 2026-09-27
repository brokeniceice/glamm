#!/usr/bin/env python3
"""Frozen official I-JEPA patch features on the canonical CLIP crop (train/DEV only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/data/yz/groundingLMM_official/third_party/ijepa")
WEIGHT = Path("/data/yz/groundingLMM_official/checkpoints/ijepa/IN1K-vit.h.16-448px-300e.pth.tar")
OUT = Path("/data/yz/groundingLMM_official/cache/phase6i_ijepa_448")
SPATIAL = ROOT / "outputs/phase3c1_spatial_probe/cache/clip"
sys.path.insert(0, str(REPO))
from src.models.vision_transformer import vit_huge  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_tensor(row: dict) -> torch.Tensor:
    geo = row["geometry"]
    rh, rw = map(int, geo["resized_hw"])
    top, left, bottom, right = map(int, geo["crop_box_yxyx"])
    with Image.open(row["image_path"]) as src:
        image = src.convert("RGB")
    if (image.height, image.width) != tuple(geo["original_hw"]):
        raise RuntimeError(f"image geometry drift: {row['sample_id']}")
    image = image.resize((rw, rh), Image.Resampling.BICUBIC)
    image = image.crop((left, top, right, bottom))
    if image.size != (336, 336):
        raise RuntimeError(f"CLIP crop drift: {row['sample_id']}")
    image = image.resize((448, 448), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = x.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (x - mean) / std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, required=True, choices=(0, 1))
    args = p.parse_args()
    if not WEIGHT.is_file() or not REPO.is_dir():
        raise RuntimeError("official I-JEPA source/checkpoint absent")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = vit_huge(img_size=[448], patch_size=16).to(device).eval()
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    if "target_encoder" not in checkpoint:
        raise RuntimeError(f"unexpected official checkpoint keys: {list(checkpoint)}")
    state = checkpoint["target_encoder"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del checkpoint
    model.requires_grad_(False)
    for split in ("train", "val"):
        paths = sorted((SPATIAL / split).glob("shard_*.pt"))
        if not paths:
            raise RuntimeError(f"missing canonical CLIP cache: {split}")
        outdir = OUT / split
        outdir.mkdir(parents=True, exist_ok=True)
        completed = []
        for index, path in enumerate(paths):
            if index % 2 != args.worker:
                continue
            dest = outdir / path.name
            if dest.exists():
                payload = torch.load(dest, map_location="cpu", weights_only=False)
                if payload.get("source_path") != str(path.resolve()):
                    raise RuntimeError(f"cache provenance drift: {dest}")
                completed.append({"path": str(dest), "n": len(payload["sample_ids"])})
                continue
            source = torch.load(path, map_location="cpu", weights_only=False)
            rows = source["records"]
            features = []
            with torch.inference_mode():
                for start in range(0, len(rows), 4):
                    x = torch.stack([image_tensor(row) for row in rows[start:start + 4]]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens = model(x)
                    if tokens.shape[1:] != (784, 1280):
                        raise RuntimeError(f"I-JEPA grid drift: {tokens.shape}")
                    grid = tokens.transpose(1, 2).reshape(-1, 1280, 28, 28).float()
                    aligned = F.interpolate(grid, (24, 24), mode="bilinear", align_corners=False)
                    features.append(aligned.to(torch.bfloat16).cpu())
            tensor = torch.cat(features)
            if tensor.shape != (len(rows), 1280, 24, 24):
                raise RuntimeError(f"aligned feature drift: {tensor.shape}")
            payload = {"schema": "phase6i_ijepa_448_clip_crop_v1", "source_path": str(path.resolve()),
                       "sample_ids": [str(row["sample_id"]) for row in rows], "features": tensor,
                       "weight_path": str(WEIGHT), "source": "official target_encoder"}
            temp = dest.with_suffix(".tmp")
            torch.save(payload, temp)
            os.replace(temp, dest)
            completed.append({"path": str(dest), "n": len(rows)})
            print(json.dumps({"worker": args.worker, "split": split, "shard": index,
                              "n": len(rows), "completed_shards": len(completed)}), flush=True)
        status = {"status": "COMPLETE", "worker": args.worker, "split": split,
                  "source_shards": len(paths), "completed": completed,
                  "checkpoint_sha256": sha(WEIGHT), "repo_commit": "52c1ae95d05f743e000e8f10a1f3a79b10cff048"}
        temp = OUT / f"worker_{args.worker}_{split}.json.tmp"
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        os.replace(temp, OUT / f"worker_{args.worker}_{split}.json")


if __name__ == "__main__":
    main()
