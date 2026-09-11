#!/usr/bin/env python3
"""Cache matched frozen CLIP-CLS and Phase4C-A F24-GAP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_forensic_adapter import CLIPSpatialArm

OUT = ROOT / "outputs/phase6b_classification_attribution"
MANIFEST_ROOT = ROOT / "outputs/data_audits/unified_forensics_split_v1"
CLIP_REV = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CLIP_ID = "openai/clip-vit-large-patch14-336"
ADAPTER_CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase4c_a_clip_forensic_adapter/forensic_adapter/selected.pt")
EXPECTED_CLIP_HASH = "ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(values) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(named) -> str:
    h = hashlib.sha256()
    for name, value in sorted(named):
        t = value.detach().contiguous().cpu()
        h.update(name.encode() + b"\0")
        h.update(str(t.dtype).encode() + b"\0")
        h.update(json.dumps(list(t.shape)).encode() + b"\0")
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_rows(split: str):
    path = MANIFEST_ROOT / f"{split}_combined.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    expected = 17672 if split == "train" else 2212
    if len(rows) != expected or len({r["sample_id"] for r in rows}) != expected:
        raise RuntimeError(f"{split} manifest count/uniqueness failure")
    if any(int(r["class_label"]) != (r["forensics_domain"] == "fake") for r in rows):
        raise RuntimeError(f"{split} label/domain conflict")
    return path, rows


def resolve_path(row):
    p = Path(str(row.get("image_path") or ""))
    if p.is_file():
        return p.resolve()
    base = Path("/data/yz/myLISA_storage/AIGC/SynthScars") if row["forensics_domain"] == "fake" else ROOT / "datasets"
    p = base / row["image_relpath"]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.resolve()


class ImageRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = resolve_path(row)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"decode failed: {path}")
        return i, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_FEATURE_EXTRACTION_OR_TRAINING":
        raise RuntimeError("protocol is not frozen")

    cache_home = Path("/data/yz/myLISA_storage/checkpoints/.hf_cache")
    os.environ.setdefault("HF_HOME", str(cache_home))
    processor = CLIPImageProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV, local_files_only=True)
    # Canonical P1 stores the frozen vision tower in BF16.  The historical
    # Phase3C.1 hash covers the CLIPVisionTower wrapper state_dict, hence the
    # `vision_tower.` prefix and buffers (not just named_parameters).
    clip = CLIPVisionModel.from_pretrained(
        CLIP_ID, revision=CLIP_REV, local_files_only=True
    ).to(device=args.device, dtype=torch.bfloat16).eval()
    for x in clip.parameters():
        x.requires_grad_(False)
    clip_hash = tensor_hash(
        [(f"vision_tower.{name}", value) for name, value in clip.state_dict().items()]
    )
    if clip_hash != EXPECTED_CLIP_HASH:
        raise RuntimeError(f"CLIP parameter hash drift: {clip_hash}")

    saved = torch.load(ADAPTER_CKPT, map_location="cpu")
    adapter = CLIPSpatialArm(blocks=3)
    adapter.load_state_dict(saved["model"], strict=True)
    adapter = adapter.to(args.device).eval()
    for x in adapter.parameters():
        x.requires_grad_(False)
    adapter_hash = tensor_hash(adapter.named_parameters())

    split_meta = {}
    for split in ("train", "val"):
        manifest_path, rows = load_rows(split)
        paths = [str(resolve_path(row)) for row in rows]
        # Full preflight before the first model batch: no silent population reduction.
        if len(paths) != len(rows):
            raise RuntimeError("path population drift")
        ids = [r["sample_id"] for r in rows]
        labels = [int(r["class_label"]) for r in rows]
        image_keys = [r.get("content_sha256") or r.get("identity_hash", {}).get("value") or p for r, p in zip(rows, paths)]
        clip_features = torch.empty((len(rows), 1024), dtype=torch.float16)
        f24_features = torch.empty((len(rows), 256), dtype=torch.float16)

        def collate(batch):
            idx, images = zip(*batch)
            pixels = processor(images=list(images), return_tensors="pt")["pixel_values"]
            return torch.tensor(idx), pixels

        loader = DataLoader(ImageRows(rows), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate,
                            persistent_workers=args.workers > 0)
        cursor = 0
        with torch.inference_mode():
            for index, pixels in loader:
                expected_index = torch.arange(cursor, cursor + len(index))
                if not torch.equal(index, expected_index):
                    raise RuntimeError("DataLoader order drift")
                pixels = pixels.to(args.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = clip(pixels, output_hidden_states=True).hidden_states[-2]
                    cls = hidden[:, 0]
                    patch = hidden[:, 1:].transpose(1, 2).reshape(-1, 1024, 24, 24)
                    f24 = adapter(patch.float(), return_features=True)["F_forensic"]
                    gap = f24.mean(dim=(2, 3))
                clip_features[cursor:cursor + len(index)] = cls.cpu().to(torch.float16)
                f24_features[cursor:cursor + len(index)] = gap.cpu().to(torch.float16)
                cursor += len(index)
                if cursor % 1024 < len(index):
                    print(json.dumps({"split": split, "done": cursor, "total": len(rows)}), flush=True)
        if cursor != len(rows) or not torch.isfinite(clip_features).all() or not torch.isfinite(f24_features).all():
            raise RuntimeError(f"{split} incomplete/nonfinite feature cache")
        payload = {
            "schema": "phase6b_matched_feature_cache_v1", "split": split,
            "sample_ids": ids, "labels": torch.tensor(labels, dtype=torch.int64),
            "image_keys": image_keys, "image_paths": paths,
            "clip_cls": clip_features, "f24_gap": f24_features,
        }
        out_path = OUT / "features" / split / "features.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        split_meta[split] = {
            "n": len(rows), "feature_file": str(out_path.resolve()), "feature_file_sha256": sha256_file(out_path),
            "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path),
            "sample_order_sha256": canonical_hash(ids), "label_sha256": canonical_hash(labels),
            "image_identity_sha256": canonical_hash(image_keys), "clip_shape": list(clip_features.shape),
            "f24_gap_shape": list(f24_features.shape), "dtype": "float16", "failures": 0,
        }

    feature_manifest = {
        "schema": "phase6b_feature_manifest_v1", "status": "COMPLETE",
        "clip": {"identity": CLIP_ID, "revision": CLIP_REV, "parameter_sha256": clip_hash,
                 "preprocessing": processor.to_dict()},
        "forensic_arm": {"checkpoint": str(ADAPTER_CKPT), "checkpoint_sha256": sha256_file(ADAPTER_CKPT),
                         "parameter_sha256": adapter_hash, "selected_epoch": int(saved["epoch"])},
        "r1_checkpoint": {"path": str((ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt").resolve()),
                          "sha256": sha256_file(ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt")},
        "splits": split_meta,
        "integrity": {"same_forward_for_both_features": True, "same_sample_id_label_image": True,
                      "population_failures": 0, "backbone_gradients": False},
        "firewall": {"internal_test_loaded": False, "official1000_loaded": False, "external_loaded": False}
    }
    (OUT / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "splits": split_meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
