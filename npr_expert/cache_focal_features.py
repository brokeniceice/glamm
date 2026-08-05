"""Precompute deterministic frozen FOCAL mean+max features for all datasets."""

import argparse
import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from .data import build_external_datasets, load_aigi_jsonl
from .focal_extractor import FrozenFOCALViT
from .train import DEFAULT_FOCAL_WEIGHTS
from .transforms import DualExpertTransform


ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_CACHE_PATH = "/data/yz/myLISA_storage/checkpoints/FOCAL/focal_vit_l_mean_max_all.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="缓存冻结 FOCAL ViT-L 的 512 维 mean+max 特征")
    parser.add_argument("--image-root", default="datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--train-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/train.jsonl")
    parser.add_argument("--val-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/val.jsonl")
    parser.add_argument("--test-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/test.jsonl")
    parser.add_argument("--deepfakejudge-jsonl", default="datasets/DeepfakeJudge/dfj-detect/data.jsonl")
    parser.add_argument("--deepfakejudge-root", default="datasets/DeepfakeJudge/dfj-detect")
    parser.add_argument("--fakebench-root", default="datasets/Fakebench")
    parser.add_argument("--loki-json", default="datasets/LOKI/true_or_false.json")
    parser.add_argument("--loki-root", default="datasets/LOKI")
    parser.add_argument("--focal-weights", default=DEFAULT_FOCAL_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = DualExpertTransform(lambda image: image).focal_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as error:
            raise RuntimeError(f"无法读取图像：{image_path}") from error
        return tensor, image_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_cache(path, features, weights_path, weights_sha256):
    payload = {
        "format": "focal_vit_l_mean_max_v1",
        "weights": str(weights_path),
        "weights_sha256": weights_sha256,
        "feature_dtype": "float16",
        "features": features,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def collect_paths(args):
    identity = lambda image: image
    datasets = []
    for jsonl_path in (args.train_jsonl, args.val_jsonl, args.test_jsonl):
        if Path(jsonl_path).is_file():
            datasets.append(load_aigi_jsonl(jsonl_path, args.image_root, identity))
    external_args = SimpleNamespace(
        deepfakejudge_jsonl=args.deepfakejudge_jsonl,
        deepfakejudge_root=args.deepfakejudge_root,
        fakebench_root=args.fakebench_root,
        loki_json=args.loki_json,
        loki_root=args.loki_root,
    )
    datasets.extend(build_external_datasets(external_args, identity).values())
    return list(dict.fromkeys(path for dataset in datasets for path, _label, _source in dataset.samples))


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.save_every < 1:
        raise ValueError("batch-size/save-every 必须大于 0，num-workers 不能小于 0")
    device = torch.device(args.device)
    paths = collect_paths(args)
    output_path = Path(args.output)
    weights_digest = sha256(args.focal_weights)
    features = {}
    if output_path.is_file():
        existing = torch.load(output_path, map_location="cpu")
        if existing.get("format") != "focal_vit_l_mean_max_v1":
            raise ValueError(f"已有缓存格式不兼容：{output_path}")
        if existing.get("weights_sha256") != weights_digest:
            raise ValueError("已有缓存使用了不同的 FOCAL 权重")
        features = existing["features"]
    remaining_paths = [path for path in paths if path not in features]
    print(f"FOCAL 缓存：总计 {len(paths)}，已有 {len(features)}，待提取 {len(remaining_paths)}")
    if not remaining_paths:
        return

    loader_options = {
        "dataset": ImagePathDataset(remaining_paths),
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_options)
    model = FrozenFOCALViT(args.focal_weights, micro_batch_size=args.batch_size).to(device)
    start = time.time()
    completed_since_save = 0
    for tensors, batch_paths in loader:
        pooled = model(tensors).cpu().half()
        for image_path, feature in zip(batch_paths, pooled):
            features[image_path] = feature.clone()
        completed_since_save += len(batch_paths)
        completed = len(features)
        if completed_since_save >= args.save_every or completed == len(paths):
            save_cache(output_path, features, args.focal_weights, weights_digest)
            elapsed = time.time() - start
            processed = len(paths) - len(remaining_paths) + completed_since_save
            rate = completed_since_save / max(elapsed, 1e-6)
            print(f"已缓存 {completed}/{len(paths)}，当前阶段 {rate:.2f} image/s，保存至 {output_path}")
            completed_since_save = 0
            start = time.time()


if __name__ == "__main__":
    main()
