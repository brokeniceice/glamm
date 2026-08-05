"""Evaluate the official FOCAL ViT-L as a zero-shot image-level detector.

No model parameter is trained.  The published 64x64 dense descriptors are
L2-normalized and clustered per image with the official torch-kmeans setup.
The minority cluster is treated as the predicted forged mask.  Image-level
scores are derived from that mask and calibrated only on AIGI validation data.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity

from .data import load_aigi_jsonl, load_loki
from .focal_extractor import FrozenFOCALViT


DEFAULT_WEIGHTS = "/data/yz/myLISA_storage/checkpoints/FOCAL/FOCAL_ViT_weights.pth"
SCORE_NAMES = ("area", "separation", "area_x_separation")


def parse_args():
    parser = argparse.ArgumentParser(description="官方 FOCAL 零训练图像级评估")
    parser.add_argument("--aigi-root", default="datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--aigi-val-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/val.jsonl")
    parser.add_argument("--aigi-test-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/test.jsonl")
    parser.add_argument("--loki-json", default="datasets/LOKI/true_or_false.json")
    parser.add_argument("--loki-root", default="datasets/LOKI")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="official_focal_zeroshot")
    parser.add_argument("--resume-dir", default=None)
    return parser.parse_args()


class OfficialFOCALImageDataset(Dataset):
    """Read images using the same OpenCV/resize/divide-by-255 path as FOCAL."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, source = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图像：{image_path}")
        image = image[..., ::-1]
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, int(label), source, image_path


def build_loader(samples, args):
    options = dict(
        dataset=OfficialFOCALImageDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_records(path, records):
    with Path(path).open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def derive_cluster_scores(features, labels):
    """Return official minority masks and image-level heuristic scores."""
    batch, pixels, channels = features.shape
    counts_one = labels.sum(dim=1)
    minority_is_one = counts_one <= (pixels - counts_one)
    masks = torch.where(minority_is_one[:, None], labels.bool(), ~labels.bool())
    area = masks.float().mean(dim=1)

    cluster_zero = (~labels.bool()).to(features.dtype)
    cluster_one = labels.bool().to(features.dtype)
    center_zero = torch.einsum("bn,bnc->bc", cluster_zero, features)
    center_one = torch.einsum("bn,bnc->bc", cluster_one, features)
    center_zero = F.normalize(center_zero, dim=1)
    center_one = F.normalize(center_one, dim=1)
    separation = 1.0 - (center_zero * center_one).sum(dim=1)
    return masks, area, separation, area * separation


@torch.no_grad()
def evaluate_dataset(name, samples, encoder, clustering, args, run_dir):
    result_path = run_dir / f"{name}_scores.jsonl"
    existing = load_records(result_path)
    completed = {record["path"] for record in existing}
    remaining = [sample for sample in samples if sample[0] not in completed]
    print(f"{name}: 总计 {len(samples)}，已完成 {len(completed)}，待处理 {len(remaining)}")
    if not remaining:
        return existing

    start = time.time()
    processed = 0
    for images, labels, sources, paths in build_loader(remaining, args):
        images = images.to(args.device, non_blocking=True)
        spatial = encoder(images.float()).permute(0, 2, 3, 1)
        features = F.normalize(spatial, dim=3).flatten(1, 2)
        cluster_result = clustering(x=features, k=2)
        masks, area, separation, joint = derive_cluster_scores(
            features, cluster_result.labels
        )
        foreground_pixels = masks.sum(dim=1)
        batch_records = []
        for index, path in enumerate(paths):
            batch_records.append(
                {
                    "path": path,
                    "label": int(labels[index]),
                    "source": sources[index],
                    "foreground_pixels": int(foreground_pixels[index]),
                    "total_pixels": int(masks.shape[1]),
                    "area": float(area[index]),
                    "separation": float(separation[index]),
                    "area_x_separation": float(joint[index]),
                }
            )
        append_records(result_path, batch_records)
        existing.extend(batch_records)
        processed += len(batch_records)
        elapsed = time.time() - start
        rate = processed / max(elapsed, 1e-6)
        print(
            f"\r{name}: {len(completed) + processed}/{len(samples)} "
            f"({rate:.2f} image/s)",
            end="",
            flush=True,
        )
    print()
    return existing


def compute_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    real = labels == 0
    fake = labels == 1
    ap = average_precision_score(labels, scores) if fake.any() else 0.0
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, predictions)),
        "ap": float(ap),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "real_acc": float(accuracy_score(labels[real], predictions[real])) if real.any() else None,
        "fake_acc": float(accuracy_score(labels[fake], predictions[fake])) if fake.any() else None,
    }


def select_threshold(records, score_name):
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    scores = np.asarray([record[score_name] for record in records], dtype=np.float64)
    unique = np.unique(scores)
    if len(unique) == 1:
        return float(unique[0])
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2.0,
         [np.nextafter(unique[-1], np.inf)])
    )
    best = None
    for threshold in candidates:
        metrics = compute_metrics(labels, scores, threshold)
        key = (metrics["macro_f1"], metrics["acc"])
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def summarize(records, thresholds):
    labels = [record["label"] for record in records]
    results = {}
    for score_name in SCORE_NAMES:
        scores = [record[score_name] for record in records]
        metrics = compute_metrics(labels, scores, thresholds[score_name])
        real_scores = [score for score, label in zip(scores, labels) if label == 0]
        fake_scores = [score for score, label in zip(scores, labels) if label == 1]
        metrics["real_score_mean"] = float(np.mean(real_scores)) if real_scores else None
        metrics["fake_score_mean"] = float(np.mean(fake_scores)) if fake_scores else None
        results[score_name] = metrics
    return results


def summarize_by_source(records, thresholds):
    sources = sorted({record["source"] for record in records})
    return {
        source: summarize(
            [record for record in records if record["source"] == source], thresholds
        )
        for source in sources
    }


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("官方 FOCAL 零样本评估要求 CUDA")

    torch.manual_seed(666666)
    torch.cuda.manual_seed_all(666666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.resume_dir)
        if args.resume_dir
        else Path(args.output_dir) / f"{args.run_name}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=bool(args.resume_dir))
    atomic_json(run_dir / "config.json", vars(args))

    placeholder = lambda image: image
    aigi_val = load_aigi_jsonl(args.aigi_val_jsonl, args.aigi_root, placeholder).samples
    aigi_test = load_aigi_jsonl(args.aigi_test_jsonl, args.aigi_root, placeholder).samples
    loki_dataset = load_loki(args.loki_json, args.loki_root, placeholder)
    if loki_dataset is None:
        raise FileNotFoundError("LOKI 数据集不可用")

    focal = FrozenFOCALViT(args.weights).image_encoder.to(args.device).eval()
    clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)
    print(f"输出目录：{run_dir}")
    print("官方链路：OpenCV Resize(1024) -> /255 -> ViT-L -> L2 -> cosine KMeans(k=2)")

    records = {}
    for name, samples in (
        ("aigi_val", aigi_val),
        ("aigi_test", aigi_test),
        ("loki", loki_dataset.samples),
    ):
        records[name] = evaluate_dataset(
            name, samples, focal, clustering, args, run_dir
        )

    thresholds = {
        score_name: select_threshold(records["aigi_val"], score_name)
        for score_name in SCORE_NAMES
    }
    evaluation = {
        "protocol": {
            "training": False,
            "weights": args.weights,
            "input": "OpenCV RGB Resize(1024,1024), float32 / 255",
            "features": "FOCAL ViT-L [256,64,64], per-pixel L2 normalization",
            "clustering": "torch-kmeans 0.2.0 KMeans(k=2, CosineSimilarity, official defaults)",
            "mask_orientation": "minority cluster is forged",
            "threshold_calibration": "maximize macro-F1 on AIGI val",
        },
        "thresholds": thresholds,
        "aigi_val": summarize(records["aigi_val"], thresholds),
        "aigi_test": summarize(records["aigi_test"], thresholds),
        "loki": summarize(records["loki"], thresholds),
        "aigi_test_by_source": summarize_by_source(records["aigi_test"], thresholds),
    }
    atomic_json(run_dir / "evaluation.json", evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(f"完成：{run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
