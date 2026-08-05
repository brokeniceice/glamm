"""NPR+SRM 训练和评估使用的数据集。"""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _clean_relative_path(path):
    path = str(path)
    return path[2:] if path.startswith("./") else path


def _source_from_path(path):
    stem = Path(path).stem
    parts = stem.split("_")
    if stem.startswith("code_stable-diffusion"):
        return "code_stable-diffusion"
    if stem.startswith("generators_stable-diffusion"):
        return "generators_stable-diffusion"
    if stem.startswith("imagenet_ai_0419") and len(parts) >= 4:
        return "_".join(parts[:4])
    return parts[0] if parts else "unknown"


class BinaryImageDataset(Dataset):
    """由 ``(图像路径, 标签, 来源)`` 三元组构成的二分类数据集。"""

    def __init__(self, samples, transform, name="dataset"):
        self.samples = list(samples)
        self.transform = transform
        self.name = name
        self.focal_feature_cache = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, source = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"无法读取图像：{image_path}") from error
        if self.focal_feature_cache is not None:
            if image_path not in self.focal_feature_cache:
                raise KeyError(f"FOCAL 特征缓存缺少图像：{image_path}")
            if not hasattr(self.transform, "npr_transform"):
                raise TypeError("FOCAL 缓存要求 DualExpertTransform")
            model_input = {
                "npr": self.transform.npr_transform(image),
                "focal": self.focal_feature_cache[image_path],
            }
        else:
            model_input = self.transform(image)
        return model_input, torch.tensor(label, dtype=torch.float32), source

    def set_focal_feature_cache(self, cache):
        self.focal_feature_cache = cache
        return self

    def source_counts(self):
        return Counter(source for _path, _label, source in self.samples)

    def subset_by_source(self, source):
        samples = [sample for sample in self.samples if sample[2] == source]
        dataset = BinaryImageDataset(samples, self.transform, name=f"{self.name}:{source}")
        dataset.focal_feature_cache = self.focal_feature_cache
        return dataset


def load_aigi_jsonl(jsonl_path, image_root, transform):
    """读取 AIGI-Holmes JSONL；存在 mask 的样本标为伪造。"""

    samples = []
    jsonl_path = Path(jsonl_path)
    image_root = Path(image_root)
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            images = item.get("images", [])
            if not images:
                raise ValueError(f"{jsonl_path}:{line_number} 缺少 images 字段")
            relative_path = _clean_relative_path(images[0])
            mask_path = item.get("mask", "")
            label = int(isinstance(mask_path, str) and bool(mask_path.strip()))
            samples.append(
                (str(image_root / relative_path), label, _source_from_path(relative_path))
            )
    return BinaryImageDataset(samples, transform, name=jsonl_path.stem)


def load_deepfakejudge(jsonl_path, image_root, transform):
    samples = []
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        return None
    image_root = Path(image_root)
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            images = item.get("images", [])
            if not images:
                continue
            relative_path = _clean_relative_path(images[0])
            label = int(str(item.get("answer", "")).strip().lower() == "fake")
            source = str(item.get("source") or _source_from_path(relative_path))
            samples.append((str(image_root / relative_path), label, source))
    return BinaryImageDataset(samples, transform, "deepfakejudge") if samples else None


def _list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        str(path)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "__MACOSX" not in path.parts
    )


def load_fakebench(data_root, transform):
    samples = []
    for image_path in _list_images(Path(data_root) / "fake_images"):
        source = Path(image_path).name.split("-")[0] or "fakebench_fake"
        samples.append((image_path, 1, source))
    samples.extend((image_path, 0, "fakebench_real") for image_path in _list_images(Path(data_root) / "real_images"))
    return BinaryImageDataset(samples, transform, "fakebench") if samples else None


def load_loki(json_path, image_root, transform):
    samples = []
    json_path = Path(json_path)
    if not json_path.is_file():
        return None
    dataset_root = Path(image_root) if image_root else json_path.parent
    content = json_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    items = json.loads(content) if content.startswith("[") else [
        json.loads(line) for line in content.splitlines() if line.strip()
    ]
    seen = set()
    for item in items:
        images = item.get("images", [])
        image_path = item.get("image_path") or item.get("image") or (images[0] if images else "")
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        question_type = str(item.get("question_type", "")).lower()
        label = int("image_tf_image_fake_" in question_type)
        resolved_path = Path(image_path)
        if not resolved_path.is_absolute():
            resolved_path = dataset_root / resolved_path
        samples.append((str(resolved_path), label, question_type or "loki"))
    return BinaryImageDataset(samples, transform, "loki") if samples else None


def build_external_datasets(args, transform):
    """构造可用的外部测试集；缺失的数据集会被跳过。"""

    candidates = {
        "deepfakejudge": load_deepfakejudge(
            args.deepfakejudge_jsonl, args.deepfakejudge_root, transform
        ),
        "fakebench": load_fakebench(args.fakebench_root, transform),
        "loki": load_loki(args.loki_json, args.loki_root, transform),
    }
    return {name: dataset for name, dataset in candidates.items() if dataset is not None}
