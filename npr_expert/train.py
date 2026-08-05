"""官方 NPR+SRM 的独立二分类训练入口。"""

import argparse
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import build_external_datasets, load_aigi_jsonl
from .official_npr_srm import OfficialNPRSRM
from .transforms import build_npr_focal_transform, build_npr_transform


DEFAULT_FOCAL_WEIGHTS = "/data/yz/myLISA_storage/checkpoints/FOCAL/FOCAL_ViT_weights.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="训练独立的官方 NPR+SRM 真假二分类器")
    parser.add_argument("--image-root", default="datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--train-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/train.jsonl")
    parser.add_argument("--val-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/val.jsonl")
    parser.add_argument("--test-jsonl", default="datasets/AIGI-Holmes-Dataset/dataset/test.jsonl")
    parser.add_argument("--deepfakejudge-jsonl", default="datasets/DeepfakeJudge/dfj-detect/data.jsonl")
    parser.add_argument("--deepfakejudge-root", default="datasets/DeepfakeJudge/dfj-detect")
    parser.add_argument("--fakebench-root", default="datasets/Fakebench")
    parser.add_argument("--loki-json", default="datasets/LOKI/true_or_false.json")
    parser.add_argument("--loki-root", default="datasets/LOKI")
    parser.add_argument("--output-dir", default="checkpoints_stage1")
    parser.add_argument("--run-name", default="official_npr_srm")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--load-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr-decay-every", type=int, default=20)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.9)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--sampler-mode",
        choices=("shuffle", "class_balanced", "source_balanced", "source_class_balanced"),
        default="shuffle",
    )
    parser.add_argument("--sampler-weight-power", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=0, help="仅用于调试；0 表示使用全部样本")
    parser.add_argument("--max-val-samples", type=int, default=0, help="仅用于调试；0 表示使用全部样本")
    parser.add_argument("--resume", default=None, help="恢复当前训练器生成的 last.pth 或 best.pth")
    parser.add_argument("--init-checkpoint", default=None, help="仅初始化 NPR+SRM 可训练权重，不恢复优化器")
    parser.add_argument("--focal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal-weights", default=DEFAULT_FOCAL_WEIGHTS)
    parser.add_argument("--focal-cache", default=None, help="预计算的 FOCAL 512 维特征缓存")
    parser.add_argument("--focal-input-size", type=int, default=1024)
    parser.add_argument("--focal-micro-batch-size", type=int, default=1)
    parser.add_argument("--focal-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--focal-only",
        action="store_true",
        help="仅使用冻结 FOCAL 的 512 维 mean+max 特征和可训练分类头，不运行 NPR/SRM",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-by-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后按生成来源评估 AIGI test（默认开启）",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("epochs 必须大于 0")
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    if args.crop_size > args.load_size:
        raise ValueError("crop-size 不能大于 load-size")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold 必须位于 (0, 1) 区间")
    if args.focal_input_size != 1024:
        raise ValueError("作者 FOCAL ViT-L 权重要求 focal-input-size=1024")
    if args.focal_micro_batch_size < 1:
        raise ValueError("focal-micro-batch-size 必须大于 0")
    if args.focal and not Path(args.focal_weights).is_file():
        raise FileNotFoundError(f"找不到 FOCAL 权重：{args.focal_weights}")
    if args.focal_only and not args.focal:
        raise ValueError("--focal-only 不能与 --no-focal 同时使用")
    if args.focal_only and args.init_checkpoint:
        raise ValueError("纯 FOCAL 实验不应使用 NPR+SRM 的 --init-checkpoint")
    if args.focal_cache and not Path(args.focal_cache).is_file():
        raise FileNotFoundError(f"找不到 FOCAL 特征缓存：{args.focal_cache}")
    if args.resume and args.init_checkpoint:
        raise ValueError("resume 与 init-checkpoint 不能同时使用")
    for path_name in ("train_jsonl", "val_jsonl"):
        path = Path(getattr(args, path_name))
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path_name}：{path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device}，但当前 PyTorch 无法使用 CUDA")


def build_sampler(dataset, mode, weight_power, seed):
    if mode == "shuffle":
        return None
    keys = []
    for _path, label, source in dataset.samples:
        if mode == "class_balanced":
            key = label
        elif mode == "source_balanced":
            key = source
        else:
            key = (source, label)
        keys.append(key)
    counts = Counter(keys)
    weights = torch.tensor([(1.0 / counts[key]) ** weight_power for key in keys], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def build_loader(dataset, args, *, shuffle=False, sampler=None):
    options = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": torch.device(args.device).type == "cuda",
    }
    if args.num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(**options)


def compute_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return {key: 0.0 for key in ("acc", "ap", "macro_f1", "real_acc", "fake_acc")}
    predictions = scores >= threshold
    real_mask = labels == 0
    fake_mask = labels == 1
    ap = average_precision_score(labels, scores) if fake_mask.any() else 0.0
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "ap": float(ap),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "real_acc": float(accuracy_score(labels[real_mask], predictions[real_mask])) if real_mask.any() else 0.0,
        "fake_acc": float(accuracy_score(labels[fake_mask], predictions[fake_mask])) if fake_mask.any() else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    labels = []
    scores = []
    for model_inputs, batch_labels, _sources in loader:
        images, focal_images = move_model_inputs(model_inputs, device)
        logits = model(images, focal_images=focal_images)
        scores.extend(torch.sigmoid(logits.float()).flatten().cpu().tolist())
        labels.extend(batch_labels.flatten().tolist())
    return compute_metrics(labels, scores, threshold)


def move_model_inputs(model_inputs, device):
    """Move NPR input to GPU while leaving large FOCAL images on pinned CPU."""
    if isinstance(model_inputs, dict):
        images = model_inputs["npr"].to(device, non_blocking=True)
        focal_images = model_inputs["focal"]
        if device.type != "cuda":
            focal_images = focal_images.to(device)
        return images, focal_images
    return model_inputs.to(device, non_blocking=True), None


def evaluate_datasets(model, datasets, args, device, title):
    print(f"\n{title}")
    results = {}
    for name, dataset in datasets.items():
        metrics = evaluate(model, build_loader(dataset, args), device, args.threshold)
        results[name] = metrics
        print(
            f"{name:<16} N={len(dataset):<7} ACC={metrics['acc']:.4f} "
            f"AP={metrics['ap']:.4f} F1={metrics['macro_f1']:.4f} "
            f"Real_ACC={metrics['real_acc']:.4f} Fake_ACC={metrics['fake_acc']:.4f}"
        )
    return results


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_ap, patience, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_ap": best_val_ap,
        "patience": patience,
        "meta": {
            "format": (
                "frozen_focal_linear_v1"
                if args.focal_only
                else "official_npr_srm_focal_v2" if args.focal else "official_npr_srm_v1"
            ),
            "preprocess": (
                f"NPR:Resize({args.load_size},{args.load_size})->Crop({args.crop_size})->ImageNetNormalize;"
                f"FOCAL:Resize({args.focal_input_size},{args.focal_input_size})->ToTensor"
                if args.focal
                else f"Resize({args.load_size},{args.load_size})->Crop({args.crop_size})->ImageNetNormalize"
            ),
            "positive_class": "fake",
            "args": vars(args),
        },
    }
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_initial_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"无法识别初始化权重：{path}")
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"初始化权重包含未知参数：{result.unexpected_keys}")


def load_focal_feature_cache(path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "focal_vit_l_mean_max_v1":
        raise ValueError(f"无法识别 FOCAL 特征缓存：{path}")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"FOCAL 特征缓存缺少 features：{path}")
    return features


def load_training_checkpoint(path, model, optimizer, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("resume 仅支持当前训练器生成的检查点")
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_val_ap", -1.0)),
        int(checkpoint.get("patience", 0)),
    )


def append_history(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, content):
    Path(path).write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def describe_device(device):
    if device.type != "cuda":
        return str(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (
        f"{device} {torch.cuda.get_device_name(device)}，"
        f"空闲 {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB"
    )


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        run_dir = Path(args.resume).resolve().parent
        if not run_dir.is_dir():
            raise FileNotFoundError(f"恢复目录不存在：{run_dir}")
        config_name = f"resume_config_{timestamp}.json"
    else:
        run_dir = Path(args.output_dir) / f"{args.run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        config_name = "training_config.json"
    (run_dir / config_name).write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    transform_builder = build_npr_focal_transform if args.focal else build_npr_transform
    transform_options = {"focal_input_size": args.focal_input_size} if args.focal else {}
    train_transform = transform_builder(
        args.load_size, args.crop_size, training=True, **transform_options
    )
    val_transform = transform_builder(
        args.load_size, args.crop_size, training=False, **transform_options
    )
    test_transform = transform_builder(
        args.load_size, args.crop_size, training=False, no_crop=True, **transform_options
    )
    train_dataset = load_aigi_jsonl(args.train_jsonl, args.image_root, train_transform)
    val_dataset = load_aigi_jsonl(args.val_jsonl, args.image_root, val_transform)
    if args.max_train_samples > 0:
        train_dataset.samples = train_dataset.samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_dataset.samples = val_dataset.samples[: args.max_val_samples]
    if not train_dataset.samples or not val_dataset.samples:
        raise ValueError("训练集和验证集都必须至少包含一个样本")
    train_sampler = build_sampler(
        train_dataset, args.sampler_mode, args.sampler_weight_power, args.seed
    )
    train_loader = build_loader(train_dataset, args, shuffle=True, sampler=train_sampler)
    val_loader = build_loader(val_dataset, args)

    test_dataset = None
    standard_test_datasets = {}
    if args.evaluate_test and Path(args.test_jsonl).is_file():
        test_dataset = load_aigi_jsonl(args.test_jsonl, args.image_root, test_transform)
        standard_test_datasets["aigi_test"] = test_dataset
    if args.evaluate_external:
        standard_test_datasets.update(build_external_datasets(args, test_transform))

    if args.focal_cache:
        focal_feature_cache = load_focal_feature_cache(args.focal_cache)
        train_dataset.set_focal_feature_cache(focal_feature_cache)
        val_dataset.set_focal_feature_cache(focal_feature_cache)
        for dataset in standard_test_datasets.values():
            dataset.set_focal_feature_cache(focal_feature_cache)

    model = OfficialNPRSRM(
        focal_weights=args.focal_weights if args.focal and not args.focal_cache else None,
        focal_micro_batch_size=args.focal_micro_batch_size,
        focal_gate_init=args.focal_gate_init,
        use_cached_focal=bool(args.focal and args.focal_cache),
        focal_only=args.focal_only,
    )
    if args.init_checkpoint:
        load_initial_checkpoint(args.init_checkpoint, model)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(args.beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, best_val_ap, patience = 1, -1.0, 0
    if args.resume:
        start_epoch, best_val_ap, patience = load_training_checkpoint(
            args.resume, model, optimizer, scaler, device
        )

    print("=" * 72)
    print("独立 NPR+SRM 训练")
    print(f"设备：       {describe_device(device)}")
    print(f"训练集：     {args.train_jsonl}（{len(train_dataset)}）")
    print(f"验证集：     {args.val_jsonl}（{len(val_dataset)}）")
    print(f"采样：       {args.sampler_mode}")
    model_name = (
        "冻结 FOCAL ViT-L + LayerNorm + 线性分类头"
        if args.focal_only
        else "官方 NPR + SRM + 冻结 FOCAL ViT-L" if args.focal else "官方 NPR + SRM"
    )
    print(f"模型：       {model_name}，单 logit BCE")
    if args.focal:
        focal_source = args.focal_cache or args.focal_weights
        print(f"FOCAL：      {focal_source}，micro-batch={args.focal_micro_batch_size}")
    print(f"输出目录：   {run_dir}")
    print("=" * 72)

    before_evaluation_path = run_dir / "evaluation_before_training.json"
    if args.resume and before_evaluation_path.is_file():
        before_training_results = json.loads(before_evaluation_path.read_text(encoding="utf-8"))
        print(f"已复用训练前评估结果：{before_evaluation_path}")
    else:
        before_training_results = evaluate_datasets(
            model,
            standard_test_datasets,
            args,
            device,
            "训练前测试结果",
        )
        write_json(before_evaluation_path, before_training_results)

    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    history_path = run_dir / "history.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for model_inputs, labels, _sources in train_loader:
            images, focal_images = move_model_inputs(model_inputs, device)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images, focal_images=focal_images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += ((torch.sigmoid(logits.detach()) >= args.threshold) == labels.bool()).sum().item()

        if args.lr_decay_every > 0 and epoch % args.lr_decay_every == 0:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * args.lr_decay_gamma, args.min_lr)

        val_metrics = evaluate(model, val_loader, device, args.threshold)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - epoch_start,
            "val": val_metrics,
        }
        improved = val_metrics["ap"] > best_val_ap
        if improved:
            best_val_ap = val_metrics["ap"]
            patience = 0
        else:
            patience += 1
        record["best_val_ap"] = best_val_ap
        append_history(history_path, record)
        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val_ap, patience, args)
        if improved:
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val_ap, patience, args)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] loss={record['train_loss']:.4f} "
            f"train_acc={record['train_acc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_ap={val_metrics['ap']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"lr={record['lr']:.6g} time={record['elapsed_seconds']:.1f}s"
        )
        if improved:
            print(f"======> 已保存新的最佳权重：{best_path}")
        elif patience >= args.early_stop_patience:
            print(f"======> 验证集 AP 连续 {patience} 个 epoch 未提升，提前停止")
            break

    if not best_path.is_file():
        raise RuntimeError("训练未生成最佳权重")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"], strict=True)

    after_training_results = evaluate_datasets(
        model,
        standard_test_datasets,
        args,
        device,
        "最佳权重训练后测试结果",
    )
    by_source_results = {}
    if args.eval_by_source and test_dataset is not None:
        source_datasets = {
            source: test_dataset.subset_by_source(source)
            for source in sorted(test_dataset.source_counts())
        }
        by_source_results = evaluate_datasets(
            model, source_datasets, args, device, "AIGI 按生成来源测试结果"
        )
    write_json(
        run_dir / "evaluation.json",
        {
            "before_training": before_training_results,
            "after_training": after_training_results,
            "aigi_by_source_after_training": by_source_results,
        },
    )
    print(f"\n训练完成，最佳验证 AP={best_val_ap:.4f}")
    print(f"最佳权重：{best_path}")


if __name__ == "__main__":
    main()
