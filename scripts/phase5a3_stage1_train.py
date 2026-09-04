#!/usr/bin/env python3
"""External, auditable Stage-1 runner for the unmodified official LEGION model.

The wrapper corrects three launcher/data-loop defects without editing LEGION:
the official one-epoch hard break, random-with-replacement HybridSegDataset,
and validation truncation to 1,000 rows.  Model, collate, prompt, loss, LoRA,
optimizer, scheduler, and checkpoint implementation remain official.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/LEGION_official"
sys.path.insert(0, str(OFFICIAL))
# The pinned repository's utils file only defines its used GCG prompt, while
# train.py eagerly imports unused caption/region/segmentation dataset modules.
# Supply inert import-time placeholders; this runner never instantiates them.
import dataset.utils.utils as dataset_prompts  # noqa: E402
import dataset.utils as dataset_utils_package  # noqa: E402
dataset_utils_package.__path__.append(str(ROOT / "dataset/utils"))
for missing_name in ("CAPTION_QUESTIONS", "REGION_QUESTIONS", "REGION_GROUP_QUESTIONS", "SEG_QUESTIONS"):
    if not hasattr(dataset_prompts, missing_name):
        setattr(dataset_prompts, missing_name, [])
spec = importlib.util.spec_from_file_location(
    "legion_official_loc_train", OFFICIAL / "scripts/loc_exp/train.py"
)
official = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(official)

from dataset.dataset import custom_collate_fn  # noqa: E402
from dataset.gcg_datasets.GranDf_gcg_ds import LegionGCGDataset  # noqa: E402
from tools.utils import dict_to_cuda  # noqa: E402


LOSS_KEYS = ("loss", "ce_loss", "mask_bce_loss", "mask_dice_loss", "mask_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--vision-pretrained", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CompleteLegionDataset(LegionGCGDataset):
    """Official dataset behavior, except no validation first-1000 truncation."""

    def _load_annotations(self, ann_file):
        with open(ann_file, "r", encoding="utf-8") as handle:
            return json.load(handle)


class WeightedLegionDataset(CompleteLegionDataset):
    def __getitem__(self, key):
        index, grad_weight, metric_weight = key if isinstance(key, tuple) else (key, 1.0, 1.0)
        return (*super().__getitem__(int(index)), float(grad_weight), float(metric_weight), int(index))


class ExactDistributedSchedule:
    """Equal-rank schedule with zero-weight padding and exact-once real samples."""

    def __init__(self, length: int, rank: int, world: int, grad_accum: int, seed: int, epoch: int):
        rng = random.Random(seed + epoch)
        indices = list(range(length))
        rng.shuffle(indices)
        global_slots = world * grad_accum
        self.local: list[tuple[int, float, float]] = []
        for start in range(0, length, global_slots):
            group = indices[start:start + global_slots]
            real_count = len(group)
            scale = global_slots / real_count
            padded = group + [group[0]] * (global_slots - real_count)
            weights = [scale] * real_count + [0.0] * (global_slots - real_count)
            for micro in range(grad_accum):
                slot = micro * world + rank
                self.local.append((padded[slot], weights[slot], float(weights[slot] > 0)))
        self.real_weight_sum = sum(metric_weight for _, _, metric_weight in self.local)
        self.padding_slots = sum(metric_weight == 0 for _, _, metric_weight in self.local)

    def __iter__(self):
        return iter(self.local)

    def __len__(self):
        return len(self.local)


class ExactValidationSchedule:
    def __init__(self, length: int, rank: int, world: int):
        padded = list(range(length))
        while len(padded) % world:
            padded.append(0)
        self.local = [(index, 1.0 if position < length else 0.0, 1.0 if position < length else 0.0)
                      for position, index in enumerate(padded) if position % world == rank]

    def __iter__(self):
        return iter(self.local)

    def __len__(self):
        return len(self.local)


def weighted_collate(batch, tokenizer, local_rank: int, inference: bool = False):
    samples = [value[:-3] for value in batch]
    result = custom_collate_fn(
        samples, tokenizer=tokenizer, use_mm_start_end=True,
        local_rank=local_rank, inference=inference,
    )
    result["gradient_weights"] = torch.tensor([value[-3] for value in batch], dtype=torch.float32)
    result["metric_weights"] = torch.tensor([value[-2] for value in batch], dtype=torch.float32)
    result["dataset_indices"] = torch.tensor([value[-1] for value in batch], dtype=torch.long)
    return result


def official_args(cli: argparse.Namespace) -> argparse.Namespace:
    annotation_path = Path(cli.dataset_dir) / "train/annotations/train.json"
    train_count = len(json.loads(annotation_path.read_text(encoding="utf-8")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = 8
    steps = math.ceil(train_count / (world * grad_accum))
    values = [
        "--version", cli.base, "--dataset_dir", cli.dataset_dir,
        "--vision_pretrained", cli.vision_pretrained, "--vision-tower", cli.vision_tower,
        "--exp_name", "phase5a3_stage1", "--lora_r", "8", "--lr", "1e-4",
        "--ce_loss_weight", "1.0", "--dice_loss_weight", "0.2", "--bce_loss_weight", "0.4",
        "--pretrained", "--use_segm_data", "--seg_dataset", "Legion",
        "--segm_sample_rates", "1", "--val_dataset", "Legion", "--epochs", "3",
        "--batch_size", "1", "--val_batch_size", "1",
        "--grad_accumulation_steps", str(grad_accum), "--epoch_samples", str(train_count),
        "--steps_per_epoch", str(steps), "--workers", str(cli.workers),
        "--log_base_dir", cli.run_dir, "--local_rank", str(cli.local_rank),
        "--print_freq", "25",
    ]
    return official.parse_args(values)


def make_datasets(args, tokenizer):
    common = dict(
        dataset_dir=args.dataset_dir, tokenizer=tokenizer,
        global_image_encoder=args.vision_tower, epoch_samples=args.epoch_samples,
        precision=args.precision, image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
    )
    train = WeightedLegionDataset(**common, validation=False, random_sampling=False)
    val = WeightedLegionDataset(**common, validation=True, random_sampling=False)
    return train, val


def make_loader(dataset, sampler, args, tokenizer, inference=False):
    return DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=args.workers,
        pin_memory=False, collate_fn=lambda batch: weighted_collate(
            batch, tokenizer, args.local_rank, inference=inference
        ),
    )


def prepare_batch(batch: dict) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    gradient_weights = batch.pop("gradient_weights")
    metric_weights = batch.pop("metric_weights")
    indices = batch.pop("dataset_indices")
    batch = dict_to_cuda(batch)
    gradient_weights = gradient_weights.cuda(non_blocking=True)
    metric_weights = metric_weights.cuda(non_blocking=True)
    indices = indices.cuda(non_blocking=True)
    for key in ("global_enc_images", "grounding_enc_images"):
        if batch[key] is not None:
            batch[key] = batch[key].bfloat16()
    return batch, gradient_weights, metric_weights, indices


def reduce_sums(sums: dict[str, float], count: float) -> dict[str, float]:
    values = torch.tensor([sums[key] for key in LOSS_KEYS] + [count], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(values)
    total_count = values[-1].item()
    return {key: values[idx].item() / total_count for idx, key in enumerate(LOSS_KEYS)}


def validate(engine, loader) -> dict[str, float]:
    engine.train()  # official loss path requires training mode; gradients remain disabled
    sums = {key: 0.0 for key in LOSS_KEYS}
    count = 0.0
    with torch.no_grad():
        for packed in loader:
            batch, _, metric_weights, _ = prepare_batch(packed)
            output = engine(**batch)
            weight = float(metric_weights.item())
            if weight:
                for key in LOSS_KEYS:
                    sums[key] += float(output[key].item())
                count += 1.0
    return reduce_sums(sums, count)


def train_epoch(engine, loader, args, epoch: int) -> dict[str, float]:
    engine.train()
    sums = {key: 0.0 for key in LOSS_KEYS}
    count = 0.0
    start = time.time()
    for micro_step, packed in enumerate(loader):
        batch, gradient_weights, metric_weights, _ = prepare_batch(packed)
        output = engine(**batch)
        grad_weight = gradient_weights[0]
        metric_weight = metric_weights[0]
        for key in LOSS_KEYS:
            sums[key] += float(output[key].item()) * float(metric_weight.item())
        count += float(metric_weight.item())
        engine.backward(output["loss"] * grad_weight)
        engine.step()
        optimizer_step = (micro_step + 1) // args.grad_accumulation_steps
        if args.local_rank == 0 and (micro_step + 1) % (25 * args.grad_accumulation_steps) == 0:
            elapsed = time.time() - start
            print(json.dumps({
                "event": "train_progress", "epoch": epoch + 1,
                "optimizer_step": optimizer_step, "steps_per_epoch": args.steps_per_epoch,
                "elapsed_seconds": elapsed, "raw_loss": float(output["loss"].item()),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }), flush=True)
    return reduce_sums(sums, count)


def preflight_samples(dataset) -> list[dict]:
    output = []
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[(index, 1.0, 1.0)]
        masks = sample[5]
        conversation = sample[4][0]
        output.append({
            "index": index, "image_path": sample[0], "conversation_seg_count": conversation.count("[SEG]"),
            "mask_count": int(masks.shape[0]), "mask_shape": list(masks.shape),
            "mask_foreground_pixels": int(masks.sum().item()),
            "gradient_weight": sample[-3], "metric_weight": sample[-2],
        })
    return output


def main() -> None:
    cli = parse_args()
    args = official_args(cli)
    expected_global_batch = int(os.environ.get("WORLD_SIZE", "1")) * args.batch_size * args.grad_accumulation_steps
    if expected_global_batch != 16:
        raise RuntimeError(f"Stage-1 protocol gate failed: effective_global_batch={expected_global_batch}, expected 16")
    seed_all(cli.seed)
    run_dir = Path(cli.run_dir)
    tokenizer = official.setup_tokenizer_and_special_tokens(args)
    model = official.initialize_model(args, tokenizer)
    official.prepare_model_for_training(model, tokenizer, args)
    trainable = [(name, list(param.shape), param.numel()) for name, param in model.named_parameters()
                 if param.requires_grad]
    engine, _, scheduler = official.initialize_deepspeed(model, tokenizer, args)
    train_dataset, val_dataset = make_datasets(args, tokenizer)
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()

    if cli.mode == "preflight":
        # Hugging Face from_pretrained() leaves the parent module in eval mode.
        # The official train() function calls model.train() before its first
        # forward; reproduce that exact state for the one-batch preflight too.
        engine.train()
        schedule = ExactDistributedSchedule(len(train_dataset), rank, world, args.grad_accumulation_steps, cli.seed, 0)
        loader = make_loader(train_dataset, schedule, args, tokenizer)
        torch.cuda.reset_peak_memory_stats()
        packed = next(iter(loader))
        batch, gradient_weights, _, _ = prepare_batch(packed)
        # ZeRO-2 partitions/clears ``param.grad`` during backward, so inspecting
        # parameter.grad afterwards incorrectly reports no gradients.  Hooks
        # observe the tensors before DeepSpeed moves them into partition buffers.
        grad_observation = {"finite": 0, "nonzero": 0}
        handles = []
        def observe_gradient(gradient):
            grad_observation["finite"] += int(torch.isfinite(gradient).all().item())
            grad_observation["nonzero"] += int(torch.count_nonzero(gradient).item() > 0)
        for param in engine.module.parameters():
            if param.requires_grad:
                handles.append(param.register_hook(observe_gradient))
        output = engine(**batch)
        engine.backward(output["loss"] * gradient_weights[0])
        for handle in handles:
            handle.remove()
        finite_grad_tensors = grad_observation["finite"]
        nonzero_grad_tensors = grad_observation["nonzero"]
        record = {
            "status": "PASS" if torch.isfinite(output["loss"]).item() and nonzero_grad_tensors else "FAIL",
            "source_commit": "d21535dd45f6fea509337a83095966f0b86ac924",
            "base": str(Path(cli.base).resolve()), "public_legion_LE_used": False,
            "train_count": len(train_dataset), "validation_count": len(val_dataset),
            "world_size": world, "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "effective_global_batch": world * args.batch_size * args.grad_accumulation_steps,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "total_optimizer_steps": args.epochs * args.steps_per_epoch,
            "learning_rate": args.lr,
            "scheduler": "WarmupDecayLR",
            "warmup_num_steps": 100,
            "initialization_checkpoint_index_sha256": file_sha256(Path(cli.base) / "pytorch_model.bin.index.json"),
            "validation_manifest": str(Path(args.dataset_dir) / "train/annotations/test.json"),
            "validation_manifest_sha256": file_sha256(Path(args.dataset_dir) / "train/annotations/test.json"),
            "trainable_parameter_count": sum(value[2] for value in trainable),
            "trainable_parameters": [{"name": n, "shape": s, "numel": c} for n, s, c in trainable],
            "losses": {key: float(output[key].item()) for key in LOSS_KEYS},
            "finite_grad_tensors": finite_grad_tensors, "nonzero_grad_tensors": nonzero_grad_tensors,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu_name": torch.cuda.get_device_name(),
            "sample_alignment": preflight_samples(train_dataset) if rank == 0 else [],
            "loader_firewall": {
                "train": str(Path(args.dataset_dir) / "train/annotations/train.json"),
                "validation": str(Path(args.dataset_dir) / "train/annotations/test.json"),
                "internal_test": False, "official1000": False, "external_benchmark": False,
            },
        }
        if rank == 0:
            dump(run_dir / "preflight.json", record)
            print(json.dumps({"event": "preflight_complete", "status": record["status"]}), flush=True)
        torch.distributed.barrier()
        if record["status"] != "PASS":
            raise RuntimeError("Stage-1 preflight failed")
        return

    history = []
    checkpoint_root = run_dir / "deepspeed_checkpoints"
    for epoch in range(args.epochs):
        seed_all(cli.seed + epoch)
        schedule = ExactDistributedSchedule(
            len(train_dataset), rank, world, args.grad_accumulation_steps, cli.seed, epoch
        )
        loader = make_loader(train_dataset, schedule, args, tokenizer)
        val_schedule = ExactValidationSchedule(len(val_dataset), rank, world)
        val_loader = make_loader(val_dataset, val_schedule, args, tokenizer)
        torch.cuda.reset_peak_memory_stats()
        train_losses = train_epoch(engine, loader, args, epoch)
        val_losses = validate(engine, val_loader)
        tag = f"epoch{epoch + 1}_global_step{(epoch + 1) * args.steps_per_epoch}"
        client_state = {
            "epoch": epoch + 1, "train_losses": train_losses, "validation_losses": val_losses,
            "train_count": len(train_dataset), "validation_count": len(val_dataset),
            "zero_weight_padding_slots_global": world * sum(s.padding_slots for s in [schedule])
            if world == 1 else (world * args.grad_accumulation_steps - len(train_dataset) % (world * args.grad_accumulation_steps)) % (world * args.grad_accumulation_steps),
        }
        engine.save_checkpoint(str(checkpoint_root), tag=tag, client_state=client_state)
        row = {
            **client_state, "tag": tag, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if rank == 0:
            dump(run_dir / "history.json", history)
            print(json.dumps({"event": "epoch_complete", **row}), flush=True)
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
