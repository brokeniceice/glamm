#!/usr/bin/env python3
"""External Stage-2 LEGION classifier runner with frozen Real=0/Fake=1 labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, Trainer, TrainingArguments, default_data_collator, set_seed


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/LEGION_official"
sys.path.insert(0, str(OFFICIAL))
spec = importlib.util.spec_from_file_location(
    "legion_official_cls_train", OFFICIAL / "scripts/cls/train.py"
)
official = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(official)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--stage1-le", required=True)
    parser.add_argument("--vision-pretrained", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
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


class FrozenClsDataset(Dataset):
    def __init__(self, path: str, vision_tower: str):
        self.path = Path(path).resolve()
        self.rows = json.loads(self.path.read_text(encoding="utf-8"))
        self.processor = CLIPImageProcessor.from_pretrained(vision_tower, local_files_only=True)
        labels = [int(row["label"]) for row in self.rows]
        if any(label not in (0, 1) for label in labels):
            raise ValueError("Stage-2 labels must use official LEGION Real=1/Fake=0 semantics")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to decode frozen sample {row['sample_id']}: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return {
            "global_enc_images": pixels, "cls_gt_list": int(row["label"]),
        }


class ControlledTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            global_enc_images=inputs["global_enc_images"],
            cls_gt_list=inputs["cls_gt_list"], train_cls=True,
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = eval_pred.label_ids
    if isinstance(labels, tuple):
        labels = labels[0]
    return {"accuracy": float((np.argmax(predictions, axis=1) == labels).mean())}


def official_args(cli: argparse.Namespace) -> argparse.Namespace:
    return make_official_args(cli)


def make_official_args(cli: argparse.Namespace) -> argparse.Namespace:
    # The official parser reads process argv, so temporarily give it only its supported flags.
    saved = sys.argv
    sys.argv = [saved[0], "--version", cli.stage1_le, "--vision_pretrained", cli.vision_pretrained,
                "--vision_tower", cli.vision_tower, "--pretrained", "--epochs", "3",
                "--lr", "1e-3", "--batch_size", str(cli.batch_size),
                "--val_batch_size", str(cli.batch_size), "--workers", str(cli.workers),
                "--train_json_file", cli.train_json, "--test_json_file", cli.val_json,
                "--save_path", str(Path(cli.run_dir) / "final_model"),
                "--log_base_dir", cli.run_dir, "--exp_name", "phase5a3_stage2"]
    try:
        return official.parse_args()
    finally:
        sys.argv = saved


def load(cli: argparse.Namespace):
    args = official_args(cli)
    tokenizer = official.setup_tokenizer_and_special_tokens(args)
    # The official Stage-2 loader enables low_cpu_mem_usage while changing the
    # class from LegionForCausalLM to LegionForCls.  Parameters absent from the
    # Stage-1 checkpoint (notably the new prediction head, and in this stack the
    # nested CLIP reload) can remain meta tensors and cannot be moved to CUDA.
    # Materialize the exact same official class normally; this changes loading
    # memory only, not weights, initialization, or architecture.
    model_args = {key: getattr(args, key) for key in (
        "train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
        "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
        "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
        "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx",
    )}
    model_args["num_level_reg_features"] = 4
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = official.LegionForCls.from_pretrained(
        args.version, torch_dtype=dtype, low_cpu_mem_usage=False, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    official.prepare_model_for_training(model, tokenizer, args)
    trainable = [(name, list(param.shape), param.numel()) for name, param in model.named_parameters()
                 if param.requires_grad]
    if not trainable or any("prediction_head" not in name for name, _, _ in trainable):
        raise RuntimeError("Stage-2 trainable firewall failed: non-head parameter is trainable")
    return args, tokenizer, model, trainable


def main() -> None:
    cli = parse_args()
    if cli.batch_size != 64:
        raise RuntimeError(f"Stage-2 protocol gate failed: per-device batch={cli.batch_size}, expected 64")
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    set_seed(cli.seed)
    run_dir = Path(cli.run_dir)
    args, tokenizer, model, trainable = load(cli)
    if args.grad_accumulation_steps != 1:
        raise RuntimeError("Stage-2 protocol gate failed: gradient accumulation must be 1")
    train_dataset = FrozenClsDataset(cli.train_json, cli.vision_tower)
    val_dataset = FrozenClsDataset(cli.val_json, cli.vision_tower)
    if len(train_dataset) != 17672 or len(val_dataset) != 2212:
        raise RuntimeError("Stage-2 frozen train/validation count drift")

    if cli.mode == "preflight":
        model.cuda()
        model.train()
        loader = DataLoader(train_dataset, batch_size=cli.batch_size, shuffle=False,
                            num_workers=cli.workers, pin_memory=False)
        batch = next(iter(loader))
        torch.cuda.reset_peak_memory_stats()
        outputs = model(
            global_enc_images=batch["global_enc_images"].cuda().bfloat16(),
            cls_gt_list=batch["cls_gt_list"].cuda(), train_cls=True,
        )
        outputs["loss"].backward()
        finite = torch.isfinite(outputs["loss"]).item()
        grad_nonzero = sum(
            int(param.grad is not None and torch.count_nonzero(param.grad).item() > 0)
            for param in model.parameters() if param.requires_grad
        )
        record = {
            "status": "PASS" if finite and grad_nonzero else "FAIL",
            "initialization": str(Path(cli.stage1_le).resolve()),
            "public_legion_LE_used": False, "batch_size": cli.batch_size,
            "per_device_batch": cli.batch_size, "world_size": 1,
            "gradient_accumulation_steps": 1, "effective_global_batch": cli.batch_size,
            "epochs": 3, "steps_per_epoch": int(np.ceil(len(train_dataset) / cli.batch_size)),
            "total_optimizer_steps": 3 * int(np.ceil(len(train_dataset) / cli.batch_size)),
            "learning_rate": 1e-3, "scheduler": "cosine", "weight_decay": 0.0,
            "loss": float(outputs["loss"].item()), "nonzero_grad_tensors": grad_nonzero,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu_name": torch.cuda.get_device_name(),
            "train_count": len(train_dataset), "validation_count": len(val_dataset),
            "labels": {"real": 1, "fake": 0},
            "initialization_checkpoint_index_sha256": file_sha256(Path(cli.stage1_le) / "pytorch_model.bin.index.json"),
            "validation_manifest": str(Path(cli.val_json).resolve()),
            "validation_manifest_sha256": file_sha256(Path(cli.val_json)),
            "trainable_parameter_count": sum(value[2] for value in trainable),
            "trainable_parameters": [{"name": n, "shape": s, "numel": c} for n, s, c in trainable],
            "loader_firewall": {"internal_test": False, "official1000": False, "external_benchmark": False},
        }
        dump(run_dir / f"preflight_batch{cli.batch_size}.json", record)
        print(json.dumps({"event": "stage2_preflight_complete", **record}), flush=True)
        if record["status"] != "PASS":
            raise RuntimeError("Stage-2 preflight failed")
        return

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), num_train_epochs=3,
        per_device_train_batch_size=cli.batch_size, per_device_eval_batch_size=cli.batch_size,
        gradient_accumulation_steps=1, evaluation_strategy="epoch", save_strategy="epoch",
        logging_strategy="steps", logging_steps=20, learning_rate=1e-3, weight_decay=0.0,
        adam_beta1=0.9, adam_beta2=0.95, bf16=True, fp16=False,
        gradient_checkpointing=True, dataloader_num_workers=cli.workers,
        report_to=[], load_best_model_at_end=True, metric_for_best_model="accuracy",
        greater_is_better=True, remove_unused_columns=False, label_names=["cls_gt_list"],
        lr_scheduler_type="cosine", seed=cli.seed, data_seed=cli.seed,
    )
    trainer = ControlledTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer, data_collator=default_data_collator,
    )
    result = trainer.train()
    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    epoch_metrics = [entry for entry in trainer.state.log_history if "eval_accuracy" in entry]
    summary = {
        "status": "COMPLETE", "initialization": str(Path(cli.stage1_le).resolve()),
        "public_legion_LE_used": False, "train_count": len(train_dataset),
        "validation_count": len(val_dataset), "labels": {"real": 1, "fake": 0},
        "epochs": 3, "lr": 1e-3, "scheduler": "cosine", "batch_size": cli.batch_size,
        "steps_per_epoch": int(np.ceil(len(train_dataset) / cli.batch_size)),
        "trainable_parameter_count": sum(value[2] for value in trainable),
        "trainable_parameters": [{"name": n, "shape": s, "numel": c} for n, s, c in trainable],
        "validation_by_epoch": epoch_metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric, "train_metrics": result.metrics,
        "final_model": str(final_dir),
    }
    dump(run_dir / "training_summary.json", summary)
    print(json.dumps({"event": "stage2_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
