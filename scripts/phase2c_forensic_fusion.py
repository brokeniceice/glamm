#!/usr/bin/env python3
"""Phase 2C frozen-feature cache, expert audit, fusion training and scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.unified import CANONICAL_PROMPT_SHA256, UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from model.forensic_fusion import ResidualForensicFusion
from model.llava import conversation as conversation_lib
from npr_expert.official_npr_srm import OfficialNPRSRM
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import file_sha256, load_model


OUTPUT_ROOT = REPO_ROOT / "outputs/phase2c_forensic_fusion"
CACHE_SCHEMA = "phase2c_frozen_features_v1"
PREPROCESS_VERSION = "NPR:Resize256-CenterCrop224-ToTensor-ImageNetNormalize:v1"
VARIANT_DIRS = {"npr": "B1_npr", "srm": "B2_srm", "npr_srm": "B3_npr_srm"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cache", "combine", "audit", "train", "test", "all"))
    parser.add_argument("--config", default="configs/phase2c_npr_srm.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-device", default="cuda:1")
    parser.add_argument("--head-device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--cache-tag", default=None)
    parser.add_argument("--parts", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_DIRS), default=tuple(VARIANT_DIRS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_path(split):
    return OUTPUT_ROOT / "feature_cache" / f"{split}.pt"


def load_config(path):
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))


def load_expert(config, device):
    checkpoint = (REPO_ROOT / config["forensics"]["expert_checkpoint"]).resolve()
    expert = OfficialNPRSRM.from_checkpoint(checkpoint, freeze=True)
    expert.to(device).eval()
    return expert, checkpoint


def image_tensor(path, transform):
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def cache_metadata(config, split, checkpoint_path, expert_checkpoint, rows):
    manifest = (REPO_ROOT / config["data"]["manifest_dir"] / f"{split}_combined.jsonl").resolve()
    preprocess_hash = sha256_text(PREPROCESS_VERSION)
    return {
        "schema": CACHE_SCHEMA,
        "split": split,
        "samples": len(rows),
        "phase2a_checkpoint": str(checkpoint_path),
        "phase2a_checkpoint_sha256": file_sha256(checkpoint_path),
        "phase2a_step": 2500,
        "phase2a_epoch": 5,
        "canonical_prompt_sha256": CANONICAL_PROMPT_SHA256,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "npr_checkpoint": str(expert_checkpoint),
        "npr_checkpoint_sha256": file_sha256(expert_checkpoint),
        "srm_config_sha256": file_sha256(REPO_ROOT / "npr_expert/official_npr_srm.py"),
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_sha256": preprocess_hash,
        "code_git_commit": git_commit(),
        "image_sha256": [row.get("content_sha256") or row.get("identity_hash", {}).get("value") for row in rows],
    }


def extract_split(config, split, cli):
    destination = cache_path(cli.cache_tag or split)
    if destination.exists() and not cli.force:
        print(f"cache exists: {destination}", flush=True)
        return
    device = torch.device(cli.device)
    expert_device = torch.device(cli.expert_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_path = (REPO_ROOT / config["base_checkpoint"]["path"]).resolve()
    model, tokenizer, checkpoint_meta = load_model(
        config=load_config("configs/phase2a_unified_baseline_full.yaml"),
        checkpoint_path=checkpoint_path,
        device=device,
        expected_step=2500,
        expected_epoch=5,
    )
    expert, expert_checkpoint = load_expert(config, expert_device)
    transform = build_npr_transform(load_size=256, crop_size=224, training=False)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16,
        use_mm_start_end=True, max_new_tokens=1,
    )
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer,
        "openai/clip-vit-large-patch14-336", split=split,
        datasets_root=REPO_ROOT / config["data"]["datasets_root"],
        synthscars_root=REPO_ROOT / config["data"]["synthscars_root"],
        image_size=1024,
    )
    start_index = max(0, int(cli.start_index))
    end_index = len(dataset) if cli.end_index is None else min(len(dataset), int(cli.end_index))
    if cli.max_samples is not None:
        end_index = min(end_index, start_index + int(cli.max_samples))
    if not 0 <= start_index < end_index <= len(dataset):
        raise ValueError(f"invalid cache range [{start_index}, {end_index}) for {len(dataset)} samples")
    limit = end_index - start_index
    rows = dataset.rows[start_index:end_index]
    storage = {key: [] for key in (
        "base_logits", "lm_logits", "h_cls", "npr", "srm", "expert_npr_logits",
        "expert_srm_logits", "expert_joint_logits",
    )}
    sample_ids, labels, sources, contents, image_paths = [], [], [], [], []
    total_glamm_time = total_expert_time = 0.0
    captured = []

    def capture_cls(_module, inputs, _output):
        captured.append(inputs[0].detach())

    hook = model.classification_head.register_forward_hook(capture_cls)
    try:
        for start in range(start_index, end_index, cli.batch_size):
            samples = [dataset[index] for index in range(start, min(start + cli.batch_size, end_index))]
            batch = backend._batch_many(samples, "")
            batch["grounding_enc_images"] = None
            captured.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            with torch.no_grad():
                output = model.model_forward(**batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_glamm_time += time.perf_counter() - before
            if len(captured) != 1 or captured[0].shape[0] != len(samples):
                raise RuntimeError(f"classification hook mismatch: {[tuple(x.shape) for x in captured]}")
            expert_inputs = torch.stack([
                image_tensor(sample["image_path"], transform) for sample in samples
            ]).to(expert_device, non_blocking=True)
            if expert_device.type == "cuda":
                torch.cuda.synchronize(expert_device)
            before = time.perf_counter()
            with torch.no_grad():
                branches = expert.extract_branch_features(expert_inputs)
                branch_logits = {
                    "npr": expert.fc1(branches["npr"]),
                    "srm": expert.fc1(branches["srm_gated"]),
                    "joint": expert.fc1(branches["npr"] + branches["srm_gated"]),
                }
            if expert_device.type == "cuda":
                torch.cuda.synchronize(expert_device)
            total_expert_time += time.perf_counter() - before
            storage["base_logits"].append(output["cls_logits"].detach().float().cpu())
            storage["lm_logits"].append(output["lm_verdict_logits"].detach().float().cpu())
            storage["h_cls"].append(captured[0].float().cpu().half())
            storage["npr"].append(branches["npr"].cpu().half())
            storage["srm"].append(branches["srm_gated"].cpu().half())
            storage["expert_npr_logits"].append(branch_logits["npr"].float().cpu())
            storage["expert_srm_logits"].append(branch_logits["srm"].float().cpu())
            storage["expert_joint_logits"].append(branch_logits["joint"].float().cpu())
            for sample in samples:
                sample_ids.append(sample["sample_id"])
                labels.append(int(sample["cls_label"]))
                sources.append(sample["source"])
                contents.append(sample.get("content_category"))
                image_paths.append(sample["image_path"])
            processed = min(start + cli.batch_size, end_index) - start_index
            print(f"cache {split}: {processed}/{limit} range=[{start_index},{end_index})", flush=True)
    finally:
        hook.remove()
    tensors = {key: torch.cat(value, dim=0) for key, value in storage.items()}
    payload = {
        "metadata": {
            **cache_metadata(config, split, checkpoint_path, expert_checkpoint, rows),
            "start_index": start_index, "end_index": end_index,
            "checkpoint_load": checkpoint_meta,
            "glamm_seconds": total_glamm_time,
            "expert_seconds": total_expert_time,
            "glamm_ms_per_image": 1000.0 * total_glamm_time / limit,
            "expert_ms_per_image": 1000.0 * total_expert_time / limit,
        },
        "sample_ids": sample_ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "sources": sources,
        "content_categories": contents,
        "image_paths": image_paths,
        **tensors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    write_json(destination.with_suffix(".metadata.json"), payload["metadata"])
    del model, expert, dataset, payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def combine_cache_parts(split, part_names):
    parts = [torch.load(cache_path(name), map_location="cpu") for name in part_names]
    expected_start = 0
    for part in parts:
        if int(part["metadata"]["start_index"]) != expected_start:
            raise ValueError(f"non-contiguous cache part at {part['metadata']['start_index']}, expected {expected_start}")
        expected_start = int(part["metadata"]["end_index"])
    tensor_keys = ("labels", "base_logits", "lm_logits", "h_cls", "npr", "srm",
                   "expert_npr_logits", "expert_srm_logits", "expert_joint_logits")
    list_keys = ("sample_ids", "sources", "content_categories", "image_paths")
    payload = {
        "metadata": {
            **parts[0]["metadata"], "samples": sum(len(part["labels"]) for part in parts),
            "start_index": 0, "end_index": expected_start,
            "combined_from": [str(cache_path(name)) for name in part_names],
            "image_sha256": sum((part["metadata"]["image_sha256"] for part in parts), []),
            "glamm_seconds": sum(part["metadata"]["glamm_seconds"] for part in parts),
            "expert_seconds": sum(part["metadata"]["expert_seconds"] for part in parts),
        },
        **{key: torch.cat([part[key] for part in parts]) for key in tensor_keys},
        **{key: sum((part[key] for part in parts), []) for key in list_keys},
    }
    count = len(payload["labels"])
    payload["metadata"]["glamm_ms_per_image"] = 1000 * payload["metadata"]["glamm_seconds"] / count
    payload["metadata"]["expert_ms_per_image"] = 1000 * payload["metadata"]["expert_seconds"] / count
    destination = cache_path(split)
    torch.save(payload, destination)
    write_json(destination.with_suffix(".metadata.json"), payload["metadata"])
    print(f"combined {count} samples into {destination}", flush=True)


def binary_metrics(labels, fake_probabilities, threshold=0.5):
    from sklearn.metrics import (
        average_precision_score, precision_recall_fscore_support, roc_auc_score,
        roc_curve,
    )
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(fake_probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    has_both_classes = len(np.unique(labels)) == 2
    if has_both_classes:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        fpr95 = float(fpr[np.where(tpr >= 0.95)[0][0]]) if np.any(tpr >= 0.95) else None
        roc_auc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    else:
        fpr95 = roc_auc = auprc = None
    bins = np.minimum((probabilities * 15).astype(int), 14)
    ece = 0.0
    for index in range(15):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return {
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision), "recall": float(recall), "fake_recall": float(recall),
        "f1": float(f1), "roc_auc": roc_auc,
        "auprc": auprc, "fpr_at_95_tpr": fpr95,
        "brier": float(np.mean((probabilities - labels) ** 2)), "ece_15": float(ece),
    }


def expert_probability(single_logit):
    return torch.sigmoid(single_logit.reshape(-1).float()).numpy()


def two_class_probability(logits):
    return logits.float().softmax(dim=-1)[:, 1].numpy()


def complementarity_rows(cache):
    labels = cache["labels"].numpy()
    probabilities = {
        "glamm": two_class_probability(cache["base_logits"]),
        "npr": expert_probability(cache["expert_npr_logits"]),
        "srm": expert_probability(cache["expert_srm_logits"]),
        "npr_srm_historical": expert_probability(cache["expert_joint_logits"]),
    }
    predictions = {key: value >= 0.5 for key, value in probabilities.items()}
    errors = {key: pred != labels for key, pred in predictions.items()}
    summary = {"metrics": {key: binary_metrics(labels, value) for key, value in probabilities.items()}}
    pairs = {}
    for expert in ("npr", "srm", "npr_srm_historical"):
        intersection = int(np.logical_and(errors["glamm"], errors[expert]).sum())
        union = int(np.logical_or(errors["glamm"], errors[expert]).sum())
        pairs[expert] = {
            "error_overlap": intersection,
            "error_jaccard": float(intersection / union) if union else 1.0,
            "glamm_wrong_expert_correct": int(np.logical_and(errors["glamm"], ~errors[expert]).sum()),
            "expert_wrong_glamm_correct": int(np.logical_and(errors[expert], ~errors["glamm"]).sum()),
            "fraction_glamm_errors_corrected": float(
                np.logical_and(errors["glamm"], ~errors[expert]).sum() / errors["glamm"].sum()
            ) if errors["glamm"].sum() else 0.0,
        }
    summary["pairs"] = pairs
    rows = []
    for index, sample_id in enumerate(cache["sample_ids"]):
        row = {"sample_id": sample_id, "gt": int(labels[index])}
        for key in probabilities:
            row[f"{key}_prob_fake"] = float(probabilities[key][index])
            row[f"{key}_pred"] = int(predictions[key][index])
        rows.append(row)
    return summary, rows


def run_audit():
    cache = torch.load(cache_path("val"), map_location="cpu")
    summary, rows = complementarity_rows(cache)
    output = OUTPUT_ROOT / "expert_audit"
    expert = OfficialNPRSRM.from_checkpoint(cache["metadata"]["npr_checkpoint"], freeze=True)
    named = dict(expert.named_parameters())
    srm_names = [name for name in named if name.startswith("srm_")]
    shared_names = [name for name in named if name.startswith("fc1.")]
    npr_names = [name for name in named if name not in set(srm_names + shared_names)]
    parameter_counts = {
        "expert_total": sum(parameter.numel() for parameter in named.values()),
        "expert_frozen": sum(parameter.numel() for parameter in named.values() if not parameter.requires_grad),
        "npr_backbone": sum(named[name].numel() for name in npr_names),
        "srm_backend_and_gate": sum(named[name].numel() for name in srm_names),
        "shared_classifier": sum(named[name].numel() for name in shared_names),
        "phase2c_expert_trainable": sum(parameter.numel() for parameter in named.values() if parameter.requires_grad),
    }
    provenance = {
        "npr": {
            "paper": "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection (CVPR 2024)",
            "source_repository": "https://github.com/chuangchuangtan/NPR-DeepfakeDetection",
            "source_commit_audited": "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a",
            "implementation_path": str((REPO_ROOT / "npr_expert/official_npr_srm.py").resolve()),
            "implementation_sha256": file_sha256(REPO_ROOT / "npr_expert/official_npr_srm.py"),
            "algorithm_parity": "nearest down/up residual x-interpolate(x,0.5), ResNet bottleneck stem/layer1/layer2, GAP",
            "input_resolution": [224, 224], "load_resolution": [256, 256],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "output_feature_shape": [512],
        },
        "srm": {
            "source": "project historical NPR enhancement; not part of the official NPR repository",
            "implementation_path": str((REPO_ROOT / "npr_expert/official_npr_srm.py").resolve()),
            "kernel_config_sha256": file_sha256(REPO_ROOT / "npr_expert/official_npr_srm.py"),
            "fixed_filters": True, "num_fixed_5x5_rgb_filters": 9,
            "backend": "Conv9-32, Conv32-128, Conv128-512 with BN/ReLU and GAP",
            "backend_frozen_phase2c": True, "output_feature_shape": [512],
        },
        "historical_checkpoint": cache["metadata"]["npr_checkpoint"],
        "historical_checkpoint_sha256": cache["metadata"]["npr_checkpoint_sha256"],
        "historical_training_dataset": "AIGI-Holmes train/val; external sets were evaluation-only in that historical run",
        "important_ablation_note": "The historical checkpoint has a shared classifier trained on NPR+gated-SRM sum. Standalone branch scores apply that frozen shared head to one contribution; fusion uses pre-head branch features.",
        "preprocess_version": cache["metadata"]["preprocess_version"],
        "preprocess_sha256": cache["metadata"]["preprocess_sha256"],
        "parameter_counts": parameter_counts,
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "validation_metrics_and_complementarity.json", summary)
    write_jsonl(output / "validation_predictions.jsonl", rows)
    write_json(OUTPUT_ROOT / "B0_phase2a/val/metrics.json", summary["metrics"]["glamm"])
    print(json.dumps(summary, indent=2), flush=True)


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache):
        self.cache = cache
    def __len__(self):
        return len(self.cache["labels"])
    def __getitem__(self, index):
        return {key: self.cache[key][index] for key in (
            "base_logits", "lm_logits", "h_cls", "npr", "srm", "labels"
        )}


def balanced_batches(labels, steps, batch_size, seed):
    if batch_size % 2:
        raise ValueError("balanced batch size must be even")
    generator = torch.Generator().manual_seed(seed)
    classes = [torch.where(labels == value)[0] for value in (0, 1)]
    orders = [indices[torch.randperm(len(indices), generator=generator)] for indices in classes]
    offsets = [0, 0]
    half = batch_size // 2
    for _ in range(steps):
        batch = []
        for value in (0, 1):
            if offsets[value] + half > len(orders[value]):
                orders[value] = classes[value][torch.randperm(len(classes[value]), generator=generator)]
                offsets[value] = 0
            batch.append(orders[value][offsets[value]:offsets[value] + half])
            offsets[value] += half
        merged = torch.cat(batch)
        yield merged[torch.randperm(len(merged), generator=generator)]


@torch.no_grad()
def score_model(model, cache, device):
    model.eval()
    logits, deltas = [], []
    for start in range(0, len(cache["labels"]), 512):
        sl = slice(start, start + 512)
        output = model(
            cache["base_logits"][sl].to(device), cache["h_cls"][sl].to(device),
            npr=cache["npr"][sl].to(device), srm=cache["srm"][sl].to(device),
        )
        logits.append(output.logits.cpu())
        deltas.append(output.scaled_delta_logits.cpu())
    logits = torch.cat(logits)
    deltas = torch.cat(deltas)
    probabilities = two_class_probability(logits)
    metrics = binary_metrics(cache["labels"].numpy(), probabilities)
    metrics["cls_loss"] = float(F.cross_entropy(logits, cache["labels"]).item())
    return logits, deltas, probabilities, metrics


def train_variant(config, variant, device):
    train_cache = torch.load(cache_path("train"), map_location="cpu")
    val_cache = torch.load(cache_path("val"), map_location="cpu")
    semantic_dim = int(train_cache["h_cls"].shape[1])
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = ResidualForensicFusion(
        variant, semantic_dim=semantic_dim, d_fuse=int(config["forensics"]["d_fuse"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    root = OUTPUT_ROOT / VARIANT_DIRS[variant]
    (root / "train").mkdir(parents=True, exist_ok=True)
    zero_logits, _, _, zero_metrics = score_model(model, val_cache, device)
    base = val_cache["base_logits"].float()
    zero_diff = float((zero_logits - base).abs().max().item())
    zero_report = {
        "max_abs_logit_diff": zero_diff,
        "prediction_equal": bool(torch.equal(zero_logits.argmax(1), base.argmax(1))),
        "metrics": zero_metrics,
    }
    if zero_diff != 0.0 or not zero_report["prediction_equal"]:
        raise AssertionError(f"alpha=0 baseline parity failed: {zero_report}")
    write_json(root / "zero_init_parity.json", zero_report)
    write_json(root / "parameter_audit.json", {
        **model.trainable_parameter_audit(),
        "phase2a_trainable": 0, "expert_trainable": 0,
        "optimizer_parameter_ids_equal_fusion_ids": {
            id(p) for group in optimizer.param_groups for p in group["params"]
        } == {id(p) for p in model.parameters() if p.requires_grad},
    })
    max_steps = int(config["training"]["max_optimizer_steps"])
    interval = int(config["training"]["validation_interval"])
    batch_size = int(config["training"]["effective_global_batch"])
    history = []
    best_loss = math.inf
    best_step = None
    for step, indices in enumerate(balanced_batches(
        train_cache["labels"], max_steps, batch_size, int(config["experiment"]["seed"])
    ), start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train_cache["base_logits"][indices].to(device),
            train_cache["h_cls"][indices].to(device),
            npr=train_cache["npr"][indices].to(device),
            srm=train_cache["srm"][indices].to(device),
        )
        labels = train_cache["labels"][indices].to(device)
        loss = F.cross_entropy(output.logits, labels)
        loss.backward()
        optimizer.step()
        record = {
            "step": step, "train_cls_loss": float(loss.detach()), "alpha": float(model.alpha.detach()),
            "delta_logit_norm": float(output.delta_logits.detach().norm(dim=1).mean()),
            "baseline_logit_norm": float(train_cache["base_logits"][indices].float().norm(dim=1).mean()),
            "fusion_logit_norm": float(output.logits.detach().norm(dim=1).mean()),
        }
        if step % interval == 0:
            val_logits, _, val_probabilities, val_metrics = score_model(model, val_cache, device)
            base_pred = val_cache["base_logits"].argmax(1)
            fusion_pred = val_logits.argmax(1)
            gt = val_cache["labels"]
            record["validation"] = {
                **val_metrics,
                "wrong_to_correct": int(((base_pred != gt) & (fusion_pred == gt)).sum()),
                "correct_to_wrong": int(((base_pred == gt) & (fusion_pred != gt)).sum()),
                "real_to_fake_flip": int(((base_pred == 0) & (fusion_pred == 1)).sum()),
                "fake_to_real_flip": int(((base_pred == 1) & (fusion_pred == 0)).sum()),
                "net_corrected_errors": int((fusion_pred == gt).sum() - (base_pred == gt).sum()),
            }
            checkpoint = {
                "model": model.state_dict(), "variant": variant, "semantic_dim": semantic_dim,
                "d_fuse": model.d_fuse, "step": step, "val_cls_loss": val_metrics["cls_loss"],
                "selector": "min_val_cls_loss", "config": config,
            }
            torch.save(checkpoint, root / "train" / f"step_{step:04d}.pt")
            if val_metrics["cls_loss"] < best_loss:
                best_loss, best_step = val_metrics["cls_loss"], step
                torch.save(checkpoint, root / "train" / "best.pt")
        history.append(record)
        if step % 50 == 0:
            print(f"{variant} step={step} loss={float(loss):.6f} alpha={float(model.alpha):.6f}", flush=True)
    write_jsonl(root / "train" / "metrics.jsonl", history)
    write_json(root / "train" / "selection.json", {
        "selector": "min_val_cls_loss", "candidate_steps": list(range(interval, max_steps + 1, interval)),
        "best_step": best_step, "best_val_cls_loss": best_loss,
        "test_or_external_used": False,
    })
    best_model, _ = load_fusion(root, device)
    val_logits, val_deltas, val_probabilities, val_metrics = score_model(best_model, val_cache, device)
    base_probabilities = two_class_probability(val_cache["base_logits"])
    base_predictions = val_cache["base_logits"].argmax(1)
    val_predictions = val_logits.argmax(1)
    gt = val_cache["labels"]
    val_metrics.update({
        "selected_step": best_step,
        "absolute_delta": {
            key: val_metrics[key] - binary_metrics(gt.numpy(), base_probabilities)[key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "auprc", "brier", "ece_15")
        },
        "wrong_to_correct": int(((base_predictions != gt) & (val_predictions == gt)).sum()),
        "correct_to_wrong": int(((base_predictions == gt) & (val_predictions != gt)).sum()),
        "net_corrected_errors": int((val_predictions == gt).sum() - (base_predictions == gt).sum()),
    })
    write_json(root / "val/metrics.json", val_metrics)
    write_jsonl(root / "val/predictions.jsonl", [{
        "sample_id": sample_id, "gt": int(gt[index]),
        "base_prob_fake": float(base_probabilities[index]),
        "fusion_prob_fake": float(val_probabilities[index]),
        "base_pred": int(base_predictions[index]), "fusion_pred": int(val_predictions[index]),
        "scaled_delta_norm": float(val_deltas[index].norm()),
    } for index, sample_id in enumerate(val_cache["sample_ids"])])


def load_fusion(root, device):
    checkpoint = torch.load(root / "train/best.pt", map_location="cpu")
    model = ResidualForensicFusion(
        checkpoint["variant"], semantic_dim=checkpoint["semantic_dim"], d_fuse=checkpoint["d_fuse"]
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def test_variant(variant, device):
    root = OUTPUT_ROOT / VARIANT_DIRS[variant]
    model, checkpoint = load_fusion(root, device)
    cache = torch.load(cache_path("test"), map_location="cpu")
    logits, deltas, probabilities, metrics = score_model(model, cache, device)
    base_probabilities = two_class_probability(cache["base_logits"])
    base_metrics = binary_metrics(cache["labels"].numpy(), base_probabilities)
    predictions = logits.argmax(1)
    base_predictions = cache["base_logits"].argmax(1)
    lm_predictions = cache["lm_logits"].argmax(1)
    labels = cache["labels"]
    metrics.update({
        "selected_step": int(checkpoint["step"]),
        "absolute_delta": {key: metrics[key] - base_metrics[key] for key in (
            "accuracy", "precision", "recall", "f1", "roc_auc", "auprc", "brier", "ece_15"
        )},
        "wrong_to_correct": int(((base_predictions != labels) & (predictions == labels)).sum()),
        "correct_to_wrong": int(((base_predictions == labels) & (predictions != labels)).sum()),
        "net_corrected_errors": int((predictions == labels).sum() - (base_predictions == labels).sum()),
        "cls_lm_agreement": float(predictions.eq(lm_predictions).float().mean()),
        "lm_accuracy": float(lm_predictions.eq(labels).float().mean()),
        "base_cls_lm_agreement": float(base_predictions.eq(lm_predictions).float().mean()),
        "base_metrics": base_metrics,
    })
    rows = []
    for index, sample_id in enumerate(cache["sample_ids"]):
        rows.append({
            "sample_id": sample_id, "gt": int(labels[index]), "source": cache["sources"][index],
            "content_category": cache["content_categories"][index],
            "base_prob_fake": float(base_probabilities[index]), "base_pred": int(base_predictions[index]),
            "fusion_prob_fake": float(probabilities[index]), "fusion_pred": int(predictions[index]),
            "lm_pred": int(lm_predictions[index]),
            "scaled_delta_norm": float(deltas[index].norm()),
            "delta_fake_logit": float(deltas[index, 1]),
        })
    write_json(root / "test/metrics.json", metrics)
    write_jsonl(root / "test/predictions.jsonl", rows)
    print(json.dumps({variant: metrics}, indent=2), flush=True)


def main(argv=None):
    cli = parse_args(argv)
    config = load_config(cli.config)
    if cli.command in {"cache", "all"}:
        for split in cli.splits:
            extract_split(config, split, cli)
    if cli.command == "combine":
        if len(cli.splits) != 1 or not cli.parts:
            raise ValueError("combine requires exactly one --splits value and --parts")
        combine_cache_parts(cli.splits[0], cli.parts)
    if cli.command in {"audit", "all"}:
        run_audit()
    head_device = torch.device(cli.head_device)
    if cli.command in {"train", "all"}:
        for variant in cli.variants:
            variant_config = load_config(f"configs/phase2c_{variant}.yaml")
            train_variant(variant_config, variant, head_device)
    if cli.command in {"test", "all"}:
        for variant in cli.variants:
            test_variant(variant, head_device)


if __name__ == "__main__":
    main()
