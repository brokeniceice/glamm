#!/usr/bin/env python3
"""Phase 3C.2: P1+B3(P3) and official1000 corruption robustness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import _localization_record
from eval.forensics_eval import GLaMMForensicsBackend
from model.forensic_fusion import ResidualForensicFusion
from model.llava import conversation as conversation_lib
from npr_expert.official_npr_srm import OfficialNPRSRM
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase2c_forensic_fusion import (
    balanced_batches,
    binary_metrics,
    two_class_probability,
)
from scripts.phase3a_evaluate import preserve_spatial_prediction, summarize_localization
from tools.phase3c2 import CONDITIONS, apply_classification_gate, corrupt_rgb, mcnemar_exact


CONFIG_PATH = ROOT / "configs/phase3c2_p3.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


CFG = config()
OUT = (ROOT / CFG["experiment"]["output_root"]).resolve()
CKPT = Path(CFG["fusion"]["checkpoint_root"]).resolve()
CACHE = OUT / "cache"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", default="cuda:0")
    cache = sub.add_parser("cache")
    cache.add_argument("--split", choices=("train", "val", "test", "official1000"), required=True)
    cache.add_argument("--start", type=int, required=True)
    cache.add_argument("--end", type=int, required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--batch-size", type=int, default=8)
    combine = sub.add_parser("combine")
    combine.add_argument("--split", choices=("train", "val", "test", "official1000"), required=True)
    train = sub.add_parser("train")
    train.add_argument("--device", default="cuda:0")
    original = sub.add_parser("evaluate-original")
    original.add_argument("--device", default="cuda:0")
    robust = sub.add_parser("robust-eval")
    robust.add_argument("--model", choices=("c0", "p1"), required=True)
    robust.add_argument("--condition", choices=CONDITIONS, required=True)
    robust.add_argument("--device", default="cuda:0")
    robust.add_argument("--max-samples", type=int, default=None)
    sub.add_parser("analyze")
    sub.add_parser("finalize")
    return parser.parse_args(argv)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_for(split: str) -> tuple[Path, str]:
    if split == "official1000":
        return (ROOT / CFG["data"]["official1000_manifest_dir"]).resolve(), "test"
    return (ROOT / CFG["data"]["manifest_dir"]).resolve(), split


def checkpoint_spec(name: str) -> dict:
    return CFG["base_checkpoint"] if name == "p1" else CFG["control_checkpoint"]


def model_config(spec: dict) -> dict:
    return yaml.safe_load((ROOT / spec["model_config"]).read_text(encoding="utf-8"))


def load_glamm(name: str, device: torch.device):
    spec = checkpoint_spec(name)
    path = (ROOT / spec["path"]).resolve()
    if file_sha256(path) != spec["sha256"]:
        raise AssertionError(f"{name} checkpoint hash mismatch")
    cfg = model_config(spec)
    return (*load_model(
        cfg, path, device,
        expected_step=int(spec["optimizer_step"]), expected_epoch=int(spec["epoch"]),
    ), cfg)


def load_expert(device: torch.device):
    path = (ROOT / CFG["expert"]["checkpoint"]).resolve()
    if file_sha256(path) != CFG["expert"]["sha256"]:
        raise AssertionError("NPR+SRM checkpoint hash mismatch")
    expert = OfficialNPRSRM.from_checkpoint(path, freeze=True).to(device).eval()
    return expert, path


def dataset_for(tokenizer, cfg: dict, split: str) -> UnifiedForensicsDataset:
    manifest_dir, dataset_split = manifest_for(split)
    return UnifiedForensicsDataset(
        manifest_dir, tokenizer, cfg["model"]["vision_tower"], split=dataset_split,
        datasets_root=(ROOT / CFG["data"]["datasets_root"]).resolve(),
        synthscars_root=(ROOT / CFG["data"]["synthscars_root"]).resolve(),
        image_size=int(cfg["model"]["image_size"]),
        target_protocol=cfg["forensics"].get("target_protocol", "historical"),
    )


def cache_shard_path(split: str, start: int, end: int) -> Path:
    return CACHE / split / "shards" / f"{start:06d}_{end:06d}.pt"


def combined_cache_path(split: str) -> Path:
    return CACHE / split / "combined.pt"


def cache_range(split: str, start: int, end: int, device_name: str, batch_size: int) -> None:
    destination = cache_shard_path(split, start, end)
    if destination.exists():
        print(f"cache shard exists: {destination}", flush=True)
        return
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, checkpoint_meta, cfg = load_glamm("p1", device)
    expert, expert_path = load_expert(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16,
        use_mm_start_end=True, max_new_tokens=1,
    )
    dataset = dataset_for(tokenizer, cfg, split)
    end = min(int(end), len(dataset))
    if not 0 <= start < end <= len(dataset):
        raise ValueError(f"invalid range [{start},{end}) for {split} n={len(dataset)}")
    transform = build_npr_transform(load_size=256, crop_size=224, training=False)
    storage = {key: [] for key in (
        "base_logits", "lm_logits", "h_cls", "npr", "srm",
        "expert_npr_logits", "expert_srm_logits", "expert_joint_logits",
    )}
    sample_ids, labels, sources, contents, image_paths = [], [], [], [], []
    captured = []
    hook = model.classification_head.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    try:
        for offset in range(start, end, batch_size):
            samples = [dataset[index] for index in range(offset, min(offset + batch_size, end))]
            batch = backend._batch_many(samples, "")
            batch["grounding_enc_images"] = None
            captured.clear()
            with torch.no_grad():
                output = model.model_forward(**batch)
            if len(captured) != 1 or captured[0].shape[0] != len(samples):
                raise RuntimeError(f"classification hook mismatch: {[tuple(x.shape) for x in captured]}")
            expert_inputs = []
            for sample in samples:
                with Image.open(sample["image_path"]) as image:
                    expert_inputs.append(transform(image.convert("RGB")))
            expert_inputs = torch.stack(expert_inputs).to(device)
            with torch.no_grad():
                branches = expert.extract_branch_features(expert_inputs)
                branch_logits = {
                    "npr": expert.fc1(branches["npr"]),
                    "srm": expert.fc1(branches["srm_gated"]),
                    "joint": expert.fc1(branches["npr"] + branches["srm_gated"]),
                }
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
            print(f"cache {split} {min(offset + batch_size, end)}/{end}", flush=True)
    finally:
        hook.remove()
    payload = {
        "metadata": {
            "schema": "phase3c2_p1_frozen_features_v1", "split": split,
            "start_index": start, "end_index": end, "samples": end - start,
            "P1_checkpoint": checkpoint_meta, "P1_checkpoint_sha256": CFG["base_checkpoint"]["sha256"],
            "expert_checkpoint": str(expert_path), "expert_checkpoint_sha256": CFG["expert"]["sha256"],
            "manifest": str((manifest_for(split)[0] / f"{manifest_for(split)[1]}_combined.jsonl").resolve()),
            "config_sha256": file_sha256(CONFIG_PATH),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "sample_ids": sample_ids, "labels": torch.tensor(labels, dtype=torch.long),
        "sources": sources, "content_categories": contents, "image_paths": image_paths,
        **{key: torch.cat(values) for key, values in storage.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    write_json(destination.with_suffix(".json"), payload["metadata"])
    print(f"saved {destination}", flush=True)


def combine_cache(split: str) -> None:
    destination = combined_cache_path(split)
    if destination.exists():
        print(f"combined cache exists: {destination}", flush=True)
        return
    shard_paths = sorted((CACHE / split / "shards").glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no cache shards for {split}")
    shards = [torch.load(path, map_location="cpu") for path in shard_paths]
    expected = 0
    for shard in shards:
        if int(shard["metadata"]["start_index"]) != expected:
            raise AssertionError(f"non-contiguous {split} cache at {expected}")
        expected = int(shard["metadata"]["end_index"])
    tensor_keys = (
        "labels", "base_logits", "lm_logits", "h_cls", "npr", "srm",
        "expert_npr_logits", "expert_srm_logits", "expert_joint_logits",
    )
    list_keys = ("sample_ids", "sources", "content_categories", "image_paths")
    payload = {
        "metadata": {
            **shards[0]["metadata"], "start_index": 0, "end_index": expected,
            "samples": sum(len(shard["labels"]) for shard in shards),
            "combined_from": [str(path) for path in shard_paths],
        },
        **{key: torch.cat([shard[key] for shard in shards]) for key in tensor_keys},
        **{key: sum((shard[key] for shard in shards), []) for key in list_keys},
    }
    manifest_dir, dataset_split = manifest_for(split)
    expected_count = sum(1 for line in (manifest_dir / f"{dataset_split}_combined.jsonl").open() if line.strip())
    if len(payload["labels"]) != expected_count or len(set(payload["sample_ids"])) != expected_count:
        raise AssertionError(f"{split} cache count/uniqueness mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    write_json(destination.with_suffix(".json"), payload["metadata"])
    print(f"combined {split}: {expected_count}", flush=True)


@torch.no_grad()
def score_p3(model, cache: dict, device: torch.device):
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


def load_p3(device: torch.device):
    checkpoint = torch.load(CKPT / "best.pt", map_location="cpu")
    model = ResidualForensicFusion(
        "npr_srm", semantic_dim=int(checkpoint["semantic_dim"]),
        d_fuse=int(checkpoint["d_fuse"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def train_p3(device_name: str) -> None:
    device = torch.device(device_name)
    train_cache = torch.load(combined_cache_path("train"), map_location="cpu")
    val_cache = torch.load(combined_cache_path("val"), map_location="cpu")
    semantic_dim = int(train_cache["h_cls"].shape[1])
    seed = int(CFG["experiment"]["seed"])
    torch.manual_seed(seed)
    model = ResidualForensicFusion(
        "npr_srm", semantic_dim=semantic_dim, d_fuse=int(CFG["fusion"]["d_fuse"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(CFG["training"]["learning_rate"]),
        weight_decay=float(CFG["training"]["weight_decay"]),
    )
    zero_logits, _, _, zero_metrics = score_p3(model, val_cache, device)
    zero = {
        "max_abs_logit_diff": float((zero_logits - val_cache["base_logits"].float()).abs().max()),
        "prediction_equal": bool(torch.equal(zero_logits.argmax(1), val_cache["base_logits"].argmax(1))),
        "metrics": zero_metrics,
    }
    if zero["max_abs_logit_diff"] != 0.0 or not zero["prediction_equal"]:
        raise AssertionError(f"zero-init parity failed: {zero}")
    write_json(OUT / "audit/zero_init_parity.json", zero)
    write_json(OUT / "audit/parameter_audit.json", {
        **model.trainable_parameter_audit(), "P1_trainable": 0, "expert_trainable": 0,
        "optimizer_parameter_ids_equal_fusion_ids": {
            id(p) for group in optimizer.param_groups for p in group["params"]
        } == {id(p) for p in model.parameters() if p.requires_grad},
    })
    CKPT.mkdir(parents=True, exist_ok=True)
    max_steps = int(CFG["training"]["max_optimizer_steps"])
    interval = int(CFG["training"]["validation_interval"])
    batch_size = int(CFG["training"]["effective_global_batch"])
    best_loss, best_step, history = math.inf, None, []
    for step, indices in enumerate(balanced_batches(train_cache["labels"], max_steps, batch_size, seed), 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(
            train_cache["base_logits"][indices].to(device), train_cache["h_cls"][indices].to(device),
            npr=train_cache["npr"][indices].to(device), srm=train_cache["srm"][indices].to(device),
        )
        loss = F.cross_entropy(output.logits, train_cache["labels"][indices].to(device))
        loss.backward(); optimizer.step()
        record = {"step": step, "train_cls_loss": float(loss.detach()), "alpha": float(model.alpha.detach())}
        if step % interval == 0:
            val_logits, _, _, val_metrics = score_p3(model, val_cache, device)
            base_pred, pred, labels = val_cache["base_logits"].argmax(1), val_logits.argmax(1), val_cache["labels"]
            record["validation"] = {
                **val_metrics,
                "wrong_to_correct": int(((base_pred != labels) & (pred == labels)).sum()),
                "correct_to_wrong": int(((base_pred == labels) & (pred != labels)).sum()),
            }
            checkpoint = {
                "model": model.state_dict(), "variant": "npr_srm", "semantic_dim": semantic_dim,
                "d_fuse": int(CFG["fusion"]["d_fuse"]), "step": step,
                "val_cls_loss": val_metrics["cls_loss"], "selector": "min_val_cls_loss",
                "P1_sha256": CFG["base_checkpoint"]["sha256"], "expert_sha256": CFG["expert"]["sha256"],
                "config": CFG,
            }
            torch.save(checkpoint, CKPT / f"step_{step:04d}.pt")
            if val_metrics["cls_loss"] < best_loss:
                best_loss, best_step = val_metrics["cls_loss"], step
                torch.save(checkpoint, CKPT / "best.pt")
        history.append(record)
        if step % 50 == 0:
            print(f"P3 step={step} loss={float(loss):.6f} alpha={float(model.alpha):.6f}", flush=True)
    write_json(OUT / "training/selection.json", {
        "selector": "min_val_cls_loss", "candidate_steps": list(range(interval, max_steps + 1, interval)),
        "selected_step": best_step, "best_val_cls_loss": best_loss,
        "test_or_official1000_used": False, "checkpoint": str(CKPT / "best.pt"),
        "checkpoint_sha256": file_sha256(CKPT / "best.pt"),
    })
    metrics_path = OUT / "training/metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")


def classification_result(cache: dict, model, device: torch.device) -> tuple[dict, list[dict]]:
    logits, deltas, probabilities, p3_metrics = score_p3(model, cache, device)
    labels = cache["labels"].numpy()
    p1_prob = two_class_probability(cache["base_logits"])
    p1_pred = cache["base_logits"].argmax(1).numpy()
    p3_pred = logits.argmax(1).numpy()
    result = {
        "P1": binary_metrics(labels, p1_prob), "P3": p3_metrics,
        "P3_minus_P1": {
            key: p3_metrics[key] - binary_metrics(labels, p1_prob)[key]
            for key in ("accuracy", "precision", "recall", "f1", "brier", "ece_15")
        },
        "paired_mcnemar": mcnemar_exact(p1_pred, p3_pred, labels),
    }
    rows = [{
        "sample_id": sample_id, "gt": int(labels[i]),
        "P1_prob_fake": float(p1_prob[i]), "P1_pred": int(p1_pred[i]),
        "P3_prob_fake": float(probabilities[i]), "P3_pred": int(p3_pred[i]),
        "scaled_delta_norm": float(deltas[i].norm()),
    } for i, sample_id in enumerate(cache["sample_ids"])]
    return result, rows


def original_g0_source(split: str) -> Path:
    return ROOT / "outputs/phase3a_phrase_grounding/evaluation" / (
        "internal" if split == "test" else "official1000"
    ) / "G0/predictions.jsonl"


def g0_comparison(rows: list[dict], predictions: list[dict]) -> dict:
    pmap = {row["sample_id"]: row for row in predictions}
    source_ids = [row["sample_id"] for row in rows]
    if not set(source_ids).issubset(pmap):
        raise AssertionError("classification/G0 sample set mismatch")
    p1_gated = [apply_classification_gate(row, pmap[row["sample_id"]]["P1_pred"] == 1) for row in rows]
    p3_gated = [apply_classification_gate(row, pmap[row["sample_id"]]["P3_pred"] == 1) for row in rows]
    return {
        "canonical_P1": summarize_localization(rows, autoregressive=True),
        "canonical_P3": summarize_localization(rows, autoregressive=True),
        "canonical_exact_by_design": True,
        "classification_gated_P1": summarize_localization(p1_gated, autoregressive=True),
        "classification_gated_P3": summarize_localization(p3_gated, autoregressive=True),
    }


def evaluate_original(device_name: str) -> None:
    device = torch.device(device_name)
    p3, checkpoint = load_p3(device)
    for split, label in (("test", "internal_test"), ("official1000", "official1000")):
        cache = torch.load(combined_cache_path(split), map_location="cpu")
        cls, predictions = classification_result(cache, p3, device)
        root = OUT / "evaluation" / label
        write_json(root / "classification_metrics.json", cls)
        predictions_path = root / "classification_predictions.jsonl"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
        g0_rows = read_jsonl(original_g0_source(split))
        write_json(root / "g0_metrics.json", g0_comparison(g0_rows, predictions))
    write_json(OUT / "evaluation/original_provenance.json", {
        "P3_checkpoint": str(CKPT / "best.pt"), "P3_checkpoint_sha256": file_sha256(CKPT / "best.pt"),
        "selected_step": int(checkpoint["step"]),
        "canonical_G0_note": "P3 is classification-only; canonical P3 G0 is exactly P1 G0",
    })


def corrupted_sample(dataset: UnifiedForensicsDataset, index: int, condition: str) -> tuple[dict, np.ndarray, str]:
    sample = dataset[index]
    image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(sample["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    changed = corrupt_rgb(
        image, condition, seed=int(CFG["experiment"]["seed"]), sample_id=sample["sample_id"],
    )
    sample["global_enc_image"] = dataset.global_enc_processor.preprocess(
        changed, return_tensors="pt"
    )["pixel_values"][0]
    resized = dataset.transform.apply_image(changed)
    sample["resize"] = resized.shape[:2]
    sample["grounding_enc_image"] = dataset.grounding_enc_processor(
        torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    )
    return sample, changed, hashlib.sha256(changed.tobytes()).hexdigest()


def robust_eval(model_name: str, condition: str, device_name: str, max_samples: int | None) -> None:
    device = torch.device(device_name)
    model, tokenizer, checkpoint_meta, cfg = load_glamm(model_name, device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = dataset_for(tokenizer, cfg, "official1000")
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    expert = transform = p3 = None
    if model_name == "p1":
        expert, _ = load_expert(device)
        transform = build_npr_transform(load_size=256, crop_size=224, training=False)
        p3, _ = load_p3(device)
    root = OUT / "robustness" / condition / model_name
    cls_path, g0_path = root / "classification.jsonl", root / "G0.jsonl"
    cls_done = {row["sample_id"] for row in read_jsonl(cls_path)}
    g0_done = {row["sample_id"] for row in read_jsonl(g0_path)}
    captured = []
    hook = model.classification_head.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    try:
        for index in range(limit):
            sample_id = dataset.rows[index]["sample_id"]
            if sample_id in cls_done and sample_id in g0_done:
                continue
            sample, image, image_hash = corrupted_sample(dataset, index, condition)
            captured.clear()
            batch = backend._batch(sample, "")
            batch["grounding_enc_images"] = None
            with torch.no_grad():
                output = model.model_forward(**batch)
            base_logits = output["cls_logits"].detach().float()
            base_prob = float(base_logits.softmax(-1)[0, 1])
            base_pred = int(base_logits.argmax(-1)[0])
            cls_row = {
                "sample_id": sample_id, "gt": int(sample["cls_label"]), "condition": condition,
                "corrupted_rgb_sha256": image_hash,
                f"{model_name}_prob_fake": base_prob, f"{model_name}_pred": base_pred,
            }
            if model_name == "p1":
                if len(captured) != 1:
                    raise RuntimeError("P1 classification hidden-state hook mismatch")
                expert_input = transform(Image.fromarray(image, mode="RGB"))[None].to(device)
                with torch.no_grad():
                    branches = expert.extract_branch_features(expert_input)
                    fused = p3(
                        base_logits, captured[0], npr=branches["npr"], srm=branches["srm_gated"],
                    )
                cls_row.update({
                    "P3_prob_fake": float(fused.logits.softmax(-1)[0, 1]),
                    "P3_pred": int(fused.logits.argmax(-1)[0]),
                })
            if sample_id not in g0_done:
                generated = backend.generate_localization_batch(
                    [sample], provide_gt_fake=False, generation_mode="unified_fake_generate"
                )[0]
                record = _localization_record(
                    sample, generated, "unified_fake_generate", uses_gt_authenticity=False,
                    uses_gt_explanation=False, classification_gate=False,
                )
                record = preserve_spatial_prediction(
                    root, "G0", sample, generated, record, save_spatial=False,
                )
                record["condition"] = condition
                record["corrupted_rgb_sha256"] = image_hash
                append_jsonl(g0_path, record)
                g0_done.add(sample_id)
            if sample_id not in cls_done:
                append_jsonl(cls_path, cls_row)
                cls_done.add(sample_id)
            if (index + 1) % 10 == 0:
                print(f"robust {condition} {model_name}: {index + 1}/{limit}", flush=True)
    finally:
        hook.remove()
    write_json(root / "provenance.json", {
        "condition": condition, "model": model_name, "samples": limit,
        "checkpoint": checkpoint_meta, "config_sha256": file_sha256(CONFIG_PATH),
        "perturbation": CFG["robustness"]["conditions"][condition],
    })


def model_condition_metrics(model_name: str, condition: str) -> dict:
    root = OUT / "robustness" / condition / ("p1" if model_name in {"p1", "p3"} else "c0")
    cls_rows = read_jsonl(root / "classification.jsonl")
    g0_rows = read_jsonl(root / "G0.jsonl")
    if len(cls_rows) != 1000 or len(g0_rows) != 1000:
        raise AssertionError(f"incomplete robustness {condition}/{model_name}: {len(cls_rows)}/{len(g0_rows)}")
    labels = np.asarray([row["gt"] for row in cls_rows])
    if model_name == "p3":
        probabilities = np.asarray([row["P3_prob_fake"] for row in cls_rows])
        predictions = np.asarray([row["P3_pred"] for row in cls_rows])
    else:
        probabilities = np.asarray([row[f"{model_name}_prob_fake"] for row in cls_rows])
        predictions = np.asarray([row[f"{model_name}_pred"] for row in cls_rows])
    prediction_map = {row["sample_id"]: bool(predictions[i]) for i, row in enumerate(cls_rows)}
    gated = [apply_classification_gate(row, prediction_map[row["sample_id"]]) for row in g0_rows]
    return {
        "classification": binary_metrics(labels, probabilities),
        "canonical_G0": summarize_localization(g0_rows, autoregressive=True),
        "classification_gated_G0": summarize_localization(gated, autoregressive=True),
    }


def analyze() -> dict:
    result = {"protocol": CFG["robustness"], "conditions": {}}
    for condition in CONDITIONS:
        result["conditions"][condition] = {
            model: model_condition_metrics(model, condition) for model in ("c0", "p1", "p3")
        }
        result["conditions"][condition]["p3"]["canonical_G0_shared_with_P1"] = True
    original = result["conditions"]["original"]
    for condition in CONDITIONS:
        for model in ("c0", "p1", "p3"):
            current = result["conditions"][condition][model]
            base = original[model]
            current["delta_vs_original"] = {
                "classification_accuracy": current["classification"]["accuracy"] - base["classification"]["accuracy"],
                "canonical_G0_fg_iou": current["canonical_G0"]["per_image_mean"]["foreground_iou"] - base["canonical_G0"]["per_image_mean"]["foreground_iou"],
                "gated_G0_fg_iou": current["classification_gated_G0"]["per_image_mean"]["foreground_iou"] - base["classification_gated_G0"]["per_image_mean"]["foreground_iou"],
            }
    write_json(OUT / "robustness/summary.json", result)
    return result


def fmt(value: float) -> str:
    return f"{value:.6f}"


def generate_report(summary: dict) -> str:
    internal = json.loads((OUT / "evaluation/internal_test/classification_metrics.json").read_text())
    official = json.loads((OUT / "evaluation/official1000/classification_metrics.json").read_text())
    original_g0 = json.loads((OUT / "evaluation/official1000/g0_metrics.json").read_text())
    selection = json.loads((OUT / "training/selection.json").read_text())
    rows = []
    for condition in CONDITIONS:
        for model in ("c0", "p1", "p3"):
            value = summary["conditions"][condition][model]
            rows.append(
                f"| {condition} | {model.upper()} | {fmt(value['classification']['accuracy'])} | "
                f"{fmt(value['canonical_G0']['per_image_mean']['foreground_iou'])} | "
                f"{fmt(value['classification_gated_G0']['per_image_mean']['foreground_iou'])} | "
                f"{value['delta_vs_original']['classification_accuracy']:+.6f} | "
                f"{value['delta_vs_original']['canonical_G0_fg_iou']:+.6f} |"
            )
    gate_p1 = original_g0["classification_gated_P1"]["per_image_mean"]["foreground_iou"]
    gate_p3 = original_g0["classification_gated_P3"]["per_image_mean"]["foreground_iou"]
    report = f"""# Phase 3C.2 — P3（P1 + NPR/SRM B3）与官方1000鲁棒性

## 1. 协议

P3 在冻结 Phase 3A P1（step 3500 / epoch 7，SHA256 `{CFG['base_checkpoint']['sha256']}`）上复用 Phase 2C B3 的 classification-only residual late-fusion 结构。NPR/SRM checkpoint 冻结，P1、LLM、LoRA、CLIP、SAM、NPR、SRM 均未训练；只训练 P3 fusion head。selector 仅使用 internal validation classification loss，internal test 和 official1000 不参与选模。

P3 selected step={selection['selected_step']}，checkpoint SHA256=`{selection['checkpoint_sha256']}`。canonical G0 不经过 classification gate，因此 P3 与 P1 逐样本共用完全相同的生成和掩码；另行报告 classification-gated G0 作为系统级指标。

## 2. Internal test 分类

P1 Accuracy={internal['P1']['accuracy']:.6f}，P3={internal['P3']['accuracy']:.6f}，P3−P1={internal['P3_minus_P1']['accuracy']:+.6f}；F1 变化={internal['P3_minus_P1']['f1']:+.6f}。paired McNemar：base-correct/P3-wrong={internal['paired_mcnemar']['base_correct_candidate_wrong']}，base-wrong/P3-correct={internal['paired_mcnemar']['base_wrong_candidate_correct']}，net={internal['paired_mcnemar']['net_corrected']:+d}，exact p={internal['paired_mcnemar']['exact_two_sided_p']:.6g}。

## 3. Official1000 原始条件

P1 fake recall/accuracy={official['P1']['accuracy']:.6f}，P3={official['P3']['accuracy']:.6f}，变化={official['P3_minus_P1']['accuracy']:+.6f}。canonical G0 mean FG IoU（P1=P3）={original_g0['canonical_P1']['per_image_mean']['foreground_iou']:.6f}；classification-gated G0：P1={gate_p1:.6f}，P3={gate_p3:.6f}，变化={gate_p3-gate_p1:+.6f}。

## 4. Official1000 五条件鲁棒性

JPEG70/80 表示 JPEG quality factor；Gaussian5/10 表示 RGB 0–255 像素尺度的确定性高斯噪声 σ=5/10（即约 5/255、10/255）。扰动在所有模型预处理之前施加，GT mask geometry 不变。

| 条件 | 模型 | 分类准确率/假召回 | canonical G0 FG IoU | gated G0 FG IoU | Δ分类 vs 原始 | Δcanonical IoU vs 原始 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 5. 解释边界

- P3 的 canonical G0 与 P1 相等是架构约束，不是 P3 获得了定位增益。
- classification-gated G0 的变化只来自分类决策翻转。
- official1000 为 Fake-only，分类数值是 fake recall/accuracy，不能单独衡量 FPR 或完整 balanced accuracy。
- 本阶段只检验当前 P1 上的 B3 对应方案，不重新解释 Phase 2C 的 B0 结果，也不证明 NPR/SRM 具有像素级定位能力。

完整 artifact：`outputs/phase3c2_p3_robustness/`；P3 checkpoint 实体：`{CKPT}`。
"""
    return report


def finalize() -> None:
    summary_path = OUT / "robustness/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else analyze()
    report = generate_report(summary)
    docs = ROOT / "docs/phase3c2_p3_robustness.md"
    docs.write_text(report, encoding="utf-8")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports/phase3c2_p3_robustness.md").write_text(report, encoding="utf-8")
    required = [
        CONFIG_PATH, CKPT / "best.pt", OUT / "training/selection.json",
        OUT / "evaluation/internal_test/classification_metrics.json",
        OUT / "evaluation/official1000/classification_metrics.json",
        OUT / "robustness/summary.json", docs,
    ]
    manifest = {
        "status": "COMPLETE", "phase": "Phase 3C.2", "report": str(docs),
        "P1_primary_unchanged": True, "P3_classification_only": True,
        "P3_checkpoint": str(CKPT / "best.pt"),
        "files": [{"path": str(path), "sha256": file_sha256(path)} for path in required],
    }
    write_json(OUT / "completion_manifest.json", manifest)


def preflight(device_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {
        "P1_checkpoint_hash": file_sha256((ROOT / CFG["base_checkpoint"]["path"]).resolve()),
        "C0_checkpoint_hash": file_sha256((ROOT / CFG["control_checkpoint"]["path"]).resolve()),
        "expert_checkpoint_hash": file_sha256((ROOT / CFG["expert"]["checkpoint"]).resolve()),
        "physical_gpu": int(CFG["runtime"]["physical_gpu"]),
        "logical_device": device_name,
        "conditions": list(CONDITIONS),
        "output_root": str(OUT), "checkpoint_root": str(CKPT),
    }
    checks["passed"] = (
        checks["P1_checkpoint_hash"] == CFG["base_checkpoint"]["sha256"]
        and checks["C0_checkpoint_hash"] == CFG["control_checkpoint"]["sha256"]
        and checks["expert_checkpoint_hash"] == CFG["expert"]["sha256"]
    )
    write_json(OUT / "audit/preflight.json", checks)
    if not checks["passed"]:
        raise AssertionError(checks)
    print(json.dumps(checks, indent=2), flush=True)


def main(argv=None):
    cli = parse_args(argv)
    if cli.command == "preflight": preflight(cli.device)
    elif cli.command == "cache": cache_range(cli.split, cli.start, cli.end, cli.device, cli.batch_size)
    elif cli.command == "combine": combine_cache(cli.split)
    elif cli.command == "train": train_p3(cli.device)
    elif cli.command == "evaluate-original": evaluate_original(cli.device)
    elif cli.command == "robust-eval": robust_eval(cli.model, cli.condition, cli.device, cli.max_samples)
    elif cli.command == "analyze": analyze()
    elif cli.command == "finalize": finalize()


if __name__ == "__main__":
    main()
