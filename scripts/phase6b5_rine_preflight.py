#!/usr/bin/env python3
"""Phase 6B.5 implementation audit and one-batch RINE-on-C1 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RINE_REPO = ROOT / "external/RINE_official"
RINE_COMMIT = "9b7fd5857cc205d0412be6aeee0d7611b95bd620"
CLIP_PATH = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
C1_EXACT = ROOT / "outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554"
TRAIN_MANIFEST = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/train.json"
VAL_MANIFEST = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/val.json"
CACHED_C1 = ROOT / "outputs/phase6b2_fusion/features/train/c1_exact_logits.pt"
OUT = ROOT / "outputs/phase6b5_rine_preflight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dump(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def load_rows(path: Path, expected: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise RuntimeError(f"Frozen manifest count/uniqueness drift: {path}")
    if any(int(row["label"]) not in (0, 1) for row in rows):
        raise RuntimeError(f"Invalid class label in {path}")
    return rows


def balanced_rows(rows: list[dict], per_class: int) -> list[dict]:
    # Stage-2 manifest semantics are Real=1, Fake=0.  Alternation ensures each
    # class has positive pairs for the official supervised contrastive loss.
    by_label = {
        label: [row for row in rows if int(row["label"]) == label][:per_class]
        for label in (0, 1)
    }
    if any(len(values) != per_class for values in by_label.values()):
        raise RuntimeError("Could not form deterministic balanced preflight batch")
    return [value for pair in zip(by_label[0], by_label[1]) for value in pair]


def preprocess(rows: list[dict], processor: CLIPImageProcessor) -> torch.Tensor:
    images = []
    for row in rows:
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to decode internal TRAIN sample: {row['image_path']}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return processor(images=images, return_tensors="pt")["pixel_values"]


def load_c1_head() -> tuple[torch.nn.Module, str]:
    index_path = C1_EXACT / "pytorch_model.bin.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shards = {weight_map[name] for name in weight_map if name.startswith("prediction_head.")}
    if len(shards) != 1:
        raise RuntimeError(f"Unexpected C1 prediction-head shards: {sorted(shards)}")
    state = torch.load(C1_EXACT / next(iter(shards)), map_location="cpu")
    head_state = {
        name.removeprefix("prediction_head."): tensor.clone()
        for name, tensor in state.items()
        if name.startswith("prediction_head.")
    }
    del state
    expected = {"0.weight", "0.bias", "2.weight", "2.bias"}
    if set(head_state) != expected:
        raise RuntimeError(f"C1 head keys changed: {sorted(head_state)}")
    head = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 2)
    )
    head.load_state_dict(head_state, strict=True)
    return head, tensor_sha256(head_state.items())


def source_snapshot() -> dict:
    paths = [
        ROOT / "model/GLaMM.py",
        ROOT / "model/SAM/build_sam.py",
        ROOT / "model/sam_forensic_rectifier.py",
        ROOT / "external/LEGION_official/model/Legion.py",
        C1_EXACT / "pytorch_model.bin.index.json",
    ]
    return {
        str(path.resolve()): {
            "sha256": sha256(path), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    }


def main() -> None:
    args = parse_args()
    if args.batch_per_class < 2:
        raise ValueError("SupCon preflight requires at least two samples per class")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from model.rine_on_c1 import (
        RINE_OFFICIAL_COMMIT,
        RINE_PROJ_DIM,
        RINE_Q,
        RINE_XI,
        RINEOnHFCLIP,
        official_rine_loss,
    )

    if RINE_OFFICIAL_COMMIT != RINE_COMMIT:
        raise RuntimeError("Implementation/source commit mismatch")
    actual_commit = os.popen(f"git -C {RINE_REPO} rev-parse HEAD").read().strip()
    official_dirty = os.popen(f"git -C {RINE_REPO} status --porcelain").read().strip()
    if actual_commit != RINE_COMMIT or official_dirty:
        raise RuntimeError(f"RINE repo identity/cleanliness failure: {actual_commit}, {official_dirty!r}")

    before = source_snapshot()
    train_rows = load_rows(TRAIN_MANIFEST, 17672)
    val_rows = load_rows(VAL_MANIFEST, 2212)
    sample_rows = balanced_rows(train_rows, args.batch_per_class)
    accessed_ids = [row["sample_id"] for row in sample_rows]

    protocol = {
        "schema": "phase6b5_rine_preflight_protocol_v1",
        "status": "FROZEN_BEFORE_FORWARD",
        "official_rine": {
            "repo": str(RINE_REPO.resolve()), "commit": actual_commit,
            "source_model": "src/models.py", "source_loss": "src/utils.py::SupConLoss",
            "released_configuration_basis": "official best 4-class ProGAN configuration",
        },
        "architecture": {
            "backbone": "frozen openai/clip-vit-large-patch14-336 via current HF tower",
            "blocks": 24, "input_dim": 1024, "q_nproj": RINE_Q,
            "projection_dim": RINE_PROJ_DIM, "tie_softmax_dim": 1,
            "dropout": 0.5, "binary_logit_count": 1,
        },
        "loss": {"bce": "BCEWithLogitsLoss(reduction=sum)", "supcon": True,
                 "supcon_temperature": 0.07, "xi": RINE_XI},
        "labels": {"manifest": {"real": 1, "fake": 0},
                   "rine_loss": {"real": 0, "fake": 1},
                   "conversion": "rine_fake_label = 1 - manifest_label"},
        "manifests": {
            "train": {"path": str(TRAIN_MANIFEST.resolve()), "n": len(train_rows),
                      "sha256": sha256(TRAIN_MANIFEST)},
            "validation": {"path": str(VAL_MANIFEST.resolve()), "n": len(val_rows),
                           "sha256": sha256(VAL_MANIFEST)},
        },
        "firewall": {"internal_train": True, "internal_validation_manifest_audit": True,
                     "internal_test": False, "OOD": False, "AIDE": False,
                     "LLM_fusion": False, "formal_training": False},
    }
    dump("protocol.json", protocol)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    vision = CLIPVisionModel.from_pretrained(CLIP_PATH, local_files_only=True)
    vision = vision.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    model = RINEOnHFCLIP(vision).to(device)
    model.train()

    named_trainable = [(name, parameter) for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    illegal = [name for name, _ in named_trainable if not name.startswith("rine.")]
    trainable_count = sum(parameter.numel() for _, parameter in named_trainable)
    expected_count = 6_323_201
    if illegal or trainable_count != expected_count:
        raise RuntimeError(
            f"Trainable firewall failed: illegal={illegal}, count={trainable_count}"
        )
    trainable_manifest = {
        "status": "PASS", "allowed_prefixes": ["rine.alpha", "rine.proj1",
        "rine.proj2", "rine.head"], "forbidden_trainable": illegal,
        "trainable_parameter_count": trainable_count,
        "expected_official_q2_d1024_24block_count": expected_count,
        "parameters": [
            {"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
             "dtype": str(parameter.dtype)}
            for name, parameter in named_trainable
        ],
        "frozen_parameter_count": sum(parameter.numel() for parameter in vision.parameters()),
        "clip_all_frozen": not any(parameter.requires_grad for parameter in vision.parameters()),
    }
    dump("trainable_parameters.json", trainable_manifest)

    pixels = preprocess(sample_rows, processor).to(device=device, dtype=torch.bfloat16)
    fake_labels = torch.tensor(
        [1 - int(row["label"]) for row in sample_rows], device=device, dtype=torch.long
    )
    torch.cuda.reset_peak_memory_stats(device)
    logits, embedding, block_cls = model(pixels)
    captured = list(model._captured)

    # Recreate official OpenAI-CLIP sequence-first stacking exactly.  Equality
    # proves the adapter only changes layout, not the selected token/value.
    official_layout = torch.stack([value.transpose(0, 1) for value in captured], dim=2)[0]
    layout_max_abs = float((block_cls - official_layout).abs().max().item())
    if layout_max_abs != 0.0:
        raise RuntimeError(f"HF-to-official layout bridge changed values: {layout_max_abs}")

    losses = official_rine_loss(logits, embedding, fake_labels)
    if not all(bool(torch.isfinite(value).item()) for value in losses.values()):
        raise RuntimeError("RINE loss is non-finite")
    losses["total"].backward()
    grad_rows = []
    for name, parameter in named_trainable:
        gradient = parameter.grad
        grad_rows.append({
            "name": name,
            "present": gradient is not None,
            "finite": bool(torch.isfinite(gradient).all().item()) if gradient is not None else False,
            "nonzero": int(torch.count_nonzero(gradient).item()) if gradient is not None else 0,
            "norm": float(gradient.float().norm().item()) if gradient is not None else None,
        })
    if not all(row["present"] and row["finite"] for row in grad_rows):
        raise RuntimeError("Missing or non-finite RINE trainable gradient")
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("Frozen CLIP received gradients")

    block_audit = {
        "status": "PASS", "block_count": len(captured),
        "hook_module_names": model.hook_names,
        "per_layer_full_shapes": [list(value.shape) for value in captured],
        "per_layer_cls_shapes": [list(value[:, 0, :].shape) for value in captured],
        "stacked_cls_shape": list(block_cls.shape),
        "token_count": int(captured[0].shape[1]), "hidden_dim": int(captured[0].shape[2]),
        "official_openai_layout_shape": list(official_layout.shape),
        "hf_vs_official_layout_max_abs_difference": layout_max_abs,
        "semantic_mapping": (
            "official hooks every OpenAI CLIP visual module named ln_2; HF CLIPEncoderLayer "
            "uses layer_norm2 at the same pre-MLP residual-block position; sequence-first "
            "official [S,B,D] is transposed from HF [B,S,D], and CLS remains token index 0"
        ),
        "not_used_as_rine_input": "HF output_hidden_states[-2] (the C1-Exact single-layer feature)",
    }
    dump("block_hook_audit.json", block_audit)

    # Independently exercise the unchanged C1-Exact forward contract on the
    # same current CLIP and compare it with the previously frozen TRAIN logits.
    c1_head, c1_head_hash = load_c1_head()
    c1_head = c1_head.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    with torch.inference_mode():
        current_outputs = vision(pixel_values=pixels, output_hidden_states=True)
        c1_official_logits = c1_head(current_outputs.hidden_states[-2][:, 0]).float()
        c1_real_fake = c1_official_logits[:, [1, 0]].cpu()
    cached = torch.load(CACHED_C1, map_location="cpu")
    positions = {sample_id: index for index, sample_id in enumerate(cached["sample_ids"])}
    cached_logits = torch.stack([cached["logits_real_fake"][positions[sid]] for sid in accessed_ids])
    c1_max_abs = float((c1_real_fake - cached_logits).abs().max().item())
    c1_disagreements = int(
        (c1_real_fake.argmax(1) != cached_logits.argmax(1)).sum().item()
    )
    c1_compatibility = {
        "status": "PASS" if c1_disagreements == 0 and torch.isfinite(c1_real_fake).all() else "FAIL",
        "checkpoint": str(C1_EXACT.resolve()), "prediction_head_sha256": c1_head_hash,
        "feature": "hidden_states[-2][:,0]", "logits_shape": list(c1_real_fake.shape),
        "all_logits_finite": bool(torch.isfinite(c1_real_fake).all().item()),
        "comparison": str(CACHED_C1.resolve()),
        "prediction_disagreement_count": c1_disagreements,
        "logit_max_abs_difference_due_to_batch_replay": c1_max_abs,
        "conclusion": "C1-Exact unchanged standalone forward remains executable",
    }
    if c1_compatibility["status"] != "PASS":
        raise RuntimeError("C1-Exact compatibility check failed")
    dump("c1_exact_compatibility.json", c1_compatibility)

    after = source_snapshot()
    localization_unchanged = before == after
    localization_audit = {
        "status": "PASS" if localization_unchanged else "FAIL",
        "source_and_checkpoint_index_snapshot_before": before,
        "source_and_checkpoint_index_snapshot_after": after,
        "exact_snapshot_equal": localization_unchanged,
        "localization_modules_loaded_by_rine_wrapper": [],
        "localization_parameters_in_optimizer": [],
        "llm_parameters_in_optimizer": [],
        "sam_parameters_in_optimizer": [],
        "reason": (
            "RINE-on-C1 owns only the frozen standalone CLIPVisionModel and RINE head; "
            "LLM, R1, SAM and localization modules are neither imported into the model "
            "nor passed to backward/optimizer"
        ),
    }
    if not localization_unchanged:
        raise RuntimeError("Localization source/checkpoint identity changed during preflight")
    dump("localization_invariance.json", localization_audit)

    initialized_state = {
        "schema": "phase6b5_rine_initialized_head_v1", "seed": args.seed,
        "official_rine_commit": RINE_COMMIT,
        "q": RINE_Q, "projection_dim": RINE_PROJ_DIM, "xi": RINE_XI,
        "state_dict": {name: value.detach().cpu() for name, value in model.rine.state_dict().items()},
    }
    torch.save(initialized_state, OUT / "rine_head_initialized_seed3407.pt")

    status = {
        "schema": "phase6b5_rine_preflight_v1",
        "status": "PASS",
        "ready_for_training": True,
        "device": args.device, "gpu": torch.cuda.get_device_name(device),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "batch_size": len(sample_rows), "sample_ids": accessed_ids,
        "labels_fake_positive": fake_labels.cpu().tolist(),
        "outputs": {"logits_shape": list(logits.shape),
                    "embedding_shape": list(embedding.shape),
                    "block_cls_shape": list(block_cls.shape)},
        "loss": {name: float(value.detach().item()) for name, value in losses.items()},
        "gradients": grad_rows,
        "checks": {
            "official_repo_fixed_and_clean": True,
            "block_shapes": True, "hook_semantics": True,
            "trainable_firewall": True, "forward_backward": True,
            "bce_finite": True, "supcon_finite": True,
            "clip_frozen_no_grad": True,
            "localization_state_unchanged": localization_unchanged,
            "c1_exact_still_runs": c1_compatibility["status"] == "PASS",
            "internal_train_only_images": True,
            "internal_validation_manifest_only": True,
            "ood_accessed": False, "formal_training_started": False,
        },
    }
    dump("preflight.json", status)
    dump("status.json", {"status": "COMPLETE", "ready_for_training": True})
    model.close_hooks()
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
