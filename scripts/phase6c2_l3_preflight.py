#!/usr/bin/env python3
"""Freeze and audit the Phase 6C.2 L3 initialization contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c2_multiseg_training/l3_preflight"
L2 = Path("/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2/best/checkpoint/mp_rank_00_model_states.pt")
L1 = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"
L2_SHA = "dbd7daa8322fe77c8bbfd80223a98ec1e6a4a2c64b4de3b09b09e592ef71d6dd"
L1_SHA = "9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def group_hash(state: dict[str, torch.Tensor], predicate) -> dict:
    selected = {key: value for key, value in state.items() if predicate(key)}
    return {
        "tensor_count": len(selected),
        "element_count": sum(value.numel() for value in selected.values()),
        "sha256": tensor_state_sha256(selected),
    }


def main() -> None:
    if sha256(L2) != L2_SHA or sha256(L1) != L1_SHA:
        raise RuntimeError("L2/L1 selected checkpoint provenance drift")
    l2_payload = torch.load(L2, map_location="cpu", weights_only=False)
    l1_payload = torch.load(L1, map_location="cpu", weights_only=False)
    if (l2_payload.get("epoch"), l2_payload.get("optimizer_step")) != (3, 1500):
        raise RuntimeError("selected L2 must be epoch 3 / optimizer step 1500")
    if set(l1_payload) != {
        "schema", "arm", "epoch", "optimizer_updates", "utility_state",
        "rectifier_state", "optimizer", "validation_g0", "config_sha256",
    }:
        raise RuntimeError("L1 checkpoint key surface drift")
    if l1_payload.get("arm") != "r1" or l1_payload.get("epoch") != 9:
        raise RuntimeError("selected L1 provenance drift")

    base = l2_payload["module"]
    base_before = tensor_state_sha256(base)
    forbidden = {
        "llm_and_lora": group_hash(base, lambda k: "model.layers." in k),
        "seg_projection": group_hash(base, lambda k: "text_hidden_fcs" in k),
        "classifier": group_hash(base, lambda k: "classification_head" in k),
        "grounding_and_mask_decoder": group_hash(base, lambda k: "grounding_encoder" in k),
        "vision_and_backbone": group_hash(base, lambda k: "vision_tower" in k or "mm_projector" in k),
        "embeddings_and_lm_head": group_hash(base, lambda k: "embed_tokens" in k or "lm_head" in k),
    }

    # L3 stores the two R1-specific modules beside an immutable L2 base
    # reference.  No L1 tensor is loaded into the L2 module namespace.
    imported = {
        "utility_state": {key: value.clone() for key, value in l1_payload["utility_state"].items()},
        "rectifier_state": {key: value.clone() for key, value in l1_payload["rectifier_state"].items()},
    }
    base_after = tensor_state_sha256(base)
    if base_before != base_after:
        raise RuntimeError("forbidden L2 base overwrite")

    trainable_utility = {
        key: value for key, value in imported["utility_state"].items()
        if not key.startswith("language_source.") and not key.startswith("forensic_source.")
        and key not in {"temperature_l", "temperature_f"}
    }
    if sum(value.numel() for value in trainable_utility.values()) != 371803:
        raise RuntimeError("L1 utility trainable scope drift")
    result = {
        "schema": "phase6c2_l3_initialization_audit_v1",
        "status": "PASS",
        "L3_base": {
            "kind": "selected_L2_module",
            "path": str(L2), "sha256": L2_SHA, "epoch": 3, "optimizer_step": 1500,
            "module_tensor_count": len(base), "module_state_sha256_before": base_before,
            "module_state_sha256_after": base_after,
        },
        "L1_import": {
            "path": str(L1), "sha256": L1_SHA, "selected_epoch": 9,
            "allowed_top_level_keys": ["utility_state", "rectifier_state"],
            "utility": group_hash(imported["utility_state"], lambda _: True),
            "rectifier": group_hash(imported["rectifier_state"], lambda _: True),
            "utility_keys": sorted(imported["utility_state"]),
            "rectifier_keys": sorted(imported["rectifier_state"]),
        },
        "forbidden_overwrite_audit": {
            "status": "PASS", "L1_keys_loaded_into_L2_module": [],
            "L2_module_exact_before_after": base_before == base_after,
            "groups": forbidden,
        },
        "trainable_parameter_manifest": {
            "utility_non_source": group_hash(trainable_utility, lambda _: True),
            "rectifier": group_hash(imported["rectifier_state"], lambda _: True),
            "frozen": [
                "selected L2 LLM/LoRA/SEG projection/classifier/backbone/SAM mask decoder",
                "L1 utility language_source and forensic_source heads",
            ],
        },
        "runtime_contract": {
            "T": "sum(K)", "SEG_states": "flatten in annotation order",
            "image_features": "gather/repeat by image index", "R1_parameters": "one shared copy",
            "regroup": "exclusive-prefix offsets", "loss": "global per-mask mean",
            "optimizer": "AdamW", "lr": 0.0001, "weight_decay": 0.0001,
            "epochs": 10, "scheduler": "none", "selector": "internal validation Fake canonical G0 mean IoU; tie earlier",
        },
        "firewall": {"internal_test_accessed": False, "external_accessed": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "initialization_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "l2_exact": result["forbidden_overwrite_audit"]["L2_module_exact_before_after"],
        "utility_tensors": result["L1_import"]["utility"]["tensor_count"],
        "rectifier_tensors": result["L1_import"]["rectifier"]["tensor_count"],
    }))


if __name__ == "__main__":
    main()
