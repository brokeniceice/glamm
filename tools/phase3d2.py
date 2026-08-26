"""Pure contracts for Phase 3D.2 direct spatial-path optimization."""

from __future__ import annotations

import hashlib
from typing import Iterable


TRAINABLE_GROUPS = ("text_hidden_fcs", "mask_decoder")


def parameter_group(name: str) -> str:
    if "text_hidden_fcs" in name:
        return "text_hidden_fcs"
    if "grounding_encoder.mask_decoder" in name:
        return "mask_decoder"
    if "lora_" in name:
        return "lora"
    if "embed_tokens" in name:
        return "token_embedding"
    if "lm_head" in name:
        return "lm_head"
    if "classification_head" in name:
        return "classification_head"
    if "vision_tower" in name:
        return "vision_tower"
    if "mm_projector" in name:
        return "mm_projector"
    if "region_encoder" in name:
        return "region_encoder"
    if "grounding_encoder.image_encoder" in name:
        return "grounding_image_encoder"
    if "grounding_encoder" in name:
        return "other_grounding_encoder"
    if ".model.layers." in name or ".model.norm." in name:
        return "llm_base"
    return "all_other"


def is_spatial_trainable(name: str) -> bool:
    return parameter_group(name) in TRAINABLE_GROUPS


def stable_fake_schedule(rows: list[dict], seed: int, exposures: int) -> list[int]:
    values = [index for index, row in enumerate(rows) if int(row["class_label"]) == 1]
    if not values:
        raise ValueError("training manifest has no Fake samples")
    values.sort(key=lambda index: hashlib.sha256(
        f"{seed}:phase3d2:{rows[index]['sample_id']}".encode("utf-8")
    ).hexdigest())
    if exposures > len(values):
        raise ValueError("preregistered exposures exceed unique Fake training samples")
    return values[:exposures]


def validate_optimizer_groups(groups: Iterable[dict]) -> None:
    names = [str(group.get("name")) for group in groups]
    if names != list(TRAINABLE_GROUPS):
        raise ValueError(f"optimizer groups must be exactly {TRAINABLE_GROUPS}, got {names}")
