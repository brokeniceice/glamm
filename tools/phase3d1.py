"""Frozen pure contracts for Phase 3D.1 group-relative policy optimization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from tools.phase3d0 import reward_candidates
from tools.phase3d0r import grounding_components, new_rewards, semantic_phrase_score


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_relative_advantages(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    """Population-standardized group advantages; constant groups produce exact zeros."""
    rewards = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError("a rollout group must contain at least two scalar rewards")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    std = float(rewards.std(ddof=0))
    if std <= eps:
        return [0.0] * len(rewards)
    advantages = (rewards - rewards.mean()) / (std + eps)
    # Remove floating residual so every group has an exact zero-sum policy weight.
    advantages -= advantages.mean()
    return [float(value) for value in advantages]


def score_r3(gt_class: int, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Call the frozen Phase-3D.0 historical R3 formula without reformulation."""
    output = []
    for row in rows:
        components = {key: float(row[key]) for key in (
            "R_cls", "R_struct", "R_phrase_lex", "R_mask", "R_align",
        )}
        output.append(float(reward_candidates(int(gt_class), components)["R3"]))
    return output


def score_q2(
    gt_class: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    authoritative_phrase: str | None,
    idf: Mapping[str, float],
    embeddings: Mapping[str, np.ndarray],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Call the frozen Phase-3D.0-R Q2 semantic and relative-grounding formula."""
    fake = int(gt_class) == 1
    grounds = (
        grounding_components([float(row["R_mask"]) for row in rows])
        if fake else [None] * len(rows)
    )
    enriched = []
    for row, ground in zip(rows, grounds):
        value = dict(row)
        if fake:
            value.update(semantic_phrase_score(
                authoritative_phrase, value.get("normalized_phrase"), idf, embeddings,
            ))
            value.update(ground)
        else:
            value.update({
                "R_phrase_sem": 0.0, "R_key_soft": 0.0, "R_sentence_sem": 0.0,
                "P_spatial_contra": 0.0, "P_neg_contra": 0.0,
                "R_ground_rel": 0.0, "C_ground": 0.0, "R_rank_mask": 0.0,
            })
        value.update(new_rewards(int(gt_class), value))
        enriched.append(value)
    return [float(row["Q2"]) for row in enriched], enriched

