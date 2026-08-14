"""Numerically guarded representation and mask comparisons."""

from __future__ import annotations

import torch


def tensor_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    left, right = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Tensor shapes differ: {left.shape} != {right.shape}")
    delta = left - right
    return {
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "l2_distance": float(delta.norm()),
        "relative_l2_to_second": float(delta.norm() / right.norm().clamp_min(1e-12)),
        "norm_ratio_first_over_second": float(left.norm() / right.norm().clamp_min(1e-12)),
        "max_abs_difference": float(delta.abs().max()) if delta.numel() else 0.0,
    }


def binary_mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    left, right = a.bool(), b.bool()
    intersection = (left & right).sum()
    union = (left | right).sum()
    return float(intersection / union) if union else 1.0


def recovery(iou_intervention: float, iou_g0: float, iou_tf: float, *, guard: float = .01) -> dict:
    gain = float(iou_intervention - iou_g0)
    denominator = float(iou_tf - iou_g0)
    return {"recovery_from_G0": gain,
            "fraction_of_TF_gap_recovered": gain / denominator if denominator > guard else None,
            "ratio_guard": guard}
