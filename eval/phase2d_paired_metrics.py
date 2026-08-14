"""Phase 2D paired localization metrics with explicit aggregation semantics."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Iterable, Mapping, Sequence


def safe_ratio(numerator: int | float, denominator: int | float, *, empty: float = 1.0) -> float:
    return float(numerator) / float(denominator) if denominator else float(empty)


def per_image_mask_metrics(row: Mapping[str, object], total_pixels: int) -> dict[str, float]:
    """Derive valid per-image metrics from persisted TP/FP/FN counts.

    This function intentionally does not expose any value as a decomposition of
    a global-pixel aggregate.  The returned fg/bg mIoU is the mean of the two
    IoUs *for this image*.
    """
    tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
    tn = int(total_pixels) - tp - fp - fn
    if tn < 0:
        raise ValueError("Mask confusion counts exceed the image pixel count")
    fg_iou = safe_ratio(tp, tp + fp + fn)
    bg_iou = safe_ratio(tn, tn + fp + fn)
    return {
        "foreground_iou": fg_iou,
        "background_iou": bg_iou,
        "per_image_fg_bg_miou": (fg_iou + bg_iou) / 2.0,
        "foreground_f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty distribution")

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": len(ordered), "mean": mean(ordered), "median": median(ordered),
        "std_population": pstdev(ordered), "min": ordered[0], "q1": quantile(.25),
        "q3": quantile(.75), "p90": quantile(.90), "p95": quantile(.95), "max": ordered[-1],
        "positive_proportion": mean(v > 1e-12 for v in ordered),
        "near_zero_proportion_abs_le_0_01": mean(abs(v) <= .01 for v in ordered),
        "negative_proportion": mean(v < -1e-12 for v in ordered),
    }


def aggregate_global(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    rows = list(rows)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn,
        "global_foreground_iou": safe_ratio(tp, tp + fp + fn),
        "global_foreground_f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
    }
