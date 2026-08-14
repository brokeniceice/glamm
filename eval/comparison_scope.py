"""Safety helpers for metric comparisons with explicit scientific scope."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


class ComparisonScopeError(ValueError):
    """Raised when two reported values do not have compatible scopes."""


@dataclass(frozen=True)
class ComparisonScope:
    dataset: str
    split: str
    category: str
    n: int
    target_representation: str
    metric_family: str
    aggregation: str
    threshold: str
    label_provenance: str = "authoritative"

    def normalized(self) -> dict:
        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, str):
                value[key] = " ".join(item.strip().lower().split())
        return value


_COMPARABILITY_FIELDS = (
    "dataset",
    "split",
    "category",
    "target_representation",
    "metric_family",
    "aggregation",
    "threshold",
)


def assert_comparable_scope(a: ComparisonScope, b: ComparisonScope) -> None:
    """Reject a numerical delta unless all comparison-defining fields match."""

    left, right = a.normalized(), b.normalized()
    mismatches = {
        field: {"left": left[field], "right": right[field]}
        for field in _COMPARABILITY_FIELDS
        if left[field] != right[field]
    }
    if mismatches:
        detail = ", ".join(
            f"{field}={values['left']!r} vs {values['right']!r}"
            for field, values in mismatches.items()
        )
        raise ComparisonScopeError(f"Incompatible comparison scope: {detail}")
    if left["label_provenance"] != "authoritative" or right["label_provenance"] != "authoritative":
        raise ComparisonScopeError("Proxy content labels cannot establish official category parity")


def scoped_delta(
    left_value: float,
    right_value: float,
    left_scope: ComparisonScope,
    right_scope: ComparisonScope,
) -> float:
    assert_comparable_scope(left_scope, right_scope)
    return float(left_value) - float(right_value)


def derived_count_weighted_diagnostic(
    counts: Mapping[str, int], values: Mapping[str, float]
) -> dict:
    """Compute the explicitly non-official, count-weighted category diagnostic."""

    if set(counts) != set(values):
        raise ValueError("Counts and metric values must have identical category keys")
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("Category count total must be positive")
    weighted = sum(int(counts[key]) * float(values[key]) for key in counts) / total
    return {
        "value_percent": weighted,
        "total_count": total,
        "derived": True,
        "assumption": "category metrics are linearly additive by image count",
        "paper_reported_overall": False,
    }


def validate_authoritative_mapping(
    rows: Iterable[Mapping[str, object]], expected_counts: Mapping[str, int]
) -> dict:
    """Validate a future official mapping without inferring any missing labels."""

    rows = list(rows)
    image_ids = [str(row["image_id"]) for row in rows]
    hashes = [str(row["image_sha256"]) for row in rows]
    counts = {category: 0 for category in expected_counts}
    for row in rows:
        if row.get("mapping_method") != "authoritative_metadata":
            raise ComparisonScopeError("Proxy labels cannot be marked as official")
        category = str(row["content_type"]).lower()
        if category not in counts:
            raise ValueError(f"Unexpected content category: {category}")
        counts[category] += 1
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Authoritative mapping contains duplicate image IDs")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Authoritative mapping contains duplicate image SHA256 values")
    if counts != dict(expected_counts):
        raise ValueError(f"Category counts mismatch: {counts} != {dict(expected_counts)}")
    return {"unique_images": len(rows), "unique_sha256": len(hashes), "category_counts": counts}

