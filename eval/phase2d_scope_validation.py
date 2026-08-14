"""Dataset identity checks used by the Phase 2D artifact join."""

from __future__ import annotations

from typing import Mapping, Sequence


def index_unique(rows: Sequence[Mapping[str, object]], *, name: str) -> dict[str, Mapping[str, object]]:
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"{name} contains duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def assert_exact_identity(reference: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]],
                          *, reference_name: str, candidate_name: str) -> None:
    left = index_unique(reference, name=reference_name)
    right = index_unique(candidate, name=candidate_name)
    if set(left) != set(right):
        raise ValueError(
            f"Dataset identity mismatch: {reference_name} n={len(left)}, {candidate_name} n={len(right)}, "
            f"only_reference={len(set(left)-set(right))}, only_candidate={len(set(right)-set(left))}"
        )


def reject_proxy_category(scope: Mapping[str, object]) -> None:
    if scope.get("label_provenance") != "authoritative":
        raise ValueError("Proxy category labels cannot be used as official categories")
