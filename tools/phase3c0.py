"""Pure helpers for Phase 3C.0 controlled localization diagnostics."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence


TARGET_FIELD_RE = re.compile(r"Target\s+regions?\s*:\s*$", flags=re.IGNORECASE)


def sha256_ints(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def locate_phrase_token_span(
    generated_token_ids: Sequence[int], tokenizer: Any, seg_token_id: int,
) -> tuple[int, int] | None:
    """Return ``[phrase_start, seg_position)`` without re-tokenizing generated text.

    The end of the frozen ``Target regions:`` prefix is located only at an
    existing token boundary.  Everything before that boundary and the exact
    ``[SEG]`` token/suffix therefore remains byte-for-byte identical.
    """
    ids = [int(value) for value in generated_token_ids]
    seg_positions = [index for index, value in enumerate(ids) if value == int(seg_token_id)]
    if len(seg_positions) != 1:
        return None
    seg_position = seg_positions[0]
    candidates = []
    for boundary in range(1, seg_position + 1):
        decoded = tokenizer.decode(ids[:boundary], skip_special_tokens=False)
        match = TARGET_FIELD_RE.search(decoded)
        if match:
            candidates.append(boundary)
    if not candidates:
        return None
    return min(candidates), seg_position


def _encode_replacement(tokenizer: Any, text: str) -> list[int]:
    normalized = " ".join(str(text).split())
    if not normalized:
        raise ValueError("Replacement phrase must be non-empty")
    values = tokenizer(" " + normalized, add_special_tokens=False).input_ids
    if not values:
        raise ValueError("Replacement phrase tokenized to an empty sequence")
    return [int(value) for value in values]


def repair_phrase_tokens(
    generated_token_ids: Sequence[int], tokenizer: Any, seg_token_id: int,
    authoritative_phrase: str,
) -> dict[str, Any]:
    ids = [int(value) for value in generated_token_ids]
    span = locate_phrase_token_span(ids, tokenizer, seg_token_id)
    if span is None:
        raise ValueError("Generated sequence lacks one parseable Target regions span")
    phrase_start, seg_position = span
    replacement = _encode_replacement(tokenizer, authoritative_phrase)
    repaired = ids[:phrase_start] + replacement + ids[seg_position:]
    repaired_seg = phrase_start + len(replacement)
    return {
        "generated_token_ids": repaired,
        "phrase_start": phrase_start,
        "original_seg_position": seg_position,
        "repaired_seg_position": repaired_seg,
        "original_phrase_token_ids": ids[phrase_start:seg_position],
        "replacement_phrase_token_ids": replacement,
        "prefix_exact": repaired[:phrase_start] == ids[:phrase_start],
        "seg_suffix_exact": repaired[repaired_seg:] == ids[seg_position:],
        "prefix_sha256": sha256_ints(ids[:phrase_start]),
        "seg_suffix_sha256": sha256_ints(ids[seg_position:]),
    }


def insert_phrase_tokens(
    generated_token_ids: Sequence[int], tokenizer: Any, seg_token_id: int,
    authoritative_phrase: str,
) -> dict[str, Any]:
    ids = [int(value) for value in generated_token_ids]
    positions = [index for index, value in enumerate(ids) if value == int(seg_token_id)]
    if len(positions) != 1:
        raise ValueError("Phrase insertion requires exactly one [SEG]")
    seg_position = positions[0]
    insertion = _encode_replacement(tokenizer, f"Target regions: {authoritative_phrase}")
    updated = ids[:seg_position] + insertion + ids[seg_position:]
    updated_seg = seg_position + len(insertion)
    return {
        "generated_token_ids": updated,
        "insertion_token_ids": insertion,
        "original_seg_position": seg_position,
        "inserted_seg_position": updated_seg,
        "prefix_exact": updated[:seg_position] == ids[:seg_position],
        "seg_suffix_exact": updated[updated_seg:] == ids[seg_position:],
    }


def classify_failure(a_iou: float, d_iou: float, threshold: float = 0.30) -> str:
    if a_iou > threshold:
        return "G0_GOOD"
    if d_iou >= 0.70:
        return "LANGUAGE_RECOVERABLE"
    if d_iou <= threshold:
        return "PERSISTENT"
    return "INTERMEDIATE"

