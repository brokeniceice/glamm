"""Pure, testable contracts for Phase 3B fixed-policy replay."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch

IGNORE_INDEX = -100


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_phrase(value: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def phrase_overlap(reference: str | None, prediction: str | None) -> dict:
    ref, pred = Counter(normalize_phrase(reference)), Counter(normalize_phrase(prediction))
    common = sum((ref & pred).values())
    precision = common / sum(pred.values()) if pred else 0.0
    recall = common / sum(ref.values()) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "normalized_exact_match": bool(ref and ref == pred),
        "normalized_token_precision": precision,
        "normalized_token_recall": recall,
        "normalized_token_f1": f1,
    }


def replay_eligibility(generated_ids: list[int], seg_token_id: int) -> dict:
    positions = [i for i, token in enumerate(generated_ids) if int(token) == int(seg_token_id)]
    return {
        "seg_count": len(positions),
        "seg_positions": positions,
        "usable_seg_predictor_position": positions[0] - 1 if len(positions) == 1 and positions[0] > 0 else None,
        "replay_eligible": len(positions) == 1 and positions[0] > 0,
        "eligibility_rule": "exactly_one_seg_with_preceding_predictor_token",
    }


def deterministic_replay_selection(sample_ids: list[str], *, seed: int, optimizer_step: int,
                                   fraction: float = 0.5) -> list[str]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ranked = sorted(
        sample_ids,
        key=lambda sid: hashlib.sha256(f"{seed}:{optimizer_step}:{sid}".encode()).hexdigest(),
    )
    count = int(len(ranked) * fraction + 0.5)
    return ranked[:count]


def build_mask_only_batch(batch: dict, selected_sample_ids: list[str], *, tokenizer,
                          replay_records: dict[str, dict] | None) -> dict:
    """Select Fake rows and build B0 gold or B1 exact-token mask-only input."""
    positions = {sid: index for index, sid in enumerate(batch["sample_ids"])}
    indices = [positions[sid] for sid in selected_sample_ids]
    if not indices:
        raise ValueError("mask-only batch cannot be empty")
    sequences = []
    for sid, index in zip(selected_sample_ids, indices):
        if int(batch["cls_labels"][index]) != 1 or not bool(batch["seg_valid"][index]):
            raise ValueError(f"Replay selected non-Fake sample: {sid}")
        if replay_records is None:
            sequence = batch["input_ids"][index][batch["attention_masks"][index].bool()].detach().cpu()
        else:
            record = replay_records[sid]
            if not record["replay_eligible"]:
                raise ValueError(f"Replay selected ineligible sample: {sid}")
            sequence = torch.tensor(record["full_input_token_ids"], dtype=torch.long)
        sequences.append(sequence)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    attention = input_ids.ne(tokenizer.pad_token_id)
    offsets = torch.arange(len(indices) + 1, dtype=torch.long)
    def select_list(key):
        return [batch[key][i] for i in indices]
    return {
        "image_paths": select_list("image_paths"),
        "global_enc_images": batch["global_enc_images"][indices],
        "grounding_enc_images": batch["grounding_enc_images"][indices],
        "bboxes": None,
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention,
        "masks_list": select_list("masks_list"),
        "label_list": select_list("label_list"),
        "resize_list": select_list("resize_list"),
        "offset": offsets,
        "questions_list": select_list("questions_list"),
        "sampled_classes_list": select_list("sampled_classes_list"),
        "cls_labels": None,
        "seg_valid": torch.ones(len(indices), dtype=torch.bool),
        "inference": False,
        "sample_ids": list(selected_sample_ids),
        "sources": select_list("sources"),
        "content_categories": select_list("content_categories"),
        "token_strategy": "fixed_cls_query",
        "replay_mask_only": True,
    }
