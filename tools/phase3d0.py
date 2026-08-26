"""Pure contracts for Phase 3D.0 evidence-aware rollout/reward preflight."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.phase3b_replay import phrase_overlap


REWARD_NAMES = ("R0", "R1", "R2", "R3")
COMPONENT_NAMES = ("R_cls", "R_struct", "R_phrase_lex", "R_mask", "R_align")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_rank(seed: int, namespace: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode()).hexdigest()


def _proportional_quotas(groups: Mapping[tuple, list[dict]], total: int) -> dict[tuple, int]:
    population = sum(len(values) for values in groups.values())
    if total > population:
        raise ValueError(f"requested {total} from population {population}")
    exact = {key: total * len(values) / population for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(math.floor(value))) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), repr(key)))
    while remaining:
        changed = False
        for key in order:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise RuntimeError("unable to allocate deterministic stratified quotas")
    return quotas


def deterministic_stratified_subset(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: int, label: int,
) -> list[dict[str, Any]]:
    eligible = [dict(row, dataset_index=index) for index, row in enumerate(rows)
                if int(row["class_label"]) == int(label)]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in eligible:
        key = (str(row.get("content_category")), str(row.get("source")))
        groups[key].append(row)
    quotas = _proportional_quotas(groups, count)
    selected = []
    for key, values in groups.items():
        ranked = sorted(values, key=lambda row: stable_rank(seed, f"label{label}:{key}", row["sample_id"]))
        selected.extend(ranked[:quotas[key]])
    return sorted(selected, key=lambda row: stable_rank(seed, f"final-label{label}", row["sample_id"]))


def normalize_output(text: str | None) -> str:
    return " ".join(str(text or "").lower().split())


def text_tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\W\s]", str(text or "").lower(), flags=re.UNICODE)


def normalized_edit_distance(left: str | None, right: str | None) -> float:
    a, b = text_tokens(left), text_tokens(right)
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        current = [i]
        for j, bv in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (av != bv)))
        previous = current
    return previous[-1] / max(len(a), len(b), 1)


def parse_structure(parsed: Mapping[str, Any], generated_ids: Sequence[int], seg_token_id: int) -> dict[str, Any]:
    positions = [i for i, value in enumerate(generated_ids) if int(value) == int(seg_token_id)]
    usable = len(positions) == 1 and positions[0] > 0
    verdict = parsed.get("verdict")
    target_present = bool(parsed.get("target_field_present"))
    phrase = " ".join(str(parsed.get("target_region") or "").split())
    if verdict == "FAKE":
        valid = usable and target_present and bool(phrase)
    elif verdict == "REAL":
        valid = not usable and not target_present
    else:
        valid = False
    return {
        "verdict_parse_valid": verdict in {"REAL", "FAKE"},
        "seg_count": len(positions), "seg_positions": positions,
        "usable_seg": usable, "usable_predictor_position": positions[0] - 1 if usable else None,
        "target_field_present": target_present, "normalized_phrase": phrase,
        "structural_validity": bool(valid),
        "seg_status": "USABLE" if usable else "SEG_UNAVAILABLE",
    }


def reward_components(
    *, gt_label: int, parsed: Mapping[str, Any], generated_ids: Sequence[int], seg_token_id: int,
    authoritative_phrase: str | None, foreground_iou: float | None, eps: float = 1e-8,
) -> dict[str, Any]:
    structure = parse_structure(parsed, generated_ids, seg_token_id)
    verdict = parsed.get("verdict")
    r_cls = float((verdict == "FAKE") if int(gt_label) == 1 else (verdict == "REAL"))
    r_struct = float(structure["structural_validity"])
    if int(gt_label) == 1:
        overlap = phrase_overlap(authoritative_phrase, parsed.get("target_region"))
        r_phrase = float(overlap["normalized_token_f1"]) if structure["usable_seg"] else 0.0
        r_mask = float(foreground_iou or 0.0) if structure["usable_seg"] else 0.0
        r_align = 2 * r_phrase * r_mask / (r_phrase + r_mask + eps)
    else:
        overlap = None; r_phrase = 0.0; r_mask = 0.0; r_align = 0.0
    components = {
        "R_cls": r_cls, "R_struct": r_struct, "R_phrase_lex": r_phrase,
        "R_mask": r_mask, "R_align": r_align,
    }
    if any(not 0.0 <= value <= 1.0 + 1e-9 for value in components.values()):
        raise ValueError(f"reward component out of range: {components}")
    return {**structure, "phrase_overlap": overlap, **components}


def reward_candidates(gt_label: int, components: Mapping[str, float]) -> dict[str, float]:
    c = components
    if int(gt_label) == 1:
        values = {
            "R0": c["R_mask"],
            "R1": c["R_phrase_lex"],
            "R2": .20*c["R_cls"] + .10*c["R_struct"] + .30*c["R_phrase_lex"] + .40*c["R_mask"],
            "R3": .20*c["R_cls"] + .10*c["R_struct"] + .20*c["R_phrase_lex"] + .30*c["R_mask"] + .20*c["R_align"],
        }
    else:
        value = .70*c["R_cls"] + .30*c["R_struct"]
        values = {"R0": c["R_cls"], "R1": c["R_cls"], "R2": value, "R3": value}
    if any(not 0.0 <= value <= 1.0 + 1e-9 for value in values.values()):
        raise ValueError(f"candidate reward out of range: {values}")
    return values


def group_diversity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("empty rollout group")
    texts = [str(row.get("decoded_text") or "") for row in records]
    normalized = [normalize_output(value) for value in texts]
    phrases = [normalize_output(row.get("normalized_phrase")) for row in records]
    pairs = [normalized_edit_distance(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    result = {
        "number_unique_decoded_outputs": len(set(texts)),
        "number_unique_normalized_outputs": len(set(normalized)),
        "unique_target_phrase_count": len({value for value in phrases if value}),
        "valid_trajectory_count": sum(bool(row.get("structural_validity")) for row in records),
        "pairwise_text_edit_diversity": float(np.mean(pairs)) if pairs else 0.0,
    }
    for reward in REWARD_NAMES:
        values = np.asarray([float(row[reward]) for row in records], dtype=float)
        result[f"{reward}_mean"] = float(values.mean())
        result[f"{reward}_std"] = float(values.std(ddof=0))
    for key, name in (("R_mask", "mask_iou"), ("R_phrase_lex", "phrase_f1")):
        values = [float(row[key]) for row in records]
        result[f"{name}_range"] = max(values) - min(values)
    return result


def pareto_dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = ("R_cls", "R_struct", "R_phrase_lex", "R_mask")
    return all(float(left[k]) >= float(right[k]) for k in keys) and any(
        float(left[k]) > float(right[k]) for k in keys
    )


def pareto_statistics(groups: Iterable[Sequence[Mapping[str, Any]]], reward: str) -> dict[str, Any]:
    correct = tie = violation = count = 0
    for records in groups:
        for i, left in enumerate(records):
            for j, right in enumerate(records):
                if i == j or not pareto_dominates(left, right):
                    continue
                count += 1
                delta = float(left[reward]) - float(right[reward])
                if delta > 1e-12: correct += 1
                elif abs(delta) <= 1e-12: tie += 1
                else: violation += 1
    return {
        "dominance_pair_count": count,
        "correct_count": correct, "tie_count": tie, "violation_count": violation,
        "correct_ranking_rate": correct / count if count else None,
        "tie_rate": tie / count if count else None,
        "violation_rate": violation / count if count else None,
    }


def choose_top(records: Sequence[Mapping[str, Any]], reward: str) -> Mapping[str, Any]:
    return max(records, key=lambda row: (float(row[reward]), -int(row["rollout_index"])))


def histogram(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))
