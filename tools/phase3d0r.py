"""Pure scoring contracts for Phase 3D.0-R reward reformulation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np


STOPWORDS = frozenset({
    "a", "an", "the", "of", "on", "in", "at", "with", "to", "for", "from",
    "by", "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "region", "regions", "area", "areas",
})
NEGATIONS = frozenset({"no", "not", "without"})
SPATIAL_PAIRS = (
    ("left", "right"), ("upper", "lower"), ("top", "bottom"),
    ("front", "back"), ("inner", "outer"), ("inside", "outside"),
    ("center", "peripheral"), ("foreground", "background"),
)
SPATIAL = frozenset(x for pair in SPATIAL_PAIRS for x in pair)
CURATED_SYNONYMS = (
    ("malformed", "abnormal"), ("distorted", "deformed"),
    ("fingers", "digits"), ("finger", "digit"), ("person", "human"),
)


def normalize_phrase(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def content_tokens(text: str | None) -> list[str]:
    return [token for token in normalize_phrase(text).split() if token not in STOPWORDS]


def content_phrase(text: str | None) -> str:
    return " ".join(content_tokens(text))


def build_idf(phrases: Iterable[str]) -> dict[str, float]:
    documents = [set(content_tokens(value)) for value in phrases]
    count = len(documents)
    df = Counter(token for document in documents for token in document)
    raw = {token: math.log((count + 1) / (freq + 1)) + 1 for token, freq in df.items()}
    maximum = max(raw.values(), default=1.0)
    return {token: value / maximum for token, value in sorted(raw.items())}


def token_weight(token: str, idf: Mapping[str, float], spatial_multiplier: float = 2.0) -> float:
    if token in STOPWORDS:
        return 0.0
    value = float(idf.get(token, 1.0))
    return value * (spatial_multiplier if token in SPATIAL else 1.0)


def contradiction_flags(reference: str | None, generated: str | None) -> dict:
    ref = set(content_tokens(reference)); gen = set(content_tokens(generated))
    ambiguous = sorted({f"{a}/{b}" for a, b in SPATIAL_PAIRS if {a, b} <= ref or {a, b} <= gen})
    conflicts = sorted({f"{a}/{b}" for a, b in SPATIAL_PAIRS
                        if (a in ref and b in gen and b not in ref and a not in gen)
                        or (b in ref and a in gen and a not in ref and b not in gen)})
    neg_mismatch = bool(ref & NEGATIONS) != bool(gen & NEGATIONS)
    return {
        "P_spatial_contra": float(bool(conflicts)), "spatial_conflicts": conflicts,
        "AMBIGUOUS_SPATIAL_PHRASE": bool(ambiguous), "ambiguous_spatial_pairs": ambiguous,
        "P_neg_contra": 0.0,
        "NEGATION_MISMATCH": neg_mismatch,
        "negation_penalty_applied": False,
        "negation_note": "scope not inferred without a deterministic dependency parser",
    }


def soft_keyword_score(reference: str | None, generated: str | None, idf: Mapping[str, float],
                       embeddings: Mapping[str, np.ndarray]) -> dict:
    ref = content_tokens(reference); gen = content_tokens(generated)
    if not ref or not gen:
        return {"Recall_soft": 0.0, "Precision_soft": 0.0, "R_key_soft": 0.0}
    matrix = np.asarray([[float(np.clip(np.dot(embeddings[a], embeddings[b]), 0, 1)) for b in gen] for a in ref])
    rw = np.asarray([token_weight(x, idf) for x in ref]); gw = np.asarray([token_weight(x, idf) for x in gen])
    recall = float(np.dot(rw, matrix.max(axis=1)) / max(rw.sum(), 1e-8))
    precision = float(np.dot(gw, matrix.max(axis=0)) / max(gw.sum(), 1e-8))
    score = 2 * precision * recall / (precision + recall + 1e-8)
    return {"Recall_soft": recall, "Precision_soft": precision, "R_key_soft": score}


def semantic_phrase_score(reference: str | None, generated: str | None, idf: Mapping[str, float],
                          embeddings: Mapping[str, np.ndarray]) -> dict:
    if not content_tokens(reference) or not content_tokens(generated):
        flags = contradiction_flags(reference, generated)
        return {"R_key_soft": 0.0, "Recall_soft": 0.0, "Precision_soft": 0.0,
                "sentence_cosine_raw": None, "R_sentence_sem": 0.0,
                "R_phrase_sem_base": 0.0, "R_phrase_sem": 0.0, **flags}
    key = soft_keyword_score(reference, generated, idf, embeddings)
    rphrase, gphrase = content_phrase(reference), content_phrase(generated)
    cosine = float(np.clip(np.dot(embeddings[rphrase], embeddings[gphrase]), -1, 1))
    sentence = float(np.clip((cosine + 1) / 2, 0, 1))
    flags = contradiction_flags(reference, generated)
    base = .65 * key["R_key_soft"] + .35 * sentence
    score = float(np.clip(base * (1 - .5 * flags["P_spatial_contra"])
                          * (1 - .5 * flags["P_neg_contra"]), 0, 1))
    return {**key, "sentence_cosine_raw": cosine, "R_sentence_sem": sentence,
            "R_phrase_sem_base": base, "R_phrase_sem": score, **flags}


def average_percentile_ranks(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    result = []
    for value in values:
        lower = sum(x < value for x in values)
        equal = sum(x == value for x in values)
        result.append((lower + (equal - 1) / 2) / (len(values) - 1))
    return result


def grounding_components(values: Sequence[float], range_scale=.10, ceiling_scale=.30) -> list[dict]:
    low, high = min(values), max(values); span = high - low
    c_range = float(np.clip(span / range_scale, 0, 1)); c_ceiling = float(np.clip(high / ceiling_scale, 0, 1))
    c_ground = c_range * c_ceiling
    return [{"R_rank_mask": rank, "C_range": c_range, "C_ceiling": c_ceiling,
             "C_ground": c_ground, "R_ground_rel": c_ground * rank,
             "mask_group_min": low, "mask_group_max": high, "mask_group_range": span}
            for rank in average_percentile_ranks(values)]


def new_rewards(gt_class: int, row: Mapping[str, float]) -> dict[str, float]:
    if int(gt_class) == 0:
        value = .75 * float(row["R_cls"]) + .25 * float(row["R_struct"])
        return {"Q1": value, "Q2": value, "Q3": value}
    phrase = float(row["R_phrase_sem"]); ground = row.get("R_ground_rel")
    q1 = .25*float(row["R_cls"]) + .10*float(row["R_struct"]) + .65*phrase
    if ground is None:
        return {"Q1": q1, "Q2": None, "Q3": None}
    return {"Q1": q1,
            "Q2": .25*float(row["R_cls"]) + .10*float(row["R_struct"]) + .55*phrase + .10*float(ground),
            "Q3": .25*float(row["R_cls"]) + .10*float(row["R_struct"]) + .45*phrase + .20*float(ground)}


def choose_top(records: Sequence[Mapping], reward: str) -> Mapping:
    return max(records, key=lambda row: (float(row[reward]), -int(row["rollout_index"])))


class FrozenSentenceEncoder:
    """Exact MiniLM sentence-transformers mean-pooling inference contract."""

    def __init__(self, model_id: str, revision: str, cache_dir: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch; self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True).to(self.device)
        self.model.eval(); self.model.requires_grad_(False)
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("semantic encoder freeze contract violated")

    def encode(self, texts: Iterable[str], batch_size: int = 128) -> dict[str, np.ndarray]:
        import torch.nn.functional as functional
        values = sorted(set(str(value) for value in texts))
        result = {}
        with self.torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = values[start:start + batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = functional.normalize(pooled, p=2, dim=1).cpu().numpy()
                result.update(zip(batch, pooled))
        return result
