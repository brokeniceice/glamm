"""Deterministic, inspectable text diagnostics for Phase 2D."""

from __future__ import annotations

import re
from typing import Iterable, Mapping

STOPWORDS = frozenset("a an the of in on at to and or is are was were be been being with for from this that picture image entire almost seems appears found following artifact artifacts examining upon elaborate".split())


def terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2 and token not in STOPWORDS}


def analyze_grounding_text(generated: str, refs: Iterable[Mapping[str, object]]) -> dict:
    refs = list(refs)
    target = terms(" ".join(str(ref.get("phrase") or "") for ref in refs))
    reference = terms(" ".join(
        f"{ref.get('phrase') or ''} {ref.get('explanation') or ''}" for ref in refs
    ))
    generated_terms = terms(generated)
    target_overlap = target & generated_terms
    reference_overlap = reference & generated_terms
    target_recall = len(target_overlap) / len(target) if target else None
    reference_recall = len(reference_overlap) / len(reference) if reference else None
    hallucinated = generated_terms - reference
    if target_recall is None:
        text_status = "ANNOTATION_TARGET_UNAVAILABLE"
    elif target_recall >= .8:
        text_status = "TARGET_PRESENT"
    elif target_recall > 0:
        text_status = "PARTIAL_TARGET"
    else:
        text_status = "MISSING_TARGET"
    return {
        "method": "deterministic_lexical_overlap_v1", "target_terms": sorted(target),
        "generated_target_terms": sorted(target_overlap), "target_term_recall": target_recall,
        "reference_term_recall": reference_recall, "hallucinated_terms": sorted(hallucinated),
        "hallucinated_term_count": len(hallucinated), "text_status": text_status,
        "limitations": "词形和同义词不归一；不能可靠判定空间关系或语义正确性。",
    }


def assign_taxonomy(text: Mapping[str, object], g0_iou: float, tf_iou: float,
                    *, semantic_contradiction: bool, annotation_ambiguous: bool = False) -> str:
    if annotation_ambiguous:
        return "G_GT_ANNOTATION_AMBIGUITY"
    if semantic_contradiction:
        return "E_CLASSIFICATION_SEMANTIC_CONTRADICTION"
    status = text["text_status"]
    if status == "TARGET_PRESENT" and g0_iou <= .30 and tf_iou >= .70:
        return "F_SEG_STATE_ANOMALY_CANDIDATE"
    if status == "PARTIAL_TARGET":
        return "B_PARTIAL_GROUNDING"
    if status == "MISSING_TARGET" and int(text["hallucinated_term_count"]) > 0:
        return "D_HALLUCINATED_GROUNDING_CANDIDATE"
    if status == "MISSING_TARGET":
        return "C_WRONG_OR_MISSING_ENTITY"
    if status == "TARGET_PRESENT" and g0_iou <= .30:
        return "A_CORRECT_LOOKING_TEXT_POOR_MASK"
    return "H_OTHER_UNRESOLVED"
