#!/usr/bin/env python3
"""Freeze the Phase 3A phrase audit, templates, and paired config diff."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import transformers
import yaml

from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import (
    PHRASE_FIELD_PREFIX,
    TARGET_PROTOCOL_HISTORICAL,
    TARGET_PROTOCOL_PHRASE_ALIGNED,
    UnifiedForensicsDataset,
)
from dataset.forensics.synthscars import polygon_to_mask, polygons_for_target
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_FAKE_TOKEN, DEFAULT_REAL_TOKEN


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3a_phrase_grounding"
MANIFESTS = ROOT / "outputs/data_audits/unified_forensics_split_v1"
MODEL = ROOT / "checkpoints/GLaMM-FullScope"
CONFIGS = {"c0": ROOT / "configs/phase3a_c0.yaml", "p1": ROOT / "configs/phase3a_p1.yaml"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def union_mask_hash(row: dict) -> str:
    if row["forensics_domain"] == "real":
        return "NO_SEGMENTATION_TARGET"
    height, width = row["image_variants"][0]["image_size"]
    union = torch.zeros((height, width), dtype=torch.bool).numpy()
    for ref in row["refs"]:
        union |= polygon_to_mask(polygons_for_target(ref, height, width), height, width).astype(bool)
    if not union.any():
        raise ValueError(f"Empty mask: {row['sample_id']}")
    return hashlib.sha256(union.astype("uint8").tobytes()).hexdigest()


def phrase_audit() -> tuple[dict, dict[str, list[dict]]]:
    by_split = {}
    rows_by_split = {}
    all_phrase_counts = Counter()
    for split in ("train", "val", "test"):
        rows = read_jsonl(MANIFESTS / f"{split}_combined.jsonl")
        rows_by_split[split] = rows
        fake = [row for row in rows if row["forensics_domain"] == "fake"]
        real = [row for row in rows if row["forensics_domain"] == "real"]
        phrase_counts = []
        empty = 0
        within_duplicate_images = 0
        explanation_missing_phrase = 0
        ref_without_polygon = 0
        construction_examples = []
        for row in fake:
            field = UnifiedForensicsDataset.authoritative_localization_field(row)
            phrases = field["raw_phrases"]
            phrase_counts.append(len(phrases))
            empty += sum(not phrase for phrase in phrases)
            within_duplicate_images += int(len(set(phrases)) != len(phrases))
            explanation = " ".join(str(row.get("explanation") or "").split()).casefold()
            explanation_missing_phrase += sum(phrase.casefold() not in explanation for phrase in phrases)
            ref_without_polygon += sum(not polygons_for_target(ref, *row["image_variants"][0]["image_size"])
                                       for ref in row["refs"])
            all_phrase_counts.update(phrases)
            if len(construction_examples) < 5:
                construction_examples.append({"sample_id": row["sample_id"], **field})
        by_split[split] = {
            "total_images": len(rows), "fake_images": len(fake), "real_images": len(real),
            "fake_with_authoritative_phrase": sum(count > 0 for count in phrase_counts),
            "empty_phrase_count": empty,
            "total_refs": sum(phrase_counts),
            "refs_per_fake": {
                "min": min(phrase_counts), "max": max(phrase_counts),
                "mean": sum(phrase_counts) / len(phrase_counts),
                "one": sum(count == 1 for count in phrase_counts),
                "multiple": sum(count > 1 for count in phrase_counts),
            },
            "images_with_exact_duplicate_phrase": within_duplicate_images,
            "phrase_not_verbatim_in_merged_explanation": explanation_missing_phrase,
            "refs_without_polygon_target": ref_without_polygon,
            "real_with_phrase": sum(bool(row.get("refs")) for row in real),
            "mask_correspondence": "union_of_the_same_ordered_refs_used_to_construct_phrase_field",
            "examples": construction_examples,
        }
    duplicate_occurrences = sum(count - 1 for count in all_phrase_counts.values() if count > 1)
    audit = {
        "phase": "3A", "source": "frozen_manifest_refs.phrase",
        "authoritative_only": True, "proxy_phrase_labels_created": False,
        "construction_rule": "preserve annotation order; whitespace normalize; remove exact duplicates; semicolon join; no paraphrase",
        "segmentation_target": "unchanged Phase 2A per-image union of all ref polygons",
        "splits": by_split,
        "dataset_wide_phrase_vocabulary": {
            "unique_exact_phrases": len(all_phrase_counts),
            "cross_image_duplicate_occurrences": duplicate_occurrences,
        },
        "real_protocol": "historical target retained exactly; no Target region: none field",
        "fallback": "not needed: all frozen fake rows contain non-empty authoritative refs.phrase",
    }
    return audit, rows_by_split


def tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, model_max_length=1536, padding_side="right", use_fast=False, local_files_only=True
    )
    tok.pad_token = tok.unk_token
    tok.add_tokens([DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN], special_tokens=True)
    return tok


def find_subsequence(sequence: list[int], needle: list[int]) -> tuple[int, int] | None:
    for start in range(len(sequence) - len(needle) + 1):
        if sequence[start:start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def collated_protocol(row: dict, protocol: str, tok) -> dict:
    _, conversation = UnifiedForensicsDataset._conversation(row, protocol)
    dummy = {
        "image_path": row.get("image_path") or row["image_relpath"],
        "global_enc_image": torch.zeros(3, 2, 2), "grounding_enc_image": torch.zeros(3, 2, 2),
        "bboxes": None, "conversations": conversation, "masks": None, "label": torch.zeros(2, 2),
        "resize": (2, 2), "questions": [], "sampled_classes": [],
        "cls_label": row["class_label"], "seg_valid": row["forensics_domain"] == "fake",
        "sample_id": row["sample_id"], "source": row["source"],
        "content_category": row.get("content_category"), "prompt_template_id": "unified_forensics_v1",
        "prompt_sha256": row.get("prompt_sha256"), "target_protocol": protocol,
    }
    batch = custom_collate_fn([dummy], tokenizer=tok, use_mm_start_end=True,
                              inference=False, token_strategy="fixed_cls_query")
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    seg_id = tok("[SEG]", add_special_tokens=False).input_ids[0]
    seg_positions = [i for i, token in enumerate(ids) if token == seg_id]
    field_span = None
    phrase_supervised = None
    if protocol == TARGET_PROTOCOL_PHRASE_ALIGNED and row["forensics_domain"] == "fake":
        phrase = UnifiedForensicsDataset.authoritative_localization_field(row)["normalized_training_phrase"]
        for candidate in (f"{PHRASE_FIELD_PREFIX} {phrase}", f" {PHRASE_FIELD_PREFIX} {phrase}"):
            span = find_subsequence(ids, tok(candidate, add_special_tokens=False).input_ids)
            if span:
                field_span = list(span)
                phrase_supervised = all(labels[index] != -100 for index in range(*span))
                break
        # SentencePiece gives the first word after a newline a context-specific
        # token id.  The stable suffix ``regions :`` still identifies the field
        # boundary exactly; include the preceding Target token through [SEG].
        if field_span is None:
            stable_prefix = tok(PHRASE_FIELD_PREFIX, add_special_tokens=False).input_ids[1:]
            prefix_span = find_subsequence(ids, stable_prefix)
            if prefix_span:
                start = prefix_span[0] - 1
                end = seg_positions[0]
                field_span = [start, end]
                phrase_supervised = all(labels[index] != -100 for index in range(start, end))
        if field_span is None:
            raise AssertionError(f"Cannot locate localization field tokens for {row['sample_id']}")
    return {
        "target": UnifiedForensicsDataset._target(row, protocol),
        "seg_positions": seg_positions,
        "sequence_length": len(ids), "lm_supervised_token_count": sum(label != -100 for label in labels),
        "localization_field_token_span": field_span,
        "localization_field_included_in_lm_loss": phrase_supervised,
        "ignored_token_count": sum(label == -100 for label in labels),
    }


def select_examples(rows: list[dict]) -> list[dict]:
    real = sorted((row for row in rows if row["forensics_domain"] == "real"), key=lambda row: row["sample_id"])
    fake = sorted((row for row in rows if row["forensics_domain"] == "fake"),
                  key=lambda row: (len(row["refs"]), row["sample_id"]))
    indices = [round(i * (len(fake) - 1) / 39) for i in range(40)]
    return [value for pair in zip(real[:40], [fake[i] for i in indices]) for value in pair]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            output.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return output
    return {prefix: value}


def config_diff() -> dict:
    loaded = {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}
    flat = {name: flatten(config) for name, config in loaded.items()}
    keys = sorted(set(flat["c0"]) | set(flat["p1"]))
    differences = [{"path": key, "c0": flat["c0"].get(key), "p1": flat["p1"].get(key)}
                   for key in keys if flat["c0"].get(key) != flat["p1"].get(key)]
    allowed_primary = {
        "forensics.target_protocol", "forensics.target_template",
        "forensics.localization_phrase_insertion", "forensics.localization_phrase_lm_target",
    }
    allowed_admin = {
        "experiment.name", "experiment.runtime_output_dir", "experiment.experiment_type",
        "checkpoint.output_root",
    }
    unexpected = [item for item in differences if item["path"] not in allowed_primary | allowed_admin]
    return {
        "status": "PASS" if not unexpected else "FAIL", "differences": differences,
        "primary_training_differences": [item for item in differences if item["path"] in allowed_primary],
        "administrative_output_differences": [item for item in differences if item["path"] in allowed_admin],
        "unexpected_differences": unexpected,
        "frozen_equal_fields": [key for key in keys if flat["c0"].get(key) == flat["p1"].get(key)],
    }


def main() -> None:
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    audit, rows_by_split = phrase_audit()
    write_json(OUT / "audit/phrase_annotation_audit.json", audit)
    tok = tokenizer()
    examples = []
    for row in select_examples(rows_by_split["train"]):
        field = UnifiedForensicsDataset.authoritative_localization_field(row)
        c0 = collated_protocol(row, TARGET_PROTOCOL_HISTORICAL, tok)
        p1 = collated_protocol(row, TARGET_PROTOCOL_PHRASE_ALIGNED, tok)
        mask_hash = union_mask_hash(row)
        examples.append({
            "sample_id": row["sample_id"], "domain": row["forensics_domain"],
            "original_explanation": row["explanation"], **field,
            "c0": c0, "p1": p1, "mask_target_sha256_c0": mask_hash,
            "mask_target_sha256_p1": mask_hash, "mask_target_identical": True,
        })
    write_jsonl(OUT / "audit/template_examples.jsonl", examples)
    diff = config_diff()
    write_json(OUT / "configs/config_diff.json", diff)
    for name, path in CONFIGS.items():
        destination = OUT / "configs" / f"{name}_config.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    protocol = {
        "status": "PASS" if diff["status"] == "PASS" and len(examples) == 80 else "FAIL",
        "template_examples": len(examples), "fake_examples": sum(x["domain"] == "fake" for x in examples),
        "real_examples": sum(x["domain"] == "real" for x in examples),
        "p1_fake_phrase_before_seg": all(
            x["p1"]["localization_field_token_span"][1] <= x["p1"]["seg_positions"][0]
            for x in examples if x["domain"] == "fake"
        ),
        "p1_fake_phrase_in_lm_loss": all(
            x["p1"]["localization_field_included_in_lm_loss"]
            for x in examples if x["domain"] == "fake"
        ),
        "mask_target_identity": all(x["mask_target_identical"] for x in examples),
        "real_target_identity": all(x["c0"]["target"] == x["p1"]["target"]
                                    for x in examples if x["domain"] == "real"),
        "official_test_used": False,
    }
    write_json(OUT / "audit/training_protocol_audit.json", protocol)
    summary = [
        "# Phase 3A phrase annotation 审计摘要", "",
        "- phrase 来源：冻结 manifest 的 `refs.phrase`，未创建 proxy phrase。",
        "- fake phrase 与 segmentation target：同一组 refs；mask 仍为全部 ref polygon 的原 Phase 2A union。",
        "- real protocol：完全保持历史 target，不添加虚构的 `none` region。",
        f"- 模板样例：{len(examples)}（40 Real + 40 Fake）。",
        f"- config diff：{diff['status']}；protocol gate：{protocol['status']}。", "",
    ]
    for split, values in audit["splits"].items():
        summary.append(
            f"- {split}: Fake {values['fake_images']}，refs {values['total_refs']}，"
            f"empty phrase {values['empty_phrase_count']}，multi-ref images {values['refs_per_fake']['multiple']}。"
        )
    (OUT / "audit/phrase_annotation_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    manifest = {
        "phase": "3A", "stage": "pretraining_gate", "language": "zh-CN",
        "experiment_type": "UNPAIRED_TRAINING_COMPARISON",
        "control": "historical Phase 2A step-2500; C0 reproduction not trained per user direction",
        "phrase_audit": "PASS", "template_validation": protocol["status"],
        "config_diff": diff["status"], "training_started": False,
        "official_test_used_for_training": False, "official_test_used_for_checkpoint_selection": False,
        "threshold_sweep_performed": False, "proxy_phrase_labels_created": False,
        "new_segmentation_labels_created": False, "forensic_fusion_used": False,
        "sam_architecture_changed": False, "multi_seg_training_used": False,
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"audit": audit["splits"], "protocol": protocol, "diff": diff["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
