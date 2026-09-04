#!/usr/bin/env python3
"""Build auditable LEGION adapters from the frozen internal train/val manifests.

This program deliberately has no path or option for internal test, official1000,
or external benchmarks. Stage 1 preserves official LEGION per-reference
semantics: one authoritative phrase, one [SEG], and one mask per source ref.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np
from pycocotools import mask as coco_mask
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


SOURCE = ROOT / "outputs/data_audits/unified_forensics_split_v1"
OUT = ROOT / "outputs/phase5a3_legion_retrained/data"
DATASETS_ROOT = (ROOT / "datasets").resolve()
ORIGINAL_TRAIN_ANNOTATIONS = DATASETS_ROOT / "SynthScars/train/annotations/train.json"
STAGE1 = OUT / "stage1" / "train" / "annotations"
STAGE2 = OUT / "stage2"
ALLOWED_INPUTS = {
    "train_fake": SOURCE / "train_fake.jsonl",
    "train_real": SOURCE / "train_real.jsonl",
    "val_fake": SOURCE / "val_fake.jsonl",
    "val_real": SOURCE / "val_real.jsonl",
}
TOKENIZER_SOURCE = ROOT / "checkpoints/GLaMM-FullScope"
MODEL_MAX_LENGTH = 1536
IMAGE_TOKEN_EXPANSION = 575
TRAIN_TEXT_LIMIT = MODEL_MAX_LENGTH - IMAGE_TOKEN_EXPANSION
GCG_QUESTION = (
    "Please provide a detailed analysis of artifacts in this photo, considering physical artifacts "
    "(e.g., optical display issues, violations of physical laws, and spatial/perspective errors), "
    "structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion "
    "artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material "
    "misrepresentation). Output with interleaved segmentation masks for the corresponding parts of "
    "the answer."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(key: str) -> list[dict]:
    path = ALLOWED_INPUTS[key]
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def official_ref_polygons(
    ref: dict, height: int, width: int
) -> tuple[list[list[float]], list[str], list[str]]:
    polygons: list[list[float]] = []
    errors: list[str] = []
    warnings: list[str] = []
    ref_polygons = ref.get("segmentation") or ref.get("polygons") or []
    for polygon in ref_polygons:
        flat = [float(value) for value in polygon]
        if len(flat) < 6 or len(flat) % 2:
            errors.append("invalid_polygon_coordinate_count")
        else:
            if not all(0 <= value <= (width if idx % 2 == 0 else height)
                       for idx, value in enumerate(flat)):
                warnings.append("polygon_coordinate_out_of_bounds_clipped_by_coco")
            polygons.append(flat)
    if not polygons:
        errors.append("no_valid_polygon")
    else:
        decoded_union = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            rles = coco_mask.frPyObjects([np.asarray(polygon)], height, width)
            decoded_union |= coco_mask.decode(rles).astype(np.uint8).squeeze()
        if not decoded_union.any():
            errors.append("empty_union_after_official_coco_decode")
    return polygons, errors, warnings


def legion_text_audit(caption: str, phrases: list[str], tokenizer) -> dict:
    spans = [(caption.find(phrase), caption.find(phrase) + len(phrase)) for phrase in phrases]
    if any(start < 0 for start, _ in spans):
        return {"sequence_length": None, "seg_positions": [], "seg_survives_truncation": False}
    tagged = caption
    for start, end in sorted(spans, key=lambda value: value[0], reverse=True):
        tagged = f"{tagged[:start]}<p> {tagged[start:end]} </p> [SEG]{tagged[end:]}"
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n{GCG_QUESTION}")
    conv.append_message(conv.roles[1], tagged)
    prompt = conv.get_prompt().replace(
        DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    )
    token_ids = tokenizer_image_token(prompt, tokenizer)
    seg_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    positions = [idx for idx, value in enumerate(token_ids) if value == seg_id]
    return {
        "sequence_length": len(token_ids), "seg_positions": positions,
        "expected_seg_count": len(phrases),
        "seg_survives_truncation": (
            len(positions) == len(phrases) and all(position < TRAIN_TEXT_LIMIT for position in positions)
        ),
    }


def stage1_records(row: dict, tokenizer, raw_by_id: dict[str, dict]) -> list[tuple[dict | None, list[str], dict]]:
    image_reasons: list[str] = []
    if row.get("forensics_domain") != "fake" or int(row.get("class_label", -1)) != 1:
        image_reasons.append("not_fake")
    grouped_image_path = Path(str(row.get("image_path") or ""))
    if not grouped_image_path.is_file():
        image_reasons.append("missing_image")
    warnings: list[str] = []

    annotation_ids = [str(value) for value in row.get("annotation_ids") or []]
    captions = row.get("annotation_captions") or []
    if len(annotation_ids) != len(captions):
        image_reasons.append("annotation_id_caption_count_mismatch")
    results = []
    for annotation_index, annotation_id in enumerate(annotation_ids):
        reasons = list(image_reasons)
        raw_record = raw_by_id.get(annotation_id)
        if raw_record is None:
            reasons.append("annotation_id_missing_from_original_train_json")
            raw_record = {}
        caption = str(raw_record.get("caption") or "").strip('"').strip()
        if not caption:
            reasons.append("empty_original_annotation_caption")
        source_refs = raw_record.get("refs") or []
        if not source_refs:
            reasons.append("no_refs_for_original_annotation")
        raw_image_name = str(raw_record.get("img_file_name") or "")
        annotation_image_path = grouped_image_path.parent / raw_image_name
        if not raw_image_name:
            reasons.append("annotation_source_image_name_missing")
        if not annotation_image_path.is_file():
            reasons.append("missing_original_annotation_image")
            actual_size = None
        else:
            try:
                with Image.open(annotation_image_path) as image:
                    actual_size = (image.height, image.width)
                    image.verify()
            except Exception as exc:
                actual_size = None
                reasons.append(f"annotation_image_decode:{type(exc).__name__}")
        adapter_refs: list[dict] = []
        matched_phrases: list[str] = []
        annotation_warnings = list(warnings)
        for ref_index, ref in enumerate(source_refs):
            phrase = str(ref.get("sentence") or "")
            if not phrase:
                reasons.append(f"ref_{ref_index}:empty_authoritative_phrase")
            polygons: list[list[float]] = []
            polygon_errors: list[str] = []
            polygon_warnings: list[str] = []
            if actual_size is not None:
                polygons, polygon_errors, polygon_warnings = official_ref_polygons(ref, *actual_size)
            # Official LEGION silently ignores a ref whose sentence is not a
            # verbatim caption span, so only matched refs participate in text
            # and mask decoding. Keep the raw ref in JSON either way.
            matched = bool(phrase) and phrase in caption
            if matched:
                matched_phrases.append(phrase)
                reasons.extend(f"ref_{ref_index}:{value}" for value in polygon_errors
                               if value != "empty_union_after_official_coco_decode")
                if "empty_union_after_official_coco_decode" in polygon_errors:
                    annotation_warnings.append(f"ref_{ref_index}:official_empty_mask_preserved")
            else:
                annotation_warnings.append(f"ref_{ref_index}:official_parser_drops_unmatched_phrase")
            annotation_warnings.extend(f"ref_{ref_index}:{value}" for value in polygon_warnings)
            adapter_refs.append(dict(ref))
        if not matched_phrases:
            reasons.append("official_parser_yields_zero_refs")
        text_audit = legion_text_audit(caption, matched_phrases, tokenizer) if caption and matched_phrases else {
            "sequence_length": None, "seg_positions": [], "expected_seg_count": 0,
            "seg_survives_truncation": False,
        }
        if not text_audit["seg_survives_truncation"]:
            surviving = sum(position < TRAIN_TEXT_LIMIT for position in text_audit.get("seg_positions") or [])
            annotation_warnings.append(
                f"official_collate_and_loss_truncate_masks:{len(matched_phrases)}_to_{surviving}"
            )
        reasons = sorted(set(reasons))
        sample_key = f"{row.get('sample_id')}:{annotation_id}"
        audit = {
            "sample_id": sample_key, "image_sample_id": row.get("sample_id"),
            "annotation_id": annotation_id, "source_ref_count": len(source_refs),
            "official_effective_ref_count": len(matched_phrases),
            "adapter_ref_count": len(adapter_refs) if not reasons else 0,
            "adapter_seg_count": len(matched_phrases) if not reasons else 0,
            "per_ref_polygon_counts": [len(value["segmentation"]) for value in adapter_refs],
            "image_size": list(actual_size) if actual_size else None,
            "matched_phrases": matched_phrases, "reasons": reasons,
            "warnings": sorted(set(annotation_warnings)), "text_audit": text_audit,
        }
        record = None if reasons else {
            sample_key: {
                "sample_id": sample_key, "img_file_name": str(annotation_image_path.resolve()),
                "caption": raw_record["caption"], "refs": adapter_refs,
            }
        }
        results.append((record, reasons, audit))
    if not annotation_ids:
        results.append((None, ["no_original_annotation_ids"], {
            "sample_id": row.get("sample_id"), "image_sample_id": row.get("sample_id"),
            "reasons": ["no_original_annotation_ids"], "warnings": warnings,
        }))
    return results


def cls_record(row: dict) -> dict:
    explicit = str(row.get("image_path") or "")
    image_path = Path(explicit) if explicit else DATASETS_ROOT / str(row.get("image_relpath") or "")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing classification image: {row.get('sample_id')} {image_path}")
    frozen_label = 0 if row.get("forensics_domain") == "real" else 1
    if int(row.get("class_label", -1)) != frozen_label:
        raise ValueError(f"Frozen class mismatch: {row.get('sample_id')}")
    official_legion_label = 1 if row.get("forensics_domain") == "real" else 0
    return {"sample_id": row["sample_id"], "image_path": str(image_path.resolve()), "label": official_legion_label}


def main() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_SOURCE, model_max_length=MODEL_MAX_LENGTH, padding_side="right",
        use_fast=False, local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    rows = {key: read_jsonl(key) for key in ALLOWED_INPUTS}
    raw_items = json.loads(ORIGINAL_TRAIN_ANNOTATIONS.read_text(encoding="utf-8"))
    raw_by_id = {str(next(iter(item))): next(iter(item.values())) for item in raw_items}
    expected = {"train_fake": 8836, "train_real": 8836, "val_fake": 1106, "val_real": 1106}
    if {key: len(value) for key, value in rows.items()} != expected:
        raise RuntimeError("Frozen internal train/val counts drifted")

    train_ids = {row["sample_id"] for row in rows["train_fake"] + rows["train_real"]}
    val_ids = {row["sample_id"] for row in rows["val_fake"] + rows["val_real"]}
    if len(train_ids) != 17672 or len(val_ids) != 2212 or train_ids & val_ids:
        raise RuntimeError("Train/validation identity firewall failed")

    conversion: dict[str, dict] = {}
    for split_key, output_name in (("train_fake", "train.json"), ("val_fake", "test.json")):
        converted: list[dict] = []
        audits: list[dict] = []
        exclusions: Counter[str] = Counter()
        warnings: Counter[str] = Counter()
        for row in rows[split_key]:
            for record, reasons, audit in stage1_records(row, tokenizer, raw_by_id):
                audits.append(audit)
                warnings.update(audit.get("warnings") or [])
                if record is None:
                    exclusions.update(reasons)
                else:
                    converted.append(record)
        output = STAGE1 / output_name
        dump(output, converted)
        identity = Counter()
        for item in converted:
            value = next(iter(item.values()))
            annotation_id = value["sample_id"].rsplit(":", 1)[1]
            original = raw_by_id[annotation_id]
            identity["missing_raw"] += int(annotation_id not in raw_by_id)
            identity["caption_diff"] += int(value["caption"] != original["caption"])
            identity["image_basename_diff"] += int(
                Path(value["img_file_name"]).name != original["img_file_name"]
            )
            identity["refs_diff"] += int(value["refs"] != original["refs"])
        audit_path = OUT / "audits" / f"{split_key}.json"
        dump(audit_path, audits)
        conversion[split_key] = {
            "candidate_images": len(rows[split_key]), "candidate_annotation_samples": len(audits),
            "usable": len(converted), "excluded": len(audits) - len(converted),
            "exclusion_reason_occurrences": dict(sorted(exclusions.items())),
            "warning_occurrences": dict(sorted(warnings.items())),
            "exact_original_field_identity": {
                "compared": len(converted),
                **{key: identity[key] for key in ("missing_raw", "caption_diff", "image_basename_diff", "refs_diff")},
            },
            "output": str(output), "output_sha256": sha256(output),
            "audit_sha256": sha256(audit_path),
        }

    stage2_train = [cls_record(row) for row in rows["train_real"] + rows["train_fake"]]
    stage2_val = [cls_record(row) for row in rows["val_real"] + rows["val_fake"]]
    stage2_train.sort(key=lambda value: value["sample_id"])
    stage2_val.sort(key=lambda value: value["sample_id"])
    dump(STAGE2 / "train.json", stage2_train)
    dump(STAGE2 / "val.json", stage2_val)

    summary = {
        "schema": "phase5a3_data_adapter_v1",
        "allowed_inputs_only": sorted(str(path) for path in ALLOWED_INPUTS.values()),
        "input_sha256": {key: sha256(path) for key, path in ALLOWED_INPUTS.items()},
        "stage1_original_annotation_lookup": {
            "path": str(ORIGINAL_TRAIN_ANNOTATIONS), "sha256": sha256(ORIGINAL_TRAIN_ANNOTATIONS),
            "policy": "frozen manifest annotation IDs select exact raw records; loader sees converted selected records only",
        },
        "stage1_mapping": {
            "sample_unit": "one original annotation_id/caption entry, matching official SynthScars JSON",
            "caption": "exact original SynthScars train.json caption selected by frozen annotation ID",
            "phrase": "one verbatim authoritative ref phrase per original annotation",
            "segmentation": "one independent refs[i] polygon mask per phrase; no union",
            "unmatched_phrase": "preserved in adapter JSON and silently dropped by official parser",
            "long_sample": "preserved; official collate truncates tokens and Legion loss truncates GT masks to pred count",
            "seg_tokens_per_sample": "equal to official parser matched ref count",
            "core_code_modified": False,
            "official_collate_text_limit": TRAIN_TEXT_LIMIT,
            "tokenizer_source": str(TOKENIZER_SOURCE),
            "tokenizer_model_sha256": sha256(TOKENIZER_SOURCE / "tokenizer.model"),
        },
        "stage1": conversion,
        "stage2": {
            "train": {"total": len(stage2_train), "real_1": 8836, "fake_0": 8836,
                      "sha256": sha256(STAGE2 / "train.json")},
            "validation": {"total": len(stage2_val), "real_1": 1106, "fake_0": 1106,
                           "sha256": sha256(STAGE2 / "val.json")},
        },
        "identity_firewall": {"train_unique": len(train_ids), "val_unique": len(val_ids),
                              "train_val_overlap": len(train_ids & val_ids)},
        "forbidden_loader_inputs": {
            "internal_test": False, "official1000": False, "LOKI": False,
            "RAISE": False, "AIGI_test": False, "external_benchmarks": False,
        },
    }
    summary["canonical_sha256"] = canonical_sha(summary)
    dump(OUT / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
