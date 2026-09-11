#!/usr/bin/env python3
"""Final frozen localization supervisor and resumable workers.

Protocol
--------
Historical reuse (no model inference):
  * SynthScars Official1000: P1/R1 historical G0; LEGION public/retrained L-FREE.
  * LOKI229: P1/R1 historical G1; LEGION public/retrained L-FREE.

Fresh evaluation:
  * X-AIGD official labeled_test: P1/R1 G1; LEGION public/retrained L-FREE.
  * PAL4VST official test: P1/R1 G1; LEGION public/retrained L-FREE.

The file is both the top-level supervisor (no --worker argument) and the worker
entry point used by that supervisor.  It never modifies raw benchmark data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/final_evaluation/localization"
LOGS = ROOT / "outputs/final_evaluation/logs"
STATUS = OUT / "supervisor_status.json"
DATA_GATE = ROOT / "outputs/final_eval_datasets/data_pipeline_status.json"
LEAKAGE_AUDIT = ROOT / "outputs/final_eval_datasets/leakage_audit/summary.json"
LEAKAGE_OVERRIDE = ROOT / "outputs/final_eval_datasets/leakage_override.json"

GLAMM_PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
LEGION_PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
PHYSICAL_GPU = int(os.environ.get("FINAL_EVAL_LOCALIZATION_GPU", "2"))

P1_PATH = Path("/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt")
P1_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"
R1_PATH = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"
R1_SHA = "9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5"

LEGION_REPO = ROOT / "external/LEGION_official"
LEGION_COMMIT = "d21535dd45f6fea509337a83095966f0b86ac924"
CLIP_PATH = Path("/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
LEGION_PUBLIC = Path("/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa")
LEGION_PUBLIC_REVISION = "f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa"
LEGION_RETRAINED_CANDIDATES = (
    Path("/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/stage1_le"),
    ROOT / "checkpoints/phase5a3_legion_retrained/stage1_le",
)
LEGION_RETRAINED_SHA = "6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844"

MANIFESTS = {
    "synthscars": ROOT / "datasets/SynthScars/manifests/eval_manifest.jsonl",
    "loki": ROOT / "datasets/LOKI/manifests/localization_eval_manifest.jsonl",
    "xaigd": ROOT / "datasets/X-AIGD/manifests/eval_manifest.jsonl",
    "pal4vst": ROOT / "datasets/PAL4VST/manifests/eval_manifest.jsonl",
}
GT_TYPES = {
    "synthscars": "official_polygon_union",
    "loki": "bounding_box_union",
    "xaigd": "official_human_artifact_polygon_union",
    "pal4vst": "official_pixel_artifact_mask",
}
SPLITS = {
    "synthscars": "official_test_Official1000",
    "loki": "frozen_localization_229",
    "xaigd": "labeled_test",
    "pal4vst": "official_test",
}
MODELS = ("p1", "r1", "legion_intermediate", "legion_retrained")
DATASETS = ("synthscars", "loki", "xaigd", "pal4vst")
FAILURE_POLICY_VERSION = "full_n_no_seg_zero_v1"

HIST_P1_R1 = ROOT / "outputs/p1_r1_reusable_matrix/jobs"
HIST_LEGION_PUBLIC_OFFICIAL = ROOT / "outputs/phase5a2_legion_public_reference/original/shards"
HIST_LEGION_RETRAINED_OFFICIAL = ROOT / "outputs/phase5a4_legion_retrained_controlled/official1000/original/shards"
HIST_LEGION_PUBLIC_LOKI = ROOT / "outputs/phase5a2_legion_loki_reference/original/shards"
HIST_LEGION_RETRAINED_LOKI = ROOT / "outputs/phase5a4_legion_retrained_loki/original/shards"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("No checkpoint candidate exists: " + ", ".join(map(str, candidates)))


def git_head(path: Path) -> str:
    value = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return value


def manifest_info(dataset: str) -> tuple[list[dict], dict]:
    path = MANIFESTS[dataset]
    if not path.is_file():
        raise FileNotFoundError(path)
    values = rows(path)
    ids = [str(row["sample_id"]) for row in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate sample IDs in {dataset}")
    expected = {"synthscars": 1000, "loki": 229, "xaigd": 2419}.get(dataset)
    if expected is not None and len(values) != expected:
        raise RuntimeError(f"{dataset} population drift: {len(values)} != {expected}")
    if dataset == "pal4vst" and not values:
        raise RuntimeError("PAL4VST official test manifest is empty")
    return values, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n": len(values),
        "ordered_sample_id_sha256": ordered_id_sha(ids),
    }


def checkpoint_identity(model: str) -> dict:
    if model == "p1":
        if not P1_PATH.is_file() or sha256_file(P1_PATH) != P1_SHA:
            raise RuntimeError("P1 checkpoint provenance drift")
        return {"path": str(P1_PATH.resolve()), "sha256": P1_SHA, "kind": "file"}
    if model == "r1":
        if not R1_PATH.is_file() or sha256_file(R1_PATH) != R1_SHA:
            raise RuntimeError("R1 checkpoint provenance drift")
        return {"path": str(R1_PATH.resolve()), "sha256": R1_SHA, "kind": "file"}
    if model == "legion_intermediate":
        if not LEGION_PUBLIC.is_dir():
            raise FileNotFoundError(LEGION_PUBLIC)
        return {
            "path": str(LEGION_PUBLIC.resolve()), "hf_revision": LEGION_PUBLIC_REVISION,
            "kind": "huggingface_directory", "role": "public_intermediate_LE",
        }
    if model == "legion_retrained":
        path = first_existing(LEGION_RETRAINED_CANDIDATES)
        identity_paths = (
            ROOT / "checkpoints/phase5a3_legion_retrained/stage1_le_identity.json",
            Path("/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/stage1_le_identity.json"),
        )
        identity = next((read_json(p) for p in identity_paths if p.is_file()), None)
        if identity is not None and identity.get("canonical_sha256") != LEGION_RETRAINED_SHA:
            raise RuntimeError("LEGION-retrained Stage-1 checkpoint provenance drift")
        return {
            "path": str(path), "canonical_sha256": LEGION_RETRAINED_SHA,
            "kind": "huggingface_directory", "role": "retrained_stage1_LE",
        }
    raise ValueError(model)


def metric_from_binary(binary, target) -> dict:
    import numpy as np
    pred = np.asarray(binary, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise RuntimeError(f"metric shape mismatch: {pred.shape} vs {truth.shape}")
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    union = tp + fp + fn
    f1d = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "foreground_iou": float(tp / union if union else 1.0),
        "foreground_f1": float(2 * tp / f1d if f1d else 1.0),
    }


def enforce_failure_policy(record: dict) -> dict:
    """Keep confusion pixels for an all-zero prediction, but score protocol failures as zero.

    Empty-GT/empty-prediction remains a perfect per-image match only when the
    model actually emitted a valid segmentation trajectory.  A missing [SEG],
    missing mask, invalid shape, or invalid R1 q-seg is a full-N failure and
    receives foreground IoU/F1 zero regardless of GT emptiness.
    """
    value = dict(record)
    status = str(value.get("status") or "")
    failure = status in {
        "EMPTY_OR_NO_SEG", "NO_VALID_QSEG", "NO_SEG", "EMPTY_MASK",
        "INVALID_SHAPE", "MASK_WITHOUT_SEG",
    }
    if value.get("seg_triggered") is False or value.get("valid_q_seg") is False:
        failure = True
    if "seg_count" in value and int(value.get("seg_count") or 0) <= 0:
        failure = True
    if failure:
        value["foreground_iou"] = 0.0
        value["foreground_f1"] = 0.0
        value["failure_score_forced_zero"] = True
    else:
        value["failure_score_forced_zero"] = False
    value["failure_policy_version"] = FAILURE_POLICY_VERSION
    return value


def summarize(records: list[dict]) -> dict:
    import numpy as np
    if not records:
        raise RuntimeError("cannot summarize empty localization records")
    iou = np.asarray([float(row["foreground_iou"]) for row in records], dtype=np.float64)
    f1 = np.asarray([float(row["foreground_f1"]) for row in records], dtype=np.float64)
    tp = sum(int(row["tp"]) for row in records)
    fp = sum(int(row["fp"]) for row in records)
    fn = sum(int(row["fn"]) for row in records)
    tn = sum(int(row["tn"]) for row in records) if all("tn" in row for row in records) else None
    gt_foreground = np.asarray([int(row["tp"]) + int(row["fn"]) for row in records], dtype=np.int64)
    nonempty = gt_foreground > 0
    return {
        "n": len(records),
        "empty_gt_images": int((~nonempty).sum()),
        "nonempty_gt_images": int(nonempty.sum()),
        "mean_foreground_iou": float(iou.mean()),
        "median_foreground_iou": float(np.median(iou)),
        "mean_foreground_f1": float(f1.mean()),
        "mean_foreground_iou_nonempty_gt": float(iou[nonempty].mean()) if nonempty.any() else None,
        "mean_foreground_f1_nonempty_gt": float(f1[nonempty].mean()) if nonempty.any() else None,
        "global_foreground_iou": float(tp / max(1, tp + fp + fn)),
        "global_foreground_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "confusion_pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "threshold_logit": 0.0,
    }


def result_path(model: str, dataset: str) -> Path:
    return OUT / model / dataset / "results.json"


def prediction_path(model: str, dataset: str) -> Path:
    return OUT / model / dataset / "predictions.jsonl"


def write_result(
    model: str,
    dataset: str,
    records: list[dict],
    *,
    execution: str,
    inference_condition: str,
    source_artifact: str | None = None,
    source_artifact_sha256: str | None = None,
    notes: str | None = None,
    extra: dict | None = None,
) -> dict:
    manifest_rows, manifest = manifest_info(dataset)
    ids = [str(row["sample_id"]) for row in manifest_rows]
    records = [enforce_failure_policy(compact_record(row)) for row in records]
    if [str(row["sample_id"]) for row in records] != ids:
        raise RuntimeError(f"{model}/{dataset} sample identity/order mismatch")
    # Persist the exact ordered per-image records for historical reuse as well
    # as fresh workers.  The finalizer validates identity/order and recomputes
    # every aggregate from this file; a summary-only historical cell is not an
    # auditable result.
    atomic_jsonl(prediction_path(model, dataset), records)
    value = {
        "schema": "final_eval_localization_result_v1",
        "status": "COMPLETE",
        "model": model,
        "dataset": dataset,
        "split": SPLITS[dataset],
        "execution": execution,
        "inference_condition": inference_condition,
        "checkpoint": checkpoint_identity(model),
        "manifest": manifest,
        "gt_type": GT_TYPES[dataset],
        "metrics": summarize(records),
        "metric_semantics": (
            "X-AIGD category-agnostic primary comparison uses dataset-global foreground IoU/F1 from TP/FP/FN. Official empty-label samples are retained as all-zero GT; per-image means are supplementary."
            if dataset == "xaigd" else
            "Frozen project localization metrics: fixed mask-logit > 0; per-image and dataset-global foreground IoU/F1 are reported."
        ),
        "failure_policy": {
            "version": FAILURE_POLICY_VERSION,
            "no_seg_or_invalid_prediction": "all-zero prediction with per-image foreground IoU/F1 forced to 0",
            "valid_segmentation_empty_prediction_on_empty_gt": "per-image foreground IoU/F1 = 1",
            "global_metrics": "computed from TP/FP/FN and therefore unaffected by zero-denominator per-image convention",
        },
        "source_artifact": source_artifact,
        "source_artifact_sha256": source_artifact_sha256,
        "notes": notes,
        "updated_at_utc": now(),
    }
    if extra:
        value.update(extra)
    atomic_json(result_path(model, dataset), value)
    return value


def compact_record(row: dict) -> dict:
    keys = (
        "sample_id", "ordinal", "foreground_iou", "foreground_f1", "tp", "fp", "fn", "tn",
        "status", "seg_triggered", "has_pred_mask", "valid_q_seg", "generated_text",
        "generator/source", "artifact_categories", "gt_foreground_pixels", "gt_is_empty",
        "failure_score_forced_zero", "failure_policy_version",
    )
    return {key: row.get(key) for key in keys if key in row}


def load_sharded_predictions(root: Path, ranges: tuple[tuple[int, int], ...]) -> tuple[list[dict], list[Path]]:
    files = [root / f"{start:04d}_{end:04d}.predictions.jsonl" for start, end in ranges]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return [row for path in files for row in rows(path)], files


def materialize_historical_reuse() -> None:
    # P1/R1 Official1000 G0.
    synth_rows, synth_info = manifest_info("synthscars")
    synth_ids = [str(r["sample_id"]) for r in synth_rows]
    official_job_path = HIST_P1_R1 / "official_g0.json"
    job = read_json(official_job_path)
    if job.get("status") != "COMPLETE" or job.get("population") != "official1000" or job.get("mode") != "g0":
        raise RuntimeError("historical P1/R1 Official1000 G0 artifact is not frozen COMPLETE")
    for model, key in (("p1", "P1"), ("r1", "R1")):
        source_records = job[key]["records"]
        if [str(r["sample_id"]) for r in source_records] != synth_ids:
            raise RuntimeError(f"historical {model} Official1000 sample drift")
        write_result(
            model, "synthscars", source_records,
            execution="REUSED_HISTORICAL", inference_condition="G0",
            source_artifact=str(official_job_path.resolve()),
            source_artifact_sha256=sha256_file(official_job_path),
            notes="Historical frozen Official1000 G0 result; current final manifest was rebuilt from the same source sample identities.",
            extra={"historical_manifest_identity_match": True, "current_manifest_sha256": synth_info["sha256"]},
        )

    # LEGION Official1000 L-FREE.
    for model, source_root in (
        ("legion_intermediate", HIST_LEGION_PUBLIC_OFFICIAL),
        ("legion_retrained", HIST_LEGION_RETRAINED_OFFICIAL),
    ):
        source_records, source_files = load_sharded_predictions(source_root, ((0, 500), (500, 1000)))
        if [str(r["sample_id"]) for r in source_records] != synth_ids:
            raise RuntimeError(f"historical {model} Official1000 sample drift")
        write_result(
            model, "synthscars", source_records,
            execution="REUSED_HISTORICAL", inference_condition="L-FREE",
            source_artifact=";".join(str(p.resolve()) for p in source_files),
            source_artifact_sha256=hashlib.sha256("".join(sha256_file(p) for p in source_files).encode()).hexdigest(),
            notes="Historical official LEGION L-FREE: free explanation/[SEG], all masks union, logit > 0, full-N.",
        )

    # P1/R1 LOKI229 G1.
    loki_rows, _ = manifest_info("loki")
    loki_ids = [str(r["sample_id"]) for r in loki_rows]
    loki_job_path = HIST_P1_R1 / "loki_g1.json"
    job = read_json(loki_job_path)
    if job.get("status") != "COMPLETE" or job.get("population") != "loki" or job.get("mode") != "g1":
        raise RuntimeError("historical P1/R1 LOKI G1 artifact is not frozen COMPLETE")
    for model, key in (("p1", "P1"), ("r1", "R1")):
        source_records = job[key]["records"]
        if [str(r["sample_id"]) for r in source_records] != loki_ids:
            raise RuntimeError(f"historical {model} LOKI sample drift")
        write_result(
            model, "loki", source_records,
            execution="REUSED_HISTORICAL", inference_condition="G1",
            source_artifact=str(loki_job_path.resolve()), source_artifact_sha256=sha256_file(loki_job_path),
            notes="LOKI P1/R1 uses known-Fake G1 continuation; GT is the union of 687 official boxes on 229 images.",
            extra={"strict_cross_model_prompt_parity_with_legion": False},
        )

    # LEGION LOKI229 L-FREE.
    for model, source_root in (
        ("legion_intermediate", HIST_LEGION_PUBLIC_LOKI),
        ("legion_retrained", HIST_LEGION_RETRAINED_LOKI),
    ):
        source_records, source_files = load_sharded_predictions(source_root, ((0, 115), (115, 229)))
        if [str(r["sample_id"]) for r in source_records] != loki_ids:
            raise RuntimeError(f"historical {model} LOKI sample drift")
        write_result(
            model, "loki", source_records,
            execution="REUSED_HISTORICAL", inference_condition="L-FREE",
            source_artifact=";".join(str(p.resolve()) for p in source_files),
            source_artifact_sha256=hashlib.sha256("".join(sha256_file(p) for p in source_files).encode()).hexdigest(),
            notes="LOKI LEGION uses official image-only L-FREE; GT remains bounding-box-derived union.",
            extra={"strict_cross_model_prompt_parity_with_p1_r1": False},
        )


def _xaigd_annotation_map() -> dict[str, dict]:
    path = ROOT / "datasets/X-AIGD/processed/labeled_test_records.jsonl"
    values = rows(path)
    result = {str(row["sample_id"]): row for row in values}
    if len(result) != 2419:
        raise RuntimeError(f"X-AIGD annotation record drift: {len(result)}")
    return result


class GroundTruthProvider:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.xaigd = _xaigd_annotation_map() if dataset == "xaigd" else None

    def mask(self, row: dict, height: int, width: int):
        import cv2
        import numpy as np
        if self.dataset == "xaigd":
            source = self.xaigd.get(str(row["sample_id"]))
            if source is None:
                raise RuntimeError(f"X-AIGD annotation missing: {row['sample_id']}")
            if int(source["width"]) != width or int(source["height"]) != height:
                raise RuntimeError(f"X-AIGD geometry drift: {row['sample_id']}")
            target = np.zeros((height, width), dtype=np.uint8)
            # Official X-AIGD permits labeled_test records with no artifact polygons.
            # The official evaluator starts from an all-zero GT mask and simply
            # returns it when labels == []; these samples must remain in the split.
            labels = source.get("labels") or []
            for label in labels:
                points = label.get("points") or []
                if len(points) < 3:
                    raise RuntimeError(f"X-AIGD polygon has <3 points: {row['sample_id']}")
                polygon = np.asarray(points, dtype=np.int32)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                cv2.fillPoly(target, [polygon], 255)
            result = target > 0
        elif self.dataset == "pal4vst":
            path = Path(row["gt_annotation_path"])
            target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if target is None:
                raise OSError(path)
            if target.shape != (height, width):
                raise RuntimeError(f"PAL4VST GT geometry drift: {row['sample_id']} {target.shape} != {(height, width)}")
            result = target > 0
        else:
            raise ValueError(self.dataset)
        # Empty GT is valid in both external releases: X-AIGD can have
        # labels=[], and PAL4VST ships official all-background masks.
        return result


class ExternalArtifactDataset:
    """GLaMM-compatible Fake-only adapter for X-AIGD/PAL4VST final manifests."""
    def __init__(self, dataset: str, tokenizer, global_image_encoder: str, image_size: int):
        import torch
        from transformers import CLIPImageProcessor
        from model.SAM.utils.transforms import ResizeLongestSide
        self.dataset = dataset
        self.rows, _ = manifest_info(dataset)
        self.gt = GroundTruthProvider(dataset)
        self.global_processor = CLIPImageProcessor.from_pretrained(global_image_encoder, local_files_only=True)
        self.transform = ResizeLongestSide(image_size)
        self.torch = torch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        import cv2
        import numpy as np
        import torch
        from dataset.forensics.unified import (
            CANONICAL_PROMPT_SHA256, CANONICAL_PROMPT_TEMPLATE_ID,
            CANONICAL_UNIFIED_QUESTION, UnifiedForensicsDataset,
        )
        row = self.rows[index]
        image_path = Path(row["image_path"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        target = self.gt.mask(row, height, width)
        resized = self.transform.apply_image(rgb)
        manifest_row = {
            **row,
            "forensics_domain": "fake",
            "class_label": 1,
            "explanation": "",
        }
        return {
            "image_path": str(image_path.resolve()),
            "global_enc_image": self.global_processor.preprocess(rgb, return_tensors="pt")["pixel_values"][0],
            "grounding_enc_image": UnifiedForensicsDataset.grounding_enc_processor(
                torch.from_numpy(resized).permute(2, 0, 1).contiguous()
            ),
            "bboxes": None,
            "conversations": [],
            "masks": torch.from_numpy(target.astype(np.float32)).unsqueeze(0),
            "label": torch.full((height, width), UnifiedForensicsDataset.IGNORE_LABEL, dtype=torch.long),
            "resize": resized.shape[:2],
            "questions": [CANONICAL_UNIFIED_QUESTION],
            "sampled_classes": ["synthetic artifact"],
            "cls_label": 1,
            "seg_valid": True,
            "sample_id": str(row["sample_id"]),
            "source": row.get("generator/source") or row.get("dataset"),
            "content_category": None,
            "manifest_row": manifest_row,
            "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256,
            "target_protocol": "final_external_g1",
        }


def empty_metric_record(sample_id: str, target) -> dict:
    import numpy as np
    truth = np.asarray(target, dtype=bool)
    # Keep exactly the same binary-mask metric semantics as a normal all-zero
    # prediction. For empty GT, empty prediction has per-image IoU/F1=1.
    value = metric_from_binary(np.zeros_like(truth, dtype=bool), truth)
    value["sample_id"] = sample_id
    return value


def run_glamm_pair(dataset_name: str, device_name: str) -> None:
    import numpy as np
    import torch
    import yaml
    from eval.forensics import _localization_record
    from model.llava import conversation as conversation_lib
    from scripts import p1_r1_reusable_matrix as prm
    from scripts.phase3a_evaluate import preserve_spatial_prediction

    if dataset_name not in ("xaigd", "pal4vst"):
        raise ValueError(dataset_name)
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    checkpoint_identity("p1"); checkpoint_identity("r1")
    cfg = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    model, tokenizer, backend, _ = prm.load_p1(device)
    utility, rectifier, sam = prm.load_r1(device)
    source = prm.load_evidence_source(prm.P4F_CFG, "forensic_rect", device)
    dataset = ExternalArtifactDataset(
        dataset_name, tokenizer, cfg["model"]["vision_tower"], int(cfg["model"]["image_size"])
    )
    core = prm.core_model(model)
    vision_tower = model.get_model().get_vision_tower()

    work_root = OUT / "_work" / f"glamm_pair_{dataset_name}"
    work_path = work_root / "paired_predictions.jsonl"
    existing = rows(work_path)
    for paired in existing:
        if "p1" in paired:
            paired["p1"] = enforce_failure_policy(paired["p1"])
        if "r1" in paired:
            paired["r1"] = enforce_failure_policy(paired["r1"])
    indexed = {str(row["sample_id"]): row for row in existing}
    expected_ids = [str(row["sample_id"]) for row in dataset.rows]
    if not set(indexed).issubset(set(expected_ids)):
        raise RuntimeError(f"{dataset_name} GLaMM resume scope drift")

    worker_status = work_root / "worker_status.json"
    state = {
        "schema": "final_eval_glamm_pair_worker_v1", "status": "RUNNING",
        "dataset": dataset_name, "models": ["p1", "r1"], "inference_condition": "G1",
        "pid": os.getpid(), "started_at_utc": now(), "device": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest_sha256": sha256_file(MANIFESTS[dataset_name]),
    }
    atomic_json(worker_status, state)
    try:
        for ordinal in range(len(dataset)):
            sample_id = expected_ids[ordinal]
            if sample_id in indexed:
                continue
            sample = dataset[ordinal]
            output = backend.generate_localization(
                sample, provide_gt_fake=True, generation_mode="unified_prompt_gt_fake_prefix"
            )
            record = _localization_record(
                sample, output, "G1", uses_gt_authenticity=True,
                uses_gt_explanation=False, classification_gate=False,
            )
            record = preserve_spatial_prediction(
                work_root, "G1", sample, output, record, save_spatial=False
            )
            target = sample["masks"].bool().any(0)
            gt_pixels = int(target.sum().item())
            p1 = enforce_failure_policy(compact_record({
                **record, "ordinal": ordinal,
                "status": "OK" if record["seg_triggered"] and record["has_pred_mask"] else "EMPTY_OR_NO_SEG",
                "gt_foreground_pixels": gt_pixels, "gt_is_empty": gt_pixels == 0,
            }))

            qseg, raw_clip = prm.context_for("g1", record, sample, backend, core, vision_tower)
            if qseg is None:
                r1 = empty_metric_record(sample_id, target.cpu().numpy())
                r1.update({"ordinal": ordinal, "valid_q_seg": False, "status": "NO_VALID_QSEG"})
            else:
                h, w = target.shape
                sg = prm.geometry_for("sam", (h, w))
                cg = prm.geometry_for("clip", (h, w))
                raw = model.get_grounding_encoder_embs(
                    sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                )
                if raw_clip is None:
                    tokens, _ = vision_tower(
                        sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16)
                    )
                    raw_clip = prm.clip_grid(tokens).cpu()
                raw_clip = raw_clip.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    forensic = source(raw_clip, return_features=True)
                s64 = prm.sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
                qone = qseg[None].to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type=device.type, enabled=False):
                    low_p1 = sam(qone, raw.to(torch.bfloat16))
                zl = prm.sam_lowres_to_original_normalized(low_p1, sg, output_hw=(256, 256)).to(torch.bfloat16)
                f24 = forensic["F_forensic"].to(torch.bfloat16)
                zf = forensic["logits"].to(torch.bfloat16)
                batch = {
                    "S64": s64, "q_seg": qone, "z_L": zl, "F24": f24, "z_F24": zf,
                    "clip_geometries": [cg],
                    "valid_g0": torch.ones(1, dtype=torch.bool, device=device),
                    "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                    "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device),
                    "forensic_off": torch.zeros(1, dtype=torch.bool, device=device),
                }
                uout = prm.hc.utility_forward(utility, batch)
                sc = prm.sam_coordinates(sg, grid=64)[None].to(device)
                cc = prm.clip_coordinates(cg, grid=24)[None].to(device)
                valid = torch.ones(1, 576, dtype=torch.bool, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    p4f = rectifier(raw, f24, sc, cc, valid)
                gate = prm.ha.gate_to_sam_grid(uout["U"], sc) * p4f["support"].reshape(1, 1, 64, 64).float()
                adapted = prm.ha.gated_embedding(raw, p4f["image_embeddings"], gate)
                with torch.autocast(device_type=device.type, enabled=False):
                    low = sam(qone, adapted.to(torch.bfloat16))
                r1 = prm.metric_record(sample_id, prm.inverse_sam_logits(low, sg), target)
                r1["tn"] = int(target.numel()) - int(r1["tp"]) - int(r1["fp"]) - int(r1["fn"])
                r1.update({"ordinal": ordinal, "valid_q_seg": True, "status": "OK"})

            r1.update({"gt_foreground_pixels": gt_pixels, "gt_is_empty": gt_pixels == 0})
            r1 = enforce_failure_policy(r1)
            paired = {
                "sample_id": sample_id, "ordinal": ordinal,
                "generator/source": dataset.rows[ordinal].get("generator/source"),
                "artifact_categories": dataset.rows[ordinal].get("artifact_categories"),
                "p1": p1, "r1": compact_record(r1),
            }
            append_jsonl(work_path, paired)
            indexed[sample_id] = paired
            if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(dataset):
                print(json.dumps({"worker": "glamm_pair", "dataset": dataset_name, "done": ordinal + 1, "total": len(dataset)}), flush=True)

        ordered = [indexed[sid] for sid in expected_ids]
        if len(ordered) != len(expected_ids):
            raise RuntimeError(f"{dataset_name} paired result incomplete")
        atomic_jsonl(work_path, ordered)
        for model in ("p1", "r1"):
            records_out = []
            for row in ordered:
                value = dict(row[model])
                value["sample_id"] = row["sample_id"]
                value["ordinal"] = row["ordinal"]
                value["generator/source"] = row.get("generator/source")
                value["artifact_categories"] = row.get("artifact_categories")
                records_out.append(value)
            atomic_jsonl(prediction_path(model, dataset_name), records_out)
            write_result(
                model, dataset_name, records_out,
                execution="RUN_NEW", inference_condition="G1",
                source_artifact=str(work_path.resolve()), source_artifact_sha256=sha256_file(work_path),
                notes="Known-Fake G1 canonical unified prompt with structural [FAKE] continuation prefix; no GT explanation or localization phrase is supplied.",
                extra={"generation_mode": "unified_prompt_gt_fake_prefix", "mask_logit_threshold": 0.0},
            )
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update({"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error)})
        raise
    finally:
        state["updated_at_utc"] = now()
        atomic_json(worker_status, state)


def run_legion(model_name: str, dataset_name: str, device_name: str) -> None:
    import argparse as _argparse
    import cv2
    import numpy as np
    import torch
    from scripts import phase5a2_legion_shared_evaluate as legacy

    if model_name not in ("legion_intermediate", "legion_retrained"):
        raise ValueError(model_name)
    if dataset_name not in ("xaigd", "pal4vst"):
        raise ValueError(dataset_name)
    if git_head(LEGION_REPO) != LEGION_COMMIT:
        raise RuntimeError("official LEGION source commit drift")
    model_dir = Path(checkpoint_identity(model_name)["path"])
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    manifest_rows, _ = manifest_info(dataset_name)
    gt = GroundTruthProvider(dataset_name)

    dest = OUT / model_name / dataset_name
    pred_path = dest / "predictions.jsonl"
    old = [enforce_failure_policy(row) for row in rows(pred_path)]
    indexed = {str(row["sample_id"]): row for row in old}
    expected_ids = [str(row["sample_id"]) for row in manifest_rows]
    if not set(indexed).issubset(set(expected_ids)):
        raise RuntimeError(f"{model_name}/{dataset_name} resume scope drift")

    worker_path = dest / "worker_status.json"
    state = {
        "schema": "final_eval_legion_localization_worker_v1", "status": "RUNNING",
        "model": model_name, "dataset": dataset_name, "inference_condition": "L-FREE",
        "pid": os.getpid(), "started_at_utc": now(), "device": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_dir": str(model_dir.resolve()), "legion_source_commit": LEGION_COMMIT,
        "manifest_sha256": sha256_file(MANIFESTS[dataset_name]),
    }
    atomic_json(worker_path, state)
    try:
        official, model, tokenizer, clip_processor, transform, prompt = legacy.load_model(
            LEGION_REPO.resolve(), model_dir.resolve(), CLIP_PATH.resolve(), device
        )
        inference_args = _argparse.Namespace(conv_type="llava_v1", use_mm_start_end=True, image_size=1024)
        original_evaluate = type(model).evaluate
        observed: list[dict] = []

        def observe_evaluate(instance, *args, **kwargs):
            generated_ids, pred_masks = original_evaluate(instance, *args, **kwargs)
            observed.append({
                "seg_count": int((generated_ids == instance.seg_token_idx).sum().item()),
                "generated_shape": list(generated_ids.shape),
                "pred_mask_shapes": [list(mask.shape) for mask in pred_masks],
            })
            return generated_ids, pred_masks

        type(model).evaluate = observe_evaluate
        try:
            for ordinal, row in enumerate(manifest_rows):
                sid = str(row["sample_id"])
                if sid in indexed:
                    continue
                bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise OSError(row["image_path"])
                height, width = bgr.shape[:2]
                target = gt.mask(row, height, width)
                observed.clear()
                explanation, pred_masks, phrases = official.inference(
                    model, prompt, bgr, tokenizer, clip_processor, transform, inference_args
                )
                if len(observed) != 1:
                    raise RuntimeError(f"LEGION evaluate observation mismatch for {sid}: {len(observed)}")
                observation = observed[0]
                masks = (
                    pred_masks[0].detach().float().cpu().numpy()
                    if pred_masks and pred_masks[0].numel()
                    else np.empty((0, height, width), dtype=np.float32)
                )
                if masks.ndim != 3 or (masks.shape[0] and tuple(masks.shape[-2:]) != (height, width)):
                    status = "INVALID_SHAPE"
                    binary = np.zeros((height, width), dtype=bool)
                elif masks.shape[0] == 0:
                    status = "EMPTY_MASK" if observation["seg_count"] else "NO_SEG"
                    binary = np.zeros((height, width), dtype=bool)
                else:
                    status = "OK" if observation["seg_count"] else "MASK_WITHOUT_SEG"
                    binary = np.any(masks > 0.0, axis=0)
                value = enforce_failure_policy({
                    "sample_id": sid, "ordinal": ordinal, "status": status,
                    "seg_count": observation["seg_count"], "has_pred_mask": bool(masks.shape[0]),
                    "explanation": explanation, "generated_phrases": phrases,
                    "generator/source": row.get("generator/source"),
                    "artifact_categories": row.get("artifact_categories"),
                    "gt_foreground_pixels": int(np.asarray(target, dtype=bool).sum()),
                    "gt_is_empty": not bool(np.asarray(target, dtype=bool).any()),
                    **metric_from_binary(binary, target),
                })
                append_jsonl(pred_path, value)
                indexed[sid] = value
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(manifest_rows):
                    print(json.dumps({"worker": model_name, "dataset": dataset_name, "done": ordinal + 1, "total": len(manifest_rows)}), flush=True)
        finally:
            type(model).evaluate = original_evaluate

        ordered = [indexed[sid] for sid in expected_ids]
        if len(ordered) != len(expected_ids):
            raise RuntimeError(f"{model_name}/{dataset_name} incomplete")
        atomic_jsonl(pred_path, ordered)
        coverage = {
            "valid_seg_and_mask": sum(row["status"] == "OK" and row["seg_count"] > 0 and row["has_pred_mask"] for row in ordered),
            "status_counts": dict(sorted(Counter(str(row["status"]) for row in ordered).items())),
        }
        write_result(
            model_name, dataset_name, ordered,
            execution="RUN_NEW", inference_condition="L-FREE",
            source_artifact=str(pred_path.resolve()), source_artifact_sha256=sha256_file(pred_path),
            notes="Official LEGION image-only L-FREE prompt; free generation; all [SEG]-conditioned masks unioned at original resolution; mask logit > 0.",
            extra={"coverage": coverage, "legion_source_commit": LEGION_COMMIT, "mask_logit_threshold": 0.0},
        )
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update({"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error)})
        raise
    finally:
        state["updated_at_utc"] = now()
        atomic_json(worker_path, state)


def result_is_current(model: str, dataset: str, inference_condition: str) -> bool:
    path = result_path(model, dataset)
    if not path.is_file():
        return False
    try:
        value = read_json(path)
        _, manifest = manifest_info(dataset)
        return (
            value.get("status") == "COMPLETE"
            and value.get("inference_condition") == inference_condition
            and value.get("failure_policy", {}).get("version") == FAILURE_POLICY_VERSION
            and value.get("manifest", {}).get("sha256") == manifest["sha256"]
            and value.get("manifest", {}).get("n") == manifest["n"]
        )
    except Exception:
        return False


def gpu_free(gpu: int, limit_mib: int = 2048) -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return int(out) < limit_mib
    except Exception:
        return False


def wait_gpu(state: dict) -> None:
    while not gpu_free(PHYSICAL_GPU):
        state.update({"status": "WAITING", "stage": f"gpu_{PHYSICAL_GPU}"})
        atomic_json(STATUS, state)
        time.sleep(60)


def run_subprocess(command: list[str], log_name: str, state: dict, stage: str) -> None:
    wait_gpu(state)
    state.update({"status": "RUNNING", "stage": stage, "command": command})
    atomic_json(STATUS, state)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (LOGS / log_name).open("a", buffering=1, encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def preflight() -> dict:
    for path in (DATA_GATE, LEAKAGE_AUDIT, LEAKAGE_OVERRIDE):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = read_json(DATA_GATE)
    audit = read_json(LEAKAGE_AUDIT)
    override = read_json(LEAKAGE_OVERRIDE)
    if gate.get("status") != "COMPLETE":
        raise RuntimeError(f"data gate is not COMPLETE: {gate}")
    if audit.get("status") == "BLOCKED_OVERLAP":
        if override.get("status") != "ACTIVE":
            raise RuntimeError("leakage audit is BLOCKED_OVERLAP and no ACTIVE override exists")
    elif audit.get("status") not in ("PASS", "COMPLETE"):
        raise RuntimeError(f"unexpected leakage audit status: {audit.get('status')}")
    manifests = {}
    for dataset in DATASETS:
        _, manifests[dataset] = manifest_info(dataset)
    xaigd_annotations = _xaigd_annotation_map()
    xaigd_empty_gt = sum(not (row.get("labels") or []) for row in xaigd_annotations.values())
    pal_rows, _ = manifest_info("pal4vst")
    pal_gt = GroundTruthProvider("pal4vst")
    pal_empty_gt = 0
    import cv2
    for row in pal_rows:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(row["image_path"])
        height, width = image.shape[:2]
        pal_empty_gt += int(not pal_gt.mask(row, height, width).any())
    checkpoint_identity("p1"); checkpoint_identity("r1")
    checkpoint_identity("legion_intermediate"); checkpoint_identity("legion_retrained")
    if git_head(LEGION_REPO) != LEGION_COMMIT:
        raise RuntimeError("official LEGION checkout commit drift")
    return {
        "data_gate": gate, "leakage_audit": audit, "leakage_override": override,
        "manifests": manifests,
        "xaigd_empty_label_gt_images": xaigd_empty_gt,
        "xaigd_empty_label_policy": "official labels=[] retained as all-zero GT",
        "pal4vst_empty_gt_images": pal_empty_gt,
        "pal4vst_empty_gt_policy": "official all-zero masks retained in the full test split",
        "failure_policy_version": FAILURE_POLICY_VERSION,
    }


def run_supervisor() -> None:
    state = {
        "schema": "final_eval_localization_supervisor_v2",
        "status": "RUNNING", "stage": "preflight", "pid": os.getpid(),
        "started_at_utc": now(), "physical_gpu": PHYSICAL_GPU,
        "protocol": {
            "synthscars": {"p1_r1": "historical_G0", "legion": "historical_L-FREE"},
            "loki": {"p1_r1": "historical_G1", "legion": "historical_L-FREE"},
            "xaigd": {"p1_r1": "fresh_G1", "legion": "fresh_L-FREE"},
            "pal4vst": {"p1_r1": "fresh_G1", "legion": "fresh_L-FREE"},
        },
    }
    atomic_json(STATUS, state)
    try:
        state["preflight"] = preflight()
        state.update({"stage": "historical_reuse"}); atomic_json(STATUS, state)
        materialize_historical_reuse()

        script = Path(__file__).resolve()
        for dataset in ("xaigd", "pal4vst"):
            if not (result_is_current("p1", dataset, "G1") and result_is_current("r1", dataset, "G1")):
                run_subprocess(
                    [str(GLAMM_PYTHON), str(script), "--worker", "glamm_pair", "--dataset", dataset, "--device", "cuda:0"],
                    f"localization_glamm_pair_{dataset}.log", state, f"glamm_pair_{dataset}",
                )
            for model in ("legion_intermediate", "legion_retrained"):
                if result_is_current(model, dataset, "L-FREE"):
                    continue
                run_subprocess(
                    [str(LEGION_PYTHON), str(script), "--worker", "legion", "--model", model, "--dataset", dataset, "--device", "cuda:0"],
                    f"localization_{model}_{dataset}.log", state, f"{model}_{dataset}",
                )

        missing = []
        expected_conditions = {
            "synthscars": {"p1": "G0", "r1": "G0", "legion_intermediate": "L-FREE", "legion_retrained": "L-FREE"},
            "loki": {"p1": "G1", "r1": "G1", "legion_intermediate": "L-FREE", "legion_retrained": "L-FREE"},
            "xaigd": {"p1": "G1", "r1": "G1", "legion_intermediate": "L-FREE", "legion_retrained": "L-FREE"},
            "pal4vst": {"p1": "G1", "r1": "G1", "legion_intermediate": "L-FREE", "legion_retrained": "L-FREE"},
        }
        for dataset in DATASETS:
            for model in MODELS:
                if not result_is_current(model, dataset, expected_conditions[dataset][model]):
                    missing.append(f"{model}/{dataset}")
        if missing:
            raise RuntimeError(f"localization matrix incomplete: {missing}")
        state.update({"status": "COMPLETE", "stage": "complete", "matrix_cells": 16, "updated_at_utc": now()})
        atomic_json(STATUS, state)
    except BaseException as error:
        state.update({"status": "FAILED", "stage": state.get("stage"), "exception_type": type(error).__name__, "exception": str(error), "updated_at_utc": now()})
        atomic_json(STATUS, state)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("glamm_pair", "legion"), default=None)
    parser.add_argument("--model", choices=("legion_intermediate", "legion_retrained"), default=None)
    parser.add_argument("--dataset", choices=("xaigd", "pal4vst"), default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.worker is None:
        run_supervisor()
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required in worker mode")
    if args.worker == "glamm_pair":
        run_glamm_pair(args.dataset, args.device)
    else:
        if args.model is None:
            raise SystemExit("--model is required for legion worker")
        run_legion(args.model, args.dataset, args.device)


if __name__ == "__main__":
    main()
