#!/usr/bin/env python3
"""Prepare frozen manifests, provenance, and preregistered reward definitions for Phase 3D.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3d0 import canonical_json_sha256, deterministic_stratified_subset
from tools.phase3b_replay import file_sha256


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d0_reward_preflight.yaml")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_record(row: dict, dataset_index: int, population: str) -> dict:
    return {
        "sample_id": row["sample_id"], "dataset_index": int(dataset_index),
        "population": population, "class_label": int(row["class_label"]),
        "forensics_domain": row["forensics_domain"], "source": row["source"],
        "content_category": row.get("content_category"),
        "image_sha256": row.get("image_sha256"),
        "annotation_relpath": row.get("annotation_relpath"),
    }


def ordered_ids_sha256(records: list[dict]) -> str:
    return hashlib.sha256("".join(f"{row['sample_id']}\n" for row in records).encode()).hexdigest()


def distribution(records: list[dict]) -> dict:
    result = {}
    for field in ("class_label", "source", "content_category"):
        values = {}
        for row in records:
            key = str(row.get(field))
            values[key] = values.get(key, 0) + 1
        result[field] = dict(sorted(values.items()))
    return result


def main(argv=None):
    cli = args(argv)
    config_path = (ROOT / cli.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["experiment"]["output_root"]).resolve()
    tensor_root = Path(cfg["experiment"]["tensor_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True); tensor_root.mkdir(parents=True, exist_ok=True)
    masks_link = output / "masks"
    # Repository ``outputs`` may itself be the canonical /data symlink.  In
    # that topology output and tensor_root resolve to the same directory, so a
    # second link would be self-referential.
    if output == tensor_root:
        masks_link.mkdir(parents=True, exist_ok=True)
    else:
        (tensor_root / "masks").mkdir(parents=True, exist_ok=True)
        if not masks_link.exists() and not masks_link.is_symlink():
            masks_link.symlink_to(tensor_root / "masks", target_is_directory=True)
    for dirname in (
        "audit", "manifests", "sampling", "greedy", "rollouts", "reward_components",
        "reward_candidates", "ranking", "diversity", "statistics", "human_review",
        "qualitative", "reports",
    ):
        (output / dirname).mkdir(parents=True, exist_ok=True)

    selector_path = (ROOT / cfg["source"]["selector"]).resolve()
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    checkpoint = Path(selector["selected_checkpoint"]).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    expected = cfg["source"]["checkpoint_sha256"]
    if checkpoint_hash != expected or checkpoint_hash != selector["checkpoint_sha256"]:
        raise RuntimeError(f"P1 checkpoint SHA mismatch: {checkpoint_hash}")
    if int(selector["optimizer_step"]) != 3500 or int(selector["epoch"]) != 7:
        raise RuntimeError("P1 selector is not step 3500 / epoch 7")

    manifest_dir = (ROOT / cfg["data"]["manifest_dir"]).resolve()
    train_rows = load_jsonl(manifest_dir / "train_combined.jsonl")
    val_rows = load_jsonl(manifest_dir / "val_combined.jsonl")
    prohibited_files = [
        manifest_dir / "test_combined.jsonl", manifest_dir / "official_synthscars_test.jsonl",
        manifest_dir / "raise_heldout_real_clean.jsonl",
    ]
    prohibited_ids = {row["sample_id"] for path in prohibited_files if path.exists() for row in load_jsonl(path)}
    dev_real = deterministic_stratified_subset(train_rows, count=512, seed=3407, label=0)
    dev_fake = deterministic_stratified_subset(train_rows, count=512, seed=3407, label=1)
    dev = [manifest_record(row, row["dataset_index"], "reward_dev") for row in dev_real + dev_fake]
    dev.sort(key=lambda row: hashlib.sha256(f"3407:dev-order:{row['sample_id']}".encode()).hexdigest())
    val = [manifest_record(row, index, "reward_confirmation") for index, row in enumerate(val_rows)]
    if len(dev) != 1024 or sum(row["class_label"] == 0 for row in dev) != 512:
        raise RuntimeError("reward-dev population mismatch")
    if len(val) != 2212 or sum(row["class_label"] == 0 for row in val) != 1106:
        raise RuntimeError("reward-validation population mismatch")
    if {row["sample_id"] for row in dev + val} & prohibited_ids:
        raise RuntimeError("prohibited test/external sample entered Phase 3D.0")
    reward_dev_manifest = {
        "status": "FROZEN", "seed": 3407, "population": "internal_train_only",
        "stratification": "deterministic proportional content_category_x_source within class",
        "count": len(dev), "ordered_ids_sha256": ordered_ids_sha256(dev),
        "distribution": distribution(dev), "records": dev,
    }
    dump(output / "manifests/reward_dev_manifest.json", reward_dev_manifest)
    dump(output / "reward_dev_manifest.json", reward_dev_manifest)
    reward_val_manifest = {
        "status": "FROZEN", "population": "complete_internal_validation",
        "count": len(val), "ordered_ids_sha256": ordered_ids_sha256(val),
        "distribution": distribution(val), "records": val,
    }
    dump(output / "manifests/reward_val_manifest.json", reward_val_manifest)
    dump(output / "reward_val_manifest.json", reward_val_manifest)

    synth_root = Path(cfg["data"]["synthscars_root"])
    if not synth_root.is_absolute(): synth_root = (ROOT / synth_root).resolve()
    train_ann = synth_root / "train/annotations/train.json"
    test_ann = synth_root / "test/annotations/test.json"
    sample_fake = next(row for row in train_rows if int(row["class_label"]) == 1)
    refs_fields = sorted(sample_fake["refs"][0])
    mask_provenance = {
        "status": "FROZEN_BEFORE_ROLLOUTS",
        "dataset_name": "LEGION / SynthScars",
        "annotation_source_paths": {"train": str(train_ann), "test": str(test_ann)},
        "annotation_sha256": {"train": file_sha256(train_ann), "test": file_sha256(test_ann)},
        "annotation_schema": "annotation-centric JSON; outer annotation id -> img_file_name, caption, refs[]",
        "official_ref_fields": ["sentence", "explanation", "bbox", "segmentation"],
        "manifest_ref_fields_after_lossless_adapter": refs_fields,
        "exact_segmentation_field_used": "refs[].segmentation polygons, retained as refs[].polygons",
        "target_construction": "rasterize each official ref polygon after source-to-canonical coordinate scaling; boolean OR all ref masks per image",
        "derived_union_from_official_synthscars_annotations": True,
        "multiple_official_ref_masks_unioned": True,
        "resize_coordinate_handling": "polygons_for_target scales source_image_size coordinates to canonical decoded image size before COCO rasterization",
        "evaluator_target_representation": "single per-image all-ref binary union target",
        "legion_paper_evaluation_unit_exact_parity_proven": False,
        "forbidden_transformations": ["erosion", "dilation", "boundary_refinement", "new_masks", "relabel", "threshold_tuning"],
        "mask_logit_threshold": "> 0",
        "implementation": {
            "adapter": "dataset/forensics/synthscars.py",
            "dataset_target": "UnifiedForensicsDataset._fake_union_mask",
        },
    }
    dump(output / "audit/mask_provenance.json", mask_provenance)
    dump(output / "mask_provenance.json", mask_provenance)
    reward_definition = {
        "status": "FROZEN_BEFORE_FULL_VALIDATION",
        "components": {
            "R_cls": "generated LM verdict equals frozen GT class label",
            "R_struct": "emitted-structure-only binary validity",
            "R_phrase_lex": "deterministic normalized-token F1; lexical agreement, not semantic correctness",
            "R_mask": "trajectory mask per-image FG IoU against frozen derived union target; logit > 0",
            "R_align": "2*R_phrase_lex*R_mask/(R_phrase_lex+R_mask+eps)",
        },
        "candidates": cfg["reward"]["candidates"],
        "fake_weights": cfg["reward"]["fake_weights"],
        "real_weights": cfg["reward"]["real_weights"],
        "free_form_explanation_reward_included": False,
        "grid_search_allowed": False,
        "definition_sha256": None,
    }
    reward_definition["definition_sha256"] = canonical_json_sha256({k: v for k, v in reward_definition.items() if k != "definition_sha256"})
    dump(output / "reward_definition.json", reward_definition)
    provenance = {
        "phase": "Phase 3D.0 — Evidence-Aware Reward / Rollout Preflight",
        "status": "PREPARED", "read_only_no_training": True,
        "config": str(config_path), "config_sha256": file_sha256(config_path),
        "selector": str(selector_path), "selector_sha256": file_sha256(selector_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "optimizer_step": 3500, "logical_epoch": 7,
        "manifest_dir": str(manifest_dir), "tensor_root": str(tensor_root),
        "allowed_route_inputs": ["internal_train_reward_dev", "complete_internal_validation"],
        "prohibited_route_inputs": ["internal_test", "official_synthscars1000", "RAISE", "LOKI", "FakeBench", "external_test"],
        "no_optimizer": True, "no_scheduler": True, "no_backward": True,
        "no_model_update": True, "phase3d1_started": False,
        "prepared_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    dump(output / "provenance.json", provenance)
    dump(output / "audit/preparation_audit.json", {
        "status": "PASS", "dev_val_disjoint": not ({r["sample_id"] for r in dev} & {r["sample_id"] for r in val}),
        "prohibited_overlap_count": 0, "checkpoint_verified": True,
        "reward_definition_frozen": True, "mask_provenance_complete": True,
    })
    print(json.dumps({"status": "PREPARED", "dev": len(dev), "val": len(val), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
