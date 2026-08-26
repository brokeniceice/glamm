#!/usr/bin/env python3
"""Freeze Phase 4C-A protocol and recover the exact Phase 3C.1 CLIP source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4c_a import cache_paths, dump, load_shard


def main():
    config_path = ROOT / "configs/phase4c_a_clip_forensic_adapter.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["experiment"]["output_root"]
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    output.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    source = ROOT / config["source"]["phase3c1_root"]
    audit = json.loads((source / "audit/clip.json").read_text())
    train_complete = json.loads((source / "cache/clip/train/complete.json").read_text())
    val_complete = json.loads((source / "cache/clip/val/complete.json").read_text())
    raw_metrics = json.loads((source / "validation/clip/metrics.json").read_text())["all_val_fake"]
    raw_checkpoint = torch.load(source / "probes/clip/selected.pt", map_location="cpu")
    train_paths = cache_paths(source, "train"); val_paths = cache_paths(source, "val")
    first = load_shard(train_paths[0]); first_val = load_shard(val_paths[0])
    train_ids=[]; val_ids=[]
    for path in train_paths: train_ids += [r["sample_id"] for r in load_shard(path)["records"]]
    for path in val_paths: val_ids += [r["sample_id"] for r in load_shard(path)["records"]]
    checks = {
        "audit_pass": audit["status"] == "PASS",
        "feature_shape_exact": audit["feature_shape"] == [1, 1024, 24, 24],
        "train_count_exact": len(train_ids) == 8836 == train_complete["samples"],
        "val_count_exact": len(val_ids) == 1106 == val_complete["samples"],
        "unique_train_ids": len(set(train_ids)) == len(train_ids),
        "unique_val_ids": len(set(val_ids)) == len(val_ids),
        "train_val_disjoint": not set(train_ids).intersection(val_ids),
        "preprocessing_exact": train_complete["preprocessing_sha256"] == val_complete["preprocessing_sha256"] == config["source"]["preprocessing_sha256"],
        "clip_hash_exact": train_complete["source_parameter_hash_before"] == train_complete["source_parameter_hash_after"] == val_complete["source_parameter_hash_before"] == val_complete["source_parameter_hash_after"] == config["source"]["clip_parameter_hash"],
        "p1_hash_exact": train_complete["source_checkpoint_sha256"] == val_complete["source_checkpoint_sha256"] == config["source"]["p1_checkpoint_sha256"],
        "raw_probe_shape_exact": tuple(raw_checkpoint["state_dict"]["weight"].shape) == (1,1024,1,1),
        "target_shape_exact": list(first["targets"].shape[1:]) == [336,336],
        "val_original_masks_present": "original_masks" in first_val,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CLIP feature audit failed: {[k for k,v in checks.items() if not v]}")
    spec = {
        "status": "PASS", "recovered_from_code_and_artifacts": True,
        "clip_checkpoint": "openai/clip-vit-large-patch14-336 as embedded in canonical P1",
        "p1_checkpoint": train_complete["source_checkpoint"],
        "p1_checkpoint_sha256": train_complete["source_checkpoint_sha256"],
        "vision_tower_module": "model.get_model().get_vision_tower().vision_tower",
        "exact_vision_layer": -2,
        "layer_evidence": "train.py argparse default mm_vision_select_layer=-2; CLIPVisionTower.feature_select uses hidden_states[self.select_layer]",
        "cls_token_handling": "removed by feature_select via image_features[:, 1:] (select_feature=patch)",
        "spatial_token_extraction": "576 patch tokens transposed from [B,N,C] and reshaped to [B,C,24,24]",
        "token_count": 576, "channel_dimension": 1024, "spatial_grid": [24,24],
        "probe_input_tensor": {"shape": ["B",1024,24,24], "cached_dtype": "torch.bfloat16", "training_cast": "torch.float32"},
        "image_preprocessing": train_complete["preprocessing_version"],
        "normalization": "CLIPImageProcessor model-defined rescale and normalize",
        "forward_interpolation": "resize shortest edge to 336 then center crop 336x336",
        "target_resize": "nearest-neighbor application of the same resize-shortest and center-crop geometry to the official annotation-derived all-reference union target",
        "evaluation_resize": "bilinear logits to 336 crop; paste into resized canvas with exterior logit -100; bilinear resize to original image",
        "mask_threshold": "original-space logit > 0",
        "validation_sample_count": len(val_ids), "validation_sample_ids": val_ids,
        "validation_sample_ids_sha256": val_complete["sample_ids_sha256"],
        "train_sample_count": len(train_ids), "train_sample_ids_sha256": train_complete["sample_ids_sha256"],
        "source_parameter_hash": train_complete["source_parameter_hash_before"],
        "checks": checks,
    }
    dump(output / "audits/clip_feature_spec.json", spec); dump(output / "clip_feature_spec.json", spec)
    dataset = {
        "status":"FROZEN", "train":{"population":"canonical internal train Fake", "n":len(train_ids), "sample_ids_sha256":train_complete["sample_ids_sha256"]},
        "validation":{"population":"canonical internal validation Fake", "n":len(val_ids), "sample_ids_sha256":val_complete["sample_ids_sha256"]},
        "target":"official annotation-derived per-image all-reference union evidence region",
        "target_is_pseudo_mask":False, "internal_test_loaded":False, "official1000_loaded":False,
    }
    dump(output / "dataset_manifest.json", dataset)
    architecture = {
        "arm_a":"frozen raw CLIP 1024x24x24 -> Conv2d(1024,1,1,bias=True), historical exact reuse",
        "arm_b":"frozen raw CLIP -> Conv2d(1024,256,1) -> Conv2d(256,1,1,bias=True)",
        "arm_c":"same projection -> 3 x [GN32,DWConv3x3,GELU,PWConv1x1,residual] -> same dense head",
        "forbidden_components_present":False, "CLIP_trainable":False,
    }
    dump(output / "architecture_manifest.json", architecture)
    dump(output / "training_config.json", {**config["training"], "optimizer":config["optimizer"], "loss":config["loss"], "selector":config["selector"], "frozen_before_results":True})
    experiment = {"status":"PREPARED", "name":config["experiment"]["name"], "seed":config["experiment"]["seed"], "phase3c1_raw_baseline_reused":True,
                  "raw_baseline_selected_epoch":raw_checkpoint["epoch"], "raw_baseline_selector":list(raw_checkpoint["selector"]),
                  "internal_test_access":False,"official1000_access":False,"threshold_tuning":False}
    dump(output / "experiment_manifest.json", experiment)
    raw_records = [json.loads(line) for line in (source / "validation/clip/normal_predictions.jsonl").read_text().splitlines() if line]
    # Add metrics required by Phase 4C-A while preserving the historical values.
    tp=sum(r["tp"] for r in raw_records); fp=sum(r["fp"] for r in raw_records); fn=sum(r["fn"] for r in raw_records)
    raw = {**raw_metrics, "global_foreground_iou":tp/(tp+fp+fn), "global_foreground_f1":2*tp/(2*tp+fp+fn),
           "threshold_logit":0.0,"artifact_reused":True,"selected_epoch":raw_checkpoint["epoch"],
           "source_predictions":str((source/"validation/clip/normal_predictions.jsonl").resolve())}
    dump(output / "raw_clip_metrics.json", raw)
    fairness = {"status":"FROZEN_PENDING_RUNTIME_HASH_CHECK", "seed":config["experiment"]["seed"],
                "common_projection_initialization":True,"common_dense_head_initialization":True,
                "same_sample_order":True,"same_batch_composition":True,"same_optimizer":True,"same_scheduler":True,
                "only_core_difference":"Arm C adds exactly three local forensic residual blocks"}
    dump(output / "fairness_manifest.json", fairness)
    report = f"""# Phase 4C-A exact CLIP feature audit

The matched Phase 3C.1 source is the frozen CLIP vision tower embedded in canonical P1 (`openai/clip-vit-large-patch14-336`, P1 SHA256 `{spec['p1_checkpoint_sha256']}`). The tower selects hidden state **-2**, removes the CLS token, and reshapes 576 patch tokens into a **1024 x 24 x 24** spatial tensor.

The image path uses `CLIPImageProcessor`: resize the shortest edge to 336, center-crop 336, rescale, and CLIP-normalize. The official annotation-derived per-image all-reference union mask follows the same geometry with nearest-neighbor interpolation. Evaluation bilinearly upsamples logits into the crop, fills unseen crop exterior with background logit -100, resizes to original resolution, and thresholds at logit 0.

Train and validation caches contain {len(train_ids)} and {len(val_ids)} unique Fake samples. Validation identity SHA256 is `{val_complete['sample_ids_sha256']}`. Frozen CLIP parameter hash is `{spec['source_parameter_hash']}` before and after extraction. The historical raw linear probe is therefore an exact protocol match and is reused without retraining.
"""
    (ROOT / "docs/phase4c_a_clip_feature_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status":"PASS","feature_shape":[1024,24,24],"train":len(train_ids),"val":len(val_ids),"raw_mean_iou":raw["mean_foreground_iou"]},indent=2))


if __name__ == "__main__": main()

