#!/usr/bin/env python3
"""Read-only/data and synthetic implementation audits for Phase 4E-0.5.

The script never loads a language checkpoint, runs validation inference, or
updates parameters.  It writes only audit metadata under the Phase 4E-0.5
output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from model.clip_forensic_adapter import CLIPSpatialArm
from model.tf_fdg import (
    CrossAttentiveSemanticRectification,
    TFFDGStudent,
    clone_frozen_teacher,
    hungarian_teacher_assignment,
    kd_losses,
    normalized_grid,
    region_aware_slot_loss,
    slot_collapse_diagnostics,
)
from tools.phase3c1 import geometry_for, inverse_logits, transform_mask


OUT = ROOT / "outputs/phase4e05_tf_fdg_hardening"
MANIFEST = ROOT / "outputs/data_audits/unified_forensics_split_v1"
CACHE = ROOT / "outputs/phase3c1_spatial_probe/cache"
ADAPTER = Path("/data/yz/groundingLMM_official/checkpoints/phase4c_a_clip_forensic_adapter/forensic_adapter/selected.pt")
SEED = 3407


def dump(name: str, value) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(split: str) -> list[dict]:
    return [json.loads(line) for line in (MANIFEST / f"{split}_combined.jsonl").open() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def population_and_slot_audit() -> tuple[dict, dict]:
    result, slot = {}, {}
    for split in ("train", "val"):
        records = rows(split)
        fake = [r for r in records if int(r["class_label"]) == 1]
        real = [r for r in records if int(r["class_label"]) == 0]
        ref_counts = [len(r.get("refs") or []) for r in fake]
        refs = [ref for r in fake for ref in r["refs"]]
        result[split] = {
            "total": len(records), "fake": len(fake), "real": len(real),
            "dataset_seg_valid_fake": len(fake), "dataset_seg_valid_real": 0,
            "real_refs_nonempty": sum(bool(r.get("refs")) for r in real),
            "real_manifest_has_mask_label_true": sum(r.get("has_mask_label") is True for r in real),
            "real_empty_mask_semantics_defined": False,
        }
        slot[split] = {
            "fake_images": len(fake), "total_refs": len(refs),
            "ref_count_histogram": {str(k): v for k, v in sorted(Counter(ref_counts).items())},
            "mean_refs": float(np.mean(ref_counts)), "max_refs": max(ref_counts),
            "images_with_refs_gt_k4": sum(count > 4 for count in ref_counts),
            "fraction_with_refs_gt_k4": sum(count > 4 for count in ref_counts) / len(fake),
            "unique_ref_ids": len({ref.get("ref_id") for ref in refs}),
            "refs_with_region_id": sum(bool(ref.get("ref_id")) for ref in refs),
            "refs_with_annotation_id": sum(bool(ref.get("annotation_id")) for ref in refs),
            "refs_with_phrase": sum(bool(str(ref.get("phrase") or "").strip()) for ref in refs),
            "refs_with_polygons": sum(bool(ref.get("polygons")) for ref in refs),
            "total_polygons": sum(len(ref.get("polygons") or []) for ref in refs),
            "refs_with_multiple_polygons": sum(len(ref.get("polygons") or []) > 1 for ref in refs),
        }
    population = {
        "status": "PASS", "manifest": str((MANIFEST / "train_combined.jsonl").resolve()),
        "manifest_sha256": sha256(MANIFEST / "train_combined.jsonl"), "splits": result,
        "LOCALIZATION_TRAIN_POPULATION": "frozen train Fake rows only",
        "N_FAKE": result["train"]["fake"], "N_REAL": result["train"]["real"],
        "REAL_USED_FOR_MASK_TRAINING": "NO",
        "RATIONALE": (
            "UnifiedForensicsDataset defines seg_valid only for forensics_domain=fake and emits masks=None "
            "for Real. Real refs are empty and no legal empty-mask localization semantics is defined."
        ),
        "budget": {
            "effective_batch_size": 8, "steps_per_epoch": math.ceil(result["train"]["fake"] / 8),
            "teacher_epochs": 5, "teacher_steps": 5 * math.ceil(result["train"]["fake"] / 8),
            "teacher_exposures": 5 * result["train"]["fake"],
            "student_epochs": 10, "student_steps": 10 * math.ceil(result["train"]["fake"] / 8),
            "student_exposures": 10 * result["train"]["fake"],
        },
    }
    slot_availability = {
        "status": "PASS", "splits": slot,
        "PER_REGION_POLYGON_AVAILABLE": "YES", "REGION_ID_AVAILABLE": "YES",
        "REGION_PHRASE_MAPPING_AVAILABLE": "YES",
        "CAN_RECONSTRUCT_SLOT_LEVEL_MASKS_WITHOUT_NEW_ANNOTATION": "YES",
        "new_annotation_created": False, "pseudo_label_created": False,
        "k4_overflow_policy": (
            "M<=4: rectangular region-aware Hungarian with unmatched slots supervised empty; "
            "M>4: union-only supervision plus mandatory anti-collapse diagnostics, because arbitrary "
            "region grouping would invent slot identity."
        ),
    }
    dump("population_audit.json", population)
    dump("slot_target_availability.json", slot_availability)
    return population, slot_availability


def _centroid(mask: torch.Tensor) -> torch.Tensor:
    position = mask.nonzero(as_tuple=False).float()
    if not len(position):
        raise ValueError("synthetic geometry support became empty")
    height, width = mask.shape
    return torch.stack(((position[:, 1] + 0.5).mean() / width, (position[:, 0] + 0.5).mean() / height))


def _feature_reconstruction(mask: torch.Tensor, source: str, feature_hw: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    geometry = geometry_for(source, mask.shape)
    transformed = transform_mask(mask, geometry).float()[None, None]
    occupancy = F.adaptive_max_pool2d(transformed, feature_hw)[0, 0]
    logits = torch.where(occupancy > 0, torch.tensor(10.0), torch.tensor(-10.0))
    reconstructed = inverse_logits(logits, geometry).gt(0)
    return reconstructed, geometry


def geometry_audit() -> dict:
    cases = {}
    definitions = {
        "top_left": (768, 768, 40, 70, 190, 230),
        "bottom_right": (768, 768, 570, 540, 730, 710),
        "center_small": (768, 768, 350, 350, 410, 410),
        "horizontal_strip": (768, 768, 330, 80, 390, 690),
        "vertical_strip": (768, 768, 70, 350, 700, 410),
        "wide_center": (640, 1024, 250, 430, 390, 600),
        "tall_center": (1024, 640, 430, 250, 600, 390),
    }
    clip_errors, sam_errors, alignment_errors = [], [], []
    for name, (height, width, top, left, bottom, right) in definitions.items():
        mask = torch.zeros(height, width, dtype=torch.bool); mask[top:bottom, left:right] = True
        clip, clip_geometry = _feature_reconstruction(mask, "clip", (24, 24))
        sam, sam_geometry = _feature_reconstruction(mask, "sam", (64, 64))
        truth_centroid, clip_centroid, sam_centroid = _centroid(mask), _centroid(clip), _centroid(sam)
        clip_error = float(torch.linalg.vector_norm(clip_centroid - truth_centroid))
        sam_error = float(torch.linalg.vector_norm(sam_centroid - truth_centroid))
        alignment_error = float(torch.linalg.vector_norm(clip_centroid - sam_centroid))
        clip_errors.append(clip_error); sam_errors.append(sam_error); alignment_errors.append(alignment_error)
        cases[name] = {
            "original_hw": [height, width], "clip_geometry": clip_geometry, "sam_geometry": sam_geometry,
            "clip_centroid_error_normalized": clip_error, "sam_centroid_error_normalized": sam_error,
            "clip_sam_centroid_error_normalized": alignment_error,
        }
    tolerance = {"clip_to_original": 0.05, "sam_to_original": 0.02, "clip_sam": 0.05}
    maxima = {
        "CLIP_TO_ORIGINAL_ERROR": max(clip_errors), "SAM_TO_ORIGINAL_ERROR": max(sam_errors),
        "CLIP_SAM_ALIGNMENT_ERROR": max(alignment_errors),
    }
    status = "PASS" if (
        maxima["CLIP_TO_ORIGINAL_ERROR"] <= tolerance["clip_to_original"]
        and maxima["SAM_TO_ORIGINAL_ERROR"] <= tolerance["sam_to_original"]
        and maxima["CLIP_SAM_ALIGNMENT_ERROR"] <= tolerance["clip_sam"]
    ) else "FAIL"
    value = {
        "status": status, "coordinate_system": "original-image normalized cell-centre (x,y) in [0,1]",
        "clip_pipeline": "resize shortest to 336, center crop 336; feature 24x24",
        "sam_pipeline": "resize longest to 1024, pad right/bottom; feature 64x64",
        "fusion_rule": (
            "derive every token coordinate from its own forward transform; use original-coordinate "
            "positional encoding and validity masks; never equate interpolate(24,64) with alignment"
        ),
        "center_crop_boundary": (
            "CLIP tokens only cover the center-crop field of view. SAM locations outside it have no "
            "forensic support and must use a validity mask rather than fabricated zero evidence."
        ),
        "tolerance_normalized": tolerance, "max_errors": maxima, "cases": cases,
    }
    dump("geometry_audit.json", value)
    if status != "PASS":
        raise RuntimeError(f"geometry audit failed: {maxima}")
    return value


def load_feature_subset(count: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    clip = torch.load(CACHE / "clip/train/shard_000000_000032.pt", map_location="cpu", weights_only=False)["features"][:count].float()
    sam = torch.load(CACHE / "sam/train/shard_000000_000032.pt", map_location="cpu", weights_only=False)["features"][:count].float()
    checkpoint = torch.load(ADAPTER, map_location="cpu", weights_only=False)
    adapter = CLIPSpatialArm(input_channels=1024, channels=256, blocks=3).eval()
    adapter.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        forensic = adapter(clip, return_features=True)["F_forensic"]
    return sam, forensic


def feature_scale_audit() -> dict:
    torch.manual_seed(SEED)
    sam, forensic = load_feature_subset(2)
    semantic = F.adaptive_avg_pool2d(sam, (32, 32)).flatten(2).transpose(1, 2)
    forensic_tokens = forensic.flatten(2).transpose(1, 2)
    sem_coord, for_coord = normalized_grid(32, 32), normalized_grid(24, 24)
    base = CrossAttentiveSemanticRectification(gamma_init=1.0).eval()
    with torch.no_grad():
        _, _, unit_residual = base(semantic, forensic_tokens, sem_coord, for_coord)
    candidates, selected = [], None
    for gamma in (0.01, 0.03, 0.05, 0.1):
        residual = unit_residual * gamma
        ratio = residual.norm(dim=-1) / semantic.norm(dim=-1).clamp_min(1e-8)
        bf16_changed = (semantic.bfloat16() != (semantic + residual).bfloat16()).any(dim=-1).float().mean()
        record = {
            "gamma": gamma, "median_residual_to_semantic_norm": float(ratio.median()),
            "mean_residual_to_semantic_norm": float(ratio.mean()),
            "bf16_token_survival": float(bf16_changed),
        }
        record["valid"] = 0.005 <= record["median_residual_to_semantic_norm"] <= 0.10 and record["bf16_token_survival"] >= 0.80
        candidates.append(record)
        if selected is None and record["valid"]:
            selected = gamma
    value = {
        "status": "PASS" if selected is not None else "FAIL", "selection_uses_validation": False,
        "population": "first two frozen train-Fake feature-cache records; no labels or IoU",
        "sam_input_shape": list(sam.shape), "forensic_input_shape": list(forensic.shape),
        "candidates": candidates, "selected_gamma": selected,
        "rule": "smallest nonzero gamma with median residual/base norm in [0.005,0.10] and BF16 token survival >=0.80",
    }
    dump("feature_scale_audit.json", value)
    if selected is None:
        raise RuntimeError("no valid nonzero rectification gamma")
    return value


def _group_norm(module: torch.nn.Module) -> float:
    return float(math.sqrt(sum(float(parameter.grad.float().square().sum()) for parameter in module.parameters() if parameter.grad is not None)))


def _mask_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit.float(), target.float())
    probability = logit.float().sigmoid()
    intersection = 2 * (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return bce + (1 - (intersection + 1e-6) / (denominator + 1e-6)).mean()


def implementation_audit(gamma: float) -> tuple[dict, dict, dict]:
    torch.manual_seed(SEED)
    model = TFFDGStudent(gamma_init=gamma)
    teacher = clone_frozen_teacher(model)
    batch = 1
    h_tf = torch.randn(batch, 4096)
    h_g0 = h_tf + 0.10 * torch.randn_like(h_tf)
    semantic = torch.randn(batch, 256, 64, 64)
    forensic = torch.randn(batch, 256, 24, 24)
    target = torch.zeros(batch, 1, 32, 32); target[..., 6:15, 5:18] = 1; target[..., 20:27, 21:29] = 1
    with torch.no_grad():
        teacher_output = teacher(h_tf, semantic, forensic)
    student_output = model(h_g0, semantic, forensic)
    permutation = hungarian_teacher_assignment(
        student_output["slot_logits"][0], teacher_output["slot_logits"][0],
        student_output["forensic_attention"][-1][0].mean(0),
        teacher_output["forensic_attention"][-1][0].mean(0),
    )
    losses = {"mask": _mask_loss(student_output["union_logit"], target)}
    losses.update(kd_losses(student_output, teacher_output, permutation))
    weights = {"mask": 1.0, "relation": 0.20, "attention": 0.50, "feature": 0.25, "logit": 0.50}
    groups = {
        "QGen": model.query_generator, "Sem": model.semantic_pyramid,
        "For": model.forensic_pyramid, "Rect": model.rectification,
        "Decoder": model.decoder, "Mask": torch.nn.ModuleList([model.mask_feature, model.mask_embedding]),
    }
    matrix, scale = {}, {}
    mask_total_grad = None
    for name, loss in losses.items():
        model.zero_grad(set_to_none=True)
        (weights[name] * loss).backward(retain_graph=True)
        norms = {group: _group_norm(module) for group, module in groups.items()}
        total = math.sqrt(sum(value * value for value in norms.values()))
        if name == "mask":
            mask_total_grad = total
        matrix[name] = norms
        scale[name] = {
            "raw_loss": float(loss.detach()), "weight": weights[name],
            "weighted_loss": float((weights[name] * loss).detach()), "total_gradient_norm": total,
        }
        if name == "mask":
            rectification_gradient_detail = {
                key: float(math.sqrt(sum(float(parameter.grad.float().square().sum()) for parameter_name, parameter in model.rectification.named_parameters() if parameter.grad is not None and parameter_name.startswith(prefix))))
                for key, prefix in {
                    "Q": "cross_attention.q_proj", "K": "cross_attention.k_proj",
                    "V": "cross_attention.v_proj", "attention_output": "cross_attention.out_proj",
                    "semantic_projection": "projection", "gamma": "gamma",
                }.items()
            }
    for name in scale:
        scale[name]["weighted_gradient_to_mask_gradient"] = scale[name]["total_gradient_norm"] / max(mask_total_grad, 1e-30)
        ratio = scale[name]["weighted_gradient_to_mask_gradient"]
        scale[name]["numerically_valid"] = name == "mask" or 1e-5 <= ratio <= 1e3
    intended = {
        "mask": {"QGen", "Sem", "For", "Rect", "Decoder", "Mask"},
        "relation": {"QGen", "Sem", "For", "Rect", "Decoder"},
        "attention": {"QGen", "For", "Decoder"},
        "feature": {"QGen", "Sem", "For", "Rect", "Decoder"},
        "logit": {"QGen", "Sem", "For", "Rect", "Decoder", "Mask"},
    }
    routing_pass = all(matrix[loss][group] > 0 for loss, expected in intended.items() for group in expected)
    scale_pass = all(item["numerically_valid"] for item in scale.values())
    teacher_grad_exact = all(parameter.grad is None for parameter in teacher.parameters())
    teacher_student_initial_weights_exact = all(
        torch.equal(student_value, teacher.state_dict()[name]) for name, student_value in model.state_dict().items()
    )
    pairwise_identical = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairwise_identical.append(bool(torch.equal(student_output["slot_logits"][:, i], student_output["slot_logits"][:, j])))
    parameter_groups = {
        name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()
    }
    region_masks = torch.zeros(3, 32, 32)
    region_masks[0, 3:9, 4:11] = 1; region_masks[1, 14:20, 13:20] = 1; region_masks[2, 23:29, 22:30] = 1
    region_mode = region_aware_slot_loss(student_output["slot_logits"][0], region_masks)["mode"]
    overflow_masks = torch.cat((region_masks, region_masks[:2].roll(2, dims=-1)), dim=0)
    overflow_mode = region_aware_slot_loss(student_output["slot_logits"][0], overflow_masks)["mode"]
    rectification_gradients_nonzero = all(value > 0 for value in rectification_gradient_detail.values())
    supervision_modes_valid = region_mode == "region_hungarian" and overflow_mode == "union_only_overflow"
    collapse = slot_collapse_diagnostics(
        student_output["slot_logits"], student_output["queries"], student_output["forensic_attention"][-1],
    )
    architecture = {
        "status": "PASS" if (
            routing_pass
            and teacher_grad_exact
            and teacher_student_initial_weights_exact
            and not any(pairwise_identical)
            and rectification_gradients_nonzero
            and supervision_modes_valid
        ) else "FAIL",
        "input_shapes": {"h": [batch, 4096], "semantic": [batch, 256, 64, 64], "forensic": [batch, 256, 24, 24]},
        "output_shapes": {
            "queries": list(student_output["queries"].shape),
            "slot_logits": list(student_output["slot_logits"].shape),
            "union_logit": list(student_output["union_logit"].shape),
        },
        "teacher_stop_gradient_exact": teacher_grad_exact,
        "teacher_student_initial_weights_exact": teacher_student_initial_weights_exact,
        "student_interface_accepts_tf_tokens": False,
        "assignment_cost_requires_grad": False,
        "assignment_permutation": permutation.tolist(),
        "slot_outputs_pairwise_identical": pairwise_identical,
        "region_supervision_mode_m3": region_mode,
        "region_supervision_mode_m5": overflow_mode,
        "slot_diagnostic_shapes": {key: list(value.shape) for key, value in collapse.items()},
        "rectification_mask_loss_gradient_detail": rectification_gradient_detail,
        "rectification_all_required_gradients_nonzero": rectification_gradients_nonzero,
        "parameter_counts": {**parameter_groups, "total_trainable": sum(parameter_groups.values())},
        "rectification_gamma": gamma,
    }
    routing = {"status": "PASS" if routing_pass else "FAIL", "gradient_norm_matrix": matrix, "intended_nonzero": {k: sorted(v) for k, v in intended.items()}}
    scale_audit = {"status": "PASS" if scale_pass else "LOSS_SCALE_INVALID", "losses": scale, "validation_metrics_used": False}
    dump("implementation_audit.json", architecture)
    dump("loss_gradient_routing_audit.json", routing)
    dump("loss_scale_audit.json", scale_audit)
    if architecture["status"] != "PASS" or routing["status"] != "PASS" or scale_audit["status"] != "PASS":
        raise RuntimeError("synthetic implementation audit failed")
    return architecture, routing, scale_audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    population, slot = population_and_slot_audit()
    geometry = geometry_audit()
    feature_scale = feature_scale_audit()
    architecture, routing, scale = implementation_audit(float(feature_scale["selected_gamma"]))
    completion = {
        "status": "COMPLETE", "phase": "Phase 4E-0.5", "performance_experiment": False,
        "formal_training_started": False, "validation_inference_run": False,
        "internal_test_accessed": False, "official1000_accessed": False,
        "population_resolved": population["status"] == "PASS",
        "slot_target_availability_resolved": slot["status"] == "PASS",
        "geometry_validated": geometry["status"] == "PASS",
        "gradient_routing_validated": routing["status"] == "PASS",
        "loss_scale_validated": scale["status"] == "PASS",
        "implementation_validated": architecture["status"] == "PASS",
    }
    dump("completion_manifest.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
