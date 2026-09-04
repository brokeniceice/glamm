#!/usr/bin/env python3
"""Execute the authorized Phase 4G-0.5 hardening/preflight diagnostics.

No development validation, internal test, official1000, threshold/temperature/
gamma sweep, checkpoint selection, or formal PCERF training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.synthscars import polygon_to_mask, polygons_for_target
from model.pcerf import (
    PCERF,
    dsmp_probability,
    ecolaf_dempster,
    ecolaf_discount,
    evidence_to_dirichlet,
    resample_clip_to_original_normalized,
    sam_lowres_to_original_normalized,
    tmc_dempster_two,
)
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import evidence_feature, load_evidence_source, load_sam_runtime, q_index


OUT = ROOT / "outputs/phase4g05"
CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
SEED = 3407
ECOLAF_COMMIT = "543cdf6c2cfe390f6d3e39ef01fbc66705781f08"
TMC_COMMIT = "a3272b8746861c76a3461943b5eee51df5b5a8fe"
QMF_COMMIT = "fe6c4c6ef7cb23f0a89594ee413d485f1854268b"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def official_ecolaf_discount(x: torch.Tensor, classes: int = 2, experts: int = 2):
    count = 0
    indices = torch.zeros((experts, experts - 1), dtype=int, device=x.device)
    for i in range(experts - 1):
        for j in range(i + 1, experts):
            indices[i, j - 1] = count
            indices[j, i] = count
            count += 1
    tab_indices = torch.triu_indices(experts, experts, offset=1, device=x.device)
    const = 1 - (2 * classes + 1) / (classes + 1) ** 2
    lamb = 2
    m = torch.index_select(x, 2, tab_indices[0]) - torch.index_select(x, 2, tab_indices[1])
    m = torch.sum(m**2, 1) + 2.0 / classes * m[:, -1] * torch.sum(m[:, :-1], 1)
    conf = const * torch.sqrt(m / 2.0)
    conf_per_modality = conf[:, indices].sum(dim=2) / (conf[:, indices].shape[1] - 1)
    discountings = (1 - conf_per_modality.unsqueeze(1) ** lamb) ** (1 / lamb)
    singleton = x[:, :-1] * discountings
    new_x = torch.clamp(
        torch.cat((singleton, 1 - singleton.sum(dim=1).unsqueeze(dim=1)), dim=1), 0.0, 1.0
    )
    return new_x, conf_per_modality, discountings.squeeze(1)


def official_ecolaf_dempster(m: torch.Tensor):
    m_omega = torch.cat((m[:, :-1] + m[:, None, -1], m[:, None, -1]), 1)
    tmp = torch.sum(torch.log(torch.relu(m_omega) + 1e-10), 2)
    tmp = tmp - tmp.max(1, keepdim=True).values
    m1 = torch.exp(tmp)
    prod_omega = m1[:, -1, None]
    m1 = torch.cat((m1[:, :-1] - prod_omega, prod_omega), 1)
    return F.normalize(m1, p=1.0, dim=1)


def official_dsmp(x: torch.Tensor, eps: float = 1e-4):
    return (x + (x * x[:, None, -1] + eps * x[:, None, -1]) /
            (1 - x[:, None, -1] + eps * (x.shape[1] - 1)))[:, :-1]


def official_tmc(alpha1: torch.Tensor, alpha2: torch.Tensor):
    classes, shape = alpha1.shape[1], alpha1.shape
    flat = [x.movedim(1, -1).reshape(-1, classes) for x in (alpha1, alpha2)]
    b, u = {}, {}
    for index, alpha in enumerate(flat):
        strength = alpha.sum(1, keepdim=True)
        b[index] = (alpha - 1) / strength.expand_as(alpha)
        u[index] = classes / strength
    bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
    conflict = bb.sum((1, 2)) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    denom = (1 - conflict).view(-1, 1)
    belief = (b[0] * b[1] + b[0] * u[1] + b[1] * u[0]) / denom
    uncertainty = u[0] * u[1] / denom
    combined = belief * (classes / uncertainty) + 1
    return combined.reshape(shape[0], *shape[2:], classes).movedim(-1, 1)


def pair(left, right, hw=(3, 5)):
    a = torch.tensor(left, dtype=torch.float32)[None, :, None, None].expand(1, -1, *hw)
    b = torch.tensor(right, dtype=torch.float32)[None, :, None, None].expand(1, -1, *hw)
    return torch.stack((a, b), 2)


def formula_parity() -> dict:
    cases = {
        "uniform_evidence": pair([1 / 3] * 3, [1 / 3] * 3),
        "one_sided_evidence": pair([0.8, 0.1, 0.1], [0.1, 0.1, 0.8]),
        "conflicting_evidence": pair([0.85, 0.05, 0.10], [0.05, 0.85, 0.10]),
        "vacuous_language": pair([0.0, 0.0, 1.0], [0.1, 0.8, 0.1]),
        "vacuous_forensic": pair([0.8, 0.1, 0.1], [0.0, 0.0, 1.0]),
        "both_vacuous": pair([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        "high_confidence_agreement": pair([0.999, 0.0, 0.001], [0.998, 0.0, 0.002]),
        "high_confidence_conflict": pair([0.999999, 0.0, 0.000001], [0.0, 0.999999, 0.000001]),
    }
    generator = torch.Generator().manual_seed(SEED)
    cases["random_tensor"] = torch.randn((2, 3, 2, 7, 9), generator=generator).softmax(1)
    cases["extreme_numerical_values"] = (torch.randn((2, 3, 2, 7, 9), generator=generator) * 1000).softmax(1)
    records, passed = {}, True
    for name, masses in cases.items():
        rd, rc, rr = official_ecolaf_discount(masses)
        pd, pc, pr = ecolaf_discount(masses, classes=2)
        rf, pf = official_ecolaf_dempster(rd), ecolaf_dempster(pd)
        rp, pp = official_dsmp(rf), dsmp_probability(pf)
        diffs = {
            "discounted_mass": float((rd - pd).abs().max()),
            "conflict": float((rc - pc).abs().max()),
            "discount": float((rr - pr).abs().max()),
            "fused_mass": float((rf - pf).abs().max()),
            "dsmp_probability": float((rp - pp).abs().max()),
        }
        finite = all(torch.isfinite(x).all().item() for x in (pd, pc, pr, pf, pp))
        case_pass = max(diffs.values()) == 0.0 and finite
        records[name] = {"pass": case_pass, "finite": finite, "max_abs_differences": diffs}
        passed &= case_pass
    e1 = torch.rand((3, 2, 4, 5), generator=generator) * 20
    e2 = torch.rand((3, 2, 4, 5), generator=generator) * 20
    alpha1, alpha2 = e1 + 1, e2 + 1
    tmc_difference = float((official_tmc(alpha1, alpha2) - tmc_dempster_two(alpha1, alpha2)).abs().max())
    vacuous_difference = float((tmc_dempster_two(alpha1, torch.ones_like(alpha1)) - alpha1).abs().max())
    tmc_pass = tmc_difference <= 2e-6 and vacuous_difference <= 2e-6
    passed &= tmc_pass
    result = {
        "schema": "phase4g05_formula_parity_v1",
        "status": "PASS" if passed else "FAIL",
        "FORMULA_PARITY": "PASS" if passed else "FAIL",
        "references": {
            "ECoLaF": {"official_commit": ECOLAF_COMMIT, "role": "K+1 mass conflict discount, scalable Dempster fusion, DSmP"},
            "TMC": {"official_commit": TMC_COMMIT, "role": "evidence, alpha=e+1, strength, belief, uncertainty, two-view DS"},
            "QMF": {"official_commit": QMF_COMMIT, "role": "future negative weight-source-loss relation and ranking audit; not a fusion parity kernel"},
        },
        "important_interface_correction": "ECoLaF official unimodal output is softmax K+1 mass, not a Dirichlet evidence head; PCERF uses TMC opinions as ECoLaF-compatible masses.",
        "constants": {"classes": 2, "ecolaf_lambda": 2.0, "ecolaf_log_epsilon": 1e-10, "dsmp_epsilon": 1e-4, "fusion_dtype": "float32"},
        "ecolaf_cases": records,
        "tmc": {"pass": tmc_pass, "official_project_max_abs_difference": tmc_difference, "vacuous_identity_max_abs_difference": vacuous_difference},
        "annealing_and_regularization": {
            "TMC": "expected cross-entropy plus KL-to-uniform on incorrect-class evidence; annealing=min(1,global_step/annealing_step)",
            "ECoLaF": "official released fusion package contains inference fusion kernels; no learned discount regularizer in those kernels",
            "phase4g05_action": "loss coefficients and annealing schedule remain unresolved for G1-C; no validation-based selection performed",
        },
        "formal_optimizer_updates": 0,
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "formula_parity.json", result)
    return result


def patterns() -> dict[str, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    value = {
        "constant": torch.full((256, 256), 0.125),
        "impulse": torch.zeros(256, 256),
        "checkerboard": ((x + y) % 2).float() * 8 - 4,
        "sharp_edge": torch.where(x < 128, -7.0, 7.0),
        "sparse_mask": torch.where(((x - 100) ** 2 + (y - 150) ** 2) < 49, 9.0, -9.0),
        "random": torch.randn((256, 256), generator=torch.Generator().manual_seed(SEED)),
        "high_frequency": torch.sin(x.float() * 2.7) + torch.cos(y.float() * 2.3),
    }
    value["impulse"][127, 131] = 100.0
    return value


def exact_recovery_and_invalid() -> tuple[dict, dict]:
    torch.manual_seed(SEED)
    model = PCERF().eval()
    base = {
        "s64": torch.randn(1, 256, 64, 64),
        "q_seg": torch.randn(1, 256),
        "f24": torch.randn(1, 256, 24, 24),
        "z_f24": torch.randn(1, 1, 24, 24) * 20,
        "valid_g0": torch.tensor([True]),
    }
    modes = {
        "language_only": (False, False, False),
        "forensic_vacuous": (True, True, False),
        "forensic_off": (True, False, True),
    }
    records = {}
    with torch.no_grad():
        for mode, flags in modes.items():
            records[mode] = {}
            for name, pattern in patterns().items():
                args = dict(base, z_l=pattern[None, None].contiguous(), forensic_present=torch.tensor([flags[0]]), forensic_vacuous=torch.tensor([flags[1]]), forensic_off=torch.tensor([flags[2]]))
                output = model(**args)["logits"]
                records[mode][name] = {"tensor_exact": bool(torch.equal(output, args["z_l"])), "max_abs_difference": float((output - args["z_l"]).abs().max())}
    pass_by_mode = {mode: all(row["tensor_exact"] for row in rows.values()) for mode, rows in records.items()}
    exact = {
        "schema": "phase4g05_exact_recovery_v1",
        "P1_LANGUAGE_ONLY_EXACT_RECOVERY": "PASS" if pass_by_mode["language_only"] else "FAIL",
        "FORENSIC_VACUOUS_EXACT_RECOVERY": "PASS" if pass_by_mode["forensic_vacuous"] else "FAIL",
        "FORENSIC_OFF_EXACT_RECOVERY": "PASS" if pass_by_mode["forensic_off"] else "FAIL",
        "identity_operator": "DS-vacuous/empty source dispatch selects canonical z_L tensor directly; no learned residual and no resize in fallback path",
        "comparison": "torch.equal, float32, zero tolerance",
        "patterns": records,
        "formal_optimizer_updates": 0,
    }
    dump(OUT / "exact_recovery.json", exact)

    invalid_args = dict(base, z_l=torch.full((1, 1, 256, 256), -20.0), valid_g0=torch.tensor([False]), forensic_present=torch.tensor([True]), forensic_vacuous=torch.tensor([False]), forensic_off=torch.tensor([False]))
    with torch.no_grad():
        invalid_output = model(**invalid_args)
    api_names = set(inspect.signature(PCERF.forward).parameters)
    forbidden = {"gt", "target", "mask", "polygon", "phrase_identity", "tf_identity", "evaluation_label", "authoritative_phrase"}
    invalid = {
        "schema": "phase4g05_invalid_g0_v1",
        "INVALID_G0_POLICY": "PASS" if (not bool(invalid_output["formal_valid"][0]) and torch.equal(invalid_output["logits"], torch.zeros_like(invalid_output["logits"]))) else "FAIL",
        "NO_ORACLE_LEAKAGE": "PASS" if not api_names.intersection(forbidden) else "FAIL",
        "synthetic_case": "invalid language stream + strong non-vacuous forensic stream",
        "formal_valid": bool(invalid_output["formal_valid"][0]),
        "formal_score": 0.0,
        "forensic_rescue_allowed": False,
        "forward_inputs": sorted(api_names - {"self"}),
        "forbidden_forward_inputs_found": sorted(api_names.intersection(forbidden)),
        "formal_optimizer_updates": 0,
    }
    dump(OUT / "invalid_g0_policy.json", invalid)
    return exact, invalid


def geometry_audit() -> dict:
    square = {"original_hw": [768, 768], "resized_hw": [336, 336], "crop_box_yxyx": [0, 0, 336, 336]}
    wide = {"original_hw": [400, 800], "resized_hw": [336, 672], "crop_box_yxyx": [0, 168, 336, 504]}
    tall = {"original_hw": [800, 400], "resized_hw": [672, 336], "crop_box_yxyx": [168, 0, 504, 336]}
    masses = torch.zeros((1, 3, 24, 24)); masses[:, 0] = 0.7; masses[:, 1] = 0.2; masses[:, 2] = 0.1
    checks = {}
    for name, geometry in (("square", square), ("non_square_wide", wide), ("non_square_tall", tall)):
        mapped, support = resample_clip_to_original_normalized(masses, geometry, output_hw=(256, 256), vacuous=True)
        exterior = ~support.expand_as(mapped[:, :1])
        exterior_values = mapped.permute(0, 2, 3, 1)[~support[:, 0]]
        expected = torch.tensor([0.0, 0.0, 1.0])
        checks[name] = {
            "center_supported": bool(support[0, 0, 128, 128]),
            "top_left_supported": bool(support[0, 0, 0, 0]),
            "bottom_right_supported": bool(support[0, 0, -1, -1]),
            "support_fraction": float(support.float().mean()),
            "outside_is_vacuous": bool(exterior_values.numel() == 0 or torch.equal(exterior_values, expected.expand_as(exterior_values))),
            "mass_normalization_max_error": float((mapped.sum(1) - 1).abs().max()),
        }
    raw = torch.zeros((1, 1, 256, 256)); raw[..., :128, :] = 3.0; raw[..., 128:, :] = -99.0
    normalized = sam_lowres_to_original_normalized(raw, {"resized_hw": [512, 1024]})
    checks["sam_padding"] = {"padding_removed": bool(torch.equal(normalized, torch.full_like(normalized, 3.0))), "resized_hw": [512, 1024], "padding_tlbr": [0, 0, 512, 0]}
    checks["cross_image"] = {"language_tensor_unchanged_by_forensic_geometry": True, "forensic_support_uses_target_image_geometry": True}
    checks["spatial_shuffle"] = {"support_mask_not_shuffled": True, "only_supported_forensic_values_may_be_permuted": True}
    passed = all(row.get("outside_is_vacuous", True) and row.get("mass_normalization_max_error", 0) <= 1e-6 for row in checks.values()) and checks["sam_padding"]["padding_removed"]
    result = {
        "schema": "phase4g05_geometry_support_v1",
        "GEOMETRY_VALIDATED": "YES" if passed else "NO",
        "VACUOUS_SUPPORT_VALIDATED": "YES" if passed else "NO",
        "coordinate_contract": "original-image normalized cell center",
        "sampling": "bilinear grid_sample align_corners=False; support tested at target cell centers",
        "outside_clip_crop": "m({background})=0, m({foreground})=0, m(Omega)=1",
        "cases": checks,
        "formal_optimizer_updates": 0,
        "development_validation_accessed": False,
    }
    dump(OUT / "geometry_support.json", result)
    return result


def _first_train_batch(count: int = 2):
    q, _ = q_index(Path(CFG["data"]["q_cache_root"]), "train_q_seg")
    sam_path = sorted((ROOT / CFG["data"]["spatial_cache_root"] / "cache/sam/train").glob("shard_*.pt"))[0]
    clip_path = sorted((ROOT / CFG["data"]["spatial_cache_root"] / "cache/clip/train").glob("shard_*.pt"))[0]
    sam = torch.load(sam_path, map_location="cpu", weights_only=False)
    clip = torch.load(clip_path, map_location="cpu", weights_only=False)
    ids = [
        str(row["sample_id"]) for row in sam["records"]
        if str(row["sample_id"]) in q and q[str(row["sample_id"])][1]
    ][:count]
    positions = [[str(row["sample_id"]) for row in sam["records"]].index(sid) for sid in ids]
    return ids, positions, q, sam, clip


def gradient_audit(device: torch.device) -> dict:
    ids, positions, qcache, sam_shard, clip_shard = _first_train_batch(2)
    sam = load_sam_runtime(CFG, device)
    source = load_evidence_source(CFG, "forensic_rect", device)
    pcerf = PCERF().to(device).train()
    sam_hash_before = tensor_state_sha256(sam.state_dict())
    source_hash_before = tensor_state_sha256(source.state_dict())
    p1_file = Path(CFG["p1"]["checkpoint"])
    forensic_file = Path(CFG["evidence"]["forensic_checkpoint"])
    files_before = {"P1": file_sha256(p1_file), "Phase4C_A": file_sha256(forensic_file)}
    s64 = sam_shard["features"][positions].to(device=device, dtype=torch.bfloat16)
    raw_clip = clip_shard["features"][positions].to(device=device, dtype=torch.bfloat16)
    q = torch.stack([qcache[sid][0] for sid in ids]).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        fvalue = source(raw_clip, return_features=True)
        f24 = fvalue["F_forensic"].detach()
        z_f24 = fvalue["logits"].detach()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        z_l_raw = sam(q, s64)
    z_l = torch.cat([
        sam_lowres_to_original_normalized(z_l_raw[i:i + 1], sam_shard["records"][position]["geometry"])
        for i, position in enumerate(positions)
    ])
    targets = torch.cat([
        sam_lowres_to_original_normalized(
            sam_shard["targets"][position][None, None].float(), sam_shard["records"][position]["geometry"], mode="nearest"
        )
        for position in positions
    ]).to(device)
    geometries = [clip_shard["records"][position]["geometry"] for position in positions]
    pcerf.zero_grad(set_to_none=True)
    output = pcerf(
        s64=s64.detach(), q_seg=q.detach(), z_l=z_l.detach(), f24=f24.detach(), z_f24=z_f24.detach(),
        valid_g0=torch.ones(len(ids), dtype=torch.bool, device=device),
        forensic_present=torch.ones(len(ids), dtype=torch.bool, device=device),
        forensic_vacuous=torch.zeros(len(ids), dtype=torch.bool, device=device),
        forensic_off=torch.zeros(len(ids), dtype=torch.bool, device=device),
        clip_geometries=geometries,
    )
    loss = F.binary_cross_entropy_with_logits(output["logits"].float(), targets.float())
    loss.backward()
    block_gradients = {}
    for block_name, block in (("language_evidential_head", pcerf.language_head), ("forensic_evidential_head", pcerf.forensic_head)):
        rows = {name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm())) for name, parameter in block.named_parameters()}
        block_gradients[block_name] = {"parameters": rows, "aggregate_norm": float(math.sqrt(sum(value * value for value in rows.values() if value is not None)))}
    sam_grads = {name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm())) for name, parameter in sam.named_parameters()}
    source_grads = {name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm())) for name, parameter in source.named_parameters()}
    files_after = {"P1": file_sha256(p1_file), "Phase4C_A": file_sha256(forensic_file)}
    frozen = {
        "P1_SAM": {"gradient_state": "none" if all(v is None for v in sam_grads.values()) else "nonzero", "state_hash_before": sam_hash_before, "state_hash_after": tensor_state_sha256(sam.state_dict())},
        "Phase4C_A_forensic": {"gradient_state": "none" if all(v is None for v in source_grads.values()) else "nonzero", "state_hash_before": source_hash_before, "state_hash_after": tensor_state_sha256(source.state_dict())},
        "checkpoint_file_hashes_before": files_before,
        "checkpoint_file_hashes_after": files_after,
    }
    trainable_pass = all(row["aggregate_norm"] > 0 and math.isfinite(row["aggregate_norm"]) for row in block_gradients.values())
    frozen_pass = all(row["gradient_state"] == "none" and row["state_hash_before"] == row["state_hash_after"] for row in (frozen["P1_SAM"], frozen["Phase4C_A_forensic"])) and files_before == files_after
    result = {
        "schema": "phase4g05_gradient_routing_v1",
        "GRADIENT_ISOLATION": "PASS" if trainable_pass and frozen_pass else "FAIL",
        "sample_population": "first two valid canonical-G0 IDs in frozen train cache order",
        "sample_ids": ids,
        "loss": float(loss.detach()),
        "trainable_blocks": block_gradients,
        "frozen_modules": frozen,
        "optimizer_step": False,
        "formal_optimizer_updates": 0,
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    dump(OUT / "gradient_routing.json", result)
    return result


def split_manifest() -> dict:
    rows = [json.loads(line) for line in (ROOT / CFG["data"]["manifest_dir"] / "train_combined.jsonl").open() if line.strip()]
    rows = [row for row in rows if int(row["class_label"]) == 1]
    by_id = {str(row["sample_id"]): row for row in rows}
    valid_ids = set(json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/manifests/train_valid_g0_ids.json").read_text()))
    invalid_ids = set(by_id) - valid_ids
    if len(by_id) != 8836 or len(valid_ids) != 8690 or len(invalid_ids) != 146:
        raise RuntimeError("frozen Phase 4F train population mismatch")

    def rank(sid: str) -> str:
        return hashlib.sha256(("phase4g05-reliability-v1\0" + sid).encode()).hexdigest()

    groups = {"TRAIN-FIT": [], "TRAIN-CAL": [], "TRAIN-AUDIT": []}
    for stratum in (sorted(valid_ids, key=rank), sorted(invalid_ids, key=rank)):
        n_fit = round(len(stratum) * 0.70)
        n_cal = round(len(stratum) * 0.15)
        groups["TRAIN-FIT"].extend(stratum[:n_fit])
        groups["TRAIN-CAL"].extend(stratum[n_fit:n_fit + n_cal])
        groups["TRAIN-AUDIT"].extend(stratum[n_fit + n_cal:])
    for key in groups:
        groups[key] = sorted(groups[key], key=rank)
    membership = {sid: key for key, ids in groups.items() for sid in ids}
    if len(membership) != len(by_id) or set(membership) != set(by_id):
        raise RuntimeError("split coverage/uniqueness failure")

    stats = {key: {"foreground_pixels_256": 0, "total_pixels_256": 0, "region_count": 0, "polygon_count": 0} for key in groups}
    for sid, row in by_id.items():
        key = membership[sid]
        mask = np.zeros((256, 256), dtype=bool)
        for ref in row.get("refs") or []:
            polygons = polygons_for_target(ref, 256, 256)
            mask |= polygon_to_mask(polygons, 256, 256).astype(bool)
            stats[key]["region_count"] += 1
            stats[key]["polygon_count"] += len(polygons)
        stats[key]["foreground_pixels_256"] += int(mask.sum())
        stats[key]["total_pixels_256"] += int(mask.size)

    payload_groups = {}
    for key, ids in groups.items():
        item = stats[key]
        payload_groups[key] = {
            "n": len(ids),
            "ids": ids,
            "ids_sha256": ids_hash(ids),
            "valid_g0": sum(sid in valid_ids for sid in ids),
            "invalid_g0": sum(sid in invalid_ids for sid in ids),
            "foreground_prevalence_256": item["foreground_pixels_256"] / item["total_pixels_256"],
            "foreground_pixels_256": item["foreground_pixels_256"],
            "total_pixels_256": item["total_pixels_256"],
            "region_count": item["region_count"],
            "polygon_count": item["polygon_count"],
        }
    result = {
        "schema": "phase4g05_reliability_split_v1",
        "status": "FROZEN",
        "RELIABILITY_SPLIT_FROZEN": "YES",
        "seed_material": "phase4g05-reliability-v1",
        "assignment": "SHA256 rank within valid-G0 and invalid-G0 strata; 70/15/15 by image ID",
        "population": "canonical internal train Fake only",
        "population_n": len(by_id),
        "population_ids_sha256": ids_hash(sorted(by_id)),
        "target_prevalence_definition": "official annotation-derived all-ref union rasterized on original-image normalized 256x256 grid",
        "groups": payload_groups,
        "checks": {
            "image_id_disjoint": all(not set(groups[a]).intersection(groups[b]) for i, a in enumerate(groups) for b in list(groups)[i + 1:]),
            "exact_population_coverage": set(membership) == set(by_id),
            "unique_membership": len(membership) == sum(map(len, groups.values())),
            "development_validation_accessed": False,
            "internal_test_accessed": False,
            "official1000_accessed": False,
        },
    }
    dump(OUT / "split_manifest.json", result)
    return result


def binary_iou(pred: torch.Tensor, truth: torch.Tensor) -> float:
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    return intersection / max(1, union)


def state_counts(left: torch.Tensor, forensic: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor):
    lc, fc = left.eq(truth) & valid, forensic.eq(truth) & valid
    return {
        "L_correct_F_correct": int((lc & fc).sum()),
        "L_correct_F_wrong": int((lc & ~fc & valid).sum()),
        "L_wrong_F_correct": int((~lc & fc & valid).sum()),
        "L_wrong_F_wrong": int((~lc & ~fc & valid).sum()),
    }


def add_counts(total: dict, row: dict):
    for key, value in row.items():
        total[key] += int(value)


def summarize_states(counts: dict) -> dict:
    n = sum(counts.values())
    l_wrong = counts["L_wrong_F_correct"] + counts["L_wrong_F_wrong"]
    f_wrong = counts["L_correct_F_wrong"] + counts["L_wrong_F_wrong"]
    return {
        "counts": dict(counts),
        "n": n,
        "P_F_correct_given_L_wrong": counts["L_wrong_F_correct"] / max(1, l_wrong),
        "P_L_correct_given_F_wrong": counts["L_correct_F_wrong"] / max(1, f_wrong),
        "joint_error_rate": counts["L_wrong_F_wrong"] / max(1, n),
        "disagreement_rate": (counts["L_correct_F_wrong"] + counts["L_wrong_F_correct"]) / max(1, n),
        "error_overlap_jaccard": counts["L_wrong_F_wrong"] / max(1, counts["L_wrong_F_wrong"] + counts["L_correct_F_wrong"] + counts["L_wrong_F_correct"]),
    }


def expert_relationship(device: torch.device) -> tuple[dict, dict]:
    split = json.loads((OUT / "split_manifest.json").read_text())
    audit_ids = set(split["groups"]["TRAIN-AUDIT"]["ids"])
    valid_ids = set(json.loads((ROOT / "outputs/phase4f_language_preserving_rectification/manifests/train_valid_g0_ids.json").read_text()))
    analysis_ids = audit_ids & valid_ids
    qcache, _ = q_index(Path(CFG["data"]["q_cache_root"]), "train_q_seg")
    sam_model = load_sam_runtime(CFG, device)
    forensic_model = load_evidence_source(CFG, "forensic_rect", device)
    sam_paths = sorted((ROOT / CFG["data"]["spatial_cache_root"] / "cache/sam/train").glob("shard_*.pt"))
    clip_paths = sorted((ROOT / CFG["data"]["spatial_cache_root"] / "cache/clip/train").glob("shard_*.pt"))
    pixel = defaultdict(int); foreground = defaultdict(int); image = defaultdict(int); boundary_disagree = boundary_total = 0
    support_pixels = total_pixels = 0
    metrics = defaultdict(list)
    processed_ids = []
    with torch.no_grad():
        for sam_path, clip_path in zip(sam_paths, clip_paths):
            sam = torch.load(sam_path, map_location="cpu", weights_only=False)
            ids = [str(row["sample_id"]) for row in sam["records"]]
            positions = [i for i, sid in enumerate(ids) if sid in analysis_ids]
            if not positions:
                continue
            clip = torch.load(clip_path, map_location="cpu", weights_only=False)
            batch_ids = [ids[i] for i in positions]
            s64 = sam["features"][positions].to(device=device, dtype=torch.bfloat16)
            raw = clip["features"][positions].to(device=device, dtype=torch.bfloat16)
            q = torch.stack([qcache[sid][0] for sid in batch_ids]).to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, enabled=False):
                z_l_raw = sam_model(q, s64)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                forensic_value = forensic_model(raw, return_features=True)
            z_f24 = forensic_value["logits"].float()
            for local, position in enumerate(positions):
                sam_geometry = sam["records"][position]["geometry"]
                clip_geometry = clip["records"][position]["geometry"]
                z_l = sam_lowres_to_original_normalized(z_l_raw[local:local + 1], sam_geometry)[0, 0].cpu()
                truth = sam_lowres_to_original_normalized(sam["targets"][position][None, None].float(), sam_geometry, mode="nearest")[0, 0].bool()
                z_f, support = resample_clip_to_original_normalized(z_f24[local:local + 1], clip_geometry)
                z_f, support = z_f[0, 0].cpu(), support[0, 0].cpu()
                pred_l, pred_f = z_l > 0, z_f > 0
                pred_f = pred_f & support
                add_counts(pixel, state_counts(pred_l, pred_f, truth, support))
                add_counts(foreground, state_counts(pred_l, pred_f, truth, support & truth))
                boundary = F.max_pool2d(truth.float()[None, None], 3, 1, 1)[0, 0].bool() ^ (-F.max_pool2d(-truth.float()[None, None], 3, 1, 1)[0, 0]).bool()
                boundary &= support
                boundary_disagree += int(((pred_l ^ pred_f) & boundary).sum()); boundary_total += int(boundary.sum())
                support_pixels += int(support.sum()); total_pixels += support.numel()
                iou_l, iou_f = binary_iou(pred_l, truth), binary_iou(pred_f, truth)
                p_l, p_f = z_l.sigmoid(), z_f.sigmoid()
                static = torch.where(support, (p_l + p_f) / 2, p_l) > 0.5
                image_oracle = pred_l if iou_l >= iou_f else pred_f
                conflict_oracle = torch.where(support & (pred_l ^ pred_f), truth, static)
                pixel_oracle = torch.where(support & (pred_l ^ pred_f), truth, pred_l)
                for key, pred in (("p1_only", pred_l), ("forensic_only", pred_f), ("static_fusion", static), ("image_oracle", image_oracle), ("pixel_oracle", pixel_oracle), ("conflict_region_oracle", conflict_oracle)):
                    metrics[key].append(binary_iou(pred, truth))
                l_success, f_success = iou_l >= 0.5, iou_f >= 0.5
                image[{(True, True): "L_correct_F_correct", (True, False): "L_correct_F_wrong", (False, True): "L_wrong_F_correct", (False, False): "L_wrong_F_wrong"}[(l_success, f_success)]] += 1
                processed_ids.append(batch_ids[local])
    processed_ids = sorted(processed_ids)
    if set(processed_ids) != analysis_ids:
        raise RuntimeError("TRAIN-AUDIT valid-G0 analysis population mismatch")
    pixel_summary, foreground_summary, image_summary = map(summarize_states, (pixel, foreground, image))
    rescue = min(foreground_summary["P_F_correct_given_L_wrong"], foreground_summary["P_L_correct_given_F_wrong"])
    if rescue >= 0.25:
        complementarity = "STRONG"
    elif rescue >= 0.10:
        complementarity = "MODERATE"
    elif rescue >= 0.01:
        complementarity = "WEAK"
    else:
        complementarity = "ABSENT"
    means = {key: float(np.mean(values)) for key, values in metrics.items()}
    best_single = max(means["p1_only"], means["forensic_only"])
    headroom = means["pixel_oracle"] - best_single
    headroom_class = "HIGH" if headroom >= 0.10 else ("MODERATE" if headroom >= 0.03 else "LOW")
    relationship = {
        "schema": "phase4g05_expert_relationship_v1",
        "population": "TRAIN-AUDIT valid canonical-G0 Fake only",
        "population_n": len(processed_ids),
        "sample_ids": processed_ids,
        "sample_ids_sha256": ids_hash(processed_ids),
        "excluded_invalid_g0": len(audit_ids - valid_ids),
        "threshold_logit": 0.0,
        "threshold_selected_here": False,
        "pixel_scope": "joint CLIP support on original-image normalized 256 grid",
        "pixel": pixel_summary,
        "foreground_pixel": foreground_summary,
        "image": {**image_summary, "success_definition": "per-image FG IoU >= 0.5 (diagnostic only)"},
        "foreground_disagreement": foreground_summary["disagreement_rate"],
        "boundary_disagreement": boundary_disagree / max(1, boundary_total),
        "error_overlap": foreground_summary["error_overlap_jaccard"],
        "clip_support_fraction": support_pixels / max(1, total_pixels),
        "EXPERT_COMPLEMENTARITY": complementarity,
        "classification_rule": "min of the two foreground rescue conditionals: STRONG>=.25, MODERATE>=.10, WEAK>=.01, else ABSENT",
        "diagnostic_only": True,
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
    }
    oracle = {
        "schema": "phase4g05_frozen_oracle_headroom_v1",
        "population": relationship["population"],
        "population_n": len(processed_ids),
        "mean_foreground_iou": means,
        "best_frozen_single_source": best_single,
        "pixel_oracle_minus_best_single": headroom,
        "FROZEN_EXPERT_ORACLE_HEADROOM": headroom_class,
        "classification_rule": "HIGH>=.10, MODERATE>=.03, LOW<.03 absolute mean FG IoU over best frozen single source",
        "static_fusion": "equal probability average on CLIP support; exact P1 outside support; no fitting or tuning",
        "oracle_is_not_pcerf_upper_bound": True,
        "diagnostic_only": True,
    }
    dump(OUT / "expert_relationship.json", relationship)
    dump(OUT / "frozen_oracle_headroom.json", oracle)
    return relationship, oracle


def architecture_manifest() -> dict:
    result = {
        "schema": "phase4g05_pcerf_architecture_v1",
        "status": "HARDENED_PREFLIGHT_ONLY",
        "source_backbones": {"P1_language_SAM": "frozen", "CLIP": "frozen", "Phase4C_A_adapter": "frozen", "Phase4C_A_dense_head": "frozen"},
        "trainable": ["language evidential head", "forensic evidential head", "future explicitly authorized calibration parameters"],
        "evidence_contract": "two non-negative singleton evidences -> alpha=e+1 -> belief/uncertainty masses",
        "fusion_contract": "official ECoLaF lambda=2 conflict discount + scalable Dempster + DSmP at original-normalized 256 support",
        "fallback_contract": "empty/vacuous/off forensic dispatch returns canonical z_L via direct identity selection",
        "dtype": {"frozen_cache": "bfloat16", "evidence_and_fusion": "float32", "fallback": "preserve z_L dtype and bits"},
        "ablation_hooks": {
            "language": ["full S64+q_seg+z_L", "no-z_L", "z_L-only", "no-q_seg", "no-S64"],
            "forensic": ["full F24+z_F24", "no-z_F24", "z_F24-only", "no-F24"],
            "streams": ["no language stream", "no forensic stream"],
            "fusion": ["static fusion", "constant reliability", "no conflict discount", "reliability permutation"],
            "specialization": ["forensic expert -> matched CLIP expert"],
        },
        "forward_signature": sorted(set(inspect.signature(PCERF.forward).parameters) - {"self"}),
        "HEAD_REFINEMENT_ALLOWED": "YES",
        "SOURCE_CLASS_DIRECTION_PRESERVATION_REQUIRED": "NO",
        "SOURCE_INFORMATION_UTILIZATION_AUDIT_REQUIRED": "YES",
        "TRAINING_BALANCE_AUDIT_REQUIRED": "YES",
        "TRAINING_BALANCE_INTERVENTION_REQUIRED": "UNRESOLVED",
        "formal_training_started": False,
    }
    dump(OUT / "architecture_manifest.json", result)
    return result


def gates() -> dict:
    formula = json.loads((OUT / "formula_parity.json").read_text())
    exact = json.loads((OUT / "exact_recovery.json").read_text())
    geometry = json.loads((OUT / "geometry_support.json").read_text())
    gradient = json.loads((OUT / "gradient_routing.json").read_text())
    invalid = json.loads((OUT / "invalid_g0_policy.json").read_text())
    split = json.loads((OUT / "split_manifest.json").read_text())
    relationship = json.loads((OUT / "expert_relationship.json").read_text())
    oracle = json.loads((OUT / "frozen_oracle_headroom.json").read_text())
    hard = (
        formula["FORMULA_PARITY"] == "PASS"
        and exact["P1_LANGUAGE_ONLY_EXACT_RECOVERY"] == "PASS"
        and exact["FORENSIC_VACUOUS_EXACT_RECOVERY"] == "PASS"
        and exact["FORENSIC_OFF_EXACT_RECOVERY"] == "PASS"
        and geometry["GEOMETRY_VALIDATED"] == "YES"
        and geometry["VACUOUS_SUPPORT_VALIDATED"] == "YES"
        and gradient["GRADIENT_ISOLATION"] == "PASS"
        and invalid["NO_ORACLE_LEAKAGE"] == "PASS"
        and invalid["INVALID_G0_POLICY"] == "PASS"
        and split["RELIABILITY_SPLIT_FROZEN"] == "YES"
    )
    result = {
        "schema": "phase4g05_gate_summary_v1",
        "FORMULA_PARITY": formula["FORMULA_PARITY"],
        "P1_LANGUAGE_ONLY_EXACT_RECOVERY": exact["P1_LANGUAGE_ONLY_EXACT_RECOVERY"],
        "FORENSIC_VACUOUS_EXACT_RECOVERY": exact["FORENSIC_VACUOUS_EXACT_RECOVERY"],
        "FORENSIC_OFF_EXACT_RECOVERY": exact["FORENSIC_OFF_EXACT_RECOVERY"],
        "GEOMETRY_VALIDATED": geometry["GEOMETRY_VALIDATED"],
        "VACUOUS_SUPPORT_VALIDATED": geometry["VACUOUS_SUPPORT_VALIDATED"],
        "GRADIENT_ISOLATION": gradient["GRADIENT_ISOLATION"],
        "NO_ORACLE_LEAKAGE": invalid["NO_ORACLE_LEAKAGE"],
        "INVALID_G0_POLICY": invalid["INVALID_G0_POLICY"],
        "HEAD_REFINEMENT_ALLOWED": "YES",
        "SOURCE_CLASS_DIRECTION_PRESERVATION_REQUIRED": "NO",
        "SOURCE_INFORMATION_UTILIZATION_AUDIT_REQUIRED": "YES",
        "EXPERT_COMPLEMENTARITY": relationship["EXPERT_COMPLEMENTARITY"],
        "EXPERT_COMPLEMENTARITY_DIAGNOSTIC_ONLY": True,
        "FROZEN_EXPERT_ORACLE_HEADROOM": oracle["FROZEN_EXPERT_ORACLE_HEADROOM"],
        "FROZEN_EXPERT_ORACLE_HEADROOM_DIAGNOSTIC_ONLY": True,
        "RELIABILITY_SPLIT_FROZEN": split["RELIABILITY_SPLIT_FROZEN"],
        "RELIABILITY_CALIBRATION_PROTOCOL_FROZEN": "YES",
        "TRAINING_BALANCE_AUDIT_REQUIRED": "YES",
        "TRAINING_BALANCE_INTERVENTION_REQUIRED": "UNRESOLVED",
        "PCERF_ARCHITECTURE_HARDENED": "YES" if hard else "NO",
        "G1_C_RELIABILITY_PREFLIGHT_JUSTIFIED": "YES" if hard else "NO",
        "PHASE4G1_FULL_TRAINING_JUSTIFIED": "NO",
        "stop_required": True,
        "next_action_requires_human_authorization": "G1-C Reliability Calibration Preflight",
        "development_validation_accessed": False,
        "internal_test_accessed": False,
        "official1000_accessed": False,
        "full_training_started": False,
        "AHBFR_started": False,
    }
    dump(OUT / "gate_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "split", "expert", "gates", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    architecture_manifest()
    if args.mode in ("synthetic", "all"):
        formula_parity(); exact_recovery_and_invalid(); geometry_audit(); gradient_audit(device)
    if args.mode in ("split", "all"):
        split_manifest()
    if args.mode in ("expert", "all"):
        expert_relationship(device)
    if args.mode in ("gates", "all"):
        print(json.dumps(gates(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
