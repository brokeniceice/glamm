#!/usr/bin/env python3
"""Read-only audit of Rectifier out_proj and residual projection matrices.

The audit does not modify any model.  It measures the two matrices directly,
collects frozen A0/A1 R2 representations, and reports how much of the R2
principal variance and the A1-A0 difference direction survives each projection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.phase6g9_rectifier_internal_conversion_audit as g9
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from scripts.phase6g7_staged_r1 import Store
from tools.phase4f import Phase4FStore

OUT = ROOT / "outputs/phase6g9_rectifier_internal_conversion_audit"


def matrix_payload(rectifier):
    attention = rectifier.rectification.cross_attention
    return {
        "out_proj": attention.out_proj.weight.detach().float().cpu(),
        "out_proj_bias": attention.out_proj.bias.detach().float().cpu(),
        "proj": rectifier.rectification.projection.weight.detach().float().cpu(),
        "proj_bias": rectifier.rectification.projection.bias.detach().float().cpu(),
        "gamma": float(rectifier.rectification.gamma.detach().float().cpu()),
    }


def spectrum(matrix):
    value = matrix.double()
    singular = torch.linalg.svdvals(value)
    squared = singular * singular
    total = squared.sum().clamp_min(1e-30)
    probability = squared / total
    shannon = float(torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum()))
    participation = float((squared.sum() ** 2) / squared.pow(2).sum().clamp_min(1e-30))
    stable = float(squared.sum() / (singular[0] ** 2).clamp_min(1e-30))
    rank = int((singular > 1e-6 * singular[0]).sum())
    s_max, s_min = float(singular[0]), float(singular[-1])
    return {
        "shape": list(value.shape),
        "rank": rank,
        "effective_rank_shannon": shannon,
        "effective_rank_participation": participation,
        "stable_rank": stable,
        "condition_number_full": (s_max / s_min if s_min > 1e-12 else float("inf")),
        "condition_number_rank": (
            s_max / float(singular[rank - 1])
            if rank > 0 and float(singular[rank - 1]) > 1e-12
            else float("inf")
        ),
        "singular_values_top10": [float(x) for x in singular[:10]],
        "singular_values_bottom5": [float(x) for x in singular[-5:]],
        "s_max": s_max,
        "s_min": s_min,
    }


def principal_angles(left, right, k=32):
    u_left, _, _ = torch.linalg.svd(left.double(), full_matrices=False)
    u_right, _, _ = torch.linalg.svd(right.double(), full_matrices=False)
    values = torch.linalg.svdvals(u_left[:, :k].T @ u_right[:, :k]).clamp(-1.0, 1.0)
    angles = torch.arccos(values) * 180.0 / math.pi
    return {
        "k": k,
        "mean_deg": float(angles.mean()),
        "max_deg": float(angles.max()),
        "min_deg": float(angles.min()),
    }


def covariance_retention(covariance, matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    order = torch.argsort(eigenvalues, descending=True)[:64]
    values = eigenvalues[order].clamp_min(0)
    vectors = eigenvectors[:, order]
    projected = matrix.double() @ vectors
    retention = projected.pow(2).sum(0) / vectors.pow(2).sum(0).clamp_min(1e-30)
    weighted = float((values * retention).sum() / values.sum().clamp_min(1e-30))
    return weighted, [float(x) for x in retention], [float(x) for x in values]


def bottom_energy(vector, matrix, fraction=0.25):
    left, singular, _ = torch.linalg.svd(matrix.double(), full_matrices=False)
    count = max(1, int(round(len(singular) * fraction)))
    bottom = left[:, -count:]
    vector = vector.double().flatten()
    return float((bottom.T @ vector).pow(2).sum() / vector.pow(2).sum().clamp_min(1e-30))


def covariance_subspace_fraction(covariance, matrix, k):
    _, _, right = torch.linalg.svd(matrix.double(), full_matrices=False)
    basis = right[:k].T
    return float(
        torch.trace(basis.T @ covariance.double() @ basis) / torch.trace(covariance.double()).clamp_min(1e-30)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    runtime = g9.load_runtimes(device)
    recorders = {arm: g9.TapRecorder(runtime[f"{arm.lower()}_rectifier"]) for arm in g9.ARMS}
    p4_store, fs_store = Phase4FStore(g9.hd.CFG, "val"), Store("val")
    dev = load_dev("g0")
    val_cache = load_c1_cache("val", dev["sample_ids"])

    m0, m1 = matrix_payload(runtime["a0_rectifier"]), matrix_payload(runtime["a1_rectifier"])
    result = {"matrices": {}, "comparison": {}, "representation_retention": {}}
    for arm, payload in (("A0", m0), ("A1", m1)):
        result["matrices"][arm] = {name: spectrum(payload[name]) for name in ("out_proj", "proj")}
        result["matrices"][arm]["gamma"] = payload["gamma"]
        result["matrices"][arm]["bias_norm_out"] = float(payload["out_proj_bias"].norm())
        result["matrices"][arm]["bias_norm_proj"] = float(payload["proj_bias"].norm())

    for name in ("out_proj", "proj"):
        w0, w1 = m0[name], m1[name]
        u0, _, _ = torch.linalg.svd(w0.float(), full_matrices=False)
        u1, _, _ = torch.linalg.svd(w1.float(), full_matrices=False)
        result["comparison"][name] = {
            "relative_frobenius_difference": float((w1 - w0).norm() / w0.norm().clamp_min(1e-30)),
            "principal_angles_top32": principal_angles(w0, w1, 32),
            "top_singular_vector_cosines": [
                float(torch.cosine_similarity(u0[:, i], u1[:, i], dim=0)) for i in range(10)
            ],
            "composition_metrics_A0": spectrum(m0["proj"] @ m0["out_proj"]),
            "composition_metrics_A1": spectrum(m1["proj"] @ m1["out_proj"]),
        }

    valid_indices = [i for i, sid in enumerate(dev["sample_ids"]) if bool(val_cache["valid"][i])]
    valid_indices = valid_indices[: args.samples]
    sums = {key: torch.zeros(256, dtype=torch.float64) for key in ("A0", "A1", "D")}
    moments = {key: torch.zeros(256, 256, dtype=torch.float64) for key in ("A0", "A1", "D")}
    token_count = 0
    for begin in range(0, len(valid_indices), args.batch):
        batch = valid_indices[begin:begin + args.batch]
        ids = [dev["sample_ids"][i] for i in batch]
        s64, sc, cc, a0_evidence, a1_evidence = g9.evidence_batch(runtime, p4_store, fs_store, ids, device)
        valid = torch.ones(len(ids), 576, dtype=torch.bool, device=device)
        taps0, _, _ = recorders["A0"].run(s64, a0_evidence, sc, cc, valid)
        taps1, _, _ = recorders["A1"].run(s64, a1_evidence, sc, cc, valid)
        x0 = taps0["R2"].permute(0, 2, 3, 1).reshape(-1, 256).double().cpu()
        x1 = taps1["R2"].permute(0, 2, 3, 1).reshape(-1, 256).double().cpu()
        delta = x1 - x0
        for key, value in (("A0", x0), ("A1", x1), ("D", delta)):
            sums[key] += value.sum(0)
            moments[key] += value.T @ value
        token_count += x0.shape[0]
        print(json.dumps({"stage": "R2_MOMENTS", "done": begin + len(batch), "total": len(valid_indices)}), flush=True)

    for key in ("A0", "A1", "D"):
        mean = sums[key] / token_count
        covariance = moments[key] / token_count - mean[:, None] * mean[None, :]
        result["representation_retention"][key] = {
            "n_tokens": token_count,
            "mean_norm": float(mean.norm()),
            "cov_trace": float(torch.diagonal(covariance).sum()),
            "out_proj": covariance_retention(covariance, m0["out_proj"])[0],
            "proj": covariance_retention(covariance, m0["proj"])[0],
            "composition": covariance_retention(covariance, m0["proj"] @ m0["out_proj"])[0],
            "mean_direction_retention_out_A0": float((m0["out_proj"].double() @ mean).norm() / mean.norm().clamp_min(1e-30)),
            "mean_direction_retention_composition_A0": float(
                ((m0["proj"].double() @ m0["out_proj"].double()) @ mean).norm() / mean.norm().clamp_min(1e-30)
            ),
            "mean_direction_bottom_energy_out_A0": bottom_energy(mean, m0["out_proj"], 0.25),
            "mean_direction_bottom_energy_composition_A0": bottom_energy(mean, m0["proj"] @ m0["out_proj"], 0.25),
        }

    delta_mean = sums["D"] / token_count
    delta_covariance = moments["D"] / token_count - delta_mean[:, None] * delta_mean[None, :]
    for label, matrix in (
        ("A0_out_proj", m0["out_proj"]),
        ("A0_composition", m0["proj"] @ m0["out_proj"]),
        ("A1_out_proj", m1["out_proj"]),
        ("A1_composition", m1["proj"] @ m1["out_proj"]),
    ):
        result["representation_retention"]["D"][label] = covariance_retention(delta_covariance, matrix)[0]
    for label, matrix in (
        ("A0_out", m0["out_proj"]),
        ("A0_composition", m0["proj"] @ m0["out_proj"]),
        ("A1_out", m1["out_proj"]),
        ("A1_composition", m1["proj"] @ m1["out_proj"]),
    ):
        result["representation_retention"]["D"][f"mean_direction_retention_{label}"] = float(
            (matrix.double() @ delta_mean).norm() / delta_mean.norm().clamp_min(1e-30)
        )
        result["representation_retention"]["D"][f"mean_direction_bottom_energy_{label}"] = bottom_energy(
            delta_mean, matrix, 0.25
        )

    result["protocol"] = {
        "samples": len(valid_indices),
        "tokens": token_count,
        "device": str(device),
        "r2_source": "frozen A0/A1 Rectifier R2 on internal validation",
        "no_model_modification": True,
    }
    result["final_interpretation"] = {
        "A0_R2_variance_ratio_after_out_proj": result["representation_retention"]["A0"]["out_proj"],
        "A0_R2_variance_ratio_after_proj": result["representation_retention"]["A0"]["proj"],
        "A0_R2_variance_ratio_after_composition_and_gamma": (m0["gamma"] ** 2)
        * result["representation_retention"]["A0"]["composition"],
        "A1_R2_variance_ratio_after_out_proj": result["representation_retention"]["A1"]["out_proj"],
        "A1_R2_variance_ratio_after_proj": result["representation_retention"]["A1"]["proj"],
        "A1_R2_variance_ratio_after_composition_and_gamma": (m1["gamma"] ** 2)
        * result["representation_retention"]["A1"]["composition"],
        "D_R2_variance_ratio_after_A0_composition_and_gamma": (m0["gamma"] ** 2)
        * result["representation_retention"]["D"]["A0_composition"],
        "D_R2_variance_ratio_after_A1_composition_and_gamma": (m1["gamma"] ** 2)
        * result["representation_retention"]["D"]["A1_composition"],
        "D_mean_direction_norm_ratio_after_A0_composition_and_gamma": m0["gamma"]
        * result["representation_retention"]["D"]["mean_direction_retention_A0_composition"],
        "D_mean_direction_norm_ratio_after_A1_composition_and_gamma": m1["gamma"]
        * result["representation_retention"]["D"]["mean_direction_retention_A1_composition"],
    }
    result["subspace_fractions"] = {}
    for key, covariance in (
        ("A0", moments["A0"] / token_count - (sums["A0"] / token_count)[:, None] * (sums["A0"] / token_count)[None, :]),
        ("A1", moments["A1"] / token_count - (sums["A1"] / token_count)[:, None] * (sums["A1"] / token_count)[None, :]),
        ("D", moments["D"] / token_count - (sums["D"] / token_count)[:, None] * (sums["D"] / token_count)[None, :]),
    ):
        result["subspace_fractions"][key] = {}
        for label, matrix in (("A0_composition", m0["proj"] @ m0["out_proj"]),
                              ("A1_composition", m1["proj"] @ m1["out_proj"])):
            result["subspace_fractions"][key][label] = {
                str(k): covariance_subspace_fraction(covariance, matrix, k) for k in (8, 16, 32, 64, 128)
            }
    output = OUT / "projection_matrix_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
