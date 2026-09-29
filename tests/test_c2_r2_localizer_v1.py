"""Focused Phase6L1 contracts before the formal GPU preflight."""

import math

import torch

from model.c2_r2_localizer_v1 import C2R2LocalizerV1, RADII
from tools.phase3c1 import geometry_for


def _inputs():
    return {"raw_clip_grid": torch.randn(1, 1024, 24, 24),
            "attention_map": torch.softmax(torch.randn(1, 8, 576), dim=-1).reshape(1, 8, 24, 24),
            "evidence_map": torch.randn(1, 512, 24, 24),
            "r_prime": torch.randn(1, 4096), "q_seg": torch.randn(1, 256),
            "s64": torch.randn(1, 256, 64, 64), "z_l": torch.randn(1, 1, 256, 256),
            "clip_geometry": [geometry_for("clip", (512, 768))]}


def test_structured_offsets_have_euclidean_radii_and_distinct_points():
    model = C2R2LocalizerV1()
    bias = model.bridge.offset_head.bias.detach().reshape(8, 4, 2)
    points = 4 * torch.tanh(bias)
    for head in range(8):
        for point, radius in enumerate(RADII):
            assert math.isclose(float(points[head, point].norm()), radius, abs_tol=1e-5)
        assert torch.pdist(points[head]).min() > 0.05
    assert torch.count_nonzero(model.bridge.offset_head.weight) == 0
    assert torch.allclose(model.alpha, torch.full_like(model.alpha, .01))


def test_step_zero_identity_and_second_backward_reaches_alpha():
    torch.manual_seed(3407)
    model = C2R2LocalizerV1().train()
    inputs = _inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(**inputs)
    assert torch.count_nonzero(output["delta_raw"]) == 0
    assert torch.count_nonzero(output["deltaS"]) == 0
    assert torch.equal(output["S_adapt"], inputs["s64"])
    output["S_adapt"].square().mean().backward()
    assert model.adapter.output.weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = model(**inputs)
    output["S_adapt"].square().mean().backward()
    assert model.alpha.grad is not None and torch.isfinite(model.alpha.grad).all()
    assert model.alpha.grad.norm() > 0
    assert model.bridge.offset_head.bias.grad is not None
    assert model.bridge.offset_head.bias.grad.norm() > 0
