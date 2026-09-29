"""CPU contracts for the new C2-native R2 branch."""

import torch

from model.c2_r2_localizer import C2R2Localizer, GeometryAwareBridge, resample_clip_to_sam_lattice
from tools.phase3c1 import geometry_for


def _inputs(batch=1):
    return {
        "raw_clip_grid": torch.randn(batch, 1024, 24, 24),
        "attention_map": torch.softmax(torch.randn(batch, 8, 576), dim=-1).reshape(batch, 8, 24, 24),
        "evidence_map": torch.randn(batch, 512, 24, 24),
        "r_prime": torch.randn(batch, 1, 4096),
        "q_seg": torch.randn(batch, 256),
        "s64": torch.randn(batch, 256, 64, 64),
        "z_l": torch.randn(batch, 1, 256, 256),
        "clip_geometry": [geometry_for("clip", (512, 768)) for _ in range(batch)],
    }


def test_full_shape_and_step_zero_identity():
    torch.manual_seed(3407)
    with torch.no_grad():
        model = C2R2Localizer().eval()
        inputs = _inputs(2)
        output = model(**inputs)
    for name, expected in {"F24_star": (2, 256, 24, 24), "F64": (2, 256, 64, 64),
                           "R64": (2, 256, 64, 64), "deltaS": (2, 256, 64, 64),
                           "S_adapt": (2, 256, 64, 64)}.items():
        assert output[name].shape == expected
    assert torch.count_nonzero(output["deltaS"]) == 0
    assert torch.equal(output["S_adapt"], inputs["s64"])
    assert all(torch.isfinite(value).all() for value in output.values())


def test_geometry_support_respects_crop_and_sam_padding():
    feature = torch.ones(1, 256, 24, 24)
    mapped, support = resample_clip_to_sam_lattice(feature, [geometry_for("clip", (512, 768))])
    assert 0 < support.sum() < 64 * 64
    assert torch.count_nonzero(mapped.masked_select(~support.expand_as(mapped))) == 0
    assert not support[0, 0, -1].any()  # SAM padding on the short side


def test_deformable_validity_and_all_invalid_queries():
    torch.manual_seed(3407)
    bridge = GeometryAwareBridge().eval()
    s = torch.randn(1, 256, 64, 64)
    z = torch.zeros(1, 1, 64, 64)
    g = torch.zeros(1, 256)
    with torch.no_grad():
        all_valid = bridge(s, z, s, torch.ones_like(z, dtype=torch.bool), g)
        assert torch.count_nonzero(all_valid["bridge_offsets"]) == 0
        assert all_valid["bridge_valid"].all()
        assert torch.allclose(all_valid["bridge_attention"].sum(dim=2),
                              torch.ones_like(all_valid["bridge_attention"].sum(dim=2)))
        partial = torch.zeros_like(z, dtype=torch.bool)
        partial[:, :, :32] = True
        partial_result = bridge(s, z, s, partial, g)
        assert torch.count_nonzero(partial_result["R64"][:, :, 32:]) == 0
        invalid = bridge(s, z, s, torch.zeros_like(partial), g)
        assert torch.count_nonzero(invalid["R64"]) == 0
        assert torch.count_nonzero(invalid["bridge_attention"]) == 0
        assert not invalid["bridge_valid"].any()
        assert all(torch.isfinite(v).all() for v in (all_valid["R64"], partial_result["R64"], invalid["R64"]))


def test_zero_head_first_gradient_then_upstream_gradient():
    torch.manual_seed(3407)
    model = C2R2Localizer().train()
    inputs = _inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(**inputs)
    (output["S_adapt"].square().mean()).backward()
    assert model.adapter.output.weight.grad is not None
    assert model.adapter.output.weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = model(**inputs)
    (output["S_adapt"].square().mean()).backward()
    for module in (model.evidence, model.bridge, model.conditioner, *model.adapter.blocks, model.adapter.output):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
                   for p in module.parameters())
