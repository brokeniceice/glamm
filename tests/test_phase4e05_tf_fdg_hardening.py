import pytest
import torch

from model.tf_fdg import (
    CrossAttentiveSemanticRectification,
    TFFDGStudent,
    clone_frozen_teacher,
    coordinate_resample,
    normalized_grid,
    region_aware_slot_loss,
    soft_union,
)


def test_soft_union_is_probabilistic_or_not_weighted_average():
    logits = torch.tensor([[[[0.0]], [[0.0]]]])
    probability, continuous_logit = soft_union(logits)
    assert probability.item() == pytest.approx(0.75)
    assert continuous_logit.sigmoid().item() == pytest.approx(0.75)


def test_region_supervision_has_explicit_k4_overflow_rule():
    logits = torch.randn(4, 16, 16)
    within = torch.zeros(3, 16, 16); within[0, 1:3, 1:3] = 1; within[1, 6:8, 6:8] = 1; within[2, 11:13, 11:13] = 1
    overflow = torch.cat((within, within[:2].roll(1, -1)), dim=0)
    assert region_aware_slot_loss(logits, within)["mode"] == "region_hungarian"
    assert region_aware_slot_loss(logits, overflow)["mode"] == "union_only_overflow"


def test_rectification_rejects_zero_gamma_and_preserves_gradient():
    with pytest.raises(ValueError):
        CrossAttentiveSemanticRectification(gamma_init=0.0)
    module = CrossAttentiveSemanticRectification(gamma_init=0.01)
    semantic, forensic = torch.randn(1, 16, 256), torch.randn(1, 9, 256)
    output, _, _ = module(semantic, forensic, normalized_grid(4, 4), normalized_grid(3, 3))
    output.square().mean().backward()
    assert module.gamma.grad is not None and module.gamma.grad.abs().item() > 0
    for projection in (module.cross_attention.q_proj, module.cross_attention.k_proj, module.cross_attention.v_proj):
        assert projection.weight.grad is not None and projection.weight.grad.norm().item() > 0


def test_coordinate_resample_marks_out_of_field_locations_invalid():
    tokens = torch.ones(1, 4, 8)
    source = torch.tensor([[0.40, 0.40], [0.60, 0.40], [0.40, 0.60], [0.60, 0.60]])
    target = torch.tensor([[0.50, 0.50], [0.05, 0.05]])
    values, support = coordinate_resample(tokens, source, target)
    assert support.tolist() == [[True, False]]
    assert values[0, 0].sum() > 0 and values[0, 1].sum() == 0


def test_teacher_copy_is_exact_and_frozen():
    torch.manual_seed(3407)
    student = TFFDGStudent()
    teacher = clone_frozen_teacher(student)
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in student.state_dict().items())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_registered_ablation_flags_preserve_output_contract():
    hidden = torch.randn(1, 4096)
    semantic = torch.randn(1, 256, 64, 64)
    forensic = torch.randn(1, 256, 24, 24)
    for use_forensic, use_rectification in ((False, False), (True, False)):
        model = TFFDGStudent(use_forensic=use_forensic, use_rectification=use_rectification)
        output = model(hidden, semantic, forensic)
        assert output["slot_logits"].shape == (1, 4, 32, 32)
        assert output["union_logit"].shape == (1, 1, 32, 32)
        assert torch.isfinite(output["union_logit"]).all()
        if not use_rectification:
            assert torch.count_nonzero(output["rectification_residual"]) == 0
