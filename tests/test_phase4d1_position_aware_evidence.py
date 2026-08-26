import torch

from model.position_aware_evidence_reader import (
    PositionAwareEvidenceReader,
    fixed_2d_sincos_position,
)


def test_fixed_position_shape_and_determinism():
    a = fixed_2d_sincos_position()
    b = fixed_2d_sincos_position()
    assert a.shape == (576, 256)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    assert a.std() > 0


def test_zero_feature_retains_fixed_position():
    reader = PositionAwareEvidenceReader()
    value = reader.positioned_source(torch.zeros(2, 256, 24, 24))
    assert value.shape == (2, 576, 256)
    assert torch.equal(value[0], reader.position_2d)
    assert torch.equal(value[0], value[1])


def test_shuffle_content_does_not_shuffle_position_lattice():
    reader = PositionAwareEvidenceReader()
    feature = (torch.arange(576, dtype=torch.float32) / 576).reshape(1, 576, 1).expand(-1, -1, 256)
    permutation = torch.arange(575, -1, -1)
    correct = reader.positioned_source(feature[:, permutation])
    forbidden = reader.positioned_source(feature)[:, permutation]
    assert not torch.equal(correct, forbidden)
    assert torch.allclose(correct - feature[:, permutation], reader.position_2d[None],
                          atol=1e-7, rtol=0)


def test_beta_zero_is_exact_query_identity():
    reader = PositionAwareEvidenceReader()
    q = torch.randn(3, 256)
    spatial = torch.randn(3, 256, 24, 24)
    result = reader(q, spatial)
    assert torch.equal(result["q_final"][:, 0], q)
    assert torch.count_nonzero(result["q_residual"]) == 0
