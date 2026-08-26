import torch

from model.sam_forensic_rectifier import GeometryAwareSAMRectifier


def tensors(batch=1):
    generator = torch.Generator().manual_seed(3407)
    s64 = torch.randn(batch, 256, 64, 64, generator=generator)
    f24 = torch.randn(batch, 256, 24, 24, generator=generator)
    sc = torch.rand(batch, 4096, 2, generator=generator)
    fc = torch.rand(batch, 576, 2, generator=generator)
    return s64, f24, sc, fc


def test_rectifier_off_is_exact_tensor_bypass():
    model = GeometryAwareSAMRectifier(0.01)
    s64, f24, sc, fc = tensors()
    output = model(s64, f24, sc, fc, enabled=False)
    assert torch.equal(output["image_embeddings"], s64)
    assert torch.count_nonzero(output["residual"]) == 0


def test_only_rectifier_is_trainable_and_receives_gradients():
    model = GeometryAwareSAMRectifier(0.03)
    s64, f24, sc, fc = tensors()
    output = model(s64, f24, sc, fc)
    output["image_embeddings"].square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_support_has_token_shape():
    model = GeometryAwareSAMRectifier(0.01)
    s64, f24, sc, fc = tensors(2)
    output = model(s64, f24, sc, fc)
    assert output["support"].shape == (2, 4096)
    assert output["image_embeddings"].shape == s64.shape
