import torch
from model.evidence_reader import EvidenceReader


def test_reader_zero_beta_exact_identity():
    torch.manual_seed(3407); model=EvidenceReader(); q=torch.randn(3,256); s=torch.randn(3,256,24,24)
    out=model(q,s)
    assert torch.equal(out["q_final"][:,0],q)
    assert out["attention"].shape==(3,4,1,576)


def test_reader_two_step_gradient_behavior():
    torch.manual_seed(3407); model=EvidenceReader(); q=torch.randn(2,256); s=torch.randn(2,256,24,24)
    loss=model(q,s)["q_final"].square().mean(); loss.backward()
    assert model.beta.grad is not None and model.beta.grad.abs()>0
    assert all(p.grad is None or torch.equal(p.grad,torch.zeros_like(p.grad)) for n,p in model.named_parameters() if n!="beta")
    with torch.no_grad(): model.beta.fill_(0.1)
    model.zero_grad(); model(q,s)["q_final"].square().mean().backward()
    assert any(p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters() if "cross_attention" in n)

