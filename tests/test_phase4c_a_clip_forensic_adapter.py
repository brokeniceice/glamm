import torch
from model.clip_forensic_adapter import CLIPSpatialArm, LocalForensicBlock


def test_phase4c_a_shapes_and_residual_blocks():
    x=torch.randn(2,1024,24,24)
    b=CLIPSpatialArm(blocks=0); c=CLIPSpatialArm(blocks=3)
    assert b(x).shape == c(x).shape == (2,1,24,24)
    assert len(c.forensic_blocks)==3
    assert all(isinstance(v,LocalForensicBlock) for v in c.forensic_blocks)
    assert all(v.depthwise.groups==256 and v.depthwise.kernel_size==(3,3) for v in c.forensic_blocks)


def test_phase4c_a_projection_head_can_be_matched():
    torch.manual_seed(3407); p=torch.nn.Conv2d(1024,256,1); h=torch.nn.Conv2d(256,1,1)
    b=CLIPSpatialArm(blocks=0); c=CLIPSpatialArm(blocks=3)
    for m in (b,c): m.projection.load_state_dict(p.state_dict()); m.dense_head.load_state_dict(h.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(b.projection.parameters(),c.projection.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(b.dense_head.parameters(),c.dense_head.parameters()))

