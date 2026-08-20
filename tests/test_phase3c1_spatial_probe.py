import torch

from npr_expert.official_npr_srm import OfficialNPRSRM
from model.GLaMM import calculate_dice_loss, compute_sigmoid_cross_entropy
from tools.phase3c1 import binary_metrics, geometry_for, inverse_logits, probe_loss, transform_mask


def test_npr_and_srm_spatial_apis_preserve_historical_pooled_features():
    torch.manual_seed(3407)
    model = OfficialNPRSRM().eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        old_npr = model.extract_npr_features(images)
        old_srm = model.extract_srm_features(images)
        spatial_npr = model.extract_npr_spatial_features(images)
        spatial_srm = model.extract_srm_spatial_features(images)
    assert spatial_npr.ndim == spatial_srm.ndim == 4
    assert torch.equal(old_npr, model.avgpool(spatial_npr).flatten(1).float())
    assert torch.equal(old_srm, model.srm_pool(spatial_srm).flatten(1).float())


def test_geometry_round_trip_preserves_visible_full_foreground():
    mask = torch.ones(1, 80, 120)
    for source in ("sam", "clip", "npr", "srm", "focal"):
        geometry = geometry_for(source, (80, 120))
        transformed = transform_mask(mask, geometry)
        restored = inverse_logits(transformed.float().mul(200).sub(100), geometry)
        prediction = restored.gt(0)
        if source in {"clip", "npr", "srm"}:
            assert prediction.any()
            assert not prediction.all()  # unseen crop exterior is forced background
        else:
            assert prediction.all()


def test_probe_loss_fixed_weighting_and_metrics():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    losses = probe_loss(logits, target)
    assert torch.allclose(losses["total"], 2.0 * losses["bce"] + 0.5 * losses["dice"])
    assert torch.allclose(losses["bce"], compute_sigmoid_cross_entropy(logits[:, 0], target[:, 0], 1))
    assert torch.allclose(losses["dice"], calculate_dice_loss(logits[:, 0], target[:, 0], 1))
    metrics = binary_metrics(logits[0, 0], target[0, 0])
    assert metrics["foreground_iou"] == 1.0
    assert metrics["foreground_f1"] == 1.0
