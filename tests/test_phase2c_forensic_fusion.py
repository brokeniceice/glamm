from pathlib import Path

import torch
import yaml
from PIL import Image

from model.forensic_fusion import ResidualForensicFusion
from npr_expert.official_npr_srm import OfficialNPRSRM
from npr_expert.transforms import build_npr_transform
from scripts.phase2c_forensic_fusion import balanced_batches


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("npr", "srm", "npr_srm")


def make_inputs(batch=4):
    return {
        "base_logits": torch.randn(batch, 2),
        "h_cls": torch.randn(batch, 16),
        "npr": torch.randn(batch, 8),
        "srm": torch.randn(batch, 8),
    }


def test_alpha_zero_is_bit_exact_baseline_parity():
    inputs = make_inputs()
    for variant in VARIANTS:
        model = ResidualForensicFusion(variant, semantic_dim=16, npr_dim=8, srm_dim=8, d_fuse=4)
        output = model(**inputs)
        assert torch.equal(output.logits, inputs["base_logits"])
        assert float(model.alpha) == 0.0


def test_only_fusion_parameters_are_trainable_and_optimizable():
    frozen_phase2a = torch.nn.Linear(16, 2).requires_grad_(False)
    frozen_expert = OfficialNPRSRM().requires_grad_(False)
    fusion = ResidualForensicFusion("npr_srm", semantic_dim=16, npr_dim=8, srm_dim=8, d_fuse=4)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in fusion.parameters()}
    assert not any(parameter.requires_grad for parameter in frozen_phase2a.parameters())
    assert not any(parameter.requires_grad for parameter in frozen_expert.parameters())
    assert all(parameter.requires_grad for parameter in fusion.parameters())


def test_base_classification_head_is_detached_from_residual_loss():
    inputs = make_inputs()
    inputs["base_logits"].requires_grad_(True)
    model = ResidualForensicFusion("npr", semantic_dim=16, npr_dim=8, d_fuse=4)
    model.alpha.data.fill_(0.1)
    model(**inputs).logits.sum().backward()
    assert inputs["base_logits"].grad is None


def test_expert_branch_decomposition_preserves_historical_joint_forward():
    expert = OfficialNPRSRM().eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = expert(images)
        actual = expert.branch_logits(images)
    assert torch.equal(actual["npr_srm"], expected)
    assert actual["npr"].shape == actual["srm"].shape == (2, 1)


def test_srm_kernels_are_buffers_not_trainable_parameters():
    expert = OfficialNPRSRM()
    assert "srm_kernels" in dict(expert.named_buffers())
    assert "srm_kernels" not in dict(expert.named_parameters())


def test_npr_srm_preprocessing_is_domain_agnostic_and_deterministic():
    transform = build_npr_transform(training=False)
    image = Image.new("RGB", (300, 280), color=(21, 42, 84))
    real = transform(image)
    fake = transform(image)
    assert torch.equal(real, fake)
    assert real.shape == (3, 224, 224)


def test_feature_cache_roundtrip_is_exact(tmp_path):
    payload = {"sample_ids": ["a", "b"], "h_cls": torch.randn(2, 16).half()}
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu")
    assert restored["sample_ids"] == payload["sample_ids"]
    assert torch.equal(restored["h_cls"], payload["h_cls"])


def test_fusion_cannot_mutate_lm_generation_or_masks():
    immutable_outputs = {
        "lm_logits": torch.randn(2, 7, 11),
        "g0_token_ids": torch.randint(0, 10, (2, 5)),
        "g0_masks": torch.randn(2, 8, 8),
        "g1_masks": torch.randn(2, 8, 8),
        "tf_masks": torch.randn(2, 8, 8),
    }
    copies = {key: value.clone() for key, value in immutable_outputs.items()}
    inputs = make_inputs(batch=2)
    model = ResidualForensicFusion("npr_srm", semantic_dim=16, npr_dim=8, srm_dim=8, d_fuse=4)
    model.alpha.data.fill_(1.0)
    assert not torch.equal(model(**inputs).logits, inputs["base_logits"])
    for key, value in immutable_outputs.items():
        assert torch.equal(value, copies[key]), key


def test_joint_gate_only_uses_changed_classification_prediction():
    mask = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool)
    base_gate = torch.tensor([False])
    fusion_gate = torch.tensor([True])
    base_joint = mask & base_gate[:, None, None]
    fusion_joint = mask & fusion_gate[:, None, None]
    assert not torch.equal(base_joint, fusion_joint)
    assert torch.equal(fusion_joint, mask)


def test_balanced_effective_global_batch_is_twenty():
    labels = torch.tensor([0] * 100 + [1] * 100)
    for batch in balanced_batches(labels, steps=10, batch_size=20, seed=3407):
        assert len(batch) == 20
        assert int(labels[batch].sum()) == 10


def test_configs_freeze_base_and_forbid_external_selection():
    for variant in VARIANTS:
        config = yaml.safe_load((ROOT / f"configs/phase2c_{variant}.yaml").read_text())
        assert config["freeze_phase2a"] is True
        assert config["forensics"]["classification_only"] is True
        assert config["forensics"]["expert_frozen"] is True
        assert config["generation_path_modified"] is False
        assert config["localization_path_modified"] is False
        assert config["training"]["selector"] == "min_val_cls_loss"
        assert config["training"]["max_optimizer_steps"] == 1500
        assert set(config["training"]["prohibited_selection_splits"]) == {
            "test", "loki", "raise", "synthscars_official"
        }


def test_source_and_sample_ids_are_not_model_inputs():
    inputs = make_inputs()
    model = ResidualForensicFusion("npr_srm", semantic_dim=16, npr_dim=8, srm_dim=8, d_fuse=4)
    first = model(**inputs).logits
    metadata = {"sample_ids": ["changed"] * 4, "sources": ["other"] * 4}
    assert metadata
    second = model(**inputs).logits
    assert torch.equal(first, second)
