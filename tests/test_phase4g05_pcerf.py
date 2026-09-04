import inspect

import pytest
import torch
import torch.nn.functional as F

from model.pcerf import (
    PCERF,
    PCERFAblation,
    dsmp_probability,
    ecolaf_dempster,
    ecolaf_discount,
    evidence_to_dirichlet,
    resample_clip_to_original_normalized,
    sam_lowres_to_original_normalized,
    tmc_expected_ce_kl,
    tmc_dempster_two,
)


def official_ecolaf_discount(x, classes=2, experts=2):
    count = 0
    indices = torch.zeros((experts, experts - 1), dtype=int, device=x.device)
    for i in range(experts - 1):
        for j in range(i + 1, experts):
            indices[i, j - 1] = count
            indices[j, i] = count
            count += 1
    tab_indices = torch.triu_indices(experts, experts, offset=1, device=x.device)
    const = 1 - (2 * classes + 1) / (classes + 1) ** 2
    lamb = 2
    m = torch.index_select(x, 2, tab_indices[0]) - torch.index_select(x, 2, tab_indices[1])
    m = torch.sum(m**2, 1) + 2.0 / classes * m[:, -1, ...] * torch.sum(m[:, :-1, ...], 1)
    conf = const * torch.sqrt(m / 2.0)
    conf_per_modality = conf[:, indices, ...].sum(dim=2)
    conf_per_modality = conf_per_modality / (conf_per_modality.shape[1] - 1)
    discountings = (1 - conf_per_modality.unsqueeze(1) ** lamb) ** (1 / lamb)
    new_x_om = x[:, :-1, :, ...] * discountings
    new_x = torch.clamp(
        torch.cat((new_x_om, 1 - new_x_om.sum(dim=1).unsqueeze(dim=1)), dim=1),
        min=0.0,
        max=1.0,
    )
    return new_x, conf_per_modality, discountings.squeeze(1)


def official_ecolaf_dempster(m):
    m_omega = torch.cat((m[:, :-1, :, ...] + m[:, None, -1, :, ...], m[:, None, -1, :, ...]), 1)
    tmp = torch.sum(torch.log(torch.relu(m_omega) + 1e-10), 2)
    tmp = tmp - tmp.max(1, keepdim=True).values
    m1 = torch.exp(tmp)
    prod_omega = m1[:, -1, None]
    m1 = m1[:, :-1, ...] - prod_omega
    m1 = torch.cat((m1, prod_omega), 1)
    return F.normalize(m1, p=1.0, dim=1)


def official_dsmp(x, eps=1e-4):
    return (x + (x * x[:, None, -1, ...] + eps * x[:, None, -1, ...]) /
            (1 - x[:, None, -1, ...] + eps * (x.shape[1] - 1)))[:, :-1]


def official_tmc(alpha1, alpha2):
    classes = alpha1.shape[1]
    shape = alpha1.shape
    flat1 = alpha1.movedim(1, -1).reshape(-1, classes)
    flat2 = alpha2.movedim(1, -1).reshape(-1, classes)
    b, u = {}, {}
    for v, alpha in enumerate((flat1, flat2)):
        strength = torch.sum(alpha, dim=1, keepdim=True)
        evidence = alpha - 1
        b[v] = evidence / strength.expand(evidence.shape)
        u[v] = classes / strength
    bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
    conflict = torch.sum(bb, dim=(1, 2)) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    belief = (
        b[0] * b[1] + b[0] * u[1].expand_as(b[0]) + b[1] * u[0].expand_as(b[0])
    ) / (1 - conflict).view(-1, 1).expand_as(b[0])
    uncertainty = u[0] * u[1] / (1 - conflict).view(-1, 1).expand_as(u[0])
    strength = classes / uncertainty
    combined = belief * strength.expand_as(belief) + 1
    return combined.reshape(*shape[:1], *shape[2:], classes).movedim(-1, 1)


def mass_pair(left, right, hw=(3, 5)):
    left = torch.tensor(left, dtype=torch.float32)[None, :, None, None].expand(1, -1, *hw)
    right = torch.tensor(right, dtype=torch.float32)[None, :, None, None].expand(1, -1, *hw)
    return torch.stack((left, right), dim=2)


@pytest.mark.parametrize(
    "masses",
    [
        mass_pair([1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]),
        mass_pair([0.8, 0.1, 0.1], [0.0, 0.0, 1.0]),
        mass_pair([0.0, 0.0, 1.0], [0.1, 0.8, 0.1]),
        mass_pair([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        mass_pair([0.89, 0.01, 0.10], [0.86, 0.02, 0.12]),
        mass_pair([0.999999, 0.0, 0.000001], [0.0, 0.999999, 0.000001]),
    ],
)
def test_ecolaf_official_formula_parity(masses):
    expected_discounted, expected_conflict, expected_discount = official_ecolaf_discount(masses)
    actual_discounted, actual_conflict, actual_discount = ecolaf_discount(masses, classes=2)
    torch.testing.assert_close(actual_discounted, expected_discounted, rtol=0, atol=0)
    torch.testing.assert_close(actual_conflict, expected_conflict, rtol=0, atol=0)
    torch.testing.assert_close(actual_discount, expected_discount, rtol=0, atol=0)
    expected_fused = official_ecolaf_dempster(expected_discounted)
    actual_fused = ecolaf_dempster(actual_discounted)
    torch.testing.assert_close(actual_fused, expected_fused, rtol=0, atol=0)
    torch.testing.assert_close(dsmp_probability(actual_fused), official_dsmp(expected_fused), rtol=0, atol=0)


def test_random_and_extreme_ecolaf_parity_is_finite():
    generator = torch.Generator().manual_seed(3407)
    raw = torch.randn((4, 3, 2, 7, 9), generator=generator) * 1000
    masses = raw.softmax(dim=1)
    expected, _, _ = official_ecolaf_discount(masses)
    actual, _, _ = ecolaf_discount(masses, classes=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fused = ecolaf_dempster(actual)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused.sum(1), torch.ones_like(fused[:, 0]), rtol=1e-6, atol=1e-6)


def test_tmc_parity_and_vacuous_identity():
    generator = torch.Generator().manual_seed(3407)
    e1 = torch.rand((3, 2, 4, 5), generator=generator) * 20
    e2 = torch.rand((3, 2, 4, 5), generator=generator) * 20
    alpha1, alpha2 = e1 + 1, e2 + 1
    torch.testing.assert_close(tmc_dempster_two(alpha1, alpha2), official_tmc(alpha1, alpha2), rtol=2e-6, atol=2e-6)
    vacuous = torch.ones_like(alpha1)
    torch.testing.assert_close(tmc_dempster_two(alpha1, vacuous), alpha1, rtol=1e-6, atol=1e-6)
    opinion = evidence_to_dirichlet(torch.zeros_like(e1))
    assert torch.equal(opinion["masses"][:, -1], torch.ones_like(opinion["masses"][:, -1]))


def test_tmc_source_loss_is_finite_and_differentiable():
    evidence = torch.full((2, 2, 4, 5), 0.5, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 5), generator=torch.Generator().manual_seed(3407))
    losses = tmc_expected_ce_kl(evidence, target, kl_coefficient=0.5)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert evidence.grad is not None and torch.isfinite(evidence.grad).all()


def synthetic_inputs(pattern):
    return dict(
        s64=torch.randn(1, 256, 64, 64),
        q_seg=torch.randn(1, 256),
        z_l=pattern[None, None].contiguous(),
        f24=torch.randn(1, 256, 24, 24),
        z_f24=torch.randn(1, 1, 24, 24) * 20,
        valid_g0=torch.tensor([True]),
    )


def patterns():
    y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    result = {
        "constant": torch.full((256, 256), 0.125),
        "impulse": torch.zeros(256, 256),
        "checkerboard": ((x + y) % 2).float() * 8 - 4,
        "sharp_edge": torch.where(x < 128, -7.0, 7.0),
        "sparse_mask": torch.where(((x - 100) ** 2 + (y - 150) ** 2) < 49, 9.0, -9.0),
        "random": torch.randn((256, 256), generator=torch.Generator().manual_seed(3407)),
        "high_frequency": torch.sin(x.float() * 2.7) + torch.cos(y.float() * 2.3),
    }
    result["impulse"][127, 131] = 100.0
    return result


@pytest.mark.parametrize("condition", ["absent", "vacuous", "off"])
def test_p1_exact_recovery_all_patterns(condition):
    model = PCERF().eval()
    with torch.no_grad():
        for pattern in patterns().values():
            kwargs = synthetic_inputs(pattern)
            kwargs.update(
                forensic_present=torch.tensor([condition != "absent"]),
                forensic_vacuous=torch.tensor([condition == "vacuous"]),
                forensic_off=torch.tensor([condition == "off"]),
            )
            result = model(**kwargs)
            assert torch.equal(result["logits"], kwargs["z_l"])


def test_invalid_g0_cannot_be_rescued_by_strong_forensic_stream():
    model = PCERF().eval()
    kwargs = synthetic_inputs(torch.full((256, 256), -20.0))
    kwargs.update(
        valid_g0=torch.tensor([False]),
        forensic_present=torch.tensor([True]),
        forensic_vacuous=torch.tensor([False]),
        forensic_off=torch.tensor([False]),
    )
    with torch.no_grad():
        result = model(**kwargs)
    assert not bool(result["formal_valid"][0])
    assert torch.equal(result["logits"], torch.zeros_like(result["logits"]))


@pytest.mark.parametrize(
    "ablation",
    [
        PCERFAblation(static_fusion=True),
        PCERFAblation(constant_reliability=True),
        PCERFAblation(sample_only_reliability=True),
        PCERFAblation(conflict_discount=False),
        PCERFAblation(language_stream=False),
        PCERFAblation(forensic_stream=False),
    ],
)
def test_mandatory_fusion_and_stream_ablation_hooks_are_executable(ablation):
    model = PCERF().eval()
    kwargs = synthetic_inputs(torch.zeros((256, 256)))
    kwargs.update(
        forensic_present=torch.tensor([True]),
        forensic_vacuous=torch.tensor([False]),
        forensic_off=torch.tensor([False]),
        ablation=ablation,
    )
    with torch.no_grad():
        output = model(**kwargs)
    assert output["logits"].shape == kwargs["z_l"].shape
    assert torch.isfinite(output["logits"]).all()


def test_forward_api_has_no_oracle_inputs():
    names = set(inspect.signature(PCERF.forward).parameters)
    forbidden = {"gt", "target", "mask", "polygon", "phrase_identity", "tf_identity", "evaluation_label"}
    assert not names.intersection(forbidden)


def test_non_square_clip_support_is_vacuous_outside_crop():
    geometry = {
        "original_hw": [400, 800],
        "resized_hw": [336, 672],
        "crop_box_yxyx": [0, 168, 336, 504],
    }
    masses = torch.zeros((1, 3, 24, 24))
    masses[:, 0] = 0.7
    masses[:, 1] = 0.2
    masses[:, 2] = 0.1
    mapped, support = resample_clip_to_original_normalized(masses, geometry, output_hw=(64, 64), vacuous=True)
    assert bool(support[0, 0, 32, 32])
    assert not bool(support[0, 0, 32, 0])
    assert not bool(support[0, 0, 32, -1])
    assert torch.equal(mapped[0, :, 32, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(mapped.sum(1), torch.ones_like(mapped[:, 0]), rtol=1e-6, atol=1e-6)


def test_sam_padding_removed_before_normalized_coordinates():
    geometry = {"resized_hw": [512, 1024]}
    raw = torch.zeros((1, 1, 256, 256))
    raw[..., :128, :] = 3.0
    raw[..., 128:, :] = -99.0
    normalized = sam_lowres_to_original_normalized(raw, geometry)
    assert torch.equal(normalized, torch.full_like(normalized, 3.0))
