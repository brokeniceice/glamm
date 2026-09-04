import inspect

import pytest
import torch

from model.csculf import CSCULF, adapted_ecolaf_fuse
from model.pcerf import dsmp_probability, ecolaf_fuse


def full_geometry():
    return {"resized_hw": [224, 224], "crop_box_yxyx": [0, 0, 224, 224]}


def partial_geometry():
    return {"resized_hw": [256, 320], "crop_box_yxyx": [16, 48, 240, 272]}


def inputs(batch=1, geometry=None):
    generator = torch.Generator().manual_seed(3407)
    return {
        "s64": torch.randn(batch, 256, 64, 64, generator=generator),
        "q_seg": torch.randn(batch, 256, generator=generator),
        "z_l": torch.randn(batch, 1, 256, 256, generator=generator),
        "f24": torch.randn(batch, 256, 24, 24, generator=generator),
        "z_f24": torch.randn(batch, 1, 24, 24, generator=generator),
        "clip_geometries": [geometry or full_geometry() for _ in range(batch)],
        "valid_g0": torch.ones(batch, dtype=torch.bool),
        "forensic_present": torch.ones(batch, dtype=torch.bool),
        "forensic_vacuous": torch.zeros(batch, dtype=torch.bool),
        "forensic_off": torch.zeros(batch, dtype=torch.bool),
    }


def random_mass(batch=2, height=8, width=8):
    generator = torch.Generator().manual_seed(3407)
    value = torch.rand(batch, 3, height, width, generator=generator)
    return value / value.sum(1, keepdim=True)


def test_adapted_utility_one_is_official_ecolaf_exact():
    left, right = random_mass(), random_mass()
    result = adapted_ecolaf_fuse(left, right, torch.ones_like(left[:, :1]))
    official = ecolaf_fuse(torch.stack((left, right), dim=2), classes=2)
    assert torch.equal(result["official_discounted_masses"], result["adapted_discounted_masses"])
    assert torch.equal(result["fused_mass"], official["fused_mass"])
    assert torch.equal(result["probability"], official["probability"])


def test_adapted_utility_zero_is_strict_language_only():
    left, right = random_mass(), random_mass()
    result = adapted_ecolaf_fuse(left, right, torch.zeros_like(left[:, :1]))
    assert torch.equal(result["fused_mass"], left)
    assert torch.equal(result["probability"], dsmp_probability(left))
    assert torch.equal(result["effective_forensic_contribution"], torch.zeros_like(result["effective_forensic_contribution"]))


def test_effective_forensic_contribution_monotonic():
    left, right = random_mass(batch=1), random_mass(batch=1)
    values = []
    for utility in (0.0, 0.25, 0.5, 0.75, 1.0):
        values.append(adapted_ecolaf_fuse(left, right, torch.full_like(left[:, :1], utility))["effective_forensic_contribution"])
    assert all(bool((right_value >= left_value).all()) for left_value, right_value in zip(values, values[1:]))


def test_forward_shapes_support_and_explicit_features():
    model = CSCULF().eval(); kwargs = inputs(geometry=partial_geometry())
    with torch.no_grad(): result = model(**kwargs)
    assert result["L64"].shape == (1, 64, 64, 64)
    assert result["Fctx64"].shape == (1, 64, 64, 64)
    assert result["comparison"].shape == (1, 263, 64, 64)
    assert result["U_F64"].shape == (1, 1, 64, 64)
    assert result["U_F256"].shape == (1, 1, 256, 256)
    assert bool((result["U_F64"][~result["support64"]] == 0).all())
    assert bool((result["mass_F64"][:, :-1][~result["support64"].expand(-1, 2, -1, -1)] == 0).all())
    assert bool((result["mass_F64"][:, -1:][~result["support64"]] == 1).all())
    for key in ("Lr", "Fr", "product", "absolute_difference", "local_cosine", "p_L", "p_F", "conflict"):
        assert key in result["comparison_parts"]


@pytest.mark.parametrize("condition", ["absent", "vacuous", "off", "unsupported"])
def test_exact_p1_fallback(condition):
    geometry = {"resized_hw": [224, 224], "crop_box_yxyx": [300, 300, 400, 400]} if condition == "unsupported" else full_geometry()
    model = CSCULF().eval(); kwargs = inputs(geometry=geometry)
    kwargs["forensic_present"][:] = condition != "absent"
    kwargs["forensic_vacuous"][:] = condition == "vacuous"
    kwargs["forensic_off"][:] = condition == "off"
    with torch.no_grad(): result = model(**kwargs)
    assert torch.equal(result["logits"], kwargs["z_l"])


def test_invalid_g0_policy_cannot_be_rescued():
    model = CSCULF().eval(); kwargs = inputs(); kwargs["valid_g0"][:] = False
    with torch.no_grad(): result = model(**kwargs)
    assert not bool(result["formal_valid"][0])
    assert torch.equal(result["logits"], torch.zeros_like(kwargs["z_l"]))


def test_spatial_shuffle_changes_all_registered_compatibility_tensors():
    model = CSCULF().eval(); kwargs = inputs()
    permutation = torch.randperm(24 * 24, generator=torch.Generator().manual_seed(3407))
    shuffled = dict(kwargs)
    shuffled["f24"] = kwargs["f24"].flatten(2)[:, :, permutation].reshape_as(kwargs["f24"])
    shuffled["z_f24"] = kwargs["z_f24"].flatten(2)[:, :, permutation].reshape_as(kwargs["z_f24"])
    with torch.no_grad(): matched, changed = model(**kwargs), model(**shuffled)
    for key in ("product", "absolute_difference", "local_cosine"):
        assert not torch.equal(matched["comparison_parts"][key], changed["comparison_parts"][key])
    assert not torch.equal(matched["exchange"]["attention_L_from_F"], changed["exchange"]["attention_L_from_F"])
    assert not torch.equal(matched["comparison"], changed["comparison"])
    assert not torch.equal(matched["utility_prehead"], changed["utility_prehead"])


def test_cross_image_forensic_changes_interaction_with_language_fixed():
    model = CSCULF().eval(); kwargs = inputs(batch=2)
    crossed = dict(kwargs); crossed["f24"] = kwargs["f24"].flip(0); crossed["z_f24"] = kwargs["z_f24"].flip(0)
    with torch.no_grad(): matched, changed = model(**kwargs), model(**crossed)
    assert torch.equal(matched["L64"], changed["L64"])
    for key in ("Fr", "comparison", "utility_prehead"):
        assert not torch.equal(matched[key], changed[key])


def test_utility_intervention_preserves_sources_and_features_bit_exact():
    model = CSCULF().eval(); kwargs = inputs(batch=2)
    with torch.no_grad(): matched = model(**kwargs)
    replacement = matched["U_F64"].flip(0).clone()
    with torch.no_grad(): intervened = model(**kwargs, utility_override=replacement)
    for key in ("p_L64", "p_F64", "mass_L64", "mass_F64", "mass_L256", "mass_F256", "L64", "Fctx64", "aligned_F64", "aligned_z_F64"):
        assert torch.equal(matched[key], intervened[key]), key
    assert bool((intervened["U_F64"][~intervened["support64"]] == 0).all())


def test_gradient_isolation_and_no_refinement_parameters():
    model = CSCULF().train(); kwargs = inputs()
    source_before = {name: value.detach().clone() for name, value in model.language_source.state_dict().items()}
    source_before.update({f"F.{name}": value.detach().clone() for name, value in model.forensic_source.state_dict().items()})
    result = model(**kwargs); loss = result["U_F64"].mean() + result["fused"]["probability"].mean(); loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    assert trainable and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool((parameter.grad != 0).any()) for _, parameter in trainable)
    assert all(parameter.grad is None or bool((parameter.grad == 0).all()) for _, parameter in frozen)
    assert not any("refinement" in name.lower() for name, _ in model.named_parameters())
    for name, value in model.language_source.state_dict().items(): assert torch.equal(value, source_before[name])
    for name, value in model.forensic_source.state_dict().items(): assert torch.equal(value, source_before[f"F.{name}"])


def test_no_oracle_or_condition_identity_in_forward_signature():
    names = set(inspect.signature(CSCULF.forward).parameters)
    forbidden = {"gt", "target", "mask", "polygon", "phrase", "tf_identity", "condition", "evaluation_label"}
    assert names.isdisjoint(forbidden)
