from tools.phase3d2 import TRAINABLE_GROUPS, is_spatial_trainable, parameter_group, stable_fake_schedule


def test_strict_trainable_boundary():
    assert is_spatial_trainable("base.model.text_hidden_fcs.0.0.weight")
    assert is_spatial_trainable("base.model.grounding_encoder.mask_decoder.mask_tokens.weight")
    for name in (
        "base.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base.model.embed_tokens.weight", "base.lm_head.weight",
        "base.classification_head.weight", "base.model.vision_tower.weight",
        "base.model.mm_projector.weight", "base.model.region_encoder.weight",
        "base.model.grounding_encoder.image_encoder.patch_embed.proj.weight",
    ):
        assert not is_spatial_trainable(name)


def test_parameter_groups_are_exhaustive_for_key_boundaries():
    assert TRAINABLE_GROUPS == ("text_hidden_fcs", "mask_decoder")
    assert parameter_group("x.grounding_encoder.mask_decoder.a") == "mask_decoder"
    assert parameter_group("x.grounding_encoder.image_encoder.a") == "grounding_image_encoder"
    assert parameter_group("x.grounding_encoder.prompt_encoder.a") == "other_grounding_encoder"


def test_fake_schedule_is_unique_deterministic_and_fake_only():
    rows = [
        {"sample_id": "r", "class_label": 0},
        {"sample_id": "f1", "class_label": 1},
        {"sample_id": "f2", "class_label": 1},
        {"sample_id": "f3", "class_label": 1},
    ]
    first = stable_fake_schedule(rows, 3407, 2)
    assert first == stable_fake_schedule(rows, 3407, 2)
    assert len(first) == len(set(first)) == 2
    assert all(rows[index]["class_label"] == 1 for index in first)
