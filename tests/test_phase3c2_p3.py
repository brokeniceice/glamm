import numpy as np

from tools.phase3c2 import CONDITIONS, apply_classification_gate, corrupt_rgb


def test_corruptions_are_deterministic_and_geometry_preserving():
    image = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(16, 12, 3)
    for condition in CONDITIONS:
        first = corrupt_rgb(image, condition, seed=3407, sample_id="x")
        second = corrupt_rgb(image, condition, seed=3407, sample_id="x")
        assert first.shape == image.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)
    assert np.array_equal(corrupt_rgb(image, "original", seed=3407, sample_id="x"), image)


def test_gaussian_seed_is_sample_specific():
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    left = corrupt_rgb(image, "gaussian5", seed=3407, sample_id="left")
    right = corrupt_rgb(image, "gaussian5", seed=3407, sample_id="right")
    assert not np.array_equal(left, right)


def test_gate_turns_rejected_fake_into_empty_prediction():
    row = {"tp": 4, "fp": 2, "fn": 3, "intersection": 4, "union": 9,
           "image_iou": 4 / 9, "image_pixel_f1": 8 / 13,
           "foreground_iou": 4 / 9, "foreground_f1": 8 / 13}
    rejected = apply_classification_gate(row, False)
    assert rejected["tp"] == 0 and rejected["fp"] == 0
    assert rejected["fn"] == 7 and rejected["union"] == 7
    assert rejected["foreground_iou"] == 0.0
    accepted = apply_classification_gate(row, True)
    assert accepted["tp"] == 4 and accepted["image_iou"] == 4 / 9
