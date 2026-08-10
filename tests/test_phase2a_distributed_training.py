import unittest

import torch

from dataset.forensics.distributed import sampler_epoch_indices
from dataset.forensics.unified import CANONICAL_PROMPT_SHA256, CANONICAL_PROMPT_TEMPLATE_ID
from tools.distributed_loss import globally_normalized_components


class DummyDataset:
    def __init__(self, pairs=20):
        self.rows = [
            {"class_label": label, "sample_id": f"{label}:{index}"}
            for label in (0, 1) for index in range(pairs)
        ]

    def __len__(self):
        return len(self.rows)


class Phase2ADistributedTrainingTest(unittest.TestCase):
    def test_world_size_two_sampler_has_no_rank_overlap_and_is_balanced(self):
        dataset = DummyDataset()
        ranks = sampler_epoch_indices(dataset, world_size=2, seed=3407, epoch=0)
        self.assertTrue(set(ranks[0]).isdisjoint(ranks[1]))
        self.assertEqual(set(ranks[0]) | set(ranks[1]), set(range(len(dataset))))
        for indices in ranks:
            for start in range(0, len(indices), 2):
                self.assertEqual({dataset.rows[i]["class_label"] for i in indices[start:start + 2]}, {0, 1})

    def test_distributed_sampler_is_deterministic_and_changes_by_epoch(self):
        dataset = DummyDataset()
        first = sampler_epoch_indices(dataset, world_size=2, seed=3407, epoch=0)
        repeated = sampler_epoch_indices(dataset, world_size=2, seed=3407, epoch=0)
        next_epoch = sampler_epoch_indices(dataset, world_size=2, seed=3407, epoch=1)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)

    @staticmethod
    def _scaled_average(values, counts, count_key, loss_key, world_size, gas):
        global_count = sum(count[count_key] for count in counts)
        total = 0.0
        for value, count in zip(values, counts):
            output = {
                "ce_loss": torch.tensor(0.0), "cls_loss": torch.tensor(0.0),
                "mask_bce_loss": torch.tensor(0.0), "mask_dice_loss": torch.tensor(0.0),
            }
            output[loss_key] = torch.tensor(value)
            scaled = globally_normalized_components(
                output, count,
                {"text_tokens": sum(c["text_tokens"] for c in counts),
                 "classification_samples": sum(c["classification_samples"] for c in counts),
                 "valid_masks": sum(c["valid_masks"] for c in counts)},
                world_size=world_size, accumulation_steps=gas,
            )
            total += float(scaled[loss_key]) / (world_size * gas)
        expected = sum(value * count[count_key] for value, count in zip(values, counts)) / global_count
        return total, expected

    def test_distributed_text_token_global_normalization(self):
        counts = [
            {"text_tokens": 10, "classification_samples": 2, "valid_masks": 1},
            {"text_tokens": 100, "classification_samples": 2, "valid_masks": 1},
            {"text_tokens": 20, "classification_samples": 2, "valid_masks": 1},
            {"text_tokens": 200, "classification_samples": 2, "valid_masks": 1},
        ]
        actual, expected = self._scaled_average([1, 2, 3, 4], counts, "text_tokens", "ce_loss", 2, 2)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_distributed_mask_global_valid_count_normalization(self):
        counts = [
            {"text_tokens": 1, "classification_samples": 2, "valid_masks": 0},
            {"text_tokens": 1, "classification_samples": 2, "valid_masks": 1},
            {"text_tokens": 1, "classification_samples": 2, "valid_masks": 2},
            {"text_tokens": 1, "classification_samples": 2, "valid_masks": 1},
        ]
        actual, expected = self._scaled_average(
            [0, 1, 3, 5], counts, "valid_masks", "mask_bce_loss", 2, 2
        )
        self.assertAlmostEqual(actual, expected, places=6)

    def test_real_only_distributed_mask_loss_is_zero(self):
        output = {key: torch.tensor(2.0) for key in (
            "ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss"
        )}
        counts = {"text_tokens": 20, "classification_samples": 4, "valid_masks": 0}
        scaled = globally_normalized_components(output, counts, counts, world_size=2, accumulation_steps=5)
        self.assertEqual(float(scaled["mask_bce_loss"]), 0.0)
        self.assertEqual(float(scaled["mask_dice_loss"]), 0.0)

    def test_classification_distributed_parity(self):
        counts = [{"text_tokens": 1, "classification_samples": 2, "valid_masks": 1}] * 10
        actual, expected = self._scaled_average(
            list(range(1, 11)), counts, "classification_samples", "cls_loss", 2, 5
        )
        self.assertAlmostEqual(actual, expected, places=6)

    def test_effective_batch_and_exposure_equality(self):
        single = 2 * 1 * 10
        dual = 2 * 2 * 5
        self.assertEqual(single, dual)
        self.assertEqual(single * 5000, dual * 5000)
        self.assertEqual(dual * 5000, 100000)

    def test_canonical_prompt_fingerprint_is_frozen(self):
        self.assertEqual(CANONICAL_PROMPT_TEMPLATE_ID, "unified_forensics_v1")
        self.assertEqual(
            CANONICAL_PROMPT_SHA256,
            "32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d",
        )


if __name__ == "__main__":
    unittest.main()
