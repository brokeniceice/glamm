import importlib.util
import unittest
from collections import Counter


SPEC = importlib.util.spec_from_file_location(
    "split_unified_forensics", "scripts/data/split_unified_forensics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitUnifiedForensicsTest(unittest.TestCase):
    def test_category_targets_sum_and_follow_eight_one_one(self):
        self.assertEqual(MODULE.category_split_targets(5824), {"train": 4659, "val": 583, "test": 582})
        self.assertEqual(sum(MODULE.category_split_targets(1941).values()), 1941)

    def test_phash_pair_is_never_split(self):
        rows = [
            {"sample_id": f"x-{index}", "content_category": "human"}
            for index in range(10)
        ]
        pairs = [{"left_sample_id": "x-0", "right_sample_id": "x-1"}]
        assignment = MODULE.assign_category_groups(rows, pairs, "human", "seed")
        self.assertEqual(assignment["x-0"], assignment["x-1"])
        self.assertEqual(Counter(assignment.values()), Counter({"train": 8, "val": 1, "test": 1}))

    def test_official_overlap_removes_fake_and_category_matched_real(self):
        real = [
            {"sample_id": "real-human", "content_category": "human"},
            {"sample_id": "real-animal", "content_category": "animal"},
        ]
        fake = [
            {"sample_id": "fake-human", "content_category": "human"},
            {"sample_id": "fake-animal", "content_category": "animal"},
        ]
        leakage = [{
            "left_sample_id": "fake-human", "right_sample_id": "official-human",
            "left_domain": "synthscars_fake", "right_domain": "synthscars_official_test",
        }]
        kept_real, kept_fake, audit = MODULE.remove_official_test_leakage(real, fake, leakage, "seed")
        self.assertEqual([row["content_category"] for row in kept_real], ["animal"])
        self.assertEqual([row["sample_id"] for row in kept_fake], ["fake-animal"])
        self.assertEqual(audit["post_mitigation_cross_leakage_count"], 0)


if __name__ == "__main__":
    unittest.main()
