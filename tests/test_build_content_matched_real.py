import importlib.util
import unittest
from collections import Counter


SPEC = importlib.util.spec_from_file_location(
    "build_content_matched_real", "scripts/data/build_content_matched_real.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildContentMatchedRealTest(unittest.TestCase):
    def test_pass_gpt_replaces_unknown_and_ambiguous_is_ineligible(self):
        real = [{"sample_id": "p", "source": "PASS", "content_category": "unknown"}]
        gpt = [{
            "sample_id": "p", "gpt_label": "ambiguous", "model": "gpt-test",
            "gpt_confidence": 0.9, "gpt_reason": "mixed", "response_id": "r",
        }]
        merged = MODULE.merge_real_labels(real, gpt)
        self.assertEqual(merged[0]["content_category"], "ambiguous")
        self.assertFalse(merged[0]["content_matching_eligible"])

    def test_selection_exactly_matches_fake_counts_without_replacement(self):
        rows = []
        for category in MODULE.CATEGORIES:
            for index in range(4):
                rows.append({"sample_id": f"{category}-{index}", "content_category": category})
        targets = Counter({category: 2 for category in MODULE.CATEGORIES})
        selected = MODULE.select_matched_real(rows, targets, "seed")
        self.assertEqual(Counter(row["content_category"] for row in selected), targets)
        self.assertEqual(len({row["sample_id"] for row in selected}), 8)


if __name__ == "__main__":
    unittest.main()
