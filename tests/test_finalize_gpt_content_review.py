import importlib.util
import unittest


SPEC = importlib.util.spec_from_file_location(
    "finalize_gpt_content_review", "scripts/data/finalize_gpt_content_review.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeGPTContentReviewTest(unittest.TestCase):
    def test_gpt_label_wins_even_when_human_disagrees(self):
        rows = [{
            "review_index": 0,
            "sample_id": "x",
            "source": "source",
            "domain": "real_candidate",
            "gpt_label": "animal",
            "model": "gpt-test",
            "gpt_confidence": 0.9,
            "gpt_reason": "animal dominates",
            "clip_label": "object",
            "clip_margin": 0.01,
            "human_label": "human",
            "human_status": "reviewed",
            "response_id": "response",
        }]
        final_rows, summary = MODULE.finalize(rows)
        self.assertEqual(final_rows[0]["final_content_label"], "animal")
        self.assertEqual(final_rows[0]["label_authority"], "gpt")
        self.assertEqual(final_rows[0]["human_label_ignored"], "human")
        self.assertEqual(summary["clip_gpt_agreement_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
