import importlib.util
import unittest


SPEC = importlib.util.spec_from_file_location("analyze_content_labels", "scripts/data/analyze_content_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContentLabelAnalysisTest(unittest.TestCase):
    def test_low_margin_and_view_disagreement_are_flagged(self):
        rows = [{
            "sample_id": "x",
            "source": "PASS",
            "domain": "real_candidate",
            "source_content_category": "unknown",
            "predicted_category": "animal",
            "top1_top2_margin": 0.01,
            "view_agreement": False,
        }]
        thresholds = {category: {"margin": 0.02} for category in MODULE.CATEGORIES}
        annotated = MODULE.annotate_predictions(rows, thresholds)[0]
        self.assertIn("low_calibrated_margin", annotated["review_reasons"])
        self.assertIn("view_disagreement", annotated["review_reasons"])
        self.assertEqual(annotated["review_status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
