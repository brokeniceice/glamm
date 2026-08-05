import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "review_content_labels", "scripts/data/review_content_labels.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"test")
        write_jsonl(self.root / "review_candidates.jsonl", [{
            "sample_id": "sample-1",
            "source": "test",
            "domain": "real_candidate",
            "image_path": str(self.image),
            "top1_top2_margin": 0.01,
            "view_predictions": ["animal", "object"],
            "view_agreement": False,
            "review_reasons": ["view_disagreement"],
        }])
        write_jsonl(self.root / "manual_corrections_template.jsonl", [{
            "sample_id": "sample-1",
            "auto_label": "animal",
            "manual_label": None,
            "reviewer_status": "pending",
            "review_reason": ["view_disagreement"],
        }])

    def tearDown(self):
        self.temporary.cleanup()

    def test_update_is_persisted_and_public_state_reflects_it(self):
        store = MODULE.ReviewStore(self.root)
        store.update(0, "human", "reviewed")
        persisted = MODULE.load_jsonl(self.root / "manual_corrections_template.jsonl")[0]
        self.assertEqual(persisted["manual_label"], "human")
        self.assertEqual(persisted["reviewer_status"], "reviewed")
        self.assertEqual(store.public_state()["rows"][0]["manual_label"], "human")

    def test_reject_requires_null_label(self):
        store = MODULE.ReviewStore(self.root)
        with self.assertRaises(ValueError):
            store.update(0, "animal", "rejected")

    def test_candidate_order_must_match_correction_order(self):
        write_jsonl(self.root / "manual_corrections_template.jsonl", [{
            "sample_id": "different",
            "auto_label": "animal",
            "manual_label": None,
            "reviewer_status": "pending",
        }])
        with self.assertRaises(ValueError):
            MODULE.ReviewStore(self.root)


if __name__ == "__main__":
    unittest.main()
