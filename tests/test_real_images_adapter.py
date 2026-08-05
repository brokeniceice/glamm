import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.forensics.real_images import REAL_EXPLANATION, UnifiedRealImageAdapter


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class UnifiedRealImageAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

        coco = self.root / "COCO2017"
        (coco / "images").mkdir(parents=True)
        image_path = coco / "images" / "0001.jpg"
        Image.new("RGB", (8, 6), color="white").save(image_path)
        write_jsonl(coco / "manifests" / "selected.jsonl", [{
            "id": 1,
            "file_name": "0001.jpg",
            "category": "human",
            "instance_categories": ["person"],
            "license_metadata": {"name": "test license"},
            "flickr_url": "https://example.test/1",
        }])
        write_jsonl(coco / "manifests" / "checksums_sha256.jsonl", [{
            "file_name": "0001.jpg",
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "bytes": image_path.stat().st_size,
        }])

        raise_root = self.root / "RAISE-1k"
        (raise_root / "images").mkdir(parents=True)
        raise_path = raise_root / "images" / "r001.TIF"
        Image.new("RGB", (7, 5), color="gray").save(raise_path)
        write_jsonl(raise_root / "manifests" / "selected.jsonl", [{
            "raise_id": "r001",
            "file_name": "r001.TIF",
            "keywords": "people; outdoor",
        }])
        write_jsonl(raise_root / "manifests" / "checksums_sha256.jsonl", [{
            "file_name": "r001.TIF",
            "sha256": hashlib.sha256(raise_path.read_bytes()).hexdigest(),
            "bytes": raise_path.stat().st_size,
        }])

    def tearDown(self):
        self.temporary.cleanup()

    def test_combines_sources_and_preserves_roles(self):
        adapter = UnifiedRealImageAdapter(self.root, sources=("COCO2017", "RAISE-1k"))
        self.assertEqual(len(adapter), 2)
        self.assertEqual(adapter.samples[0]["source_role"], "candidate_train")
        self.assertEqual(adapter.samples[1]["source_role"], "heldout_test")
        self.assertEqual(adapter.samples[0]["content_category"], "human")
        self.assertEqual(adapter.samples[1]["content_category"], "human")

    def test_real_sample_has_fixed_explanation_and_native_zero_mask(self):
        sample = UnifiedRealImageAdapter(self.root, sources=("COCO2017",))[0]
        self.assertEqual(sample["class_label"], 0)
        self.assertEqual(sample["verdict_token"], "[REAL]")
        self.assertEqual(sample["explanation"], REAL_EXPLANATION)
        self.assertEqual(sample["union_evidence_mask"].shape, (6, 8))
        self.assertEqual(sample["union_evidence_mask"].dtype, np.uint8)
        self.assertFalse(sample["union_evidence_mask"].any())
        self.assertEqual(sample["refs"], [])

    def test_manifest_record_does_not_embed_zero_mask(self):
        adapter = UnifiedRealImageAdapter(self.root, sources=("COCO2017",))
        record = next(adapter.iter_manifest_records())
        self.assertNotIn("union_evidence_mask", record)
        self.assertEqual(record["identity_hash"]["algorithm"], "sha256")


if __name__ == "__main__":
    unittest.main()
