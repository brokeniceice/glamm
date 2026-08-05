import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.forensics.synthscars import SynthScarsAdapter, SynthScarsFormatError


def make_ref(sentence, explanation, polygon):
    return {
        "sentence": sentence,
        "explanation": explanation,
        "bbox": None,
        "segmentation": [polygon],
    }


class SynthScarsAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for split in ("train", "test"):
            (self.root / split / "annotations").mkdir(parents=True)
            (self.root / split / "images").mkdir(parents=True)
            Image.new("RGB", (10, 10), color="white").save(self.root / split / "images" / "sample.png")
            annotations = []
            if split == "train":
                annotations = [
                    {
                        "10": {
                            "img_file_name": "sample.png",
                            "caption": "The left artifact is malformed.",
                            "refs": [make_ref("left artifact", "It is malformed.", [1, 1, 4, 1, 4, 4, 1, 4])],
                        }
                    },
                    {
                        "20": {
                            "img_file_name": "sample.png",
                            "caption": "The right artifact is distorted.",
                            "refs": [make_ref("right artifact", "It is distorted.", [6, 6, 9, 6, 9, 9, 6, 9])],
                        }
                    },
                ]
            (self.root / split / "annotations" / f"{split}.json").write_text(json.dumps(annotations))

    def tearDown(self):
        self.temporary.cleanup()

    def test_groups_duplicate_annotations_by_image_and_retains_ids(self):
        adapter = SynthScarsAdapter(self.root, split="train")
        self.assertEqual(adapter.annotation_count, 2)
        self.assertEqual(len(adapter), 1)
        sample = adapter[0]
        self.assertEqual(sample["annotation_ids"], ["10", "20"])
        self.assertEqual(len(sample["ref_masks"]), 2)
        self.assertEqual(sample["ref_mask_valid"], [True, True])
        self.assertEqual(sample["invalid_ref_ids"], [])
        self.assertIn("left artifact", sample["explanation"])
        self.assertIn("right artifact", sample["explanation"])

    def test_union_mask_is_binary_and_contains_every_ref_mask(self):
        sample = SynthScarsAdapter(self.root, split="train")[0]
        union = sample["union_evidence_mask"]
        self.assertEqual(union.dtype, np.uint8)
        self.assertEqual(set(np.unique(union)), {0, 1})
        for ref_mask in sample["ref_masks"]:
            self.assertTrue(np.all(union[ref_mask.astype(bool)] == 1))
        expected = np.logical_or.reduce([mask.astype(bool) for mask in sample["ref_masks"]]).astype(np.uint8)
        np.testing.assert_array_equal(union, expected)

    def test_manifest_does_not_embed_raster_masks(self):
        record = next(iter(SynthScarsAdapter(self.root, split="train").iter_manifest_records()))
        self.assertNotIn("union_evidence_mask", record)
        self.assertNotIn("ref_masks", record)
        self.assertEqual(record["class_label"], 1)
        self.assertEqual(record["verdict_token"], "[FAKE]")

    def test_rejects_invalid_polygon(self):
        annotation_path = self.root / "train" / "annotations" / "train.json"
        data = json.loads(annotation_path.read_text())
        data[0]["10"]["refs"][0]["segmentation"] = [[1, 2, 3]]
        annotation_path.write_text(json.dumps(data))
        with self.assertRaises(SynthScarsFormatError):
            SynthScarsAdapter(self.root, split="train")

    def test_preserves_empty_ref_explanation_when_caption_is_available(self):
        annotation_path = self.root / "train" / "annotations" / "train.json"
        data = json.loads(annotation_path.read_text())
        data[0]["10"]["refs"][0]["explanation"] = ""
        annotation_path.write_text(json.dumps(data))
        sample = SynthScarsAdapter(self.root, split="train")[0]
        self.assertEqual(sample["ref_explanations"][0], "")
        self.assertTrue(sample["explanation"])

    def test_groups_same_stem_variants_and_scales_polygons_to_canonical_size(self):
        Image.new("RGB", (20, 20), color="white").save(self.root / "train" / "images" / "sample.jpg")
        annotation_path = self.root / "train" / "annotations" / "train.json"
        data = json.loads(annotation_path.read_text())
        data.append({
            "30": {
                "img_file_name": "sample.jpg",
                "caption": "A third artifact is visible.",
                "refs": [make_ref("third artifact", "It is visible.", [5, 5, 10, 5, 10, 10, 5, 10])],
            }
        })
        annotation_path.write_text(json.dumps(data))
        adapter = SynthScarsAdapter(self.root, split="train")
        self.assertEqual(len(adapter), 1)
        sample = adapter[0]
        self.assertEqual(sample["image_name"], "sample.jpg")
        self.assertEqual(len(sample["image_variants"]), 2)
        self.assertEqual(sample["image_size"], [20, 20])
        self.assertEqual(len(sample["ref_masks"]), 3)


if __name__ == "__main__":
    unittest.main()
