import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SPEC = importlib.util.spec_from_file_location("audit_image_phash", "scripts/data/audit_image_phash.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ImagePhashTest(unittest.TestCase):
    def test_identical_images_have_identical_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.png"
            right = Path(directory) / "right.png"
            image = Image.new("RGB", (64, 48), "white")
            image.putpixel((10, 10), (0, 0, 0))
            image.save(left)
            image.save(right)
            self.assertEqual(MODULE.perceptual_hash(str(left))[0], MODULE.perceptual_hash(str(right))[0])

    def test_multi_index_finds_hamming_distance_four(self):
        records = [
            {"sample_id": "a", "domain": "real_candidate", "source": "A", "phash64": "0000000000000000"},
            {"sample_id": "b", "domain": "synthscars_fake", "source": "B", "phash64": "000000000000000f"},
        ]
        pairs = MODULE.near_duplicate_pairs(records, threshold=4)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["hamming_distance"], 4)


if __name__ == "__main__":
    unittest.main()
