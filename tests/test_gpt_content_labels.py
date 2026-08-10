import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SPEC = importlib.util.spec_from_file_location("label_content_gpt", "scripts/data/label_content_gpt.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GPTContentLabelsTest(unittest.TestCase):
    def test_request_uses_image_input_and_strict_schema(self):
        config = {
            "categories": ["human", "animal", "object", "scene", "ambiguous"],
            "reasoning_effort": "low",
            "image_detail": "low",
            "prompt": "classify",
        }
        request = MODULE.build_request("gpt-test", config, "abcd")
        image = request["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        self.assertEqual(image["detail"], "low")
        self.assertTrue(image["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertIn("ambiguous", request["text"]["format"]["schema"]["properties"]["label"]["enum"])

    def test_extracts_responses_api_output_text(self):
        payload = {"output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"label":"human","confidence":0.9,"reason":"person"}'}
        ]}]}
        parsed = json.loads(MODULE.extract_output_text(payload))
        self.assertEqual(parsed["label"], "human")

    def test_encode_image_resizes_and_returns_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.png"
            Image.new("RGB", (1200, 600), "white").save(path)
            encoded, metadata = MODULE.encode_image(path, 300, 80)
            self.assertTrue(encoded)
            self.assertEqual(metadata["original_size"], [1200, 600])
            self.assertEqual(metadata["api_image_size"], [300, 150])

    def test_normalize_seed_uses_new_scope_identity_and_old_gpt_result(self):
        item = {
            "review_index": 8000,
            "sample_id": "pass-1",
            "source": "PASS",
            "domain": "real_candidate",
            "image_path": "/new/path.jpg",
            "clip_label": "scene",
            "clip_margin": 0.01,
            "human_label": None,
            "human_status": "not_requested",
        }
        seed = {
            "sample_id": "pass-1",
            "gpt_label": "object",
            "gpt_confidence": 0.95,
            "gpt_reason": "An object dominates.",
            "model": "gpt-test",
            "response_id": "response",
            "usage": {"total_tokens": 10},
        }
        normalized = MODULE.normalize_seed(item, seed, "gpt-test")
        self.assertEqual(normalized["review_index"], 8000)
        self.assertEqual(normalized["gpt_label"], "object")
        self.assertTrue(normalized["reused_seed_prediction"])


if __name__ == "__main__":
    unittest.main()
