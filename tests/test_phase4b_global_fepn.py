import unittest

import torch

from model.GLaMM import GLaMMForCausalLM, extract_seg_predictor_hidden
from model.fepn import ForensicEvidenceProjector
from tools.utils import IMAGE_TOKEN_INDEX


class Phase4BGlobalFEPNTest(unittest.TestCase):
    def test_projector_fixed_shape_and_nonzero_gradient(self):
        torch.manual_seed(3407)
        projector = ForensicEvidenceProjector(128, 256, 4, 4096)
        features = torch.randn(3, 128, requires_grad=True)
        tokens = projector(features)
        self.assertEqual(tuple(tokens.shape), (3, 4, 4096))
        self.assertTrue(torch.isfinite(tokens).all())
        tokens.square().mean().backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in projector.parameters()), 0.0)
        self.assertEqual(projector.parameter_count, 4_243_712)

    def test_four_tokens_shift_cls_and_seg_mapping_by_exactly_four(self):
        dummy = type("Dummy", (), {"forensic_evidence_token_count": 4})()
        cls, seg = 71, 72
        raw = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, cls, 13, seg]])
        cls_mask = GLaMMForCausalLM._create_expanded_token_mask(dummy, raw, cls)
        self.assertEqual(cls_mask.nonzero(as_tuple=False).tolist(), [[0, 582]])
        hidden = torch.arange(585, dtype=torch.float32).reshape(1, 585, 1)
        states, positions = extract_seg_predictor_hidden(hidden, raw, seg, image_expansion=579)
        self.assertEqual(positions[0].tolist(), [583])
        self.assertEqual(float(states[0][0, 0]), 583.0)

    def test_original_mapping_remains_unchanged_when_evidence_count_absent(self):
        dummy = type("Dummy", (), {})()
        cls = 71
        raw = torch.tensor([[11, IMAGE_TOKEN_INDEX, 12, cls]])
        mask = GLaMMForCausalLM._create_expanded_token_mask(dummy, raw, cls)
        self.assertEqual(mask.nonzero(as_tuple=False).tolist(), [[0, 578]])
        self.assertEqual(mask.shape[1], raw.shape[1] + 575)


if __name__ == "__main__":
    unittest.main()
