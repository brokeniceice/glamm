import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import build_forensics_prediction_records
from model.GLaMM import GLaMMForCausalLM, mask_gradient_rows, per_sample_causal_text_loss
from model.llava import conversation as conversation_lib
from model.llava.model.language_model.llava_llama import LlavaConfig
from tools.utils import DEFAULT_CLS_TOKEN, DEFAULT_FAKE_TOKEN, DEFAULT_REAL_TOKEN, IGNORE_INDEX


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "checkpoints/GLaMM-FullScope"


class DummyGlobalProcessor:
    def preprocess(self, image, return_tensors="pt"):
        assert return_tensors == "pt"
        value = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return {"pixel_values": value}


class DummyGroundingEncoder(nn.Module):
    pass


class DummyRegionEncoder(nn.Module):
    def forward(self, features, boxes):
        return features


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@unittest.skipUnless(TOKENIZER_PATH.is_dir(), "local GLaMM tokenizer is required for smoke tests")
class UnifiedForensicsPipelineTest(unittest.TestCase):
    def test_per_sample_text_loss_normalization_math(self):
        # Sample 0 has one supervised token with CE=a; sample 1 has three with
        # CE=b. Per-sample mode must return (a+b)/2 rather than (a+3b)/4.
        logits = torch.zeros(2, 5, 2)
        labels = torch.full((2, 5), IGNORE_INDEX, dtype=torch.long)
        labels[0, 1] = 0
        labels[1, 1:] = 1
        logits[0, 0] = torch.tensor([2.0, 0.0])
        logits[1, :4] = torch.tensor([0.0, 1.0])
        a = torch.nn.functional.cross_entropy(logits[0, 0].unsqueeze(0), torch.tensor([0]))
        b = torch.nn.functional.cross_entropy(logits[1, 0].unsqueeze(0), torch.tensor([1]))
        actual = per_sample_causal_text_loss(logits, labels)
        self.assertTrue(torch.allclose(actual, (a + b) / 2))
        token_mean = (a + 3 * b) / 4
        self.assertFalse(torch.allclose(actual, token_mean))

    def test_optional_old_vocab_gradient_masking(self):
        gradient = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        masked = mask_gradient_rows(gradient, [1, 4])
        self.assertTrue(torch.equal(masked[1], gradient[1]))
        self.assertTrue(torch.equal(masked[4], gradient[4]))
        self.assertEqual(float(masked[[0, 2, 3, 5]].abs().sum()), 0.0)

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, use_fast=False, local_files_only=True, model_max_length=256,
            padding_side="right"
        )
        cls.tokenizer.pad_token = cls.tokenizer.unk_token
        cls.tokenizer.add_tokens(
            [DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN], special_tokens=True
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.datasets_root = self.root / "datasets"
        self.synth_root = self.datasets_root / "SynthScars"
        real_path = self.datasets_root / "COCO2017/images/real.jpg"
        fake_path = self.synth_root / "train/images/fake.png"
        real_path.parent.mkdir(parents=True)
        fake_path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 6), color=(64, 128, 192)).save(real_path)
        Image.new("RGB", (8, 6), color=(192, 128, 64)).save(fake_path)
        self.real_row = {
            "sample_id": "real:COCO2017:1", "source": "COCO2017",
            "forensics_domain": "real", "class_label": 0, "content_category": "object",
            "image_name": "real.jpg", "image_relpath": "COCO2017/images/real.jpg",
            "explanation": "legacy text", "refs": [], "has_mask_label": True,
        }
        self.fake_row = {
            "sample_id": "synthscars:train:fake", "source": "SynthScars",
            "forensics_domain": "fake", "class_label": 1, "content_category": "object",
            "image_name": "fake.png", "image_relpath": "train/images/fake.png",
            "explanation": "The object has an impossible warped edge.",
            "refs": [{
                "ref_id": "1:0", "source_image_size": [6, 8],
                "polygons": [[1, 1, 6, 1, 6, 4, 1, 4]],
            }],
            "has_mask_label": True,
        }
        for split in ("train", "val", "test"):
            rows = [dict(self.real_row), dict(self.fake_row)]
            for row in rows:
                row["dataset_split"] = split
            write_jsonl(self.root / "manifests" / f"{split}_combined.jsonl", rows)
        self.dataset = UnifiedForensicsDataset(
            self.root / "manifests", self.tokenizer, "unused", split="train",
            datasets_root=self.datasets_root, synthscars_root=self.synth_root,
            image_size=8, global_enc_processor=DummyGlobalProcessor(),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_special_tokens_are_single_tokens_and_survive_save_load(self):
        for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN):
            self.assertEqual(len(self.tokenizer(token, add_special_tokens=False).input_ids), 1)
        saved = self.root / "saved-tokenizer"
        self.tokenizer.save_pretrained(saved)
        loaded = AutoTokenizer.from_pretrained(saved, use_fast=False, local_files_only=True)
        for token in (DEFAULT_CLS_TOKEN, DEFAULT_REAL_TOKEN, DEFAULT_FAKE_TOKEN):
            self.assertEqual(len(loaded(token, add_special_tokens=False).input_ids), 1)
            self.assertEqual(
                loaded(token, add_special_tokens=False).input_ids,
                self.tokenizer(token, add_special_tokens=False).input_ids,
            )

    def test_real_sample_has_no_mask_and_fixed_target(self):
        sample = self.dataset[0]
        self.assertEqual(sample["cls_label"], 0)
        self.assertFalse(sample["seg_valid"])
        self.assertIsNone(sample["masks"])
        self.assertIn("[REAL] No identifiable synthetic artifact evidence is detected.",
                      sample["conversations"][0])
        self.assertNotIn("[SEG]", sample["conversations"][0])

    def test_fake_sample_has_union_mask_and_seg_target(self):
        sample = self.dataset[1]
        self.assertEqual(sample["cls_label"], 1)
        self.assertTrue(sample["seg_valid"])
        self.assertEqual(tuple(sample["masks"].shape), (1, 6, 8))
        self.assertGreater(sample["masks"].sum().item(), 0)
        self.assertIn("[FAKE] The object has an impossible warped edge. [SEG]",
                      sample["conversations"][0])

    def test_mixed_batch_collate_preserves_per_sample_segmentation_state_and_labels(self):
        batch = custom_collate_fn(
            [self.dataset[0], self.dataset[1]], tokenizer=self.tokenizer,
            token_strategy="fixed_cls_query", inference=False,
        )
        self.assertEqual(batch["seg_valid"].tolist(), [False, True])
        self.assertIsNone(batch["masks_list"][0])
        self.assertEqual(tuple(batch["masks_list"][1].shape), (1, 6, 8))
        self.assertEqual(tuple(batch["grounding_enc_images"].shape), (2, 3, 1024, 1024))
        self.assertEqual(batch["cls_labels"].tolist(), [0, 1])
        cls_id = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids[0]
        real_id = self.tokenizer(DEFAULT_REAL_TOKEN, add_special_tokens=False).input_ids[0]
        fake_id = self.tokenizer(DEFAULT_FAKE_TOKEN, add_special_tokens=False).input_ids[0]
        for ids, labels in zip(batch["input_ids"], batch["labels"]):
            self.assertTrue(torch.all(labels[ids == cls_id] == IGNORE_INDEX))
        self.assertTrue(torch.all(batch["labels"][0][batch["input_ids"][0] == real_id] == real_id))
        self.assertTrue(torch.all(batch["labels"][1][batch["input_ids"][1] == fake_id] == fake_id))

    def test_partial_assistant_inference_does_not_use_training_label_parser(self):
        evaluation_sample = dict(self.dataset[0])
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], "The <image> provides an overview of the picture.\nAnalyze it.")
        conv.append_message(conv.roles[1], "")
        evaluation_sample["conversations"] = [conv.get_prompt()]
        batch = custom_collate_fn(
            [evaluation_sample], tokenizer=self.tokenizer,
            token_strategy="fixed_cls_query", inference=True,
        )
        cls_id = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids[0]
        self.assertTrue(batch["input_ids"].eq(cls_id).any())
        self.assertTrue(batch["labels"].eq(IGNORE_INDEX).all())

    def test_training_truncation_preserves_late_seg_token(self):
        evaluation_sample = dict(self.dataset[1])
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], "The <image> provides an overview of the picture.\nAnalyze it.")
        conv.append_message(conv.roles[1], "[FAKE] " + "artifact " * 400 + "[SEG]")
        evaluation_sample["conversations"] = [conv.get_prompt()]
        batch = custom_collate_fn(
            [evaluation_sample], tokenizer=self.tokenizer,
            token_strategy="fixed_cls_query", inference=False,
        )
        seg_id = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.assertEqual(int(batch["input_ids"].eq(seg_id).sum()), 1)
        self.assertEqual(batch["seg_preserving_truncation_count"], 1)
        self.assertLessEqual(batch["input_ids"].shape[1], self.tokenizer.model_max_length)

    def _tiny_model(self):
        import model.GLaMM as glamm_module
        import model.llava.llava_with_region_arch as region_arch

        config = LlavaConfig(
            vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        kwargs = dict(
            out_dim=8, train_mask_decoder=False, ce_loss_weight=1.0,
            dice_loss_weight=0.5, bce_loss_weight=2.0, cls_loss_weight=1.0,
            cls_token_idx=self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False).input_ids[0],
            real_token_idx=self.tokenizer(DEFAULT_REAL_TOKEN, add_special_tokens=False).input_ids[0],
            fake_token_idx=self.tokenizer(DEFAULT_FAKE_TOKEN, add_special_tokens=False).input_ids[0],
            seg_token_idx=self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
            token_strategy="fixed_cls_query", with_region=False,
        )
        patches = (
            mock.patch.object(glamm_module, "build_sam_vit_h", return_value=DummyGroundingEncoder()),
            mock.patch.object(region_arch, "MLVLROIQueryModule", return_value=DummyRegionEncoder()),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return GLaMMForCausalLM(config, **kwargs)

    def test_single_batch_forward_exposes_cls_and_lm_verdict(self):
        model = self._tiny_model()
        cls_id = model.cls_token_idx
        real_id = model.real_token_idx
        fake_id = model.fake_token_idx
        input_ids = torch.tensor([[1, cls_id, real_id, 2], [1, cls_id, fake_id, 2]])
        labels = input_ids.clone()
        labels[:, :2] = IGNORE_INDEX
        output = model.model_forward(
            global_enc_images=None, grounding_enc_images=None, bboxes=None,
            input_ids=input_ids, labels=labels, attention_masks=torch.ones_like(input_ids),
            offset=torch.tensor([0, 1, 2]), masks_list=[None, None],
            label_list=[None, None], resize_list=[None, None], cls_labels=torch.tensor([0, 1]),
            seg_valid=torch.tensor([False, False]),
        )
        self.assertEqual(tuple(output["cls_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["lm_verdict_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["cls_pred"].shape), (2,))
        output["loss"].backward()
        self.assertIsNotNone(model.classification_head.weight.grad)

    def test_real_segmentation_is_skipped(self):
        model = self._tiny_model()
        ce = torch.tensor(0.25, requires_grad=True)
        result = model._compute_loss_components(
            [torch.randn(1, 4, 4, requires_grad=True)], [None],
            type("Output", (), {"loss": ce})(), torch.tensor([False]),
        )
        self.assertEqual(result["mask_loss"].item(), 0.0)
        self.assertEqual(result["seg_unexpected_pred_count"].item(), 1)

    def test_fake_segmentation_has_gradient_and_no_silent_truncation(self):
        model = self._tiny_model()
        prediction = torch.zeros(1, 4, 4, requires_grad=True)
        target = torch.ones(1, 4, 4)
        result = model._compute_loss_components(
            [prediction], [target], type("Output", (), {"loss": prediction.sum() * 0})(),
            torch.tensor([True]),
        )
        result["loss"].backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        mismatch = model._compute_loss_components(
            [prediction.detach().repeat(2, 1, 1)], [target],
            type("Output", (), {"loss": prediction.sum() * 0})(), torch.tensor([True]),
        )
        self.assertEqual(mismatch["seg_count_mismatch_count"].item(), 1)
        self.assertEqual(mismatch["mask_loss"].item(), 0.0)

    def test_mixed_mask_loss_is_not_diluted_by_real_samples(self):
        model = self._tiny_model()
        prediction = torch.zeros(1, 4, 4)
        target = torch.ones(1, 4, 4)
        output = type("Output", (), {"loss": prediction.sum() * 0})()
        fake_only = model._compute_loss_components(
            [prediction], [target], output, torch.tensor([True]),
        )
        mixed = model._compute_loss_components(
            [torch.randn(1, 4, 4), prediction], [None, target], output,
            torch.tensor([False, True]),
        )
        self.assertTrue(torch.allclose(fake_only["mask_bce_loss"], mixed["mask_bce_loss"]))
        self.assertTrue(torch.allclose(fake_only["mask_dice_loss"], mixed["mask_dice_loss"]))

    def test_cls_hidden_has_no_future_target_leakage(self):
        model = self._tiny_model().eval()
        captured = []
        hook = model.classification_head.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            sequences = (
                torch.tensor([[1, model.cls_token_idx, model.real_token_idx, 3, 4]]),
                torch.tensor([[1, model.cls_token_idx, model.fake_token_idx, 5, model.seg_token_idx]]),
            )
            for input_ids in sequences:
                labels = input_ids.clone()
                labels[:, :2] = IGNORE_INDEX
                model.model_forward(
                    global_enc_images=None, grounding_enc_images=None, bboxes=None,
                    input_ids=input_ids, labels=labels, attention_masks=torch.ones_like(input_ids),
                    offset=torch.tensor([0, 1]), masks_list=[None], label_list=[None], resize_list=[None],
                    cls_labels=torch.tensor([1]), seg_valid=torch.tensor([False]),
                )
        finally:
            hook.remove()
        self.assertEqual(len(captured), 2)
        self.assertTrue(torch.allclose(captured[0], captured[1], atol=1e-6, rtol=1e-5))

    def test_cls_real_and_fake_embedding_rows_receive_gradient(self):
        model = self._tiny_model().eval()
        rows = [
            [1, model.cls_token_idx, model.real_token_idx, 3],
            [1, model.cls_token_idx, model.fake_token_idx, 4],
        ]
        input_ids = torch.tensor(rows)
        labels = input_ids.clone()
        labels[:, :2] = IGNORE_INDEX
        result = model.model_forward(
            global_enc_images=None, grounding_enc_images=None, bboxes=None,
            input_ids=input_ids, labels=labels, attention_masks=torch.ones_like(input_ids),
            offset=torch.tensor([0, 1, 2]), masks_list=[None, None], label_list=[None, None],
            resize_list=[None, None], cls_labels=torch.tensor([0, 1]),
            seg_valid=torch.tensor([False, False]),
        )
        result["loss"].backward()
        grad = model.get_input_embeddings().weight.grad
        for token_id in (model.cls_token_idx, model.real_token_idx, model.fake_token_idx):
            self.assertGreater(float(grad[token_id].abs().sum()), 0.0)

    def test_fixed_query_generation_does_not_force_cls_and_model_roundtrip(self):
        model = self._tiny_model().eval()
        prompt = torch.tensor([[1, model.cls_token_idx]])
        with mock.patch.object(model, "generate", wraps=model.generate) as generate_spy:
            sequences, masks, results = model.evaluate(
                None, None, prompt, [None], [None], max_tokens_new=2, force_cls_token=True
            )
        self.assertTrue(torch.equal(sequences[:, :prompt.shape[1]], prompt))
        self.assertEqual(masks, [])
        self.assertNotIn("forced_decoder_ids", generate_spy.call_args.kwargs)
        self.assertEqual(tuple(results["lm_verdict_probabilities"].shape), (1, 2))

        saved = self.root / "tiny-model"
        model.save_pretrained(saved)
        with mock.patch("model.GLaMM.build_sam_vit_h", return_value=DummyGroundingEncoder()), \
             mock.patch("model.llava.llava_with_region_arch.MLVLROIQueryModule",
                        return_value=DummyRegionEncoder()):
            loaded = GLaMMForCausalLM.from_pretrained(saved)
        self.assertEqual(loaded.config.token_strategy, "fixed_cls_query")
        self.assertEqual(loaded.cls_token_idx, model.cls_token_idx)
        self.assertEqual(loaded.real_token_idx, model.real_token_idx)
        self.assertEqual(loaded.fake_token_idx, model.fake_token_idx)

    def test_evaluation_records_include_both_predictions_and_agreement(self):
        batch = {
            "sample_ids": ["x"], "image_paths": ["x.jpg"], "sources": ["SynthScars"],
            "content_categories": ["object"], "cls_labels": torch.tensor([1]),
        }
        output = {
            "cls_pred": torch.tensor([1]), "cls_prob": torch.tensor([0.8]),
            "lm_verdict_pred": torch.tensor([0]), "lm_verdict_prob": torch.tensor([0.3]),
            "cls_lm_agree": torch.tensor([False]),
        }
        record = build_forensics_prediction_records(batch, output)[0]
        self.assertEqual(record["cls_pred"], "fake")
        self.assertEqual(record["lm_verdict_pred"], "real")
        self.assertFalse(record["cls_lm_agree"])


if __name__ == "__main__":
    unittest.main()
