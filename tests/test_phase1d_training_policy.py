import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import train
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    CANONICAL_UNIFIED_QUESTION,
    UnifiedForensicsDataset,
)
from eval.forensics import evaluate_joint_localization, evaluate_unified_fake_generation_localization
from eval.forensics_eval import GLaMMForensicsBackend
from model.GLaMM import diagnostic_domain_text_losses, per_sample_causal_text_loss
from scripts.phase1d_training_policy import vocabulary_drift


def fake_sample():
    return {
        "sample_id": "fake", "image_path": "fake.png", "source": "unit",
        "content_category": "object", "cls_label": 1, "seg_valid": True,
        "masks": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.float32),
    }


class CanonicalBackend:
    def __init__(self, cls_pred=1):
        self.cls_pred = cls_pred
        self.calls = []

    def generate_localization(self, item, *, provide_gt_fake, generation_mode=None):
        mode = generation_mode or "joint"
        self.asserted_mode = mode
        prompt = GLaMMForensicsBackend._conversation(
            "", question=CANONICAL_UNIFIED_QUESTION, continue_assistant=False
        )
        self.calls.append(prompt)
        return {
            "cls_pred": self.cls_pred, "cls_prob_fake": 0.9 if self.cls_pred else 0.1,
            "lm_verdict_pred": 1, "lm_verdict_prob_fake": 0.9,
            "cls_lm_agree": bool(self.cls_pred == 1), "generated_explanation": "evidence",
            "seg_triggered": True,
            "pred_mask": torch.tensor([[[2.0, -2.0], [-2.0, -2.0]]]),
            "generated_token_ids": [17, 18, 19], "seg_position": 2,
            "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256, "raw_prompt_text": prompt,
            "assistant_prefix": "[CLS]",
        }


class TinyPolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.classification_head = torch.nn.Linear(1, 2)
        self.text_hidden_fcs = torch.nn.Linear(1, 1)
        self.grounding_encoder = torch.nn.Module()
        self.grounding_encoder.mask_decoder = torch.nn.Linear(1, 1)
        self.region_encoder = torch.nn.Linear(1, 1)
        self.embed_tokens = torch.nn.Embedding(8, 2)
        self.lm_head = torch.nn.Linear(2, 8, bias=False)

    def forward(self, x):
        hidden = self.embed_tokens(torch.zeros(x.shape[0], dtype=torch.long))
        logits = self.lm_head(hidden)
        cls = self.classification_head(x)
        mask = self.grounding_encoder.mask_decoder(self.text_hidden_fcs(x))
        return logits, cls, mask, logits.mean() + cls.mean() + mask.mean() + self.lora_A.sum()


def optimizer_args():
    return SimpleNamespace(
        lr=3e-4, lora_lr=3e-4, classification_head_lr=3e-4,
        text_hidden_fcs_lr=3e-4, mask_decoder_lr=3e-4,
        embedding_lr=3e-5, lm_head_lr=4e-5,
    )


class Phase1DTrainingPolicyTest(unittest.TestCase):
    def test_joint_equals_g0_when_all_fake_classifications_pass(self):
        backend = CanonicalBackend(cls_pred=1)
        g0_records, g0 = evaluate_unified_fake_generation_localization([fake_sample()], backend)
        joint_records, joint = evaluate_joint_localization([fake_sample()], backend)
        self.assertEqual(backend.calls[0], backend.calls[1])
        for field in ("generated_token_ids", "seg_position", "intersection", "union", "tp", "fp", "fn"):
            self.assertEqual(g0_records[0][field], joint_records[0][field])
        for field in ("mean_iou", "global_iou", "mean_pixel_f1", "global_pixel_f1"):
            self.assertEqual(g0[field], joint[field])

    def test_joint_real_gate_is_zero_while_g0_scores_mask(self):
        backend = CanonicalBackend(cls_pred=0)
        _, g0 = evaluate_unified_fake_generation_localization([fake_sample()], backend)
        joint_records, joint = evaluate_joint_localization([fake_sample()], backend)
        self.assertEqual(g0["mean_iou"], 1.0)
        self.assertEqual(joint["mean_iou"], 0.0)
        self.assertFalse(joint_records[0]["classification_gate_passed"])

    def test_g1_prefix_is_structural_and_has_no_terminal_eos(self):
        prompt = GLaMMForensicsBackend._conversation(
            "[FAKE]", question=CANONICAL_UNIFIED_QUESTION, continue_assistant=True
        )
        self.assertTrue(prompt.endswith("[FAKE]"))

    def test_freezing_unused_region_encoder_preserves_outputs(self):
        torch.manual_seed(5)
        model = TinyPolicyModel().eval()
        x = torch.tensor([[0.25]])
        before = tuple(value.detach().clone() for value in model(x))
        frozen = train.freeze_unused_region_encoder(model)
        after = tuple(value.detach().clone() for value in model(x))
        self.assertGreater(frozen, 0)
        self.assertFalse(any(p.requires_grad for p in model.region_encoder.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_optimizer_groups_are_disjoint_and_use_independent_lrs(self):
        model = TinyPolicyModel()
        train.freeze_unused_region_encoder(model)
        groups = train.build_optimizer_parameter_groups(model, optimizer_args())
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        by_name = {group["name"]: group for group in groups}
        self.assertEqual(by_name["embeddings"]["lr"], 3e-5)
        self.assertEqual(by_name["lm_head"]["lr"], 4e-5)
        region_ids = {id(parameter) for parameter in model.region_encoder.parameters()}
        self.assertTrue(region_ids.isdisjoint(ids))

    def test_vocabulary_drift_audit_separates_new_and_old_rows(self):
        initial = torch.zeros(6, 2)
        final = initial.clone(); final[1] = torch.tensor([3.0, 4.0]); final[4] = torch.tensor([0.0, 2.0])
        audit = vocabulary_drift(initial, final, [1])
        self.assertEqual(audit["new_token_rows"]["mean_l2_delta"], 5.0)
        self.assertEqual(audit["old_vocabulary_rows"]["max_l2_delta"], 2.0)

    def test_token_mean_and_per_sample_text_loss_math(self):
        logits = torch.tensor([
            [[0.0, 0.0], [3.0, -3.0], [0.0, 0.0]],
            [[0.0, 0.0], [-3.0, 3.0], [-3.0, 3.0]],
        ])
        labels = torch.tensor([[-100, 0, -100], [-100, 0, 0]])
        shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
        token_mean = F.cross_entropy(
            shifted_logits.reshape(-1, 2), shifted_labels.reshape(-1), ignore_index=-100
        )
        valid = shifted_labels.ne(-100)
        losses = F.cross_entropy(
            shifted_logits.reshape(-1, 2), shifted_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape_as(shifted_labels)
        expected_sample = ((losses * valid).sum(1) / valid.sum(1)).mean()
        self.assertFalse(torch.allclose(token_mean, expected_sample))
        self.assertTrue(torch.allclose(per_sample_causal_text_loss(logits, labels), expected_sample))

    def test_real_fake_diagnostic_ce_is_domain_specific(self):
        logits = torch.tensor([[[3.0, -3.0], [0.0, 0.0]], [[-3.0, 3.0], [0.0, 0.0]]])
        labels = torch.tensor([[-100, 0], [-100, 0]])
        real, fake = diagnostic_domain_text_losses(logits, labels, torch.tensor([0, 1]))
        self.assertLess(real, fake)

    def test_true_validation_total_and_selector_ignore_ce_only(self):
        output = {
            "ce_loss": torch.tensor(1.0), "cls_loss": torch.tensor(2.0),
            "mask_bce_loss": torch.tensor(3.0), "mask_dice_loss": torch.tensor(4.0),
            "loss": torch.tensor(10.0),
        }
        self.assertEqual(float(train.validation_total_loss_from_output(output)), 10.0)
        self.assertEqual(train.update_best_validation_total_loss(4.0, 5.0), (True, 4.0))
        with self.assertRaises(ValueError):
            train.validation_total_loss_from_output(dict(output, loss=torch.tensor(1.0)))

    def test_checkpoint_saves_last_and_best_independently(self):
        class Engine:
            def __init__(self):
                self.paths = []

            def save_checkpoint(self, path):
                self.paths.append(path)

        with tempfile.TemporaryDirectory() as temporary:
            engine = Engine()
            original_barrier = torch.distributed.barrier
            torch.distributed.barrier = lambda: None
            try:
                train.save_checkpoint(
                    engine, SimpleNamespace(log_dir=temporary, local_rank=0),
                    epoch=2, metric_name="loss", metric_value="1.25", is_best=True,
                )
            finally:
                torch.distributed.barrier = original_barrier
            self.assertEqual(
                {path.rsplit("/", 1)[-1] for path in engine.paths},
                {"ckpt_model_last_epoch", "ckpt_model_best"},
            )

    def test_prompt_fingerprint_train_eval_is_identical(self):
        _, conversations = UnifiedForensicsDataset._conversation(
            {"forensics_domain": "fake", "explanation": "artifact"}
        )
        eval_prompt = GLaMMForensicsBackend._conversation(
            "[FAKE] artifact [SEG]", question=CANONICAL_UNIFIED_QUESTION
        )
        self.assertEqual(conversations[0], eval_prompt)
        self.assertEqual(len(CANONICAL_PROMPT_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
