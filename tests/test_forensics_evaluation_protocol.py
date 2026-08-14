import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessor, LogitsProcessorList

from eval.forensics import (
    TF_MINIMAL_CONTEXT_TEMPLATE,
    compute_binary_mask_metrics,
    evaluate_detection,
    evaluate_gt_fake_generation_localization,
    evaluate_joint_localization,
    evaluate_teacher_forced_full_context,
    evaluate_teacher_forced_minimal_context,
    evaluate_unified_fake_generation_localization,
    evaluate_unified_gt_fake_prefix_localization,
    evaluate_legacy_gt_fake_generation_localization,
    summarize_localization_records,
    write_evaluation_outputs,
)
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.phase1c_seg_diagnosis import (
    audit_weight_tying,
    classify_generation_stop,
    module_participates_in_loss,
    probability_trace,
    summarize_probability_trace,
)
from model.GLaMM import extract_seg_predictor_hidden


def sample(sample_id="fake", *, label=1, seg_valid=True, content="object", mask=None):
    return {
        "sample_id": sample_id,
        "image_path": f"{sample_id}.png",
        "source": "unit",
        "content_category": content,
        "cls_label": label,
        "seg_valid": seg_valid,
        "masks": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.float32) if mask is None else mask,
    }


def output(*, cls_pred=1, lm_pred=1, pred_mask=None, triggered=True):
    return {
        "cls_pred": cls_pred,
        "cls_prob_fake": 0.9 if cls_pred else 0.1,
        "lm_verdict_pred": lm_pred,
        "lm_verdict_prob_fake": 0.9 if lm_pred else 0.1,
        "cls_lm_agree": cls_pred == lm_pred,
        "generated_explanation": "model generated evidence",
        "seg_triggered": triggered,
        "pred_mask": pred_mask,
    }


class RecordingBackend:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detection(self, item):
        self.calls.append(("detection", item["sample_id"]))
        return self.result

    def generate_localization(self, item, *, provide_gt_fake, generation_mode=None):
        self.calls.append(("generate", item["sample_id"], provide_gt_fake, generation_mode))
        return self.result

    def teacher_forced_localization(self, item, *, context):
        self.calls.append(("teacher", item["sample_id"], context))
        return self.result


class ForceSchedule(LogitsProcessor):
    def __init__(self, prompt_length, tokens):
        self.prompt_length = prompt_length
        self.tokens = tokens

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        forced = self.tokens[step]
        scores.fill_(-float("inf"))
        scores[:, forced] = 0
        return scores


class ForensicsEvaluationProtocolTest(unittest.TestCase):
    @staticmethod
    def perfect_logits():
        return torch.tensor([[[2.0, -2.0], [-2.0, -2.0]]])

    @staticmethod
    def tiny_causal_model():
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=31,
            pad_token_id=0,
        )
        return LlamaForCausalLM(config).eval()

    def test_teacher_forced_seg_indexing_uses_preceding_hidden(self):
        hidden = torch.arange(3 * 4, dtype=torch.float32).reshape(1, 3, 4)
        states, positions = extract_seg_predictor_hidden(hidden, torch.tensor([[4, 5, 9]]), 9)
        self.assertEqual(positions[0].tolist(), [1])
        self.assertTrue(torch.equal(states[0][0], hidden[0, 1]))
        self.assertFalse(torch.equal(states[0][0], hidden[0, 2]))

    def test_causal_full_sequence_has_no_future_seg_leak(self):
        model = self.tiny_causal_model()
        prefix = torch.tensor([[1, 7]])
        full = torch.tensor([[1, 7, 9]])
        with torch.no_grad():
            prefix_hidden = model(prefix, output_hidden_states=True).hidden_states[-1]
            full_hidden = model(full, output_hidden_states=True).hidden_states[-1]
        self.assertTrue(torch.allclose(prefix_hidden[0, -1], full_hidden[0, 1], atol=1e-6, rtol=1e-5))

    def test_gt_fake_generate_ignores_real_classification_prediction(self):
        backend = RecordingBackend(output(cls_pred=0, pred_mask=self.perfect_logits()))
        records, metrics = evaluate_gt_fake_generation_localization([sample()], backend)
        self.assertEqual(records[0]["image_iou"], 1.0)
        self.assertEqual(records[0]["image_pixel_f1"], 1.0)
        self.assertEqual(metrics["mean_iou"], 1.0)

    def test_g0_fake_only_localization_ignores_classification_gate(self):
        backend = RecordingBackend(output(cls_pred=0, pred_mask=self.perfect_logits()))
        records, metrics = evaluate_unified_fake_generation_localization([sample()], backend)
        self.assertEqual(records[0]["image_iou"], 1.0)
        self.assertFalse(records[0]["classification_gates_localization"])
        self.assertEqual(metrics["mean_iou"], 1.0)

    def test_g1_fake_only_localization_ignores_classification_gate(self):
        backend = RecordingBackend(output(cls_pred=0, pred_mask=self.perfect_logits()))
        records, metrics = evaluate_unified_gt_fake_prefix_localization([sample()], backend)
        self.assertEqual(records[0]["image_iou"], 1.0)
        self.assertFalse(records[0]["classification_gates_localization"])
        self.assertEqual(metrics["mean_iou"], 1.0)

    def test_legacy_g2_protocol_remains_explicitly_reproducible(self):
        backend = RecordingBackend(output(pred_mask=self.perfect_logits()))
        records, _ = evaluate_legacy_gt_fake_generation_localization([sample()], backend)
        self.assertEqual(records[0]["eval_mode"], "legacy_gt_fake_generate")
        self.assertEqual(backend.calls[-1][-1], "legacy_gt_fake_generate")

    def test_seg_probability_and_eos_trace_extraction(self):
        scores = [torch.tensor([[0.0, 1.0, 3.0, 2.0]]), torch.tensor([[0.0, 1.0, 2.0, 4.0]])]
        trace = probability_trace(scores, [2, 3], seg_token_id=2, eos_token_id=3)
        self.assertEqual(trace[0]["rank_seg"], 1)
        self.assertGreater(trace[0]["p_seg"], trace[0]["p_eos"])
        self.assertLess(trace[1]["p_seg"], trace[1]["p_eos"])
        summary = summarize_probability_trace(trace, eos_position=1)
        self.assertAlmostEqual(summary["p_seg_before_eos"], trace[1]["p_seg"])
        self.assertAlmostEqual(summary["p_eos_before_eos"], trace[1]["p_eos"])

    def test_generation_failure_reason_classification(self):
        eos = classify_generation_stop([5, 9], seg_token_id=8, eos_token_id=9, max_new_tokens=4)
        maximum = classify_generation_stop([5, 6, 7, 4], seg_token_id=8, eos_token_id=9, max_new_tokens=4)
        empty = classify_generation_stop([], seg_token_id=8, eos_token_id=9, max_new_tokens=4)
        self.assertEqual(eos["stop_reason"], "EOS_BEFORE_SEG")
        self.assertEqual(maximum["stop_reason"], "MAX_NEW_TOKENS_BEFORE_SEG")
        self.assertEqual(empty["stop_reason"], "EMPTY_GENERATION")

    def test_region_encoder_participation_audit_helper(self):
        self.assertFalse(module_participates_in_loss(0, 0.0))
        self.assertTrue(module_participates_in_loss(1, 0.0))
        self.assertTrue(module_participates_in_loss(0, 0.25))

    def test_embedding_lm_head_weight_tying_audit(self):
        shared = torch.nn.Parameter(torch.ones(3, 2))
        tied = audit_weight_tying(shared, shared, config_tie_word_embeddings=True)
        untied = audit_weight_tying(
            shared, torch.nn.Parameter(shared.detach().clone()), config_tie_word_embeddings=False
        )
        self.assertTrue(tied["parameter_identity"])
        self.assertTrue(tied["data_ptr_equal"])
        self.assertFalse(untied["parameter_identity"])
        self.assertFalse(untied["data_ptr_equal"])

    def test_generated_fake_vs_prefilled_fake_alignment_prompt_has_no_terminal_eos(self):
        prompt = GLaMMForensicsBackend._conversation(
            "[FAKE]", question=UNIFIED_FORENSICS_QUESTION, continue_assistant=True
        )
        legacy = GLaMMForensicsBackend._conversation(
            "[FAKE]", question=UNIFIED_FORENSICS_QUESTION, continue_assistant=False
        )
        self.assertTrue(prompt.endswith("[FAKE]"))
        self.assertNotEqual(prompt, legacy)
        self.assertTrue(legacy.startswith(prompt))

    def test_joint_propagates_classification_error_to_zero(self):
        backend = RecordingBackend(output(cls_pred=0, pred_mask=self.perfect_logits()))
        records, metrics = evaluate_joint_localization([sample()], backend)
        self.assertEqual(records[0]["image_iou"], 0.0)
        self.assertEqual(records[0]["image_pixel_f1"], 0.0)
        self.assertEqual(metrics["classification_pass_rate"], 0.0)

    def test_missing_seg_in_gt_fake_is_zero_and_counted(self):
        backend = RecordingBackend(output(pred_mask=None, triggered=False))
        records, metrics = evaluate_gt_fake_generation_localization([sample()], backend)
        self.assertTrue(records[0]["seg_trigger_failure"])
        self.assertEqual(records[0]["image_iou"], 0.0)
        self.assertEqual(metrics["seg_trigger_failure_count"], 1)

    def test_localization_prediction_preserves_explanation_audit_fields(self):
        item = sample()
        item["manifest_row"] = {"explanation": "  GT   artifact evidence  "}
        result = output(pred_mask=self.perfect_logits())
        result["repetition_statistics"] = {"is_repetition_loop": True}
        records, _ = evaluate_gt_fake_generation_localization(
            [item], RecordingBackend(result)
        )
        self.assertEqual(records[0]["gt_explanation"], "GT artifact evidence")
        self.assertEqual(records[0]["generated_explanation"], "model generated evidence")
        self.assertTrue(records[0]["repetition_flag"])

    def test_missing_seg_in_joint_is_zero_and_grounding_failure(self):
        backend = RecordingBackend(output(cls_pred=1, pred_mask=None, triggered=False))
        records, metrics = evaluate_joint_localization([sample()], backend)
        self.assertEqual(records[0]["image_pixel_f1"], 0.0)
        self.assertTrue(records[0]["grounding_activation_failure"])
        self.assertEqual(metrics["seg_trigger_rate"], 0.0)

    def test_tf_full_context_uses_forward_not_generation(self):
        result = output(pred_mask=self.perfect_logits())
        result["generated_explanation"] = None
        backend = RecordingBackend(result)
        records, _ = evaluate_teacher_forced_full_context([sample()], backend)
        self.assertEqual(backend.calls, [("teacher", "fake", "full")])
        self.assertTrue(records[0]["uses_gt_explanation"])
        self.assertIsNone(records[0]["generated_explanation"])

    def test_gt_information_flags_and_minimal_template_are_fixed(self):
        backend = RecordingBackend(output(pred_mask=self.perfect_logits()))
        generated, _ = evaluate_gt_fake_generation_localization([sample()], backend)
        full, _ = evaluate_teacher_forced_full_context([sample()], backend)
        minimal, _ = evaluate_teacher_forced_minimal_context([sample()], backend)
        self.assertTrue(generated[0]["uses_gt_authenticity"])
        self.assertFalse(generated[0]["uses_gt_explanation"])
        self.assertTrue(full[0]["uses_gt_explanation"])
        self.assertFalse(minimal[0]["uses_gt_explanation"])
        self.assertEqual(TF_MINIMAL_CONTEXT_TEMPLATE, "Artifact evidence: [SEG]")

    def test_glamm_gt_fake_backend_prefix_contains_no_gt_explanation(self):
        captured = {}

        class DummyModel:
            seg_token_idx = 9

            def evaluate(self, *args, **kwargs):
                cls = {
                    "cls_pred": torch.tensor([0]), "cls_prob": torch.tensor([0.1]),
                    "lm_verdict_pred": torch.tensor([1]), "lm_verdict_prob": torch.tensor([0.9]),
                    "cls_lm_agree": torch.tensor([False]),
                }
                return torch.tensor([[1, 2, 7]]), [], cls

        class DummyTokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=False):
                return "generated"

        backend = object.__new__(GLaMMForensicsBackend)
        backend.model = DummyModel()
        backend.tokenizer = DummyTokenizer()
        backend.max_new_tokens = 8

        def fake_batch(item, assistant_content):
            captured["assistant_content"] = assistant_content
            return {
                "global_enc_images": None, "grounding_enc_images": None,
                "input_ids": torch.tensor([[1, 2]]), "resize_list": [(2, 2)],
                "label_list": [torch.zeros(2, 2)], "bboxes": None,
            }

        backend._batch = fake_batch
        item = sample()
        item["manifest_row"] = {"explanation": "SECRET GT EXPLANATION"}
        backend.generate_localization(item, provide_gt_fake=True)
        self.assertEqual(captured["assistant_content"], "[FAKE]")
        self.assertNotIn("SECRET", captured["assistant_content"])

    def test_real_is_excluded_from_every_fake_localization_denominator(self):
        real = sample("real", label=0, seg_valid=False)
        fake = sample()
        backend = RecordingBackend(output(pred_mask=self.perfect_logits()))
        evaluators = (
            evaluate_gt_fake_generation_localization,
            evaluate_teacher_forced_full_context,
            evaluate_teacher_forced_minimal_context,
            evaluate_joint_localization,
        )
        for evaluator in evaluators:
            records, metrics = evaluator([real, fake], backend)
            self.assertEqual([record["sample_id"] for record in records], ["fake"])
            self.assertEqual(metrics["num_gt_fake"], 1)

    def test_zero_logit_and_sigmoid_half_thresholds_are_equivalent(self):
        logits = torch.tensor([[-2.0, 0.0, 2.0]])
        self.assertTrue(torch.equal(logits > 0, logits.sigmoid() > 0.5))

    def test_mean_image_and_global_pixel_aggregation_are_distinct_and_correct(self):
        gt_small = torch.tensor([[[1]]], dtype=torch.float32)
        gt_large = torch.tensor([[[1, 1, 1]]], dtype=torch.float32)
        first = compute_binary_mask_metrics(torch.tensor([[[2.0]]]), gt_small)
        second = compute_binary_mask_metrics(torch.tensor([[[2.0, -2.0, -2.0]]]), gt_large)
        records = [
            {**first, "content_type": "object", "seg_triggered": True, "has_pred_mask": True},
            {**second, "content_type": "object", "seg_triggered": True, "has_pred_mask": True},
        ]
        metrics = summarize_localization_records(records, autoregressive=True)
        self.assertAlmostEqual(metrics["mean_iou"], (1 + 1 / 3) / 2)
        self.assertAlmostEqual(metrics["global_iou"], 2 / 4)
        self.assertNotEqual(metrics["mean_iou"], metrics["global_iou"])

    def test_generation_hidden_state_alignment_uses_predictor_step(self):
        model = self.tiny_causal_model()
        prefix = torch.tensor([[1]])
        explanation_token, seg_token = 7, 9
        with torch.no_grad():
            generated = model.generate(
                prefix,
                max_new_tokens=2,
                return_dict_in_generate=True,
                output_hidden_states=True,
                logits_processor=LogitsProcessorList([
                    ForceSchedule(prefix.shape[1], [explanation_token, seg_token])
                ]),
            )
            full_hidden = model(generated.sequences, output_hidden_states=True).hidden_states[-1]
        states, positions = extract_seg_predictor_hidden(full_hidden, generated.sequences, seg_token)
        generation_predictor = generated.hidden_states[1][-1][0, -1]
        self.assertEqual(positions[0].tolist(), [1])
        self.assertTrue(torch.allclose(states[0][0], generation_predictor, atol=1e-5, rtol=1e-4))

    def test_detection_reports_two_heads_agreement_quadrants_and_output_files(self):
        items = [sample("real", label=0, seg_valid=False), sample()]
        backend = RecordingBackend(output(cls_pred=1, lm_pred=0, pred_mask=None))
        records, metrics = evaluate_detection(items, backend)
        self.assertEqual(len(records), 2)
        self.assertEqual(metrics["outcome_counts"]["cls_wrong_lm_correct"], 1)
        self.assertEqual(metrics["outcome_counts"]["cls_correct_lm_wrong"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            predictions, summary = write_evaluation_outputs(temporary, "detection", records, metrics)
            self.assertEqual(predictions.name, "forensics_predictions_detection.jsonl")
            self.assertEqual(summary.name, "forensics_metrics_detection.json")
            self.assertTrue(Path(predictions).is_file())


if __name__ == "__main__":
    unittest.main()
