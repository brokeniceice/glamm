"""Exact full-forward and cache-stepwise replay helpers."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from eval.inference_trace import raw_seg_positions, unwrap_glamm
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


@torch.no_grad()
def replay_stepwise_predictor_hidden(model, batch: Mapping[str, Any], input_ids: torch.Tensor,
                                     *, prompt_length: int) -> dict[str, Any]:
    """Replay a fixed sequence using the same cache update semantics as generation."""
    core = unwrap_glamm(model)
    seg_position, predictor_position = raw_seg_positions(input_ids, core.seg_token_idx)
    if prompt_length <= 0 or predictor_position < prompt_length - 1:
        raise ValueError("Invalid prompt length for the replay sequence")
    prompt = input_ids[:, :prompt_length]
    attention = prompt.ne(core.config.pad_token_id)
    output = LlavaLlamaForCausalLM.forward(
        core, input_ids=prompt, attention_mask=attention, images=batch["global_enc_images"],
        bboxes=batch["bboxes"], use_cache=True, output_hidden_states=True, return_dict=True,
    )
    past = output.past_key_values
    current_raw_position = prompt_length - 1
    predictor_hidden = output.hidden_states[:, -1]
    step_count = 0
    while current_raw_position < predictor_position:
        current_raw_position += 1
        token = input_ids[:, current_raw_position:current_raw_position + 1]
        output = LlavaLlamaForCausalLM.forward(
            core, input_ids=token, attention_mask=None, past_key_values=past,
            images=batch["global_enc_images"], bboxes=batch["bboxes"], use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        past = output.past_key_values
        predictor_hidden = output.hidden_states[:, -1]
        step_count += 1
    return {
        "llm_predictor_hidden": predictor_hidden.detach(), "seg_token_position_raw": seg_position,
        "predictor_position_raw": predictor_position, "prompt_length": prompt_length,
        "cache_steps_after_prompt": step_count,
        "execution_semantics": "prompt_full_forward_then_one_raw_token_per_cached_step",
    }
