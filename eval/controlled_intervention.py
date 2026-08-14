"""Same-image controlled segmentation interventions and integrity checks."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from eval.inference_trace import unwrap_glamm


@torch.no_grad()
def decode_projected_embedding(model, trace: Mapping[str, Any], projected: torch.Tensor,
                               *, resize: tuple[int, int], original_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    core = unwrap_glamm(model)
    sparse, dense = core.model.grounding_encoder.prompt_encoder(
        points=None, boxes=None, masks=None, text_embeds=projected.unsqueeze(1)
    )
    sparse = sparse.to(projected.dtype)
    return decode_sam_prompt(
        model, trace, sparse=sparse, dense=dense, resize=resize, original_size=original_size
    ) | {"projected_embedding": projected.detach()}


@torch.no_grad()
def decode_predictor_hidden(model, trace: Mapping[str, Any], predictor_hidden: torch.Tensor,
                            *, resize: tuple[int, int], original_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    core = unwrap_glamm(model)
    projected = core.model.text_hidden_fcs[0](predictor_hidden)
    return decode_projected_embedding(
        model, trace, projected, resize=resize, original_size=original_size
    ) | {"llm_predictor_hidden": predictor_hidden.detach()}


@torch.no_grad()
def decode_sam_prompt(model, trace: Mapping[str, Any], *, sparse: torch.Tensor, dense: torch.Tensor,
                      resize: tuple[int, int], original_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    core = unwrap_glamm(model)
    low_res, _ = core.model.grounding_encoder.mask_decoder(
        image_embeddings=trace["image_embedding"].unsqueeze(0),
        image_pe=core.model.grounding_encoder.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    postprocessed = core.model.grounding_encoder.postprocess_masks(
        low_res, input_size=resize, original_size=original_size
    )[:, 0]
    return {"sparse_prompt_embedding": sparse.detach(), "dense_prompt_embedding": dense.detach(),
            "low_res_mask_logits": low_res.detach(), "postprocessed_mask_logits": postprocessed.detach(),
            "binary_mask": postprocessed.gt(0).detach()}


def validate_intervention(base: Mapping[str, Any], donor: Mapping[str, Any], *,
                          base_sample_id: str, donor_sample_id: str, intervention: str,
                          checkpoint_sha256: str, donor_checkpoint_sha256: str,
                          threshold: float = 0.0) -> dict[str, Any]:
    violations = []
    if base_sample_id != donor_sample_id:
        violations.append("sample_identity")
    if checkpoint_sha256 != donor_checkpoint_sha256:
        violations.append("checkpoint")
    if threshold != 0.0:
        violations.append("mask_threshold")
    if not torch.equal(base["image_embedding"], donor["image_embedding"]):
        violations.append("image_embedding")
    allowed = {
        "H2a_predictor_hidden_swap": "llm_predictor_hidden",
        "H2b_projection_swap": "projected_embedding",
        "H2c_sam_prompt_swap": "sparse_prompt_embedding+dense_prompt_embedding",
    }
    if intervention not in allowed:
        violations.append("unknown_intervention")
    return {"status": "VALID" if not violations else "INTERVENTION_SCOPE_VIOLATION",
            "violations": violations, "declared_difference": allowed.get(intervention),
            "sample_id": base_sample_id, "threshold": threshold}
