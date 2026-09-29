"""Phase 6J C2 fusion, installed without changing the historical C1 model."""

from __future__ import annotations

import math

import torch
from torch import nn

from model.GLaMM import GLaMMForCausalLM


class PreLNCrossAttention(nn.Module):
    """One RINE query reads the already projected spatial CLIP patches."""

    def __init__(self, hidden_size: int, inner_dim: int = 512, heads: int = 8):
        super().__init__()
        if inner_dim % heads:
            raise ValueError("inner_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_v = nn.LayerNorm(hidden_size)
        self.w_q = nn.Linear(hidden_size, inner_dim)
        self.w_k = nn.Linear(hidden_size, inner_dim)
        self.w_v = nn.Linear(hidden_size, inner_dim)
        self.w_o = nn.Linear(inner_dim, hidden_size)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)
        self.last_stats: dict[str, float] = {}
        self.last_gradient_norms: dict[str, float] = {}
        # Optional read-only Phase6K capture. The numerical forward below is unchanged.
        self.capture_spatial_intermediates = False
        self.last_spatial_intermediates = None
        for name, parameter in self.named_parameters():
            parameter.register_hook(self._gradient_hook(name))

    def _gradient_hook(self, name: str):
        def record(gradient):
            self.last_gradient_norms[name] = float(gradient.detach().float().norm())
            return gradient
        return record

    def forward(self, r: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if r.ndim != 3 or r.shape[1] != 1 or patches.ndim != 3:
            raise ValueError("Expected forensic token [B,1,H] and CLIP patches [B,N,H]")
        if r.shape[0] != patches.shape[0] or r.shape[2] != patches.shape[2]:
            raise ValueError("C2 forensic token and CLIP patches have incompatible shapes")
        batch, count, _ = patches.shape
        q = self.w_q(self.ln_r(r)).view(batch, 1, self.heads, self.head_dim).transpose(1, 2)
        normalized_patches = self.ln_v(patches)
        k = self.w_k(normalized_patches).view(batch, count, self.heads, self.head_dim).transpose(1, 2)
        v = self.w_v(normalized_patches).view(batch, count, self.heads, self.head_dim).transpose(1, 2)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        probabilities = scores.softmax(dim=-1)
        context = (probabilities.to(v.dtype) @ v).transpose(1, 2).reshape(batch, 1, -1)
        if self.capture_spatial_intermediates:
            if count != 24 * 24 or self.heads != 8 or self.head_dim != 64:
                raise ValueError("Phase6K requires the trained 8x64, 24x24 C2 layout")
            weighted = probabilities.float().squeeze(2).unsqueeze(-1) * v.float()
            self.last_spatial_intermediates = {
                "A": probabilities.detach().squeeze(2).reshape(batch, 8, 24, 24).cpu(),
                "E": weighted.detach().permute(0, 1, 3, 2).reshape(batch, 512, 24, 24).cpu(),
                "head_context": context.detach().reshape(batch, 8, 64).cpu(),
                "fused_input": r.detach().cpu(),
            }
        delta = self.w_o(context).to(r.dtype)
        fused = r + delta
        with torch.no_grad():
            p = probabilities.detach().float().clamp_min(1e-12)
            r_norm = r.detach().float().norm(dim=-1).mean()
            self.last_stats = {
                "residual_ratio": float((delta.detach().float().norm(dim=-1).mean() / r_norm.clamp_min(1e-12))),
                "fused_norm": float(fused.detach().float().norm(dim=-1).mean()),
                "attention_entropy": float((-(p * p.log()).sum(dim=-1)).mean()),
                "normalized_attention_entropy": float((-(p * p.log()).sum(dim=-1)).mean() / math.log(count)),
                "attention_max_mass": float(probabilities.detach().amax(dim=-1).float().mean()),
            }
        return fused


class C2GLaMMForCausalLM(GLaMMForCausalLM):
    """C1 model with a single additional pre-LN fusion block."""

    def enable_rine_conditioning(self, rine_checkpoint, *, projector_dtype=None):
        super().enable_rine_conditioning(rine_checkpoint, projector_dtype=projector_dtype)
        self.c2_cross_attention = PreLNCrossAttention(int(self.config.hidden_size))
        if projector_dtype is not None:
            self.c2_cross_attention.to(dtype=projector_dtype)

    def encode_images(self, images):
        cached = getattr(self, "_c2_cached_image_features", None)
        if cached is not None:
            self._c2_cached_image_features = None
            return cached
        return super().encode_images(images)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, bboxes,
        forensic_evidence_tokens=None,
    ):
        if (
            forensic_evidence_tokens is None or images is None or input_ids.shape[1] == 1
        ):
            return super().prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, bboxes,
                forensic_evidence_tokens=forensic_evidence_tokens,
            )
        if isinstance(images, list) or images.ndim != 4:
            raise ValueError("C2 expects one [B,C,H,W] image tensor per batch")
        if getattr(self, "_c2_cached_image_features", None) is not None:
            raise RuntimeError("Nested C2 image-feature preparation")
        features = super().encode_images(images)
        self._c2_cached_image_features = features
        try:
            fused = self.c2_cross_attention(forensic_evidence_tokens, features[0])
            return super().prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, bboxes,
                forensic_evidence_tokens=fused,
            )
        finally:
            self._c2_cached_image_features = None
