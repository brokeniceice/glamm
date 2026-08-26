"""Phase 4C-B single-layer language-query evidence reader."""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceReader(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if embed_dim != 256 or num_heads != 4 or dropout != 0.0:
            raise ValueError("Phase 4C-B reader architecture is frozen to 256D/4-head/dropout0")
        self.query_norm = nn.LayerNorm(embed_dim)
        self.source_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, q_seg: torch.Tensor, spatial: torch.Tensor,
                *, need_weights: bool = True):
        if q_seg.ndim == 2:
            q_seg = q_seg[:, None, :]
        if spatial.ndim == 4:
            spatial = spatial.flatten(2).transpose(1, 2)
        if q_seg.ndim != 3 or q_seg.shape[1:] != (1, 256):
            raise ValueError(f"q_seg must be [B,1,256], got {tuple(q_seg.shape)}")
        if spatial.ndim != 3 or spatial.shape[-1] != 256:
            raise ValueError(f"spatial must be [B,N,256], got {tuple(spatial.shape)}")
        evidence, weights = self.cross_attention(
            self.query_norm(q_seg), self.source_norm(spatial), self.source_norm(spatial),
            need_weights=need_weights, average_attn_weights=False,
        )
        final = q_seg + self.beta * evidence
        return {"q_final": final, "q_evidence": evidence, "attention": weights}

