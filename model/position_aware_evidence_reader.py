"""Phase 4D-1 fixed-position variant of the Phase 4C-B Evidence Reader."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def fixed_2d_sincos_position(height: int = 24, width: int = 24,
                            embed_dim: int = 256) -> torch.Tensor:
    """Return a deterministic [H*W,D] 2D sine/cosine lattice."""
    if embed_dim % 4:
        raise ValueError("2D sine/cosine encoding requires embed_dim divisible by 4")
    axis_dim = embed_dim // 2
    frequencies = torch.arange(axis_dim // 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * frequencies / (axis_dim // 2))

    def encode(values: torch.Tensor) -> torch.Tensor:
        phase = values[:, None].float() * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=1)

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.cat((encode(yy.reshape(-1)), encode(xx.reshape(-1))), dim=1)


class PositionAwareEvidenceReader(nn.Module):
    """The Phase 4C-B reader with one fixed 2D position buffer added to K/V."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.0, height: int = 24, width: int = 24):
        super().__init__()
        if embed_dim != 256 or num_heads != 4 or dropout != 0.0:
            raise ValueError("Phase 4D-1 reader is frozen to 256D/4-head/dropout0")
        if (height, width) != (24, 24):
            raise ValueError("Phase 4D-1 spatial lattice is frozen to 24x24")
        self.query_norm = nn.LayerNorm(embed_dim)
        self.source_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.beta = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "position_2d",
            fixed_2d_sincos_position(height, width, embed_dim),
            persistent=True,
        )

    def positioned_source(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim == 4:
            spatial = spatial.flatten(2).transpose(1, 2)
        if spatial.ndim != 3 or spatial.shape[1:] != (576, 256):
            raise ValueError(f"spatial must be [B,576,256], got {tuple(spatial.shape)}")
        return spatial + self.position_2d.to(device=spatial.device, dtype=spatial.dtype)[None]

    def forward(self, q_seg: torch.Tensor, spatial: torch.Tensor,
                *, need_weights: bool = True):
        if q_seg.ndim == 2:
            q_seg = q_seg[:, None, :]
        if q_seg.ndim != 3 or q_seg.shape[1:] != (1, 256):
            raise ValueError(f"q_seg must be [B,1,256], got {tuple(q_seg.shape)}")
        positioned = self.positioned_source(spatial)
        normalized = self.source_norm(positioned)
        evidence, weights = self.cross_attention(
            self.query_norm(q_seg), normalized, normalized,
            need_weights=need_weights, average_attn_weights=False,
        )
        residual = self.beta * evidence
        final = q_seg + residual
        return {
            "q_final": final,
            "q_evidence": evidence,
            "q_residual": residual,
            "attention": weights,
        }
