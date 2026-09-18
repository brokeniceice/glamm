"""Patch-aligned cross-layer attention for frozen CLIP dense tokens."""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossLayerPatchAttention(nn.Module):
    """Attend from block-17 to the two layer tokens at the same patch only."""

    def __init__(self, dim: int = 1024, heads: int = 8, gamma_init: float = 0.01):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _tokens(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if value.ndim == 4:
            b, c, h, w = value.shape
            return value.flatten(2).transpose(1, 2), (h, w)
        if value.ndim == 3:
            return value, None
        raise ValueError(f"expected BCHW or BNC, got {tuple(value.shape)}")

    def forward(self, f11: torch.Tensor, f17: torch.Tensor, *, need_weights: bool = False):
        x11, spatial = self._tokens(f11)
        x17, spatial17 = self._tokens(f17)
        if x11.shape != x17.shape or spatial != spatial17 or x17.shape[-1] != self.dim:
            raise ValueError(f"aligned layer shape mismatch: {tuple(x11.shape)} vs {tuple(x17.shape)}")
        b, n, d = x17.shape
        query = x17.reshape(b * n, 1, d)
        key_value = torch.stack((x11, x17), dim=2).reshape(b * n, 2, d)
        attended, weights = self.attention(
            query, key_value, key_value, need_weights=need_weights,
            average_attn_weights=False,
        )
        fused = (query + self.gamma * attended).reshape(b, n, d)
        if spatial is not None:
            fused = fused.transpose(1, 2).reshape(b, d, *spatial)
        if need_weights:
            # [B*N, heads, 1, 2] -> [B,N,heads,2]
            weights = weights.squeeze(2).reshape(b, n, self.heads, 2)
            return fused, weights
        return fused


class ThreeLayerPatchAttention(nn.Module):
    """Attend from block-17 to block-11/17/22 at the same patch only."""

    def __init__(self, dim: int = 1024, heads: int = 8, gamma_init: float = 0.01):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, f11: torch.Tensor, f17: torch.Tensor, f22: torch.Tensor,
                *, need_weights: bool = False):
        x11, spatial = CrossLayerPatchAttention._tokens(f11)
        x17, spatial17 = CrossLayerPatchAttention._tokens(f17)
        x22, spatial22 = CrossLayerPatchAttention._tokens(f22)
        if x11.shape != x17.shape or x17.shape != x22.shape or not (spatial == spatial17 == spatial22):
            raise ValueError("aligned three-layer shape mismatch")
        b, n, d = x17.shape
        query = x17.reshape(b * n, 1, d)
        key_value = torch.stack((x11, x17, x22), dim=2).reshape(b * n, 3, d)
        attended, weights = self.attention(
            query, key_value, key_value, need_weights=need_weights,
            average_attn_weights=False,
        )
        fused = (query + self.gamma * attended).reshape(b, n, d)
        if spatial is not None:
            fused = fused.transpose(1, 2).reshape(b, d, *spatial)
        if need_weights:
            weights = weights.squeeze(2).reshape(b, n, self.heads, 3)
            return fused, weights
        return fused
