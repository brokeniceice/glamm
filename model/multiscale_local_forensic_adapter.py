"""Minimal multi-scale local forensic adapter for Phase 6G.4."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleLocalForensicAdapter(nn.Module):
    def __init__(self, gamma_init: float = 0.01):
        super().__init__()
        self.projection = nn.Conv2d(1024, 256, 1)
        self.branches = nn.ModuleList([
            nn.Identity(),
            nn.Conv2d(64, 64, 3, padding=1, dilation=1, groups=64),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, groups=64),
            nn.Conv2d(64, 64, 3, padding=3, dilation=3, groups=64),
        ])
        self.norm = nn.GroupNorm(32, 256)
        self.activation = nn.GELU()
        self.channel_mixing = nn.Conv2d(256, 256, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.dense_head = nn.Conv2d(256, 1, 1, bias=True)

    def forward(self, fused_feature: torch.Tensor, *, return_features: bool = False):
        base = self.projection(fused_feature)
        chunks = base.chunk(4, dim=1)
        local = torch.cat([branch(value) for branch, value in zip(self.branches, chunks)], dim=1)
        delta = self.channel_mixing(self.activation(self.norm(local)))
        residual = self.gamma * delta
        output = base + residual
        logits = self.dense_head(output)
        if return_features:
            return {"logits": logits, "F_base": base, "delta": delta,
                    "residual": residual, "F_out": output}
        return logits
