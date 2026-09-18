"""Minimal spatial-prior interaction adapter for dense forensic prediction."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip_forensic_adapter import LocalForensicBlock


class SpatialPriorStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128),
            nn.Conv2d(128, 256, 1), nn.GroupNorm(32, 256), nn.GELU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.stem(rgb), size=(24, 24), mode="bilinear", align_corners=False)


class SpatialPriorInteractionAdapter(nn.Module):
    def __init__(self, heads: int = 8, gamma_init: float = 0.01):
        super().__init__()
        self.projection = nn.Conv2d(1024, 256, 1)
        self.spatial_prior = SpatialPriorStem()
        self.interaction = nn.MultiheadAttention(256, heads, batch_first=True)
        self.gamma_spatial = nn.Parameter(torch.tensor(float(gamma_init)))
        self.forensic_blocks = nn.Sequential(*(LocalForensicBlock(256, 32) for _ in range(3)))
        self.dense_head = nn.Conv2d(256, 1, 1, bias=True)
        self.heads = heads

    def forward(self, clip_feature: torch.Tensor, rgb: torch.Tensor, *,
                return_features: bool = False, need_weights: bool = False):
        clip = self.projection(clip_feature)
        spatial = self.spatial_prior(rgb)
        q = clip.flatten(2).transpose(1, 2)
        kv = spatial.flatten(2).transpose(1, 2)
        attended, weights = self.interaction(
            q, kv, kv, need_weights=need_weights, average_attn_weights=False,
        )
        residual = self.gamma_spatial * attended
        mixed = (q + residual).transpose(1, 2).reshape_as(clip)
        forensic = self.forensic_blocks(mixed)
        logits = self.dense_head(forensic)
        if return_features:
            return {"logits": logits, "F0": clip, "F_spatial": spatial,
                    "F_interaction": mixed, "F_forensic": forensic,
                    "residual_tokens": residual, "attention": weights}
        return logits
