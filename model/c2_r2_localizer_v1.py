"""Phase6L1 R2-v1: distinct radial bridge initialization and LayerScale.

The inherited R2-v0 architecture is otherwise unchanged.  Phase6L0 source
and checkpoints remain immutable comparison artifacts.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from model.c2_r2_localizer import C2R2Localizer


RADII = (0.5, 1.5, 2.5, 3.5)


class C2R2LocalizerV1(C2R2Localizer):
    def __init__(self) -> None:
        super().__init__()
        head = self.bridge.offset_head
        with torch.no_grad():
            head.weight.zero_()
            bias = head.bias.reshape(self.bridge.heads, self.bridge.samples, 2)
            for h in range(self.bridge.heads):
                theta = 2 * math.pi * h / self.bridge.heads
                direction = (math.cos(theta), math.sin(theta))
                for k, radius in enumerate(RADII):
                    for xy in range(2):
                        desired = radius * direction[xy]
                        fraction = max(-0.999, min(0.999, desired / self.bridge.max_offset))
                        bias[h, k, xy] = math.atanh(fraction)
        self.alpha = nn.Parameter(torch.full((1, 256, 1, 1), 1e-2))

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        raw = output["deltaS"]
        delta = self.alpha * raw
        output["delta_raw"] = raw
        output["deltaS"] = delta
        output["S_adapt"] = kwargs["s64"].float() + delta
        return output
