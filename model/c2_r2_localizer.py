"""C2-native R2: geometry-aligned forensic evidence to a frozen-SAM residual.

Only this module's parameters are trainable. C2 outputs and S64 are inputs;
the caller decodes S_adapt with the frozen C2 SAM without torch.no_grad().
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tools.phase3c1 import geometry_for
from tools.phase4e1 import sam_coordinates


def _projection(channels_in: int, channels_out: int, groups: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels_in, channels_out, 1),
                         nn.GroupNorm(groups, channels_out), nn.GELU())


class _LocalForensicResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, 256)
        self.depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise = nn.Conv2d(256, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(F.gelu(self.depthwise(self.norm(x))))


class EvidenceConsolidator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip = _projection(1024, 256, 32)
        self.evidence = _projection(512, 256, 32)
        self.attention = _projection(8, 32, 8)
        self.fuse = _projection(544, 256, 32)
        self.blocks = nn.Sequential(*[_LocalForensicResidualBlock() for _ in range(3)])

    def forward(self, raw_clip_grid: torch.Tensor, attention_map: torch.Tensor,
                evidence_map: torch.Tensor) -> torch.Tensor:
        if raw_clip_grid.shape[1:] != (1024, 24, 24) or attention_map.shape[1:] != (8, 24, 24) or evidence_map.shape[1:] != (512, 24, 24):
            raise ValueError("Expected raw CLIP [B,1024,24,24], A [B,8,24,24], E [B,512,24,24]")
        if not (raw_clip_grid.shape[0] == attention_map.shape[0] == evidence_map.shape[0]):
            raise ValueError("Evidence batch mismatch")
        return self.blocks(self.fuse(torch.cat((self.clip(raw_clip_grid),
                                                self.evidence(evidence_map),
                                                self.attention(attention_map)), dim=1)))


def resample_clip_to_sam_lattice(
    feature24: torch.Tensor, clip_geometries: Sequence[Mapping[str, object]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use audited SAM cell centers and CLIP crop geometry on the padded S64 grid.

    This differs from an original-normalized 64-grid for non-square images:
    the right/bottom SAM cells include padding, and must have no CLIP support.
    """
    if feature24.ndim != 4 or feature24.shape[-2:] != (24, 24):
        raise ValueError("feature24 must be [B,C,24,24]")
    if len(clip_geometries) != feature24.shape[0]:
        raise ValueError("one CLIP geometry per sample is required")
    grids, supports = [], []
    for geometry in clip_geometries:
        if geometry.get("source") != "clip" or geometry.get("kind") != "resize_shortest_center_crop":
            raise ValueError("Expected audited CLIP center-crop geometry")
        expected = geometry_for("clip", geometry["original_hw"])
        if tuple(geometry["resized_hw"]) != tuple(expected["resized_hw"]) or tuple(geometry["crop_box_yxyx"]) != tuple(expected["crop_box_yxyx"]):
            raise ValueError("CLIP geometry differs from canonical geometry_for")
        xy = sam_coordinates(geometry_for("sam", geometry["original_hw"]), grid=64).reshape(64, 64, 2).to(feature24.device)
        resized_h, resized_w = map(float, geometry["resized_hw"])
        top, left, bottom, right = map(float, geometry["crop_box_yxyx"])
        x, y = xy[..., 0] * resized_w, xy[..., 1] * resized_h
        support = (x >= left) & (x < right) & (y >= top) & (y < bottom)
        grids.append(torch.stack((2 * (x - left) / (right - left) - 1,
                                  2 * (y - top) / (bottom - top) - 1), dim=-1))
        supports.append(support)
    grid = torch.stack(grids).float()
    support64 = torch.stack(supports)[:, None]
    mapped = F.grid_sample(feature24.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.where(support64, mapped, torch.zeros_like(mapped)), support64


class GlobalConditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(256)
        self.r_norm = nn.LayerNorm(4096)
        self.r_projection = nn.Linear(4096, 256)
        self.fusion = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, q_seg: torch.Tensor, r_prime: torch.Tensor) -> torch.Tensor:
        if q_seg.ndim == 3 and q_seg.shape[1] == 1:
            q_seg = q_seg[:, 0]
        if r_prime.ndim == 3 and r_prime.shape[1] == 1:
            r_prime = r_prime[:, 0]
        if q_seg.ndim != 2 or q_seg.shape[1] != 256 or r_prime.ndim != 2 or r_prime.shape[1] != 4096 or q_seg.shape[0] != r_prime.shape[0]:
            raise ValueError("Expected q_seg [B,256] and r_prime [B,4096], optional singleton axes")
        return self.fusion(torch.cat((self.q_norm(q_seg),
                                      F.gelu(self.r_projection(self.r_norm(r_prime)))), dim=1))


class GeometryAwareBridge(nn.Module):
    heads = 8
    samples = 4
    head_dim = 32
    max_offset = 4.0

    def __init__(self) -> None:
        super().__init__()
        self.z_encoder = _projection(1, 32, 8)
        self.base = _projection(288, 256, 32)
        self.q_norm = nn.GroupNorm(32, 256)
        self.q_film = nn.Linear(256, 512)
        # A tiny nonzero scale keeps g trainable on the second backward after
        # the zero residual head's first update. Exact zero would delay it.
        nn.init.normal_(self.q_film.weight, std=1e-4)
        nn.init.zeros_(self.q_film.bias)
        self.query = nn.Conv2d(256, 256, 1)
        self.key = nn.Conv2d(256, 256, 1)
        self.value = nn.Conv2d(256, 256, 1)
        self.offset_head = nn.Conv2d(256, self.heads * self.samples * 2, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.output = nn.Conv2d(256, 256, 1)

    def forward(self, s64: torch.Tensor, z64: torch.Tensor, f64: torch.Tensor,
                support64: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
        b = s64.shape[0]
        if s64.shape != (b, 256, 64, 64) or f64.shape != s64.shape or z64.shape != (b, 1, 64, 64) or support64.shape != (b, 1, 64, 64) or g.shape != (b, 256):
            raise ValueError("Bridge expects S64/F64 [B,256,64,64], z64/support [B,1,64,64], g [B,256]")
        base = self.base(torch.cat((s64, self.z_encoder(z64)), dim=1))
        gamma, beta = self.q_film(g).chunk(2, dim=1)
        query_state = self.q_norm(base) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        offsets = self.max_offset * torch.tanh(self.offset_head(query_state).reshape(b, 8, 4, 2, 64, 64))
        y = (torch.arange(64, device=s64.device, dtype=torch.float32) + .5) * (2 / 64) - 1
        x = (torch.arange(64, device=s64.device, dtype=torch.float32) + .5) * (2 / 64) - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        reference = torch.stack((xx, yy), dim=0)[None, None, None]
        grid = (reference + offsets.float() * (2 / 64)).permute(0, 1, 2, 4, 5, 3).reshape(b * 8, 4 * 64, 64, 2)

        def sample(tensor: torch.Tensor, channels: int) -> torch.Tensor:
            source = tensor.reshape(b * 8, channels, 64, 64)
            result = F.grid_sample(source.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            return result.reshape(b, 8, channels, 4, 64, 64).permute(0, 1, 3, 2, 4, 5)

        keys = sample(self.key(f64), 32)
        values = sample(self.value(f64), 32)
        support_source = support64.float().expand(b, 8, 64, 64).reshape(b * 8, 1, 64, 64)
        sampled_support = F.grid_sample(support_source, grid, mode="bilinear", padding_mode="zeros", align_corners=False).reshape(b, 8, 4, 64, 64)
        valid = (sampled_support > 0.5) & (grid.reshape(b, 8, 4, 64, 64, 2).abs() <= 1).all(dim=-1)
        queries = self.query(query_state).reshape(b, 8, 32, 64, 64).float()
        scores = (queries[:, :, None] * keys).sum(dim=3) / math.sqrt(32)
        scores = scores.masked_fill(~valid, -1e4)
        attention = scores.softmax(dim=2) * valid.float()
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-12)
        response = (attention[:, :, :, None] * values).sum(dim=2).reshape(b, 256, 64, 64)
        any_valid = valid.any(dim=2).any(dim=1)[:, None]
        response = self.output(response) * any_valid
        return {"R64": response, "bridge_attention": attention,
                "bridge_offsets": offsets, "bridge_valid": valid}


class ConditionalResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, 256)
        self.film = nn.Linear(256, 512)
        nn.init.normal_(self.film.weight, std=1e-4)
        nn.init.zeros_(self.film.bias)
        self.depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise_in = nn.Conv2d(256, 512, 1)
        self.pointwise_out = nn.Conv2d(512, 256, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(g).chunk(2, dim=1)
        conditioned = self.norm(x) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return x + self.pointwise_out(F.gelu(self.pointwise_in(F.gelu(self.depthwise(conditioned)))))


class SAMResidualAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.z_encoder = _projection(1, 32, 8)
        self.input = _projection(545, 256, 32)
        self.blocks = nn.ModuleList([ConditionalResidualBlock() for _ in range(3)])
        self.output = nn.Conv2d(256, 256, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, s64: torch.Tensor, r64: torch.Tensor, z64: torch.Tensor,
                support64: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = self.input(torch.cat((s64, r64, self.z_encoder(z64), support64.float()), dim=1))
        for block in self.blocks:
            x = block(x, g)
        return self.output(x)


class C2R2Localizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evidence = EvidenceConsolidator()
        self.conditioner = GlobalConditioner()
        self.bridge = GeometryAwareBridge()
        self.adapter = SAMResidualAdapter()

    def forward(self, *, raw_clip_grid: torch.Tensor, attention_map: torch.Tensor,
                evidence_map: torch.Tensor, r_prime: torch.Tensor, q_seg: torch.Tensor,
                s64: torch.Tensor, z_l: torch.Tensor,
                clip_geometry: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if z_l.ndim != 4 or z_l.shape[:2] != (s64.shape[0], 1):
            raise ValueError("z_l must be [B,1,H,W]")
        f24 = self.evidence(raw_clip_grid.float(), attention_map.float(), evidence_map.float())
        f64, support64 = resample_clip_to_sam_lattice(f24, clip_geometry)
        z64 = F.interpolate(z_l.float(), size=(64, 64), mode="bilinear", align_corners=False)
        g = self.conditioner(q_seg.float(), r_prime.float())
        bridge = self.bridge(s64.float(), z64, f64, support64, g)
        delta = self.adapter(s64.float(), bridge["R64"], z64, support64, g)
        return {"F24_star": f24, "F64": f64, "support64": support64,
                **bridge, "deltaS": delta, "S_adapt": s64 + delta}
