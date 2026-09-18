"""Deterministic full-FOV acquisition for frozen CLIP forensic evidence.

The adapter and evidential head are deliberately applied independently to
each 336x336 tile.  This module contains geometry only and has no labels,
thresholds, selectors, or trainable parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.image_transforms import resize

from model.pcerf import evidence_to_dirichlet


TILE_SIZE = 336
STRIDE = 280
PATCH_GRID = 24


@dataclass(frozen=True)
class TileGeometry:
    resized_hw: tuple[int, int]
    tile_boxes_yxyx: tuple[tuple[int, int, int, int], ...]
    full_grid_hw: tuple[int, int]


def tile_origins(long_size: int, *, tile_size: int = TILE_SIZE, stride: int = STRIDE) -> list[int]:
    if long_size < tile_size:
        raise ValueError("resized long side cannot be smaller than the tile")
    if long_size == tile_size:
        return [0]
    values = list(range(0, long_size - tile_size + 1, stride))
    end = long_size - tile_size
    if values[-1] != end:
        values.append(end)
    return values


def full_fov_geometry(original_hw: Sequence[int]) -> TileGeometry:
    height, width = map(int, original_hw)
    if min(height, width) <= 0:
        raise ValueError("invalid original image geometry")
    if height <= width:
        resized_h = TILE_SIZE
        resized_w = int(TILE_SIZE * width / height)
        boxes = tuple((0, x, TILE_SIZE, x + TILE_SIZE) for x in tile_origins(resized_w))
    else:
        resized_w = TILE_SIZE
        resized_h = int(TILE_SIZE * height / width)
        boxes = tuple((y, 0, y + TILE_SIZE, TILE_SIZE) for y in tile_origins(resized_h))
    full_h = max(1, int(round(PATCH_GRID * resized_h / TILE_SIZE)))
    full_w = max(1, int(round(PATCH_GRID * resized_w / TILE_SIZE)))
    return TileGeometry((resized_h, resized_w), boxes, (full_h, full_w))


def preprocess_tiles(image: Image.Image, processor, geometry: TileGeometry) -> torch.Tensor:
    """Reproduce CLIP resize exactly, then disable resize/crop per tile."""
    rgb = np.asarray(image.convert("RGB"))
    resized = resize(rgb, size=geometry.resized_hw, resample=processor.resample)
    tiles = [resized[top:bottom, left:right] for top, left, bottom, right in geometry.tile_boxes_yxyx]
    if any(tile.shape[:2] != (TILE_SIZE, TILE_SIZE) for tile in tiles):
        raise RuntimeError("tile extraction produced a non-336 tile")
    return processor.preprocess(
        tiles, return_tensors="pt", do_resize=False, do_center_crop=False,
    )["pixel_values"]


def full_grid_coordinates(geometry: TileGeometry, *, device=None) -> torch.Tensor:
    height, width = geometry.full_grid_hw
    resized_h, resized_w = map(float, geometry.resized_hw)
    # Preserve both the legacy clip_coordinates floating-point operation order
    # and its CPU construction before the final device transfer.  Constructing
    # the same divisions directly on CUDA differs by ~6e-8 on some cells,
    # enough to change BF16 locality-biased attention rounding.
    y = ((torch.arange(height, dtype=torch.float32) + 0.5)
         * resized_h / height) / resized_h
    x = ((torch.arange(width, dtype=torch.float32) + 0.5)
         * resized_w / width) / resized_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(device=device)


def _tile_sample_and_weight(
    tile: torch.Tensor, box: tuple[int, int, int, int], geometry: TileGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one tile on the canonical full grid with a geometry-only Hann feather."""
    if tile.ndim != 4 or tile.shape[0] != 1:
        raise ValueError("one tile tensor must be [1,C,H,W]")
    full_h, full_w = geometry.full_grid_hw
    resized_h, resized_w = geometry.resized_hw
    top, left, bottom, right = map(float, box)
    y = (torch.arange(full_h, device=tile.device, dtype=torch.float32) + 0.5) * resized_h / full_h
    x = (torch.arange(full_w, device=tile.device, dtype=torch.float32) + 0.5) * resized_w / full_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    support = (yy >= top) & (yy < bottom) & (xx >= left) & (xx < right)
    u = ((xx - left) / (right - left)).clamp(0.0, 1.0)
    v = ((yy - top) / (bottom - top)).clamp(0.0, 1.0)
    # Cell centres never lie exactly on a tile boundary for the square parity
    # case.  Clamp protects unusual fractional end anchors from zero/zero.
    weight = (torch.sin(torch.pi * u) * torch.sin(torch.pi * v)).clamp_min(1e-6) * support
    grid = torch.stack((2.0 * u - 1.0, 2.0 * v - 1.0), dim=-1)[None]
    sampled = F.grid_sample(
        tile.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return sampled, weight[None, None]


def stitch_tiles(
    tiles: torch.Tensor, geometry: TileGeometry, *, order: Sequence[int] | None = None,
    normalize_channels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feather-and-normalize tile tensors into one native-density full-FOV grid."""
    if tiles.ndim != 4 or tiles.shape[0] != len(geometry.tile_boxes_yxyx):
        raise ValueError("tile count/geometry mismatch")
    indices = list(range(tiles.shape[0])) if order is None else list(map(int, order))
    if sorted(indices) != list(range(tiles.shape[0])):
        raise ValueError("order must be a permutation of every tile")
    # With one tile there is no overlap to fuse.  Preserve a strict FP32
    # identity so the finalized tensor casts back to the exact legacy BF16
    # values at learned-module boundaries; multiplying/dividing by a feather
    # window would introduce avoidable roundoff before that cast.
    if tiles.shape[0] == 1:
        result = tiles[:1].float()
        if normalize_channels:
            # Evidential masses were normalized by the frozen source head.
            # Avoid a second normalization here; the aligned mapping performs
            # the same post-resampling normalization as the legacy path.
            if not bool(torch.allclose(
                result.sum(dim=1), torch.ones_like(result[:, 0]), atol=2e-6, rtol=0.0,
            )):
                raise RuntimeError("single-tile evidential masses are not normalized")
        support = torch.ones(
            (1, 1, *geometry.full_grid_hw), device=tiles.device, dtype=torch.bool,
        )
        return result, support
    shape = (1, tiles.shape[1], *geometry.full_grid_hw)
    numerator = torch.zeros(shape, device=tiles.device, dtype=torch.float32)
    denominator = torch.zeros((1, 1, *geometry.full_grid_hw), device=tiles.device, dtype=torch.float32)
    for index in indices:
        sampled, weight = _tile_sample_and_weight(
            tiles[index:index + 1], geometry.tile_boxes_yxyx[index], geometry,
        )
        numerator.add_(sampled * weight)
        denominator.add_(weight)
    support = denominator > 0
    if not bool(support.all()):
        raise RuntimeError("full-FOV stitch contains uncovered cells")
    result = numerator / denominator.clamp_min(1e-12)
    if normalize_channels:
        result = result.clamp_min(0.0)
        result = result / result.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return result, support


def align_full_grid(
    value: torch.Tensor, support: torch.Tensor, output_hw: tuple[int, int], *,
    normalize_channels: bool = False, vacuous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4 or support.shape != (value.shape[0], 1, *value.shape[-2:]):
        raise ValueError("full-grid value/support mismatch")
    out_h, out_w = output_hw
    y = (torch.arange(out_h, device=value.device, dtype=torch.float32) + 0.5) / out_h
    x = (torch.arange(out_w, device=value.device, dtype=torch.float32) + 0.5) / out_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((2.0 * xx - 1.0, 2.0 * yy - 1.0), dim=-1)[None].expand(value.shape[0], -1, -1, -1)
    aligned = F.grid_sample(
        value.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    aligned_support = F.interpolate(support.float(), output_hw, mode="nearest").bool()
    if normalize_channels:
        aligned = aligned.clamp_min(0.0)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-12)
    aligned = torch.where(aligned_support, aligned, torch.zeros_like(aligned))
    if vacuous:
        vacuous_value = torch.zeros_like(aligned)
        vacuous_value[:, -1] = 1.0
        aligned = torch.where(aligned_support, aligned, vacuous_value)
    return aligned, aligned_support


@torch.no_grad()
def acquire_full_fov(
    image: Image.Image, processor, vision, adapter, forensic_head, temperature: torch.Tensor,
    device: torch.device, *, order: Sequence[int] | None = None,
) -> dict[str, object]:
    """Run the frozen source independently per tile and stitch F/z/masses."""
    geometry = full_fov_geometry((image.height, image.width))
    pixels = preprocess_tiles(image, processor, geometry).to(device=device, dtype=torch.bfloat16)
    vision_output = vision(pixels, output_hidden_states=True)
    patch = vision_output.hidden_states[-2][:, 1:]
    raw = patch.transpose(1, 2).reshape(len(geometry.tile_boxes_yxyx), patch.shape[-1], 24, 24)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        adapted = adapter(raw, return_features=True)
        evidence = forensic_head(adapted["F_forensic"], adapted["logits"]) / temperature
    masses = evidence_to_dirichlet(evidence.float())["masses"]
    f_full, support_f = stitch_tiles(adapted["F_forensic"], geometry, order=order)
    z_full, support_z = stitch_tiles(adapted["logits"], geometry, order=order)
    mass_full, support_mass = stitch_tiles(masses, geometry, order=order, normalize_channels=True)
    if not torch.equal(support_f, support_z) or not torch.equal(support_f, support_mass):
        raise RuntimeError("F/z/mass support drift")
    return {
        "geometry": geometry, "raw_tiles": raw, "F_tiles": adapted["F_forensic"],
        "z_tiles": adapted["logits"], "mass_tiles": masses,
        "F_full": f_full, "z_full": z_full, "mass_full": mass_full,
        "support_full": support_f, "coordinates": full_grid_coordinates(geometry, device=device),
    }


__all__ = [
    "PATCH_GRID", "STRIDE", "TILE_SIZE", "TileGeometry", "acquire_full_fov",
    "align_full_grid", "full_fov_geometry", "full_grid_coordinates", "preprocess_tiles",
    "stitch_tiles", "tile_origins",
]
