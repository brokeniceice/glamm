"""Phase6N0: frozen-basis, evidence-authorized C2-native R2."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

# Reuse only the audited CLIP-crop -> padded-SAM geometry function. No R2-v0
# module or checkpoint is instantiated or loaded by this architecture.
from model.c2_r2_localizer import resample_clip_to_sam_lattice


def conv_norm_gelu(input_channels: int, output_channels: int, kernel: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel, padding=kernel // 2),
                         nn.GroupNorm(8, output_channels), nn.GELU())


class SemanticContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = conv_norm_gelu(256, 64)
        self.z = conv_norm_gelu(1, 16)
        self.q = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU())
        self.merge = conv_norm_gelu(144, 64, 3)

    def forward(self, s64: torch.Tensor, z_l: torch.Tensor, q_seg: torch.Tensor) -> torch.Tensor:
        z64 = F.interpolate(z_l.float(), size=(64, 64), mode="bilinear", align_corners=False)
        q64 = self.q(q_seg.float())[:, :, None, None].expand(-1, -1, 64, 64)
        return self.merge(torch.cat((self.s(s64.float()), self.z(z64), q64), dim=1))


class ForensicContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = conv_norm_gelu(256, 64)
        self.e = conv_norm_gelu(512, 64)
        self.a = conv_norm_gelu(8, 16)
        self.r = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 64), nn.GELU())
        self.merge = conv_norm_gelu(209, 64, 3)

    def forward(self, f64: torch.Tensor, e64: torch.Tensor, a64: torch.Tensor,
                r_prime: torch.Tensor, support64: torch.Tensor) -> torch.Tensor:
        r64 = self.r(r_prime.float())[:, :, None, None].expand(-1, -1, 64, 64)
        value = self.merge(torch.cat((self.f(f64.float()), self.e(e64.float()),
                                      self.a(a64.float()), r64, support64.float()), dim=1))
        return value * support64.float()


class LocalResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(8, 64)
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise = nn.Conv2d(64, 64, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(F.gelu(self.depthwise(self.norm(value))))


class SourceAwareMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = conv_norm_gelu(257, 64, 3)
        self.local = LocalResidualBlock()

    def forward(self, language: torch.Tensor, forensic: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(language.float(), forensic.float(), dim=1, eps=1e-8)[:, None]
        features = torch.cat((language, forensic, language * forensic,
                              (language - forensic).abs(), cosine), dim=1)
        return self.local(self.fuse(features))


class CoefficientHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(64, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(hidden))


class AuthorityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(64, 1, 1)  # historical ordinary Conv initialization

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(hidden))


class C2LowRankEvidenceR2(nn.Module):
    def __init__(self, basis: torch.Tensor):
        super().__init__()
        if basis.shape != (256, 2) or not bool(torch.isfinite(basis).all()):
            raise ValueError("frozen intervention basis must be finite [256,2]")
        gram = basis.float().T @ basis.float()
        if float((gram - torch.eye(2)).abs().max()) > 1e-5:
            raise ValueError("frozen intervention basis is not orthonormal")
        self.register_buffer("basis", basis.detach().float().clone(), persistent=True)
        self.semantic = SemanticContext()
        self.forensic = ForensicContext()
        self.mixer = SourceAwareMixer()
        self.coefficient = CoefficientHead()
        self.authority = AuthorityHead()

    def forward(self, *, f24: torch.Tensor, attention_map: torch.Tensor,
                evidence_map: torch.Tensor, r_prime: torch.Tensor,
                q_seg: torch.Tensor, s64: torch.Tensor, z_l: torch.Tensor,
                clip_geometry: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        n = s64.shape[0]
        if (s64.shape != (n, 256, 64, 64) or f24.shape != (n, 256, 24, 24) or
            attention_map.shape != (n, 8, 24, 24) or evidence_map.shape != (n, 512, 24, 24) or
            r_prime.shape != (n, 4096) or q_seg.shape != (n, 256) or z_l.shape != (n, 1, 256, 256)):
            raise ValueError("Phase6N0 frozen input shape drift")
        f64, support64 = resample_clip_to_sam_lattice(f24, clip_geometry)
        a64, a_support = resample_clip_to_sam_lattice(attention_map, clip_geometry)
        e64, e_support = resample_clip_to_sam_lattice(evidence_map, clip_geometry)
        if not torch.equal(support64, a_support) or not torch.equal(support64, e_support):
            raise RuntimeError("forensic geometry support drift")
        language = self.semantic(s64, z_l, q_seg)
        forensic = self.forensic(f64, e64, a64, r_prime, support64)
        hidden = self.mixer(language, forensic)
        coefficients = self.coefficient(hidden)
        authority_logit = self.authority(hidden)
        gate = authority_logit.sigmoid()
        direction = torch.einsum("bkhw,ck->bchw", coefficients.float(), self.basis.float())
        delta = support64.float() * gate.float() * direction
        adapted = s64.float() + delta
        return {"a": coefficients, "authority_logit": authority_logit, "g": gate,
                "direction": direction, "deltaS": delta, "S_adapt": adapted,
                "support64": support64, "L64": language, "Fctx64": forensic,
                "H64": hidden}
