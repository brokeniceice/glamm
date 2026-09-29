"""Phase6K frozen C2 spatial transfer; projections are the only trainable modules."""
from __future__ import annotations

import torch
from torch import nn
from model.csculf import build_comparison_features
from model.pcerf import evidence_to_dirichlet, ecolaf_discount


class AfterProjection(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ('a', 'e'):
            raise ValueError(arm)
        self.arm = arm
        self.projection = nn.Conv2d(8 if arm == 'a' else 512, 256, 1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, forensic: torch.Tensor, spatial: torch.Tensor):
        residual = self.projection(spatial.float())
        return (forensic.float() + residual).to(forensic.dtype), residual


def utility_forward_full_input_gradient(model, batch):
    """Same frozen Utility values with the F24 input-gradient path retained.

    The historical helper detaches F24/z_F24 because it was written for Utility
    parameter training. Phase6K freezes those parameters and optimizes an
    upstream projection, so those detaches would silently omit this branch.
    """
    s64, qseg, zl = batch['S64'], batch['q_seg'], batch['z_L']
    f24, zf, geometries = batch['F24'], batch['z_F24'], batch['clip_geometries']
    evidence_l = model.language_source(s64.detach(), qseg.detach(), zl.detach()) / model.temperature_l
    evidence_f = model.forensic_source(f24, zf) / model.temperature_f
    opinion_l, opinion_f = evidence_to_dirichlet(evidence_l), evidence_to_dirichlet(evidence_f)
    mass_l = opinion_l['masses']
    mass_f, support = model._map_forensic(opinion_f['masses'], geometries, output_hw=(64,64), vacuous=True)
    aligned_f, mapped_support = model._map_forensic(f24, geometries, output_hw=(64,64), vacuous=False)
    aligned_zf, _ = model._map_forensic(zf, geometries, output_hw=(64,64), vacuous=False)
    if not torch.equal(support, mapped_support):
        raise RuntimeError('support drift')
    p_l = opinion_l['posterior']
    p_f = mass_f[:,:-1] + mass_f[:,-1:] / 2.0
    l64 = model.language_context(s64, qseg, zl)
    # ForensicContext64.forward detaches its inputs for the historical fit.
    # Execute its existing submodules in the same order without those detaches.
    ctx = model.forensic_context
    f64 = ctx.merge(torch.cat((ctx.f(aligned_f.float()), ctx.z(aligned_zf.float()), support.float()), dim=1)) * support.float()
    lr, fr, rectification = model.rectification(l64, f64, support)
    lr, fr, exchange = model.exchange(lr, fr, support)
    conflict = ecolaf_discount(torch.stack((mass_l,mass_f),dim=2),classes=2)[1]
    comparison, parts = build_comparison_features(lr,fr,p_l,p_f,conflict,support)
    hidden = model.utility_head.net[:-1](comparison)
    utility_logit = model.utility_head.net[-1](hidden)
    utility = utility_logit.sigmoid() * support.float()
    return {'utility_logit':utility_logit,'U':utility,'support':support,'p_L':p_l,'p_F':p_f,
            'mass_L':mass_l,'mass_F':mass_f,'L64':l64,'Fctx64':f64,'aligned_F64':aligned_f,
            'aligned_z_F64':aligned_zf,'Lr':lr,'Fr':fr,'comparison':comparison,
            'rectification':rectification,'exchange':exchange,'utility_hidden':hidden,'parts':parts}
