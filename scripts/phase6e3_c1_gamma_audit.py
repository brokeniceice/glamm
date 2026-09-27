#!/usr/bin/env python3
"""Select Rectifier gamma from frozen C1-valid training geometry only."""
from __future__ import annotations

import json

import torch

from scripts import phase6e3_c1_native_staged as staged


def main():
    path = staged.OUT / "rectifier/c1_gamma_audit.json"
    if path.exists():
        record = json.loads(path.read_text())
        staged.require(record["status"] == "PASS", "existing C1 gamma audit failed")
        print(json.dumps(record), flush=True)
        return
    ids, _, train_store, _, cache, _ = staged.load_inputs()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ordered = staged.f4.deterministic_order(ids, 0)
    positions = {sid: i for i, sid in enumerate(ids)}
    selected_ids = [sid for sid in ordered if bool(cache["valid"][positions[sid]])][:8]
    s64, raw, _, sc, cc = staged.e2.phase4f_spatial_batch(train_store, selected_ids, device)
    source = staged.load_evidence_source(staged.CFG, "forensic_rect", device)
    with torch.no_grad():
        evidence = staged.evidence_feature(source, raw)
    base = s64.float().flatten(2).transpose(1, 2)
    candidates = []
    for gamma in staged.CFG["architecture"]["gamma_candidates"]:
        rectifier = staged.load_rectifier(staged.CFG, float(gamma), device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = rectifier(s64, evidence, sc, cc,
                               torch.ones(len(selected_ids), 576, dtype=torch.bool, device=device))
        residual = output["residual"].float().flatten(2).transpose(1, 2)
        ratio = residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-12)
        support = output["support"]
        median = float(ratio[support].median())
        changed = (output["image_embeddings"].to(torch.bfloat16) != s64.to(torch.bfloat16)).flatten(2).any(1)
        survival = float(changed[support].float().mean())
        passed = (float(staged.CFG["architecture"]["residual_ratio_range"][0]) <= median <=
                  float(staged.CFG["architecture"]["residual_ratio_range"][1]) and
                  survival >= float(staged.CFG["architecture"]["minimum_bf16_token_survival"]))
        candidates.append({"gamma": float(gamma), "median_ratio": median,
                           "bf16_survival": survival, "pass": passed})
    passing = [row for row in candidates if row["pass"]]
    staged.require(bool(passing), "no C1 gamma candidate passed")
    record = {"status": "PASS", "selected_gamma": passing[0]["gamma"],
              "sample_ids": selected_ids, "candidates": candidates,
              "uses_c1_validity": True, "uses_p1_query": False,
              "validation_or_test_used": False}
    staged.dump(path, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
