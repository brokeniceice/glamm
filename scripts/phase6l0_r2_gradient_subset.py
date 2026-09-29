#!/usr/bin/env python3
"""Real-sample two-backward R2/SAM gradient gate before full cache completes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.c2_r2_localizer import C2R2Localizer
from scripts.phase6l0_r2_preflight import OUT, c2_native_sam_state, load_native_runtime, dump, require
from scripts.phase6l0_r2_train import forward, grad_norm, gradient_modules
from tools.phase3c1 import geometry_for
from tools.phase4e1 import tensor_state_sha256


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    c2 = torch.load("/data/yz/groundingLMM_official/cache/phase6l0_r2/train/shard_000000.pt",
                    map_location="cpu", weights_only=False)
    sam_source = torch.load(ROOT / "outputs/phase3c1_spatial_probe/cache/sam/train/shard_000000_000032.pt",
                            map_location="cpu", weights_only=False)
    clip_source = torch.load(ROOT / "outputs/phase3c1_spatial_probe/cache/clip/train/shard_000000_000032.pt",
                             map_location="cpu", weights_only=False)
    index = int(c2["valid_c2_g0"].nonzero()[0])
    require(index < 32 and c2["sample_ids"][index] == sam_source["records"][index]["sample_id"] ==
            clip_source["records"][index]["sample_id"], "real gradient sample ID drift")
    state, _ = c2_native_sam_state()
    sam = load_native_runtime(state, device)
    frozen_before = tensor_state_sha256(sam.state_dict())
    torch.manual_seed(3407)
    model = C2R2Localizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    batch = {"sample_ids": [c2["sample_ids"][index]],
             "raw_clip_grid": clip_source["features"][index:index+1].to(device),
             "attention_map": c2["A"][index:index+1].to(device),
             "evidence_map": c2["E"][index:index+1].to(device),
             "r_prime": c2["r_prime"][index:index+1].to(device),
             "q_seg": c2["q_seg"][index:index+1].to(device),
             "s64": sam_source["features"][index:index+1].to(device),
             "z_l": c2["z_L"][index:index+1].to(device),
             "target": sam_source["targets"][index:index+1].to(device),
             "clip_geometry": [clip_source["records"][index]["geometry"]]}
    require(batch["clip_geometry"][0] == geometry_for("clip", batch["clip_geometry"][0]["original_hw"]),
            "real gradient geometry drift")
    first_output, first_low, first_loss = forward(model, sam, batch)
    require(torch.count_nonzero(first_output["deltaS"]) == 0 and
            torch.equal(first_output["S_adapt"], batch["s64"]) and
            torch.equal(first_low, batch["z_l"]), "real R2 step0/C2-G0 parity failed")
    first_loss["total"].backward()
    first_grad = {k: grad_norm(v) for k, v in gradient_modules(model).items()}
    require(first_grad["W_out"] > 0 and all(p.grad is None for p in sam.parameters()),
            "real first backward firewall failed")
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    second_output, _second_low, second_loss = forward(model, sam, batch)
    second_loss["total"].backward()
    second_grad = {k: grad_norm(v) for k, v in gradient_modules(model).items()}
    require(all(value > 0 and torch.isfinite(torch.tensor(value)) for value in second_grad.values()) and
            all(p.grad is None for p in sam.parameters()) and
            frozen_before == tensor_state_sha256(sam.state_dict()),
            f"real second backward or frozen SAM integrity failed: {second_grad}")
    dump(OUT / "real_subset_gradient_gate.json", {"status": "PASS",
         "sample_id": batch["sample_ids"][0], "first_loss": float(first_loss["total"]),
         "second_loss": float(second_loss["total"]), "first_grad_norms": first_grad,
         "second_grad_norms": second_grad, "sam_state_sha256": frozen_before,
         "C2_G0_step0_exact": True, "frozen_SAM_grad_none": True})
    print(json.dumps({"status": "PASS", "sample_id": batch["sample_ids"][0]}), flush=True)


if __name__ == "__main__": main()
