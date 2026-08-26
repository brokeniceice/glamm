#!/usr/bin/env python3
"""Export the full validation P1-equivalent epoch-0 spatial logits for audit panels."""
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.evidence_reader import EvidenceReader
from scripts.phase4c_b_train import evaluate, load_p1, pair_paths, q_index
from tools.phase4c_b import load_source_model


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase4c_b_evidence_reader.yaml").read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    cache = Path(cfg["experiment"]["cache_root"])
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    _, core = load_p1(cfg, device)
    source = load_source_model(cfg, "clip_reader", device)
    source._phase4c_b_arm = "clip_reader"
    state = torch.load(Path(cfg["experiment"]["checkpoint_root"]) /
                       "clip_reader/epoch_0.pt", map_location="cpu")
    reader = EvidenceReader().to(device); reader.load_state_dict(state["reader"])
    qcache = q_index(cache / "validation", "G0")
    pairs = pair_paths(ROOT / cfg["frozen_spatial_cache"]["phase3c1_root"], "val")
    metrics, records, low, _, _ = evaluate(reader, source, core, qcache, pairs, device)
    if len(low) != 1106 or metrics["n"] != 1106:
        raise RuntimeError("P1 spatial export is not full validation")
    path = out / "evaluation/G0/P1/epoch0_spatial.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_ids": [r["sample_id"] for r in records],
                "low_res_logits": low}, path)
    print({"status": "COMPLETE", "n": len(low), "metrics": metrics})


if __name__ == "__main__":
    main()
