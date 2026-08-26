#!/usr/bin/env python3
"""Re-evaluate all trained Phase 4C-B checkpoints on the full fixed validation set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.evidence_reader import EvidenceReader
from scripts.phase4c_b_train import evaluate, frozen_hash, load_p1, pair_paths, q_index
from tools.phase4c_b import dump, file_sha256, load_source_model, tensor_hash


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("clip_reader", "forensic_reader"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def main():
    args = cli()
    cfg = yaml.safe_load((ROOT / "configs/phase4c_b_evidence_reader.yaml").read_text())
    out = ROOT / cfg["experiment"]["output_root"]
    cache = Path(cfg["experiment"]["cache_root"])
    ckroot = Path(cfg["experiment"]["checkpoint_root"]) / args.arm
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, core = load_p1(cfg, device)
    p1_sam_before = frozen_hash(core)
    source = load_source_model(cfg, args.arm, device)
    source._phase4c_b_arm = args.arm
    source_before = tensor_hash(source.state_dict().items())
    qcache = q_index(cache / "validation", "G0")
    pairs = pair_paths(ROOT / cfg["frozen_spatial_cache"]["phase3c1_root"], "val")
    history_path = out / "training" / args.arm / "history.json"
    history = json.loads(history_path.read_text())
    by_epoch = {int(row["epoch"]): row for row in history}
    for epoch in range(11):
        state = torch.load(ckroot / f"epoch_{epoch}.pt", map_location="cpu")
        reader = EvidenceReader().to(device)
        reader.load_state_dict(state["reader"])
        metrics, _, _, _, diag = evaluate(reader, source, core, qcache, pairs, device)
        if metrics["n"] != 1106:
            raise RuntimeError(f"full validation expected 1106, got {metrics['n']}")
        by_epoch[epoch]["validation"] = metrics
        by_epoch[epoch]["diagnostics"] = diag
        state["metrics"] = metrics
        state["diagnostics"] = diag
        torch.save(state, ckroot / f"epoch_{epoch}.pt")
        print(json.dumps({"arm": args.arm, "epoch": epoch, "validation": metrics}), flush=True)
    history = [by_epoch[e] for e in range(11)]
    dump(history_path, history)
    best = max(history, key=lambda r: (r["validation"]["mean_foreground_iou"],
                                       r["validation"]["mean_foreground_f1"]))
    selected = torch.load(ckroot / f"epoch_{best['epoch']}.pt", map_location="cpu")
    torch.save(selected, ckroot / "selected.pt")
    reader = EvidenceReader().to(device)
    reader.load_state_dict(selected["reader"])
    metrics, records, low, attention, diag = evaluate(
        reader, source, core, qcache, pairs, device, save_attention=True)
    evalroot = out / "evaluation" / "G0" / args.arm
    write_records(evalroot / "selected_predictions.jsonl", records)
    torch.save({"sample_ids": [r["sample_id"] for r in records],
                "low_res_logits": low, "attention": attention},
               evalroot / "selected_spatial.pt")
    selector = {
        "status": "COMPLETE", "arm": args.arm,
        "primary": cfg["selector"]["primary"], "tie_break": cfg["selector"]["tie_break"],
        "candidates": [{"epoch": r["epoch"], **r["validation"], "beta": r["beta"]}
                       for r in history],
        "selected_epoch": best["epoch"], "selected_metrics": metrics,
        "selected_beta": float(reader.beta.detach()),
        "selected_checkpoint": str((ckroot / "selected.pt").resolve()),
        "fallback_P1_allowed": True, "test_used": False, "official1000_used": False,
        "full_validation_re_evaluation": True,
    }
    dump(out / f"selector_{args.arm}.json", selector)
    dump(out / f"checkpoint_metrics_{args.arm}.json", {"history": history, "selected": selector})
    initial = selected["initial_hash"]
    epoch10 = torch.load(ckroot / "epoch_10.pt", map_location="cpu")
    update = {
        "status": "PASS", "arm": args.arm,
        "training_reader_changed": tensor_hash(epoch10["reader"].items()) != initial,
        "selected_reader_changed": tensor_hash(selected["reader"].items()) != initial,
        "reader_initial_hash": initial,
        "reader_selected_hash": tensor_hash(selected["reader"].items()),
        "P1_SAM_unchanged": frozen_hash(core) == p1_sam_before,
        "source_unchanged": tensor_hash(source.state_dict().items()) == source_before,
        "P1_checkpoint_hash_unchanged": file_sha256(Path(cfg["p1"]["checkpoint"])) == cfg["p1"]["checkpoint_sha256"],
        "source_checkpoint_hash_unchanged": file_sha256(Path(cfg["phase4c_a"]["clip_proj_checkpoint"] if args.arm == "clip_reader" else cfg["phase4c_a"]["forensic_adapter_checkpoint"])) == (cfg["phase4c_a"]["clip_proj_sha256"] if args.arm == "clip_reader" else cfg["phase4c_a"]["forensic_adapter_sha256"]),
    }
    dump(out / "training" / args.arm / "parameter_update.json", update)
    print(json.dumps({"status": "COMPLETE", "arm": args.arm,
                      "selected_epoch": best["epoch"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
