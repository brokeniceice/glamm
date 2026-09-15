#!/usr/bin/env python3
"""Resumable canonical-G0 C1 SEG-query cache for Phase 6E.2 (internal only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c2_p3 import dataset_for
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase4gf_formal_localization import load_dev
from tools.phase4c_b import file_sha256

C1 = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"
CFG_PATH = ROOT / "configs/phase6d3_c1_rine_conditioned_p1.yaml"
OUT = Path("/data/yz/groundingLMM_official/cache/phase6e2_c1_specific_r1")
AUDIT = ROOT / "outputs/phase6e2_c1_specific_r1/cache"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=100)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if file_sha256(C1) != C1_SHA:
        raise RuntimeError("C1 checkpoint SHA256 drift")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_model(cfg, C1, device, expected_step=2500, expected_epoch=5)
    model.eval().requires_grad_(False)
    if not hasattr(model, "rine_conditioner"):
        raise RuntimeError("C1 missing RINE conditioner")
    model.rine_conditioner.eval().requires_grad_(False)
    if model.training or model.rine_conditioner.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("C1/RINE freeze audit failed")
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]))
    summary = {"schema": "phase6e2_c1_g0_cache_v1", "status": "RUNNING", "c1_sha256": C1_SHA,
               "model_meta": meta, "splits": {}, "firewall": {"internal_test": False, "official1000": False}}
    dump(AUDIT / "status.json", summary)
    for split in ("train", "val"):
        ds = dataset_for(tokenizer, cfg, split)
        fake_indices = [i for i, row in enumerate(ds.rows) if int(row["class_label"]) == 1]
        expected_ids = train_ids() if split == "train" else load_dev("g0")["sample_ids"]
        actual_ids = [str(ds.rows[i]["sample_id"]) for i in fake_indices]
        if actual_ids != expected_ids:
            raise RuntimeError(f"{split} ordered Fake IDs differ from Phase4H-D population")
        root = OUT / split
        root.mkdir(parents=True, exist_ok=True)
        completed = sorted(root.glob("shard_*.pt"))
        done = 0
        for path in completed:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["sample_ids"] != expected_ids[done:done + len(payload["sample_ids"])]:
                raise RuntimeError(f"{split} resume-prefix drift at {path}")
            done += len(payload["sample_ids"])
        began = time.time()
        buffer = []
        for begin in range(done, len(fake_indices), args.batch_size):
            samples = [ds[i] for i in fake_indices[begin:begin + args.batch_size]]
            outputs = backend.generate_localization_batch(samples, provide_gt_fake=False,
                                                          generation_mode="unified_fake_generate")
            for sample, output in zip(samples, outputs):
                q = output.get("projected_seg_embeddings")
                q = torch.empty((0, 256), dtype=torch.bfloat16) if q is None else q.detach().cpu().to(torch.bfloat16)
                buffer.append({"sample_id": str(sample["sample_id"]), "q": q,
                               "seg_count": int(q.shape[0]), "seg_triggered": bool(output["seg_triggered"]),
                               "generated_token_ids": output["generated_token_ids"],
                               "prompt_sha256": output["prompt_sha256"]})
                if len(buffer) >= args.shard_size or begin + len(samples) >= len(fake_indices):
                    ordinal = len(sorted(root.glob("shard_*.pt")))
                    ids = [x["sample_id"] for x in buffer]
                    counts = torch.tensor([x["seg_count"] for x in buffer], dtype=torch.long)
                    valid = counts.eq(1)
                    qone = torch.full((len(buffer), 256), float("nan"), dtype=torch.bfloat16)
                    for j, row in enumerate(buffer):
                        if row["seg_count"] == 1:
                            qone[j] = row["q"][0]
                    payload = {"schema": "phase6e2_c1_g0_cache_shard_v1", "split": split,
                               "sample_ids": ids, "q_seg": qone, "valid": valid, "seg_count": counts,
                               "records": [{k: v for k, v in x.items() if k != "q"} for x in buffer],
                               "c1_sha256": C1_SHA}
                    path = root / f"shard_{ordinal:06d}.pt"
                    tmp = path.with_suffix(".pt.tmp")
                    torch.save(payload, tmp)
                    os.replace(tmp, path)
                    done += len(buffer)
                    buffer = []
                    print(json.dumps({"stage": "C1_G0_CACHE", "split": split, "done": done,
                                      "total": len(fake_indices)}), flush=True)
        shards = sorted(root.glob("shard_*.pt"))
        ids, counts = [], []
        for path in shards:
            value = torch.load(path, map_location="cpu", weights_only=False)
            ids += value["sample_ids"]
            counts += value["seg_count"].tolist()
        if ids != expected_ids:
            raise RuntimeError(f"{split} completed ID order drift")
        split_summary = {"n": len(ids), "valid_exactly_one_seg": sum(x == 1 for x in counts),
                         "invalid_no_seg": sum(x == 0 for x in counts),
                         "invalid_multiple_seg": sum(x > 1 for x in counts),
                         "seconds_this_run": time.time() - began,
                         "shards": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in shards]}
        summary["splits"][split] = split_summary
        dump(AUDIT / f"{split}.json", split_summary)
    summary["status"] = "COMPLETE"
    dump(AUDIT / "status.json", summary)


if __name__ == "__main__":
    main()
