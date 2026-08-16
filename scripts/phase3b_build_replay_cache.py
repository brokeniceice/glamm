#!/usr/bin/env python3
"""Freeze canonical P1 G0 token sequences for every internal-train Fake sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.phase3b_replay import (
    canonical_json_sha256, file_sha256, phrase_overlap, replay_eligibility,
)
from tools.utils import IMAGE_TOKEN_INDEX


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3b_b1_generated_replay.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-canonical", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cache_root() -> Path:
    return ROOT / "outputs/phase3b_generated_replay/cache"


def merge_shards(cli) -> None:
    root = cache_root()
    paths = [root / f"shard_{i:02d}_of_{cli.num_shards:02d}.jsonl" for i in range(cli.num_shards)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cache shards: {missing}")
    records = [row for path in paths for row in read_jsonl(path)]
    records.sort(key=lambda row: row["train_fake_index"])
    if len({r["sample_id"] for r in records}) != len(records):
        raise RuntimeError("Duplicate sample IDs across replay shards")
    destination = root / "train_fake_p1_g0.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    eligible = [r for r in records if r["replay_eligible"]]
    stats = {
        "status": "FROZEN",
        "cache_path": str(destination.resolve()),
        "cache_sha256": file_sha256(destination),
        "num_train_fake": len(records),
        "num_replay_eligible": len(eligible),
        "replay_eligible_rate": len(eligible) / len(records) if records else None,
        "seg_count_histogram": {
            str(k): sum(r["seg_count"] == k for r in records)
            for k in sorted({r["seg_count"] for r in records})
        },
        "phrase_target_presence_rate": sum(r["generated_phrase"] is not None for r in records) / len(records),
        "phrase_normalized_exact_rate": sum(r["phrase_metrics"]["normalized_exact_match"] for r in records) / len(records),
        "phrase_mean_token_f1": sum(r["phrase_metrics"]["normalized_token_f1"] for r in records) / len(records),
        "generation_length": {
            "mean": sum(r["generated_length"] for r in records) / len(records),
            "min": min(r["generated_length"] for r in records),
            "max": max(r["generated_length"] for r in records),
        },
        "source_checkpoint": records[0]["source_checkpoint"] if records else None,
        "source_checkpoint_sha256": records[0]["source_checkpoint_sha256"] if records else None,
        "source_config_sha256": records[0]["source_config_sha256"] if records else None,
        "prompt_sha256": records[0]["prompt_sha256"] if records else None,
        "generation_batch_size": cli.batch_size,
        "canonical_historical_batch_size_identity": cli.batch_size == 1,
        "no_quality_filtering": True,
    }
    write_json(root / "cache_stats.json", stats)
    write_json(root / "cache_manifest.json", {
        **stats, "num_shards": cli.num_shards,
        "shards": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in paths],
    })
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main(argv=None):
    cli = args(argv)
    if cli.merge:
        merge_shards(cli); return
    if not 0 <= cli.shard_index < cli.num_shards:
        raise ValueError("invalid shard index")
    config_path = (ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(config["source"]["checkpoint"]).resolve()
    actual_hash = file_sha256(source)
    if actual_hash != config["source"]["checkpoint_sha256"]:
        raise RuntimeError(f"P1 checkpoint hash mismatch: {actual_hash}")
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    model, tokenizer, state = load_model(
        config, source, device,
        expected_step=int(config["source"]["optimizer_step"]),
        expected_epoch=int(config["source"]["epoch"]),
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / config["data"]["manifest_dir"], tokenizer, config["model"]["vision_tower"],
        split="train", datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"],
        image_size=int(config["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    fake_indices = [i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 1]
    indexed = [(fake_pos, index) for fake_pos, index in enumerate(fake_indices)
               if fake_pos % cli.num_shards == cli.shard_index]
    if cli.max_samples is not None:
        indexed = indexed[:cli.max_samples]
    root = cache_root(); root.mkdir(parents=True, exist_ok=True)
    output = root / f"shard_{cli.shard_index:02d}_of_{cli.num_shards:02d}.jsonl"
    if cli.reset and output.exists():
        output.unlink()
    done = {r["sample_id"] for r in read_jsonl(output)}
    pending = [(pos, idx) for pos, idx in indexed if dataset.rows[idx]["sample_id"] not in done]
    config_hash = file_sha256(config_path)
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), cli.batch_size):
            entries = pending[start:start + cli.batch_size]
            samples = [dataset[index] for _, index in entries]
            batch = backend._batch_many(samples, "", question=UNIFIED_FORENSICS_QUESTION)
            input_ids = batch["input_ids"]
            with torch.no_grad():
                generated = model.generate(
                    images=batch["global_enc_images"], input_ids=input_ids, bboxes=batch["bboxes"],
                    max_new_tokens=max(2, int(config["evaluation"]["max_new_tokens"])),
                    num_beams=1, output_hidden_states=False, return_dict_in_generate=True,
                    output_scores=False, use_cache=True,
                ).sequences
            canonical_outputs = (
                backend.generate_localization_batch(
                    samples, provide_gt_fake=False, generation_mode="unified_fake_generate"
                ) if cli.verify_canonical else None
            )
            prompt_len = input_ids.shape[1]
            for row_index, ((fake_pos, dataset_index), sample) in enumerate(zip(entries, samples)):
                gen = generated[row_index, prompt_len:]
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    eos = gen.eq(eos_id).nonzero(as_tuple=False).flatten()
                    if eos.numel(): gen = gen[:int(eos[0]) + 1]
                generated_ids = [int(x) for x in gen.detach().cpu().tolist()]
                if canonical_outputs is not None and generated_ids != canonical_outputs[row_index]["generated_token_ids"]:
                    raise RuntimeError(f"Direct replay generation diverged from canonical G0 for {sample['sample_id']}")
                prompt_ids = [int(x) for x in input_ids[row_index].detach().cpu().tolist()]
                decoded_ids = gen[gen.ne(IMAGE_TOKEN_INDEX)]
                decoded = tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
                parsed = parse_phrase_aligned_generation(decoded)
                reference = sample["localization_field"]["normalized_training_phrase"]
                eligibility = replay_eligibility(generated_ids, model.seg_token_idx)
                record = {
                    "sample_id": sample["sample_id"], "train_fake_index": fake_pos,
                    "dataset_index": dataset_index, "image_path": sample["image_path"],
                    "source": sample["source"], "content_category": sample["content_category"],
                    "prompt_token_ids": prompt_ids, "generated_token_ids": generated_ids,
                    "full_input_token_ids": prompt_ids + generated_ids,
                    "decoded_text": decoded, "generated_verdict": parsed["verdict"],
                    "generated_phrase": parsed["target_region"], "phrase_parse": parsed,
                    "authoritative_phrase": reference,
                    "phrase_metrics": phrase_overlap(reference, parsed["target_region"]),
                    "prompt_length": len(prompt_ids), "generated_length": len(generated_ids),
                    "full_input_length": len(prompt_ids) + len(generated_ids),
                    **eligibility,
                    "source_checkpoint": str(source), "source_checkpoint_sha256": actual_hash,
                    "source_optimizer_step": int(state["optimizer_step"]),
                    "source_epoch": int(state["epoch"]), "source_config": str(config_path),
                    "source_config_sha256": config_hash,
                    "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 400, "use_cache": True},
                    "generation_config_sha256": canonical_json_sha256({"do_sample": False, "num_beams": 1, "max_new_tokens": 400, "use_cache": True}),
                    "prompt_template_id": sample["prompt_template_id"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "no_correction_or_quality_filtering": True,
                    "canonical_g0_exact_token_verified": bool(cli.verify_canonical),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            print(f"shard {cli.shard_index}: {min(start + cli.batch_size, len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
