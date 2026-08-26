#!/usr/bin/env python3
"""Replay frozen reward-dev token IDs through the frozen P1 mask path only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import trace_full_forward_batch
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase3d0_generate import model_state_sha256, save_mask, spatial_metrics, trainable_state_sha256
from tools.phase3d0 import reward_candidates, reward_components

EXPECTED_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-config", default="configs/phase3a_p1.yaml")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
def token_hash(groups):
    digest = hashlib.sha256()
    for group in groups:
        digest.update(group["sample_id"].encode() + b"\0")
        for row in group["rollouts"]: digest.update(json.dumps(row["generated_token_ids"], separators=(",", ":")).encode() + b"\0")
    return digest.hexdigest()


def main():
    cli = args(); cfg = yaml.safe_load((ROOT / "configs/phase3d0r_reward_reformulation.yaml").read_text())
    model_cfg = yaml.safe_load((ROOT / cli.model_config).read_text()); source = (ROOT / cfg["experiment"]["source_root"]).resolve()
    if not 0 <= cli.shard_index < cli.num_shards: raise ValueError("invalid shard")
    out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    output_path = out / f"reward_dev/mask_replay_fake_A_shard{cli.shard_index:02d}_of_{cli.num_shards:02d}.jsonl"
    source_groups = rows(source / "rollouts/dev_A_text_shard00_of_01.jsonl")
    all_source_fake = [group for group in source_groups if int(group["rollouts"][0]["gt_class"]) == 1]
    source_fake = [group for index, group in enumerate(all_source_fake) if index % cli.num_shards == cli.shard_index]
    if len(source_groups) != 1024 or len(all_source_fake) != 512 or any(len(x["rollouts"]) != 8 for x in source_groups):
        raise RuntimeError("frozen reward-dev A population mismatch")
    done_rows = rows(output_path) if output_path.exists() else []; done = {x["sample_id"] for x in done_rows}
    manifest = json.loads((source / "manifests/reward_dev_manifest.json").read_text())
    manifest_by_id = {row["sample_id"]: row for row in manifest["records"]}
    selector = json.loads((ROOT / "outputs/phase3a_phrase_grounding/selection/p1_selector.json").read_text())
    checkpoint = Path(selector["selected_checkpoint"]).resolve(); checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != EXPECTED_SHA: raise RuntimeError(f"P1 checkpoint mismatch: {checkpoint_hash}")
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device); torch.cuda.set_device(device)
    model, tokenizer, _ = load_model(model_cfg, checkpoint, device, expected_step=3500, expected_epoch=7)
    model.eval(); model.requires_grad_(False)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    dataset = UnifiedForensicsDataset(
        ROOT / model_cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"], split="train",
        datasets_root=model_cfg["data"]["datasets_root"], synthscars_root=model_cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned")
    before = model_state_sha256(model); trainable_before = trainable_state_sha256(model); start = time.time(); masks = 0
    for ordinal, group in enumerate(source_fake, start=1):
        if group["sample_id"] in done: continue
        entry = manifest_by_id[group["sample_id"]]; sample = dataset[int(entry["dataset_index"])]
        if sample["sample_id"] != group["sample_id"]: raise RuntimeError("reward-dev dataset index mismatch")
        repeated = [sample] * 8; batch = backend._batch_many(repeated, "", question=UNIFIED_FORENSICS_QUESTION)
        token_rows = [torch.tensor(row["generated_token_ids"], dtype=torch.long, device=device) for row in group["rollouts"]]
        eligible = [index for index, row in enumerate(group["rollouts"]) if bool(row["usable_seg"])]
        traces = {}
        if eligible:
            replay_batch = backend._batch_many([sample] * len(eligible), "", question=UNIFIED_FORENSICS_QUESTION)
            full_rows = [torch.cat((batch["input_ids"][index], token_rows[index])) for index in eligible]
            full_ids = torch.nn.utils.rnn.pad_sequence(full_rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            originals = [tuple(replay_batch["label_list"][i].shape) for i in range(len(eligible))]
            with torch.no_grad(): trace_rows = trace_full_forward_batch(model, replay_batch, full_ids, original_sizes=originals)
            traces = dict(zip(eligible, trace_rows))
        gt = torch.as_tensor(sample["masks"]).bool(); authoritative = sample["localization_field"]["normalized_training_phrase"]
        enriched = []
        for index, original in enumerate(group["rollouts"]):
            row = dict(original); trace = traces.get(index); logits = None if trace is None else trace["postprocessed_mask_logits"]
            metrics = spatial_metrics(logits, gt); parsed = parse_phrase_aligned_generation(row["decoded_text"])
            components = reward_components(gt_label=1, parsed=parsed, generated_ids=row["generated_token_ids"],
                seg_token_id=model.seg_token_idx, authoritative_phrase=authoritative, foreground_iou=metrics["foreground_iou"])
            row.update(metrics); row.update(components); row.update(reward_candidates(1, components))
            saved = save_mask(out, "reward_dev", "A_FIXED_TOKEN_REPLAY", group["sample_id"], f"rollout_{index:02d}", logits)
            masks += int(saved["binary_mask_path"] is not None); row.update(saved); enriched.append(row)
        append(output_path, {"sample_id": group["sample_id"], "dataset_index": entry["dataset_index"],
               "population": "reward_dev", "setting": "A", "fixed_token_mask_replay": True,
               "source_text_rollout_path": str(source / "rollouts/dev_A_text_shard00_of_01.jsonl"), "rollouts": enriched})
        done.add(group["sample_id"])
        if ordinal % 10 == 0: print(f"phase3d0r fixed-token mask replay shard={cli.shard_index} {ordinal}/{len(source_fake)} elapsed={time.time()-start:.1f}s", flush=True)
    final = rows(output_path); after = model_state_sha256(model); trainable_after = trainable_state_sha256(model)
    if len(final) != len(source_fake) or len({x["sample_id"] for x in final}) != len(source_fake): raise RuntimeError("reward-dev mask replay incomplete")
    source_hash = token_hash(source_fake); output_hash = token_hash(final)
    audit = {"status": "COMPLETE", "authorization": "user explicitly authorized reward-dev mask calculation without rerollout",
      "shard_index": cli.shard_index, "num_shards": cli.num_shards,
      "source_groups": len(source_fake), "trajectories": len(source_fake)*8, "source_token_ids_sha256": source_hash,
      "replay_token_ids_sha256": output_hash, "token_ids_exact": source_hash == output_hash,
      "model_generate_called": False, "fixed_token_full_sequence_no_cache_replay": True,
      "mask_threshold": "> 0", "new_rollouts": False, "text_changed": False, "K_changed": False,
      "checkpoint_sha256": checkpoint_hash, "model_state_sha256_before": before, "model_state_sha256_after": after,
      "model_state_exact": before == after, "trainable_state_exact": trainable_before == trainable_after,
      "all_requires_grad_false": not any(p.requires_grad for p in model.parameters()), "optimizer_created": False,
      "backward_called": False, "phase3d1_started": False, "new_binary_masks_written": masks,
      "output": str(output_path), "elapsed_seconds": time.time()-start}
    if not all((audit["token_ids_exact"], audit["model_state_exact"], audit["trainable_state_exact"])): raise RuntimeError(audit)
    dump(out / f"audit/reward_dev_fixed_token_mask_replay_shard{cli.shard_index:02d}_of_{cli.num_shards:02d}.json", audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
