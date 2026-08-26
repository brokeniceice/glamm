#!/usr/bin/env python3
"""Resumable frozen-P1 greedy and stochastic rollout generation for Phase 3D.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import MASK_LOGIT_THRESHOLD, compute_binary_mask_metrics, compute_empty_prediction_metrics
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import trace_full_forward, trace_full_forward_batch
from eval.phase3a_metrics import parse_phrase_aligned_generation
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from tools.phase3d0 import parse_structure, reward_candidates, reward_components, stable_rank
from tools.utils import IMAGE_TOKEN_INDEX


EXPECTED_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"


def args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3d0_reward_preflight.yaml")
    parser.add_argument("--model-config", default="configs/phase3a_p1.yaml")
    parser.add_argument("--population", choices=("dev", "val"), required=True)
    parser.add_argument("--setting", choices=("A", "B"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--include-greedy", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-model-hash", action="store_true", help="Smoke-test only")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args(argv)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        flat = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, flat.numel(), 16 * 1024 * 1024):
            digest.update(flat[start:start + 16 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def trainable_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad:
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def stable_seed(base: int, population: str, setting: str, sample_id: str) -> int:
    return int(stable_rank(base, f"phase3d0:{population}:{setting}", sample_id)[:8], 16)


def trim_generated(row: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    if eos_id is not None:
        positions = row.eq(eos_id).nonzero(as_tuple=False).flatten()
        if positions.numel(): row = row[:int(positions[0]) + 1]
    return row


def selected_logprob(scores, row_index: int, token_ids: list[int]) -> tuple[float | None, float | None]:
    if not scores or not token_ids: return None, None
    values = []
    for step, token in enumerate(token_ids[:len(scores)]):
        values.append(float(torch.log_softmax(scores[step][row_index].float(), dim=-1)[token]))
    total = float(sum(values))
    return total, total / len(values) if values else None


def spatial_metrics(logits: torch.Tensor | None, gt: torch.Tensor | None) -> dict[str, float | int | None]:
    if gt is None:
        return {"foreground_iou": None, "foreground_f1": None, "background_iou": None,
                "fg_bg_miou": None, "tp": None, "fp": None, "fn": None, "tn": None}
    target = gt.bool().any(dim=0).cpu() if gt.ndim == 3 else gt.bool().cpu()
    if logits is not None and logits.numel():
        value = logits.detach().float().cpu()
        metric = compute_binary_mask_metrics(value, target)
        binary = value.amax(dim=0).gt(MASK_LOGIT_THRESHOLD)
    else:
        metric = compute_empty_prediction_metrics(target); binary = torch.zeros_like(target)
    tn = int((~binary & ~target).sum()); bg_den = tn + int(metric["fp"]) + int(metric["fn"])
    bg_iou = tn / bg_den if bg_den else 1.0
    return {
        "foreground_iou": float(metric["image_iou"]), "foreground_f1": float(metric["image_pixel_f1"]),
        "background_iou": float(bg_iou), "fg_bg_miou": float((metric["image_iou"] + bg_iou) / 2),
        "tp": int(metric["tp"]), "fp": int(metric["fp"]), "fn": int(metric["fn"]), "tn": tn,
    }


def tensor_paths(tensor_root: Path, population: str, setting: str, sample_id: str, name: str):
    stem = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
    base = tensor_root / "masks" / population / setting / stem
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.logits.pt", base / f"{name}.binary.pt"


def save_mask(tensor_root: Path, population: str, setting: str, sample_id: str, name: str,
              logits: torch.Tensor | None) -> dict[str, str | None]:
    if logits is None or not logits.numel():
        return {"mask_logits_path": None, "binary_mask_path": None}
    logits_path, binary_path = tensor_paths(tensor_root, population, setting, sample_id, name)
    torch.save(logits.detach().cpu().to(torch.bfloat16), logits_path)
    torch.save(logits.detach().cpu().gt(MASK_LOGIT_THRESHOLD), binary_path)
    return {"mask_logits_path": str(logits_path), "binary_mask_path": str(binary_path)}


def stochastic_group(model, tokenizer, backend, sample, *, K: int, setting: dict, seed: int,
                     text_only: bool, tensor_root: Path, population: str, setting_name: str):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    repeated = [sample] * K
    batch = backend._batch_many(repeated, "", question=UNIFIED_FORENSICS_QUESTION)
    with torch.no_grad():
        generated = model.generate(
            images=batch["global_enc_images"], input_ids=batch["input_ids"], bboxes=batch["bboxes"],
            max_new_tokens=backend.max_new_tokens, num_beams=1, do_sample=True,
            temperature=float(setting["temperature"]), top_p=float(setting["top_p"]),
            use_cache=True, return_dict_in_generate=True, output_scores=True,
        )
    prompt_len = batch["input_ids"].shape[1]; eos_id = tokenizer.eos_token_id
    token_rows = [trim_generated(generated.sequences[i, prompt_len:], eos_id) for i in range(K)]
    records = []
    for rollout_index, token_row in enumerate(token_rows):
        ids = [int(value) for value in token_row.detach().cpu().tolist()]
        decoded_ids = token_row[token_row.ne(IMAGE_TOKEN_INDEX)]
        text = tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
        parsed = parse_phrase_aligned_generation(text)
        structure = parse_structure(parsed, ids, model.seg_token_idx)
        logprob, normalized_logprob = selected_logprob(generated.scores, rollout_index, ids)
        records.append({
            "sample_id": sample["sample_id"], "rollout_index": rollout_index,
            "seed": seed, "gt_class": int(sample["cls_label"]),
            "generated_token_ids": ids, "decoded_text": text,
            "verdict": parsed["verdict"], "generated_explanation": parsed["explanation"],
            "target_regions_raw": parsed["target_region"], "parse_status": parsed["parse_status"],
            "generation_length": len(ids), "sequence_log_probability": logprob,
            "length_normalized_log_probability": normalized_logprob,
            "duplicate_or_repetition_flag": "REPEATED_TARGET_FIELD" in parsed["parse_status"],
            **structure,
        })
    if text_only:
        return records
    eligible = [index for index, row in enumerate(records) if row["usable_seg"]]
    traces = {}
    if eligible:
        replay_samples = [sample] * len(eligible)
        replay_batch = backend._batch_many(replay_samples, "", question=UNIFIED_FORENSICS_QUESTION)
        full_rows = [torch.cat((batch["input_ids"][index], token_rows[index])) for index in eligible]
        full_ids = torch.nn.utils.rnn.pad_sequence(
            full_rows, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(backend.device)
        originals = [tuple(replay_batch["label_list"][i].shape) for i in range(len(eligible))]
        with torch.no_grad():
            trace_rows = trace_full_forward_batch(model, replay_batch, full_ids, original_sizes=originals)
        traces = {index: trace for index, trace in zip(eligible, trace_rows)}
    gt = torch.as_tensor(sample["masks"]).bool() if bool(sample["seg_valid"]) else None
    authoritative = sample["localization_field"]["normalized_training_phrase"] if gt is not None else None
    for index, row in enumerate(records):
        trace = traces.get(index); logits = None if trace is None else trace["postprocessed_mask_logits"]
        metrics = spatial_metrics(logits, gt)
        components = reward_components(
            gt_label=int(sample["cls_label"]), parsed=parse_phrase_aligned_generation(row["decoded_text"]),
            generated_ids=row["generated_token_ids"], seg_token_id=model.seg_token_idx,
            authoritative_phrase=authoritative, foreground_iou=metrics["foreground_iou"],
        )
        row.update(metrics); row.update(components); row.update(reward_candidates(int(sample["cls_label"]), components))
        row.update(save_mask(tensor_root, population, setting_name, sample["sample_id"], f"rollout_{index:02d}", logits))
    return records


def greedy_record(model, tokenizer, backend, sample, tensor_root: Path, population: str):
    output = backend.generate_localization(sample, provide_gt_fake=False, generation_mode="unified_fake_generate")
    ids = [int(value) for value in output["generated_token_ids"]]
    parsed = parse_phrase_aligned_generation(output["generated_text"])
    structure = parse_structure(parsed, ids, model.seg_token_idx)
    trace = None
    if structure["usable_seg"]:
        batch = backend._batch(sample, "", question=UNIFIED_FORENSICS_QUESTION)
        full_ids = torch.cat((batch["input_ids"], torch.tensor([ids], device=backend.device)), dim=1)
        trace = trace_full_forward(model, batch, full_ids, original_size=tuple(batch["label_list"][0].shape))
    logits = None if trace is None else trace["postprocessed_mask_logits"]
    canonical = output.get("pred_mask")
    canonical_exact = (
        logits is None and canonical is None
        or logits is not None and canonical is not None
        and torch.equal(logits.detach().cpu().gt(0), torch.as_tensor(canonical).detach().cpu().gt(0))
    )
    gt = torch.as_tensor(sample["masks"]).bool() if bool(sample["seg_valid"]) else None
    metrics = spatial_metrics(logits, gt)
    authoritative = sample["localization_field"]["normalized_training_phrase"] if gt is not None else None
    components = reward_components(
        gt_label=int(sample["cls_label"]), parsed=parsed, generated_ids=ids,
        seg_token_id=model.seg_token_idx, authoritative_phrase=authoritative,
        foreground_iou=metrics["foreground_iou"],
    )
    return {
        "sample_id": sample["sample_id"], "gt_class": int(sample["cls_label"]),
        "generated_token_ids": ids, "decoded_text": output["generated_text"],
        "verdict": parsed["verdict"], "generated_explanation": parsed["explanation"],
        "target_regions_raw": parsed["target_region"], "parse_status": parsed["parse_status"],
        "generation_length": len(ids), "cls_pred": int(output["cls_pred"]),
        "cls_prob_fake": float(output["cls_prob_fake"]),
        "lm_verdict_pred": int(output["lm_verdict_pred"]),
        "cls_lm_agree": bool(output["cls_lm_agree"]),
        "canonical_trace_binary_exact": bool(canonical_exact),
        **structure, **metrics, **components, **reward_candidates(int(sample["cls_label"]), components),
        **save_mask(tensor_root, population, "greedy", sample["sample_id"], "greedy", logits),
    }


def main(argv=None):
    cli = args(argv)
    if not 0 <= cli.shard_index < cli.num_shards: raise ValueError("invalid shard index")
    cfg_path = (ROOT / cli.config).resolve(); cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg_path = (ROOT / cli.model_config).resolve(); model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["experiment"]["output_root"]).resolve(); tensor_root = Path(cfg["experiment"]["tensor_root"]).resolve()
    manifest_name = "reward_dev_manifest.json" if cli.population == "dev" else "reward_val_manifest.json"
    frozen_manifest = json.loads((output / "manifests" / manifest_name).read_text(encoding="utf-8"))
    records = [row for ordinal, row in enumerate(frozen_manifest["records"]) if ordinal % cli.num_shards == cli.shard_index]
    if cli.max_samples is not None: records = records[:cli.max_samples]
    suffix = "text" if cli.text_only else "full"
    rollout_path = output / "rollouts" / f"{cli.population}_{cli.setting}_{suffix}_shard{cli.shard_index:02d}_of_{cli.num_shards:02d}.jsonl"
    greedy_path = output / "greedy" / f"{cli.population}_shard{cli.shard_index:02d}_of_{cli.num_shards:02d}.jsonl"
    audit_path = output / "audit" / f"generation_{cli.population}_{cli.setting}_{suffix}_shard{cli.shard_index:02d}.json"
    if cli.reset:
        for path in (rollout_path, greedy_path):
            if path.exists(): path.unlink()
    done = {row["sample_id"] for row in read_jsonl(rollout_path)}
    greedy_done = {row["sample_id"] for row in read_jsonl(greedy_path)}

    selector = json.loads((ROOT / cfg["source"]["selector"]).read_text(encoding="utf-8"))
    checkpoint = Path(selector["selected_checkpoint"]).resolve(); checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != EXPECTED_SHA or checkpoint_hash != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError(f"P1 checkpoint hash mismatch: {checkpoint_hash}")
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device); torch.cuda.set_device(device)
    model, tokenizer, state = load_model(model_cfg, checkpoint, device, expected_step=3500, expected_epoch=7)
    model.eval(); model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen model contract violated")
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )
    split = "train" if cli.population == "dev" else "val"
    dataset = UnifiedForensicsDataset(
        ROOT / model_cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"],
        split=split, datasets_root=model_cfg["data"]["datasets_root"],
        synthscars_root=model_cfg["data"]["synthscars_root"], image_size=int(model_cfg["model"]["image_size"]),
        target_protocol="phrase_aligned",
    )
    before = None if cli.skip_model_hash else model_state_sha256(model)
    trainable_before = trainable_state_sha256(model)
    setting = cfg["sampling"]["settings"][cli.setting]; start_time = time.time()
    processed = 0
    for ordinal, manifest_row in enumerate(records, start=1):
        sample_id = manifest_row["sample_id"]
        dataset_index = int(manifest_row["dataset_index"]); sample = dataset[dataset_index]
        if sample["sample_id"] != sample_id: raise RuntimeError(f"manifest index mismatch: {sample_id}")
        if cli.include_greedy and sample_id not in greedy_done:
            append(greedy_path, greedy_record(model, tokenizer, backend, sample, tensor_root, cli.population))
            greedy_done.add(sample_id)
        if sample_id not in done:
            group = stochastic_group(
                model, tokenizer, backend, sample, K=int(cfg["sampling"]["K"]), setting=setting,
                seed=stable_seed(int(cfg["experiment"]["seed"]), cli.population, cli.setting, sample_id),
                text_only=cli.text_only, tensor_root=tensor_root, population=cli.population,
                setting_name=cli.setting,
            )
            if len(group) != 8: raise RuntimeError(f"K mismatch for {sample_id}")
            append(rollout_path, {"sample_id": sample_id, "dataset_index": dataset_index,
                                  "population": cli.population, "setting": cli.setting,
                                  "text_only": cli.text_only, "rollouts": group})
            done.add(sample_id)
        processed += 1
        if processed % 10 == 0:
            elapsed = time.time() - start_time
            print(f"phase3d0 {cli.population}/{cli.setting}/{suffix} shard={cli.shard_index} "
                  f"progress={processed}/{len(records)} mean_s={elapsed/processed:.2f}", flush=True)
    after = None if cli.skip_model_hash else model_state_sha256(model)
    trainable_after = trainable_state_sha256(model)
    audit = {
        "status": "COMPLETE", "population": cli.population, "setting": cli.setting,
        "text_only": cli.text_only, "include_greedy": cli.include_greedy,
        "shard_index": cli.shard_index, "num_shards": cli.num_shards,
        "requested_groups": len(records), "completed_groups": len(done), "K": 8,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "phase3d0_config": str(cfg_path), "phase3d0_config_sha256": file_sha256(cfg_path),
        "sampling_setting": setting,
        "model_state_sha256_before": before, "model_state_sha256_after": after,
        "model_state_exact_identical": None if before is None else before == after,
        "trainable_state_sha256_before": trainable_before,
        "trainable_state_sha256_after": trainable_after,
        "trainable_state_exact_identical": trainable_before == trainable_after,
        "all_requires_grad_false": not any(p.requires_grad for p in model.parameters()),
        "model_eval": not model.training, "torch_no_grad_for_inference": True,
        "optimizer_created": False, "scheduler_created": False, "backward_called": False,
        "phase3d1_started": False, "elapsed_seconds": time.time() - start_time,
        "rollout_path": str(rollout_path), "greedy_path": str(greedy_path) if cli.include_greedy else None,
    }
    if not cli.skip_model_hash and not audit["model_state_exact_identical"]:
        raise RuntimeError("model state changed during Phase 3D.0")
    dump(audit_path, audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
