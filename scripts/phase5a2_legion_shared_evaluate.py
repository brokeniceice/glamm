#!/usr/bin/env python3
"""External, resumable official-LEGION L-FREE evaluation for Phase 5A-2.

The official checkout is not edited.  This runner reuses its ``inference``
function verbatim, supplies its fixed public prompt, and adds only manifest
selection, deterministic pre-model corruptions, output persistence, and the
already-frozen common GT/scoring bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PROMPT_SHA256 = "5465a2b6265db804beb9e9e339ba746d75127d673a1467d04572cb63e9fa5804"
CLIP_REPO_ID = "openai/clip-vit-large-patch14-336"
SEED = 3407
HISTORICAL_CORRUPTION_MODULE = None
PROJECT_SYNTHSCARS_MODULE = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def redirect_clip_class(cls, clip_dir: Path) -> None:
    original = cls.from_pretrained

    def redirected(_cls, pretrained_model_name_or_path, *args, **kwargs):
        requested = os.fspath(pretrained_model_name_or_path)
        resolved = os.fspath(clip_dir) if requested == CLIP_REPO_ID else requested
        return original(resolved, *args, **kwargs)

    cls.from_pretrained = classmethod(redirected)


def load_official_inference(repo: Path):
    infer_path = repo / "scripts" / "loc_exp" / "infer.py"
    spec = importlib.util.spec_from_file_location("phase5a2_official_infer", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {infer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_mask(row: dict, height: int, width: int, target_kind: str) -> np.ndarray:
    if target_kind == "loki_bbox_union":
        path = Path(row["mask_path"])
        with Image.open(path) as image:
            result = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if result.shape != (height, width):
            raise RuntimeError(f"LOKI target geometry mismatch: {row.get('sample_id')}")
        if not result.any():
            raise RuntimeError(f"empty LOKI bbox-union target: {row.get('sample_id')}")
        return result
    if target_kind not in ("synthscars_union", "internal_val_union"):
        raise RuntimeError(f"unsupported target kind: {target_kind}")
    global PROJECT_SYNTHSCARS_MODULE
    if PROJECT_SYNTHSCARS_MODULE is None:
        source = ROOT / "dataset" / "forensics" / "synthscars.py"
        spec = importlib.util.spec_from_file_location("phase5a2_project_synthscars", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load frozen SynthScars target implementation: {source}")
        PROJECT_SYNTHSCARS_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PROJECT_SYNTHSCARS_MODULE)

    result = np.zeros((height, width), dtype=bool)
    for ref in row.get("refs") or []:
        result |= PROJECT_SYNTHSCARS_MODULE.polygon_to_mask(
            PROJECT_SYNTHSCARS_MODULE.polygons_for_target(ref, height, width), height, width
        ).astype(bool)
    if not result.any():
        raise RuntimeError(f"empty GT target: {row.get('sample_id')}")
    return result


def historical_corruption_hashes(condition: str) -> dict[str, str]:
    if condition == "original":
        return {}
    path = ROOT / "outputs" / "phase3c2_p3_robustness" / "robustness" / condition / "p1" / "G0.jsonl"
    result = {}
    for row in rows(path):
        value = row.get("corrupted_rgb_sha256")
        if not value:
            raise RuntimeError(f"historical corruption hash missing: {path} {row.get('sample_id')}")
        result[str(row["sample_id"])] = str(value)
    if len(result) != 1000:
        raise RuntimeError(f"historical corruption population drift: {condition} {len(result)}")
    return result


def corrupt_rgb(image: np.ndarray, condition: str, sample_id: str) -> np.ndarray:
    global HISTORICAL_CORRUPTION_MODULE
    if HISTORICAL_CORRUPTION_MODULE is None:
        source = ROOT / "tools" / "phase3c2.py"
        spec = importlib.util.spec_from_file_location("phase5a2_historical_corruption", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load historical corruption implementation: {source}")
        HISTORICAL_CORRUPTION_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(HISTORICAL_CORRUPTION_MODULE)
    return HISTORICAL_CORRUPTION_MODULE.corrupt_rgb(image, condition, seed=SEED, sample_id=sample_id)


def metric(binary: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred, truth = binary.astype(bool), target.astype(bool)
    if pred.shape != truth.shape:
        raise RuntimeError(f"metric shape mismatch: {pred.shape} vs {truth.shape}")
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    union = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "foreground_iou": float(tp / union if union else 1.0),
        "foreground_f1": float(2 * tp / f1_denominator if f1_denominator else 1.0),
    }


def load_model(repo: Path, model_dir: Path, clip_dir: Path, device: torch.device):
    repo_string = os.fspath(repo.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    for cls in (transformers.CLIPVisionConfig, transformers.CLIPVisionModel, transformers.CLIPImageProcessor):
        redirect_clip_class(cls, clip_dir.resolve())
    official = load_official_inference(repo)
    from dataset.utils.utils import GCG_QUESTIONS
    from model.Legion import LegionForCausalLM

    prompt = GCG_QUESTIONS[0]
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != OFFICIAL_PROMPT_SHA256:
        raise RuntimeError("official L-FREE prompt drift")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, cache_dir=None, model_max_length=512, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = LegionForCausalLM.from_pretrained(
        model_dir, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, torch_dtype=torch.bfloat16
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch.bfloat16, device=device)
    model = model.bfloat16().to(device).eval()
    clip_image_processor = transformers.CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = official.ResizeLongestSide(1024)
    return official, model, tokenizer, clip_image_processor, transform, prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legion-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=("original", "jpeg70", "jpeg80", "gaussian5", "gaussian10"), required=True)
    parser.add_argument(
        "--target-kind",
        choices=("synthscars_union", "loki_bbox_union", "internal_val_union"),
        default="synthscars_union",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not all(path.exists() for path in (args.legion_repo, args.model_dir, args.clip_dir, args.manifest)):
        parser.error("one or more required paths do not exist")

    manifest_rows = rows(args.manifest)
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    expected_n = {
        "synthscars_union": 1000,
        "loki_bbox_union": 229,
        "internal_val_union": 1106,
    }[args.target_kind]
    if len(sample_ids) != expected_n or len(set(sample_ids)) != expected_n:
        raise RuntimeError("shared manifest population drift")
    if args.target_kind == "loki_bbox_union" and args.condition != "original":
        raise RuntimeError("LOKI supplement is frozen to original RGB only")
    if args.target_kind == "internal_val_union" and args.condition != "original":
        raise RuntimeError("internal-validation diagnostic is frozen to original RGB only")
    if not 0 <= args.start < args.end <= len(manifest_rows):
        raise RuntimeError(f"invalid shard [{args.start}, {args.end})")
    expected_rgb_hashes = historical_corruption_hashes(args.condition)
    condition_root = args.output_root / args.condition
    records_path = condition_root / "shards" / f"{args.start:04d}_{args.end:04d}.predictions.jsonl"
    completed = {str(row["sample_id"]) for row in rows(records_path)} if records_path.exists() else set()
    if any(sid not in set(sample_ids[args.start:args.end]) for sid in completed):
        raise RuntimeError("resume file includes sample outside its shard")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 5A-2 official L-FREE requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    worker_path = condition_root / "shards" / f"{args.start:04d}_{args.end:04d}.worker.json"
    worker = {
        "schema": "phase5a2_legion_lfree_worker_v1", "status": "RUNNING", "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "condition": args.condition, "target_kind": args.target_kind,
        "start": args.start, "end": args.end, "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest),
        "ordered_id_sha256": ordered_id_sha256(sample_ids), "completed_before_resume": len(completed),
        "model_dir": str(args.model_dir.resolve()),
        "legion_source_commit": os.popen(f"git -C {args.legion_repo.resolve()} rev-parse HEAD").read().strip(),
    }
    atomic_json(worker_path, worker)
    started = time.monotonic()
    try:
        official, model, tokenizer, clip_processor, transform, prompt = load_model(
            args.legion_repo.resolve(), args.model_dir.resolve(), args.clip_dir.resolve(), device
        )
        inference_args = argparse.Namespace(conv_type="llava_v1", use_mm_start_end=True, image_size=1024)
        original_evaluate = type(model).evaluate
        observed = []

        def observe_evaluate(instance, *evaluate_args, **evaluate_kwargs):
            generated_ids, pred_masks = original_evaluate(instance, *evaluate_args, **evaluate_kwargs)
            observed.append({
                "seg_count": int((generated_ids == instance.seg_token_idx).sum().item()),
                "generated_shape": list(generated_ids.shape),
                "pred_mask_shapes": [list(mask.shape) for mask in pred_masks],
            })
            return generated_ids, pred_masks

        type(model).evaluate = observe_evaluate
        try:
            for index in range(args.start, args.end):
                row = manifest_rows[index]
                sample_id = str(row["sample_id"])
                if sample_id in completed:
                    continue
                source_bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if source_bgr is None:
                    raise OSError(row["image_path"])
                source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
                changed_rgb = corrupt_rgb(source_rgb, args.condition, sample_id)
                rgb_sha256 = hashlib.sha256(changed_rgb.tobytes()).hexdigest()
                expected = expected_rgb_hashes.get(sample_id)
                if expected is not None and expected != rgb_sha256:
                    raise RuntimeError(f"corruption RGB hash drift for {sample_id}")
                changed_bgr = cv2.cvtColor(changed_rgb, cv2.COLOR_RGB2BGR)
                height, width = changed_rgb.shape[:2]
                target = target_mask(row, height, width, args.target_kind)
                observed.clear()
                explanation, pred_masks, phrases = official.inference(
                    model, prompt, changed_bgr, tokenizer, clip_processor, transform, inference_args
                )
                if len(observed) != 1:
                    raise RuntimeError(f"evaluate observation mismatch for {sample_id}: {len(observed)}")
                observation = observed[0]
                status = "OK"
                masks = pred_masks[0].detach().float().cpu().numpy() if pred_masks and pred_masks[0].numel() else np.empty((0, height, width), dtype=np.float32)
                if masks.ndim != 3 or (masks.shape[0] and tuple(masks.shape[-2:]) != (height, width)):
                    status = "INVALID_SHAPE"
                    binary = np.zeros((height, width), dtype=bool)
                elif masks.shape[0] == 0:
                    status = "EMPTY_MASK" if observation["seg_count"] else "NO_SEG"
                    binary = np.zeros((height, width), dtype=bool)
                else:
                    binary = np.any(masks > 0.0, axis=0)
                    if not observation["seg_count"]:
                        status = "MASK_WITHOUT_SEG"
                values = metric(binary, target)
                stem = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
                mask_path = condition_root / "masks" / f"{stem}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((binary.astype(np.uint8) * 255), mode="L").save(mask_path)
                record = {
                    "sample_id": sample_id, "ordinal": index, "condition": args.condition,
                    "target_kind": args.target_kind,
                    "image_path": str(row["image_path"]), "corrupted_rgb_sha256": rgb_sha256,
                    "original_hw": [height, width], "status": status,
                    "seg_count": observation["seg_count"], "has_pred_mask": bool(masks.shape[0]),
                    "pred_mask_shapes": observation["pred_mask_shapes"], "mask_path": str(mask_path.resolve()),
                    "foreground_pixels": int(binary.sum()), "foreground_ratio": float(binary.mean()),
                    "explanation": explanation, "generated_phrases": phrases, **values,
                }
                append_jsonl(records_path, record)
                completed.add(sample_id)
                if (index + 1) % 10 == 0 or index + 1 == args.end:
                    print(json.dumps({"stage": "PHASE5A2_LEGION", "condition": args.condition, "done": index + 1, "end": args.end, "status": status}), flush=True)
        finally:
            type(model).evaluate = original_evaluate
        torch.cuda.synchronize(device)
        worker.update({"status": "COMPLETE", "completed": len(completed), "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    except BaseException as exc:
        worker.update({"status": "FAILED", "exception_type": type(exc).__name__, "exception": str(exc)})
        raise
    finally:
        worker["finished_at_utc"] = utc_now()
        worker["elapsed_seconds"] = time.monotonic() - started
        atomic_json(worker_path, worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
