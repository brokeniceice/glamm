#!/usr/bin/env python3
"""Run the fixed LEGION smoke test without editing the official checkout.

This wrapper only supplies launcher/path plumbing and observes generated token
IDs, decoded text, mask tensor shapes, and PyTorch peak CUDA memory.  It calls
the official ``scripts/loc_exp/infer.py`` entry point unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


CLIP_REPO_ID = "openai/clip-vit-large-patch14-336"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _redirect_clip_class(cls, clip_dir: Path, redirects: list[dict]) -> None:
    original = cls.from_pretrained

    def redirected(_cls, pretrained_model_name_or_path, *args, **kwargs):
        requested = os.fspath(pretrained_model_name_or_path)
        resolved = os.fspath(clip_dir) if requested == CLIP_REPO_ID else requested
        redirects.append({"class": cls.__name__, "requested": requested, "resolved": resolved})
        return original(resolved, *args, **kwargs)

    cls.from_pretrained = classmethod(redirected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legion-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-json", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.legion_repo, args.model_dir, args.clip_dir, args.image_dir):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")

    infer_path = args.legion_repo / "scripts" / "loc_exp" / "infer.py"
    if not infer_path.is_file():
        parser.error(f"official inference entry point is missing: {infer_path}")

    observation = {
        "status": "started",
        "started_at_utc": _utc_now(),
        "legion_repo": os.fspath(args.legion_repo.resolve()),
        "model_dir": os.fspath(args.model_dir.resolve()),
        "clip_dir": os.fspath(args.clip_dir.resolve()),
        "image_dir": os.fspath(args.image_dir.resolve()),
        "output_dir": os.fspath(args.output_dir.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clip_path_redirects": [],
        "evaluate_calls": [],
        "decoded_outputs": [],
    }
    _atomic_json_dump(args.observation_json, observation)

    repo_string = os.fspath(args.legion_repo.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    # Resolve every official request for the named CLIP tower to the audited,
    # revision-pinned local snapshot.  Return values are left untouched.
    for cls in (
        transformers.CLIPVisionConfig,
        transformers.CLIPVisionModel,
        transformers.CLIPImageProcessor,
    ):
        _redirect_clip_class(cls, args.clip_dir.resolve(), observation["clip_path_redirects"])

    # Observe decoded generation text while returning the exact original value.
    original_tokenizer_loader = transformers.AutoTokenizer.from_pretrained

    def observed_tokenizer_loader(_cls, *loader_args, **loader_kwargs):
        tokenizer = original_tokenizer_loader(*loader_args, **loader_kwargs)
        original_decode = tokenizer.decode

        def observed_decode(*decode_args, **decode_kwargs):
            decoded = original_decode(*decode_args, **decode_kwargs)
            observation["decoded_outputs"].append(decoded)
            return decoded

        tokenizer.decode = observed_decode
        return tokenizer

    transformers.AutoTokenizer.from_pretrained = classmethod(observed_tokenizer_loader)

    from model.Legion import LegionForCausalLM

    original_evaluate = LegionForCausalLM.evaluate

    def observed_evaluate(model, *evaluate_args, **evaluate_kwargs):
        generated_ids, pred_masks = original_evaluate(model, *evaluate_args, **evaluate_kwargs)
        observation["evaluate_calls"].append(
            {
                "generated_shape": list(generated_ids.shape),
                "seg_token_id": int(model.seg_token_idx),
                "seg_count": int((generated_ids == model.seg_token_idx).sum().item()),
                "pred_mask_shapes": [list(mask.shape) for mask in pred_masks],
            }
        )
        return generated_ids, pred_masks

    LegionForCausalLM.evaluate = observed_evaluate

    official_argv = [
        os.fspath(infer_path),
        "--hf_model_path",
        os.fspath(args.model_dir.resolve()),
        "--image_dir",
        os.fspath(args.image_dir.resolve()),
        "--output_dir",
        os.fspath(args.output_dir.resolve()),
        "--local_rank",
        "0",
    ]
    observation["official_argv"] = official_argv

    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        sys.argv = official_argv
        runpy.run_path(os.fspath(infer_path), run_name="__main__")
        torch.cuda.synchronize()
        observation.update(
            {
                "status": "completed",
                "gpu_name": torch.cuda.get_device_name(),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        return 0
    except BaseException as exc:
        observation.update(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        observation["finished_at_utc"] = _utc_now()
        observation["elapsed_seconds"] = time.monotonic() - started
        if torch.cuda.is_available():
            observation.setdefault("gpu_name", torch.cuda.get_device_name())
            observation.setdefault("peak_memory_allocated_bytes", torch.cuda.max_memory_allocated())
            observation.setdefault("peak_memory_reserved_bytes", torch.cuda.max_memory_reserved())
        _atomic_json_dump(args.observation_json, observation)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
