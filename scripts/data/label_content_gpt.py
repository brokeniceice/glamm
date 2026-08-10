#!/usr/bin/env python3
"""Re-label sampled CLIP review images with the OpenAI Responses API.

The script uses only the Python standard library plus Pillow, so it does not
require upgrading the repository's legacy ``openai`` package. Results are
written separately from human corrections and are resumable per image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_BASE = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/data/gpt_content_labels_v1.json"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_gpt_v1",
    )
    parser.add_argument("--input-scope", choices=("review", "pass"), default="review")
    parser.add_argument(
        "--seed-predictions",
        type=Path,
        default=None,
        help="Reuse compatible GPT predictions from a prior run without new API calls.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-pending-human-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_review_items(labels_dir: Path, only_pending: bool, limit: int) -> list[dict[str, Any]]:
    candidates = load_jsonl(labels_dir / "review_candidates.jsonl")
    corrections = load_jsonl(labels_dir / "manual_corrections_template.jsonl")
    if len(candidates) != len(corrections):
        raise ValueError("Review candidate and human correction counts differ")
    items = []
    for review_index, (candidate, correction) in enumerate(zip(candidates, corrections)):
        if candidate["sample_id"] != correction["sample_id"]:
            raise ValueError(f"Review order mismatch at row {review_index + 1}")
        if only_pending and correction["reviewer_status"] != "pending":
            continue
        image_path = Path(candidate["image_path"]).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        items.append({
            "review_index": review_index,
            "sample_id": candidate["sample_id"],
            "source": candidate["source"],
            "domain": candidate["domain"],
            "image_path": str(image_path),
            "clip_label": candidate["predicted_category"],
            "clip_margin": candidate["top1_top2_margin"],
            "human_label": correction.get("manual_label"),
            "human_status": correction.get("reviewer_status", "pending"),
        })
    return items[:limit] if limit > 0 else items


def build_pass_items(labels_dir: Path, limit: int) -> list[dict[str, Any]]:
    rows = load_jsonl(labels_dir / "pass_predictions.jsonl")
    items = []
    for row in rows:
        image_path = Path(row["image_path"]).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        items.append({
            # embedding_index is globally unique and stable in the frozen CLIP run.
            "review_index": int(row["embedding_index"]),
            "sample_id": row["sample_id"],
            "source": row["source"],
            "domain": row["domain"],
            "image_path": str(image_path),
            "clip_label": row["predicted_category"],
            "clip_margin": row["top1_top2_margin"],
            "human_label": None,
            "human_status": "not_requested",
        })
    if len({item["sample_id"] for item in items}) != len(items):
        raise ValueError("Duplicate sample IDs in PASS predictions")
    return items[:limit] if limit > 0 else items


def encode_image(path: Path, max_side: int, jpeg_quality: int) -> tuple[str, dict[str, Any]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = list(image.size)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), {
        "original_size": original_size,
        "api_image_size": list(image.size),
        "api_image_bytes": len(payload),
    }


def response_schema(categories: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": categories},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False,
    }


def build_request(
    model: str, config: Mapping[str, Any], encoded_image: str
) -> dict[str, Any]:
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": config["reasoning_effort"]},
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": config["prompt"]},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": config["image_detail"],
                },
            ],
        }],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "dominant_content_label",
                "strict": True,
                "schema": response_schema(list(config["categories"])),
            },
        },
        "max_output_tokens": 400,
    }


def extract_output_text(response: Mapping[str, Any]) -> str:
    for output in response.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return str(content["text"])
            if content.get("type") == "refusal":
                raise RuntimeError(f"API refusal: {content.get('refusal', '')}")
    raise RuntimeError("Responses API result contains no output_text")


def post_response(
    api_base: str,
    api_key: str,
    request_payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        api_base.rstrip("/") + "/responses",
        data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def classify_one(
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    max_image_side: int,
    jpeg_quality: int,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    encoded, image_metadata = encode_image(Path(item["image_path"]), max_image_side, jpeg_quality)
    request_payload = build_request(model, config, encoded)
    for attempt in range(max_retries + 1):
        try:
            response = post_response(api_base, api_key, request_payload, timeout_seconds)
            parsed = json.loads(extract_output_text(response))
            if parsed["label"] not in config["categories"]:
                raise RuntimeError(f"Unexpected GPT label: {parsed['label']}")
            return {
                **item,
                "gpt_label": parsed["label"],
                "gpt_confidence": float(parsed["confidence"]),
                "gpt_reason": parsed["reason"],
                "model": response.get("model", model),
                "response_id": response.get("id"),
                "usage": response.get("usage"),
                **image_metadata,
            }
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            body = error.read().decode("utf-8", errors="replace")[:2000]
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"HTTP {error.code}: {body}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= max_retries:
                raise RuntimeError(str(error)) from error
        delay = min(30.0, 2.0 ** attempt) + random.random()
        time.sleep(delay)
    raise AssertionError("unreachable")


def shard_path(output_dir: Path, review_index: int) -> Path:
    return output_dir / "response_shards" / f"review-{review_index:04d}.json"


def valid_shard(path: Path, sample_id: str, model: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = load_json(path)
        if row.get("sample_id") == sample_id and row.get("requested_model") == model:
            return row
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def normalize_seed(item: Mapping[str, Any], seed: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        **item,
        "gpt_label": seed["gpt_label"],
        "gpt_confidence": seed["gpt_confidence"],
        "gpt_reason": seed["gpt_reason"],
        "model": seed["model"],
        "response_id": seed.get("response_id"),
        "usage": seed.get("usage"),
        "original_size": seed.get("original_size"),
        "api_image_size": seed.get("api_image_size"),
        "api_image_bytes": seed.get("api_image_bytes"),
        "requested_model": model,
        "reused_seed_prediction": True,
    }


def result_summary(results: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    manually_reviewed = [
        row for row in results
        if row["human_status"] == "reviewed" and row["human_label"] is not None
    ]
    agreements = sum(row["human_label"] == row["gpt_label"] for row in manually_reviewed)
    return {
        "schema_version": "gpt_content_labels_summary_v1",
        "completed_count": len(results),
        "failure_count": len(failures),
        "reused_seed_prediction_count": sum(
            bool(row.get("reused_seed_prediction")) for row in results
        ),
        "gpt_label_counts": dict(Counter(row["gpt_label"] for row in results)),
        "human_comparison_count": len(manually_reviewed),
        "human_gpt_agreement_count": agreements,
        "human_gpt_agreement_rate": agreements / len(manually_reviewed) if manually_reviewed else None,
        "human_gpt_disagreement_count": len(manually_reviewed) - agreements,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = load_json(config_path)
    model = args.model or config["default_model"]
    if args.input_scope == "review":
        items = build_review_items(labels_dir, args.only_pending_human_review, args.limit)
        input_path = labels_dir / "review_candidates.jsonl"
    else:
        if args.only_pending_human_review:
            raise ValueError("--only-pending-human-review is only valid for --input-scope review")
        items = build_pass_items(labels_dir, args.limit)
        input_path = labels_dir / "pass_predictions.jsonl"
    if not items:
        raise RuntimeError("No review items selected")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_image_side < 256:
        raise ValueError("--max-image-side must be at least 256")

    run_config = {
        **config,
        "requested_model": model,
        "input_scope": args.input_scope,
        "labels_dir": str(labels_dir),
        "selected_item_count": len(items),
        "only_pending_human_review": args.only_pending_human_review,
        "max_image_side": args.max_image_side,
        "jpeg_quality": args.jpeg_quality,
        "workers": args.workers,
        "config_sha256": sha256_file(config_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "human_corrections_sha256_at_start": (
            sha256_file(labels_dir / "manual_corrections_template.jsonl")
            if args.input_scope == "review" else None
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run_config.json", run_config)

    if args.dry_run:
        total_bytes = 0
        for item in items:
            _, metadata = encode_image(Path(item["image_path"]), args.max_image_side, args.jpeg_quality)
            total_bytes += metadata["api_image_bytes"]
        print(json.dumps({
            "dry_run": True,
            "selected_item_count": len(items),
            "encoded_image_bytes": total_bytes,
            "estimated_base64_bytes": math.ceil(total_bytes / 3) * 4,
            "model": model,
        }, indent=2))
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; no API requests were made")
    api_base = os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE)

    results_by_index: dict[int, dict[str, Any]] = {}
    pending = []
    seed_by_id: dict[str, dict[str, Any]] = {}
    if args.seed_predictions is not None:
        seed_path = args.seed_predictions.expanduser().resolve()
        seed_config_path = seed_path.parent / "run_config.json"
        if not seed_config_path.is_file():
            raise FileNotFoundError(f"Seed run_config.json not found beside {seed_path}")
        seed_config = load_json(seed_config_path)
        if seed_config.get("config_sha256") != run_config["config_sha256"]:
            raise ValueError("Seed predictions used a different GPT labeling config")
        if seed_config.get("requested_model") != model:
            raise ValueError("Seed predictions used a different requested model")
        seed_rows = load_jsonl(seed_path)
        seed_by_id = {row["sample_id"]: row for row in seed_rows}
        if len(seed_by_id) != len(seed_rows):
            raise ValueError("Duplicate sample IDs in seed predictions")
    for item in items:
        path = shard_path(output_dir, item["review_index"])
        existing = None if args.overwrite else valid_shard(path, item["sample_id"], model)
        if existing is not None:
            results_by_index[item["review_index"]] = existing
        elif not args.overwrite and item["sample_id"] in seed_by_id:
            seeded = normalize_seed(item, seed_by_id[item["sample_id"]], model)
            atomic_write_json(path, seeded)
            results_by_index[item["review_index"]] = seeded
        else:
            pending.append(item)
    print(f"selected={len(items)} existing={len(results_by_index)} pending={len(pending)}", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                classify_one,
                item,
                config,
                model,
                api_base,
                api_key,
                args.max_image_side,
                args.jpeg_quality,
                args.timeout_seconds,
                args.max_retries,
            ): item
            for item in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                row = future.result()
                row["requested_model"] = model
                atomic_write_json(shard_path(output_dir, item["review_index"]), row)
                results_by_index[item["review_index"]] = row
            except Exception as error:  # Keep other paid requests and record the exact failure.
                failures.append({
                    "review_index": item["review_index"],
                    "sample_id": item["sample_id"],
                    "error": f"{type(error).__name__}: {error}",
                })
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"API completed={completed}/{len(pending)} success={len(results_by_index)} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    results = [results_by_index[index] for index in sorted(results_by_index)]
    atomic_write_jsonl(output_dir / "predictions.jsonl", results)
    atomic_write_json(output_dir / "failures.json", failures)
    summary = result_summary(results, failures)
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
