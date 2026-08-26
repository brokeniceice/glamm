#!/usr/bin/env python3
"""Prepare and run the frozen Phase 3D.0-S blind multimodal GPT audit.

This script never performs model rollout or training.  It reads only frozen
Phase 3D.0-R trajectories, builds deterministic blind comparison packets, and
calls the OpenAI Responses API.  Per-request shards make the run resumable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset

DEFAULT_API_BASE = "https://api.openai.com/v1"
OUT = ROOT / "outputs/phase3d0s_gpt_audit"
SOURCE = ROOT / "outputs/phase3d0r_reward_reformulation"
PREFLIGHT = ROOT / "outputs/phase3d0_reward_preflight"
VAL_ROWS = ROOT / "outputs/data_audits/unified_forensics_split_v1/val_combined.jsonl"

SYSTEM_PROMPT = """You are an independent blinded multimodal forensic-output evaluator. Compare only Candidate A and Candidate B using the supplied original image, the annotation-derived reference-region visualization, each target-region phrase, each explanation, and each candidate mask visualization. You do not know how either candidate was selected. Do not infer or discuss candidate source, reward, model confidence, or hidden metrics. Allow semantic paraphrases; do not require word-for-word overlap. Penalize hallucinated regions and incorrect spatial relations. Return only JSON matching the requested schema."""

USER_TEMPLATE = """Evaluate which candidate provides better forensic target-region identification.

Image 1 is the original image.
Image 2 is the reference region from the official SynthScars annotation-derived target; it is not described as absolute truth.
Image 3 is Candidate A's predicted-mask overlay.
Image 4 is Candidate B's predicted-mask overlay.

Candidate A target-regions phrase:
{a_phrase}

Candidate A explanation:
{a_explanation}

Candidate B target-regions phrase:
{b_phrase}

Candidate B explanation:
{b_explanation}

Judge: (1) overall forensic target-region identification, (2) phrase semantic correctness, (3) whether each candidate's phrase corresponds to its own predicted mask, and (4) whether each explanation has visual evidence and agrees with its phrase and mask. Scores are integers 0-4, where 0 is completely wrong and 4 is correct; semantic paraphrases can score 4. Phrase-mask consistency is internal consistency, not mask-versus-reference overlap."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "A_score", "B_score", "reason", "A_phrase_semantic_score",
                 "B_phrase_semantic_score", "A_phrase_mask_consistency_score",
                 "B_phrase_mask_consistency_score", "A_explanation_evidence_consistency_score",
                 "B_explanation_evidence_consistency_score"],
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "TIE", "UNCERTAIN"]},
        "A_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "B_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "reason": {"type": "string", "maxLength": 500},
        "A_phrase_semantic_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "B_phrase_semantic_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "A_phrase_mask_consistency_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "B_phrase_mask_consistency_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "A_explanation_evidence_consistency_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "B_explanation_evidence_consistency_score": {"type": "integer", "minimum": 0, "maximum": 4}
    }
}


def cli_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase3d0s_gpt_audit.json")
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--output-root", default="outputs/phase3d0s_gpt_audit")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    return p.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(seed: int, namespace: str, value: str) -> str:
    return sha256_text(f"{seed}:{namespace}:{value}")


def choose_deterministic(rows: list[dict], count: int, seed: int, namespace: str) -> list[dict]:
    ordered = sorted(rows, key=lambda row: stable_key(seed, namespace, f"{row['sample_id']}:{row.get('rollout_index', -1)}"))
    if len(ordered) >= count:
        return ordered[:count]
    if not ordered:
        raise RuntimeError(f"No eligible cases for {namespace}")
    return [ordered[index % len(ordered)] for index in range(count)]


def resolve_image_path(row: dict) -> Path:
    candidate = row.get("image_path")
    if candidate and Path(candidate).is_file():
        return Path(candidate).resolve()
    root = ROOT / ("datasets/SynthScars" if row["forensics_domain"] == "fake" else "datasets")
    path = (root / row["image_relpath"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mask_array(path: str, height: int, width: int) -> np.ndarray:
    value = torch.load(path, map_location="cpu").bool()
    while value.ndim > 2:
        value = value.any(dim=0)
    array = value.numpy().astype(bool)
    if array.shape != (height, width):
        array = np.asarray(Image.fromarray(array.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 0
    return array


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    layer = np.asarray(color, dtype=np.float32)
    base[mask] = 0.55 * base[mask] + 0.45 * layer
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def save_api_image(image: Image.Image, path: Path, max_side: int, quality: int) -> None:
    value = image.copy()
    value.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path, format="JPEG", quality=quality, optimize=True)


def prepare_assets(pair: dict, source_row: dict, cfg: dict) -> dict:
    directory = OUT / "audit/assets" / pair["pair_id"]
    paths = {name: directory / f"{name}.jpg" for name in ("original", "reference", "left", "right")}
    if all(path.is_file() for path in paths.values()):
        return {key: str(value) for key, value in paths.items()}
    image = Image.open(resolve_image_path(source_row)).convert("RGB")
    width, height = image.size
    reference_tensor = UnifiedForensicsDataset._fake_union_mask(source_row, height, width).bool()
    while reference_tensor.ndim > 2:
        reference_tensor = reference_tensor.any(dim=0)
    reference = reference_tensor.numpy()
    left = mask_array(pair["left"]["binary_mask_path"], height, width)
    right = mask_array(pair["right"]["binary_mask_path"], height, width)
    save_api_image(image, paths["original"], cfg["max_image_side"], cfg["jpeg_quality"])
    save_api_image(overlay(image, reference, (255, 32, 32)), paths["reference"], cfg["max_image_side"], cfg["jpeg_quality"])
    save_api_image(overlay(image, left, (0, 220, 255)), paths["left"], cfg["max_image_side"], cfg["jpeg_quality"])
    save_api_image(overlay(image, right, (255, 210, 0)), paths["right"], cfg["max_image_side"], cfg["jpeg_quality"])
    return {key: str(value) for key, value in paths.items()}


def compact_candidate(row: Mapping[str, Any]) -> dict:
    required = ["sample_id", "rollout_index", "normalized_phrase", "generated_explanation", "binary_mask_path",
                "R_phrase_lex", "R_phrase_sem", "R_mask", "R_ground_rel", "P_spatial_contra", "OLD_R3", "Q2"]
    return {key: row.get(key) for key in required}


def failure_trajectories(groups: list[dict], valid_ids: set[str]) -> dict[str, list[dict]]:
    all_rows = [row for group in groups if int(group["rollouts"][0]["gt_class"]) == 1
                for row in group["rollouts"] if group["sample_id"] in valid_ids and row.get("binary_mask_path")]
    ranks = {}
    for group in groups:
        if int(group["rollouts"][0]["gt_class"]) != 1:
            continue
        old = sorted(group["rollouts"], key=lambda x: (-float(x["OLD_R3"]), int(x["rollout_index"])))
        new = sorted(group["rollouts"], key=lambda x: (-float(x["Q2"]), int(x["rollout_index"])))
        ranks[group["sample_id"]] = ({x["rollout_index"]: i for i, x in enumerate(old)},
                                      {x["rollout_index"]: i for i, x in enumerate(new)})
    result = {
        "LEXICAL_LOW_SEMANTIC_HIGH": [x for x in all_rows if x["R_phrase_lex"] < .3 and x["R_phrase_sem"] >= .7],
        "LEXICAL_HIGH_SEMANTIC_LOW": [x for x in all_rows if x["R_phrase_lex"] >= .7 and x["R_phrase_sem"] < .5],
        "SPATIAL_CONTRADICTION": [x for x in all_rows if bool(x["P_spatial_contra"])],
        "PHRASE_GOOD_MASK_POOR": [x for x in all_rows if x["R_phrase_sem"] >= .7 and x["R_mask"] <= .2],
        "OLD_R3_DOUBLE_PENALTY": []
    }
    for x in all_rows:
        old_rank, new_rank = ranks[x["sample_id"]]
        if x["R_phrase_sem"] >= .7 and x["R_mask"] <= .2 and old_rank[x["rollout_index"]] - new_rank[x["rollout_index"]] >= 2:
            result["OLD_R3_DOUBLE_PENALTY"].append(x)
    return result


def build_manifest(cfg: dict) -> tuple[list[dict], dict]:
    seed = int(cfg["seed"])
    q2 = {x["sample_id"]: x for x in read_jsonl(SOURCE / "reward_candidates/top_selected_Q2.jsonl") if int(x["gt_class"]) == 1}
    old = {x["sample_id"]: x for x in read_jsonl(SOURCE / "reward_candidates/top_selected_OLD_R3.jsonl") if int(x["gt_class"]) == 1}
    greedy = {x["sample_id"]: x for x in read_jsonl(SOURCE / "validation/scored_greedy.jsonl") if int(x["gt_class"]) == 1}
    groups = read_jsonl(SOURCE / "validation/scored_groups.jsonl")
    source_rows = {x["sample_id"]: x for x in read_jsonl(VAL_ROWS) if x["forensics_domain"] == "fake"}
    ids = sorted(sid for sid in set(q2) & set(old) & set(greedy) & set(source_rows)
                 if q2[sid].get("binary_mask_path") and old[sid].get("binary_mask_path")
                 and greedy[sid].get("binary_mask_path"))
    a_ids = [x["sample_id"] for x in choose_deterministic([q2[sid] for sid in ids], cfg["primary_q2_vs_greedy"], seed, "primary-q2-greedy")]
    b_ids = [x["sample_id"] for x in choose_deterministic([q2[sid] for sid in ids], cfg["primary_q2_vs_oldr3"], seed, "primary-q2-old")]
    pairs = []
    def add(cohort: str, stratum: str, left: dict, right: dict, ordinal: int):
        pair_id = f"P{len(pairs):04d}"
        flip = int(stable_key(seed, "blind", pair_id), 16) % 2 == 1
        candidates = [compact_candidate(left), compact_candidate(right)]
        if flip:
            candidates.reverse()
        pairs.append({"pair_id": pair_id, "cohort": cohort, "stratum": stratum,
                      "sample_id": left["sample_id"], "source_ordinal": ordinal,
                      "left": candidates[0], "right": candidates[1],
                      "left_source": "comparator" if flip else "Q2",
                      "right_source": "Q2" if flip else "comparator"})
    for i, sid in enumerate(a_ids):
        add("Q2_VS_GREEDY", "PRIMARY", q2[sid], greedy[sid], i)
    for i, sid in enumerate(b_ids):
        add("Q2_VS_OLD_R3", "PRIMARY", q2[sid], old[sid], i)
    pools = failure_trajectories(groups, set(ids))
    pool_counts = {}
    for stratum, pool in pools.items():
        selected = choose_deterministic(pool, cfg["failure_per_stratum"], seed, "failure-" + stratum)
        pool_counts[stratum] = {"eligible_trajectories": len(pool), "selected": len(selected),
                                "with_replacement": len(pool) < cfg["failure_per_stratum"]}
        for i, comparator in enumerate(selected):
            add("FAILURE_ORIENTED", stratum, q2[comparator["sample_id"]], comparator, i)
    for pair in pairs:
        pair["assets"] = prepare_assets(pair, source_rows[pair["sample_id"]], cfg)
    public = [{key: value for key, value in pair.items() if key not in {"left_source", "right_source"}} for pair in pairs]
    manifest = {"schema_version": "phase3d0s_sampling_manifest_v1", "status": "FROZEN_BEFORE_GPT_CALLS",
                "seed": seed, "pair_count": len(pairs), "cohort_counts": {
                    "Q2_VS_GREEDY": len(a_ids), "Q2_VS_OLD_R3": len(b_ids), "FAILURE_ORIENTED": sum(len(x) for x in pools.values()) * 0 + 5 * cfg["failure_per_stratum"]},
                "failure_pool_counts": pool_counts, "pairs": public}
    blind = {"schema_version": "phase3d0s_blind_mapping_v1", "status": "SEALED_UNTIL_GPT_COMPLETE",
             "seed": seed, "mapping_count": len(pairs), "mappings": [{"pair_id": x["pair_id"],
             "candidate_A_source": x["left_source"], "candidate_B_source": x["right_source"]} for x in pairs]}
    return pairs, {"manifest": manifest, "blind": blind}


def prepare(cfg: dict) -> list[dict]:
    pairs, payloads = build_manifest(cfg)
    write_json(OUT / "audit_sampling_manifest.json", payloads["manifest"])
    write_json(OUT / "sampling_manifest.json", payloads["manifest"])
    write_json(OUT / "blind/blind_mapping.json", payloads["blind"])
    write_json(OUT / "blind_mapping.json", payloads["blind"])
    os.chmod(OUT / "blind_mapping.json", 0o600)
    os.chmod(OUT / "blind/blind_mapping.json", 0o600)
    write_json(OUT / "route_gate.json", {"schema_version": "phase3d0s_route_gate_v1",
        "status": "PREPARED_AWAITING_GPT_API", "gate": None, "terminal": False,
        "evaluation_started": False, "rollout_repeated": False, "training_started": False})
    return pairs


def image_data_url(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + data


def request_payload(pair: dict, orientation: str, model: str, cfg: dict) -> dict:
    left, right = pair["left"], pair["right"]
    left_asset, right_asset = pair["assets"]["left"], pair["assets"]["right"]
    if orientation == "BA":
        left, right = right, left
        left_asset, right_asset = right_asset, left_asset
    prompt = USER_TEMPLATE.format(a_phrase=left["normalized_phrase"] or "(none)",
        a_explanation=left["generated_explanation"] or "(none)",
        b_phrase=right["normalized_phrase"] or "(none)",
        b_explanation=right["generated_explanation"] or "(none)")
    content = [{"type": "input_text", "text": prompt}]
    for path in (pair["assets"]["original"], pair["assets"]["reference"], left_asset, right_asset):
        content.append({"type": "input_image", "image_url": image_data_url(path), "detail": cfg["image_detail"]})
    payload = {"model": model, "store": False, "reasoning": {"effort": cfg["reasoning_effort"]},
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": content}], "text": {"verbosity": "low", "format": {
            "type": "json_schema", "name": "phase3d0s_blind_judgment", "strict": True, "schema": RESPONSE_SCHEMA}},
        "max_output_tokens": cfg["max_output_tokens"]}
    if cfg.get("temperature") is not None:
        payload["temperature"] = cfg["temperature"]
    return payload


def post(api_base: str, api_key: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(api_base.rstrip("/") + "/responses",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def output_text(response: dict) -> str:
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"]
                if content.get("type") == "refusal":
                    raise RuntimeError("API refusal: " + content.get("refusal", ""))
    raise RuntimeError("Responses API returned no output_text")


def evaluate(task: dict, model: str, api_base: str, api_key: str, cfg: dict) -> dict:
    payload = request_payload(task["pair"], task["orientation"], model, cfg)
    for attempt in range(cfg["max_retries"] + 1):
        try:
            response = post(api_base, api_key, payload, cfg["timeout_seconds"])
            parsed = json.loads(output_text(response))
            return {"request_id": task["request_id"], "pair_id": task["pair"]["pair_id"],
                "orientation": task["orientation"], "replicate": task["replicate"],
                "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
                "user_prompt_sha256": sha256_text(payload["input"][0]["content"][0]["text"]),
                "requested_model": model, "response_id": response.get("id"),
                "response_model": response.get("model"), "usage": response.get("usage"),
                "created_at": response.get("created_at"), "parsed": parsed, "raw_response": response}
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:3000]
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt >= cfg["max_retries"]:
                raise RuntimeError(f"HTTP {error.code}: {body}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt >= cfg["max_retries"]:
                raise RuntimeError(str(error)) from error
        time.sleep(min(30, 2 ** attempt) + random.random())
    raise AssertionError("unreachable")


def shard_path(request_id: str) -> Path:
    return OUT / "raw_responses/shards" / f"{request_id}.json"


def tasks_for(pairs: list[dict], cfg: dict) -> list[dict]:
    tasks = []
    for pair in pairs:
        for orientation in ("AB", "BA"):
            tasks.append({"request_id": f"{pair['pair_id']}_{orientation}_r0", "pair": pair,
                          "orientation": orientation, "replicate": 0})
    repeat = sorted(pairs, key=lambda x: stable_key(cfg["seed"], "repeatability", x["pair_id"]))[:cfg["repeatability_pairs"]]
    for pair in repeat:
        tasks.append({"request_id": f"{pair['pair_id']}_AB_r1", "pair": pair,
                      "orientation": "AB", "replicate": 1})
    return tasks


def valid_existing(task: dict, model: str) -> dict | None:
    path = shard_path(task["request_id"])
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("request_id") == task["request_id"] and row.get("requested_model") == model:
            return row
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def write_provenance(cfg: dict, model: str, api_base: str, status: str) -> None:
    write_json(OUT / "gpt_provenance.json", {"schema_version": "phase3d0s_gpt_provenance_v1",
        "status": status, "provider": "OpenAI-compatible user-provided API",
        "model_identifier": model, "api_endpoint_type": "Responses API /v1/responses",
        "api_base_host": urllib.request.urlparse(api_base).netloc if hasattr(urllib.request, "urlparse") else api_base.split("//")[-1].split("/")[0],
        "model_version": None, "temperature": cfg.get("temperature"),
        "temperature_parameter_sent": cfg.get("temperature") is not None,
        "temperature_note": cfg.get("temperature_note"), "max_tokens": cfg["max_output_tokens"],
        "reasoning_effort": cfg["reasoning_effort"], "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "user_prompt_template_sha256": sha256_text(USER_TEMPLATE), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "retry_policy": {"max_retries": cfg["max_retries"], "backoff": "exponential capped at 30s plus jitter",
                         "retryable_http": [429, "5xx"]},
        "raw_response_storage_format": "one full Responses API JSON object per JSON shard; compiled JSONL after completion",
        "api_key_recorded": False, "reward_values_sent_to_gpt": False, "candidate_sources_sent_to_gpt": False})


def compile_outputs(tasks: list[dict]) -> None:
    rows = [json.loads(shard_path(task["request_id"]).read_text(encoding="utf-8")) for task in tasks]
    append_jsonl(OUT / "raw_responses.jsonl", rows)
    parsed = [{key: row[key] for key in ("request_id", "pair_id", "orientation", "replicate", "requested_model", "response_id", "response_model", "usage")} | row["parsed"] for row in rows]
    append_jsonl(OUT / "parsed_scores.jsonl", parsed)
    append_jsonl(OUT / "parsed_scores/parsed_scores.jsonl", parsed)
    blind = json.loads((OUT / "blind_mapping.json").read_text(encoding="utf-8"))
    blind["status"] = "UNBLINDED_AFTER_GPT_COMPLETE"
    blind["unblinded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "blind_mapping.json", blind)
    write_json(OUT / "blind/blind_mapping.json", blind)
    write_json(OUT / "route_gate.json", {"schema_version": "phase3d0s_route_gate_v1",
        "status": "GPT_EVALUATION_COMPLETE_ANALYSIS_PENDING", "gate": None, "terminal": False,
        "completed_requests": len(rows), "evaluation_started": True, "rollout_repeated": False,
        "training_started": False})


def main(argv=None):
    global OUT
    args = cli_args(argv)
    OUT = (ROOT / args.output_root).resolve()
    cfg_path = (ROOT / args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = args.model or cfg["model"]
    workers = args.workers or cfg["default_workers"]
    pairs = prepare(cfg)
    print(f"prepared_pairs={len(pairs)} model={model} requests={len(tasks_for(pairs, cfg))}", flush=True)
    if args.prepare_only:
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; no API requests were made")
    api_base = os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE)
    write_provenance(cfg, model, api_base, "GPT_API_RUN_STARTED")
    tasks = tasks_for(pairs, cfg)
    existing = {task["request_id"]: valid_existing(task, model) for task in tasks}
    pending = [task for task in tasks if existing[task["request_id"]] is None]
    print(f"existing={len(tasks)-len(pending)} pending={len(pending)} workers={workers}", flush=True)
    if pending:
        smoke = pending[0]
        try:
            result = evaluate(smoke, model, api_base, api_key, cfg)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            lowered = message.lower()
            vision_unavailable = any(token in lowered for token in ("image input", "vision", "does not support image", "unsupported image"))
            status = "GPT_VISION_UNAVAILABLE" if vision_unavailable else "GPT_API_PREFLIGHT_FAILED"
            write_provenance(cfg, model, api_base, status)
            write_json(OUT / "route_gate.json", {"schema_version": "phase3d0s_route_gate_v1",
                "status": status, "gate": status, "terminal": True, "evaluation_started": False,
                "smoke_error": message, "rollout_repeated": False, "training_started": False})
            raise
        write_json(shard_path(smoke["request_id"]), result)
        print(f"vision_smoke_pass={smoke['request_id']} response_model={result.get('response_model')}", flush=True)
        pending = pending[1:]
    if args.smoke_only:
        write_provenance(cfg, model, api_base, "VISION_SMOKE_PASS")
        return
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, task, model, api_base, api_key, cfg): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                write_json(shard_path(task["request_id"]), future.result())
            except Exception as error:
                failures.append({"request_id": task["request_id"], "error": f"{type(error).__name__}: {error}"})
            if completed % 10 == 0 or completed == len(pending):
                print(f"completed_this_run={completed}/{len(pending)} failures={len(failures)}", flush=True)
    write_json(OUT / "audit/api_failures.json", failures)
    missing = [task["request_id"] for task in tasks if valid_existing(task, model) is None]
    if missing:
        write_provenance(cfg, model, api_base, "INCOMPLETE_RETRY_REQUIRED")
        raise RuntimeError(f"{len(missing)} requests remain incomplete; rerun the same command to resume")
    compile_outputs(tasks)
    write_provenance(cfg, model, api_base, "GPT_EVALUATION_COMPLETE")
    print("GPT_EVALUATION_COMPLETE_ANALYSIS_PENDING", flush=True)


if __name__ == "__main__":
    main()
