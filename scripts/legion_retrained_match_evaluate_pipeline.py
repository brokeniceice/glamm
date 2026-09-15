#!/usr/bin/env python3
"""Detached two-GPU frozen evaluation for LEGION-retrained-match."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/home/yz/miniconda3/envs/legion/bin/python")
MODEL_CLS = ROOT / "checkpoints/legion_retrained_match/stage2_cls/final_model"
MODEL_LE = ROOT / "checkpoints/legion_retrained_match/stage1_le"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
OUT = ROOT / "outputs/legion_retrained_match_evaluation"
STATUS = OUT / "pipeline_status.json"
LOGS = OUT / "logs"


def now(): return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); os.replace(tmp, path)


def rows(path: Path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_memory(gpu: int) -> int:
    value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)], text=True)
    return int(value.strip())


def wait_gpu(gpu: int, state: dict, limit_mib: int = 20000) -> None:
    while gpu_memory(gpu) >= limit_mib:
        state.update(status="WAITING_GPU", stage=f"wait_gpu_{gpu}", memory_used_mib=gpu_memory(gpu)); atomic(STATUS, state); time.sleep(30)


def launch(name: str, gpu: int, command: list[str]):
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    LOGS.mkdir(parents=True, exist_ok=True); handle = (LOGS / f"{name}.log").open("a", buffering=1)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def run_pair(jobs, state, stage):
    state.update(status="RUNNING", stage=stage, jobs=[job[0] for job in jobs]); atomic(STATUS, state)
    active = []
    for name, gpu, command in jobs:
        while gpu_memory(gpu) >= 20000:
            failed = [(old_name, process.returncode) for old_name, process, _ in active
                      if process.poll() is not None and process.returncode]
            if failed:
                for _, _, old_handle in active: old_handle.close()
                raise RuntimeError(f"worker failures while waiting for gpu_{gpu}: {failed}")
            state.update(status="WAITING_GPU", stage=f"wait_gpu_{gpu}", memory_used_mib=gpu_memory(gpu)); atomic(STATUS, state); time.sleep(30)
        process, handle = launch(name, gpu, command); active.append((name, process, handle))
    errors = []
    for name, process, handle in active:
        code = process.wait(); handle.close()
        if code: errors.append((name, code))
    if errors: raise RuntimeError(f"worker failures: {errors}")


def build_internal_fake_manifest() -> Path:
    source = rows(ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl")
    public = {row["sample_id"]: row for row in rows(ROOT / "datasets/Internal2208/manifests/eval_manifest.jsonl")}
    selected = []
    for row in source:
        if int(row["class_label"]) != 1: continue
        value = dict(row); value["image_path"] = public[row["sample_id"]]["image_path"]; selected.append(value)
    if len(selected) != 1104 or len({row["sample_id"] for row in selected}) != 1104:
        raise RuntimeError("internal-test Fake localization population drift")
    path = OUT / "manifests/internal_test_fake_1104.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp"); tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)); os.replace(tmp, path)
    atomic(OUT / "manifests/internal_test_fake_1104_provenance.json", {
        "status": "FROZEN", "n": 1104, "source": str((ROOT / "outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl").resolve()),
        "manifest": str(path.resolve()), "manifest_sha256": sha(path), "scope": "internal test Fake-only union GT"})
    return path


def shared_job(name, gpu, manifest, target, n):
    root = OUT / "localization" / name
    command = [str(PY), str(ROOT / "scripts/phase5a2_legion_shared_evaluate.py"),
               "--legion-repo", str(ROOT / "external/LEGION_official"), "--model-dir", str(MODEL_LE),
               "--clip-dir", str(CLIP), "--manifest", str(manifest), "--output-root", str(root),
               "--condition", "original", "--target-kind", target, "--start", "0", "--end", str(n), "--device", "cuda:0"]
    return name, gpu, command


def summarize_shared(name: str, manifest: Path):
    root = OUT / "localization" / name
    files = sorted((root / "original/shards").glob("*.predictions.jsonl")); records = [row for path in files for row in rows(path)]
    expected = [row["sample_id"] for row in rows(manifest)]; by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(expected) or set(by_id) != set(expected): raise RuntimeError(f"{name} incomplete")
    ordered = [by_id[sid] for sid in expected]; iou = np.asarray([r["foreground_iou"] for r in ordered]); f1 = np.asarray([r["foreground_f1"] for r in ordered])
    tp=sum(r["tp"] for r in ordered); fp=sum(r["fp"] for r in ordered); fn=sum(r["fn"] for r in ordered)
    result={"status":"COMPLETE","model":"legion_retrained_match","dataset":name,"condition":"L-FREE","n":len(ordered),
            "manifest":str(manifest.resolve()),"manifest_sha256":sha(manifest),"source_files":[str(p.resolve()) for p in files],
            "metrics":{"mean_foreground_iou":float(iou.mean()),"median_foreground_iou":float(np.median(iou)),
                       "mean_foreground_f1":float(f1.mean()),"global_foreground_iou":float(tp/max(1,tp+fp+fn)),
                       "global_foreground_f1":float(2*tp/max(1,2*tp+fp+fn)),
                       "seg_trigger_rate":float(np.mean([r.get("seg_count",0)>0 for r in ordered]))}}
    atomic(root / "results.json", result); return result


def main():
    state={"status":"RUNNING","stage":"preflight","started_at_utc":now(),"pid":os.getpid()}; atomic(STATUS,state)
    try:
        for path in (MODEL_CLS, MODEL_LE, CLIP):
            if not path.exists(): raise FileNotFoundError(path)
        internal = build_internal_fake_manifest()
        cls_script = str(ROOT / "scripts/legion_retrained_match_classification.py")
        run_pair([
            ("classification_gpu1",1,[str(PY),cls_script,"--datasets","internal","genimage","--device","cuda:0","--batch-size","32"]),
            ("classification_gpu0",0,[str(PY),cls_script,"--datasets","aigi_holmes","loki","raise998","--device","cuda:0","--batch-size","32"]),
        ],state,"classification")
        official = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
        loki = ROOT / "datasets/LOKI/legion_localization/manifest.jsonl"
        run_pair([shared_job("official1000",1,official,"synthscars_union",1000), shared_job("internal_test",0,internal,"internal_test_union",1104)],state,"localization_core")
        extra=str(ROOT / "scripts/legion_retrained_match_localization_extra.py")
        run_pair([shared_job("loki229",1,loki,"loki_bbox_union",229),
                  ("xaigd",0,[str(PY),extra,"--dataset","xaigd","--device","cuda:0"])],state,"localization_ood_1")
        run_pair([("pal4vst",1,[str(PY),extra,"--dataset","pal4vst","--device","cuda:0"])],state,"localization_ood_2")
        localization={
            "official1000":summarize_shared("official1000",official), "internal_test":summarize_shared("internal_test",internal),
            "loki229":summarize_shared("loki229",loki),
            "xaigd":json.loads((OUT/"localization/legion_retrained/xaigd/results.json").read_text()),
            "pal4vst":json.loads((OUT/"localization/legion_retrained/pal4vst/results.json").read_text()),
        }
        classification={name:json.loads((OUT/"classification"/name/"results.json").read_text()) for name in ("internal","aigi_holmes","genimage","loki","raise998")}
        atomic(OUT/"results.json",{"status":"COMPLETE","model":"legion_retrained_match","classification":classification,"localization":localization})
        state.update(status="COMPLETE",stage="STOP",completed_at_utc=now()); atomic(STATUS,state)
    except BaseException as error:
        state.update(status="FAILED",exception_type=type(error).__name__,exception=str(error),updated_at_utc=now()); atomic(STATUS,state); raise


if __name__ == "__main__": main()
