#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=/data/yz/groundingLMM_official/outputs/phase3c2_p3_robustness/robustness_internal
LOG="$OUT/automatic_pipeline.log"
export CUDA_VISIBLE_DEVICES=1
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$ROOT"
mkdir -p "$OUT/logs"

run_stage() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name" >>"$LOG"
  "$@" >>"$OUT/logs/${name}.log" 2>&1
  echo "[$(date -Is)] COMPLETE $name" >>"$LOG"
}

for condition in original jpeg70 jpeg80 gaussian5 gaussian10; do
  run_stage "evaluate_${condition}" "$PYTHON" scripts/phase3c2_internal_robustness.py evaluate \
    --condition "$condition" --device cuda:0
done

run_stage analyze "$PYTHON" scripts/phase3c2_internal_robustness.py analyze
run_stage tests "$PYTHON" -m pytest -q tests/test_phase3c2_p3.py tests/test_phase2c_forensic_fusion.py
run_stage finalize "$PYTHON" scripts/phase3c2_internal_robustness.py finalize
echo "[$(date -Is)] PHASE3C2_INTERNAL_ROBUSTNESS_COMPLETE" >>"$LOG"
