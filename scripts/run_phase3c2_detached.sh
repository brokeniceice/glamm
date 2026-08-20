#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=/data/yz/groundingLMM_official/outputs/phase3c2_p3_robustness
LOG="$OUT/reports/automatic_pipeline.log"
export CUDA_VISIBLE_DEVICES=1
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$ROOT"
mkdir -p "$OUT/reports"

run_stage() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name" >>"$LOG"
  "$@" >>"$OUT/reports/${name}.log" 2>&1
  echo "[$(date -Is)] COMPLETE $name" >>"$LOG"
}

run_stage preflight "$PYTHON" scripts/phase3c2_p3.py preflight --device cuda:0

for split in train val test official1000; do
  case "$split" in
    train) count=17672 ;;
    val) count=2212 ;;
    test) count=2208 ;;
    official1000) count=1000 ;;
  esac
  start=0
  while (( start < count )); do
    end=$((start + 2048))
    if (( end > count )); then end=$count; fi
    run_stage "cache_${split}_${start}_${end}" "$PYTHON" scripts/phase3c2_p3.py cache \
      --split "$split" --start "$start" --end "$end" --device cuda:0 --batch-size 8
    start=$end
  done
  run_stage "combine_${split}" "$PYTHON" scripts/phase3c2_p3.py combine --split "$split"
done

run_stage train_p3 "$PYTHON" scripts/phase3c2_p3.py train --device cuda:0
run_stage evaluate_original "$PYTHON" scripts/phase3c2_p3.py evaluate-original --device cuda:0

for condition in original jpeg70 jpeg80 gaussian5 gaussian10; do
  run_stage "robust_${condition}_c0" "$PYTHON" scripts/phase3c2_p3.py robust-eval \
    --model c0 --condition "$condition" --device cuda:0
  run_stage "robust_${condition}_p1_p3" "$PYTHON" scripts/phase3c2_p3.py robust-eval \
    --model p1 --condition "$condition" --device cuda:0
done

run_stage analyze "$PYTHON" scripts/phase3c2_p3.py analyze
run_stage phase3c2_tests "$PYTHON" -m pytest -q tests/test_phase3c2_p3.py tests/test_phase2c_forensic_fusion.py
run_stage finalize "$PYTHON" scripts/phase3c2_p3.py finalize
echo "[$(date -Is)] PHASE3C2_COMPLETE" >>"$LOG"
