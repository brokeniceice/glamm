#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT="$ROOT/outputs/phase3c0_residual_diagnosis"
LOG="$OUT/reports/evaluation.log"

mkdir -p "$OUT/reports"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec "$PYTHON" scripts/phase3c0_diagnose.py \
  --device cuda:0 \
  --output-dir outputs/phase3c0_residual_diagnosis \
  >>"$LOG" 2>&1
