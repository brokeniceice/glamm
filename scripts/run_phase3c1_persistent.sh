#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT="$ROOT/outputs/phase3c1_spatial_probe"
WAIT_PID="${1:-}"
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$ROOT"
mkdir -p "$OUT/reports"

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] persistent service waiting for pre-existing cache supervisor pid=$WAIT_PID" \
    >>"$OUT/reports/persistent_service.log"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi

echo "[$(date -Is)] persistent cache/resume starts" >>"$OUT/reports/persistent_service.log"
for source in npr srm focal sam clip; do
  for split in train val; do
    "$PYTHON" scripts/phase3c1_cache.py cache --source "$source" --split "$split" --device cuda:1 \
      >>"$OUT/reports/cache_${source}_${split}.log" 2>&1
  done
done

echo "[$(date -Is)] persistent cache complete; hand off automatic continuation" \
  >>"$OUT/reports/persistent_service.log"
exec bash scripts/run_phase3c1_after_cache.sh
