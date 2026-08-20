#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT="$ROOT/outputs/phase3c1_spatial_probe"
cd "$ROOT"
echo "[$(date -Is)] corrected project regression: pytest -q tests" \
  >>"$OUT/reports/automatic_continuation.log"
"$PYTHON" -m pytest -q tests >"$OUT/reports/regression_tests.txt" 2>&1
echo "[$(date -Is)] finalize after corrected regression" \
  >>"$OUT/reports/automatic_continuation.log"
"$PYTHON" scripts/phase3c1_finalize.py >>"$OUT/reports/automatic_continuation.log" 2>&1
echo "[$(date -Is)] COMPLETE" >>"$OUT/reports/automatic_continuation.log"
