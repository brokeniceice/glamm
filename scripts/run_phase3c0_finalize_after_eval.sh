#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT="$ROOT/outputs/phase3c0_residual_diagnosis"
LOG="$OUT/reports/finalize.log"

cd "$ROOT"
while pgrep -f '[p]hase3c0_diagnose.py' >/dev/null; do
  sleep 30
done

STATUS="$($PYTHON -c 'import json; print(json.load(open("outputs/phase3c0_residual_diagnosis/provenance.json"))["status"])')"
if [[ "$STATUS" != "EVALUATION_COMPLETE" ]]; then
  echo "Refusing Phase3C0 finalize because evaluator status is $STATUS" >>"$LOG"
  exit 1
fi

"$PYTHON" -m pytest -q tests >"$OUT/reports/regression_tests.txt" 2>&1
"$PYTHON" scripts/phase3c0_analyze.py >>"$LOG" 2>&1
"$PYTHON" scripts/phase3c0_finalize.py >>"$LOG" 2>&1
