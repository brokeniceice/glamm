#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
OUT=outputs/phase2a_unified_baseline
PY=/home/yz/miniconda3/envs/glamm_official/bin/python

while tmux has-session -t phase2a_eval_gen 2>/dev/null \
   || tmux has-session -t phase2a_eval_forward 2>/dev/null; do
  sleep 30
done

grep -q 'PHASE2A_GENERATION_EXIT_STATUS=0' "$OUT/test_eval_generation.log"
grep -q 'PHASE2A_FORWARD_EXIT_STATUS=0' "$OUT/test_eval_forward.log"

"$PY" scripts/phase2a_finalize_artifacts.py > "$OUT/finalization.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q tests > "$OUT/final_regression.log" 2>&1
"$PY" scripts/phase2a_finalize_artifacts.py >> "$OUT/finalization.log" 2>&1
echo PHASE2A_FINAL_STATUS=COMPLETE
