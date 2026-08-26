#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=outputs/phase3d0r_reward_reformulation

while [[ ! -f "$OUT/audit/reward_dev_fixed_token_mask_replay_shard00_of_02.json" || ! -f "$OUT/audit/reward_dev_fixed_token_mask_replay_shard01_of_02.json" ]]; do
  sleep 30
done

"$PYTHON" scripts/phase3d0r_dev.py > "$OUT/reports/reward_dev.log" 2>&1
"$PYTHON" scripts/phase3d0r_validate.py > "$OUT/reports/validation.log" 2>&1
"$PYTHON" -m pytest -q tests 2>&1 | tee "$OUT/regression_tests.txt" "$OUT/reports/regression_tests.txt"
"$PYTHON" scripts/phase3d0r_finalize.py > "$OUT/reports/finalize.log" 2>&1

# STOP: Phase 3D.1 and all policy training remain deliberately unstarted.
