#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official

PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUTPUT=outputs/phase3d0_reward_preflight

while [[ ! -f "$OUTPUT/audit/generation_dev_A_text_shard00.json" || ! -f "$OUTPUT/audit/generation_dev_B_text_shard00.json" ]]; do
  sleep 30
done

"$PYTHON" scripts/phase3d0_select_sampling.py --num-shards 1 \
  > "$OUTPUT/reports/select_sampling.log" 2>&1

SETTING=$("$PYTHON" -c 'import json; print(json.load(open("outputs/phase3d0_reward_preflight/sampling_protocol.json"))["selected_setting"])')

# Full generation + no-cache mask replay integration smoke on reward-dev only.
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  "$PYTHON" scripts/phase3d0_generate.py --population dev --setting "$SETTING" \
  --device cuda:0 --num-shards 97 --shard-index 0 --max-samples 1 \
  --include-greedy --skip-model-hash --reset \
  > "$OUTPUT/reports/full_replay_smoke.log" 2>&1

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  "$PYTHON" scripts/phase3d0_generate.py --population val --setting "$SETTING" \
  --device cuda:0 --num-shards 2 --shard-index 0 --include-greedy \
  > "$OUTPUT/reports/validation_shard0.log" 2>&1

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  "$PYTHON" scripts/phase3d0_generate.py --population val --setting "$SETTING" \
  --device cuda:0 --num-shards 2 --shard-index 1 --include-greedy \
  > "$OUTPUT/reports/validation_shard1.log" 2>&1

"$PYTHON" scripts/phase3d0_analyze.py --setting "$SETTING" --num-shards 2 \
  > "$OUTPUT/reports/analyze.log" 2>&1

set -o pipefail
"$PYTHON" -m pytest -q tests 2>&1 | tee "$OUTPUT/reports/regression_tests.txt"

"$PYTHON" scripts/phase3d0_finalize.py --num-shards 2 \
  > "$OUTPUT/reports/finalize.log" 2>&1

# STOP: Phase 3D.1 is deliberately not started.
