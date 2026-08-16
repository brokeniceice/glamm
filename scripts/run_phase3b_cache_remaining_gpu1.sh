#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
WAIT_PID=${1:?current shard worker PID required}
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=1
cd "$ROOT"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

for SHARD in 2 0 3 4 5; do
  "$PYTHON" scripts/phase3b_build_replay_cache.py \
    --device cuda:0 --num-shards 6 --shard-index "$SHARD" --batch-size 1 \
    >>"outputs/phase3b_generated_replay/cache/shard_${SHARD}.log" 2>&1
done

"$PYTHON" scripts/phase3b_build_replay_cache.py \
  --merge --num-shards 6 --batch-size 1 \
  >outputs/phase3b_generated_replay/cache/merge.log 2>&1
