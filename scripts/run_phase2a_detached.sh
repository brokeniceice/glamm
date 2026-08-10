#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
export PYTHONPATH=/home/yz/groundingLMM_official

monitor_gpu() {
  while true; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader,nounits
    sleep 10
  done
}

monitor_gpu >> outputs/phase2a_unified_baseline/gpu_memory_monitor.log 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

/home/yz/miniconda3/envs/glamm_official/bin/deepspeed \
  --include localhost:1 \
  --master_port=29636 \
  scripts/phase2a_distributed_train.py \
  --mode train \
  --micro-batch-size 10 \
  --gradient-accumulation-steps 2 \
  --workers 8 \
  --resume checkpoints/phase2a_unified_baseline/single/last \
  2>&1 | tee outputs/phase2a_unified_baseline/train_full.log
