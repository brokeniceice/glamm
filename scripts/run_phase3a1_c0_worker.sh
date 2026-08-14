#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
export PYTHONPATH=/home/yz/groundingLMM_official
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

exec /home/yz/miniconda3/envs/glamm_official/bin/deepspeed \
  --include localhost:0 \
  --master_port 29525 \
  scripts/phase2a_distributed_train.py \
  --config configs/phase3a1_paired_c0.yaml \
  --mode train \
  --workers 8 \
  >>outputs/phase3a1_paired_control/c0_training/logs/formal_training.log 2>&1
