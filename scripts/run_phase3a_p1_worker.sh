#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
export PYTHONPATH=/home/yz/groundingLMM_official
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec /home/yz/miniconda3/envs/glamm_official/bin/deepspeed \
  --include localhost:1 \
  --master_port 29512 \
  scripts/phase2a_distributed_train.py \
  --config configs/phase3a_p1.yaml \
  --mode train \
  --workers 8 \
  >>outputs/phase3a_phrase_grounding/logs/p1_training.log 2>&1
