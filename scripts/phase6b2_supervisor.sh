#!/usr/bin/env bash
set -euo pipefail

root="/home/yz/groundingLMM_official"
train_pid="$1"
val_pid="$2"
cd "$root"

while kill -0 "$train_pid" 2>/dev/null || kill -0 "$val_pid" 2>/dev/null; do
  sleep 30
done

test "$(/home/yz/miniconda3/envs/glamm_official/bin/python -c 'import json;print(json.load(open("outputs/phase6b2_fusion/features/train/complete.json"))["status"])')" = "COMPLETE"
test "$(/home/yz/miniconda3/envs/glamm_official/bin/python -c 'import json;print(json.load(open("outputs/phase6b2_fusion/features/val/complete.json"))["status"])')" = "COMPLETE"

CUDA_VISIBLE_DEVICES=0 /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6b2_fusion.py fuse
