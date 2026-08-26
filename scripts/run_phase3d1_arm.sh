#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/yz/groundingLMM_official
env_python=/home/yz/miniconda3/envs/glamm_official/bin/python
arm=${1:?usage: run_phase3d1_arm.sh R3|Q2 physical_gpu}
physical_gpu=${2:?usage: run_phase3d1_arm.sh R3|Q2 physical_gpu}

case "$arm:$physical_gpu" in
  R3:1|Q2:2) ;;
  *) echo "Frozen GPU assignment is R3:1 and Q2:2" >&2; exit 2 ;;
esac

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
exec "$env_python" scripts/phase3d1_train.py \
  --arm "$arm" --physical-gpu "$physical_gpu" \
  >>"outputs/phase3d1_policy_optimization/logs/${arm,,}_training.log" 2>&1
