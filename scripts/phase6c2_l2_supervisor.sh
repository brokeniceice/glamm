#!/usr/bin/env bash
set -euo pipefail

repo=/home/yz/groundingLMM_official
env_bin=/home/yz/miniconda3/envs/glamm_official/bin
checkpoint_root=/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2
initial_checkpoint=/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt
output_dir="$repo/outputs/phase6c2_multiseg_training/l2_training"
log_dir="$repo/outputs/phase6c2_multiseg_training/logs"

mkdir -p "$output_dir" "$log_dir"
cd "$repo"

export CUDA_VISIBLE_DEVICES=0,1
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

while true; do
  args=(
    --include localhost:0,1
    --master_port 29623
    scripts/phase2a_distributed_train.py
    --config configs/phase6c2_l2_p1_multi.yaml
    --mode train
    --micro-batch-size 5
    --gradient-accumulation-steps 2
    --workers 2
  )

  if [[ -f "$checkpoint_root/last/checkpoint/mp_rank_00_model_states.pt" ]]; then
    args+=(--resume "$checkpoint_root/last")
  else
    args+=(--initial-checkpoint "$initial_checkpoint")
  fi

  set +e
  "$env_bin/deepspeed" "${args[@]}" >>"$log_dir/l2_train.log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    exit 0
  fi
  printf '%s L2 exited rc=%s; retrying from latest epoch checkpoint if present\n' \
    "$(date -u +%FT%TZ)" "$rc" >>"$log_dir/l2_supervisor.log"
  sleep 20
done
