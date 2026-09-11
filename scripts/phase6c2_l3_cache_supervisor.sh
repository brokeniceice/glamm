#!/usr/bin/env bash
set -euo pipefail

repo=/home/yz/groundingLMM_official
env_bin=/home/yz/miniconda3/envs/glamm_official/bin
log_dir="$repo/outputs/phase6c2_multiseg_training/logs"
audit_dir="$repo/outputs/phase6c2_multiseg_training/l3_cache"
cache_dir=/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l3_l2_sources
mkdir -p "$log_dir" "$audit_dir" "$cache_dir"
cd "$repo"

export CUDA_VISIBLE_DEVICES=0,1
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

complete() {
  "$env_bin/python" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path('outputs/phase6c2_multiseg_training/l3_cache')/f'{sys.argv[1]}_complete.json'
raise SystemExit(0 if p.is_file() and json.load(p.open()).get('status') == 'COMPLETE' else 1)
PY
}

run_split() {
  local split=$1
  if complete "$split"; then
    return 0
  fi
  "$env_bin/torchrun" --standalone --nproc_per_node=2 \
    scripts/phase6c2_l3_cache.py --split "$split" --batch-size 2 --workers 2 \
    --images-per-shard 64 --resume --output-root "$cache_dir" --audit-root "$audit_dir" \
    >>"$log_dir/l3_cache_${split}.log" 2>&1
}

while true; do
  set +e
  run_split train && run_split val
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    exit 0
  fi
  printf '%s L3 cache exited rc=%s; retrying verified rank shards\n' \
    "$(date -u +%FT%TZ)" "$rc" >>"$log_dir/l3_cache_supervisor.log"
  sleep 20
done
