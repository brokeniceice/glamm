#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/yz/groundingLMM_official
env_python=/home/yz/miniconda3/envs/glamm_official/bin/python
output_root="$repo_root/outputs/phase3d1_policy_optimization"
cd "$repo_root"

is_complete() {
  "$env_python" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.load(p.open()).get("status")=="COMPLETE" else 1)
PY
}

while ! is_complete "$output_root/experiments/R3_OPT/run_summary.json" || \
      ! is_complete "$output_root/experiments/Q2_OPT/run_summary.json"; do
  if ! pgrep -f 'scripts/phase3d1_train.py --arm (R3|Q2)' >/dev/null; then
    echo "Phase 3D.1 training stopped before both arms completed" >&2
    exit 1
  fi
  sleep 30
done

export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
(
  export CUDA_VISIBLE_DEVICES=1
  "$env_python" scripts/phase3d1_validate_select.py --arm R3 --physical-gpu 1 \
    >>"$output_root/logs/r3_selector.log" 2>&1
) &
r3_pid=$!
(
  export CUDA_VISIBLE_DEVICES=2
  "$env_python" scripts/phase3d1_validate_select.py --arm Q2 --physical-gpu 2 \
    >>"$output_root/logs/q2_selector.log" 2>&1
) &
q2_pid=$!
wait "$r3_pid" "$q2_pid"

(
  export CUDA_VISIBLE_DEVICES=1
  "$env_python" scripts/phase3d1_selected_rollouts.py --arm R3 --physical-gpu 1 \
    >>"$output_root/logs/r3_selected_rollouts.log" 2>&1
) &
r3_pid=$!
(
  export CUDA_VISIBLE_DEVICES=2
  "$env_python" scripts/phase3d1_selected_rollouts.py --arm Q2 --physical-gpu 2 \
    >>"$output_root/logs/q2_selected_rollouts.log" 2>&1
) &
q2_pid=$!
wait "$r3_pid" "$q2_pid"

"$env_python" scripts/phase3d1_finalize.py >>"$output_root/logs/finalize.log" 2>&1
