#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/yz/groundingLMM_official
env_python=/home/yz/miniconda3/envs/glamm_official/bin/python
summary="$repo_root/outputs/phase3d2_direct_spatial_path/training/run_summary.json"
cd "$repo_root"

is_complete() {
  "$env_python" - "$summary" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.load(p.open()).get("status")=="COMPLETE" else 1)
PY
}

while ! is_complete; do
  if ! pgrep -f 'scripts/phase3d2_train.py .*--run-kind official' >/dev/null; then
    echo "Phase 3D.2 training stopped before completion" >&2
    exit 1
  fi
  sleep 15
done

exec "$env_python" scripts/phase3d2_after_training.py
