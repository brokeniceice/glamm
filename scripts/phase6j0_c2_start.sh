#!/usr/bin/env bash
set -euo pipefail
cd /home/yz/groundingLMM_official

PY=/home/yz/miniconda3/envs/glamm_official/bin/python
DS=/home/yz/miniconda3/envs/glamm_official/bin/deepspeed
OUT=outputs/phase6j0_c2/training

"$PY" - <<'PY'
import json
from pathlib import Path
p=Path('outputs/phase6j0_c2/training/preflight/dual')
validation=json.loads((p/'c2_validation_preflight.json').read_text())
train=json.loads((p/'run_summary.json').read_text())
checkpoint=Path('checkpoints/phase6j0_c2/preflight/last/checkpoint/mp_rank_00_model_states.pt')
if validation['status'] != 'PASS' or train['optimizer_steps'] != 1 or not train['all_parameter_groups_synchronized'] or not checkpoint.is_file():
    raise SystemExit('C2 train/validation preflight gate did not pass')
PY

mkdir -p "$OUT"
setsid "$DS" --include localhost:1,2 --master_port 29642 \
  scripts/phase6j0_c2_train.py \
  --config configs/phase6j0_c2_preln_cross_attention.yaml \
  --mode train --workers 4 \
  --rine-conditioned-c1 \
  --rine-checkpoint outputs/phase6b6_rine_training/selected_checkpoint.pt \
  > "$OUT/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/train.pid"
printf 'C2 formal training started, supervisor PID %s\n' "$pid"
