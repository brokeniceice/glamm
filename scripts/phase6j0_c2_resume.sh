#!/usr/bin/env bash
set -euo pipefail
cd /home/yz/groundingLMM_official

PY=/home/yz/miniconda3/envs/glamm_official/bin/python
DS=/home/yz/miniconda3/envs/glamm_official/bin/deepspeed
OUT=outputs/phase6j0_c2

"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('outputs/phase6j0_c2')
pre=root/'training/preflight/resume_mb10'
audit=json.loads((pre/'resume_audit.json').read_text())
summary=json.loads((pre/'run_summary.json').read_text())
validation=json.loads((pre/'c2_validation_preflight.json').read_text())
if audit['status']!='passed' or audit['optimizer_step']!=3500:
    raise SystemExit('Checkpoint resume audit did not pass at step 3500')
if summary['optimizer_steps']!=1 or not summary['all_parameter_groups_synchronized']:
    raise SystemExit('Resumed optimizer-step smoke did not pass')
if validation['status']!='PASS':
    raise SystemExit('Resumed validation preflight did not pass')
marker=root/'recovery/resume_started.json'
if marker.exists():
    raise SystemExit('Resume has already been launched')
for name in ('metrics.jsonl','c2_attention_diagnostics.jsonl'):
    path=root/'training'/name
    original=[json.loads(x) for x in path.read_text().splitlines() if x]
    if not original or original[-1]['optimizer_step'] != (3741 if name=='metrics.jsonl' else 3725):
        raise SystemExit(f'Unexpected failed-run tail in {name}')
    retained=[x for x in original if x['optimizer_step']<=3500]
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in retained))
marker.write_text(json.dumps({'resume_from_step':3500,'failed_last_logged_step':3741,
                              'micro_batch_per_gpu':10,'gradient_accumulation_steps':1,
                              'effective_global_batch':20},indent=2)+'\n')
PY

setsid "$DS" --include localhost:1,2 --master_port 29646 \
  scripts/phase6j0_c2_train.py \
  --config configs/phase6j0_c2_preln_cross_attention.yaml \
  --mode train --workers 2 \
  --resume checkpoints/phase6j0_c2/last \
  --rine-conditioned-c1 \
  --rine-checkpoint outputs/phase6b6_rine_training/selected_checkpoint.pt \
  > "$OUT/training/train_resume_mb10.log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$OUT/training/train.pid"
printf 'C2 resume started: train PID %s\n' "$train_pid"
