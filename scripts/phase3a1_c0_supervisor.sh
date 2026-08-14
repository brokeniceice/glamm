#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
OUT=outputs/phase3a1_paired_control/c0_training
METRICS="$OUT/run/metrics.jsonl"
STATE="$OUT/run/training_state.json"
LOG="$OUT/logs/supervisor.log"
ENV_PY=/home/yz/miniconda3/envs/glamm_official/bin/python

mkdir -p "$OUT/logs/archive"
echo "$(date -u +%FT%TZ) supervisor started" >>"$LOG"

while true; do
  rows=0
  [[ -f "$METRICS" ]] && rows=$(wc -l <"$METRICS")
  if (( rows >= 5000 )); then
    if (( rows > 5000 )); then
      echo "$(date -u +%FT%TZ) ERROR metrics has $rows rows" >>"$LOG"
      exit 1
    fi
    if ! pgrep -f 'scripts/phase2a_distributed_train.py.*configs/phase3a1_paired_c0.yaml.*--mode train' >/dev/null; then
      echo "$(date -u +%FT%TZ) training durably completed at step 5000" >>"$LOG"
      exit 0
    fi
    sleep 30
    continue
  fi

  if pgrep -f 'scripts/phase2a_distributed_train.py.*configs/phase3a1_paired_c0.yaml.*--mode train' >/dev/null; then
    sleep 30
    continue
  fi

  if [[ ! -f "$STATE" ]]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    cp "$METRICS" "$OUT/logs/archive/metrics_before_fresh_restart_${stamp}.jsonl"
    : >"$METRICS"
    echo "$(date -u +%FT%TZ) trainer exited before first durable checkpoint; archived rows and restarting the exact step-0 trajectory" >>"$LOG"
    if ! scripts/run_phase3a1_c0_worker.sh; then
      echo "$(date -u +%FT%TZ) fresh trainer restart exited nonzero" >>"$LOG"
    fi
    continue
  fi

  durable=$($ENV_PY -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["optimizer_step"]))' "$STATE")
  if (( durable <= 0 || durable >= 5000 || durable % 500 != 0 )); then
    echo "$(date -u +%FT%TZ) ERROR invalid durable checkpoint step=$durable" >>"$LOG"
    exit 1
  fi

  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  cp "$METRICS" "$OUT/logs/archive/metrics_before_resume_${stamp}.jsonl"
  "$ENV_PY" - "$METRICS" "$durable" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
durable = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
kept = [row for row in rows if int(row["optimizer_step"]) <= durable]
steps = [int(row["optimizer_step"]) for row in kept]
if steps != list(range(1, durable + 1)):
    raise SystemExit(f"canonical metrics are not contiguous through durable step {durable}")
path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
PY
  echo "$(date -u +%FT%TZ) resuming from durable step $durable; trailing rows archived" >>"$LOG"
  if ! scripts/run_phase3a1_c0_resume_worker.sh; then
    echo "$(date -u +%FT%TZ) resumed trainer exited nonzero" >>"$LOG"
  fi
done
