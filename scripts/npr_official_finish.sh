#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
PY=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=outputs/npr_official_retrain
trap 'code=$?; if (( code != 0 )); then printf "{\"status\":\"FAILED\",\"exit_code\":%d}\n" "$code" > "$OUT/pipeline_status.json"; fi' EXIT
npr_wait_started=$(date +%s)

while [[ ! -f "$OUT/checkpoint_epoch50.pt" || ! -f "$OUT/checkpoint_identity.json" ]]; do
  if (( $(date +%s) - npr_wait_started > 14400 )); then
    echo 'TRAINING_TIMEOUT_BEFORE_FINAL_CHECKPOINT' >&2
    exit 1
  fi
  if ! pgrep -f '[n]pr_official_retrain.py train' >/dev/null; then
    echo 'TRAINING_EXITED_WITHOUT_FINAL_CHECKPOINT' >&2
    exit 1
  fi
  sleep 30
done

CUDA_VISIBLE_DEVICES=1 "$PY" scripts/npr_official_retrain.py eval \
  --device cuda:0 --datasets validation internal_test loki --workers 8 --batch-size 64

CUDA_VISIBLE_DEVICES=1 "$PY" scripts/npr_official_retrain.py eval \
  --device cuda:0 --datasets aigi_holmes --workers 8 --batch-size 64 &
left=$!
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/npr_official_retrain.py eval \
  --device cuda:0 --datasets genimage --workers 8 --batch-size 64 &
right=$!
wait "$left"
wait "$right"

CUDA_VISIBLE_DEVICES=1 "$PY" scripts/npr_official_retrain.py eval \
  --device cuda:0 --datasets raise998 --workers 4 --batch-size 16
"$PY" scripts/npr_official_retrain.py finalize
"$PY" scripts/npr_official_publish.py
printf '{"status":"COMPLETE"}\n' > "$OUT/pipeline_status.json"
echo COMPLETE_STOP
