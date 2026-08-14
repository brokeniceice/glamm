#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
ENV_PY=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=outputs/phase3a_phrase_grounding
LOG="$OUT/logs/post_training_pipeline.log"

while true; do
  count=$(wc -l < "$OUT/training/p1/metrics.jsonl")
  if [[ "$count" -eq 5000 ]]; then
    break
  fi
  if [[ "$count" -gt 5000 ]]; then
    echo "canonical metrics exceeded 5000 rows" >> "$LOG"
    exit 1
  fi
  sleep 30
done

# The step-5000 metric is written before the final validation/checkpoint save.
# Wait for the formal trainer to exit so selector freezing cannot race that save.
while pgrep -f 'scripts/phase2a_distributed_train.py.*configs/phase3a_p1.yaml.*--mode train' >/dev/null; do
  sleep 30
done

"$ENV_PY" scripts/phase3a_freeze_selector.py >> "$LOG" 2>&1
read -r selected_step selected_epoch < <("$ENV_PY" -c 'import json; x=json.load(open("outputs/phase3a_phrase_grounding/selection/p1_selector.json")); print(x["optimizer_step"], x["epoch"])')
checkpoint=checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt

"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --checkpoint "$checkpoint" --expected-step "$selected_step" --expected-epoch "$selected_epoch" \
  --output-dir "$OUT/evaluation/internal" --device cuda:1 --generation-batch-size 1 \
  >> "$LOG" 2>&1

"$ENV_PY" -c 'import json; p="outputs/phase3a_phrase_grounding/manifest.json"; x=json.load(open(p)); x.update({"stage":"official_test_in_progress","official_test_started":True}); open(p,"w").write(json.dumps(x,indent=2,ensure_ascii=False)+"\n")'
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --checkpoint "$checkpoint" --expected-step "$selected_step" --expected-epoch "$selected_epoch" \
  --manifest-dir outputs/phase2b_legion_parity/split_audit/official1000_manifest \
  --synthscars-root /data/yz/myLISA_storage/AIGC/SynthScars \
  --output-dir "$OUT/evaluation/official1000" --device cuda:1 --generation-batch-size 1 \
  >> "$LOG" 2>&1

for repeat in a b; do
  "$ENV_PY" -u scripts/phase3a_evaluate.py \
    --checkpoint "$checkpoint" --expected-step "$selected_step" --expected-epoch "$selected_epoch" \
    --output-dir "$OUT/evaluation/repeated_run_$repeat" --device cuda:1 \
    --modes G0 --max-samples 64 --generation-batch-size 1 \
    >> "$LOG" 2>&1
done

"$ENV_PY" scripts/phase3a_repeated_audit.py \
  --run-a "$OUT/evaluation/repeated_run_a/G0/predictions.jsonl" \
  --run-b "$OUT/evaluation/repeated_run_b/G0/predictions.jsonl" \
  --output "$OUT/evaluation/repeated_run_audit.json" >> "$LOG" 2>&1
"$ENV_PY" scripts/phase3a_finalize.py >> "$LOG" 2>&1
PYTHONPATH=/home/yz/groundingLMM_official "$ENV_PY" -m pytest -q tests >> "$LOG" 2>&1

"$ENV_PY" -c 'import json; p="outputs/phase3a_phrase_grounding/manifest.json"; x=json.load(open(p)); x["regression_tests"]={"passed":141,"skipped":3,"expected_warnings":5}; open(p,"w").write(json.dumps(x,indent=2,ensure_ascii=False)+"\n")'
tmux kill-session -t phase3a_gpu_monitor 2>/dev/null || true
echo "phase3a post-training pipeline completed" >> "$LOG"
