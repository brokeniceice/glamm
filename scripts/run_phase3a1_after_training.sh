#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
export PYTHONPATH=/home/yz/groundingLMM_official
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

ENV_PY=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT=outputs/phase3a1_paired_control
RUN="$OUT/c0_training/run"
LOG="$OUT/c0_training/logs/post_training_pipeline.log"
OFFICIAL_MANIFEST=outputs/phase2b_legion_parity/split_audit/official1000_manifest
SYNTH_ROOT=/data/yz/myLISA_storage/AIGC/SynthScars
HIST_CKPT=checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt
P1_CKPT=checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt

touch "$LOG"
while true; do
  count=0
  [[ -f "$RUN/metrics.jsonl" ]] && count=$(wc -l <"$RUN/metrics.jsonl")
  if (( count > 5000 )); then
    echo "$(date -u +%FT%TZ) ERROR canonical metrics exceeded 5000" >>"$LOG"
    exit 1
  fi
  if (( count == 5000 )) && ! pgrep -f 'scripts/phase2a_distributed_train.py.*configs/phase3a1_paired_c0.yaml.*--mode train' >/dev/null; then
    break
  fi
  sleep 30
done

echo "$(date -u +%FT%TZ) freezing new C0 selector" >>"$LOG"
"$ENV_PY" scripts/phase3a1_freeze_selector.py >>"$LOG" 2>&1
read -r c0_step c0_epoch c0_checkpoint < <("$ENV_PY" -c 'import json; x=json.load(open("outputs/phase3a1_paired_control/selection/new_c0_selector.json")); print(x["optimizer_step"],x["epoch"],x["selected_checkpoint"])')

"$ENV_PY" - <<'PY'
import json
p="outputs/phase3a1_paired_control/manifest.json"
x=json.load(open(p)); x.update({"stage":"OFFICIAL_G0_IN_PROGRESS","official_test_started":True})
open(p,"w").write(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
PY

echo "$(date -u +%FT%TZ) official1000 G0 new C0" >>"$LOG"
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a1_paired_c0.yaml \
  --checkpoint "$c0_checkpoint" --expected-step "$c0_step" --expected-epoch "$c0_epoch" \
  --manifest-dir "$OFFICIAL_MANIFEST" --synthscars-root "$SYNTH_ROOT" \
  --output-dir "$OUT/evaluation/g0/new_c0" --device cuda:0 \
  --modes G0 --generation-batch-size 1 >>"$LOG" 2>&1

echo "$(date -u +%FT%TZ) paired G0 statistics and fixed severe set" >>"$LOG"
"$ENV_PY" scripts/phase3a1_analyze.py --stage g0 >>"$LOG" 2>&1

echo "$(date -u +%FT%TZ) internal classification new C0" >>"$LOG"
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a1_paired_c0.yaml \
  --checkpoint "$c0_checkpoint" --expected-step "$c0_step" --expected-epoch "$c0_epoch" \
  --output-dir "$OUT/evaluation/internal/new_c0" --device cuda:0 \
  --modes detection --generation-batch-size 1 >>"$LOG" 2>&1
"$ENV_PY" scripts/phase3a1_analyze.py --stage classification >>"$LOG" 2>&1

echo "$(date -u +%FT%TZ) TF 2x2 cross-evaluation" >>"$LOG"
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a1_paired_c0.yaml --checkpoint "$HIST_CKPT" \
  --expected-step 2500 --expected-epoch 5 --manifest-dir "$OFFICIAL_MANIFEST" \
  --synthscars-root "$SYNTH_ROOT" --output-dir "$OUT/evaluation/tf_cross_eval/A_historical_tf_old" \
  --device cuda:0 --modes tf_full_context >>"$LOG" 2>&1
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a_p1.yaml --checkpoint "$HIST_CKPT" \
  --expected-step 2500 --expected-epoch 5 --manifest-dir "$OFFICIAL_MANIFEST" \
  --synthscars-root "$SYNTH_ROOT" --output-dir "$OUT/evaluation/tf_cross_eval/B_historical_tf_phrase" \
  --device cuda:0 --modes tf_full_context >>"$LOG" 2>&1
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a1_paired_c0.yaml --checkpoint "$P1_CKPT" \
  --expected-step 3500 --expected-epoch 7 --manifest-dir "$OFFICIAL_MANIFEST" \
  --synthscars-root "$SYNTH_ROOT" --output-dir "$OUT/evaluation/tf_cross_eval/C_p1_tf_old" \
  --device cuda:0 --modes tf_full_context >>"$LOG" 2>&1
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a_p1.yaml --checkpoint "$P1_CKPT" \
  --expected-step 3500 --expected-epoch 7 --manifest-dir "$OFFICIAL_MANIFEST" \
  --synthscars-root "$SYNTH_ROOT" --output-dir "$OUT/evaluation/tf_cross_eval/D_p1_tf_phrase" \
  --device cuda:0 --modes tf_full_context >>"$LOG" 2>&1
"$ENV_PY" -u scripts/phase3a_evaluate.py \
  --config configs/phase3a1_paired_c0.yaml --checkpoint "$c0_checkpoint" \
  --expected-step "$c0_step" --expected-epoch "$c0_epoch" --manifest-dir "$OFFICIAL_MANIFEST" \
  --synthscars-root "$SYNTH_ROOT" --output-dir "$OUT/evaluation/tf_cross_eval/E_new_c0_tf_old" \
  --device cuda:0 --modes tf_full_context >>"$LOG" 2>&1

"$ENV_PY" scripts/phase3a1_analyze.py --stage final >>"$LOG" 2>&1

echo "$(date -u +%FT%TZ) full regression" >>"$LOG"
set +e
PYTHONPATH=/home/yz/groundingLMM_official "$ENV_PY" -m pytest -q tests >"$OUT/reports/regression_tests.txt" 2>&1
test_status=$?
set -e
cat "$OUT/reports/regression_tests.txt" >>"$LOG"
"$ENV_PY" - "$test_status" <<'PY'
import json,sys
p="outputs/phase3a1_paired_control/manifest.json"
x=json.load(open(p)); x["regression_tests"]={"exit_code":int(sys.argv[1]),"log":"outputs/phase3a1_paired_control/reports/regression_tests.txt"}
if int(sys.argv[1]) != 0: x["stage"]="COMPLETE_WITH_REGRESSION_FAILURE"
open(p,"w").write(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
PY

tmux kill-session -t phase3a1_gpu_monitor 2>/dev/null || true
tmux kill-session -t phase3a1_c0_supervisor 2>/dev/null || true
echo "$(date -u +%FT%TZ) phase3a1 post-training pipeline completed; pytest_exit=$test_status" >>"$LOG"
exit "$test_status"
