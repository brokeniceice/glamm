#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
DEEPSPEED=/home/yz/miniconda3/envs/glamm_official/bin/deepspeed
OUT="$ROOT/outputs/phase3b_generated_replay"
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export PHASE3B_PHYSICAL_GPU=0
START_STAGE="${PHASE3B_START_STAGE:-cache}"
cd "$ROOT"

mkdir -p "$OUT/logs" "$OUT/audit/preflight" "$OUT/reports"
echo "PIPELINE_START $(date -u +%FT%TZ) pid=$$ physical_gpu=0 start_stage=$START_STAGE" >"$OUT/logs/detached_pipeline_state.log"

stage() {
  echo "$1 $(date -u +%FT%TZ)" | tee -a "$OUT/logs/detached_pipeline_state.log"
}

if [[ "$START_STAGE" == "cache" ]]; then
  stage CACHE_BATCH1_RESUME
  for SHARD in 1 2 0 3 4 5; do
    "$PYTHON" scripts/phase3b_build_replay_cache.py \
      --device cuda:0 --num-shards 6 --shard-index "$SHARD" --batch-size 1 \
      >>"$OUT/cache/shard_${SHARD}.log" 2>&1
  done
  "$PYTHON" scripts/phase3b_build_replay_cache.py \
    --merge --num-shards 6 --batch-size 1 >"$OUT/cache/merge.log" 2>&1

  stage CACHE_AUDIT
  "$PYTHON" scripts/phase3b_verify_cache_batch_identity.py \
    --device cuda:0 --samples 12 >"$OUT/audit/cache_batch_identity.log" 2>&1
  "$PYTHON" scripts/phase3b_audit.py >"$OUT/audit/preregistration_audit.log" 2>&1
elif [[ "$START_STAGE" == "b1_preflight" ]]; then
  stage REUSE_FROZEN_CACHE_AND_PASSED_AUDITS
  "$PYTHON" scripts/phase3b_audit.py >"$OUT/audit/preregistration_audit.log" 2>&1
else
  echo "Unsupported PHASE3B_START_STAGE=$START_STAGE" >&2
  exit 2
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path("outputs/phase3b_generated_replay")
s=json.loads((root/"cache/cache_stats.json").read_text())
assert s["status"]=="FROZEN" and s["num_train_fake"]==8836
assert s["generation_batch_size"]==1 and s["canonical_historical_batch_size_identity"] is True
a=json.loads((root/"audit/cache_batch_identity.json").read_text())
assert a["status"]=="PASS" and a["num_checked"]==12
p=json.loads((root/"audit/preregistration_audit.json").read_text())
assert p["status"]=="PASS"
PY

stage B1_PREFLIGHT
"$DEEPSPEED" --master_port 29531 scripts/phase3b_train.py \
  --config configs/phase3b_b1_generated_replay.yaml --workers 8 --optimizer-steps 1 \
  >"$OUT/logs/b1_preflight.log" 2>&1
cp "$OUT/training/b1_generated_replay/source_initialization_audit.json" "$OUT/audit/preflight/b1_source_initialization.json"
cp "$OUT/training/b1_generated_replay/metrics.jsonl" "$OUT/audit/preflight/b1_metrics.jsonl"
cp "$OUT/training/b1_generated_replay/replay_schedule.jsonl" "$OUT/audit/preflight/b1_schedule.jsonl"
cp "$OUT/training/b1_generated_replay/run_summary.json" "$OUT/audit/preflight/b1_run_summary.json"
"$PYTHON" - <<'PY'
import json,math
from pathlib import Path
p=Path("outputs/phase3b_generated_replay/audit/preflight")
r=json.loads((p/"b1_metrics.jsonl").read_text().strip())
assert r["optimizer_step"]==1 and r["selected_replay_count"]>0
assert r["replay"]["ce_loss"]==0.0 and r["replay"]["cls_loss"]==0.0
assert math.isfinite(r["replay"]["mask_bce_loss"]) and r["replay"]["mask_bce_loss"]>0
assert math.isfinite(r["replay"]["mask_dice_loss"]) and r["replay"]["mask_dice_loss"]>0
PY

stage B1_FORMAL_TRAIN
"$DEEPSPEED" --master_port 29532 scripts/phase3b_train.py \
  --config configs/phase3b_b1_generated_replay.yaml --workers 8 \
  >"$OUT/logs/b1_formal_training.log" 2>&1

stage B0_PREFLIGHT
"$DEEPSPEED" --master_port 29533 scripts/phase3b_train.py \
  --config configs/phase3b_b0_gold_replay.yaml --workers 8 --optimizer-steps 1 \
  >"$OUT/logs/b0_preflight.log" 2>&1
cp "$OUT/training/b0_gold_replay/source_initialization_audit.json" "$OUT/audit/preflight/b0_source_initialization.json"
cp "$OUT/training/b0_gold_replay/metrics.jsonl" "$OUT/audit/preflight/b0_metrics.jsonl"
cp "$OUT/training/b0_gold_replay/replay_schedule.jsonl" "$OUT/audit/preflight/b0_schedule.jsonl"
cp "$OUT/training/b0_gold_replay/run_summary.json" "$OUT/audit/preflight/b0_run_summary.json"
"$PYTHON" - <<'PY'
import json,math
from pathlib import Path
p=Path("outputs/phase3b_generated_replay/audit/preflight")
b1=json.loads((p/"b1_source_initialization.json").read_text())
b0=json.loads((p/"b0_source_initialization.json").read_text())
assert b1["trainable_state"]==b0["trainable_state"]
s1=json.loads((p/"b1_schedule.jsonl").read_text().strip())
s0=json.loads((p/"b0_schedule.jsonl").read_text().strip())
assert s1==s0
r=json.loads((p/"b0_metrics.jsonl").read_text().strip())
assert r["optimizer_step"]==1 and r["selected_replay_count"]>0
assert r["replay"]["ce_loss"]==0.0 and r["replay"]["cls_loss"]==0.0
assert math.isfinite(r["replay"]["mask_bce_loss"]) and r["replay"]["mask_bce_loss"]>0
assert math.isfinite(r["replay"]["mask_dice_loss"]) and r["replay"]["mask_dice_loss"]>0
PY

stage B0_FORMAL_TRAIN
"$DEEPSPEED" --master_port 29534 scripts/phase3b_train.py \
  --config configs/phase3b_b0_gold_replay.yaml --workers 8 \
  >"$OUT/logs/b0_formal_training.log" 2>&1

stage VALIDATION_SELECTORS
"$PYTHON" scripts/phase3b_validate_and_select.py \
  --config configs/phase3b_b1_generated_replay.yaml --device cuda:0 --generation-batch-size 1 \
  >"$OUT/logs/b1_validation_selector.log" 2>&1
"$PYTHON" scripts/phase3b_validate_and_select.py \
  --config configs/phase3b_b0_gold_replay.yaml --device cuda:0 --generation-batch-size 1 \
  >"$OUT/logs/b0_validation_selector.log" 2>&1

stage FINAL_EVALUATION_B1_THEN_B0
"$PYTHON" scripts/phase3b_final_evaluate.py \
  --arm b1_generated_replay --device cuda:0 --generation-batch-size 1 \
  >"$OUT/logs/b1_final_evaluation.log" 2>&1
"$PYTHON" scripts/phase3b_final_evaluate.py \
  --arm b0_gold_replay --device cuda:0 --generation-batch-size 1 \
  >"$OUT/logs/b0_final_evaluation.log" 2>&1

stage ANALYSIS_AND_REGRESSION
"$PYTHON" scripts/phase3b_analyze.py >"$OUT/logs/final_analysis.log" 2>&1
"$PYTHON" -m pytest -q tests >"$OUT/reports/regression_tests.txt" 2>&1
"$PYTHON" - <<'PY'
import json,re
from pathlib import Path
root=Path("outputs/phase3b_generated_replay")
manifest=json.loads((root/"manifest.json").read_text())
text=(root/"reports/regression_tests.txt").read_text()
m=re.search(r"(\d+) passed, (\d+) skipped, (\d+) warnings",text)
assert m, text[-1000:]
manifest.update({"stage":"COMPLETE","regression_tests":{"passed":int(m.group(1)),"skipped":int(m.group(2)),"warnings":int(m.group(3))},
                 "detached_pipeline_complete":True})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
PY
stage COMPLETE
