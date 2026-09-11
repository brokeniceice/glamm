#!/usr/bin/env bash
set -uo pipefail
repo=/home/yz/groundingLMM_official
log="$repo/outputs/phase6c2_multiseg_training/logs/l2_epoch10_g0.log"
cache=/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l2_epoch10_val_g0
audit="$repo/outputs/phase6c2_multiseg_training/l2_epoch10_g0"
checkpoint=/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2/last/checkpoint/mp_rank_00_model_states.pt
cd "$repo" || exit 2
while [ ! -f "$audit/complete.json" ]; do
 CUDA_VISIBLE_DEVICES=0,1 /home/yz/miniconda3/envs/glamm_official/bin/torchrun --standalone --nproc_per_node=2 scripts/phase6c2_l2_g0_cache.py \
  --batch-size 2 --checkpoint "$checkpoint" --expected-epoch 10 --expected-step 5000 \
  --expected-sha256 9b9d52125a729fec6b8a77c7a9591c59ecf058303375d8633aa72155c36a3200 \
  --output-root "$cache" --audit-root "$audit" >>"$log" 2>&1
 code=$?; if [ "$code" -ne 0 ]; then printf '%s retry exit=%s\n' "$(date -u +%FT%TZ)" "$code" >>"$log";sleep 30;fi
done
/home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_l2_g0_summarize.py >>"$log" 2>&1
