#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yz/groundingLMM_official
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
OUT="$ROOT/outputs/phase3c1_spatial_probe"
LOG="$OUT/reports/automatic_continuation.log"
export GLAMM_PRESERVE_CUDA_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$ROOT"
mkdir -p "$OUT/reports"

echo "[$(date -Is)] waiting for five-source train/val cache" >>"$LOG"
while true; do
  ready=1
  for source in sam clip npr srm focal; do
    for split in train val; do
      marker="$OUT/cache/$source/$split/complete.json"
      if [[ ! -f "$marker" ]]; then ready=0; fi
    done
  done
  if [[ "$ready" -eq 1 ]]; then break; fi
  sleep 30
done
while pgrep -f '[p]hase3c1_cache.py' >/dev/null; do sleep 10; done

"$PYTHON" - <<'PY' >>"$LOG" 2>&1
import json
from pathlib import Path
root=Path('outputs/phase3c1_spatial_probe/cache')
for source in ('sam','clip','npr','srm','focal'):
    for split, expected in (('train',8836),('val',1106)):
        value=json.loads((root/source/split/'complete.json').read_text())
        assert value['status']=='COMPLETE', value
        assert value['samples']==expected, value
        assert value['source_parameter_hash_exact'], value
print('all cache completion gates PASS')
PY

echo "[$(date -Is)] P1 G0 one-sample exact recheck" >>"$LOG"
"$PYTHON" scripts/phase3c0_diagnose.py \
  --output-dir outputs/phase3c1_spatial_probe/audit/p1_g0_recheck \
  --device cuda:1 --max-samples 1 --skip-model-hash --reset >>"$LOG" 2>&1
"$PYTHON" - <<'PY' >>"$LOG" 2>&1
import json, torch
from pathlib import Path
old_path=Path('outputs/phase3c0_residual_diagnosis/paired/paired_conditions.jsonl')
new_path=Path('outputs/phase3c1_spatial_probe/audit/p1_g0_recheck/paired/paired_conditions.jsonl')
old=json.loads(next(line for line in old_path.read_text().splitlines() if line))
new=json.loads(next(line for line in new_path.read_text().splitlines() if line))
result={
 'sample_id_exact': old['sample_id']==new['sample_id'],
 'generated_token_ids_sha256_exact': old['A']['generated_token_ids_sha256']==new['A']['generated_token_ids_sha256'],
 'full_input_token_ids_sha256_exact': old['A']['full_input_token_ids_sha256']==new['A']['full_input_token_ids_sha256'],
 'binary_mask_exact': bool(torch.equal(torch.load(old['A']['binary_mask_path'],map_location='cpu'),torch.load(new['A']['binary_mask_path'],map_location='cpu'))),
 'trace_vs_canonical_binary_exact': bool(new['A']['trace_vs_canonical_binary_exact']),
}
result['all_exact']=all(result.values())
Path('outputs/phase3c1_spatial_probe/audit/p1_g0_output_invariance.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['all_exact'], result
print(result)
PY

for source in sam clip npr srm focal; do
  echo "[$(date -Is)] train $source" >>"$LOG"
  "$PYTHON" scripts/phase3c1_probe.py train --source "$source" --device cuda:1 >>"$OUT/reports/train_${source}.log" 2>&1
  echo "[$(date -Is)] evaluate val controls $source" >>"$LOG"
  "$PYTHON" scripts/phase3c1_probe.py evaluate --source "$source" --split val --device cuda:1 >>"$OUT/reports/evaluate_${source}.log" 2>&1
done

echo "[$(date -Is)] freeze val route gate" >>"$LOG"
"$PYTHON" scripts/phase3c1_analyze.py >>"$LOG" 2>&1

mapfile -t confirmatory_sources < <("$PYTHON" - <<'PY'
import json
for value in json.load(open('outputs/phase3c1_spatial_probe/route_gate.json'))['confirmatory_sources_authorized_after_gate_freeze']:
    print(value)
PY
)
for source in "${confirmatory_sources[@]}"; do
  for split in test official1000; do
    echo "[$(date -Is)] confirmatory cache/evaluate $source $split" >>"$LOG"
    "$PYTHON" scripts/phase3c1_cache.py cache --source "$source" --split "$split" --device cuda:1 >>"$OUT/reports/cache_${source}_${split}.log" 2>&1
    "$PYTHON" scripts/phase3c1_probe.py evaluate --source "$source" --split "$split" --device cuda:1 >>"$OUT/reports/confirmatory_${source}_${split}.log" 2>&1
  done
done

echo "[$(date -Is)] qualitative" >>"$LOG"
"$PYTHON" scripts/phase3c1_qualitative.py >>"$LOG" 2>&1
echo "[$(date -Is)] full regression" >>"$LOG"
"$PYTHON" -m pytest -q tests >"$OUT/reports/regression_tests.txt" 2>&1
echo "[$(date -Is)] finalize" >>"$LOG"
"$PYTHON" scripts/phase3c1_finalize.py >>"$LOG" 2>&1
echo "[$(date -Is)] COMPLETE" >>"$LOG"
