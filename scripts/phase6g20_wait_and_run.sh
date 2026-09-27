#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yz/groundingLMM_official
PY=/home/yz/miniconda3/envs/glamm_official/bin/python
G19="$ROOT/outputs/phase6g19_r1_c_vs_srect/results.json"
while [[ ! -s "$G19" ]]; do sleep 30; done
cd "$ROOT"
exec "$PY" scripts/phase6g20_evidence_correction_interaction.py --device cuda:1
