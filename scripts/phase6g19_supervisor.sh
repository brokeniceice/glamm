#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yz/groundingLMM_official
PY=/home/yz/miniconda3/envs/glamm_official/bin/python
cd "$ROOT"
"$PY" scripts/phase6g19_r1_c_vs_srect.py --arm A1_C --device cuda:1
"$PY" scripts/phase6g19_r1_c_vs_srect.py --arm A2_Srect --device cuda:1
"$PY" scripts/phase6g19_r1_c_vs_srect.py --finalize
