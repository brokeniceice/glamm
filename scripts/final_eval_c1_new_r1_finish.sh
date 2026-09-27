#!/usr/bin/env bash
set -euo pipefail

unit=codex-c1-new-r1-ood.service
root=/home/yz/groundingLMM_official
python=/home/yz/miniconda3/envs/glamm_official/bin/python

# This detached dependency guard stays quiet until the GPU supervisor exits.
while systemctl --user is-active --quiet "$unit"; do
    sleep 30
done

cd "$root"
# Finalize rechecks every worker marker, manifest, prediction hash, and metric.
# It can also recover a supervisor that finished inference but exited while
# writing a report under the superseded raw-C1 protocol.
exec "$python" scripts/final_eval_c1_new_r1.py --mode finalize
