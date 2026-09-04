#!/usr/bin/env bash
set -euo pipefail

while systemctl --user is-active --quiet phase5a3-glamm-download.service; do
  sleep 30
done
exec /usr/bin/env PYTHONDONTWRITEBYTECODE=1 \
  /home/yz/miniconda3/envs/legion/bin/python \
  /home/yz/groundingLMM_official/scripts/phase5a3_pipeline.py
