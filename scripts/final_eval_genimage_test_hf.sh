#!/usr/bin/env bash
set -u

revision="71c983e6262684bc2c6b6af99582e8f568c259a5"
expected_size="25408572820"
expected_sha256="1ee98d0958a5905b4e1b2f7f44bb384069704cad8f70e741512e1911e29dba97"
target_dir="/data/yz/myLISA_storage/AIGC/GenImage/raw/huggingface_test"
target="${target_dir}/genimage_test.zip"
status="/home/yz/groundingLMM_official/outputs/final_eval_datasets/downloads/genimage_test_hf_status.json"
url="https://huggingface.co/datasets/jzousz/GenImage/resolve/${revision}/genimage_test.zip?download=true"

mkdir -p "$target_dir"

write_status() {
  local state="$1"
  local actual_size="0"
  if [[ -f "$target" ]]; then actual_size="$(stat -c %s "$target")"; fi
  /home/yz/miniconda3/bin/python - "$status" "$state" "$actual_size" "$expected_size" "$revision" "$expected_sha256" <<'PY'
import json, sys
from datetime import datetime, timezone
path, state, actual, expected, revision, sha256 = sys.argv[1:]
actual, expected = int(actual), int(expected)
payload = {
    "schema": "genimage_hf_test_download_v1",
    "status": state,
    "repo_id": "jzousz/GenImage",
    "revision": revision,
    "file": "genimage_test.zip",
    "expected_bytes": expected,
    "downloaded_bytes": actual,
    "percent": round(100.0 * actual / expected, 4),
    "expected_sha256": sha256,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

write_status RUNNING
while true; do
  curl --location --fail --continue-at - \
    --retry 30 --retry-delay 30 \
    --connect-timeout 30 --speed-time 300 --speed-limit 1048576 \
    --output "$target" "$url" && break
  write_status RETRYING
  sleep 300
done

actual_size="$(stat -c %s "$target")"
if [[ "$actual_size" != "$expected_size" ]]; then
  write_status SIZE_MISMATCH
  exit 1
fi

actual_sha256="$(sha256sum "$target" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  write_status SHA256_MISMATCH
  exit 1
fi

write_status COMPLETE
