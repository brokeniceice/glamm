#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yz/myLISA_storage/AIGC/GenImage/raw/archives
PYTHON=/home/yz/miniconda3/envs/glamm_official/bin/python
export PYTHONPATH=/data/yz/myLISA_storage/AIGC/.tools/gdown

# ADM is downloaded by a dedicated resumable unit started before this queue.
while systemctl --user is-active --quiet finaldata-genimage2-ADM.service; do
  sleep 30
done
test -f "$ROOT/ADM/imagenet_ai_0508_adm.zip"

download_folder() {
  local name="$1"
  local id="$2"
  mkdir -p "$ROOT/$name"
  "$PYTHON" -m gdown --folder --continue -O "$ROOT/$name" \
    "https://drive.google.com/drive/folders/$id"
}

download_folder BigGAN 1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt
download_folder glide 1H2_4VPlla4OuKYU2CbMYx-8X0nbGJ2Mc
download_folder Midjourney 1GLZedAqYuBh0kIQCZcbY_RspPJsR7ZyE
download_folder sdv14 12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy
download_folder sdv15 1lG4WheCh3_CM2XNrRwfXVeOiG_1gZ_K2
download_folder VQDM 1-KviCgiBrBpm4e-TTdGk2RaGyV-69SaN
download_folder wukong 1h7685o7i6FNJ1wVtLtbwEWndW25YoaJk

printf 'COMPLETE\n' > "$ROOT/.download_complete"
