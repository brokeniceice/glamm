#!/usr/bin/env bash
set -euo pipefail

DEST=/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/base/GLaMM-GranD-Pretrained
REV=a2513f97c9404065cfd5849325e61d5d53456441
BASE="https://huggingface.co/MBZUAI/GLaMM-GranD-Pretrained/resolve/$REV"

mkdir -p "$DEST"
files=(
  .gitattributes README.md added_tokens.json config.json generation_config.json
  pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin
  pytorch_model.bin.index.json special_tokens_map.json tokenizer.model tokenizer_config.json
)
for name in "${files[@]}"; do
  if [[ -f "$DEST/$name" ]]; then
    echo "present $name"
    continue
  fi
  echo "downloading $name"
  curl --fail --location --retry 100 --retry-delay 10 --retry-connrefused \
       --connect-timeout 30 --continue-at - --output "$DEST/$name.part" "$BASE/$name"
  mv "$DEST/$name.part" "$DEST/$name"
done

cd "$DEST"
printf '%s  %s\n' \
  988a0eac1f70372a1e95226c139db44c56fd8de0155a190352e3a2278464fcaf pytorch_model-00001-of-00002.bin \
  557d74ccf62676279ca42a4d635b6641f1ff7208ab73e530625a9dc0f79a692c pytorch_model-00002-of-00002.bin \
  9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347 tokenizer.model \
  | sha256sum --check
echo "phase5a3 base download complete"
