#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -m surgmark.data.build_label_space \
  --jsonl \
    data/open_english/cholec/state_caption_train.jsonl \
    data/open_english/psiava/state_caption_train.jsonl \
    data/open_english/autolaparo/state_caption_train.jsonl \
  --output configs/label_space.json
