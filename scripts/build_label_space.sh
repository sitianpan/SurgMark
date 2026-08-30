#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

for dataset in cholec psiava autolaparo; do
  python -m surgmark.data.build_label_space \
    --jsonl "data/${dataset}/state_caption_train.jsonl" \
    --output "configs/${dataset}_label_space.json"
done

cp configs/cholec_label_space.json configs/label_space.json
