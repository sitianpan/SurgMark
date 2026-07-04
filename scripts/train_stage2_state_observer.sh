#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

DATASET="${DATASET:-cholec}"
MODEL="${MODEL:-checkpoints/${DATASET}_stage1_alignment}"
TRAIN_JSONL="data/open_english/${DATASET}/state_caption_train.jsonl"
LABEL_SPACE="${LABEL_SPACE:-configs/label_space.json}"
OUT_DIR="checkpoints/${DATASET}_surgmark_observer"

test -f "${LABEL_SPACE}" || bash scripts/build_label_space.sh

python -m surgmark.training.train_observer \
  --stage stage2 \
  --model "${MODEL}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --label-space "${LABEL_SPACE}" \
  --output-dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${LR:-1e-5}" \
  --hierarchy-loss-weight "${HIER_W:-0.4}" \
  --boundary-loss-weight "${BOUNDARY_W:-0.2}"
