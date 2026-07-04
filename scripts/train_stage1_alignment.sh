#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MODEL="${MODEL:-OpenGVLab/InternVL2-8B}"
DATASET="${DATASET:-cholec}"
TRAIN_JSONL="data/${DATASET}/state_caption_train.jsonl"
OUT_DIR="checkpoints/${DATASET}_stage1_alignment"

python -m surgmark.training.train_observer \
  --stage stage1 \
  --model "${MODEL}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${LR:-2e-5}"
