#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

ARGS=(
  --video-id "${VIDEO_ID:-demo}" \
  --model-path "${MODEL_PATH:-checkpoints/cholec_surgmark_observer}" \
  --label-space "${LABEL_SPACE:-configs/label_space.json}" \
  --frames-dir "${FRAMES_DIR:-data/frames/cholec/VID01}" \
  --agent-config "${AGENT_CONFIG:-configs/agent.example.json}" \
  --output-dir "${OUTPUT_DIR:-outputs/stream_agent}" \
  --window-size "${WINDOW_SIZE:-4}" \
  --stride "${STRIDE:-1}" \
  --top-k "${TOP_K:-5}" \
  --boundary-threshold "${BOUNDARY_TH:-0.85}" \
  --score-margin "${SCORE_MARGIN:-0.08}" \
  --minimum-switch-gap-sec "${MIN_SWITCH_GAP:-30}"
)
if [[ "${DRY_RUN:-}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python -m surgmark.streaming.stream_infer "${ARGS[@]}"
