#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -m surgmark.streaming.stream_infer \
  --video-id "${VIDEO_ID:-demo}" \
  --label-space "${LABEL_SPACE:-configs/label_space.json}" \
  --cached-observations "${CACHED_OBSERVATIONS:-examples/cached_observations.jsonl}" \
  --agent-config "${AGENT_CONFIG:-configs/agent.example.json}" \
  --output-dir "${OUTPUT_DIR:-outputs/cached_agent_demo}" \
  --dry-run
