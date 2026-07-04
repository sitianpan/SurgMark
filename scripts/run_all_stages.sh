#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/build_label_space.sh
bash scripts/train_stage1_alignment.sh
bash scripts/train_stage2_state_observer.sh
bash scripts/run_streaming_inference.sh
