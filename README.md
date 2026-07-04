# SurgMark

**An Agentic Hierarchical Markov State-Space Framework for Streaming Surgical Video Understanding**

## Demo

The following demo uses a cholecystectomy video as an example to show how SurgMark performs real-time surgical state understanding, procedural graph construction, and interactive surgical QA.

<video controls preload="metadata" width="100%">
  <source src="assets/videos/demo.mp4" type="video/mp4">
  Your browser does not support embedded video playback.
</video>

If the embedded player is not rendered by GitHub, open the demo video directly: [assets/videos/demo.mp4](assets/videos/demo.mp4).

## Main Idea

SurgMark is a reference implementation for streaming surgical video understanding. It models surgery as a hierarchical Markov state-space process and combines a VLM-based surgical state observer, online Markov belief tracking, a dynamic procedural memory graph, and an LLM decision agent with explicit tool actions.

The framework is designed for causal intraoperative video streams. At each time step, SurgMark observes the current video window, predicts hierarchical surgical states, updates the Markov belief, commits reliable states into a procedural memory graph, and routes surgical questions to the appropriate evidence source.

## Innovations

- **Hierarchical surgical state observer:** predicts multi-level surgical states and captions from streaming video windows.
- **Markov state-space tracking:** constrains noisy observations using procedure-aware transition priors and temporal guards.
- **Dynamic procedural memory graph:** stores completed states, state transitions, uncertainty, deviation notes, and graph-level procedural context.
- **Agentic decision module:** uses structured tools to hold, transition, revise, mark uncertainty/deviation, update the graph, and route QA evidence.
- **Streaming surgical QA:** answers questions about completed steps, the current operation, and expected next actions using observation, memory graph, SOP prior, or state belief.

## Overview

![SurgMark overview](assets/images/surgmark_overview.png)

## Data

This repository contains compact English JSONL annotations under `data/open_english/`. Raw surgical frames, private gastric data, model checkpoints, and API keys are not included.

Frame paths in the released JSONL files are relative placeholders, such as:

```text
frames/cholec/VID01/000000.png
```

Download the original public frames separately and place or symlink them under `data/frames/`.

Original public datasets:

- CholecT45: https://github.com/CAMMA-public/cholect45
- PSI-AVA / TAPIR: https://github.com/BCV-Uniandes/TAPIR
- AutoLaparo: https://github.com/ziyiwangx/AutoLaparo and https://autolaparo.github.io/

## Usage

### Environment

```bash
conda create -n surgmark python=3.10 -y
conda activate surgmark
pip install -r requirements.txt
```

For full VLM training, install the dependencies required by your Intern-compatible base model. The included model wrapper uses `trust_remote_code=True`.

### Prepare Labels

```bash
bash scripts/build_label_space.sh
```

### Training

Stage 1: frame-level semantic alignment.

```bash
DATASET=cholec MODEL=OpenGVLab/InternVL2-8B bash scripts/train_stage1_alignment.sh
```

Stage 2: clip-level state-aware training with hierarchical state and boundary heads.

```bash
DATASET=cholec MODEL=checkpoints/cholec_stage1_alignment bash scripts/train_stage2_state_observer.sh
```

### Testing

Run the cached observation path to test the Markov tracker, procedural memory graph, and agent interface without model weights.

```bash
bash scripts/build_label_space.sh
DRY_RUN=1 bash scripts/run_cached_stream_demo.sh
```

### Inference

Streaming inference without the LLM agent:

```bash
FRAMES_DIR=data/frames/cholec/VID01 bash scripts/run_streaming_inference.sh
```

Streaming inference with the decision agent:

```bash
export OPENAI_API_KEY=your_key_here
FRAMES_DIR=data/frames/cholec/VID01 bash scripts/run_agent_streaming.sh
```

The LLM configuration template is available at `configs/agent.example.json`. Do not commit real API keys.

## Repository Structure

```text
surgmark/
  data/        JSONL loading and label-space construction
  model/       hierarchical state and boundary heads
  training/    two-stage observer training entry
  streaming/   online Markov tracking and streaming inference
  agent/       memory graph, tools, LLM client, decision agent
scripts/       stage-wise runnable examples
configs/       relative-path config templates
data/open_english/
  cholec/
  psiava/
  autolaparo/
assets/
  images/
  videos/
```

## Notes

The scripts are intentionally concise and use relative paths. They are meant to expose the core method components clearly; large-scale training may require adapting batch size, distributed launch, model wrappers, and dataset-specific preprocessing.

This repository does not include model weights, private data, raw surgical frames, or API credentials.

## Contact

For any questions or inquiries, please contact us at pansitian2025@ia.ac.cn.
