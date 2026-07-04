# SurgMark: An Agentic Hierarchical Markov State-Space Framework for Streaming Surgical Video Understanding

![SurgMark overview](assets/images/surgmark_overview.png)

SurgMark is a compact reference implementation for streaming surgical video understanding. It models surgery as a hierarchical Markov state-space process and combines a VLM-based surgical state observer, online Markov belief tracking, a dynamic procedural memory graph, and an LLM decision agent with explicit tools.

The framework is designed for causal intraoperative video streams: it predicts multi-level surgical states, tracks state transitions, maintains queryable procedure memory, flags potential workflow deviations, and routes context-aware surgical QA through graph, observation, SOP prior, or visual evidence.

This repository does not include model weights, private gastric data, raw surgical frames, or API keys.

## Demo

<video src="assets/videos/demo.mp4" controls width="100%"></video>

If the video preview is not rendered by GitHub, open it directly: [assets/videos/demo.mp4](assets/videos/demo.mp4).

## Structure

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
```

## Environment

```bash
conda create -n surgmark python=3.10 -y
conda activate surgmark
pip install -r requirements.txt
```

For full VLM training, install the dependencies required by your InternVL/Intern-compatible base model. The included code uses `trust_remote_code=True` for model loading.

## Data

The repository contains only compact English JSONL annotations under `data/open_english/`. Frame paths are relative placeholders such as `frames/cholec/VID01/000000.png`; download the original frames separately and place or symlink them under `data/frames/`.

Original public datasets:

- CholecT45: https://github.com/CAMMA-public/cholect45
- PSI-AVA / TAPIR: https://github.com/BCV-Uniandes/TAPIR
- AutoLaparo: https://github.com/ziyiwangx/AutoLaparo and https://autolaparo.github.io/

## Minimal Workflow

Build the hierarchical label space:

```bash
bash scripts/build_label_space.sh
```

Stage 1 frame-level semantic alignment:

```bash
DATASET=cholec MODEL=OpenGVLab/InternVL2-8B bash scripts/train_stage1_alignment.sh
```

Stage 2 clip-level state-aware training:

```bash
DATASET=cholec MODEL=checkpoints/cholec_stage1_alignment bash scripts/train_stage2_state_observer.sh
```

Streaming inference without the LLM agent:

```bash
FRAMES_DIR=data/frames/cholec/VID01 bash scripts/run_streaming_inference.sh
```

Streaming inference with the agent:

```bash
export OPENAI_API_KEY=your_key_here
FRAMES_DIR=data/frames/cholec/VID01 bash scripts/run_agent_streaming.sh
```

Run the Markov/agent path without model weights:

```bash
bash scripts/build_label_space.sh
DRY_RUN=1 bash scripts/run_cached_stream_demo.sh
```

## Agent Design

At each streaming step, SurgMark builds:

1. observer evidence: caption, atom top-k, hierarchy logits, boundary probability;
2. Markov belief: SOP-like transition prior with duration and boundary guards;
3. procedural memory graph: accepted nodes, uncertainty, and deviation ledger;
4. tool action plan: hold, transition, revise, mark uncertainty/deviation, write graph, route QA, or inspect frame.

The LLM configuration template is `configs/agent.example.json`. Do not commit real API keys.

## Notes

The scripts are intentionally concise and use relative paths. They are meant to expose the method components cleanly; large-scale training may require adapting batch size, distributed launch, and the exact Intern-compatible model wrapper to your environment.

## Contact

For any questions or inquiries, please contact us at pansitian2025@ia.ac.cn.
