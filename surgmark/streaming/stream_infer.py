import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from surgmark.agent.decision_agent import SurgMarkAgent
from surgmark.agent.memory import ProceduralMemory
from surgmark.model.observer import SurgMarkObserver
from surgmark.streaming.markov_tracker import MarkovStateTracker


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cached_observations(path: Path):
    for row in read_jsonl(path):
        yield row


def frame_windows(frames: List[str], window_size: int, stride: int):
    for idx in range(0, len(frames), stride):
        window = frames[max(0, idx - window_size + 1): idx + 1]
        yield idx, window


def run(args):
    label_space = json.loads(Path(args.label_space).read_text(encoding="utf-8"))
    tracker = MarkovStateTracker(label_space, args.boundary_threshold, args.score_margin, args.minimum_switch_gap_sec)
    memory = ProceduralMemory(video_id=args.video_id)
    agent_cfg = json.loads(Path(args.agent_config).read_text(encoding="utf-8")) if args.agent_config else {"llm": {}, "agent": {"enabled": False}}
    agent_enabled = bool(agent_cfg.get("agent", {}).get("enabled", False))
    agent = SurgMarkAgent(agent_cfg, memory, label_space, dry_run=args.dry_run) if agent_enabled else None

    if args.cached_observations:
        observations = cached_observations(Path(args.cached_observations))
    else:
        model = SurgMarkObserver.from_pretrained(args.model_path, args.label_space)
        frames = sorted(str(p) for p in Path(args.frames_dir).glob("*"))
        observations = (
            {
                "time_sec": i * args.seconds_per_step,
                **model.observe_window(window, "Identify the current surgical state and provide a brief description.", top_k=args.top_k)["state"],
            }
            for i, window in frame_windows(frames, args.window_size, args.stride)
        )

    out_rows = []
    for step, observation in enumerate(observations):
        observation.setdefault("time_sec", step * args.seconds_per_step)
        atom, markov_info = tracker.step(observation)
        observation["markov_atom"] = atom
        executed = []
        decision = {}
        if agent is not None:
            decision = agent.decide(observation, markov_info)
            executed = agent.act(decision, observation)
        elif markov_info.get("action") == "transition":
            event = markov_info["event"]
            memory.transition_state(event["time_sec"], event["atom"], event["node_name"], event["phase"], event["step"], event["reason"])
        out = {"step": step, "observation": observation, "markov": markov_info, "agent_decision": decision, "executed": executed, "graph": memory.to_text()}
        out_rows.append(out)
        print(f"[{step:04d}] t={observation.get('time_sec')} atom={observation.get('atom')} markov={atom} tools={[x.get('tool') for x in executed]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "stream_trace.jsonl", out_rows)
    (output_dir / "graph.txt").write_text(memory.to_text(max_nodes=999), encoding="utf-8")
    print(f"saved {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="demo")
    parser.add_argument("--model-path", default="checkpoints/surgmark")
    parser.add_argument("--label-space", default="configs/label_space.json")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--cached-observations", default="")
    parser.add_argument("--agent-config", default="")
    parser.add_argument("--output-dir", default="outputs/stream_run")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seconds-per-step", type=float, default=5.0)
    parser.add_argument("--boundary-threshold", type=float, default=0.85)
    parser.add_argument("--score-margin", type=float, default=0.08)
    parser.add_argument("--minimum-switch-gap-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
