import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


LEVELS = ("phase", "cluster", "step", "atom")


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, obj: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_records(paths: Iterable[str | Path]):
    for path in paths:
        yield from read_jsonl(path)


def _record_time(record: Dict) -> tuple[float, float]:
    time = record.get("time", {})
    start = time.get("start_sec", time.get("start_frame", 0.0))
    end = time.get("end_sec", time.get("end_frame", start))
    return float(start or 0.0), float(end if end is not None else start or 0.0)


def _normalize_weights(weights: Dict[str, float], states: List[str], epsilon: float) -> Dict[str, float]:
    values = {state: max(0.0, float(weights.get(state, 0.0))) + epsilon for state in states}
    total = sum(values.values())
    return {state: value / total for state, value in values.items()}


def _default_procedural_priors(states: List[str], epsilon: float) -> Dict[str, Dict[str, float]]:
    priors = {}
    for src_idx, src in enumerate(states):
        weights = {}
        for dst_idx, dst in enumerate(states):
            offset = dst_idx - src_idx
            if offset in (0, 1):
                weights[dst] = 1.0
            elif offset == 2:
                weights[dst] = 0.5
            elif offset == -1:
                weights[dst] = 0.25
            else:
                weights[dst] = 0.05
        priors[src] = _normalize_weights(weights, states, epsilon)
    return priors


def _sequence_priors(
    records: List[Dict],
    states: List[str],
    epsilon: float,
    duration_bin_size: float,
) -> Dict:
    by_video = defaultdict(list)
    for record in records:
        atom = record.get("state", {}).get("atom")
        if atom in states:
            start, end = _record_time(record)
            by_video[str(record.get("video_id") or "unknown")].append((start, end, atom))

    transition_counts = {src: Counter() for src in states}
    duration_counts = {state: Counter() for state in states}
    for sequence in by_video.values():
        sequence.sort(key=lambda item: (item[0], item[1]))
        for (_, _, src), (_, _, dst) in zip(sequence, sequence[1:]):
            transition_counts[src][dst] += 1

        segments = []
        for start, end, atom in sequence:
            if segments and segments[-1][2] == atom:
                prev_start, prev_end, _ = segments[-1]
                segments[-1] = (prev_start, max(prev_end, end, start), atom)
            else:
                segments.append((start, max(start, end), atom))
        for index, (start, end, atom) in enumerate(segments):
            next_start = segments[index + 1][0] if index + 1 < len(segments) else end
            duration = max(duration_bin_size, max(end, next_start) - start)
            duration_bin = max(1, int(math.ceil(duration / duration_bin_size)))
            duration_counts[atom][duration_bin] += 1

    empirical = {
        src: _normalize_weights(dict(transition_counts[src]), states, epsilon)
        for src in states
    }
    durations = {}
    for state in states:
        counts = duration_counts[state]
        if not counts:
            durations[state] = {"1": 1.0}
            continue
        total = sum(counts.values())
        durations[state] = {str(duration): count / total for duration, count in sorted(counts.items())}
    return {
        "empirical_transitions": empirical,
        "duration_distributions": durations,
        "duration_bin_size": float(duration_bin_size),
        "smoothing_epsilon": float(epsilon),
    }


def build_label_space(
    records: Iterable[Dict],
    procedural_priors: Dict[str, Dict[str, float]] | None = None,
    epsilon: float = 1e-6,
    duration_bin_size: float = 30.0,
) -> Dict:
    records = list(records)
    labels = {level: [] for level in LEVELS}
    node_names = {}
    parent = {"atom_to_step": {}, "step_to_cluster": {}, "cluster_to_phase": {}}
    for rec in records:
        state = rec.get("state", {})
        for level in LEVELS:
            value = state.get(level)
            if value and value not in labels[level]:
                labels[level].append(value)
        atom = state.get("atom")
        if atom:
            node_names[atom] = state.get("node_name") or atom
            if state.get("step"):
                parent["atom_to_step"][atom] = state["step"]
        if state.get("step") and state.get("cluster"):
            parent["step_to_cluster"][state["step"]] = state["cluster"]
        if state.get("cluster") and state.get("phase"):
            parent["cluster_to_phase"][state["cluster"]] = state["phase"]
    levels = {key: sorted(values) for key, values in labels.items()}
    atoms = levels["atom"]
    priors = _sequence_priors(records, atoms, epsilon, duration_bin_size)
    if procedural_priors:
        priors["procedural_transitions"] = {
            src: _normalize_weights(procedural_priors.get(src, {}), atoms, epsilon)
            for src in atoms
        }
    else:
        priors["procedural_transitions"] = _default_procedural_priors(atoms, epsilon)
    return {
        "levels": levels,
        "node_names": node_names,
        "parents": parent,
        "priors": priors,
    }


def state_to_label_ids(state: Dict, label_space: Dict) -> Dict[str, int]:
    out = {}
    for level in LEVELS:
        values = label_space["levels"].get(level, [])
        value = state.get(level)
        out[level] = values.index(value) if value in values else -100
    return out


class SurgMarkJsonlDataset:
    def __init__(self, jsonl_path: str | Path, frames_root: str | Path = "."):
        self.path = Path(jsonl_path)
        self.frames_root = Path(frames_root)
        self.records = read_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        frames = [str(self.frames_root / p) for p in rec.get("frames", [])]
        answer = rec.get("answer", {})
        text = f"Current Node: {answer.get('node_name', '')}\nDescription: {answer.get('caption', '')}"
        return {
            "sample_id": rec.get("sample_id"),
            "frames": frames,
            "prompt": rec.get("prompt", "Identify the current surgical state and provide a brief description."),
            "target_text": text,
            "state": rec.get("state", {}),
            "time": rec.get("time", {}),
        }
