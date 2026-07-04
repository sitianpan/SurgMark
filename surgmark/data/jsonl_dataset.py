import json
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


def build_label_space(records: Iterable[Dict]) -> Dict:
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
    return {"levels": {k: sorted(v) for k, v in labels.items()}, "node_names": node_names, "parents": parent}


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
