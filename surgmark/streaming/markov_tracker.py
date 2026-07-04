from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class StateEvent:
    time_sec: float
    atom: str
    phase: str = ""
    cluster: str = ""
    step: str = ""
    node_name: str = ""
    reason: str = ""


class MarkovStateTracker:
    def __init__(self, label_space: Dict, boundary_threshold: float = 0.85, score_margin: float = 0.08, minimum_switch_gap_sec: float = 30.0):
        self.label_space = label_space
        self.atoms = label_space.get("levels", {}).get("atom", [])
        self.node_names = label_space.get("node_names", {})
        self.parents = label_space.get("parents", {})
        self.order = {atom: i for i, atom in enumerate(self.atoms)}
        self.boundary_threshold = float(boundary_threshold)
        self.score_margin = float(score_margin)
        self.minimum_switch_gap_sec = float(minimum_switch_gap_sec)
        self.current_atom = ""
        self.current_start_time = 0.0
        self.events: List[StateEvent] = []

    def transition_logit(self, src: str, dst: str) -> float:
        if not src:
            return 0.2
        if src == dst:
            return 0.35
        diff = self.order.get(dst, -999) - self.order.get(src, -999)
        if diff == 1:
            return 0.35
        if diff == 2:
            return 0.1
        if diff > 2:
            return -0.35
        return -0.3

    def candidate_scores(self, observation: Dict) -> List[Dict]:
        topk = observation.get("atom_topk") or []
        atoms = [x.get("atom") for x in topk if x.get("atom")]
        if observation.get("atom"):
            atoms.insert(0, observation["atom"])
        if self.current_atom:
            atoms.insert(0, self.current_atom)
        atoms = list(dict.fromkeys([a for a in atoms if a in self.order]))
        boundary = float(observation.get("boundary_prob", 0.0))
        time_sec = float(observation.get("time_sec", 0.0))
        elapsed = max(0.0, time_sec - self.current_start_time)
        by_atom = {x.get("atom"): float(x.get("prob", 0.0)) for x in topk}
        scores = []
        for atom in atoms:
            visual = by_atom.get(atom, 0.0)
            if atom == observation.get("atom"):
                visual = max(visual, float(observation.get("confidence", 0.0)))
            transition = self.transition_logit(self.current_atom, atom)
            boundary_term = 0.2 if atom != self.current_atom and boundary >= self.boundary_threshold else 0.0
            duration_term = -0.35 if atom != self.current_atom and elapsed < self.minimum_switch_gap_sec else 0.0
            score = 1.35 * visual + 0.75 * transition + 0.55 * boundary_term + 0.35 * duration_term
            scores.append({"atom": atom, "score": score, "visual": visual, "transition": transition})
        return sorted(scores, key=lambda x: x["score"], reverse=True)

    def step(self, observation: Dict) -> Tuple[str, Dict]:
        scores = self.candidate_scores(observation)
        if not scores:
            return self.current_atom, {"action": "hold", "reason": "no_candidate", "candidates": []}
        best = scores[0]
        keep = next((x for x in scores if x["atom"] == self.current_atom), None)
        boundary = float(observation.get("boundary_prob", 0.0))
        should_switch = not self.current_atom or best["atom"] != self.current_atom
        if self.current_atom and should_switch:
            margin = best["score"] - (keep["score"] if keep else -999.0)
            if boundary < self.boundary_threshold or margin < self.score_margin:
                return self.current_atom, {"action": "hold", "reason": "markov_guard", "candidates": scores[:5]}
        if should_switch:
            atom = best["atom"]
            time_sec = float(observation.get("time_sec", 0.0))
            event = self._make_event(atom, time_sec, "markov_transition")
            self.events.append(event)
            self.current_atom = atom
            self.current_start_time = time_sec
            return atom, {"action": "transition", "event": event.__dict__, "candidates": scores[:5]}
        return self.current_atom, {"action": "hold", "reason": "same_state", "candidates": scores[:5]}

    def _make_event(self, atom: str, time_sec: float, reason: str) -> StateEvent:
        atom_to_step = self.parents.get("atom_to_step", {})
        step_to_cluster = self.parents.get("step_to_cluster", {})
        cluster_to_phase = self.parents.get("cluster_to_phase", {})
        step = atom_to_step.get(atom, "")
        cluster = step_to_cluster.get(step, "")
        phase = cluster_to_phase.get(cluster, "")
        return StateEvent(time_sec=time_sec, atom=atom, phase=phase, cluster=cluster, step=step, node_name=self.node_names.get(atom, atom), reason=reason)
