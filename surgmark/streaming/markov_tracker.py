import math
from dataclasses import dataclass
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
    def __init__(
        self,
        label_space: Dict,
        boundary_threshold: float = 0.85,
        score_margin: float = 0.08,
        minimum_switch_gap_sec: float = 30.0,
        score_weights: Dict[str, float] | None = None,
        procedural_prior_weight: float = 1.0,
        epsilon: float | None = None,
        hierarchy_weights: Dict[str, float] | None = None,
    ):
        self.label_space = label_space
        self.atoms = label_space.get("levels", {}).get("atom", [])
        self.node_names = label_space.get("node_names", {})
        self.parents = label_space.get("parents", {})
        self.order = {atom: i for i, atom in enumerate(self.atoms)}
        self.priors = label_space.get("priors", {})
        self.empirical_transitions = self.priors.get("empirical_transitions", {})
        self.procedural_transitions = self.priors.get("procedural_transitions", {})
        self.duration_distributions = self.priors.get("duration_distributions", {})
        self.duration_bin_size = float(self.priors.get("duration_bin_size", 30.0))
        default_epsilon = self.priors.get("smoothing_epsilon", 1e-6)
        self.epsilon = float(default_epsilon if epsilon is None else epsilon)
        self.boundary_threshold = float(boundary_threshold)
        self.score_margin = float(score_margin)
        self.minimum_switch_gap_sec = float(minimum_switch_gap_sec)
        weights = score_weights or {}
        self.alpha = float(weights.get("visual", 1.35))
        self.beta = float(weights.get("transition", 0.75))
        self.gamma = float(weights.get("boundary", 0.55))
        self.delta = float(weights.get("duration", 0.35))
        self.eta = float(weights.get("hierarchy", 0.50))
        self.procedural_prior_weight = float(procedural_prior_weight)
        self.hierarchy_weights = hierarchy_weights or {"phase": 1.0, "cluster": 1.0, "step": 1.0}
        self.critical_states = set(label_space.get("critical_states", []))
        self.current_atom = ""
        self.current_start_time = 0.0
        self.events: List[StateEvent] = []

    def _default_transition_probability(self, src: str, dst: str) -> float:
        if not self.atoms:
            return 1.0
        if not src:
            return 1.0 / len(self.atoms)
        diff = self.order.get(dst, -999) - self.order.get(src, -999)
        if diff in (0, 1):
            weight = 1.0
        elif diff == 2:
            weight = 0.5
        elif diff == -1:
            weight = 0.25
        else:
            weight = 0.05
        normalizer = sum(
            1.0 if offset in (0, 1) else 0.5 if offset == 2 else 0.25 if offset == -1 else 0.05
            for offset in (idx - self.order.get(src, 0) for idx in range(len(self.atoms)))
        )
        return weight / max(normalizer, self.epsilon)

    def _transition_probability(self, table: Dict, src: str, dst: str) -> float:
        if src and src in table:
            return float(table[src].get(dst, self.epsilon))
        return self._default_transition_probability(src, dst)

    def _duration_hazard(self, atom: str, elapsed: float) -> float:
        distribution = self.duration_distributions.get(atom, {})
        if not distribution:
            return 0.05 if elapsed < self.minimum_switch_gap_sec else 0.5
        duration_bin = max(1, int(math.ceil(elapsed / max(self.duration_bin_size, self.epsilon))))
        pmf = {int(key): float(value) for key, value in distribution.items()}
        if duration_bin > max(pmf):
            return 1.0 - self.epsilon
        event_probability = pmf.get(duration_bin, 0.0)
        survival_probability = sum(probability for duration, probability in pmf.items() if duration >= duration_bin)
        return min(1.0 - self.epsilon, max(self.epsilon, event_probability / max(survival_probability, self.epsilon)))

    def _ancestor(self, atom: str, level: str) -> str:
        step = self.parents.get("atom_to_step", {}).get(atom, "")
        if level == "step":
            return step
        cluster = self.parents.get("step_to_cluster", {}).get(step, "")
        if level == "cluster":
            return cluster
        if level == "phase":
            return self.parents.get("cluster_to_phase", {}).get(cluster, "")
        return ""

    def _visual_probability(self, observation: Dict, atom: str) -> float:
        global_probs = observation.get("global_atom_probs") or {}
        if atom in global_probs:
            return float(global_probs[atom])
        for candidate in observation.get("atom_topk") or []:
            if candidate.get("atom") == atom:
                return float(candidate.get("prob", 0.0))
        if atom == observation.get("atom"):
            return float(observation.get("confidence", 0.0))
        return self.epsilon

    def _hierarchy_score(self, observation: Dict, atom: str) -> float:
        hierarchy_probs = observation.get("hierarchy_probs") or {}
        score = 0.0
        for level in ("phase", "cluster", "step"):
            ancestor = self._ancestor(atom, level)
            if not ancestor:
                continue
            probability = hierarchy_probs.get(level, {}).get(ancestor)
            if probability is None:
                probability = 1.0 if observation.get(level) == ancestor else self.epsilon
            weight = float(self.hierarchy_weights.get(level, 1.0))
            score += weight * math.log(float(probability) + self.epsilon)
        return score

    def candidates(self, observation: Dict) -> List[str]:
        candidates = []
        if self.current_atom:
            candidates.append(self.current_atom)
        if observation.get("atom"):
            candidates.append(observation["atom"])
        candidates.extend(item.get("atom") for item in observation.get("atom_topk") or [])
        if self.current_atom in self.order:
            current_index = self.order[self.current_atom]
            for offset in (-1, 1, 2):
                neighbor_index = current_index + offset
                if 0 <= neighbor_index < len(self.atoms):
                    candidates.append(self.atoms[neighbor_index])
            frequent = sorted(
                self.empirical_transitions.get(self.current_atom, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            candidates.extend(atom for atom, _ in frequent[:3])
        candidates.extend(self.critical_states)
        return list(dict.fromkeys(atom for atom in candidates if atom in self.order))

    def candidate_scores(self, observation: Dict) -> List[Dict]:
        boundary = min(1.0, max(0.0, float(observation.get("boundary_prob", 0.0))))
        time_sec = float(observation.get("time_sec", 0.0))
        elapsed = max(0.0, time_sec - self.current_start_time)
        hazard = self._duration_hazard(self.current_atom, elapsed) if self.current_atom else 0.5
        scores = []
        for atom in self.candidates(observation):
            visual = math.log(self._visual_probability(observation, atom) + self.epsilon)
            empirical = self._transition_probability(self.empirical_transitions, self.current_atom, atom)
            procedural = self._transition_probability(self.procedural_transitions, self.current_atom, atom)
            transition = math.log(empirical + self.epsilon) + self.procedural_prior_weight * math.log(procedural + self.epsilon)
            is_persistence = bool(self.current_atom) and atom == self.current_atom
            boundary_support = (1.0 - boundary) if is_persistence else boundary if self.current_atom else 1.0
            duration_support = (1.0 - hazard) if is_persistence else hazard if self.current_atom else 1.0
            boundary_score = math.log(boundary_support + self.epsilon)
            duration_score = math.log(duration_support + self.epsilon)
            hierarchy = self._hierarchy_score(observation, atom)
            score = (
                self.alpha * visual
                + self.beta * transition
                + self.gamma * boundary_score
                + self.delta * duration_score
                + self.eta * hierarchy
            )
            scores.append(
                {
                    "atom": atom,
                    "score": score,
                    "visual": visual,
                    "transition": transition,
                    "boundary": boundary_score,
                    "duration": duration_score,
                    "hierarchy": hierarchy,
                }
            )
        if not scores:
            return []
        max_score = max(item["score"] for item in scores)
        normalizer = sum(math.exp(item["score"] - max_score) for item in scores)
        for item in scores:
            item["belief"] = math.exp(item["score"] - max_score) / normalizer
        return sorted(scores, key=lambda item: item["belief"], reverse=True)

    def step(self, observation: Dict) -> Tuple[str, Dict]:
        scores = self.candidate_scores(observation)
        if not scores:
            return self.current_atom, {"action": "hold", "reason": "no_candidate", "candidates": []}
        best = scores[0]
        keep = next((item for item in scores if item["atom"] == self.current_atom), None)
        boundary = float(observation.get("boundary_prob", 0.0))
        time_sec = float(observation.get("time_sec", 0.0))
        elapsed = max(0.0, time_sec - self.current_start_time)
        should_switch = not self.current_atom or best["atom"] != self.current_atom
        if self.current_atom and should_switch:
            margin = best["belief"] - (keep["belief"] if keep else 0.0)
            critical_recovery = best["atom"] in self.critical_states
            if (
                boundary < self.boundary_threshold
                or margin < self.score_margin
                or (elapsed < self.minimum_switch_gap_sec and not critical_recovery)
            ):
                return self.current_atom, {
                    "action": "hold",
                    "reason": "delayed_commitment",
                    "belief_margin": margin,
                    "candidates": scores[:5],
                }
        if should_switch:
            atom = best["atom"]
            event = self._make_event(atom, time_sec, "markov_belief_transition")
            self.events.append(event)
            self.current_atom = atom
            self.current_start_time = time_sec
            return atom, {
                "action": "transition",
                "event": event.__dict__,
                "belief": best["belief"],
                "candidates": scores[:5],
            }
        return self.current_atom, {
            "action": "hold",
            "reason": "same_state",
            "belief": best["belief"],
            "candidates": scores[:5],
        }

    def _make_event(self, atom: str, time_sec: float, reason: str) -> StateEvent:
        atom_to_step = self.parents.get("atom_to_step", {})
        step_to_cluster = self.parents.get("step_to_cluster", {})
        cluster_to_phase = self.parents.get("cluster_to_phase", {})
        step = atom_to_step.get(atom, "")
        cluster = step_to_cluster.get(step, "")
        phase = cluster_to_phase.get(cluster, "")
        return StateEvent(
            time_sec=time_sec,
            atom=atom,
            phase=phase,
            cluster=cluster,
            step=step,
            node_name=self.node_names.get(atom, atom),
            reason=reason,
        )
