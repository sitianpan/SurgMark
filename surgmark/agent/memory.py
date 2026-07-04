from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GraphNode:
    time_sec: float
    atom: str
    node_name: str
    phase: str = ""
    step: str = ""
    evidence: str = ""


@dataclass
class ProceduralMemory:
    video_id: str = ""
    nodes: List[GraphNode] = field(default_factory=list)
    uncertainty: List[Dict] = field(default_factory=list)
    deviations: List[Dict] = field(default_factory=list)

    @property
    def current_atom(self) -> str:
        return self.nodes[-1].atom if self.nodes else ""

    def hold_state(self, time_sec: float, reason: str = "") -> Dict:
        return {"tool": "hold_state", "time_sec": time_sec, "atom": self.current_atom, "reason": reason}

    def transition_state(self, time_sec: float, atom: str, node_name: str, phase: str = "", step: str = "", evidence: str = "") -> Dict:
        node = GraphNode(time_sec=float(time_sec), atom=atom, node_name=node_name, phase=phase, step=step, evidence=evidence)
        self.nodes.append(node)
        return {"tool": "transition_state", **node.__dict__}

    def revise_state(self, atom: str, node_name: str, reason: str = "") -> Dict:
        if not self.nodes:
            return {"tool": "revise_state", "status": "skipped", "reason": "empty_memory"}
        self.nodes[-1].atom = atom
        self.nodes[-1].node_name = node_name
        self.nodes[-1].evidence = reason
        return {"tool": "revise_state", "atom": atom, "node_name": node_name, "reason": reason}

    def mark_uncertainty(self, time_sec: float, note: str) -> Dict:
        item = {"time_sec": float(time_sec), "note": note}
        self.uncertainty.append(item)
        return {"tool": "mark_uncertainty", **item}

    def mark_deviation(self, time_sec: float, note: str, severity: str = "low") -> Dict:
        item = {"time_sec": float(time_sec), "note": note, "severity": severity}
        self.deviations.append(item)
        return {"tool": "mark_deviation", **item}

    def to_text(self, max_nodes: int = 12) -> str:
        lines = [
            "## Procedural Memory Graph",
            f"video_id: {self.video_id}",
            f"current_atom: {self.current_atom or 'UNKNOWN'}",
            "",
            "| time | atom | node | phase | step | evidence |",
            "|---:|---|---|---|---|---|",
        ]
        for node in self.nodes[-max_nodes:]:
            lines.append(f"| {node.time_sec:.1f}s | {node.atom} | {node.node_name} | {node.phase} | {node.step} | {node.evidence} |")
        if self.uncertainty:
            lines.append("\nuncertainty:")
            for item in self.uncertainty[-5:]:
                lines.append(f"- {item['time_sec']:.1f}s: {item['note']}")
        if self.deviations:
            lines.append("\ndeviations:")
            for item in self.deviations[-5:]:
                lines.append(f"- {item['time_sec']:.1f}s [{item['severity']}]: {item['note']}")
        return "\n".join(lines)
