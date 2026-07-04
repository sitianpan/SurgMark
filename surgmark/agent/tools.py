from typing import Dict, List

from surgmark.agent.memory import ProceduralMemory


class AgentToolbox:
    def __init__(self, memory: ProceduralMemory, label_space: Dict):
        self.memory = memory
        self.label_space = label_space

    def node_name(self, atom: str) -> str:
        return self.label_space.get("node_names", {}).get(atom, atom)

    def apply(self, actions: List[Dict], observation: Dict) -> List[Dict]:
        executed = []
        time_sec = float(observation.get("time_sec", 0.0))
        for action in actions:
            tool = action.get("tool", "hold_state")
            if tool == "hold_state":
                executed.append(self.memory.hold_state(time_sec, action.get("reason", "")))
            elif tool == "transition_state":
                atom = action.get("atom") or observation.get("atom", "")
                executed.append(self.memory.transition_state(time_sec, atom, self.node_name(atom), evidence=action.get("reason", "")))
            elif tool == "revise_state":
                atom = action.get("atom") or observation.get("atom", "")
                executed.append(self.memory.revise_state(atom, self.node_name(atom), action.get("reason", "")))
            elif tool == "mark_uncertainty":
                executed.append(self.memory.mark_uncertainty(time_sec, action.get("note") or action.get("reason", "")))
            elif tool == "mark_deviation":
                executed.append(self.memory.mark_deviation(time_sec, action.get("note") or action.get("reason", ""), action.get("severity", "low")))
            elif tool == "write_graph":
                executed.append({"tool": "write_graph", "graph": self.memory.to_text()})
            elif tool == "answer_question":
                executed.append({"tool": "answer_question", "qa_id": action.get("qa_id", ""), "route": action.get("route", [])})
            elif tool == "inspect_frame":
                executed.append({"tool": "inspect_frame", "status": "requested"})
            else:
                executed.append({"tool": tool, "status": "unknown"})
        return executed
