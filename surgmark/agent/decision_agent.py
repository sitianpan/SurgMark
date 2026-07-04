import json
import re
from typing import Dict, List

from surgmark.agent.llm_client import OpenAICompatibleLLM
from surgmark.agent.memory import ProceduralMemory
from surgmark.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from surgmark.agent.tools import AgentToolbox


def extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class SurgMarkAgent:
    def __init__(self, config: Dict, memory: ProceduralMemory, label_space: Dict, dry_run: bool = False):
        self.config = config
        self.memory = memory
        self.toolbox = AgentToolbox(memory, label_space)
        self.dry_run = dry_run
        self.llm = None if dry_run else OpenAICompatibleLLM(config)

    def decide(self, observation: Dict, markov_decision: Dict, pending_qa: List[Dict] | None = None) -> Dict:
        if self.dry_run:
            atom = observation.get("atom", "")
            action = "transition_state" if atom and atom != self.memory.current_atom else "hold_state"
            return {"thought_summary": "dry_run", "actions": [{"tool": action, "atom": atom, "reason": "local observation"}], "confidence": 0.0}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation, markov_decision, self.memory.to_text(), pending_qa)},
        ]
        raw = self.llm.chat(messages)
        decision = extract_json(raw)
        decision["raw_llm_output"] = raw
        return decision

    def act(self, decision: Dict, observation: Dict) -> List[Dict]:
        return self.toolbox.apply(decision.get("actions", []), observation)
