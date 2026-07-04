import os
from typing import Dict, List


class OpenAICompatibleLLM:
    def __init__(self, config: Dict):
        from openai import OpenAI

        llm_cfg = config.get("llm", config)
        api_key = llm_cfg.get("api_key") or os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or provide llm.api_key outside the repository.")
        kwargs = {"api_key": api_key}
        if llm_cfg.get("base_url"):
            kwargs["base_url"] = llm_cfg["base_url"]
        self.client = OpenAI(**kwargs)
        self.model = llm_cfg.get("model", "gpt-5")
        self.temperature = float(llm_cfg.get("temperature", 0.0))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1024))

    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""
