import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from surgmark.model.state_heads import HierarchicalStateHeads


class SurgMarkObserver(nn.Module):
    def __init__(self, base_model: nn.Module, tokenizer, label_space: Dict, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.state_heads = HierarchicalStateHeads(hidden_size, label_space)
        self.label_space = label_space

    @classmethod
    def from_pretrained(cls, model_path: str, label_space_path: str):
        from transformers import AutoModel, AutoTokenizer

        label_space = json.loads(Path(label_space_path).read_text(encoding="utf-8"))
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "llm_config", {}).hidden_size
        model = cls(base_model, tokenizer, label_space, hidden_size)
        head_path = Path(model_path) / "surgmark_state_heads.pt"
        if head_path.exists():
            model.state_heads.load_state_dict(torch.load(head_path, map_location="cpu"))
        return model

    def save_heads(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_heads.state_dict(), output_dir / "surgmark_state_heads.pt")

    def state_from_logits(self, logits: Dict[str, torch.Tensor], top_k: int = 5) -> Dict:
        pred = {}
        for level in ("phase", "cluster", "step", "atom"):
            probs = logits[level].softmax(dim=-1)[0]
            values = self.label_space["levels"].get(level, [])
            idx = int(probs.argmax())
            pred[level] = values[idx] if idx < len(values) else ""
            if level == "atom":
                k = min(top_k, len(values))
                top = torch.topk(probs, k=k)
                pred["atom_topk"] = [
                    {"atom": values[int(i)], "prob": float(p)}
                    for p, i in zip(top.values, top.indices)
                ]
        pred["boundary_prob"] = float(torch.sigmoid(logits["boundary"])[0])
        pred["node_name"] = self.label_space.get("node_names", {}).get(pred.get("atom", ""), pred.get("atom", ""))
        return pred

    @torch.no_grad()
    def observe_window(self, frame_paths: List[str], prompt: str, top_k: int = 5) -> Dict:
        if not hasattr(self.base_model, "chat"):
            raise RuntimeError("The loaded base model must expose an InternVL-style chat API for open-ended captions.")
        response, history, hidden = self.base_model.chat(
            self.tokenizer,
            None,
            prompt,
            generation_config={"max_new_tokens": 128, "do_sample": False},
            history=None,
            return_history=True,
            return_hidden=True,
        )
        if hidden.dim() == 3:
            hidden = hidden[:, -1]
        logits = self.state_heads(hidden)
        state = self.state_from_logits(logits, top_k=top_k)
        return {"caption": response, "state": state}
