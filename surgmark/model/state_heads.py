from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalStateHeads(nn.Module):
    def __init__(self, hidden_size: int, label_space: Dict, dropout: float = 0.1):
        super().__init__()
        self.label_space = label_space
        self.levels = ("phase", "cluster", "step", "atom")
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            level: nn.Linear(hidden_size, len(label_space["levels"].get(level, [])))
            for level in self.levels
        })
        self.boundary = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.drop(self.norm(hidden))
        logits = {level: head(hidden) for level, head in self.heads.items()}
        logits["boundary"] = self.boundary(hidden).squeeze(-1)
        return logits

    def loss(self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], weights: Dict[str, float]):
        total = logits["boundary"].new_tensor(0.0)
        parts = {}
        for level in self.levels:
            if level not in labels or logits[level].numel() == 0:
                continue
            loss = F.cross_entropy(logits[level], labels[level], ignore_index=-100)
            parts[f"{level}_loss"] = loss.detach()
            total = total + float(weights.get(level, 1.0)) * loss
        if "boundary" in labels:
            boundary_loss = F.binary_cross_entropy_with_logits(logits["boundary"], labels["boundary"].float())
            parts["boundary_loss"] = boundary_loss.detach()
            total = total + float(weights.get("boundary", 0.2)) * boundary_loss
        parts["state_loss"] = total.detach()
        return total, parts
