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

    def _ground_truth_parent_mask(
        self,
        level: str,
        parent_level: str,
        labels: Dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        parent_maps = {
            "cluster": self.label_space.get("parents", {}).get("cluster_to_phase", {}),
            "step": self.label_space.get("parents", {}).get("step_to_cluster", {}),
            "atom": self.label_space.get("parents", {}).get("atom_to_step", {}),
        }
        child_values = self.label_space.get("levels", {}).get(level, [])
        parent_values = self.label_space.get("levels", {}).get(parent_level, [])
        relation = parent_maps.get(level, {})
        mask = torch.ones_like(logits, dtype=torch.bool)
        parent_targets = labels.get(parent_level)
        if parent_targets is None or not relation:
            return mask
        for batch_index, parent_id in enumerate(parent_targets.view(-1).tolist()):
            if parent_id < 0 or parent_id >= len(parent_values):
                continue
            parent = parent_values[parent_id]
            valid = [idx for idx, child in enumerate(child_values) if relation.get(child) == parent]
            if valid:
                mask[batch_index].fill_(False)
                mask[batch_index, valid] = True
        return mask

    def loss(self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], weights: Dict[str, float]):
        total = logits["boundary"].new_tensor(0.0)
        parts = {}
        parent_levels = {"cluster": "phase", "step": "cluster", "atom": "step"}
        for level in self.levels:
            if level not in labels or logits[level].numel() == 0:
                continue
            level_logits = logits[level]
            if level in parent_levels:
                valid_mask = self._ground_truth_parent_mask(
                    level,
                    parent_levels[level],
                    labels,
                    level_logits,
                )
                level_logits = level_logits.masked_fill(~valid_mask, torch.finfo(level_logits.dtype).min)
            loss = F.cross_entropy(level_logits, labels[level], ignore_index=-100)
            parts[f"{level}_loss"] = loss.detach()
            total = total + float(weights.get(level, 1.0)) * loss
        if "boundary" in labels:
            boundary_loss = F.binary_cross_entropy_with_logits(logits["boundary"], labels["boundary"].float())
            parts["boundary_loss"] = boundary_loss.detach()
            total = total + float(weights.get("boundary", 0.2)) * boundary_loss
        parts["state_loss"] = total.detach()
        return total, parts
