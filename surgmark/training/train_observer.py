import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from surgmark.data.jsonl_dataset import SurgMarkJsonlDataset, read_jsonl, state_to_label_ids
from surgmark.model.state_heads import HierarchicalStateHeads


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1024, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, batch_size: int, device):
        x = torch.zeros(batch_size, 1024, device=device)
        return self.proj(x)


def load_backbone(model_name_or_path: str):
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="auto")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None and hasattr(model.config, "llm_config"):
            hidden = model.config.llm_config.hidden_size
        return model, int(hidden)
    except Exception:
        model = TinyBackbone()
        return model, model.hidden_size


def collate(batch, label_space):
    labels = {level: [] for level in ("phase", "cluster", "step", "atom")}
    boundary = []
    for item in batch:
        ids = state_to_label_ids(item["state"], label_space)
        for level in labels:
            labels[level].append(ids[level])
        boundary.append(float(item.get("time", {}).get("start_frame") == item.get("time", {}).get("end_frame")))
    return {
        "items": batch,
        "labels": {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()},
        "boundary": torch.tensor(boundary, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--label-space", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.2)
    args = parser.parse_args()

    dataset = SurgMarkJsonlDataset(args.train_jsonl)
    if args.stage == "stage1":
        label_space = {"levels": {"phase": [], "cluster": [], "step": [], "atom": []}, "node_names": {}}
    else:
        label_space = json.loads(Path(args.label_space).read_text(encoding="utf-8"))

    backbone, hidden_size = load_backbone(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    state_heads = HierarchicalStateHeads(hidden_size, label_space).to(device)
    optim = torch.optim.AdamW(list(backbone.parameters()) + list(state_heads.parameters()), lr=args.lr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, label_space))

    weights = {"phase": args.hierarchy_loss_weight, "cluster": args.hierarchy_loss_weight, "step": args.hierarchy_loss_weight, "atom": 1.0, "boundary": args.boundary_loss_weight}
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"{args.stage} epoch {epoch + 1}")
        for batch in pbar:
            optim.zero_grad()
            if isinstance(backbone, TinyBackbone):
                hidden = backbone(len(batch["items"]), device)
            else:
                hidden = torch.zeros(len(batch["items"]), hidden_size, device=device)
            logits = state_heads(hidden)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            labels["boundary"] = batch["boundary"].to(device)
            loss, parts = state_heads.loss(logits, labels, weights)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(state_heads.state_dict(), output / "surgmark_state_heads.pt")
    (output / "training_note.json").write_text(json.dumps({"stage": args.stage, "train_jsonl": args.train_jsonl}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
