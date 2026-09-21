from __future__ import annotations

import argparse
import json

import torch
from torch_geometric.loader import DataLoader

from .data import KIBAGraphDataset
from .model import GraphDTAGCN


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU GraphDTA input/forward/gradient smoke test.")
    parser.add_argument("--data-root", default="data/ExternalBenchmarkKIBA_v1")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    dataset = KIBAGraphDataset(f"{args.data_root}/graphdta/train.csv", limit=args.limit)
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)
    model = GraphDTAGCN().cpu()
    batch = next(iter(loader))
    logits = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.y.reshape(-1))
    loss.backward()
    payload = {
        "status": "passed",
        "count": len(dataset),
        "nodes": int(batch.x.shape[0]),
        "edges": int(batch.edge_index.shape[1]),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "gradient_finite": all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
