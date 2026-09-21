from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import KIBADataset, PROTEIN_TO_ID, SMILES_TO_ID
from .model import DeepDTAClassifier


def main():
    parser = argparse.ArgumentParser(description="CPU DeepDTA input/forward/gradient smoke test.")
    parser.add_argument("--data-root", default="data/ExternalBenchmarkKIBA_v1")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    dataset = KIBADataset(f"{args.data_root}/graphdta/train.csv", args.limit)
    batch = next(iter(DataLoader(dataset, batch_size=min(args.limit, 2), shuffle=False)))
    model = DeepDTAClassifier(len(SMILES_TO_ID), len(PROTEIN_TO_ID)).cpu()
    logits = model(batch["drug"], batch["protein"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"])
    loss.backward()
    payload = {
        "status": "passed",
        "count": len(dataset),
        "drug_shape": list(batch["drug"].shape),
        "protein_shape": list(batch["protein"].shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "gradient_finite": all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

