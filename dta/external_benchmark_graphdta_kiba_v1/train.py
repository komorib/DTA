from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .data import KIBAGraphDataset
from .metrics import binary_metrics
from .model import GraphDTAGCN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, optimizer, criterion, device, train: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    all_labels, all_logits = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            labels = batch.y.reshape(-1)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            count = labels.numel()
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            all_labels.append(labels.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.float32)
    logits = np.concatenate(all_logits) if all_logits else np.empty(0, dtype=np.float32)
    return labels, logits, total_loss / max(total_count, 1)


def fit(args) -> Dict:
    seed_everything(args.seed)
    root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_set = KIBAGraphDataset(root / "graphdta/train.csv", limit=args.limit_train)
    valid_set = KIBAGraphDataset(root / "graphdta/valid.csv", limit=args.limit_valid)
    test_set = KIBAGraphDataset(root / "graphdta/test.csv", limit=args.limit_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_labels = np.asarray([row.label for row in train_set.records], dtype=np.float32)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)

    model = GraphDTAGCN(dropout=args.dropout).to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score = -float("inf")
    best_epoch = 0
    best_path = output / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_y, train_z, train_loss = _run_epoch(model, train_loader, optimizer, criterion, args.device, True)
        valid_y, valid_z, valid_loss = _run_epoch(model, valid_loader, optimizer, criterion, args.device, False)
        valid_metrics = binary_metrics(valid_y, valid_z)
        score = valid_metrics["ROC-AUC"]
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_metrics": valid_metrics})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if (best_epoch == 0) or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_metrics": valid_metrics}, best_path)
        if epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    valid_y, valid_z, valid_loss = _run_epoch(model, valid_loader, optimizer, criterion, args.device, False)
    test_y, test_z, test_loss = _run_epoch(model, test_loader, optimizer, criterion, args.device, False)
    result = {
        "version": "external-benchmark-graphdta-kiba-binary-v1",
        "model": "GraphDTA-GCN-sequence-pool",
        "task": "KIBA binary interaction classification",
        "data_root": str(root),
        "output": str(output),
        "seed": args.seed,
        "device": str(args.device),
        "counts": {"train": len(train_set), "valid": len(valid_set), "test": len(test_set)},
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "validation ROC-AUC",
        "validation_metrics": binary_metrics(valid_y, valid_z),
        "test_metrics": binary_metrics(test_y, test_z),
        "test_used_for_selection": False,
        "history": history,
    }
    np.savez_compressed(output / "valid_predictions.npz", labels=valid_y, logits=valid_z)
    np.savez_compressed(output / "test_predictions.npz", labels=test_y, logits=test_z)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result
