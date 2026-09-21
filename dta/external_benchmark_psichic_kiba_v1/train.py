from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PSICHICKIBADataset, list_collate
from .metrics import binary_metrics
from .model import PSICHICSequenceEdgeClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train(train)
    all_labels, all_logits = [], []
    total_loss, total_count = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for items in loader:
            moved = []
            for item in items:
                moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in item.items()})
            labels = torch.stack([item["label"] for item in moved])
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(moved)
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
    return np.concatenate(all_labels), np.concatenate(all_logits), total_loss / max(total_count, 1)


def fit(args):
    seed_everything(args.seed)
    data_root, feature_root, output = Path(args.data_root), Path(args.feature_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_set = PSICHICKIBADataset(data_root / "psichic/train.csv", feature_root, args.limit_train)
    valid_set = PSICHICKIBADataset(data_root / "psichic/valid.csv", feature_root, args.limit_valid)
    test_set = PSICHICKIBADataset(data_root / "psichic/test.csv", feature_root, args.limit_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=list_collate, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=list_collate, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=list_collate, num_workers=0)
    positives = float(sum(item["label"].item() for item in train_set))
    negatives = float(len(train_set) - positives)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1.0), device=args.device))
    model = PSICHICSequenceEdgeClassifier(dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score, best_epoch = -float("inf"), 0
    best_path, history = output / "best.pt", []
    for epoch in range(1, args.epochs + 1):
        _, _, train_loss = run_epoch(model, train_loader, optimizer, criterion, args.device, True)
        valid_y, valid_z, valid_loss = run_epoch(model, valid_loader, optimizer, criterion, args.device, False)
        valid_result = binary_metrics(valid_y, valid_z)
        entry = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_metrics": valid_result}
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        score = valid_result["ROC-AUC"]
        if best_epoch == 0 or (np.isfinite(score) and score > best_score):
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_metrics": valid_result}, best_path)
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    valid_y, valid_z, _ = run_epoch(model, valid_loader, optimizer, criterion, args.device, False)
    test_y, test_z, _ = run_epoch(model, test_loader, optimizer, criterion, args.device, False)
    result = {
        "version": "external-benchmark-psichic-sequence-edge-kiba-binary-v1",
        "model": "PSICHIC-sequence-edge",
        "task": "KIBA binary interaction classification",
        "data_root": str(data_root), "feature_root": str(feature_root), "output": str(output),
        "seed": args.seed, "device": str(args.device),
        "counts": {"train": len(train_set), "valid": len(valid_set), "test": len(test_set)},
        "best_epoch": int(checkpoint["epoch"]), "selection_metric": "validation ROC-AUC",
        "contact_map_used": False, "protein_edge_policy": "sequence_offsets_plus_minus_1_and_2",
        "validation_metrics": binary_metrics(valid_y, valid_z), "test_metrics": binary_metrics(test_y, test_z),
        "test_used_for_selection": False, "history": history,
    }
    np.savez_compressed(output / "valid_predictions.npz", labels=valid_y, logits=valid_z)
    np.savez_compressed(output / "test_predictions.npz", labels=test_y, logits=test_z)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result

