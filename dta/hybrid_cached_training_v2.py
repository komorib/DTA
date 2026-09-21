"""Cached upper-bound fusion training for LP-PDBBind."""

import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils.hybrid_affinity_dataset import LabelSpreadBatchSampler
from data_utils.hybrid_feature_cache_v2 import (
    CachedHybridFeatureDataset,
    cache_matches_provenance,
    collate_cached_hybrid_features,
)
from net.hybrid_upper_bound import HybridUpperBoundFusion
from utils.experiment_integrity import atomic_write_json, sha256_file


def upper_bound_affinity_loss(prediction, target, expert_weights):
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    weights = expert_weights.float()
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    differences = target[:, None] - target[None, :]
    pair_mask = differences > 0
    if bool(pair_mask.any()):
        prediction_difference = prediction[:, None] - prediction[None, :]
        pairwise = F.softplus(-prediction_difference[pair_mask]).mean()
    else:
        pairwise = prediction.new_zeros(())
    order = torch.argsort(target, descending=True)
    sorted_prediction = prediction[order]
    if sorted_prediction.numel() > 1:
        listmle = torch.stack(
            [
                torch.logsumexp(sorted_prediction[index:], dim=0)
                - sorted_prediction[index]
                for index in range(sorted_prediction.numel() - 1)
            ]
        ).mean()
    else:
        listmle = prediction.new_zeros(())
    load_balance = (weights.mean(dim=0) - 1.0 / weights.shape[1]).square().sum()
    loss = smooth_l1 + 0.05 * mse + 0.75 * pairwise + 0.05 * listmle + 0.01 * load_balance
    return {
        "loss": loss,
        "smooth_l1": smooth_l1,
        "mse": mse,
        "pairwise": pairwise,
        "listmle": listmle,
        "load_balance": load_balance,
    }


def is_better_validation(candidate, current, tolerance=1e-6):
    if current is None:
        return True
    ci_delta = float(candidate["CI"]) - float(current["CI"])
    if ci_delta > tolerance:
        return True
    if ci_delta < -tolerance:
        return False
    spearman_delta = float(candidate["Spearman"]) - float(current["Spearman"])
    if spearman_delta > tolerance:
        return True
    if spearman_delta < -tolerance:
        return False
    return float(candidate["RMSE"]) < float(current["RMSE"]) - tolerance


def select_preflight_batch_size(candidates, probe):
    for batch_size in candidates:
        try:
            if probe(int(batch_size)):
                return int(batch_size)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("no candidate batch size fits the selected device")


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _concordance_index(target, prediction):
    total = 0
    concordant = 0.0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            if target[left] == target[right]:
                continue
            total += 1
            sign = 1.0 if target[left] > target[right] else -1.0
            delta = prediction[left] - prediction[right]
            if sign * delta > 0.0:
                concordant += 1.0
            elif delta == 0.0:
                concordant += 0.5
    return concordant / total if total else 0.5


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    centered_target = target - target.mean()
    centered_prediction = prediction - prediction.mean()
    denominator = np.linalg.norm(centered_target) * np.linalg.norm(centered_prediction)
    pearson = float(np.dot(centered_target, centered_prediction) / denominator) if denominator else 0.0
    target_rank = _rank(target)
    prediction_rank = _rank(prediction)
    rank_denominator = np.linalg.norm(target_rank - target_rank.mean()) * np.linalg.norm(prediction_rank - prediction_rank.mean())
    spearman = float(np.dot(target_rank - target_rank.mean(), prediction_rank - prediction_rank.mean()) / rank_denominator) if rank_denominator else 0.0
    return {
        "CI": _concordance_index(target, prediction),
        "MAE": float(np.abs(target - prediction).mean()),
        "RMSE": float(np.sqrt(np.square(target - prediction).mean())),
        "Pearson": pearson,
        "Spearman": spearman,
    }


@torch.inference_mode()
def evaluate_cached_fusion(model, loader, device):
    model.eval()
    predictions, targets, keys = [], [], []
    for batch in loader:
        keys.extend(batch["keys"])
        batch = _move_to_device(batch, device)
        output = model(batch)
        predictions.extend(output["prediction"].reshape(-1).float().cpu().tolist())
        targets.extend(batch["label"].reshape(-1).float().cpu().tolist())
    result = _metrics(targets, predictions)
    return result, keys, np.asarray(targets), np.asarray(predictions)


def _cache_provenance(dataset):
    metadata = dataset.metadata
    required = {
        name: metadata.get(name)
        for name in (
            "manifest_sha256",
            "gerna_checkpoint_sha256",
            "interface_checkpoint_sha256",
        )
    }
    if not all(required.values()) or not cache_matches_provenance(metadata, required):
        raise RuntimeError("cache provenance is incomplete or invalid")
    return required


def _save_checkpoint(path, payload):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_cached_fusion(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal cached training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError("output is nonempty; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    train_dataset = CachedHybridFeatureDataset(args.train_cache)
    valid_dataset = CachedHybridFeatureDataset(args.valid_cache)
    provenance = {
        "train": _cache_provenance(train_dataset),
        "valid": _cache_provenance(valid_dataset),
    }
    train_labels = [float(train_dataset[index]["label"].item()) for index in range(len(train_dataset))]
    sampler = LabelSpreadBatchSampler(train_labels, args.batch_size, args.seed, min_gap=0.05)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_cached_hybrid_features)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_cached_hybrid_features)
    model = HybridUpperBoundFusion(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler()
    start_epoch, best, stale, history = 0, None, 0, []
    checkpoint_path = output_dir / "training_checkpoint.pth"
    if args.resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, stale = int(state["epoch"]) + 1, state["best"], int(state["stale"])
        history = list(state.get("history", []))
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses = []
        for batch in train_loader:
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch)
                components = upper_bound_affinity_loss(output["prediction"], batch["label"], output["expert_weights"])
            scaler.scale(components["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(components["loss"].detach().cpu()))
        scheduler.step()
        metrics, keys, target, prediction = evaluate_cached_fusion(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": float(optimizer.param_groups[0]["lr"]), **metrics}
        history.append(row)
        if is_better_validation(metrics, best):
            best, stale = dict(metrics), 0
            _save_checkpoint(output_dir / "Model_cached_best.pth", {"model_state_dict": model.state_dict(), "metrics": best, "epoch": epoch})
            np.savez_compressed(output_dir / "valid_predictions.npz", keys=np.asarray(keys), labels=target, predictions=prediction)
        else:
            stale += 1
        _save_checkpoint(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale, "history": history, "provenance": provenance})
        if stale >= args.patience:
            break
    _save_checkpoint(output_dir / "Model_cached_last.pth", {"model_state_dict": model.state_dict(), "metrics": best})
    atomic_write_json(output_dir / "best_metrics.json", best)
    atomic_write_json(output_dir / "run_config.json", {"seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs, "learning_rate": args.learning_rate, "device": str(device)})
    atomic_write_json(output_dir / "provenance.json", provenance)
    atomic_write_json(output_dir / "candidate_cached.json", {
        "split": "valid",
        "model_sha256": sha256_file(output_dir / "Model_cached_best.pth"),
        "valid_manifest_sha256": provenance["valid"]["manifest_sha256"],
        "variant": "cached",
    })
    with (output_dir / "metrics_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return best
