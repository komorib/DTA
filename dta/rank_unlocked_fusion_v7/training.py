"""Deterministic fold training for rank-unlocked fusion v7."""

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from dta.rank_fusion_v4.metrics import affinity_metrics

from .losses import rank_unlocked_objective
from .model import RankUnlockedFusionV7
from .selection import SEEDS, verify_oof_coverage


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _finite_positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be finite")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        raise ValueError(name + " must be finite")
    return result


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 160
    patience: int = 20
    gradient_clip: float = 1.0
    latent_widths: tuple = (1024, 640, 1024)
    model_width: int = 256
    transformer_layers: int = 3
    attention_heads: int = 8
    dropout: float = 0.10

    def __post_init__(self):
        object.__setattr__(self, "batch_size", _positive_integer(self.batch_size, "batch size"))
        object.__setattr__(
            self, "maximum_epochs", _positive_integer(self.maximum_epochs, "maximum epochs")
        )
        object.__setattr__(self, "patience", _positive_integer(self.patience, "patience"))
        object.__setattr__(self, "model_width", _positive_integer(self.model_width, "model width"))
        object.__setattr__(
            self,
            "transformer_layers",
            _positive_integer(self.transformer_layers, "transformer layers"),
        )
        object.__setattr__(
            self,
            "attention_heads",
            _positive_integer(self.attention_heads, "attention heads"),
        )
        object.__setattr__(
            self, "learning_rate", _finite_positive(self.learning_rate, "learning rate")
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_positive(self.weight_decay, "weight decay", allow_zero=True),
        )
        object.__setattr__(
            self, "gradient_clip", _finite_positive(self.gradient_clip, "gradient clip")
        )
        try:
            widths = tuple(int(width) for width in self.latent_widths)
        except (TypeError, ValueError) as error:
            raise ValueError("latent widths are invalid") from error
        if len(widths) != 3 or any(width <= 0 for width in widths):
            raise ValueError("latent widths are invalid")
        object.__setattr__(self, "latent_widths", widths)
        if self.model_width % self.attention_heads:
            raise ValueError("model width must be divisible by attention heads")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout is invalid")
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "device", str(torch.device(self.device)))

    def model_kwargs(self):
        return {
            "latent_widths": self.latent_widths,
            "model_width": self.model_width,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
        }


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device=device, non_blocking=True)
        elif key == "latents":
            result[key] = [tensor.to(device=device, non_blocking=True) for tensor in value]
        else:
            result[key] = value
    return result


def _loader(dataset, indices, config, collate_fn, *, shuffle, seed):
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("training indices must be nonempty")
    subset = torch.utils.data.Subset(dataset, values.tolist())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.device(config.device).type == "cuda",
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
    )


def _new_model(config):
    return RankUnlockedFusionV7(**config.model_kwargs()).to(config.device)


def _train_epoch(model, loader, optimizer, scheduler, config):
    model.train()
    total = 0.0
    rows = 0
    for host_batch in loader:
        batch = _move_batch(host_batch, config.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, _ = rank_unlocked_objective(
            prediction, batch["labels"], batch["base_predictions"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite v7 training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("nonfinite v7 gradient")
        optimizer.step()
        count = int(prediction.shape[0])
        total += float(loss.detach().cpu()) * count
        rows += count
    scheduler.step()
    return total / rows


def _predict_model(model, loader, config, with_labels=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for host_batch in loader:
            batch = _move_batch(host_batch, config.device)
            prediction = model(batch)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("nonfinite v7 prediction")
            predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
            if with_labels:
                labels.append(batch["labels"].detach().cpu().numpy().astype(np.float64))
    prediction_values = np.concatenate(predictions)
    if not with_labels:
        return prediction_values
    return prediction_values, np.concatenate(labels)


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_fixed_epochs(dataset, indices, seed, epochs, config, collate_fn):
    epochs = _positive_integer(epochs, "epochs")
    if epochs > config.maximum_epochs:
        raise ValueError("epochs exceed configured maximum")
    _seed_everything(seed)
    model = _new_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        loader = _loader(
            dataset,
            indices,
            config,
            collate_fn,
            shuffle=True,
            seed=int(seed) * 1000 + epoch,
        )
        loss = _train_epoch(model, loader, optimizer, scheduler, config)
        history.append({"epoch": epoch, "loss": loss})
    return {
        "schema_version": 1,
        "seed": int(seed),
        "epoch": epochs,
        "model_config": config.model_kwargs(),
        "model_state_dict": _cpu_state_dict(model),
        "history": history,
        "training_config": asdict(config),
    }


def predict_checkpoint(checkpoint, dataset, indices, config, collate_fn):
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint is invalid")
    model_config = dict(checkpoint.get("model_config", {}))
    model = RankUnlockedFusionV7(**model_config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = _loader(
        dataset, indices, config, collate_fn, shuffle=False, seed=0
    )
    return _predict_model(model, loader, config)


def select_epoch(dataset, fit_indices, stop_indices, seed, config, collate_fn):
    _seed_everything(seed)
    model = _new_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.maximum_epochs
    )
    best_score = None
    best_epoch = 1
    stale = 0
    history = []
    stop_loader = _loader(
        dataset, stop_indices, config, collate_fn, shuffle=False, seed=0
    )
    for epoch in range(1, config.maximum_epochs + 1):
        fit_loader = _loader(
            dataset,
            fit_indices,
            config,
            collate_fn,
            shuffle=True,
            seed=int(seed) * 1000 + epoch,
        )
        train_loss = _train_epoch(model, fit_loader, optimizer, scheduler, config)
        predictions, labels = _predict_model(
            model, stop_loader, config, with_labels=True
        )
        metrics = affinity_metrics(labels, predictions)
        score = (float(metrics["CI"]), -float(metrics["RMSE"]))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "stop_metrics": dict(metrics),
        })
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return best_epoch, history


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cross_fit(dataset, fold_ids, output, config, collate_fn, resume=False):
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if fold_ids.shape != (len(dataset),):
        raise ValueError("fold IDs do not match the dataset")
    folds = tuple(sorted(int(value) for value in np.unique(fold_ids)))
    if len(folds) < 2 or folds != tuple(range(len(folds))):
        raise ValueError("fold IDs must be contiguous from zero")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    oof = {seed: np.full(len(dataset), np.nan, dtype=np.float64) for seed in SEEDS}
    coverage = {seed: np.zeros(len(dataset), dtype=np.int64) for seed in SEEDS}
    checkpoints = {seed: [] for seed in SEEDS}
    histories = []
    all_indices = np.arange(len(dataset), dtype=np.int64)
    for seed in SEEDS:
        for fold in folds:
            holdout = all_indices[fold_ids == fold]
            outer_train = all_indices[fold_ids != fold]
            stop_fold = folds[(fold + 1) % len(folds)]
            stop_indices = all_indices[fold_ids == stop_fold]
            fit_indices = all_indices[(fold_ids != fold) & (fold_ids != stop_fold)]
            path = output / ("seed_%d" % seed) / ("fold_%d.pt" % fold)
            print({"event": "v7_fold_start", "seed": seed, "fold": fold, "train_count": int(outer_train.size), "holdout_count": int(holdout.size)}, flush=True)
            if path.exists():
                if not resume:
                    raise ValueError("v7 checkpoint already exists: " + str(path))
                checkpoint = torch.load(str(path), map_location="cpu")
                if checkpoint.get("seed") != seed or checkpoint.get("outer_fold") != fold:
                    raise ValueError("resume checkpoint lineage does not match")
                select_history = checkpoint.get("selection_history", [])
            else:
                best_epoch, select_history = select_epoch(
                    dataset,
                    fit_indices,
                    stop_indices,
                    seed + 100 * fold,
                    config,
                    collate_fn,
                )
                checkpoint = fit_fixed_epochs(
                    dataset,
                    outer_train,
                    seed + 100 * fold,
                    best_epoch,
                    config,
                    collate_fn,
                )
                checkpoint["seed"] = seed
                checkpoint["outer_fold"] = fold
                checkpoint["selected_epoch"] = best_epoch
                checkpoint["selection_history"] = select_history
                _atomic_torch_save(checkpoint, path)
            prediction = predict_checkpoint(
                checkpoint, dataset, holdout, config, collate_fn
            )
            oof[seed][holdout] = prediction
            coverage[seed][holdout] += 1
            checkpoints[seed].append(str(path.resolve()))
            histories.append({
                "seed": seed,
                "fold": fold,
                "selected_epoch": int(checkpoint["selected_epoch"]),
                "history": select_history,
            })
            print({"event": "v7_fold_complete", "seed": seed, "fold": fold, "selected_epoch": int(checkpoint["selected_epoch"]), "checkpoint": str(path.resolve())}, flush=True)
    verify_oof_coverage(coverage, len(dataset))
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("OOF predictions are incomplete")
    return {
        "oof_predictions": oof,
        "coverage": coverage,
        "checkpoints": checkpoints,
        "histories": histories,
    }


def predict_cross_fitted_ensemble(checkpoints, dataset, config, collate_fn):
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(SEEDS):
        raise ValueError("checkpoint inventory must contain all seeds")
    indices = np.arange(len(dataset), dtype=np.int64)
    result = {}
    for seed in SEEDS:
        members = []
        for path in checkpoints[seed]:
            checkpoint = torch.load(str(path), map_location="cpu")
            members.append(
                predict_checkpoint(checkpoint, dataset, indices, config, collate_fn)
            )
        if not members:
            raise ValueError("seed checkpoint inventory is empty")
        result[seed] = np.mean(np.stack(members, axis=0), axis=0)
    return result


__all__ = [
    "TrainingConfig",
    "cross_fit",
    "fit_fixed_epochs",
    "predict_checkpoint",
    "predict_cross_fitted_ensemble",
    "select_epoch",
]
