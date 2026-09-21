"""Cross-fitted training and resumable artifacts for calibration v6."""

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from types import MappingProxyType
import tempfile

import numpy as np
import torch

from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from .calibrators import MonotonicSplineCalibrator, PositiveAffineCalibrator
from .data import collate_rank_locked_calibration
from .model import ConditionalCalibrator, calibration_loss
from .projection import rank_locked_project


FORMAL_SEEDS = (1, 7, 21)
_ANALYTIC_CANDIDATES = {
    "affine_mse": (PositiveAffineCalibrator, {"objective": "mse"}),
    "affine_huber": (PositiveAffineCalibrator, {"objective": "huber"}),
    "spline_8_knots": (MonotonicSplineCalibrator, {"knot_count": 8}),
    "spline_16_knots": (MonotonicSplineCalibrator, {"knot_count": 16}),
}
_COMPLETED_FILES = (
    "lineage.json",
    "checkpoint_last.pt",
    "checkpoint_best.pt",
    "predictions.npz",
    "result.json",
    "diagnostics.json",
)


def _positive_integer(value, name, *, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
        or (maximum is not None and int(value) > maximum)
    ):
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (nonnegative and float(value) < 0.0)
    ):
        raise ValueError(name + " must be finite")
    return float(value)


def _finite_vector(value, name, *, count=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a finite vector") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(name + " must be a nonempty vector")
    if count is not None and result.shape != (count,):
        raise ValueError(name + " count does not match")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _canonical_json_bytes(value):
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("payload must be canonical JSON compatible") from error
    return encoded


def _plain_json(value):
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _validate_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class ConditionalTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 256
    maximum_epochs: int = 200
    patience: int = 20
    gradient_clip: float = 1.0
    device: str = "cuda:0"

    def __post_init__(self):
        object.__setattr__(
            self,
            "learning_rate",
            _finite_number(self.learning_rate, "learning rate", positive=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_number(self.weight_decay, "weight decay", nonnegative=True),
        )
        object.__setattr__(
            self, "batch_size", _positive_integer(self.batch_size, "batch size")
        )
        object.__setattr__(
            self,
            "maximum_epochs",
            _positive_integer(
                self.maximum_epochs, "maximum epochs", maximum=200
            ),
        )
        object.__setattr__(
            self, "patience", _positive_integer(self.patience, "patience")
        )
        object.__setattr__(
            self,
            "gradient_clip",
            _finite_number(self.gradient_clip, "gradient clip", positive=True),
        )
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device is invalid") from error
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        object.__setattr__(self, "device", str(device))

    def to_dict(self):
        return {
            "schema_version": 1,
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "maximum_epochs": self.maximum_epochs,
            "patience": self.patience,
            "gradient_clip": self.gradient_clip,
            "device": self.device,
        }


class NonfiniteTrainingError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        if diagnostics is None:
            diagnostics = {}
        if not isinstance(diagnostics, Mapping):
            raise ValueError("nonfinite diagnostics must be a mapping")
        self.diagnostics = _plain_json(dict(diagnostics))


@dataclass(frozen=True)
class FoldFitResult:
    fold: int
    predictions: np.ndarray
    best_epoch: int
    best_loss: float
    model_state: Mapping
    history: tuple

    def __post_init__(self):
        if isinstance(self.fold, bool) or not isinstance(self.fold, Integral):
            raise ValueError("fold must be an integer")
        predictions = _finite_vector(self.predictions, "fold predictions").copy()
        predictions.setflags(write=False)
        if not isinstance(self.model_state, Mapping):
            raise ValueError("fold model state must be a mapping")
        try:
            history = tuple(self.history)
        except TypeError as error:
            raise ValueError("fold history must be iterable") from error
        object.__setattr__(self, "fold", int(self.fold))
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(
            self, "best_epoch", _positive_integer(self.best_epoch, "best epoch")
        )
        object.__setattr__(
            self,
            "best_loss",
            _finite_number(self.best_loss, "best loss", nonnegative=True),
        )
        object.__setattr__(self, "model_state", dict(self.model_state))
        object.__setattr__(self, "history", history)


@dataclass(frozen=True)
class FinalFitResult:
    epoch: int
    final_loss: float
    model_config: Mapping
    model_state: Mapping
    history: tuple

    def __post_init__(self):
        if not isinstance(self.model_config, Mapping):
            raise ValueError("final model config must be a mapping")
        if not isinstance(self.model_state, Mapping):
            raise ValueError("final model state must be a mapping")
        try:
            history = tuple(self.history)
        except TypeError as error:
            raise ValueError("final history must be iterable") from error
        object.__setattr__(
            self, "epoch", _positive_integer(self.epoch, "final epoch", maximum=200)
        )
        object.__setattr__(
            self,
            "final_loss",
            _finite_number(self.final_loss, "final loss", nonnegative=True),
        )
        object.__setattr__(self, "model_config", _plain_json(self.model_config))
        object.__setattr__(self, "model_state", dict(self.model_state))
        object.__setattr__(self, "history", history)


@dataclass(frozen=True)
class AnalyticFoldResult:
    fold: int
    train_indices: tuple
    holdout_indices: tuple
    model: Mapping
    metrics: Mapping


@dataclass(frozen=True)
class OOFResult:
    predictions: np.ndarray
    held_out_count: np.ndarray
    fold_results: tuple
    metrics: Mapping
    seed: object = None

    def __post_init__(self):
        predictions = _finite_vector(self.predictions, "OOF predictions").copy()
        held = np.asarray(self.held_out_count, dtype=np.int64)
        if held.shape != predictions.shape or np.any(held != 1):
            raise ValueError("OOF held-out coverage must equal one per row")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("OOF metrics must be a mapping")
        predictions.setflags(write=False)
        held = held.copy()
        held.setflags(write=False)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "held_out_count", held)
        object.__setattr__(self, "fold_results", tuple(self.fold_results))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def best_epochs(self):
        values = []
        for result in self.fold_results:
            if isinstance(result, FoldFitResult):
                values.append(result.best_epoch)
        return tuple(values)


@dataclass(frozen=True)
class ConditionalInventoryResult:
    seed_predictions: Mapping
    failures: Mapping
    ensemble_predictions: object
    ensemble_unprojected: object
    projection_report: object


@dataclass(frozen=True)
class CompletedRun:
    result: Mapping
    diagnostics: Mapping
    predictions: np.ndarray
    checkpoint_last: Mapping
    checkpoint_best: Mapping
    reused: bool = True


def _dataset_vectors(dataset):
    count = len(dataset)
    if count <= 0:
        raise ValueError("dataset must be nonempty")
    labels = np.empty(count, dtype=np.float64)
    base = np.empty(count, dtype=np.float64)
    for index in range(count):
        record = dataset[index]
        if not isinstance(record, Mapping):
            raise ValueError("dataset records must be mappings")
        labels[index] = record.get("label")
        base[index] = record.get("base_prediction")
    if not np.isfinite(labels).all() or not np.isfinite(base).all():
        raise ValueError("dataset labels and base predictions must be finite")
    return labels, base


def _validated_folds(fold_ids, count):
    try:
        folds = np.asarray(fold_ids, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("fold IDs must be integer values") from error
    if folds.shape != (count,):
        raise ValueError("fold IDs count does not match dataset")
    unique = np.unique(folds)
    if unique.shape[0] < 2 or not np.array_equal(
        unique, np.arange(unique.shape[0], dtype=np.int64)
    ):
        raise ValueError("fold IDs must be contiguous and nonempty")
    return folds, tuple(int(value) for value in unique)


def _fit_analytic(candidate, score, label):
    if candidate not in _ANALYTIC_CANDIDATES:
        raise ValueError("analytic candidate is invalid")
    calibrator, arguments = _ANALYTIC_CANDIDATES[candidate]
    return calibrator.fit(score, label, **arguments)


def cross_fit_analytic(dataset, fold_ids, candidate):
    labels, base = _dataset_vectors(dataset)
    folds, unique = _validated_folds(fold_ids, len(dataset))
    predictions = np.empty(len(dataset), dtype=np.float64)
    held_out_count = np.zeros(len(dataset), dtype=np.int64)
    fold_results = []
    for fold in unique:
        train_indices = np.flatnonzero(folds != fold)
        holdout_indices = np.flatnonzero(folds == fold)
        model = _fit_analytic(
            candidate, base[train_indices], labels[train_indices]
        )
        held_predictions = model.predict(base[holdout_indices])
        predictions[holdout_indices] = held_predictions
        held_out_count[holdout_indices] += 1
        fold_results.append(
            AnalyticFoldResult(
                fold=fold,
                train_indices=tuple(int(value) for value in train_indices),
                holdout_indices=tuple(int(value) for value in holdout_indices),
                model=model.to_dict(),
                metrics=affinity_metrics(
                    labels[holdout_indices], held_predictions
                ),
            )
        )
    return OOFResult(
        predictions=predictions,
        held_out_count=held_out_count,
        fold_results=tuple(fold_results),
        metrics=affinity_metrics(labels, predictions),
    )


def cross_fit_conditional(
    dataset,
    fold_ids,
    *,
    seed,
    config,
    fit_fold_fn=None,
):
    if seed not in FORMAL_SEEDS:
        raise ValueError("conditional seed must be one of 1, 7, or 21")
    if not isinstance(config, ConditionalTrainingConfig):
        raise ValueError("conditional training config is invalid")
    if fit_fold_fn is None:
        fit_fold_fn = fit_conditional_fold
    if not callable(fit_fold_fn):
        raise ValueError("fit_fold_fn must be callable")
    labels, _ = _dataset_vectors(dataset)
    folds, unique = _validated_folds(fold_ids, len(dataset))
    predictions = np.empty(len(dataset), dtype=np.float64)
    held_out_count = np.zeros(len(dataset), dtype=np.int64)
    fold_results = []
    for fold in unique:
        train_indices = np.flatnonzero(folds != fold)
        holdout_indices = np.flatnonzero(folds == fold)
        result = fit_fold_fn(
            dataset=dataset,
            train_indices=train_indices.copy(),
            holdout_indices=holdout_indices.copy(),
            seed=seed,
            fold=fold,
            config=config,
        )
        if not isinstance(result, FoldFitResult) or result.fold != fold:
            raise ValueError("conditional fold result does not match")
        if result.predictions.shape != holdout_indices.shape:
            raise ValueError("conditional fold predictions do not match holdout")
        predictions[holdout_indices] = result.predictions
        held_out_count[holdout_indices] += 1
        fold_results.append(result)
    return OOFResult(
        predictions=predictions,
        held_out_count=held_out_count,
        fold_results=tuple(fold_results),
        metrics=affinity_metrics(labels, predictions),
        seed=seed,
    )


def _validated_indices(indices, name, count):
    try:
        values = np.asarray(indices, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " indices must be integers") from error
    if values.ndim != 1 or values.size == 0:
        raise ValueError(name + " indices must be nonempty")
    if (
        np.any(values < 0)
        or np.any(values >= count)
        or np.unique(values).shape[0] != values.shape[0]
    ):
        raise ValueError(name + " indices are invalid")
    return values


def _latent_widths(dataset):
    record = dataset[0]
    if not isinstance(record, Mapping):
        raise ValueError("dataset records must be mappings")
    latents = record.get("latents")
    if not isinstance(latents, (tuple, list)) or len(latents) != 3:
        raise ValueError("dataset latents must contain three values")
    widths = []
    for latent in latents:
        value = np.asarray(latent)
        if value.ndim != 1 or value.shape[0] <= 0 or not np.isfinite(value).all():
            raise ValueError("dataset latent is invalid")
        widths.append(int(value.shape[0]))
    return tuple(widths)


def _seed_everything(seed, stream):
    combined = int(seed) * 1009 + int(stream)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(combined % (2 ** 32))
    torch.manual_seed(combined)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(combined)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return combined


def _training_device(config):
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    return device


def _move_batch(batch, device):
    return {
        "keys": batch["keys"],
        "labels": batch["labels"].to(device),
        "base_predictions": batch["base_predictions"].to(device),
        "raw_predictions": batch["raw_predictions"].to(device),
        "standardized_predictions": batch["standardized_predictions"].to(device),
        "available": batch["available"].to(device),
        "latents": [value.to(device) for value in batch["latents"]],
    }


def _loader(dataset, indices, config, *, shuffle, generator_seed):
    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [int(value) for value in indices]),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_rank_locked_calibration,
        generator=generator,
        drop_last=False,
    )


def _cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _finite_gradients(model, *, seed, fold, epoch):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise NonfiniteTrainingError(
                "nonfinite conditional gradient",
                diagnostics={
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "parameter": name,
                    "stage": "backward",
                },
            )


def _train_epoch(model, loader, optimizer, config, device, *, seed, fold, epoch):
    model.train()
    loss_sum = 0.0
    row_count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = calibration_loss(prediction, batch["labels"])
        if not torch.isfinite(loss):
            raise NonfiniteTrainingError(
                "nonfinite conditional loss",
                diagnostics={
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "stage": "forward",
                    "sample_keys": list(batch["keys"]),
                },
            )
        loss.backward()
        _finite_gradients(model, seed=seed, fold=fold, epoch=epoch)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(norm)):
            raise NonfiniteTrainingError(
                "nonfinite conditional gradient norm",
                diagnostics={
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "stage": "gradient_clip",
                },
            )
        optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        row_count += batch_size
    if row_count == 0:
        raise ValueError("training loader is empty")
    result = loss_sum / row_count
    if not math.isfinite(result):
        raise NonfiniteTrainingError("nonfinite conditional epoch loss")
    return result


def _predict_indices(model, dataset, indices, config, device):
    loader = _loader(
        dataset, indices, config, shuffle=False, generator_seed=0
    )
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            prediction = model(batch)
            if not torch.isfinite(prediction).all():
                raise NonfiniteTrainingError(
                    "nonfinite conditional prediction",
                    diagnostics={"stage": "evaluation"},
                )
            predictions.append(prediction.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
    prediction = torch.cat(predictions, dim=0)
    label = torch.cat(labels, dim=0)
    loss = calibration_loss(prediction, label)
    if not torch.isfinite(loss):
        raise NonfiniteTrainingError(
            "nonfinite conditional evaluation loss",
            diagnostics={"stage": "evaluation"},
        )
    return np.asarray(prediction.numpy(), dtype=np.float64), float(loss)


def _new_model(dataset, fit_indices, labels, base, device):
    anchor = PositiveAffineCalibrator.fit(
        base[fit_indices], labels[fit_indices], objective="mse"
    )
    model = ConditionalCalibrator(
        _latent_widths(dataset), anchor.slope, anchor.intercept
    ).to(device)
    return model


def fit_conditional_fold(
    *,
    dataset,
    train_indices,
    holdout_indices,
    seed,
    fold,
    config,
):
    if seed not in FORMAL_SEEDS:
        raise ValueError("conditional seed must be one of 1, 7, or 21")
    if not isinstance(config, ConditionalTrainingConfig):
        raise ValueError("conditional training config is invalid")
    labels, base = _dataset_vectors(dataset)
    train_indices = _validated_indices(
        train_indices, "train", len(dataset)
    )
    holdout_indices = _validated_indices(
        holdout_indices, "holdout", len(dataset)
    )
    if np.intersect1d(train_indices, holdout_indices).size:
        raise ValueError("train and holdout indices overlap")
    if set(train_indices).union(set(holdout_indices)) != set(range(len(dataset))):
        raise ValueError("train and holdout indices do not cover the dataset")
    if isinstance(fold, bool) or not isinstance(fold, Integral) or int(fold) < 0:
        raise ValueError("fold must be a nonnegative integer")
    fold = int(fold)
    combined_seed = _seed_everything(seed, fold)
    device = _training_device(config)
    model = _new_model(dataset, train_indices, labels, base, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        dataset,
        train_indices,
        config,
        shuffle=True,
        generator_seed=combined_seed,
    )
    best_state = None
    best_epoch = None
    best_loss = float("inf")
    history = []
    stale = 0
    for epoch in range(1, config.maximum_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            seed=seed,
            fold=fold,
            epoch=epoch,
        )
        _, holdout_loss = _predict_indices(
            model, dataset, holdout_indices, config, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "holdout_loss": holdout_loss,
            }
        )
        if holdout_loss < best_loss:
            best_loss = holdout_loss
            best_epoch = epoch
            best_state = _cpu_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None or best_epoch is None:
        raise NonfiniteTrainingError("conditional fold did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    predictions, restored_loss = _predict_indices(
        model, dataset, holdout_indices, config, device
    )
    if abs(restored_loss - best_loss) > 1e-6:
        raise ValueError("restored fold checkpoint loss does not match")
    return FoldFitResult(
        fold=fold,
        predictions=predictions,
        best_epoch=best_epoch,
        best_loss=best_loss,
        model_state=best_state,
        history=tuple(history),
    )


def fit_final_conditional(*, dataset, seed, epochs, config):
    if seed not in FORMAL_SEEDS:
        raise ValueError("conditional seed must be one of 1, 7, or 21")
    if not isinstance(config, ConditionalTrainingConfig):
        raise ValueError("conditional training config is invalid")
    epochs = _positive_integer(epochs, "final epochs", maximum=200)
    if epochs > config.maximum_epochs:
        raise ValueError("final epochs exceed configured maximum")
    labels, base = _dataset_vectors(dataset)
    indices = np.arange(len(dataset), dtype=np.int64)
    combined_seed = _seed_everything(seed, 10000)
    device = _training_device(config)
    model = _new_model(dataset, indices, labels, base, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        dataset,
        indices,
        config,
        shuffle=True,
        generator_seed=combined_seed,
    )
    history = []
    final_loss = None
    for epoch in range(1, epochs + 1):
        final_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            seed=seed,
            fold=-1,
            epoch=epoch,
        )
        history.append({"epoch": epoch, "train_loss": final_loss})
    return FinalFitResult(
        epoch=epochs,
        final_loss=final_loss,
        model_config=model.config(),
        model_state=_cpu_state(model),
        history=tuple(history),
    )


def predict_conditional_checkpoint(checkpoint, dataset, *, config):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("conditional checkpoint must be a mapping")
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("conditional checkpoint fields do not match")
    required = {"latent_widths", "affine_slope", "affine_intercept"}
    if not required.issubset(model_config):
        raise ValueError("conditional model config is incomplete")
    if not isinstance(config, ConditionalTrainingConfig):
        raise ValueError("conditional training config is invalid")
    device = _training_device(config)
    model = ConditionalCalibrator(
        model_config["latent_widths"],
        model_config["affine_slope"],
        model_config["affine_intercept"],
    ).to(device)
    try:
        model.load_state_dict(dict(model_state), strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("conditional checkpoint state does not match") from error
    indices = np.arange(len(dataset), dtype=np.int64)
    predictions, _ = _predict_indices(model, dataset, indices, config, device)
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions.setflags(write=False)
    return predictions


def final_epoch_from_best_epochs(best_epochs):
    try:
        values = tuple(best_epochs)
    except TypeError as error:
        raise ValueError("best epochs must contain five values") from error
    if len(values) != 5 or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError("best epochs must contain five integer values")
    epoch = int(math.ceil(float(np.median(np.asarray(values, dtype=np.float64)))))
    return min(200, max(1, epoch))


def ensemble_then_project(base_predictions, seed_predictions, project_fn=None):
    base = _finite_vector(base_predictions, "base predictions")
    if not isinstance(seed_predictions, Mapping) or set(seed_predictions) != set(
        FORMAL_SEEDS
    ):
        raise ValueError("ensemble requires seeds 1, 7, and 21")
    members = [
        _finite_vector(
            seed_predictions[seed], "seed predictions", count=base.shape[0]
        )
        for seed in FORMAL_SEEDS
    ]
    mean = np.mean(np.stack(members, axis=0), axis=0, dtype=np.float64)
    if project_fn is None:
        project_fn = rank_locked_project
    projected, report = project_fn(base, mean)
    projected = _finite_vector(
        projected, "ensemble projected predictions", count=base.shape[0]
    ).copy()
    mean = mean.copy()
    projected.setflags(write=False)
    mean.setflags(write=False)
    return projected, report, mean


def fit_conditional_inventory(base_predictions, *, fit_seed_fn):
    base = _finite_vector(base_predictions, "base predictions")
    if not callable(fit_seed_fn):
        raise ValueError("fit_seed_fn must be callable")
    predictions = {}
    failures = {}
    for seed in FORMAL_SEEDS:
        try:
            value = fit_seed_fn(seed)
            prediction = _finite_vector(
                value, "conditional seed predictions", count=base.shape[0]
            ).copy()
            prediction.setflags(write=False)
            predictions[seed] = prediction
        except NonfiniteTrainingError as error:
            diagnostic = {
                "status": "failed_nonfinite",
                "seed": seed,
                "error": str(error),
            }
            diagnostic.update(error.diagnostics)
            failures[seed] = diagnostic
        except ValueError as error:
            if "must be finite" not in str(error):
                raise
            failures[seed] = {
                "status": "failed_nonfinite",
                "seed": seed,
                "error": str(error),
                "stage": "prediction_export",
            }
    ensemble = None
    unprojected = None
    report = None
    if not failures:
        ensemble, report, unprojected = ensemble_then_project(base, predictions)
    return ConditionalInventoryResult(
        seed_predictions=MappingProxyType(dict(predictions)),
        failures=MappingProxyType(copy.deepcopy(failures)),
        ensemble_predictions=ensemble,
        ensemble_unprojected=unprojected,
        projection_report=report,
    )


def build_run_lineage(
    *,
    candidate,
    seed,
    config,
    source_revision,
    train_identity,
    valid_identity,
    v5_package_sha256,
    v5_anchor_sha256,
    fold_map_sha256,
    normalization_sha256,
):
    if candidate != "conditional_seed_" + str(seed) or seed not in FORMAL_SEEDS:
        raise ValueError("candidate and formal seed do not match")
    if not isinstance(config, ConditionalTrainingConfig):
        raise ValueError("conditional training config is invalid")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("source revision is invalid")
    payload = {
        "schema_version": 1,
        "candidate": candidate,
        "seed": int(seed),
        "config": config.to_dict(),
        "source_revision": source_revision,
        "train_identity": _plain_json(train_identity),
        "valid_identity": _plain_json(valid_identity),
        "v5_package_sha256": _validate_sha256(
            v5_package_sha256, "v5 package hash"
        ),
        "v5_anchor_sha256": _validate_sha256(
            v5_anchor_sha256, "v5 anchor hash"
        ),
        "fold_map_sha256": _validate_sha256(
            fold_map_sha256, "fold map hash"
        ),
        "normalization_sha256": _validate_sha256(
            normalization_sha256, "normalization hash"
        ),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    payload["lineage_sha256"] = digest
    return payload


def validate_resume_lineage(actual, expected):
    try:
        actual_bytes = _canonical_json_bytes(actual)
        expected_bytes = _canonical_json_bytes(expected)
    except ValueError as error:
        raise ValueError("resume lineage does not match") from error
    if actual_bytes != expected_bytes:
        raise ValueError("resume lineage does not match")


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_bytes(path, _canonical_json_bytes(value))


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary, predictions=np.asarray(predictions, dtype=np.float64)
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_canonical_json(path, name):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(name + " is malformed") from error
    if raw != _canonical_json_bytes(value):
        raise ValueError(name + " is not canonical")
    return value


def _load_torch(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _validate_partial_run(output, expected_lineage):
    names = {path.name for path in output.iterdir()}
    allowed = set(_COMPLETED_FILES)
    if (
        not names
        or "complete.json" in names
        or "lineage.json" not in names
        or not names.issubset(allowed)
    ):
        raise ValueError("resume lineage does not match")
    try:
        actual = _load_canonical_json(output / "lineage.json", "run lineage")
        validate_resume_lineage(actual, expected_lineage)
    except ValueError as error:
        raise ValueError("resume lineage does not match") from error


def save_completed_run(
    output,
    *,
    lineage,
    checkpoint_last,
    checkpoint_best,
    result,
    diagnostics,
    predictions,
    allow_matching_partial=False,
):
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not allow_matching_partial:
            raise ValueError("completed run output already exists")
        _validate_partial_run(output, lineage)
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(checkpoint_last, Mapping) or not isinstance(
        checkpoint_best, Mapping
    ):
        raise ValueError("completed checkpoints must be mappings")
    if not isinstance(result, Mapping) or result.get("status") != "complete":
        raise ValueError("completed result must have complete status")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("completed diagnostics must be a mapping")
    predictions = _finite_vector(predictions, "completed predictions")
    _atomic_json(output / "lineage.json", lineage)
    _atomic_torch(output / "checkpoint_last.pt", dict(checkpoint_last))
    _atomic_torch(output / "checkpoint_best.pt", dict(checkpoint_best))
    _atomic_predictions(output / "predictions.npz", predictions)
    _atomic_json(output / "result.json", dict(result))
    _atomic_json(output / "diagnostics.json", dict(diagnostics))
    hashes = {
        name: sha256_file(output / name) for name in _COMPLETED_FILES
    }
    marker = {
        "schema_version": 1,
        "status": "complete",
        "lineage_sha256": lineage.get("lineage_sha256"),
        "artifacts": hashes,
    }
    _atomic_json(output / "complete.json", marker)
    return load_completed_run(output, lineage)


def load_completed_run(output, expected_lineage):
    output = Path(output).expanduser().resolve()
    try:
        names = {path.name for path in output.iterdir()}
        expected_names = set(_COMPLETED_FILES) | {"complete.json"}
        if names != expected_names:
            raise ValueError("completed run files do not match")
        lineage = _load_canonical_json(output / "lineage.json", "run lineage")
        validate_resume_lineage(lineage, expected_lineage)
        marker = _load_canonical_json(
            output / "complete.json", "completion marker"
        )
        if (
            not isinstance(marker, dict)
            or marker.get("schema_version") != 1
            or marker.get("status") != "complete"
            or marker.get("lineage_sha256")
            != expected_lineage.get("lineage_sha256")
            or set(marker.get("artifacts", {})) != set(_COMPLETED_FILES)
        ):
            raise ValueError("completion marker does not match")
        for name in _COMPLETED_FILES:
            if marker["artifacts"][name] != sha256_file(output / name):
                raise ValueError("completed artifact hash does not match")
        result = _load_canonical_json(output / "result.json", "run result")
        diagnostics = _load_canonical_json(
            output / "diagnostics.json", "run diagnostics"
        )
        if result.get("status") != "complete":
            raise ValueError("completed result status does not match")
        with np.load(str(output / "predictions.npz"), allow_pickle=False) as archive:
            if set(archive.files) != {"predictions"}:
                raise ValueError("completed prediction fields do not match")
            predictions = _finite_vector(
                archive["predictions"], "completed predictions"
            ).copy()
        checkpoint_last = _load_torch(output / "checkpoint_last.pt")
        checkpoint_best = _load_torch(output / "checkpoint_best.pt")
        if not isinstance(checkpoint_last, Mapping) or not isinstance(
            checkpoint_best, Mapping
        ):
            raise ValueError("completed checkpoints are malformed")
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError("completed run does not match") from error
    predictions.setflags(write=False)
    return CompletedRun(
        result=MappingProxyType(dict(result)),
        diagnostics=MappingProxyType(dict(diagnostics)),
        predictions=predictions,
        checkpoint_last=dict(checkpoint_last),
        checkpoint_best=dict(checkpoint_best),
        reused=True,
    )


def _fit_seed_models(
    dataset,
    fold_ids,
    *,
    seed,
    config,
    fit_fold_fn,
    fit_final_fn,
):
    oof = cross_fit_conditional(
        dataset,
        fold_ids,
        seed=seed,
        config=config,
        fit_fold_fn=fit_fold_fn,
    )
    final_epoch = final_epoch_from_best_epochs(oof.best_epochs)
    final = fit_final_fn(
        dataset=dataset,
        seed=seed,
        epochs=final_epoch,
        config=config,
    )
    return oof, final_epoch, final


def fit_conditional_seed(
    dataset,
    fold_ids,
    *,
    seed,
    config,
    output,
    lineage,
    resume,
    fit_fold_fn=None,
    fit_final_fn=None,
):
    output = Path(output).expanduser().resolve()
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    resume_partial = False
    if output.exists() and any(output.iterdir()):
        if resume and (output / "complete.json").is_file():
            return load_completed_run(output, lineage)
        if not resume:
            raise ValueError("conditional output already exists")
        _validate_partial_run(output, lineage)
        resume_partial = True
    if fit_fold_fn is None:
        fit_fold_fn = fit_conditional_fold
    if fit_final_fn is None:
        fit_final_fn = fit_final_conditional
    try:
        oof, final_epoch, final = _fit_seed_models(
            dataset,
            fold_ids,
            seed=seed,
            config=config,
            fit_fold_fn=fit_fold_fn,
            fit_final_fn=fit_final_fn,
        )
    except NonfiniteTrainingError as error:
        output.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "schema_version": 1,
            "status": "failed_nonfinite",
            "seed": seed,
            "error": str(error),
        }
        diagnostic.update(error.diagnostics)
        _atomic_json(output / "lineage.json", lineage)
        _atomic_json(output / "diagnostics.json", diagnostic)
        raise
    if not isinstance(final, FinalFitResult) or final.epoch != final_epoch:
        raise ValueError("final conditional fit result does not match")
    checkpoint = {
        "schema_version": 1,
        "candidate": "conditional_seed_" + str(seed),
        "seed": seed,
        "epoch": final.epoch,
        "model_config": final.model_config,
        "model_state_dict": final.model_state,
        "lineage": lineage,
        "oof_best_epochs": list(oof.best_epochs),
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "candidate": "conditional_seed_" + str(seed),
        "seed": seed,
        "final_epoch": final.epoch,
        "final_train_loss": final.final_loss,
        "oof_metrics": dict(oof.metrics),
        "prediction_variant": "oof_unprojected_float64",
    }
    diagnostics = {
        "schema_version": 1,
        "status": "finite",
        "held_out_count_minimum": int(oof.held_out_count.min()),
        "held_out_count_maximum": int(oof.held_out_count.max()),
        "folds": [
            {
                "fold": fold.fold,
                "best_epoch": fold.best_epoch,
                "best_loss": fold.best_loss,
                "history": list(fold.history),
            }
            for fold in oof.fold_results
        ],
        "final_history": list(final.history),
    }
    completed = save_completed_run(
        output,
        lineage=lineage,
        checkpoint_last=checkpoint,
        checkpoint_best=checkpoint,
        result=result,
        diagnostics=diagnostics,
        predictions=oof.predictions,
        allow_matching_partial=resume_partial,
    )
    return CompletedRun(
        result=completed.result,
        diagnostics=completed.diagnostics,
        predictions=completed.predictions,
        checkpoint_last=completed.checkpoint_last,
        checkpoint_best=completed.checkpoint_best,
        reused=False,
    )


__all__ = [
    "FORMAL_SEEDS",
    "AnalyticFoldResult",
    "CompletedRun",
    "ConditionalInventoryResult",
    "ConditionalTrainingConfig",
    "FinalFitResult",
    "FoldFitResult",
    "NonfiniteTrainingError",
    "OOFResult",
    "build_run_lineage",
    "cross_fit_analytic",
    "cross_fit_conditional",
    "ensemble_then_project",
    "final_epoch_from_best_epochs",
    "fit_conditional_fold",
    "fit_conditional_inventory",
    "fit_conditional_seed",
    "fit_final_conditional",
    "load_completed_run",
    "predict_conditional_checkpoint",
    "save_completed_run",
    "validate_resume_lineage",
]
