"""Additional grouped-fold repeats built on the frozen v7 architecture."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from dta.rank_unlocked_fusion_v7.training import (
    fit_fixed_epochs,
    predict_checkpoint,
    select_epoch,
)


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


def _inner_indices(all_indices, fold_ids, outer_fold, folds):
    outer_train = all_indices[fold_ids != outer_fold]
    if len(folds) == 2:
        stop = outer_train[::2]
        fit = outer_train[1::2]
    else:
        stop_fold = folds[(outer_fold + 1) % len(folds)]
        stop = all_indices[fold_ids == stop_fold]
        fit = all_indices[(fold_ids != outer_fold) & (fold_ids != stop_fold)]
    if fit.size == 0 or stop.size == 0:
        raise ValueError("repeat inner split is empty")
    return outer_train, fit, stop


def fit_repeat(
    dataset,
    fold_ids,
    *,
    repeat_name,
    fold_seed,
    model_seed,
    data_lineage_sha256=None,
    output,
    config,
    collate_fn,
    resume=False,
):
    if not isinstance(repeat_name, str) or not repeat_name:
        raise ValueError("repeat name must be nonempty")
    if data_lineage_sha256 is not None and (
        not isinstance(data_lineage_sha256, str)
        or len(data_lineage_sha256) != 64
        or any(character not in "0123456789abcdef" for character in data_lineage_sha256)
    ):
        raise ValueError("repeat data lineage SHA256 is invalid")
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if fold_ids.shape != (len(dataset),):
        raise ValueError("repeat fold IDs do not match the dataset")
    folds = tuple(sorted(int(value) for value in np.unique(fold_ids)))
    if folds != tuple(range(len(folds))) or len(folds) < 2:
        raise ValueError("repeat fold IDs must be contiguous")
    all_indices = np.arange(len(dataset), dtype=np.int64)
    predictions = np.full(len(dataset), np.nan, dtype=np.float64)
    coverage = np.zeros(len(dataset), dtype=np.int64)
    checkpoints = []
    histories = []
    output = Path(output)
    expected_training_config = asdict(config)
    for fold in folds:
        holdout = all_indices[fold_ids == fold]
        outer_train, fit_indices, stop_indices = _inner_indices(
            all_indices, fold_ids, fold, folds
        )
        path = output / repeat_name / ("fold_%d.pt" % fold)
        print({
            "event": "v8_repeat_fold_start",
            "repeat": repeat_name,
            "fold_seed": int(fold_seed),
            "model_seed": int(model_seed),
            "fold": fold,
            "train_count": int(outer_train.size),
            "holdout_count": int(holdout.size),
        }, flush=True)
        if path.exists():
            if not resume:
                raise ValueError("V8 repeat checkpoint already exists: " + str(path))
            checkpoint = torch.load(str(path), map_location="cpu")
            expected = (repeat_name, int(fold_seed), int(model_seed), fold)
            actual = (
                checkpoint.get("repeat_name"),
                checkpoint.get("fold_seed"),
                checkpoint.get("base_model_seed"),
                checkpoint.get("outer_fold"),
            )
            if actual != expected:
                raise ValueError("V8 repeat resume lineage does not match")
            if checkpoint.get("training_config") != expected_training_config:
                raise ValueError("V8 repeat resume configuration does not match")
            if checkpoint.get("data_lineage_sha256") != data_lineage_sha256:
                raise ValueError("V8 repeat resume data lineage does not match")
            selection_history = checkpoint.get("selection_history", [])
        else:
            training_seed = int(model_seed) + 1009 * fold
            best_epoch, selection_history = select_epoch(
                dataset,
                fit_indices,
                stop_indices,
                training_seed,
                config,
                collate_fn,
            )
            checkpoint = fit_fixed_epochs(
                dataset,
                outer_train,
                training_seed,
                best_epoch,
                config,
                collate_fn,
            )
            checkpoint.update({
                "repeat_name": repeat_name,
                "fold_seed": int(fold_seed),
                "base_model_seed": int(model_seed),
                "outer_fold": fold,
                "data_lineage_sha256": data_lineage_sha256,
                "selected_epoch": int(best_epoch),
                "selection_history": selection_history,
            })
            _atomic_torch_save(checkpoint, path)
        prediction = predict_checkpoint(
            checkpoint, dataset, holdout, config, collate_fn
        )
        predictions[holdout] = prediction
        coverage[holdout] += 1
        checkpoints.append(str(path.resolve()))
        histories.append({
            "repeat": repeat_name,
            "fold": fold,
            "selected_epoch": int(checkpoint["selected_epoch"]),
            "selection_history": selection_history,
        })
        print({
            "event": "v8_repeat_fold_complete",
            "repeat": repeat_name,
            "fold": fold,
            "selected_epoch": int(checkpoint["selected_epoch"]),
            "checkpoint": str(path.resolve()),
        }, flush=True)
    if not np.isfinite(predictions).all() or not np.all(coverage == 1):
        raise RuntimeError("V8 repeat OOF coverage is incomplete")
    return {
        "repeat_name": repeat_name,
        "fold_seed": int(fold_seed),
        "model_seed": int(model_seed),
        "data_lineage_sha256": data_lineage_sha256,
        "predictions": predictions,
        "coverage": coverage,
        "checkpoints": checkpoints,
        "histories": histories,
    }


def predict_repeat_ensemble(checkpoints, dataset, config, collate_fn):
    try:
        paths = tuple(checkpoints)
    except TypeError as error:
        raise ValueError("repeat checkpoints must be iterable") from error
    if not paths:
        raise ValueError("repeat checkpoints must be nonempty")
    indices = np.arange(len(dataset), dtype=np.int64)
    members = []
    for path in paths:
        checkpoint = torch.load(str(path), map_location="cpu")
        members.append(
            predict_checkpoint(checkpoint, dataset, indices, config, collate_fn)
        )
    return np.mean(np.stack(members, axis=0), axis=0)


__all__ = ["fit_repeat", "predict_repeat_ensemble"]
