"""Validation-only V26/V28 affinity selection and fold diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np

from dta.rank_fusion_v4.metrics import affinity_metrics

from .data import DATASET_INDEX


BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


def load_aligned_predictions(path, sample_ids, labels, dataset_index):
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "labels", "dataset_index", "predictions"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("prediction archive is missing: %s" % sorted(missing))
        source_ids = archive["sample_ids"].astype(str)
        if len(set(source_ids.tolist())) != source_ids.size:
            raise ValueError("prediction archive sample IDs are not unique")
        positions = {value: index for index, value in enumerate(source_ids.tolist())}
        try:
            order = np.asarray([positions[str(value)] for value in sample_ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError("prediction archive does not cover all requested samples") from error
        source_labels = np.asarray(archive["labels"], dtype=np.float64)[order]
        source_dataset = np.asarray(archive["dataset_index"], dtype=np.int64)[order]
        predictions = np.asarray(archive["predictions"], dtype=np.float64)[order]
    # V26 router archives retain manifest float64 labels, whereas V28 labels
    # pass through the float32 training collator before prediction export.
    # Accept only the expected float32 round-trip noise; IDs and dataset
    # indices remain exact lineage checks below.
    if not np.allclose(source_labels, labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("prediction archive labels differ")
    if not np.array_equal(source_dataset, dataset_index):
        raise ValueError("prediction archive dataset indices differ")
    if not np.isfinite(predictions).all():
        raise ValueError("prediction archive contains non-finite values")
    return predictions


def hash_fold_ids(sample_ids, count=5):
    folds = np.asarray(
        [int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16) % int(count) for value in sample_ids],
        dtype=np.int64,
    )
    if set(folds.tolist()) != set(range(int(count))):
        raise ValueError("one or more validation folds are empty")
    return folds


def _metrics(labels, predictions):
    return {key: float(value) for key, value in affinity_metrics(labels, predictions).items()}


def select_validation_blends(
    sample_ids,
    labels,
    dataset_index,
    anchor_predictions,
    v28_predictions,
    minimum_ci_gain=0.0005,
    maximum_rmse_increase=0.003,
    minimum_spearman_gain=0.0,
    minimum_positive_folds=3,
):
    sample_ids = np.asarray(sample_ids).astype(str)
    labels = np.asarray(labels, dtype=np.float64)
    dataset_index = np.asarray(dataset_index, dtype=np.int64)
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.float64)
    v28_predictions = np.asarray(v28_predictions, dtype=np.float64)
    selections = {}
    final = anchor_predictions.copy()
    any_promoted = False
    for name, index in DATASET_INDEX.items():
        selected = dataset_index == index
        ids = sample_ids[selected]
        target = labels[selected]
        anchor = anchor_predictions[selected]
        candidate = v28_predictions[selected]
        anchor_metrics = _metrics(target, anchor)
        expressions = []
        for weight in BLEND_WEIGHTS:
            prediction = (1.0 - weight) * anchor + weight * candidate
            metrics = _metrics(target, prediction)
            expressions.append((metrics["CI"], -metrics["RMSE"], metrics["Spearman"], -metrics["MAE"], -weight, weight, metrics, prediction))
        chosen = max(expressions)
        weight, metrics, prediction = float(chosen[5]), chosen[6], chosen[7]
        folds = hash_fold_ids(ids)
        fold_gains = []
        for fold in range(5):
            fold_mask = folds == fold
            fold_gains.append(
                float(_metrics(target[fold_mask], prediction[fold_mask])["CI"] - _metrics(target[fold_mask], anchor[fold_mask])["CI"])
            )
        ci_gain = float(metrics["CI"] - anchor_metrics["CI"])
        rmse_increase = float(metrics["RMSE"] - anchor_metrics["RMSE"])
        spearman_gain = float(metrics["Spearman"] - anchor_metrics["Spearman"])
        positive_folds = int(sum(value > 0 for value in fold_gains))
        passed = (
            weight > 0.0
            and ci_gain >= float(minimum_ci_gain)
            and rmse_increase <= float(maximum_rmse_increase)
            and spearman_gain >= float(minimum_spearman_gain)
            and positive_folds >= int(minimum_positive_folds)
        )
        effective_weight = weight if passed else 0.0
        if passed:
            final[selected] = prediction
            any_promoted = True
        selections[name] = {
            "searched_v28_weight": weight,
            "effective_v28_weight": effective_weight,
            "passed": bool(passed),
            "anchor_metrics": anchor_metrics,
            "searched_metrics": metrics,
            "effective_metrics": metrics if passed else anchor_metrics,
            "ci_gain": ci_gain,
            "rmse_increase": rmse_increase,
            "spearman_gain": spearman_gain,
            "positive_folds": positive_folds,
            "fold_ci_gains": fold_gains,
        }
    return {
        "any_promoted": bool(any_promoted),
        "criteria": {
            "minimum_ci_gain": float(minimum_ci_gain),
            "maximum_rmse_increase": float(maximum_rmse_increase),
            "minimum_spearman_gain": float(minimum_spearman_gain),
            "minimum_positive_folds": int(minimum_positive_folds),
            "blend_weights": list(BLEND_WEIGHTS),
        },
        "datasets": selections,
        "predictions": final,
    }


def apply_sealed_blends(dataset_index, anchor_predictions, v28_predictions, selection):
    result = np.asarray(anchor_predictions, dtype=np.float64).copy()
    dataset_index = np.asarray(dataset_index, dtype=np.int64)
    for name, index in DATASET_INDEX.items():
        weight = float(selection["datasets"][name]["effective_v28_weight"])
        selected = dataset_index == index
        result[selected] = (
            (1.0 - weight) * np.asarray(anchor_predictions, dtype=np.float64)[selected]
            + weight * np.asarray(v28_predictions, dtype=np.float64)[selected]
        )
    return result


__all__ = [
    "BLEND_WEIGHTS",
    "apply_sealed_blends",
    "hash_fold_ids",
    "load_aligned_predictions",
    "select_validation_blends",
]
