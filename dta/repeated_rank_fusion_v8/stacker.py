"""Positive pairwise rank stacker trained only on OOF predictions."""

import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize


def _matrix(value, name, row_count=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a finite matrix") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(name + " must be a nonempty matrix")
    if row_count is not None and result.shape[0] != row_count:
        raise ValueError(name + " row count does not match")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _vector(value, name, count=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a finite vector") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(name + " must be a nonempty vector")
    if count is not None and result.shape != (count,):
        raise ValueError(name + " shape does not match")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _positive_float(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be finite")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        raise ValueError(name + " must be finite")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _pair_differences(features, labels, maximum_pairs, seed):
    row_count = labels.shape[0]
    total_pairs = row_count * (row_count - 1) // 2
    if total_pairs <= maximum_pairs:
        left, right = np.triu_indices(row_count, k=1)
    else:
        rng = np.random.RandomState(int(seed))
        sample_count = 2 * maximum_pairs
        left = rng.randint(0, row_count, size=sample_count)
        right = rng.randint(0, row_count, size=sample_count)
        distinct = left != right
        left = left[distinct][:maximum_pairs]
        right = right[distinct][:maximum_pairs]
    difference = labels[left] - labels[right]
    eligible = difference != 0.0
    left = left[eligible]
    right = right[eligible]
    difference = difference[eligible]
    high = np.where(difference > 0.0, left, right)
    low = np.where(difference > 0.0, right, left)
    if high.size == 0:
        raise ValueError("rank stacker requires non-tied label pairs")
    return features[high] - features[low]


def fit_positive_rank_stacker(
    features,
    labels,
    *,
    regularization=0.10,
    maximum_pairs=200000,
    seed=42,
):
    features = _matrix(features, "features")
    labels = _vector(labels, "labels", count=features.shape[0])
    regularization = _positive_float(
        regularization, "regularization", allow_zero=True
    )
    maximum_pairs = _positive_integer(maximum_pairs, "maximum pairs")
    means = features.mean(axis=0)
    scales = np.maximum(features.std(axis=0), 1e-6)
    standardized = (features - means) / scales
    label_mean = float(labels.mean())
    label_scale = max(float(labels.std()), 1e-6)
    standardized_labels = (labels - label_mean) / label_scale
    pair_difference = _pair_differences(
        standardized, labels, maximum_pairs, seed
    )

    def objective(weights):
        pair_margin = pair_difference.dot(weights)
        pair_loss = np.logaddexp(0.0, -pair_margin).mean()
        pair_gradient = pair_difference.T.dot(
            -1.0 / (1.0 + np.exp(np.clip(pair_margin, -40.0, 40.0)))
        ) / pair_difference.shape[0]
        residual = standardized.dot(weights) - standardized_labels
        mse = np.square(residual).mean()
        mse_gradient = 2.0 * standardized.T.dot(residual) / labels.shape[0]
        loss = 0.80 * pair_loss + 0.20 * mse + regularization * np.dot(weights, weights)
        gradient = (
            0.80 * pair_gradient
            + 0.20 * mse_gradient
            + 2.0 * regularization * weights
        )
        return float(loss), gradient

    initial = np.full(features.shape[1], 1.0 / features.shape[1], dtype=np.float64)
    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * features.shape[1],
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not optimized.success and optimized.status not in (1, 2):
        raise RuntimeError("rank stacker optimization failed: " + optimized.message)
    weights = np.maximum(np.asarray(optimized.x, dtype=np.float64), 0.0)
    raw = standardized.dot(weights)
    centered = raw - raw.mean()
    denominator = float(np.dot(centered, centered))
    slope = 1.0 if denominator <= 1e-12 else float(
        np.dot(centered, labels - labels.mean()) / denominator
    )
    slope = max(1e-6, slope)
    intercept = float(labels.mean() - slope * raw.mean())
    return {
        "schema_version": 1,
        "feature_count": int(features.shape[1]),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "weights": weights.tolist(),
        "affine_slope": slope,
        "affine_intercept": intercept,
        "regularization": regularization,
        "maximum_pairs": maximum_pairs,
        "seed": int(seed),
        "optimizer_status": int(optimized.status),
        "optimizer_iterations": int(optimized.nit),
        "optimizer_loss": float(optimized.fun),
    }


def predict_rank_stacker(features, model):
    features = _matrix(features, "features")
    if not isinstance(model, dict):
        raise ValueError("rank stacker model must be a dictionary")
    feature_count = int(model.get("feature_count", -1))
    if features.shape[1] != feature_count:
        raise ValueError("rank stacker feature count does not match")
    means = _vector(model.get("feature_means"), "feature means", count=feature_count)
    scales = _vector(model.get("feature_scales"), "feature scales", count=feature_count)
    weights = _vector(model.get("weights"), "weights", count=feature_count)
    if np.any(scales <= 0.0) or np.any(weights < 0.0):
        raise ValueError("rank stacker scales or weights are invalid")
    slope = _positive_float(model.get("affine_slope"), "affine slope")
    intercept = float(model.get("affine_intercept"))
    if not math.isfinite(intercept):
        raise ValueError("affine intercept must be finite")
    raw = ((features - means) / scales).dot(weights)
    predictions = slope * raw + intercept
    if not np.isfinite(predictions).all():
        raise ValueError("rank stacker predictions are nonfinite")
    return predictions


def cross_fit_rank_stacker(
    features,
    labels,
    fold_ids,
    *,
    regularization,
    maximum_pairs=200000,
    seed=42,
):
    features = _matrix(features, "features")
    labels = _vector(labels, "labels", count=features.shape[0])
    folds = np.asarray(fold_ids, dtype=np.int64)
    if folds.shape != labels.shape:
        raise ValueError("stacker fold IDs do not match")
    unique_folds = tuple(sorted(int(value) for value in np.unique(folds)))
    if unique_folds != tuple(range(len(unique_folds))) or len(unique_folds) < 2:
        raise ValueError("stacker fold IDs must be contiguous")
    predictions = np.full(labels.shape, np.nan, dtype=np.float64)
    coverage = np.zeros(labels.shape, dtype=np.int64)
    models = []
    for fold in unique_folds:
        train = folds != fold
        holdout = folds == fold
        model = fit_positive_rank_stacker(
            features[train],
            labels[train],
            regularization=regularization,
            maximum_pairs=maximum_pairs,
            seed=int(seed) + 1009 * fold,
        )
        predictions[holdout] = predict_rank_stacker(features[holdout], model)
        coverage[holdout] += 1
        models.append(model)
    if not np.isfinite(predictions).all() or not np.all(coverage == 1):
        raise RuntimeError("rank stacker OOF coverage is incomplete")
    return {
        "predictions": predictions,
        "coverage": coverage,
        "fold_models": models,
    }


__all__ = [
    "cross_fit_rank_stacker",
    "fit_positive_rank_stacker",
    "predict_rank_stacker",
]
