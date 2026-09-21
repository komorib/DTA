"""OOF-only ensemble and calibration selection for fusion v7."""

import math

import numpy as np

from dta.rank_fusion_v4.metrics import affinity_metrics


SEEDS = (1, 7, 21)
_WEIGHT_CANDIDATES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)
_RESIDUAL_SCALES = (0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0)


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


def _seed_vectors(seed_predictions, count):
    if not isinstance(seed_predictions, dict) or set(seed_predictions) != set(SEEDS):
        raise ValueError("seed predictions must contain seeds 1, 7, and 21")
    return {
        seed: _vector(seed_predictions[seed], "seed prediction", count=count)
        for seed in SEEDS
    }


def verify_oof_coverage(coverage, row_count):
    if not isinstance(coverage, dict) or set(coverage) != set(SEEDS):
        raise ValueError("OOF coverage must contain seeds 1, 7, and 21")
    for seed in SEEDS:
        values = np.asarray(coverage[seed])
        if values.shape != (int(row_count),) or not np.all(values == 1):
            raise ValueError("every row must receive exactly once OOF coverage")


def _weighted_prediction(seed_predictions, weights):
    return sum(
        float(weight) * seed_predictions[seed]
        for seed, weight in zip(SEEDS, weights)
    )


def _positive_affine(prediction, labels):
    centered_prediction = prediction - prediction.mean()
    denominator = float(np.dot(centered_prediction, centered_prediction))
    if denominator <= 1e-12:
        slope = 1.0
    else:
        slope = float(
            np.dot(centered_prediction, labels - labels.mean()) / denominator
        )
        slope = max(1e-6, slope)
    intercept = float(labels.mean() - slope * prediction.mean())
    return slope, intercept


def _score(metrics):
    return (
        float(metrics["CI"]),
        -float(metrics["RMSE"]),
        -float(metrics["MAE"]),
        float(metrics["Pearson"]),
    )


def build_oof_selection(anchor, seed_predictions, labels):
    anchor = _vector(anchor, "anchor")
    labels = _vector(labels, "labels", count=anchor.shape[0])
    seeds = _seed_vectors(seed_predictions, anchor.shape[0])
    anchor_metrics = affinity_metrics(labels, anchor)
    best = None
    for weights in _WEIGHT_CANDIDATES:
        fused = _weighted_prediction(seeds, weights)
        for residual_scale in _RESIDUAL_SCALES:
            uncalibrated = anchor + residual_scale * (fused - anchor)
            slope, intercept = _positive_affine(uncalibrated, labels)
            calibrated = slope * uncalibrated + intercept
            metrics = affinity_metrics(labels, calibrated)
            candidate = {
                "seed_weights": {
                    str(seed): float(weight)
                    for seed, weight in zip(SEEDS, weights)
                },
                "residual_scale": float(residual_scale),
                "affine_slope": float(slope),
                "affine_intercept": float(intercept),
                "oof_metrics": dict(metrics),
                "anchor_metrics": dict(anchor_metrics),
            }
            candidate_score = _score(metrics)
            if best is None or candidate_score > best[0]:
                best = (candidate_score, candidate)
    if best is None:
        raise ValueError("OOF selection did not produce a candidate")
    result = best[1]
    if not all(
        math.isfinite(float(result[field]))
        for field in ("residual_scale", "affine_slope", "affine_intercept")
    ):
        raise ValueError("OOF selection is nonfinite")
    result["status"] = (
        "passed" if result["oof_metrics"]["CI"] > anchor_metrics["CI"] else "failed"
    )
    return result


def apply_selection(anchor, seed_predictions, selection):
    anchor = _vector(anchor, "anchor")
    seeds = _seed_vectors(seed_predictions, anchor.shape[0])
    if not isinstance(selection, dict):
        raise ValueError("selection must be a dictionary")
    raw_weights = selection.get("seed_weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != {"1", "7", "21"}:
        raise ValueError("selection seed weights do not match")
    weights = tuple(float(raw_weights[str(seed)]) for seed in SEEDS)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("selection seed weights are invalid")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("selection seed weights must sum to one")
    scale = float(selection["residual_scale"])
    slope = float(selection["affine_slope"])
    intercept = float(selection["affine_intercept"])
    if not all(math.isfinite(value) for value in (scale, slope, intercept)):
        raise ValueError("selection transform must be finite")
    fused = _weighted_prediction(seeds, weights)
    uncalibrated = anchor + scale * (fused - anchor)
    result = slope * uncalibrated + intercept
    if not np.isfinite(result).all():
        raise ValueError("selected prediction is nonfinite")
    return result


__all__ = ["SEEDS", "apply_selection", "build_oof_selection", "verify_oof_coverage"]
