"""Deterministic validation-only static fusion for anchored expert fusion v5."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from numbers import Integral, Real

import numpy as np

from dta.anchored_expert_fusion_v5.data import EXPERT_ORDER, fallback_predictions
from dta.rank_fusion_v4.metrics import affinity_metrics


_METRIC_NAMES = ("CI", "MAE", "RMSE", "Pearson", "Spearman")
_GATE_GAIN = 0.005
_GATE_WEIGHT_RANGE = 0.20
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SearchResult:
    weights: np.ndarray
    metrics: dict
    slope: float
    intercept: float
    stage_counts: tuple
    searched_count: int


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    held_out_indices: np.ndarray
    selected_weights: np.ndarray
    candidate_metrics: dict
    upper_bound_metrics: dict
    search_metrics: dict
    slope: float
    intercept: float
    stage_counts: tuple
    searched_count: int

    @property
    def weights(self):
        return self.selected_weights


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple


@dataclass(frozen=True)
class CrossFitReport:
    family: str
    validation_keys: tuple
    fold_ids: np.ndarray
    fold_map_hash: str
    held_out_count: np.ndarray
    predictions: np.ndarray
    pooled_metrics: dict
    upper_bound_pooled_metrics: dict
    folds: tuple
    fold_weights: np.ndarray
    deploy_weights: np.ndarray
    gate: GateResult
    validation_available: np.ndarray
    report_hash: str = ""

    @property
    def gate_passed(self):
        return self.gate.passed

    @property
    def gate_reasons(self):
        return self.gate.reasons

    @property
    def coverage_complete(self):
        return bool(np.all(self.held_out_count == 1))

    @property
    def selected_weights(self):
        return self.fold_weights

    @property
    def upper_bound_metrics(self):
        return self.upper_bound_pooled_metrics


def _normalized_keys(keys):
    if isinstance(keys, (str, bytes)):
        raise ValueError("keys must be a nonempty sequence")
    try:
        normalized = tuple(str(key) for key in keys)
    except TypeError as error:
        raise ValueError("keys must be a nonempty sequence") from error
    if not normalized or any(not key for key in normalized):
        raise ValueError("keys must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("keys must be unique")
    return normalized


def assign_folds(keys, seed=42, fold_count=5):
    """Assign each key independently with the approved SHA256 primitive."""
    normalized = _normalized_keys(keys)
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, Integral)
        or fold_count < 2
    ):
        raise ValueError("fold_count must be an integer of at least two")
    fold_count = int(fold_count)
    if len(normalized) < fold_count:
        raise ValueError("rows must be at least fold_count")
    values = []
    for key in normalized:
        payload = (str(seed) + ":" + key).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        values.append(int(digest[:16], 16) % fold_count)
    folds = np.asarray(values, dtype=np.int64)
    if set(folds.tolist()) != set(range(fold_count)):
        raise ValueError("deterministic fold assignment left an empty fold")
    return folds


def fold_map_sha256(keys, fold_ids):
    """Hash ordered keys and fold IDs without accepting or including labels."""
    normalized = _normalized_keys(keys)
    folds = np.asarray(fold_ids)
    if folds.ndim != 1 or folds.size != len(normalized):
        raise ValueError("fold_ids must match ordered keys")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must be integers")
    payload = json.dumps(
        [[key, int(fold)] for key, fold in zip(normalized, folds)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rank_percentiles(values):
    """Return stable average-tie percentile ranks in float64."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty 1D vector")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    if array.size == 1:
        ranks[0] = 0.0
        return ranks
    return ranks / float(array.size - 1)


def project_simplex(weights):
    """Euclidean projection onto the nonnegative sum-one simplex."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a nonempty 1D vector")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    valid = np.nonzero(ordered - cumulative / indices > 0.0)[0]
    rho = int(valid[-1])
    theta = cumulative[rho] / float(rho + 1)
    result = np.maximum(values - theta, 0.0)
    result /= float(result.sum())
    return result.astype(np.float64, copy=False)


def _labels(values, name):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(name + " must be a nonempty 1D vector")
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _predictions(values, name, row_count=None):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    if array.ndim != 2 or array.shape[1:] != (len(EXPERT_ORDER),):
        raise ValueError(name + " must have shape [rows, 3]")
    if row_count is not None and array.shape[0] != row_count:
        raise ValueError(name + " and labels must have matching rows")
    if array.shape[0] == 0:
        raise ValueError(name + " must be nonempty")
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _rank_columns(predictions):
    return np.column_stack(
        [rank_percentiles(predictions[:, index]) for index in range(3)]
    ).astype(np.float64, copy=False)


def _fit_affine(labels, blend):
    centered = blend - float(blend.mean())
    denominator = float(np.dot(centered, centered))
    scale = max(1.0, float(np.dot(blend, blend)))
    if denominator <= np.finfo(np.float64).eps * scale:
        return 0.0, float(labels.mean())
    label_centered = labels - float(labels.mean())
    slope = float(np.dot(centered, label_centered) / denominator)
    intercept = float(labels.mean() - slope * blend.mean())
    return slope, intercept


def _candidate_key(metrics, weights):
    return (
        -float(metrics["CI"]),
        -float(metrics["Spearman"]),
        float(metrics["RMSE"]),
        tuple(float(value) for value in weights),
    )


def _grid_points(step_units, center=None, radius_units=None):
    if center is None:
        for first in range(0, 1001, step_units):
            for second in range(0, 1001 - first, step_units):
                third = 1000 - first - second
                if third % step_units == 0:
                    yield (first, second, third)
        return
    lower_first = max(0, center[0] - radius_units)
    upper_first = min(1000, center[0] + radius_units)
    lower_first += (-lower_first) % step_units
    for first in range(lower_first, upper_first + 1, step_units):
        lower_second = max(0, center[1] - radius_units)
        upper_second = min(1000 - first, center[1] + radius_units)
        lower_second += (-lower_second) % step_units
        for second in range(lower_second, upper_second + 1, step_units):
            third = 1000 - first - second
            if third < 0 or third % step_units:
                continue
            point = (first, second, third)
            if max(abs(point[i] - center[i]) for i in range(3)) <= radius_units:
                yield point


def search_simplex_weights(
    labels,
    predictions,
    family="raw",
    train_labels=None,
    train_predictions=None,
    features_prepared=False,
):
    """Search the complete coarse grid and two bounded refinement grids."""
    if family not in ("raw", "rank"):
        raise ValueError("family must be raw or rank")
    valid_labels = _labels(labels, "labels")
    valid_features = _predictions(
        predictions, "predictions", row_count=valid_labels.size
    )
    calibration_labels = None
    calibration_features = None
    if family == "rank":
        if train_labels is None or train_predictions is None:
            raise ValueError(
                "rank family requires train_labels and train_predictions"
            )
        calibration_labels = _labels(train_labels, "train_labels")
        calibration_features = _predictions(
            train_predictions,
            "train_predictions",
            row_count=calibration_labels.size,
        )
        if not features_prepared:
            valid_features = _rank_columns(valid_features)
            calibration_features = _rank_columns(calibration_features)
    elif features_prepared:
        raise ValueError("features_prepared is only valid for rank family")

    best = None
    best_units = None
    seen = set()
    stage_counts = []
    stages = ((25, None), (5, 50), (1, 10))
    for step_units, radius_units in stages:
        center = best_units if radius_units is not None else None
        count = 0
        for units in _grid_points(step_units, center, radius_units):
            if units in seen:
                continue
            seen.add(units)
            count += 1
            weights = np.asarray(units, dtype=np.float64) / 1000.0
            blend = valid_features @ weights
            if family == "rank":
                train_blend = calibration_features @ weights
                slope, intercept = _fit_affine(
                    calibration_labels, train_blend
                )
                candidate = slope * blend + intercept
            else:
                slope, intercept = 1.0, 0.0
                candidate = blend
            metrics = affinity_metrics(valid_labels, candidate)
            key = _candidate_key(metrics, weights)
            if best is None or key < best[0]:
                best = (key, weights, metrics, slope, intercept)
                best_units = units
        stage_counts.append(count)

    return SearchResult(
        weights=best[1].copy(),
        metrics={name: float(best[2][name]) for name in _METRIC_NAMES},
        slope=float(best[3]),
        intercept=float(best[4]),
        stage_counts=tuple(int(value) for value in stage_counts),
        searched_count=int(sum(stage_counts)),
    )


def deploy_weights(fold_weights):
    values = np.asarray(fold_weights, dtype=np.float64)
    if values.shape != (5, 3):
        raise ValueError("fold_weights must have shape [5, 3]")
    if not np.isfinite(values).all():
        raise ValueError("fold_weights must be finite")
    return project_simplex(np.median(values, axis=0))


def _gate_ci(metrics):
    try:
        value = metrics["CI"]
    except (KeyError, TypeError, AttributeError):
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
    ):
        return None
    return float(value)


def evaluate_gate(
    pooled_metrics,
    upper_bound_pooled_metrics,
    fold_metrics,
    upper_bound_fold_metrics,
    fold_weights,
    held_out_count,
):
    """Evaluate all four validation-only anti-overfit conditions."""
    reasons = []
    pooled_ci = _gate_ci(pooled_metrics)
    upper_pooled_ci = _gate_ci(upper_bound_pooled_metrics)
    invalid_metrics = pooled_ci is None or upper_pooled_ci is None

    improved = 0
    if len(fold_metrics) == 5 and len(upper_bound_fold_metrics) == 5:
        for candidate, upper in zip(
            fold_metrics, upper_bound_fold_metrics
        ):
            candidate_ci = _gate_ci(candidate)
            upper_ci = _gate_ci(upper)
            if candidate_ci is None or upper_ci is None:
                invalid_metrics = True
                continue
            improved += candidate_ci > upper_ci

    if invalid_metrics:
        reasons.append("nonfinite_gate_metrics")
    if (
        pooled_ci is not None
        and upper_pooled_ci is not None
        and pooled_ci - upper_pooled_ci < _GATE_GAIN
    ):
        reasons.append("pooled_ci_gain_below_0.005")
    if improved < 4:
        reasons.append("fewer_than_4_of_5_folds_improved")

    weights = np.asarray(fold_weights, dtype=np.float64)
    unstable = (
        weights.shape != (5, 3)
        or not np.isfinite(weights).all()
        or np.any(np.ptp(weights, axis=0) > _GATE_WEIGHT_RANGE)
    )
    if unstable:
        reasons.append("fold_weight_range_exceeds_0.20")

    counts = np.asarray(held_out_count)
    if counts.ndim != 1 or counts.size == 0 or not np.all(counts == 1):
        reasons.append("held_out_coverage_not_exactly_once")
    return GateResult(passed=not reasons, reasons=tuple(reasons))


def _validate_search_result(result):
    if not isinstance(result, SearchResult):
        raise ValueError("search_fn must return SearchResult")

    metrics = result.metrics
    if not hasattr(metrics, "keys") or any(
        name not in metrics for name in _METRIC_NAMES
    ):
        raise ValueError("search_fn returned invalid metrics")
    for name in _METRIC_NAMES:
        value = metrics[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
        ):
            raise ValueError("search_fn returned invalid metrics")

    weights = np.asarray(result.weights, dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or abs(float(weights.sum()) - 1.0) > _FLOAT_TOLERANCE
    ):
        raise ValueError("search_fn returned invalid simplex weights")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        for value in (result.slope, result.intercept)
    ):
        raise ValueError("search_fn returned invalid calibration")

    stage_counts = result.stage_counts
    if (
        not isinstance(stage_counts, Sequence)
        or isinstance(stage_counts, (str, bytes))
        or len(stage_counts) == 0
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 0
            for value in stage_counts
        )
    ):
        raise ValueError("search_fn returned invalid stage_counts")
    searched_count = result.searched_count
    if (
        isinstance(searched_count, (bool, np.bool_))
        or not isinstance(searched_count, Integral)
        or searched_count < 0
        or int(searched_count) != sum(int(value) for value in stage_counts)
    ):
        raise ValueError("search_fn returned invalid searched_count")
    return weights


def cross_fit_static(
    train_labels,
    train_predictions,
    train_available,
    validation_keys,
    validation_labels,
    validation_predictions,
    validation_available,
    family,
    seed=42,
    fold_count=5,
    search_fn=search_simplex_weights,
):
    """Cross-fit a raw or rank static blend over validation folds."""
    if family not in ("raw", "rank"):
        raise ValueError("family must be raw or rank")
    if fold_count != 5:
        raise ValueError("cross_fit_static requires fold_count=5")
    keys = _normalized_keys(validation_keys)
    train_y = _labels(train_labels, "train_labels")
    valid_y = _labels(validation_labels, "validation_labels")
    if valid_y.size != len(keys):
        raise ValueError("validation labels and keys must have matching rows")
    train_raw = _predictions(
        train_predictions, "train_predictions", row_count=train_y.size
    )
    valid_raw = _predictions(
        validation_predictions,
        "validation_predictions",
        row_count=valid_y.size,
    )
    train_availability = np.asarray(train_available)
    valid_availability = np.asarray(validation_available)
    if (
        train_availability.shape != train_raw.shape
        or train_availability.dtype != np.bool_
    ):
        raise ValueError(
            "train_available must be boolean with matching [rows, 3] shape"
        )
    if (
        valid_availability.shape != valid_raw.shape
        or valid_availability.dtype != np.bool_
    ):
        raise ValueError(
            "validation_available must be boolean with matching [rows, 3] shape"
        )

    train_fallback = fallback_predictions(
        train_raw, train_availability
    ).astype(np.float64)
    valid_fallback = fallback_predictions(
        valid_raw, valid_availability
    ).astype(np.float64)
    if family == "rank":
        train_features = _rank_columns(train_fallback)
        valid_features = _rank_columns(valid_fallback)
    else:
        train_features = train_fallback
        valid_features = valid_fallback

    fold_ids = assign_folds(keys, seed=seed, fold_count=fold_count)
    predictions = np.empty(valid_y.size, dtype=np.float64)
    held_out_count = np.zeros(valid_y.size, dtype=np.int64)
    fold_results = []
    selected = []
    for fold in range(fold_count):
        held_out = fold_ids == fold
        search_rows = ~held_out
        kwargs = {
            "labels": valid_y[search_rows],
            "predictions": valid_features[search_rows],
            "family": family,
        }
        if family == "rank":
            kwargs.update(
                train_labels=train_y,
                train_predictions=train_features,
                features_prepared=True,
            )
        result = search_fn(**kwargs)
        weights = _validate_search_result(result)
        held_blend = valid_features[held_out] @ weights
        if family == "rank":
            held_predictions = (
                float(result.slope) * held_blend + float(result.intercept)
            )
        else:
            held_predictions = held_blend
        predictions[held_out] = held_predictions
        held_out_count[held_out] += 1
        upper_predictions = valid_fallback[held_out, 0]
        candidate_metrics = affinity_metrics(
            valid_y[held_out], held_predictions
        )
        upper_metrics = affinity_metrics(
            valid_y[held_out], upper_predictions
        )
        indices = np.flatnonzero(held_out).astype(np.int64)
        fold_results.append(
            FoldResult(
                fold_id=fold,
                held_out_indices=indices,
                selected_weights=weights.copy(),
                candidate_metrics={
                    name: float(candidate_metrics[name])
                    for name in _METRIC_NAMES
                },
                upper_bound_metrics={
                    name: float(upper_metrics[name])
                    for name in _METRIC_NAMES
                },
                search_metrics={
                    name: float(result.metrics[name])
                    for name in _METRIC_NAMES
                },
                slope=float(result.slope),
                intercept=float(result.intercept),
                stage_counts=tuple(
                    int(value) for value in result.stage_counts
                ),
                searched_count=int(result.searched_count),
            )
        )
        selected.append(weights.copy())

    if not np.all(held_out_count == 1):
        raise ValueError("each validation row must be held out exactly once")
    pooled = affinity_metrics(valid_y, predictions)
    upper_pooled = affinity_metrics(valid_y, valid_fallback[:, 0])
    fold_weights = np.stack(selected).astype(np.float64)
    fold_metrics = tuple(fold.candidate_metrics for fold in fold_results)
    upper_fold_metrics = tuple(
        fold.upper_bound_metrics for fold in fold_results
    )
    gate = evaluate_gate(
        pooled,
        upper_pooled,
        fold_metrics,
        upper_fold_metrics,
        fold_weights,
        held_out_count,
    )
    report = CrossFitReport(
        family=family,
        validation_keys=tuple(keys),
        fold_ids=fold_ids.copy(),
        fold_map_hash=fold_map_sha256(keys, fold_ids),
        held_out_count=held_out_count.copy(),
        predictions=predictions.copy(),
        pooled_metrics={
            name: float(pooled[name]) for name in _METRIC_NAMES
        },
        upper_bound_pooled_metrics={
            name: float(upper_pooled[name]) for name in _METRIC_NAMES
        },
        folds=tuple(fold_results),
        fold_weights=fold_weights,
        deploy_weights=deploy_weights(fold_weights),
        gate=gate,
        validation_available=valid_availability.copy(),
    )
    report = replace(report, report_hash=report_sha256(report))
    assert report.report_hash == report_sha256(report)
    return report


def report_sha256(report):
    """Return a stable hash of cross-fit outputs and provenance."""
    if not isinstance(report, CrossFitReport):
        raise ValueError("report must be a CrossFitReport")
    payload = {
        "family": report.family,
        "validation_keys": list(report.validation_keys),
        "fold_ids": report.fold_ids.astype(int).tolist(),
        "fold_map_hash": report.fold_map_hash,
        "held_out_count": report.held_out_count.astype(int).tolist(),
        "predictions": report.predictions.astype(float).tolist(),
        "pooled_metrics": report.pooled_metrics,
        "upper_bound_pooled_metrics": report.upper_bound_pooled_metrics,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "held_out_indices": fold.held_out_indices.astype(int).tolist(),
                "selected_weights": fold.selected_weights.astype(float).tolist(),
                "candidate_metrics": fold.candidate_metrics,
                "upper_bound_metrics": fold.upper_bound_metrics,
                "search_metrics": fold.search_metrics,
                "slope": fold.slope,
                "intercept": fold.intercept,
                "stage_counts": list(fold.stage_counts),
                "searched_count": fold.searched_count,
            }
            for fold in report.folds
        ],
        "fold_weights": report.fold_weights.astype(float).tolist(),
        "deploy_weights": report.deploy_weights.astype(float).tolist(),
        "gate": {
            "passed": report.gate.passed,
            "reasons": list(report.gate.reasons),
        },
        "validation_available": report.validation_available.astype(bool).tolist(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
