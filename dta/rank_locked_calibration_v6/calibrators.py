"""Train-only analytic calibrators for rank-locked calibration v6."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .projection import weighted_pava


_MIN_SLOPE = 1e-6
_MIN_INCREMENT = 1e-6


def _vector(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(name + " must be a nonempty vector")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _fit_inputs(score, label, weights):
    score = _vector(score, "score")
    label = _vector(label, "label")
    if score.shape != label.shape:
        raise ValueError("score and label must be aligned")
    if weights is None:
        weights = np.ones(score.shape[0], dtype=np.float64)
    else:
        weights = _vector(weights, "weights")
        if weights.shape != score.shape:
            raise ValueError("score and weights must be aligned")
        if np.any(weights <= 0.0):
            raise ValueError("weights must be positive")
    return score, label, weights


def _weighted_affine(score, label, weights):
    mass = float(weights.sum(dtype=np.float64))
    score_mean = float(np.dot(weights, score) / mass)
    label_mean = float(np.dot(weights, label) / mass)
    centered = score - score_mean
    variance = float(np.dot(weights, centered * centered))
    if variance == 0.0:
        slope = _MIN_SLOPE
    else:
        covariance = float(
            np.dot(weights, centered * (label - label_mean))
        )
        slope = max(_MIN_SLOPE, covariance / variance)
    intercept = float(np.dot(weights, label - slope * score) / mass)
    return slope, intercept


@dataclass(frozen=True)
class PositiveAffineCalibrator:
    slope: float
    intercept: float
    objective: str

    def __post_init__(self):
        if self.objective not in ("mse", "huber"):
            raise ValueError("affine objective must be mse or huber")
        if (
            isinstance(self.slope, bool)
            or not isinstance(self.slope, Real)
            or not math.isfinite(float(self.slope))
            or float(self.slope) < _MIN_SLOPE
        ):
            raise ValueError("affine slope must be positive and finite")
        if (
            isinstance(self.intercept, bool)
            or not isinstance(self.intercept, Real)
            or not math.isfinite(float(self.intercept))
        ):
            raise ValueError("affine intercept must be finite")
        object.__setattr__(self, "slope", float(self.slope))
        object.__setattr__(self, "intercept", float(self.intercept))

    @classmethod
    def fit(cls, score, label, objective="mse", weights=None):
        if objective not in ("mse", "huber"):
            raise ValueError("affine objective must be mse or huber")
        score, label, base_weights = _fit_inputs(score, label, weights)
        slope, intercept = _weighted_affine(score, label, base_weights)
        if objective == "huber":
            for _ in range(50):
                residual = label - (slope * score + intercept)
                absolute = np.abs(residual)
                robust = np.ones_like(absolute)
                outside = absolute > 1.0
                robust[outside] = 1.0 / absolute[outside]
                slope, intercept = _weighted_affine(
                    score, label, base_weights * robust
                )
        return cls(slope=slope, intercept=intercept, objective=objective)

    def predict(self, score):
        score = _vector(score, "score")
        result = self.slope * score + self.intercept
        if not np.isfinite(result).all():
            raise ValueError("affine predictions must be finite")
        return np.asarray(result, dtype=np.float64)

    def to_dict(self):
        return {
            "schema_version": 1,
            "type": "positive_affine",
            "objective": self.objective,
            "slope": self.slope,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload):
        required = {
            "schema_version",
            "type",
            "objective",
            "slope",
            "intercept",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("affine payload fields do not match")
        if payload["schema_version"] != 1 or payload["type"] != "positive_affine":
            raise ValueError("affine payload schema does not match")
        return cls(
            slope=payload["slope"],
            intercept=payload["intercept"],
            objective=payload["objective"],
        )


def _strict_pava(values, weights, increment):
    offsets = np.arange(values.shape[0], dtype=np.float64) * increment
    result = weighted_pava(values - offsets, weights) + offsets
    for index in range(1, result.shape[0]):
        if result[index] - result[index - 1] < increment:
            result[index] = np.nextafter(
                result[index - 1] + increment, np.inf
            )
    return result


@dataclass(frozen=True)
class MonotonicSplineCalibrator:
    knot_count: int
    x_knots: np.ndarray
    y_knots: np.ndarray
    min_increment: float = _MIN_INCREMENT

    def __post_init__(self):
        if (
            isinstance(self.knot_count, bool)
            or not isinstance(self.knot_count, Integral)
            or int(self.knot_count) not in (8, 16)
        ):
            raise ValueError("spline knot_count must be 8 or 16")
        x_knots = _vector(self.x_knots, "spline x knots").copy()
        y_knots = _vector(self.y_knots, "spline y knots").copy()
        if x_knots.shape != y_knots.shape or x_knots.shape[0] < 2:
            raise ValueError("spline knots must be aligned with at least two values")
        if np.any(np.diff(x_knots) <= 0.0):
            raise ValueError("spline x knots must be unique and increasing")
        if np.any(np.diff(y_knots) <= 0.0):
            raise ValueError("spline y knots must be strictly increasing")
        if (
            isinstance(self.min_increment, bool)
            or not isinstance(self.min_increment, Real)
            or not math.isfinite(float(self.min_increment))
            or float(self.min_increment) <= 0.0
        ):
            raise ValueError("spline minimum increment must be positive")
        x_knots.setflags(write=False)
        y_knots.setflags(write=False)
        object.__setattr__(self, "knot_count", int(self.knot_count))
        object.__setattr__(self, "x_knots", x_knots)
        object.__setattr__(self, "y_knots", y_knots)
        object.__setattr__(self, "min_increment", float(self.min_increment))

    @classmethod
    def fit(cls, score, label, knot_count, weights=None):
        if (
            isinstance(knot_count, bool)
            or not isinstance(knot_count, Integral)
            or int(knot_count) not in (8, 16)
        ):
            raise ValueError("spline knot_count must be 8 or 16")
        score, label, weights = _fit_inputs(score, label, weights)
        if np.unique(score).shape[0] < 2:
            raise ValueError("spline requires at least two unique scores")
        quantiles = np.linspace(0.0, 1.0, int(knot_count))
        initial = np.unique(np.quantile(score, quantiles))
        edges = 0.5 * (initial[:-1] + initial[1:])
        bins = np.searchsorted(edges, score, side="right")
        x_values = []
        y_values = []
        masses = []
        for index in range(initial.shape[0]):
            selected = bins == index
            if not selected.any():
                continue
            selected_weights = weights[selected]
            mass = float(selected_weights.sum(dtype=np.float64))
            x_values.append(float(initial[index]))
            y_values.append(
                float(np.dot(selected_weights, label[selected]) / mass)
            )
            masses.append(mass)
        x_knots = np.asarray(x_values, dtype=np.float64)
        if np.unique(x_knots).shape[0] < 2:
            raise ValueError("spline requires at least two unique knots")
        y_knots = _strict_pava(
            np.asarray(y_values, dtype=np.float64),
            np.asarray(masses, dtype=np.float64),
            _MIN_INCREMENT,
        )
        return cls(
            knot_count=int(knot_count),
            x_knots=x_knots,
            y_knots=y_knots,
            min_increment=_MIN_INCREMENT,
        )

    def predict(self, score):
        score = _vector(score, "score")
        result = np.interp(score, self.x_knots, self.y_knots)
        left_slope = (
            (self.y_knots[1] - self.y_knots[0])
            / (self.x_knots[1] - self.x_knots[0])
        )
        right_slope = (
            (self.y_knots[-1] - self.y_knots[-2])
            / (self.x_knots[-1] - self.x_knots[-2])
        )
        left = score < self.x_knots[0]
        right = score > self.x_knots[-1]
        result[left] = self.y_knots[0] + left_slope * (
            score[left] - self.x_knots[0]
        )
        result[right] = self.y_knots[-1] + right_slope * (
            score[right] - self.x_knots[-1]
        )
        if not np.isfinite(result).all():
            raise ValueError("spline predictions must be finite")
        return np.asarray(result, dtype=np.float64)

    def to_dict(self):
        return {
            "schema_version": 1,
            "type": "monotonic_spline",
            "knot_count": self.knot_count,
            "min_increment": self.min_increment,
            "x_knots": self.x_knots.tolist(),
            "y_knots": self.y_knots.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        required = {
            "schema_version",
            "type",
            "knot_count",
            "min_increment",
            "x_knots",
            "y_knots",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("spline payload fields do not match")
        if payload["schema_version"] != 1 or payload["type"] != "monotonic_spline":
            raise ValueError("spline payload schema does not match")
        return cls(
            knot_count=payload["knot_count"],
            x_knots=payload["x_knots"],
            y_knots=payload["y_knots"],
            min_increment=payload["min_increment"],
        )


__all__ = [
    "MonotonicSplineCalibrator",
    "PositiveAffineCalibrator",
]
