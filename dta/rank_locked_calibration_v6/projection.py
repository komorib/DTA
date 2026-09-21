"""Deterministic float64 tie-aware rank projection for calibration v6."""

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

import numpy as np


def _finite_float64_vector(value, name, *, nonempty=True):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a finite vector") from error
    if result.ndim != 1:
        raise ValueError(name + " must be a vector")
    if nonempty and result.size == 0:
        raise ValueError(name + " must be nonempty")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def weighted_pava(values, weights):
    """Return the weighted least-squares nondecreasing projection."""
    values = _finite_float64_vector(values, "values")
    weights = _finite_float64_vector(weights, "weights")
    if values.shape != weights.shape:
        raise ValueError("PAVA values and weights must be aligned")
    if np.any(weights <= 0.0):
        raise ValueError("PAVA weights must be positive")
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, float(value), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            mass = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / mass
            blocks.append([left[0], right[1], mean, mass])
    output = np.empty_like(values)
    for start, stop, mean, _ in blocks:
        output[start:stop] = mean
    return output


def _order_signature(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_groups = np.empty(values.shape[0], dtype=np.int64)
    group = 0
    sorted_groups[0] = group
    for index in range(1, values.shape[0]):
        if sorted_values[index] != sorted_values[index - 1]:
            group += 1
        sorted_groups[index] = group
    groups = np.empty_like(sorted_groups)
    groups[order] = sorted_groups
    payload = json.dumps(
        groups.tolist(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProjectionReport:
    row_count: int
    group_count: int
    tie_group_count: int
    tied_row_count: int
    epsilon: float
    minimum_distinct_gap: object
    inversion_count: int
    base_order_sha256: str
    final_order_sha256: str


def _validated_epsilon(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("epsilon must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be a positive finite number")
    return result


def rank_locked_project(base, proposed, epsilon=1e-8):
    """Project proposed values while preserving every base order relation."""
    base = _finite_float64_vector(base, "base")
    proposed = _finite_float64_vector(proposed, "proposed")
    if base.shape != proposed.shape:
        raise ValueError("base and proposed shape must match")
    epsilon = _validated_epsilon(epsilon)

    order = np.argsort(base, kind="mergesort")
    sorted_base = base[order]
    sorted_proposed = proposed[order]
    starts = [0]
    for index in range(1, base.shape[0]):
        if sorted_base[index] != sorted_base[index - 1]:
            starts.append(index)
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.append(starts[1:], base.shape[0])
    group_sizes = (stops - starts).astype(np.float64)
    group_means = np.asarray(
        [
            sorted_proposed[start:stop].mean(dtype=np.float64)
            for start, stop in zip(starts, stops)
        ],
        dtype=np.float64,
    )

    offsets = np.arange(starts.shape[0], dtype=np.float64) * epsilon
    group_values = weighted_pava(group_means - offsets, group_sizes) + offsets
    for index in range(1, group_values.shape[0]):
        if group_values[index] - group_values[index - 1] < epsilon:
            group_values[index] = np.nextafter(
                group_values[index - 1] + epsilon, np.inf
            )
    if not np.isfinite(group_values).all():
        raise ValueError("rank projection produced nonfinite values")

    final_sorted = np.empty_like(sorted_proposed)
    for value, start, stop in zip(group_values, starts, stops):
        final_sorted[start:stop] = value
    final = np.empty_like(final_sorted)
    final[order] = final_sorted

    gaps = np.diff(group_values)
    inversion_count = int(np.count_nonzero(gaps <= 0.0))
    if inversion_count:
        raise ValueError("rank projection contains an inversion")
    minimum_gap = None if gaps.size == 0 else float(gaps.min())
    if minimum_gap is not None and minimum_gap < epsilon:
        raise ValueError("rank projection distinct gap is below epsilon")
    base_hash = _order_signature(base)
    final_hash = _order_signature(final)
    if base_hash != final_hash:
        raise ValueError("rank projection changed the complete order relation")

    sizes = group_sizes.astype(np.int64)
    report = ProjectionReport(
        row_count=int(base.shape[0]),
        group_count=int(starts.shape[0]),
        tie_group_count=int(np.count_nonzero(sizes > 1)),
        tied_row_count=int(sizes[sizes > 1].sum()),
        epsilon=epsilon,
        minimum_distinct_gap=minimum_gap,
        inversion_count=0,
        base_order_sha256=base_hash,
        final_order_sha256=final_hash,
    )
    final.setflags(write=False)
    return final, report


__all__ = [
    "ProjectionReport",
    "rank_locked_project",
    "weighted_pava",
]
