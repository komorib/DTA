"""Exact ranking and affinity metrics for Rank Fusion v4."""

import math

import numpy as np


class _Fenwick:
    def __init__(self, size):
        self.values = [0] * (int(size) + 1)

    def add(self, index, value=1):
        index = int(index) + 1
        while index < len(self.values):
            self.values[index] += value
            index += index & -index

    def prefix(self, stop):
        total = 0
        index = int(stop)
        while index > 0:
            total += self.values[index]
            index -= index & -index
        return total


def _finite_vectors(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != predictions.size:
        raise ValueError("labels and predictions must have the same nonzero length")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("labels and predictions contain nonfinite values")
    return labels, predictions


def fast_concordance_index(labels, predictions):
    """Return exact repository CI semantics using grouped Fenwick-tree queries."""
    labels, predictions = _finite_vectors(labels, predictions)
    prediction_values = np.unique(predictions)
    prediction_ranks = np.searchsorted(prediction_values, predictions)
    order = np.argsort(labels, kind="mergesort")
    tree = _Fenwick(len(prediction_values))
    processed = 0
    comparable = 0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        label = labels[order[start]]
        while end < len(order) and labels[order[end]] == label:
            end += 1
        for position in range(start, end):
            rank = int(prediction_ranks[order[position]])
            lower = tree.prefix(rank)
            equal = tree.prefix(rank + 1) - lower
            concordant += float(lower) + 0.5 * float(equal)
            comparable += processed
        for position in range(start, end):
            tree.add(int(prediction_ranks[order[position]]))
            processed += 1
        start = end
    return concordant / comparable if comparable else 0.5


def _stable_rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0.0:
        return 0.0
    value = float(np.dot(left, right) / denominator)
    return value if math.isfinite(value) else 0.0


def affinity_metrics(labels, predictions):
    """Return CI, MAE, RMSE, Pearson, and stable-rank Spearman."""
    labels, predictions = _finite_vectors(labels, predictions)
    difference = labels - predictions
    return {
        "CI": fast_concordance_index(labels, predictions),
        "MAE": float(np.abs(difference).mean()),
        "RMSE": float(np.sqrt(np.square(difference).mean())),
        "Pearson": _correlation(labels, predictions),
        "Spearman": _correlation(
            _stable_rank(labels), _stable_rank(predictions)
        ),
    }
