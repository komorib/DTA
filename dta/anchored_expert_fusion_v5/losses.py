"""Targeted pair mining and anchored loss for expert fusion v5."""

import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F


def _floating_vector(value, name, nonempty=True):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    if value.ndim != 1:
        raise ValueError(name + " must be a 1D tensor")
    if nonempty and value.numel() == 0:
        raise ValueError(name + " must be nonempty")
    return value


def _validate_finite(named_tensors):
    # Valid calls synchronize once; field-specific checks run only on failure.
    checks = [torch.isfinite(tensor).all() for _, tensor in named_tensors]
    if torch.stack(checks).all().item():
        return
    for name, tensor in named_tensors:
        if not torch.isfinite(tensor).all().item():
            raise ValueError(name + " must be finite")


def _validate_mining_arguments(maximum_pairs, minimum_label_gap):
    if (
        isinstance(maximum_pairs, bool)
        or not isinstance(maximum_pairs, Integral)
        or maximum_pairs <= 0
    ):
        raise ValueError("maximum_pairs must be a positive integer")
    if (
        isinstance(minimum_label_gap, bool)
        or not isinstance(minimum_label_gap, Real)
    ):
        raise ValueError("minimum_label_gap must be finite and nonnegative")
    gap = float(minimum_label_gap)
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError("minimum_label_gap must be finite and nonnegative")
    return int(maximum_pairs), gap


def _stable_lexicographic_order(correct, margin, high, low):
    order = torch.arange(high.numel(), device=high.device)
    for key in (low, high, margin, correct):
        sort_key = key[order]
        if sort_key.dtype == torch.bool:
            sort_key = sort_key.to(dtype=torch.int64)
        order = order[torch.argsort(sort_key, stable=True)]
    return order


def _mine_targeted_pairs_validated(labels, base, maximum_pairs, gap):
    first, second = torch.triu_indices(
        labels.numel(), labels.numel(), offset=1, device=labels.device
    )
    eligible = (labels[first] - labels[second]).abs() >= gap
    first = first[eligible]
    second = second[eligible]
    if first.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=labels.device)

    first_is_high = labels[first] >= labels[second]
    high = torch.where(first_is_high, first, second)
    low = torch.where(first_is_high, second, first)
    margin = base[high] - base[low]
    correct = margin >= 0
    order = _stable_lexicographic_order(correct, margin, high, low)
    order = order[:maximum_pairs]
    return torch.stack((high[order], low[order]), dim=1).to(dtype=torch.long)


def mine_targeted_pairs(
    labels,
    base,
    maximum_pairs=4096,
    minimum_label_gap=0.5,
):
    """Mine deterministic hard pairs, oriented from high to low label."""
    labels = _floating_vector(labels, "labels")
    base = _floating_vector(base, "base")
    if labels.shape != base.shape:
        raise ValueError("labels and base must have the same shape")
    if labels.device != base.device:
        raise ValueError("labels and base must be on the same device")
    if labels.dtype != base.dtype:
        raise ValueError("labels and base must have the same dtype")
    maximum_pairs, gap = _validate_mining_arguments(
        maximum_pairs, minimum_label_gap
    )
    _validate_finite((("labels", labels), ("base", base)))
    return _mine_targeted_pairs_validated(labels, base, maximum_pairs, gap)


def anchored_residual_loss(
    final,
    base,
    labels,
    maximum_pairs=4096,
    minimum_label_gap=0.5,
):
    """Combine targeted pair, regression, and anchoring objectives."""
    final = _floating_vector(final, "final")
    base = _floating_vector(base, "base")
    labels = _floating_vector(labels, "labels")
    if final.shape != base.shape or final.shape != labels.shape:
        raise ValueError("final, base, and labels must have matching shapes")
    if final.device != base.device or final.device != labels.device:
        raise ValueError("final, base, and labels must be on the same device")
    if final.dtype != base.dtype or final.dtype != labels.dtype:
        raise ValueError("final, base, and labels must have the same dtype")
    maximum_pairs, gap = _validate_mining_arguments(
        maximum_pairs, minimum_label_gap
    )
    _validate_finite(
        (("final", final), ("base", base), ("labels", labels))
    )
    pairs = _mine_targeted_pairs_validated(
        labels, base, maximum_pairs, gap
    )
    pair_count = int(pairs.shape[0])
    if pair_count:
        high, low = pairs[:, 0], pairs[:, 1]
        targeted_pair = F.softplus(-(final[high] - final[low])).mean()
    else:
        targeted_pair = final.sum() * 0.0
    smooth_l1 = F.smooth_l1_loss(final, labels)
    anchor = ((final - base) ** 2).mean()
    total = targeted_pair + 0.25 * smooth_l1 + 0.50 * anchor
    return total, {
        "targeted_pair": targeted_pair,
        "smooth_l1": smooth_l1,
        "anchor": anchor,
        "pair_count": pair_count,
    }


__all__ = ["mine_targeted_pairs", "anchored_residual_loss"]
