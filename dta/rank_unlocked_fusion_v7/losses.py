"""Regression and CI-oriented ranking losses for fusion v7."""

import torch
from torch.nn import functional as F


def _vectors(prediction, labels, anchor=None):
    values = [prediction, labels] if anchor is None else [prediction, labels, anchor]
    if any(
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or value.ndim != 1
        for value in values
    ):
        raise ValueError("loss inputs must be floating vectors")
    if any(value.shape != prediction.shape for value in values[1:]):
        raise ValueError("loss input shapes must match")
    if any(
        value.device != prediction.device or value.dtype != prediction.dtype
        for value in values[1:]
    ):
        raise ValueError("loss inputs must share device and dtype")
    if prediction.numel() == 0 or not torch.stack(
        [torch.isfinite(value).all() for value in values]
    ).all():
        raise ValueError("loss inputs must be nonempty and finite")
    return values


def _oriented_pairs(labels, minimum_gap):
    left, right = torch.triu_indices(
        labels.numel(), labels.numel(), offset=1, device=labels.device
    )
    difference = labels[left] - labels[right]
    eligible = difference.abs() >= float(minimum_gap)
    left = left[eligible]
    right = right[eligible]
    difference = difference[eligible]
    high = torch.where(difference > 0, left, right)
    low = torch.where(difference > 0, right, left)
    return high, low


def pairwise_logistic_loss(prediction, labels, minimum_gap=1e-6, temperature=1.0):
    prediction, labels = _vectors(prediction, labels)
    if minimum_gap < 0.0 or temperature <= 0.0:
        raise ValueError("pairwise gap and temperature are invalid")
    high, low = _oriented_pairs(labels, minimum_gap)
    if high.numel() == 0:
        return prediction.sum() * 0.0, 0
    margin = (prediction[high] - prediction[low]) / float(temperature)
    return F.softplus(-margin).mean(), int(high.numel())


def hard_pairwise_loss(
    prediction,
    labels,
    anchor,
    minimum_gap=0.10,
    maximum_pairs=4096,
):
    prediction, labels, anchor = _vectors(prediction, labels, anchor)
    high, low = _oriented_pairs(labels, minimum_gap)
    if high.numel() == 0:
        return prediction.sum() * 0.0, 0
    anchor_margin = anchor[high] - anchor[low]
    order = torch.argsort(anchor_margin, stable=True)
    order = order[: int(maximum_pairs)]
    high = high[order]
    low = low[order]
    return F.softplus(-(prediction[high] - prediction[low])).mean(), int(high.numel())


def rank_unlocked_objective(prediction, labels, anchor):
    prediction, labels, anchor = _vectors(prediction, labels, anchor)
    smooth_l1 = F.smooth_l1_loss(prediction, labels)
    mse = F.mse_loss(prediction, labels)
    pairwise, pair_count = pairwise_logistic_loss(prediction, labels)
    hard_pairwise, hard_pair_count = hard_pairwise_loss(
        prediction, labels, anchor
    )
    residual = F.mse_loss(prediction, anchor)
    total = (
        0.35 * smooth_l1
        + 0.10 * mse
        + 0.35 * pairwise
        + 0.15 * hard_pairwise
        + 0.05 * residual
    )
    return total, {
        "smooth_l1": smooth_l1,
        "mse": mse,
        "pairwise": pairwise,
        "hard_pairwise": hard_pairwise,
        "residual": residual,
        "pair_count": pair_count,
        "hard_pair_count": hard_pair_count,
    }


__all__ = [
    "hard_pairwise_loss",
    "pairwise_logistic_loss",
    "rank_unlocked_objective",
]
