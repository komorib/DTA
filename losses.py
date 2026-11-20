"""Affinity-only objective from the early-K/V experiment."""

import torch
from torch.nn import functional as F


def pairwise_rank_loss(prediction, labels, minimum_gap=0.10):
    label_delta = labels[:, None] - labels[None, :]
    pred_delta = prediction[:, None] - prediction[None, :]
    upper = torch.triu(torch.ones_like(label_delta, dtype=torch.bool), diagonal=1)
    valid = upper & (label_delta.abs() >= float(minimum_gap))
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.softplus(-label_delta[valid].sign() * pred_delta[valid]).mean()


def affinity_only_loss(output, normalized_labels, minimum_gap=0.10):
    prediction = output.prediction
    smooth = F.smooth_l1_loss(prediction, normalized_labels, beta=0.5)
    mse = F.mse_loss(prediction, normalized_labels)
    rank = pairwise_rank_loss(prediction, normalized_labels, minimum_gap)
    total = smooth + 0.15 * mse + 0.40 * rank
    return total, {"smooth_l1": float(smooth.detach()),
                   "mse": float(mse.detach()), "rank": float(rank.detach()),
                   "loss_auxiliary_effective": 0.0}
