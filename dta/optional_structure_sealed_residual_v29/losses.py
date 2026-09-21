"""Losses for the V29 sealed-V22 bounded structured residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    smooth_weight: float = 1.0
    mse_weight: float = 0.25
    rank_weight: float = 0.20
    backbone_anchor_weight: float = 0.08
    delta_l2_weight: float = 0.01
    gate_center_weight: float = 0.002
    minimum_rank_gap: float = 0.10
    maximum_relative_pairs: int = 64


def _same_target_pairs(labels, target_group, valid, minimum_gap, maximum_pairs):
    left, right = [], []
    count = int(labels.shape[0])
    for first in range(count):
        if not bool(valid[first]):
            continue
        for second in range(first + 1, count):
            if not bool(valid[second]):
                continue
            if int(target_group[first]) != int(target_group[second]):
                continue
            if abs(float(labels[first].detach() - labels[second].detach())) < float(minimum_gap):
                continue
            left.append(first)
            right.append(second)
            if len(left) >= int(maximum_pairs):
                break
        if len(left) >= int(maximum_pairs):
            break
    device = labels.device
    return (
        torch.as_tensor(left, dtype=torch.long, device=device),
        torch.as_tensor(right, dtype=torch.long, device=device),
    )


def _rank_loss(prediction, labels, left, right):
    if left.numel() == 0:
        return prediction.sum() * 0.0
    sign = torch.sign(labels[left] - labels[right])
    return F.softplus(-sign * (prediction[left] - prediction[right])).mean()


def sealed_residual_loss(output, normalized_labels, batch, config=None):
    """Supervise only V22-full plus the bounded V29 structured delta."""

    config = config or LossConfig()
    labels = normalized_labels.to(output.prediction.dtype)
    valid = output.effective_has_3d.bool()
    zero = output.prediction.sum() * 0.0
    if not valid.any():
        raise RuntimeError("V29 residual training batch has no structured samples")
    target_group = batch["target_group"].long()
    left, right = _same_target_pairs(
        labels,
        target_group,
        valid,
        config.minimum_rank_gap,
        config.maximum_relative_pairs,
    )
    prediction = output.prediction
    smooth = F.smooth_l1_loss(prediction[valid], labels[valid], beta=0.5)
    mse = F.mse_loss(prediction[valid], labels[valid])
    rank = _rank_loss(prediction, labels, left, right)
    backbone_anchor = F.smooth_l1_loss(
        prediction[valid].float(),
        output.backbone_prediction[valid].detach().float(),
        beta=0.25,
    )
    delta_l2 = output.structure_delta[valid].float().square().mean()
    gate_center = (output.structure_gate[valid].float() - 0.5).square().mean()
    loss = (
        config.smooth_weight * smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.backbone_anchor_weight * backbone_anchor
        + config.delta_l2_weight * delta_l2
        + config.gate_center_weight * gate_center
    )
    components = {
        "loss": float(loss.detach()),
        "smooth": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "backbone_anchor": float(backbone_anchor.detach()),
        "delta_l2": float(delta_l2.detach()),
        "gate_center": float(gate_center.detach()),
        "relative_pairs": float(left.numel()),
        "mean_gate": float(output.structure_gate[valid].detach().float().mean()),
        "mean_absolute_delta": float(output.structure_delta[valid].detach().float().abs().mean()),
        "maximum_absolute_delta": float(output.structure_delta[valid].detach().float().abs().max()),
    }
    return loss, components


__all__ = ["LossConfig", "sealed_residual_loss"]
