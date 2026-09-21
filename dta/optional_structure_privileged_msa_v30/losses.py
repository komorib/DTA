"""V30 affinity/rank objectives with a train-only scalar MSA teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    smooth_weight: float = 1.0
    mse_weight: float = 0.25
    rank_weight: float = 0.20
    teacher_value_weight: float = 0.25
    teacher_rank_weight: float = 0.10
    backbone_anchor_weight: float = 0.04
    delta_l2_weight: float = 0.01
    gate_center_weight: float = 0.002
    mix_center_weight: float = 0.002
    minimum_rank_gap: float = 0.10
    maximum_relative_pairs: int = 64


def _same_target_pairs(values, target_group, valid, minimum_gap, maximum_pairs):
    left, right = [], []
    count = int(values.shape[0])
    for first in range(count):
        if not bool(valid[first]):
            continue
        for second in range(first + 1, count):
            if not bool(valid[second]) or int(target_group[first]) != int(target_group[second]):
                continue
            if abs(float(values[first].detach() - values[second].detach())) < float(minimum_gap):
                continue
            left.append(first)
            right.append(second)
            if len(left) >= int(maximum_pairs):
                break
        if len(left) >= int(maximum_pairs):
            break
    device = values.device
    return (
        torch.as_tensor(left, dtype=torch.long, device=device),
        torch.as_tensor(right, dtype=torch.long, device=device),
    )


def _rank_loss(prediction, values, left, right):
    if left.numel() == 0:
        return prediction.sum() * 0.0
    direction = torch.sign(values[left] - values[right])
    return F.softplus(-direction * (prediction[left] - prediction[right])).mean()


def privileged_msa_loss(output, normalized_labels, batch, config=None):
    """Train V30 only on structured LP rows with OOF MSA soft targets."""

    config = config or LossConfig()
    labels = normalized_labels.to(output.prediction.dtype)
    available = batch["teacher_available"].bool()
    valid = output.effective_has_3d.bool() & available
    if not valid.any():
        raise RuntimeError("V30 training batch has no structured teacher-supervised sample")
    teacher_values = batch["teacher_predictions_normalized"].to(output.prediction.dtype)
    target_group = batch["target_group"].long()
    label_left, label_right = _same_target_pairs(
        labels, target_group, valid, config.minimum_rank_gap, config.maximum_relative_pairs
    )
    teacher_left, teacher_right = _same_target_pairs(
        teacher_values,
        target_group,
        valid,
        config.minimum_rank_gap,
        config.maximum_relative_pairs,
    )
    smooth = F.smooth_l1_loss(output.prediction[valid], labels[valid], beta=0.5)
    mse = F.mse_loss(output.prediction[valid], labels[valid])
    rank = _rank_loss(output.prediction, labels, label_left, label_right)
    teacher_value = F.smooth_l1_loss(
        output.teacher_prediction[valid].float(), teacher_values[valid].float(), beta=0.5
    )
    teacher_rank = _rank_loss(
        output.teacher_prediction, teacher_values, teacher_left, teacher_right
    )
    backbone_anchor = F.smooth_l1_loss(
        output.prediction[valid].float(), output.backbone_prediction[valid].detach().float(), beta=0.25
    )
    delta_l2 = output.structure_delta[valid].float().square().mean()
    gate_center = (output.structure_gate[valid].float() - 0.5).square().mean()
    mix_center = (output.teacher_mix[valid].float() - 0.5).square().mean()
    total = (
        config.smooth_weight * smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.teacher_value_weight * teacher_value
        + config.teacher_rank_weight * teacher_rank
        + config.backbone_anchor_weight * backbone_anchor
        + config.delta_l2_weight * delta_l2
        + config.gate_center_weight * gate_center
        + config.mix_center_weight * mix_center
    )
    components = {
        "loss": float(total.detach()),
        "smooth": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "teacher_value": float(teacher_value.detach()),
        "teacher_rank": float(teacher_rank.detach()),
        "backbone_anchor": float(backbone_anchor.detach()),
        "delta_l2": float(delta_l2.detach()),
        "gate_center": float(gate_center.detach()),
        "mix_center": float(mix_center.detach()),
        "relative_pairs": float(label_left.numel()),
        "teacher_relative_pairs": float(teacher_left.numel()),
        "mean_gate": float(output.structure_gate[valid].detach().float().mean()),
        "mean_teacher_mix": float(output.teacher_mix[valid].detach().float().mean()),
        "mean_absolute_delta": float(output.structure_delta[valid].detach().float().abs().mean()),
        "mean_absolute_teacher_delta": float(output.teacher_delta[valid].detach().float().abs().mean()),
    }
    return total, components


__all__ = ["LossConfig", "privileged_msa_loss"]
