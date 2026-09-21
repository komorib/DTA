"""Losses for V27 privileged geometry and same-target relative co-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import DISTANCE_BOUNDARIES


@dataclass(frozen=True)
class LossConfig:
    base_smooth_weight: float = 1.0
    full_smooth_weight: float = 1.0
    mse_weight: float = 0.25
    rank_weight: float = 0.20
    full_rank_weight: float = 0.10
    binary_weight: float = 0.20
    relative_weight: float = 0.30
    relative_consistency_weight: float = 0.10
    affinity_distillation_weight: float = 0.10
    pair_distillation_weight: float = 0.15
    quality_distillation_weight: float = 0.10
    contact_weight: float = 0.10
    distance_weight: float = 0.10
    delta_l2_weight: float = 0.005
    gate_regularization_weight: float = 0.002
    binary_temperature: float = 0.50
    minimum_rank_gap: float = 0.10
    maximum_relative_pairs: int = 64


def _same_target_pairs(labels, dataset_index, target_group, valid, minimum_gap, maximum_pairs):
    left, right = [], []
    count = int(labels.shape[0])
    for first in range(count):
        if not bool(valid[first]):
            continue
        for second in range(first + 1, count):
            if not bool(valid[second]):
                continue
            if int(dataset_index[first]) != int(dataset_index[second]):
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
    target_difference = labels[left] - labels[right]
    predicted_difference = prediction[left] - prediction[right]
    signed_margin = torch.sign(target_difference) * predicted_difference
    return F.softplus(-signed_margin).mean()


def _binary_loss(prediction, labels, dataset_index, thresholds, positive_weights, temperature):
    losses = []
    for dataset in torch.unique(dataset_index).tolist():
        threshold = thresholds[int(dataset)]
        if not torch.isfinite(threshold):
            continue
        selected = dataset_index == int(dataset)
        logits = (prediction[selected] - threshold.to(prediction.dtype)) / float(temperature)
        targets = (labels[selected] >= threshold).to(logits.dtype)
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=positive_weights[int(dataset)].to(logits.dtype),
            )
        )
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def _balanced_contact_loss(logits, targets):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    targets = targets.to(logits.dtype)
    positive = targets.sum()
    negative = targets.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def privileged_pair_loss(
    model,
    output,
    normalized_labels,
    batch,
    thresholds,
    positive_weights,
    config=None,
):
    config = config or LossConfig()
    labels = normalized_labels.to(output.base_prediction.dtype)
    dataset_index = batch["dataset_index"].long()
    target_group = batch["target_group"].long()
    all_valid = torch.ones_like(dataset_index, dtype=torch.bool)
    structure_valid = output.effective_has_3d.bool()
    left, right = _same_target_pairs(
        labels,
        dataset_index,
        target_group,
        all_valid,
        config.minimum_rank_gap,
        config.maximum_relative_pairs,
    )

    base_smooth = F.smooth_l1_loss(output.base_prediction, labels, beta=0.5)
    base_mse = F.mse_loss(output.base_prediction, labels)
    rank = _rank_loss(output.base_prediction, labels, left, right)
    binary = _binary_loss(
        output.base_prediction,
        labels,
        dataset_index,
        thresholds,
        positive_weights,
        config.binary_temperature,
    )
    if left.numel():
        relative_prediction = model.relative_difference(
            output.base_latent[left], output.base_latent[right], dataset_index[left]
        )
        relative_target = labels[left] - labels[right]
        relative = F.smooth_l1_loss(relative_prediction, relative_target, beta=0.5)
        relative_consistency = F.smooth_l1_loss(
            relative_prediction,
            (output.base_prediction[left] - output.base_prediction[right]).detach(),
            beta=0.5,
        )
    else:
        relative = output.base_prediction.sum() * 0.0
        relative_consistency = relative

    if structure_valid.any():
        full_smooth = F.smooth_l1_loss(
            output.prediction[structure_valid], labels[structure_valid], beta=0.5
        )
        full_mse = F.mse_loss(output.prediction[structure_valid], labels[structure_valid])
        full_left, full_right = _same_target_pairs(
            labels,
            dataset_index,
            target_group,
            structure_valid,
            config.minimum_rank_gap,
            config.maximum_relative_pairs,
        )
        full_rank = _rank_loss(output.prediction, labels, full_left, full_right)
        affinity_distillation = F.smooth_l1_loss(
            output.base_prediction[structure_valid],
            output.prediction[structure_valid].detach(),
            beta=0.5,
        )
        pair_distillation = F.smooth_l1_loss(
            output.sequence_pair_latent[structure_valid],
            output.full_pair_latent[structure_valid].detach(),
            beta=0.5,
        )
        quality_distillation = F.smooth_l1_loss(
            output.geometry_quality_prediction[structure_valid],
            output.geometry_quality_target[structure_valid].detach(),
            beta=0.2,
        )
        pair_mask = (
            batch["pocket_mask"][:, :, None]
            & batch["structure_ligand_mask"][:, None, :]
            & structure_valid[:, None, None]
        )
        auxiliary_mask = batch.get("geometry_masked", pair_mask).bool() & pair_mask
        if not auxiliary_mask.any():
            auxiliary_mask = pair_mask
        contact_targets = batch.get("contact_label")
        if contact_targets is None:
            contact_targets = batch["residue_ligand_distance"] <= 4.5
        contact = _balanced_contact_loss(
            output.contact_logits[auxiliary_mask], contact_targets[auxiliary_mask]
        )
        boundaries = torch.as_tensor(
            DISTANCE_BOUNDARIES,
            dtype=batch["residue_ligand_distance"].dtype,
            device=batch["residue_ligand_distance"].device,
        )
        distance_targets = torch.bucketize(batch["residue_ligand_distance"], boundaries)
        distance = F.cross_entropy(
            output.distance_logits[auxiliary_mask], distance_targets[auxiliary_mask]
        )
        delta_l2 = output.structure_delta[structure_valid].square().mean()
        gate_regularization = (
            output.structure_gate[structure_valid].float() - 1.0
        ).square().mean()
    else:
        zero = output.base_prediction.sum() * 0.0
        full_smooth = zero
        full_mse = zero
        full_rank = zero
        affinity_distillation = zero
        pair_distillation = zero
        quality_distillation = zero
        contact = zero
        distance = zero
        delta_l2 = zero
        gate_regularization = zero

    total = (
        config.base_smooth_weight * base_smooth
        + config.full_smooth_weight * full_smooth
        + config.mse_weight * (base_mse + full_mse)
        + config.rank_weight * rank
        + config.full_rank_weight * full_rank
        + config.binary_weight * binary
        + config.relative_weight * relative
        + config.relative_consistency_weight * relative_consistency
        + config.affinity_distillation_weight * affinity_distillation
        + config.pair_distillation_weight * pair_distillation
        + config.quality_distillation_weight * quality_distillation
        + config.contact_weight * contact
        + config.distance_weight * distance
        + config.delta_l2_weight * delta_l2
        + config.gate_regularization_weight * gate_regularization
    )
    components = {
        "loss": float(total.detach()),
        "base_smooth": float(base_smooth.detach()),
        "full_smooth": float(full_smooth.detach()),
        "base_mse": float(base_mse.detach()),
        "full_mse": float(full_mse.detach()),
        "rank": float(rank.detach()),
        "full_rank": float(full_rank.detach()),
        "binary": float(binary.detach()),
        "relative": float(relative.detach()),
        "relative_consistency": float(relative_consistency.detach()),
        "affinity_distillation": float(affinity_distillation.detach()),
        "pair_distillation": float(pair_distillation.detach()),
        "quality_distillation": float(quality_distillation.detach()),
        "contact": float(contact.detach()),
        "distance": float(distance.detach()),
        "delta_l2": float(delta_l2.detach()),
        "gate_regularization": float(gate_regularization.detach()),
        "relative_pairs": int(left.numel()),
    }
    return total, components


__all__ = ["LossConfig", "privileged_pair_loss"]
