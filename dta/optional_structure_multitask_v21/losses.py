"""Losses for joint sequence-only and optional-structure V21 training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import DISTANCE_BOUNDARIES


@dataclass
class LossConfig:
    base_affinity_weight: float = 1.0
    structure_affinity_weight: float = 1.0
    mse_weight: float = 0.15
    base_rank_weight: float = 0.15
    structure_rank_weight: float = 0.10
    contact_weight: float = 0.10
    distance_weight: float = 0.10
    distillation_weight: float = 0.05
    minimum_rank_gap: float = 0.10


def _pairwise_rank_loss(prediction, labels, dataset_index, valid, minimum_gap):
    losses = []
    for dataset in torch.unique(dataset_index[valid]).tolist():
        selected = valid & (dataset_index == int(dataset))
        values = prediction[selected]
        target = labels[selected]
        if values.numel() < 2:
            continue
        difference = target[:, None] - target[None, :]
        pair_mask = difference > float(minimum_gap)
        if pair_mask.any():
            prediction_difference = values[:, None] - values[None, :]
            losses.append(F.softplus(-prediction_difference[pair_mask]).mean())
    if losses:
        return torch.stack(losses).mean()
    return prediction.sum() * 0.0


def _balanced_binary_loss(logits, targets):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    targets = targets.to(logits.dtype)
    positive = targets.sum()
    negative = targets.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def optional_structure_loss(output, normalized_labels, batch, config=None):
    config = config or LossConfig()
    normalized_labels = normalized_labels.to(output.prediction.dtype)
    dataset_index = batch["dataset_index"].long()
    all_valid = torch.ones_like(output.effective_has_3d, dtype=torch.bool)
    structure_valid = output.effective_has_3d

    base_smooth = F.smooth_l1_loss(output.base_prediction, normalized_labels, beta=0.5)
    base_mse = F.mse_loss(output.base_prediction, normalized_labels)
    base_rank = _pairwise_rank_loss(
        output.base_prediction,
        normalized_labels,
        dataset_index,
        all_valid,
        config.minimum_rank_gap,
    )
    if structure_valid.any():
        structure_smooth = F.smooth_l1_loss(
            output.prediction[structure_valid], normalized_labels[structure_valid], beta=0.5
        )
        structure_mse = F.mse_loss(
            output.prediction[structure_valid], normalized_labels[structure_valid]
        )
        structure_rank = _pairwise_rank_loss(
            output.prediction,
            normalized_labels,
            dataset_index,
            structure_valid,
            config.minimum_rank_gap,
        )
        distillation = F.smooth_l1_loss(
            output.base_prediction[structure_valid],
            output.prediction[structure_valid].detach(),
            beta=0.5,
        )
        pair_mask = (
            batch["pocket_mask"][:, :, None]
            & batch["structure_ligand_mask"][:, None, :]
            & structure_valid[:, None, None]
        )
        contact_targets = batch.get("contact_label")
        if contact_targets is None:
            contact_targets = batch["residue_ligand_distance"] <= 4.5
        contact = _balanced_binary_loss(
            output.contact_logits[pair_mask], contact_targets[pair_mask]
        )
        boundaries = torch.as_tensor(
            DISTANCE_BOUNDARIES,
            dtype=batch["residue_ligand_distance"].dtype,
            device=batch["residue_ligand_distance"].device,
        )
        distance_targets = torch.bucketize(batch["residue_ligand_distance"], boundaries)
        distance = F.cross_entropy(
            output.distance_logits[pair_mask], distance_targets[pair_mask]
        )
    else:
        zero = output.prediction.sum() * 0.0
        structure_smooth = zero
        structure_mse = zero
        structure_rank = zero
        distillation = zero
        contact = zero
        distance = zero

    total = (
        config.base_affinity_weight * base_smooth
        + config.structure_affinity_weight * structure_smooth
        + config.mse_weight * (base_mse + structure_mse)
        + config.base_rank_weight * base_rank
        + config.structure_rank_weight * structure_rank
        + config.contact_weight * contact
        + config.distance_weight * distance
        + config.distillation_weight * distillation
    )
    components = {
        "loss": float(total.detach()),
        "base_affinity": float(base_smooth.detach()),
        "structure_affinity": float(structure_smooth.detach()),
        "base_mse": float(base_mse.detach()),
        "structure_mse": float(structure_mse.detach()),
        "base_rank": float(base_rank.detach()),
        "structure_rank": float(structure_rank.detach()),
        "contact": float(contact.detach()),
        "distance": float(distance.detach()),
        "distillation": float(distillation.detach()),
    }
    return total, components


__all__ = ["LossConfig", "optional_structure_loss"]
