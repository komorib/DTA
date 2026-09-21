"""Phase-specific objectives for staged optional-structure V22 training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import DATASET_INDEX
from .model import DISTANCE_BOUNDARIES


@dataclass(frozen=True)
class SequenceLossConfig:
    smooth_weight: float = 1.0
    mse_weight: float = 0.50
    rank_weight: float = 0.25
    binary_weight: float = 0.20
    binary_temperature: float = 0.50
    minimum_rank_gap: float = 0.10


@dataclass(frozen=True)
class StructureLossConfig:
    smooth_weight: float = 1.0
    mse_weight: float = 0.30
    rank_weight: float = 0.20
    contact_weight: float = 0.10
    distance_weight: float = 0.10
    delta_l2_weight: float = 0.01
    gate_regularization_weight: float = 0.005
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


def sequence_only_loss(
    output,
    normalized_labels,
    batch,
    normalized_thresholds,
    positive_weights,
    config=None,
):
    config = config or SequenceLossConfig()
    prediction = output.base_prediction
    labels = normalized_labels.to(prediction.dtype)
    dataset_index = batch["dataset_index"].long()
    valid = torch.ones_like(dataset_index, dtype=torch.bool)
    smooth = F.smooth_l1_loss(prediction, labels, beta=0.5)
    mse = F.mse_loss(prediction, labels)
    rank = _pairwise_rank_loss(
        prediction, labels, dataset_index, valid, config.minimum_rank_gap
    )
    binary_losses = []
    for index in range(len(DATASET_INDEX)):
        threshold = normalized_thresholds[index]
        selected = dataset_index == index
        if selected.any() and torch.isfinite(threshold):
            target = (labels[selected] >= threshold).to(prediction.dtype)
            logits = (prediction[selected] - threshold) / float(config.binary_temperature)
            binary_losses.append(
                F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=positive_weights[index].to(prediction.dtype),
                )
            )
    binary = torch.stack(binary_losses).mean() if binary_losses else prediction.sum() * 0.0
    total = (
        config.smooth_weight * smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.binary_weight * binary
    )
    return total, {
        "loss": float(total.detach()),
        "smooth": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "binary": float(binary.detach()),
    }


def _balanced_contact_loss(logits, targets):
    targets = targets.to(logits.dtype)
    positive = targets.sum()
    negative = targets.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)


def structure_residual_loss(output, normalized_labels, batch, config=None):
    config = config or StructureLossConfig()
    labels = normalized_labels.to(output.prediction.dtype)
    dataset_index = batch["dataset_index"].long()
    valid = output.effective_has_3d
    if not valid.any():
        raise ValueError("V22 structure phase requires at least one 3D sample per batch")
    smooth = F.smooth_l1_loss(output.prediction[valid], labels[valid], beta=0.5)
    mse = F.mse_loss(output.prediction[valid], labels[valid])
    rank = _pairwise_rank_loss(
        output.prediction, labels, dataset_index, valid, config.minimum_rank_gap
    )
    pair_mask = (
        batch["pocket_mask"][:, :, None]
        & batch["structure_ligand_mask"][:, None, :]
        & valid[:, None, None]
    )
    contact_targets = batch.get("contact_label")
    if contact_targets is None:
        contact_targets = batch["residue_ligand_distance"] <= 4.5
    contact = _balanced_contact_loss(
        output.contact_logits[pair_mask], contact_targets[pair_mask]
    )
    boundaries = torch.as_tensor(
        DISTANCE_BOUNDARIES,
        dtype=batch["residue_ligand_distance"].dtype,
        device=batch["residue_ligand_distance"].device,
    )
    distance_targets = torch.bucketize(batch["residue_ligand_distance"], boundaries)
    distance = F.cross_entropy(output.distance_logits[pair_mask], distance_targets[pair_mask])
    delta_l2 = output.structure_delta[valid].square().mean()
    gate_regularization = (output.structure_gate[valid].float() - 1.0).square().mean()
    total = (
        config.smooth_weight * smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.contact_weight * contact
        + config.distance_weight * distance
        + config.delta_l2_weight * delta_l2
        + config.gate_regularization_weight * gate_regularization
    )
    return total, {
        "loss": float(total.detach()),
        "smooth": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "contact": float(contact.detach()),
        "distance": float(distance.detach()),
        "delta_l2": float(delta_l2.detach()),
        "gate_regularization": float(gate_regularization.detach()),
    }


__all__ = [
    "SequenceLossConfig",
    "StructureLossConfig",
    "sequence_only_loss",
    "structure_residual_loss",
]
