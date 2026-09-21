"""Bounded losses for V28 stable privileged-geometry pair co-training."""

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
    base_anchor_distillation_weight: float = 0.50
    privileged_affinity_distillation_weight: float = 0.02
    pair_distillation_weight: float = 0.03
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


def normalized_pair_distillation(student, teacher):
    """Scale-invariant, bounded pair matching with a stop-gradient teacher."""

    if student.shape != teacher.shape:
        raise ValueError("student and teacher pair latents must have identical shapes")
    if student.ndim != 2 or student.shape[-1] < 2:
        raise ValueError("pair latents must have shape [batch, width>=2]")
    width = int(student.shape[-1])
    student_float = student.float()
    teacher_float = teacher.detach().float()
    student_normalized = F.layer_norm(student_float, (width,))
    teacher_normalized = F.layer_norm(teacher_float, (width,))
    cosine = (1.0 - F.cosine_similarity(student_normalized, teacher_normalized, dim=-1)).mean()
    regression = F.smooth_l1_loss(
        student_normalized,
        teacher_normalized,
        beta=0.5,
    )
    return 0.5 * (cosine + regression), cosine, regression


def stable_pair_loss(
    model,
    output,
    normalized_labels,
    batch,
    thresholds,
    positive_weights,
    config=None,
    base_anchor_prediction=None,
    pair_teacher_latent=None,
    pair_distillation_scale=1.0,
):
    config = config or LossConfig()
    pair_distillation_scale = float(pair_distillation_scale)
    if not 0.0 <= pair_distillation_scale <= 1.0:
        raise ValueError("pair_distillation_scale must be in [0, 1]")
    labels = normalized_labels.to(output.base_prediction.dtype)
    dataset_index = batch["dataset_index"].long()
    target_group = batch["target_group"].long()
    all_valid = torch.ones_like(dataset_index, dtype=torch.bool)
    structure_valid = output.effective_has_3d.bool()
    zero = output.base_prediction.sum() * 0.0
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
    if base_anchor_prediction is None:
        base_anchor_distillation = zero
    else:
        if base_anchor_prediction.shape != output.base_prediction.shape:
            raise ValueError("base anchor prediction shape mismatch")
        base_anchor_distillation = F.smooth_l1_loss(
            output.base_prediction.float(),
            base_anchor_prediction.detach().float(),
            beta=0.25,
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
        relative = zero
        relative_consistency = zero

    pair_cosine = zero
    pair_regression = zero
    sequence_pair_norm = zero
    teacher_pair_norm = zero
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
        privileged_affinity_distillation = F.smooth_l1_loss(
            output.base_prediction[structure_valid],
            output.prediction[structure_valid].detach(),
            beta=0.5,
        )
        if pair_teacher_latent is None:
            pair_teacher_latent = output.full_pair_latent
        if pair_teacher_latent.shape != output.full_pair_latent.shape:
            raise ValueError("pair teacher latent shape mismatch")
        student_pair = output.sequence_pair_latent[structure_valid]
        teacher_pair = pair_teacher_latent[structure_valid]
        pair_distillation, pair_cosine, pair_regression = normalized_pair_distillation(
            student_pair,
            teacher_pair,
        )
        sequence_pair_norm = torch.linalg.vector_norm(student_pair.float(), dim=-1).mean()
        teacher_pair_norm = torch.linalg.vector_norm(teacher_pair.detach().float(), dim=-1).mean()
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
        full_smooth = zero
        full_mse = zero
        full_rank = zero
        privileged_affinity_distillation = zero
        pair_distillation = zero
        quality_distillation = zero
        contact = zero
        distance = zero
        delta_l2 = zero
        gate_regularization = zero

    effective_pair_weight = config.pair_distillation_weight * pair_distillation_scale
    total = (
        config.base_smooth_weight * base_smooth
        + config.full_smooth_weight * full_smooth
        + config.mse_weight * (base_mse + full_mse)
        + config.rank_weight * rank
        + config.full_rank_weight * full_rank
        + config.binary_weight * binary
        + config.relative_weight * relative
        + config.relative_consistency_weight * relative_consistency
        + config.base_anchor_distillation_weight * base_anchor_distillation
        + config.privileged_affinity_distillation_weight * privileged_affinity_distillation
        + effective_pair_weight * pair_distillation
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
        "base_anchor_distillation": float(base_anchor_distillation.detach()),
        "privileged_affinity_distillation": float(privileged_affinity_distillation.detach()),
        "pair_distillation": float(pair_distillation.detach()),
        "pair_cosine": float(pair_cosine.detach()),
        "pair_regression": float(pair_regression.detach()),
        "pair_distillation_scale": pair_distillation_scale,
        "effective_pair_distillation_weight": float(effective_pair_weight),
        "sequence_pair_norm": float(sequence_pair_norm.detach()),
        "teacher_pair_norm": float(teacher_pair_norm.detach()),
        "quality_distillation": float(quality_distillation.detach()),
        "contact": float(contact.detach()),
        "distance": float(distance.detach()),
        "delta_l2": float(delta_l2.detach()),
        "gate_regularization": float(gate_regularization.detach()),
        "relative_pairs": int(left.numel()),
    }
    return total, components


__all__ = ["LossConfig", "normalized_pair_distillation", "stable_pair_loss"]

