"""Deterministic float32 ranking-first losses for Rank Fusion v4."""

import torch
import torch.nn.functional as F


def _vectors(prediction, labels):
    prediction = prediction.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if prediction.numel() == 0 or prediction.shape != labels.shape:
        raise ValueError("prediction and labels must have the same nonzero shape")
    if not torch.isfinite(prediction).all() or not torch.isfinite(labels).all():
        raise ValueError("prediction or labels contain nonfinite values")
    return prediction, labels


def mine_hard_pairs(
    prediction,
    labels,
    expert_predictions,
    available,
    *,
    min_label_gap=0.10,
    max_pairs=2048,
):
    """Return higher-label/lower-label indices ordered by deterministic hardness."""
    prediction, labels = _vectors(prediction, labels)
    experts = expert_predictions.float()
    available = available.bool()
    if experts.ndim != 2 or experts.shape != available.shape:
        raise ValueError("expert_predictions and available shapes do not match")
    if experts.shape[0] != prediction.numel():
        raise ValueError("expert prediction batch does not match labels")
    if isinstance(max_pairs, bool) or int(max_pairs) <= 0:
        raise ValueError("max_pairs must be positive")
    candidates = []
    detached_prediction = prediction.detach()
    detached_labels = labels.detach()
    detached_experts = experts.detach()
    for first in range(prediction.numel()):
        for second in range(first + 1, prediction.numel()):
            difference = float(detached_labels[first] - detached_labels[second])
            if abs(difference) < float(min_label_gap):
                continue
            left, right = (first, second) if difference > 0.0 else (second, first)
            prediction_margin = float(
                detached_prediction[left] - detached_prediction[right]
            )
            incorrect = 1 if prediction_margin <= 0.0 else 0
            shared = available[left] & available[right]
            if bool(shared.any()):
                expert_margins = (
                    detached_experts[left, shared] - detached_experts[right, shared]
                )
                disagreement = float(
                    expert_margins.max() - expert_margins.min()
                )
                expert_wrong = float((expert_margins <= 0.0).float().mean())
            else:
                disagreement = 0.0
                expert_wrong = 0.0
            candidates.append(
                (
                    -incorrect,
                    -disagreement,
                    -expert_wrong,
                    -abs(difference),
                    left,
                    right,
                )
            )
    candidates.sort()
    selected = candidates[: int(max_pairs)]
    device = prediction.device
    left = torch.tensor([item[-2] for item in selected], device=device, dtype=torch.long)
    right = torch.tensor([item[-1] for item in selected], device=device, dtype=torch.long)
    return left, right


def hard_pair_logistic_loss(
    prediction,
    labels,
    expert_predictions,
    available,
    *,
    min_label_gap=0.10,
    max_pairs=2048,
):
    prediction, labels = _vectors(prediction, labels)
    left, right = mine_hard_pairs(
        prediction,
        labels,
        expert_predictions,
        available,
        min_label_gap=min_label_gap,
        max_pairs=max_pairs,
    )
    if left.numel() == 0:
        return prediction.sum() * 0.0
    return F.softplus(-(prediction[left] - prediction[right])).mean()


def _deterministic_label_order(labels):
    values = labels.detach().float().cpu().tolist()
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return torch.tensor(order, device=labels.device, dtype=torch.long)


def listmle_loss(prediction, labels):
    prediction, labels = _vectors(prediction, labels)
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    ordered = prediction[_deterministic_label_order(labels)]
    suffix_logsumexp = torch.logcumsumexp(ordered.flip(0), dim=0).flip(0)
    return (suffix_logsumexp[:-1] - ordered[:-1]).mean()


def _soft_rank(values, temperature):
    pairwise = (values[None, :] - values[:, None]) / float(temperature)
    return 1.0 + torch.sigmoid(pairwise).sum(dim=1) - 0.5


def soft_spearman_loss(prediction, labels, *, temperature=0.10):
    prediction, labels = _vectors(prediction, labels)
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    prediction_rank = _soft_rank(prediction, temperature)
    label_rank = _soft_rank(labels, temperature).detach()
    prediction_rank = prediction_rank - prediction_rank.mean()
    label_rank = label_rank - label_rank.mean()
    prediction_scale = prediction_rank.square().sum().sqrt().clamp_min(1e-6)
    label_scale = label_rank.square().sum().sqrt().clamp_min(1e-6)
    denominator = (prediction_scale * label_scale).detach()
    correlation = (prediction_rank * label_rank).sum() / denominator
    return 1.0 - correlation


def mixture_regularization(weights, available):
    weights = weights.float()
    available = available.bool()
    if weights.ndim != 2 or weights.shape != available.shape:
        raise ValueError("weights and available shapes do not match")
    if not bool(available.any(dim=1).all()):
        raise ValueError("every row requires an available expert")
    safe_weights = weights.clamp_min(1e-12)
    terms = torch.where(
        available,
        weights * torch.log(safe_weights),
        torch.zeros_like(weights),
    )
    return terms.sum(dim=1).mean()


def rank_fusion_loss(
    prediction,
    labels,
    expert_predictions,
    mixture_weights,
    available,
):
    prediction, labels = _vectors(prediction, labels)
    components = {
        "hard_pair_logistic": hard_pair_logistic_loss(
            prediction, labels, expert_predictions, available
        ),
        "listmle": listmle_loss(prediction, labels),
        "soft_spearman": soft_spearman_loss(prediction, labels),
        "smooth_l1": F.smooth_l1_loss(prediction, labels),
        "mixture_regularization": mixture_regularization(
            mixture_weights, available
        ),
    }
    components["loss"] = (
        components["hard_pair_logistic"]
        + 0.30 * components["listmle"]
        + 0.20 * components["soft_spearman"]
        + 0.10 * components["smooth_l1"]
        + 0.01 * components["mixture_regularization"]
    )
    return components
