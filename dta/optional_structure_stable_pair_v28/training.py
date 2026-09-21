"""Training and prediction utilities for V28 stable pair co-training."""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.training import (
    _amp_dtype,
    _binary_training_statistics,
    _compatible_state,
    _dataset_metrics,
    _normalizers,
    cache_lineage,
    label_statistics,
    seed_everything,
    update_ema,
)

from .data import (
    DATASET_INDEX,
    OptionalStructureCollator,
    TargetAwareBalancedBatchSampler,
    manifest_fingerprint,
    move_batch_to_device,
    sha256_file,
)
from .losses import LossConfig, stable_pair_loss
from .model import OptionalStructureStablePairDTA


VERSION = "optional-structure-stable-pair-v28"
NEW_PREFIXES = (
    "sequence_pair_encoder.",
    "base_pair_adapter.",
    "geometry_quality_head.",
    "relative_heads.",
    "pair_cross.",
    "pair_cross_pool.",
    "geometry_adapter.",
    "structure_encoder.visibility_embedding.",
)
STRUCTURE_PREFIXES = ("structure_encoder.", "structure_gate.", "geometry_adapter.", "pair_cross.")


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 20260914
    batch_size: int = 4
    evaluation_batch_size: int = 4
    gradient_accumulation: int = 4
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 1.0e-4
    pretrained_learning_rate_scale: float = 0.05
    structure_learning_rate_scale: float = 0.35
    weight_decay: float = 1.0e-4
    maximum_epochs: int = 45
    steps_per_epoch: int = 7000
    patience: int = 10
    warmup_fraction: float = 0.06
    gradient_clip: float = 1.0
    ema_decay: float = 0.995
    lp_sampling: float = 0.30
    davis_sampling: float = 0.25
    kiba_sampling: float = 0.45
    pair_batch_fraction: float = 0.70
    geometry_mask_fraction: float = 0.25
    structure_dropout: float = 0.15
    pretrained_freeze_epochs: int = 2
    pair_distillation_warmup_epochs: int = 5
    maximum_pair_distillation: float = 4.0
    protein_input_width: int = 1280
    width: int = 384
    heads: int = 8
    protein_layers: int = 2
    ligand_layers: int = 4
    cross_layers: int = 3
    latent_width: int = 768
    pair_width: int = 192
    pair_factor: int = 16
    sequence_pair_layers: int = 2
    sequence_top_pairs: int = 128
    structure_self_layers: int = 2
    structure_pair_layers: int = 2
    structure_token_layers: int = 2
    structure_top_pairs: int = 128
    maximum_protein_tokens: int = 512
    maximum_ligand_atoms: int = 128
    maximum_pocket_tokens: int = 256
    adapter_width: int = 192
    dropout: float = 0.10
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
    minimum_structure_ci_gain: float = 0.005
    maximum_structure_rmse_increase: float = 0.01
    minimum_structure_spearman_gain: float = 0.0

    def model_kwargs(self):
        names = (
            "protein_input_width",
            "width",
            "heads",
            "protein_layers",
            "ligand_layers",
            "cross_layers",
            "latent_width",
            "pair_width",
            "pair_factor",
            "sequence_pair_layers",
            "sequence_top_pairs",
            "structure_self_layers",
            "structure_pair_layers",
            "structure_token_layers",
            "structure_top_pairs",
            "maximum_protein_tokens",
            "adapter_width",
            "dropout",
        )
        result = {name: getattr(self, name) for name in names}
        result["dataset_count"] = len(DATASET_INDEX)
        return result

    def loss_config(self):
        names = tuple(LossConfig.__dataclass_fields__)
        return LossConfig(**{name: getattr(self, name) for name in names})


def atomic_torch_save(value, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_loader(dataset, config, training=False, sampler=None):
    common = {
        "dataset": dataset,
        "collate_fn": OptionalStructureCollator(
            config.maximum_ligand_atoms, config.maximum_pocket_tokens
        ),
        "num_workers": int(config.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(config.num_workers) > 0,
    }
    if int(config.num_workers) > 0:
        common["prefetch_factor"] = 2
    if training:
        return torch.utils.data.DataLoader(batch_sampler=sampler, **common)
    return torch.utils.data.DataLoader(
        batch_size=int(config.evaluation_batch_size),
        shuffle=False,
        drop_last=False,
        **common,
    )


def predict(model, dataset, config, label_stats):
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    means, standard_deviations = _normalizers(label_stats, device)
    loader = build_loader(dataset, config)
    model.eval()
    sample_ids, labels, predictions, base_predictions = [], [], [], []
    dataset_indices, gates = [], []
    with torch.no_grad():
        for batch in loader:
            sample_ids.extend(batch["sample_ids"])
            labels.append(batch["labels"].numpy().astype(np.float64))
            dataset_indices.append(batch["dataset_index"].numpy().astype(np.int64))
            moved = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output = model(moved)
            mean = means[moved["dataset_index"]]
            std = standard_deviations[moved["dataset_index"]]
            predictions.append((output.prediction * std + mean).float().cpu().numpy())
            base_predictions.append((output.base_prediction * std + mean).float().cpu().numpy())
            gates.append(output.structure_gate.float().cpu().numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions).astype(np.float64)
    base_predictions = np.concatenate(base_predictions).astype(np.float64)
    dataset_indices = np.concatenate(dataset_indices)
    gates = np.concatenate(gates).astype(np.float32)
    if not np.isfinite(predictions).all() or not np.isfinite(base_predictions).all():
        raise RuntimeError("V28 produced non-finite predictions")
    return {
        "sample_ids": tuple(sample_ids),
        "labels": labels,
        "predictions": predictions,
        "base_predictions": base_predictions,
        "dataset_index": dataset_indices,
        "structure_gate": gates,
        "metrics": _dataset_metrics(labels, predictions, dataset_indices),
        "base_metrics": _dataset_metrics(labels, base_predictions, dataset_indices),
    }


def save_predictions(path, result):
    np.savez_compressed(
        path,
        sample_ids=np.asarray(result["sample_ids"]),
        labels=result["labels"],
        predictions=result["predictions"],
        base_predictions=result["base_predictions"],
        dataset_index=result["dataset_index"],
        structure_gate=result["structure_gate"],
    )


def _selection_score(result):
    metrics = result["metrics"]
    weighted_ci = (
        0.50 * float(metrics["LP_PDBBind"]["CI"])
        + 0.25 * float(metrics["DAVIS"]["CI"])
        + 0.25 * float(metrics["KIBA"]["CI"])
    )
    macro_spearman = float(np.mean([entry["Spearman"] for entry in metrics.values()]))
    return (
        weighted_ci,
        float(metrics["LP_PDBBind"]["CI"]),
        macro_spearman,
        -float(metrics["LP_PDBBind"]["RMSE"]),
    )


def _scheduler(optimizer, config, loader_length):
    optimizer_steps = int(math.ceil(loader_length / config.gradient_accumulation))
    total_steps = max(1, config.maximum_epochs * optimizer_steps)
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _parameter_groups(model, config):
    new_parameters, structure_parameters, pretrained_parameters = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(NEW_PREFIXES):
            new_parameters.append(parameter)
        elif name.startswith(STRUCTURE_PREFIXES):
            structure_parameters.append(parameter)
        else:
            pretrained_parameters.append(parameter)
    identifiers = [id(value) for group in (new_parameters, structure_parameters, pretrained_parameters) for value in group]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != len(list(model.parameters())):
        raise RuntimeError("V28 optimizer parameter groups are not a disjoint partition")
    return [
        {"params": new_parameters, "lr": config.learning_rate, "name": "new_pair"},
        {
            "params": structure_parameters,
            "lr": config.learning_rate * config.structure_learning_rate_scale,
            "name": "structure",
        },
        {
            "params": pretrained_parameters,
            "lr": config.learning_rate * config.pretrained_learning_rate_scale,
            "name": "pretrained_shared",
        },
    ]


def _set_pretrained_trainable(model, trainable):
    count = 0
    for name, parameter in model.named_parameters():
        is_pretrained = not name.startswith(NEW_PREFIXES) and not name.startswith(STRUCTURE_PREFIXES)
        if is_pretrained:
            parameter.requires_grad_(bool(trainable))
            count += int(parameter.numel())
    if count == 0:
        raise RuntimeError("V28 did not identify pretrained parameters")
    return count


def pair_distillation_scale(epoch, config):
    freeze_epochs = int(config.pretrained_freeze_epochs)
    warmup_epochs = int(config.pair_distillation_warmup_epochs)
    if freeze_epochs < 0 or warmup_epochs < 0:
        raise ValueError("V28 freeze/warm-up epochs must be non-negative")
    if int(epoch) < freeze_epochs:
        return 0.0
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, float(int(epoch) - freeze_epochs + 1) / float(warmup_epochs))


def apply_training_masks(batch, config):
    has_3d = batch["has_3d"].bool()
    if has_3d.any() and config.structure_dropout > 0:
        keep = torch.rand_like(has_3d, dtype=torch.float32) >= float(config.structure_dropout)
        effective = has_3d & keep
        if not effective.any():
            effective = effective.clone()
            effective[torch.nonzero(has_3d, as_tuple=False)[0, 0]] = True
        batch["has_3d"] = effective
    if batch["has_3d"].any():
        pair_mask = (
            batch["pocket_mask"][:, :, None]
            & batch["structure_ligand_mask"][:, None, :]
            & batch["has_3d"][:, None, None]
        )
        masked = (
            torch.rand(pair_mask.shape, device=pair_mask.device) < float(config.geometry_mask_fraction)
        ) & pair_mask
        if pair_mask.any() and not masked.any():
            first = torch.nonzero(pair_mask, as_tuple=False)[0]
            masked[tuple(first.tolist())] = True
        batch["geometry_masked"] = masked
        batch["geometry_visible"] = pair_mask & ~masked
    return batch


def structure_validation_gate(result, config):
    full = result["metrics"]["LP_PDBBind"]
    base = result["base_metrics"]["LP_PDBBind"]
    ci_gain = float(full["CI"] - base["CI"])
    rmse_increase = float(full["RMSE"] - base["RMSE"])
    spearman_gain = float(full["Spearman"] - base["Spearman"])
    passed = (
        ci_gain >= config.minimum_structure_ci_gain
        and rmse_increase <= config.maximum_structure_rmse_increase
        and spearman_gain >= config.minimum_structure_spearman_gain
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "rmse_increase": rmse_increase,
        "spearman_gain": spearman_gain,
        "criteria": {
            "minimum_structure_ci_gain": config.minimum_structure_ci_gain,
            "maximum_structure_rmse_increase": config.maximum_structure_rmse_increase,
            "minimum_structure_spearman_gain": config.minimum_structure_spearman_gain,
        },
    }
