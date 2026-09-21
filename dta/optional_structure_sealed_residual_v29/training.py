"""Training utilities for V29 sealed-V22 bounded residual learning."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.training import (
    _amp_dtype,
    _dataset_metrics,
    _normalizers,
    cache_lineage,
    seed_everything,
)

from .data import DATASET_INDEX, OptionalStructureCollator, move_batch_to_device
from .losses import LossConfig
from .model import OptionalStructureSealedResidualDTA


VERSION = "optional-structure-sealed-residual-v29"


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 20260916
    batch_size: int = 4
    evaluation_batch_size: int = 4
    gradient_accumulation: int = 4
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    maximum_epochs: int = 35
    steps_per_epoch: int = 7000
    patience: int = 10
    warmup_fraction: float = 0.06
    gradient_clip: float = 1.0
    ema_decay: float = 0.995
    pair_batch_fraction: float = 0.70
    protein_input_width: int = 1280
    width: int = 384
    heads: int = 8
    protein_layers: int = 2
    ligand_layers: int = 4
    cross_layers: int = 3
    latent_width: int = 768
    pair_width: int = 192
    structure_self_layers: int = 2
    structure_pair_layers: int = 2
    structure_token_layers: int = 2
    structure_top_pairs: int = 128
    maximum_protein_tokens: int = 512
    maximum_ligand_atoms: int = 128
    maximum_pocket_tokens: int = 256
    adapter_width: int = 192
    dropout: float = 0.10
    residual_width: int = 192
    maximum_residual: float = 0.75
    smooth_weight: float = 1.0
    mse_weight: float = 0.25
    rank_weight: float = 0.20
    backbone_anchor_weight: float = 0.08
    delta_l2_weight: float = 0.01
    gate_center_weight: float = 0.002
    minimum_rank_gap: float = 0.10
    maximum_relative_pairs: int = 64
    minimum_structure_ci_gain: float = 0.005
    maximum_structure_rmse_increase: float = 0.01
    minimum_structure_spearman_gain: float = 0.0
    minimum_residual_ci_gain: float = -0.001
    maximum_residual_rmse_increase: float = 0.003
    minimum_residual_spearman_gain: float = -0.001

    def model_kwargs(self):
        return {
            "protein_input_width": self.protein_input_width,
            "width": self.width,
            "heads": self.heads,
            "protein_layers": self.protein_layers,
            "ligand_layers": self.ligand_layers,
            "cross_layers": self.cross_layers,
            "latent_width": self.latent_width,
            "pair_width": self.pair_width,
            "structure_self_layers": self.structure_self_layers,
            "structure_pair_layers": self.structure_pair_layers,
            "structure_token_layers": self.structure_token_layers,
            "structure_top_pairs": self.structure_top_pairs,
            "maximum_protein_tokens": self.maximum_protein_tokens,
            "dataset_count": len(DATASET_INDEX),
            "adapter_width": self.adapter_width,
            "dropout": self.dropout,
            "residual_width": self.residual_width,
            "maximum_residual": self.maximum_residual,
        }

    def loss_config(self):
        return LossConfig(
            smooth_weight=self.smooth_weight,
            mse_weight=self.mse_weight,
            rank_weight=self.rank_weight,
            backbone_anchor_weight=self.backbone_anchor_weight,
            delta_l2_weight=self.delta_l2_weight,
            gate_center_weight=self.gate_center_weight,
            minimum_rank_gap=self.minimum_rank_gap,
            maximum_relative_pairs=self.maximum_relative_pairs,
        )


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
        if sampler is None:
            raise ValueError("V29 training loader requires an explicit LP sampler")
        return torch.utils.data.DataLoader(batch_sampler=sampler, **common)
    return torch.utils.data.DataLoader(
        batch_size=int(config.evaluation_batch_size),
        shuffle=False,
        drop_last=False,
        **common,
    )


def _scheduler(optimizer, config, loader_length):
    optimizer_steps = int(math.ceil(float(loader_length) / config.gradient_accumulation))
    total_steps = max(1, int(config.maximum_epochs) * optimizer_steps)
    warmup_steps = max(1, int(float(config.warmup_fraction) * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _parameter_groups(model, config):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V29 has no trainable residual parameters")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("V29 sealed V22 backbone unexpectedly became trainable")
    return [{"params": parameters, "lr": config.learning_rate, "name": "bounded_3d_residual"}]


def _selection_score(result):
    metrics = result["metrics"]["LP_PDBBind"]
    return (
        float(metrics["CI"]),
        float(metrics["Spearman"]),
        -float(metrics["RMSE"]),
        -float(metrics["MAE"]),
    )


def predict(model, dataset, config, label_stats):
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    means, standard_deviations = _normalizers(label_stats, device)
    loader = build_loader(dataset, config)
    model.eval()
    sample_ids, labels, indices = [], [], []
    predictions, base_predictions, backbone_predictions = [], [], []
    gates, deltas, backbone_gates, backbone_deltas = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            sample_ids.extend(batch["sample_ids"])
            labels.append(batch["labels"].numpy().astype(np.float64))
            indices.append(batch["dataset_index"].numpy().astype(np.int64))
            moved = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output = model(moved)
            mean = means[moved["dataset_index"]]
            standard_deviation = standard_deviations[moved["dataset_index"]]
            predictions.append(
                (output.prediction * standard_deviation + mean).float().cpu().numpy()
            )
            base_predictions.append(
                (output.base_prediction * standard_deviation + mean).float().cpu().numpy()
            )
            backbone_predictions.append(
                (output.backbone_prediction * standard_deviation + mean).float().cpu().numpy()
            )
            gates.append(output.structure_gate.float().cpu().numpy())
            deltas.append(output.structure_delta.float().cpu().numpy())
            backbone_gates.append(output.backbone_structure_gate.float().cpu().numpy())
            backbone_deltas.append(output.backbone_structure_delta.float().cpu().numpy())
    labels = np.concatenate(labels)
    dataset_index = np.concatenate(indices)
    predictions = np.concatenate(predictions).astype(np.float64)
    base_predictions = np.concatenate(base_predictions).astype(np.float64)
    backbone_predictions = np.concatenate(backbone_predictions).astype(np.float64)
    gates = np.concatenate(gates).astype(np.float32)
    deltas = np.concatenate(deltas).astype(np.float32)
    backbone_gates = np.concatenate(backbone_gates).astype(np.float32)
    backbone_deltas = np.concatenate(backbone_deltas).astype(np.float32)
    arrays = (
        predictions,
        base_predictions,
        backbone_predictions,
        gates,
        deltas,
        backbone_gates,
        backbone_deltas,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("V29 produced non-finite predictions")
    return {
        "sample_ids": tuple(sample_ids),
        "labels": labels,
        "dataset_index": dataset_index,
        "predictions": predictions,
        "base_predictions": base_predictions,
        "backbone_predictions": backbone_predictions,
        "structure_gate": gates,
        "structure_delta": deltas,
        "backbone_structure_gate": backbone_gates,
        "backbone_structure_delta": backbone_deltas,
        "metrics": _dataset_metrics(labels, predictions, dataset_index),
        "base_metrics": _dataset_metrics(labels, base_predictions, dataset_index),
        "backbone_metrics": _dataset_metrics(labels, backbone_predictions, dataset_index),
    }


def save_predictions(path, result):
    np.savez_compressed(
        path,
        sample_ids=np.asarray(result["sample_ids"]),
        labels=result["labels"],
        dataset_index=result["dataset_index"],
        predictions=result["predictions"],
        base_predictions=result["base_predictions"],
        backbone_predictions=result["backbone_predictions"],
        structure_gate=result["structure_gate"],
        structure_delta=result["structure_delta"],
        backbone_structure_gate=result["backbone_structure_gate"],
        backbone_structure_delta=result["backbone_structure_delta"],
    )


def structure_validation_gate(result, config):
    full = result["metrics"]["LP_PDBBind"]
    base = result["base_metrics"]["LP_PDBBind"]
    ci_gain = float(full["CI"] - base["CI"])
    rmse_increase = float(full["RMSE"] - base["RMSE"])
    spearman_gain = float(full["Spearman"] - base["Spearman"])
    passed = (
        ci_gain >= float(config.minimum_structure_ci_gain)
        and rmse_increase <= float(config.maximum_structure_rmse_increase)
        and spearman_gain >= float(config.minimum_structure_spearman_gain)
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "rmse_increase": rmse_increase,
        "spearman_gain": spearman_gain,
        "criteria": {
            "minimum_structure_ci_gain": float(config.minimum_structure_ci_gain),
            "maximum_structure_rmse_increase": float(config.maximum_structure_rmse_increase),
            "minimum_structure_spearman_gain": float(config.minimum_structure_spearman_gain),
        },
    }


def residual_validation_gate(result, config):
    full = result["metrics"]["LP_PDBBind"]
    backbone = result["backbone_metrics"]["LP_PDBBind"]
    ci_gain = float(full["CI"] - backbone["CI"])
    rmse_increase = float(full["RMSE"] - backbone["RMSE"])
    spearman_gain = float(full["Spearman"] - backbone["Spearman"])
    passed = (
        ci_gain >= float(config.minimum_residual_ci_gain)
        and rmse_increase <= float(config.maximum_residual_rmse_increase)
        and spearman_gain >= float(config.minimum_residual_spearman_gain)
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "rmse_increase": rmse_increase,
        "spearman_gain": spearman_gain,
        "criteria": {
            "minimum_residual_ci_gain": float(config.minimum_residual_ci_gain),
            "maximum_residual_rmse_increase": float(config.maximum_residual_rmse_increase),
            "minimum_residual_spearman_gain": float(config.minimum_residual_spearman_gain),
        },
    }


__all__ = [
    "TrainingConfig",
    "VERSION",
    "_parameter_groups",
    "_scheduler",
    "_selection_score",
    "atomic_torch_save",
    "build_loader",
    "cache_lineage",
    "predict",
    "residual_validation_gate",
    "save_predictions",
    "seed_everything",
    "structure_validation_gate",
]
