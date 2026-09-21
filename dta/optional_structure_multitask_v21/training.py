"""Multidataset training, validation-only selection, and evaluation for V21."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.sequence_smiles_binary_probe_v1.core import DEFAULT_THRESHOLDS, binary_metrics

from .data import (
    DATASET_INDEX,
    DatasetBalancedBatchSampler,
    OptionalStructureCollator,
    manifest_fingerprint,
    move_batch_to_device,
    sha256_file,
)
from .losses import LossConfig, optional_structure_loss
from .model import OptionalStructureDTA


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 20260903
    batch_size: int = 4
    evaluation_batch_size: int = 4
    gradient_accumulation: int = 4
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 1.5e-4
    structure_learning_rate_scale: float = 0.70
    weight_decay: float = 1.0e-4
    maximum_epochs: int = 60
    steps_per_epoch: int = 3000
    patience: int = 12
    warmup_fraction: float = 0.06
    gradient_clip: float = 1.0
    ema_decay: float = 0.995
    structure_dropout: float = 0.30
    lp_sampling: float = 0.40
    davis_sampling: float = 0.25
    kiba_sampling: float = 0.35
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
    dropout: float = 0.10
    base_affinity_weight: float = 1.0
    structure_affinity_weight: float = 1.0
    mse_weight: float = 0.15
    base_rank_weight: float = 0.15
    structure_rank_weight: float = 0.10
    contact_weight: float = 0.10
    distance_weight: float = 0.10
    distillation_weight: float = 0.05
    minimum_rank_gap: float = 0.10

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
            "dropout": self.dropout,
        }

    def loss_config(self):
        return LossConfig(
            base_affinity_weight=self.base_affinity_weight,
            structure_affinity_weight=self.structure_affinity_weight,
            mse_weight=self.mse_weight,
            base_rank_weight=self.base_rank_weight,
            structure_rank_weight=self.structure_rank_weight,
            contact_weight=self.contact_weight,
            distance_weight=self.distance_weight,
            distillation_weight=self.distillation_weight,
            minimum_rank_gap=self.minimum_rank_gap,
        )


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def label_statistics(dataset):
    values = {name: [] for name in DATASET_INDEX}
    for row in dataset.rows:
        values[str(row["dataset"])].append(float(row["label"]))
    result = {}
    for name, labels in values.items():
        array = np.asarray(labels, dtype=np.float64)
        standard_deviation = float(array.std())
        if array.size == 0 or not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError("invalid training labels for %s" % name)
        result[name] = {"mean": float(array.mean()), "std": standard_deviation, "count": int(array.size)}
    return result


def _normalizers(label_stats, device):
    means = torch.empty(len(DATASET_INDEX), dtype=torch.float32, device=device)
    standard_deviations = torch.empty_like(means)
    for name, index in DATASET_INDEX.items():
        means[index] = float(label_stats[name]["mean"])
        standard_deviations[index] = float(label_stats[name]["std"])
    return means, standard_deviations


def _amp_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be bfloat16 or float16")


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


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


def cache_lineage(cache_path):
    result = {"path": str(Path(cache_path).expanduser().resolve())}
    for name in ("metadata.json", "lineage.json"):
        candidate = Path(cache_path) / name
        if candidate.is_file():
            result[name + "_sha256"] = sha256_file(candidate)
    return result


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
        batch_size=int(config.evaluation_batch_size), shuffle=False, drop_last=False, **common
    )


def _dataset_metrics(labels, predictions, dataset_index):
    result = {}
    for name, index in DATASET_INDEX.items():
        selected = dataset_index == index
        if not selected.any():
            continue
        metrics = affinity_metrics(labels[selected], predictions[selected])
        entry = {key: float(value) for key, value in metrics.items()}
        if name in DEFAULT_THRESHOLDS:
            threshold = float(DEFAULT_THRESHOLDS[name])
            binary_label = (labels[selected] >= threshold).astype(np.int8)
            if np.unique(binary_label).size == 2:
                entry["binary"] = binary_metrics(
                    binary_label, predictions[selected], threshold
                )
                entry["binary"]["affinity_threshold"] = threshold
        result[name] = entry
    return result


def predict(model, dataset, config, label_stats):
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    means, standard_deviations = _normalizers(label_stats, device)
    loader = build_loader(dataset, config)
    model.eval()
    sample_ids = []
    labels = []
    predictions = []
    base_predictions = []
    dataset_indices = []
    gates = []
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
        raise RuntimeError("V21 produced non-finite validation/test predictions")
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
    lp = result["metrics"]["LP_PDBBind"]
    macro_ci = float(np.mean([metrics["CI"] for metrics in result["metrics"].values()]))
    return (float(lp["CI"]), -float(lp["RMSE"]), macro_ci)


def fit(train_dataset, valid_dataset, output, config, protein_cache, ligand_cache, resume=False):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V21 output exists; pass --resume only for the same interrupted run")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    label_stats = label_statistics(train_dataset)
    means, standard_deviations = _normalizers(label_stats, device)
    model = OptionalStructureDTA(**config.model_kwargs()).to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    structure_parameters = list(model.structure_encoder.parameters())
    structure_parameters += list(model.structure_gate.parameters())
    structure_parameters += list(model.delta_heads.parameters())
    structure_ids = {id(parameter) for parameter in structure_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in structure_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": config.learning_rate},
            {
                "params": structure_parameters,
                "lr": config.learning_rate * config.structure_learning_rate_scale,
            },
        ],
        weight_decay=config.weight_decay,
    )
    sampler = DatasetBalancedBatchSampler(
        train_dataset,
        config.batch_size,
        config.steps_per_epoch,
        {
            "LP_PDBBind": config.lp_sampling,
            "DAVIS": config.davis_sampling,
            "KIBA": config.kiba_sampling,
        },
        config.seed,
    )
    train_loader = build_loader(train_dataset, config, training=True, sampler=sampler)
    optimizer_steps_per_epoch = int(math.ceil(len(train_loader) / config.gradient_accumulation))
    total_steps = max(1, config.maximum_epochs * optimizer_steps_per_epoch)
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_dtype = _amp_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    history = []
    best = None
    stale = 0
    start_epoch = 0
    last_path = output / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume requires %s" % last_path)
        state = torch.load(str(last_path), map_location="cpu")
        if state["config"] != asdict(config):
            raise ValueError("resume config differs from the interrupted V21 run")
        model.load_state_dict(state["model_state"], strict=True)
        ema_model.load_state_dict(state["ema_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best = state["best"]
        stale = int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
    loss_config = config.loss_config()
    selection_valid = torch.utils.data.Subset(
        valid_dataset, valid_dataset.dataset_to_indices["LP_PDBBind"]
    )
    for epoch in range(start_epoch, config.maximum_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {}
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            moved = move_batch_to_device(batch, device)
            if config.structure_dropout > 0 and moved["has_3d"].any():
                keep = torch.rand_like(moved["has_3d"], dtype=torch.float32) >= config.structure_dropout
                moved["has_3d"] = moved["has_3d"] & keep
            normalized = (
                moved["labels"] - means[moved["dataset_index"]]
            ) / standard_deviations[moved["dataset_index"]]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = optional_structure_loss(
                    output_value, normalized, moved, loss_config
                )
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V21 training loss")
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % config.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V21 gradient")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, config.ema_decay)
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1
        raw = predict(model, selection_valid, config, label_stats)
        ema = predict(ema_model, selection_valid, config, label_stats)
        variant, selected = max((("raw", raw), ("ema", ema)), key=lambda item: _selection_score(item[1]))
        score = _selection_score(selected)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "structure_learning_rate": optimizer.param_groups[1]["lr"],
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["metrics"],
            "ema": ema["metrics"],
            "selected_variant": variant,
        }
        history.append(row)
        print(json.dumps({"event": "v21_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "version": "optional-structure-multitask-v21",
                "config": asdict(config),
                "model_kwargs": config.model_kwargs(),
                "model_state": {key: value.detach().cpu() for key, value in chosen.state_dict().items()},
                "label_statistics": label_stats,
                "epoch": epoch,
                "variant": variant,
                "validation_metrics": selected["metrics"],
                "validation_base_metrics": selected["base_metrics"],
                "selection_rule": "maximize LP-PDBBind validation CI; tie-break RMSE then macro CI",
                "train_manifest_sha256": manifest_fingerprint(train_dataset.manifest_path),
                "valid_manifest_sha256": manifest_fingerprint(valid_dataset.manifest_path),
                "protein_cache": cache_lineage(protein_cache),
                "ligand_cache": cache_lineage(ligand_cache),
            }
            atomic_torch_save(checkpoint, output / "best.pt")
            save_predictions(output / "valid_predictions.npz", selected)
            best = {"score": list(score), "epoch": epoch, "variant": variant}
            stale = 0
        else:
            stale += 1
        last_state = {
            "config": asdict(config),
            "epoch": epoch,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "ema_state": {key: value.detach().cpu() for key, value in ema_model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "best": best,
            "stale": stale,
        }
        atomic_torch_save(last_state, last_path)
        (output / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if stale >= config.patience:
            break
    if not (output / "best.pt").is_file():
        raise RuntimeError("V21 training did not produce a best checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    selected_model = OptionalStructureDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    valid_result = predict(selected_model, valid_dataset, config, checkpoint["label_statistics"])
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


__all__ = [
    "TrainingConfig",
    "cache_lineage",
    "fit",
    "predict",
    "save_predictions",
]
