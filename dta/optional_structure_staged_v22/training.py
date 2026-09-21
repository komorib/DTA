"""Two-stage training, validation selection, and prediction utilities for V22."""

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
from .losses import (
    SequenceLossConfig,
    StructureLossConfig,
    sequence_only_loss,
    structure_residual_loss,
)
from .model import OptionalStructureStagedDTA


VERSION = "optional-structure-staged-v22"
STRUCTURE_PREFIXES = ("structure_encoder.", "structure_gate.", "delta_heads.")


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 20260905
    batch_size: int = 4
    evaluation_batch_size: int = 4
    gradient_accumulation: int = 4
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 1.5e-4
    lp_adapter_learning_rate_scale: float = 0.05
    weight_decay: float = 1.0e-4
    maximum_epochs: int = 30
    steps_per_epoch: int = 7000
    patience: int = 8
    warmup_fraction: float = 0.06
    gradient_clip: float = 1.0
    ema_decay: float = 0.995
    lp_sampling: float = 0.15
    davis_sampling: float = 0.25
    kiba_sampling: float = 0.60
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
    sequence_smooth_weight: float = 1.0
    sequence_mse_weight: float = 0.50
    sequence_rank_weight: float = 0.25
    sequence_binary_weight: float = 0.20
    binary_temperature: float = 0.50
    structure_smooth_weight: float = 1.0
    structure_mse_weight: float = 0.30
    structure_rank_weight: float = 0.20
    contact_weight: float = 0.10
    distance_weight: float = 0.10
    delta_l2_weight: float = 0.01
    gate_regularization_weight: float = 0.005
    minimum_rank_gap: float = 0.10
    minimum_structure_ci_gain: float = 0.005
    maximum_structure_rmse_increase: float = 0.01
    minimum_structure_spearman_gain: float = 0.0

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
        }

    def sequence_loss_config(self):
        return SequenceLossConfig(
            smooth_weight=self.sequence_smooth_weight,
            mse_weight=self.sequence_mse_weight,
            rank_weight=self.sequence_rank_weight,
            binary_weight=self.sequence_binary_weight,
            binary_temperature=self.binary_temperature,
            minimum_rank_gap=self.minimum_rank_gap,
        )

    def structure_loss_config(self):
        return StructureLossConfig(
            smooth_weight=self.structure_smooth_weight,
            mse_weight=self.structure_mse_weight,
            rank_weight=self.structure_rank_weight,
            contact_weight=self.contact_weight,
            distance_weight=self.distance_weight,
            delta_l2_weight=self.delta_l2_weight,
            gate_regularization_weight=self.gate_regularization_weight,
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
        result[name] = {
            "mean": float(array.mean()),
            "std": standard_deviation,
            "count": int(array.size),
        }
    return result


def _normalizers(label_stats, device):
    means = torch.empty(len(DATASET_INDEX), dtype=torch.float32, device=device)
    standard_deviations = torch.empty_like(means)
    for name, index in DATASET_INDEX.items():
        means[index] = float(label_stats[name]["mean"])
        standard_deviations[index] = float(label_stats[name]["std"])
    return means, standard_deviations


def _binary_training_statistics(dataset, label_stats, device):
    thresholds = torch.full((len(DATASET_INDEX),), float("nan"), device=device)
    positive_weights = torch.ones(len(DATASET_INDEX), device=device)
    for name, index in DATASET_INDEX.items():
        if name not in DEFAULT_THRESHOLDS:
            continue
        cutoff = float(DEFAULT_THRESHOLDS[name])
        thresholds[index] = (
            cutoff - float(label_stats[name]["mean"])
        ) / float(label_stats[name]["std"])
        labels = np.asarray(
            [float(row["label"]) for row in dataset.rows if row["dataset"] == name],
            dtype=np.float64,
        )
        positive = int((labels >= cutoff).sum())
        negative = int(labels.size - positive)
        if positive == 0 or negative == 0:
            raise ValueError("binary auxiliary target needs both classes for %s" % name)
        positive_weights[index] = min(50.0, max(1.0, negative / positive))
    return thresholds, positive_weights


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


class LPBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, steps_per_epoch, seed):
        self.indices = np.asarray(dataset.dataset_to_indices["LP_PDBBind"], dtype=np.int64)
        if self.indices.size == 0:
            raise ValueError("V22 structure phase requires LP-PDBBind training rows")
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            yield generator.choice(self.indices, size=self.batch_size, replace=True).tolist()


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
                entry["binary"] = binary_metrics(binary_label, predictions[selected], threshold)
                entry["binary"]["affinity_threshold"] = threshold
        result[name] = entry
    return result


def predict(model, dataset, config, label_stats):
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    means, standard_deviations = _normalizers(label_stats, device)
    loader = build_loader(dataset, config)
    model.eval()
    sample_ids, labels, predictions, base_predictions, dataset_indices, gates = [], [], [], [], [], []
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
        raise RuntimeError("V22 produced non-finite validation/test predictions")
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


def _sequence_score(result):
    metrics = list(result["base_metrics"].values())
    return (
        float(np.mean([entry["CI"] for entry in metrics])),
        float(np.mean([entry["Spearman"] for entry in metrics])),
        float(np.mean([entry["Pearson"] for entry in metrics])),
        -float(result["base_metrics"]["LP_PDBBind"]["RMSE"]),
    )


def _structure_score(result):
    metrics = result["metrics"]["LP_PDBBind"]
    return (float(metrics["CI"]), -float(metrics["RMSE"]), float(metrics["Spearman"]))


def _scheduler(optimizer, config, loader_length):
    optimizer_steps_per_epoch = int(math.ceil(loader_length / config.gradient_accumulation))
    total_steps = max(1, config.maximum_epochs * optimizer_steps_per_epoch)
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _compatible_state(model, checkpoint_path, prefixes=None, excluded_prefixes=()):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    target = model.state_dict()
    loaded = {}
    for key, value in source.items():
        if prefixes is not None and not key.startswith(tuple(prefixes)):
            continue
        if excluded_prefixes and key.startswith(tuple(excluded_prefixes)):
            continue
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            loaded[key] = value
    if not loaded:
        raise ValueError("no shape-compatible parameters found in %s" % checkpoint_path)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if unexpected:
        raise ValueError("unexpected warm-start parameters: %s" % unexpected)
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "loaded_parameter_groups": len(loaded),
        "missing_parameter_groups": len(missing),
    }


def configure_structure_trainable_parameters(model):
    """Freeze the sealed base except for the LP-specific low-LR correction path."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    structure_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(STRUCTURE_PREFIXES):
            parameter.requires_grad_(True)
            structure_parameters.append(parameter)
    lp_index = DATASET_INDEX["LP_PDBBind"]
    lp_parameters = list(model.task_adapters[lp_index].parameters())
    lp_parameters += list(model.base_heads[lp_index].parameters())
    for parameter in lp_parameters:
        parameter.requires_grad_(True)
    return structure_parameters, lp_parameters


def _save_last(path, config, epoch, model, ema_model, optimizer, scheduler, scaler, history, best, stale, sources):
    atomic_torch_save(
        {
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
            "sources": sources,
        },
        path,
    )


def fit_sequence(train_dataset, valid_dataset, output, config, protein_cache, ligand_cache, initial_checkpoint=None, resume=False):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V22 sequence output exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    label_stats = label_statistics(train_dataset)
    means, standard_deviations = _normalizers(label_stats, device)
    thresholds, positive_weights = _binary_training_statistics(train_dataset, label_stats, device)
    model = OptionalStructureStagedDTA(**config.model_kwargs()).to(device)
    sources = {}
    if initial_checkpoint is not None:
        sources["initial_base"] = _compatible_state(
            model,
            initial_checkpoint,
            excluded_prefixes=STRUCTURE_PREFIXES,
        )
    for name, parameter in model.named_parameters():
        if name.startswith(STRUCTURE_PREFIXES):
            parameter.requires_grad_(False)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
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
    loader = build_loader(train_dataset, config, training=True, sampler=sampler)
    scheduler = _scheduler(optimizer, config, len(loader))
    amp_dtype = _amp_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    history, best, stale, start_epoch = [], None, 0, 0
    last_path = output / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume requires %s" % last_path)
        state = torch.load(str(last_path), map_location="cpu")
        if state["config"] != asdict(config) or state["sources"] != sources:
            raise ValueError("V22 sequence resume lineage/config differs")
        model.load_state_dict(state["model_state"], strict=True)
        ema_model.load_state_dict(state["ema_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history, best, stale = state["history"], state["best"], int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
    loss_config = config.sequence_loss_config()
    for epoch in range(start_epoch, config.maximum_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums, batches = {}, 0
        for batch_index, batch in enumerate(loader):
            moved = move_batch_to_device(batch, device)
            normalized = (
                moved["labels"] - means[moved["dataset_index"]]
            ) / standard_deviations[moved["dataset_index"]]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved, force_no_3d=True)
                loss, components = sequence_only_loss(
                    output_value,
                    normalized,
                    moved,
                    thresholds,
                    positive_weights,
                    loss_config,
                )
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V22 sequence loss")
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % config.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.gradient_clip,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V22 sequence gradient")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, config.ema_decay)
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1
        raw, ema = predict(model, valid_dataset, config, label_stats), predict(ema_model, valid_dataset, config, label_stats)
        variant, selected = max((("raw", raw), ("ema", ema)), key=lambda item: _sequence_score(item[1]))
        score = _sequence_score(selected)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["base_metrics"],
            "ema": ema["base_metrics"],
            "selected_variant": variant,
            "selection_score": list(score),
        }
        history.append(row)
        print(json.dumps({"event": "v22_sequence_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "version": VERSION,
                "phase": "sequence",
                "config": asdict(config),
                "model_kwargs": config.model_kwargs(),
                "model_state": {key: value.detach().cpu() for key, value in chosen.state_dict().items()},
                "label_statistics": label_stats,
                "binary_training_statistics": {
                    "normalized_thresholds": thresholds.detach().cpu().tolist(),
                    "positive_weights": positive_weights.detach().cpu().tolist(),
                },
                "epoch": epoch,
                "variant": variant,
                "validation_metrics": selected["base_metrics"],
                "selection_rule": "maximize validation macro CI, then macro Spearman/Pearson, then LP RMSE",
                "train_manifest_sha256": manifest_fingerprint(train_dataset.manifest_path),
                "valid_manifest_sha256": manifest_fingerprint(valid_dataset.manifest_path),
                "protein_cache": cache_lineage(protein_cache),
                "ligand_cache": cache_lineage(ligand_cache),
                "sources": sources,
            }
            atomic_torch_save(checkpoint, output / "best.pt")
            save_predictions(output / "valid_predictions.npz", selected)
            best, stale = {"score": list(score), "epoch": epoch, "variant": variant}, 0
        else:
            stale += 1
        _save_last(last_path, config, epoch, model, ema_model, optimizer, scheduler, scaler, history, best, stale, sources)
        (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if stale >= config.patience:
            break
    if not (output / "best.pt").is_file():
        raise RuntimeError("V22 sequence phase did not produce a checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    selected_model = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    valid_result = predict(selected_model, valid_dataset, config, label_stats)
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


def fit_structure(train_dataset, valid_dataset, output, config, protein_cache, ligand_cache, sequence_checkpoint, warm_structure_checkpoint=None, resume=False):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V22 structure output exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    sequence_checkpoint = Path(sequence_checkpoint).expanduser().resolve()
    sequence_state = torch.load(str(sequence_checkpoint), map_location="cpu")
    if sequence_state.get("version") != VERSION or sequence_state.get("phase") != "sequence":
        raise ValueError("V22 structure phase requires a sealed V22 sequence checkpoint")
    if sequence_state["model_kwargs"] != config.model_kwargs():
        raise ValueError("V22 structure model config differs from the sealed sequence stage")
    if sequence_state.get("train_manifest_sha256") != manifest_fingerprint(train_dataset.manifest_path):
        raise ValueError("V22 sequence checkpoint training manifest lineage mismatch")
    if sequence_state.get("valid_manifest_sha256") != manifest_fingerprint(valid_dataset.manifest_path):
        raise ValueError("V22 sequence checkpoint validation manifest lineage mismatch")
    if sequence_state.get("protein_cache") != cache_lineage(protein_cache):
        raise ValueError("V22 sequence checkpoint protein cache lineage mismatch")
    if sequence_state.get("ligand_cache") != cache_lineage(ligand_cache):
        raise ValueError("V22 sequence checkpoint ligand cache lineage mismatch")
    model = OptionalStructureStagedDTA(**sequence_state["model_kwargs"]).to(device)
    model.load_state_dict(sequence_state["model_state"], strict=True)
    sources = {
        "sequence_checkpoint": {
            "path": str(sequence_checkpoint),
            "sha256": sha256_file(sequence_checkpoint),
        }
    }
    if warm_structure_checkpoint is not None:
        sources["warm_structure"] = _compatible_state(
            model, warm_structure_checkpoint, prefixes=("structure_encoder.",)
        )
    structure_parameters, lp_parameters = configure_structure_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": structure_parameters, "lr": config.learning_rate},
            {"params": lp_parameters, "lr": config.learning_rate * config.lp_adapter_learning_rate_scale},
        ],
        weight_decay=config.weight_decay,
    )
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    sampler = LPBatchSampler(train_dataset, config.batch_size, config.steps_per_epoch, config.seed)
    loader = build_loader(train_dataset, config, training=True, sampler=sampler)
    scheduler = _scheduler(optimizer, config, len(loader))
    amp_dtype = _amp_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    label_stats = sequence_state["label_statistics"]
    means, standard_deviations = _normalizers(label_stats, device)
    history, best, stale, start_epoch = [], None, 0, 0
    last_path = output / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume requires %s" % last_path)
        state = torch.load(str(last_path), map_location="cpu")
        if state["config"] != asdict(config) or state["sources"] != sources:
            raise ValueError("V22 structure resume lineage/config differs")
        model.load_state_dict(state["model_state"], strict=True)
        ema_model.load_state_dict(state["ema_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history, best, stale = state["history"], state["best"], int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
    loss_config = config.structure_loss_config()
    selection_valid = torch.utils.data.Subset(valid_dataset, valid_dataset.dataset_to_indices["LP_PDBBind"])
    for epoch in range(start_epoch, config.maximum_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums, batches = {}, 0
        for batch_index, batch in enumerate(loader):
            moved = move_batch_to_device(batch, device)
            normalized = (
                moved["labels"] - means[moved["dataset_index"]]
            ) / standard_deviations[moved["dataset_index"]]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = structure_residual_loss(output_value, normalized, moved, loss_config)
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V22 structure loss")
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % config.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.gradient_clip,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V22 structure gradient")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, config.ema_decay)
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1
        raw, ema = predict(model, selection_valid, config, label_stats), predict(ema_model, selection_valid, config, label_stats)
        variant, selected = max((("raw", raw), ("ema", ema)), key=lambda item: _structure_score(item[1]))
        score = _structure_score(selected)
        row = {
            "epoch": epoch,
            "structure_learning_rate": optimizer.param_groups[0]["lr"],
            "lp_adapter_learning_rate": optimizer.param_groups[1]["lr"],
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["metrics"],
            "ema": ema["metrics"],
            "selected_variant": variant,
        }
        history.append(row)
        print(json.dumps({"event": "v22_structure_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "version": VERSION,
                "phase": "structure",
                "config": asdict(config),
                "model_kwargs": sequence_state["model_kwargs"],
                "model_state": {key: value.detach().cpu() for key, value in chosen.state_dict().items()},
                "label_statistics": label_stats,
                "epoch": epoch,
                "variant": variant,
                "validation_metrics": selected["metrics"],
                "validation_base_metrics": selected["base_metrics"],
                "selection_rule": "maximize LP-PDBBind validation CI; tie-break RMSE then Spearman",
                "train_manifest_sha256": manifest_fingerprint(train_dataset.manifest_path),
                "valid_manifest_sha256": manifest_fingerprint(valid_dataset.manifest_path),
                "protein_cache": cache_lineage(protein_cache),
                "ligand_cache": cache_lineage(ligand_cache),
                "sources": sources,
            }
            atomic_torch_save(checkpoint, output / "best.pt")
            save_predictions(output / "valid_predictions.npz", selected)
            best, stale = {"score": list(score), "epoch": epoch, "variant": variant}, 0
        else:
            stale += 1
        _save_last(last_path, config, epoch, model, ema_model, optimizer, scheduler, scaler, history, best, stale, sources)
        (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if stale >= config.patience:
            break
    if not (output / "best.pt").is_file():
        raise RuntimeError("V22 structure phase did not produce a checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    selected_model = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    valid_result = predict(selected_model, valid_dataset, config, label_stats)
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


def structure_validation_gate(result, config):
    full = result["metrics"]["LP_PDBBind"]
    base = result["base_metrics"]["LP_PDBBind"]
    gain = float(full["CI"] - base["CI"])
    rmse_increase = float(full["RMSE"] - base["RMSE"])
    spearman_gain = float(full["Spearman"] - base["Spearman"])
    passed = (
        gain >= config.minimum_structure_ci_gain
        and rmse_increase <= config.maximum_structure_rmse_increase
        and spearman_gain >= config.minimum_structure_spearman_gain
    )
    return {
        "passed": bool(passed),
        "ci_gain": gain,
        "rmse_increase": rmse_increase,
        "spearman_gain": spearman_gain,
        "criteria": {
            "minimum_structure_ci_gain": config.minimum_structure_ci_gain,
            "maximum_structure_rmse_increase": config.maximum_structure_rmse_increase,
            "minimum_spearman_gain": config.minimum_structure_spearman_gain,
        },
    }


__all__ = [
    "TrainingConfig",
    "cache_lineage",
    "configure_structure_trainable_parameters",
    "fit_sequence",
    "fit_structure",
    "label_statistics",
    "predict",
    "save_predictions",
    "structure_validation_gate",
]
