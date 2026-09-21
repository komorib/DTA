"""Training and evaluation for the pretrained V12 MSA-to-pair expert."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from data_utils.hybrid_affinity_dataset import LabelSpreadBatchSampler
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from .dataset import MSAPairCollator, move_batch_to_device
from .model import MSAPairExpertV12, residue_distogram_targets


@dataclass(frozen=True)
class TrainingConfig:
    pretrained_checkpoint: str
    device: str = "cuda:0"
    seed: int = 42
    batch_size: int = 4
    evaluation_batch_size: int = 6
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 1.5e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 80
    patience: int = 14
    gradient_clip: float = 1.0
    model_width: int = 384
    layers: int = 6
    heads: int = 8
    dropout: float = 0.10
    latent_width: int = 1024
    msa_width: int = 128
    msa_layers: int = 4
    msa_heads: int = 4
    pair_factor: int = 8
    pair_width: int = 64
    row_depth: int = 64
    inference_subsets: int = 2
    rank_weight: float = 0.40
    mse_weight: float = 0.15
    contact_weight: float = 0.05
    residue_contact_weight: float = 0.03
    affinity_class_weight: float = 0.05
    distogram_weight: float = 0.05
    encoder_learning_rate_scale: float = 0.30
    ema_decay: float = 0.995
    minimum_rank_gap: float = 0.10

    def model_kwargs(self):
        return {
            "model_width": self.model_width,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "latent_width": self.latent_width,
            "msa_width": self.msa_width,
            "msa_layers": self.msa_layers,
            "msa_heads": self.msa_heads,
            "pair_factor": self.pair_factor,
            "pair_width": self.pair_width,
        }


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _autocast_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be bfloat16 or float16")


def _pairwise_rank_loss(prediction, labels, minimum_gap):
    label_difference = labels[:, None] - labels[None, :]
    prediction_difference = prediction[:, None] - prediction[None, :]
    upper = torch.triu(torch.ones_like(label_difference, dtype=torch.bool), diagonal=1)
    valid = upper & (torch.abs(label_difference) >= float(minimum_gap))
    if not valid.any():
        return prediction.sum() * 0.0
    direction = torch.sign(label_difference[valid])
    return F.softplus(-direction * prediction_difference[valid]).mean()


def _balanced_binary_loss(logits, target):
    target = target.to(logits.dtype)
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)


def msa_pair_loss(output, normalized_labels, batch, config):
    smooth = F.smooth_l1_loss(output.prediction, normalized_labels, beta=0.5)
    mse = F.mse_loss(output.prediction, normalized_labels)
    rank = _pairwise_rank_loss(output.prediction, normalized_labels, config.minimum_rank_gap)

    contact_mask = batch["contact_mask"]
    contact = _balanced_binary_loss(
        output.contact_logits[contact_mask], batch["contact_label"][contact_mask]
    )

    chain = batch["pocket_chain"]
    pocket_mask = batch["pocket_mask"]
    position = batch["pocket_position"]
    same_chain = (chain[:, :, None] == chain[:, None, :]) & (chain[:, :, None] >= 0)
    separated = torch.abs(position[:, :, None] - position[:, None, :]) >= 4
    upper = torch.triu(
        torch.ones_like(same_chain, dtype=torch.bool), diagonal=1
    )
    residue_mask = (
        same_chain
        & separated
        & upper
        & pocket_mask[:, :, None]
        & pocket_mask[:, None, :]
    )
    if residue_mask.any():
        residue_distance = torch.cdist(batch["pocket_coords"], batch["pocket_coords"])
        residue_target = residue_distance <= 8.0
        residue_contact = _balanced_binary_loss(
            output.residue_contact_logits[residue_mask], residue_target[residue_mask]
        )
    else:
        residue_contact = output.residue_contact_logits.sum() * 0.0

    distogram_target, distogram_mask, _ = residue_distogram_targets(batch)
    if distogram_mask.any():
        distogram = F.cross_entropy(
            output.distogram_logits[distogram_mask],
            distogram_target[distogram_mask],
        )
    else:
        distogram = output.distogram_logits.sum() * 0.0

    affinity_class = F.binary_cross_entropy_with_logits(
        output.affinity_class_logits, (normalized_labels >= 0.0).to(output.affinity_class_logits.dtype)
    )
    total = (
        smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.contact_weight * contact
        + config.residue_contact_weight * residue_contact
        + config.affinity_class_weight * affinity_class
        + config.distogram_weight * distogram
    )
    return total, {
        "smooth_l1": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "contact": float(contact.detach()),
        "residue_contact": float(residue_contact.detach()),
        "affinity_class": float(affinity_class.detach()),
        "distogram": float(distogram.detach()),
    }


def _update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _loader(
    dataset,
    batch_size,
    row_depth,
    *,
    training=False,
    subset_index=0,
    labels=None,
    seed=0,
    minimum_gap=0.1,
    num_workers=0,
):
    num_workers = int(num_workers)
    common = {
        "dataset": dataset,
        "collate_fn": MSAPairCollator(
            row_depth=row_depth, training=training, subset_index=subset_index
        ),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 2
    if labels is not None:
        sampler = LabelSpreadBatchSampler(labels, batch_size, seed, minimum_gap)
        return torch.utils.data.DataLoader(batch_sampler=sampler, **common), sampler
    return (
        torch.utils.data.DataLoader(
            batch_size=batch_size, shuffle=False, drop_last=False, **common
        ),
        None,
    )


def predict_model(model, dataset, config, label_mean, label_std, subsets=None):
    device = torch.device(config.device)
    amp_dtype = _autocast_dtype(config.amp_dtype)
    subset_count = int(config.inference_subsets if subsets is None else subsets)
    if subset_count <= 0:
        raise ValueError("inference subset count must be positive")
    model.eval()
    prediction_runs = []
    latent_runs = []
    reference_keys = None
    reference_labels = None
    with torch.no_grad():
        for subset_index in range(subset_count):
            loader, _ = _loader(
                dataset,
                config.evaluation_batch_size,
                config.row_depth,
                subset_index=subset_index,
                num_workers=config.num_workers,
            )
            keys = []
            labels = []
            predictions = []
            latents = []
            for batch in loader:
                keys.extend(batch["keys"])
                labels.append(batch["labels"].numpy().astype(np.float64))
                moved = move_batch_to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                    output = model(moved)
                prediction = output.prediction.float().cpu().numpy()
                prediction = prediction * float(label_std) + float(label_mean)
                predictions.append(prediction.astype(np.float64))
                latents.append(output.latent.float().cpu().numpy().astype(np.float32))
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            latents = np.concatenate(latents)
            if reference_keys is None:
                reference_keys = tuple(keys)
                reference_labels = labels
            elif tuple(keys) != reference_keys or not np.allclose(labels, reference_labels, atol=1e-6):
                raise ValueError("inference subset order changed")
            prediction_runs.append(predictions)
            latent_runs.append(latents)
    predictions = np.mean(np.stack(prediction_runs), axis=0)
    latents = np.mean(np.stack(latent_runs), axis=0).astype(np.float32)
    if not np.isfinite(predictions).all() or not np.isfinite(latents).all():
        raise RuntimeError("V12 prediction contains non-finite values")
    return {
        "keys": reference_keys,
        "labels": reference_labels,
        "predictions": predictions,
        "latents": latents,
        "metrics": affinity_metrics(reference_labels, predictions),
    }


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_pretrained_encoder(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("V12 pretraining checkpoint does not exist: %s" % checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError("V12 pretraining checkpoint schema is invalid")
    encoder_state = checkpoint.get("encoder_state")
    if not isinstance(encoder_state, dict) or not encoder_state:
        raise ValueError("V12 pretraining checkpoint has no encoder_state")
    try:
        model.msa_pair_encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "V12 pretraining checkpoint architecture does not match affinity model"
        ) from error
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "variant": str(checkpoint.get("variant", "unknown")),
        "metrics": checkpoint.get("metrics"),
    }


def fit_msa_pair_expert(train_dataset, valid_dataset, output, config):
    _seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    label_mean = float(np.mean(train_dataset.labels))
    label_std = float(np.std(train_dataset.labels))
    if not math.isfinite(label_std) or label_std <= 0.0:
        raise ValueError("training labels have zero or invalid variance")
    if not 0.0 < float(config.encoder_learning_rate_scale) <= 1.0:
        raise ValueError("encoder_learning_rate_scale must be in (0, 1]")
    model = MSAPairExpertV12(**config.model_kwargs())
    pretraining_lineage = _load_pretrained_encoder(model, config.pretrained_checkpoint)
    model = model.to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    encoder_parameters = list(model.msa_pair_encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": task_parameters, "lr": config.learning_rate},
            {
                "params": encoder_parameters,
                "lr": config.learning_rate * config.encoder_learning_rate_scale,
            },
        ],
        weight_decay=config.weight_decay,
    )
    train_loader, spread_sampler = _loader(
        train_dataset,
        config.batch_size,
        config.row_depth,
        training=True,
        labels=train_dataset.labels,
        seed=config.seed,
        minimum_gap=config.minimum_rank_gap * label_std,
        num_workers=config.num_workers,
    )
    total_steps = max(1, config.maximum_epochs * len(train_loader))
    warmup_steps = max(1, int(0.08 * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_dtype = _autocast_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    best = None
    history = []
    stale = 0
    for epoch in range(config.maximum_epochs):
        spread_sampler.set_epoch(epoch)
        model.train()
        names = (
            "loss",
            "smooth_l1",
            "mse",
            "rank",
            "contact",
            "residue_contact",
            "affinity_class",
            "distogram",
        )
        sums = {name: 0.0 for name in names}
        batches = 0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            normalized = (moved["labels"] - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = msa_pair_loss(output_value, normalized, moved, config)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V12 training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite V12 gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, config.ema_decay)
            sums["loss"] += float(loss.detach())
            for name, value in components.items():
                sums[name] += value
            batches += 1
        raw = predict_model(model, valid_dataset, config, label_mean, label_std, subsets=1)
        ema = predict_model(ema_model, valid_dataset, config, label_mean, label_std, subsets=1)
        variant_name, selected = max(
            (("raw", raw), ("ema", ema)),
            key=lambda item: (
                float(item[1]["metrics"]["CI"]),
                -float(item[1]["metrics"]["RMSE"]),
            ),
        )
        score = (float(selected["metrics"]["CI"]), -float(selected["metrics"]["RMSE"]))
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_learning_rate": optimizer.param_groups[1]["lr"],
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": dict(raw["metrics"]),
            "ema": dict(ema["metrics"]),
            "selected_variant": variant_name,
        }
        history.append(row)
        print(json.dumps({"event": "msa_v12_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > best["score"]:
            chosen_model = model if variant_name == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "config": asdict(config),
                "epoch": epoch,
                "label_mean": label_mean,
                "label_std": label_std,
                "metrics": dict(selected["metrics"]),
                "pretraining_lineage": pretraining_lineage,
                "model_state": {
                    key: value.detach().cpu() for key, value in chosen_model.state_dict().items()
                },
                "variant": variant_name,
            }
            _atomic_torch_save(checkpoint, output / "best.pt")
            best = {"score": score, "checkpoint": checkpoint}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best is None:
        raise RuntimeError("V12 training did not produce a checkpoint")
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
        handle.write("\n")
    selected_model = MSAPairExpertV12(**config.model_kwargs()).to(device)
    selected_model.load_state_dict(best["checkpoint"]["model_state"])
    return selected_model, best["checkpoint"], history


__all__ = [
    "TrainingConfig",
    "fit_msa_pair_expert",
    "msa_pair_loss",
    "predict_model",
]
