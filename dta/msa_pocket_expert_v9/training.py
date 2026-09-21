"""Training and evaluation for the MSA pocket expert."""

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

from .dataset import collate_msa_pocket_complexes, move_batch_to_device
from .model import MSAPocketExpertV9


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 42
    batch_size: int = 8
    evaluation_batch_size: int = 12
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 100
    patience: int = 18
    gradient_clip: float = 1.0
    model_width: int = 384
    layers: int = 6
    heads: int = 8
    dropout: float = 0.10
    latent_width: int = 1024
    rank_weight: float = 0.35
    mse_weight: float = 0.20
    contact_weight: float = 0.05
    ema_decay: float = 0.995
    minimum_rank_gap: float = 0.10

    def model_kwargs(self):
        return {
            "model_width": self.model_width,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "latent_width": self.latent_width,
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


def msa_pocket_loss(output, normalized_labels, batch, config):
    smooth = F.smooth_l1_loss(output.prediction, normalized_labels, beta=0.5)
    mse = F.mse_loss(output.prediction, normalized_labels)
    rank = _pairwise_rank_loss(
        output.prediction, normalized_labels, config.minimum_rank_gap
    )
    mask = batch["contact_mask"]
    contact_target = batch["contact_label"][mask].to(output.contact_logits.dtype)
    contact_logits = output.contact_logits[mask]
    positive = contact_target.sum()
    negative = contact_target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    contact = F.binary_cross_entropy_with_logits(
        contact_logits, contact_target, pos_weight=positive_weight
    )
    total = (
        smooth
        + config.mse_weight * mse
        + config.rank_weight * rank
        + config.contact_weight * contact
    )
    return total, {
        "smooth_l1": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "contact": float(contact.detach()),
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
    shuffle=False,
    labels=None,
    seed=0,
    minimum_gap=0.1,
    num_workers=0,
):
    num_workers = int(num_workers)
    common = {
        "dataset": dataset,
        "collate_fn": collate_msa_pocket_complexes,
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
            batch_size=batch_size, shuffle=shuffle, drop_last=False, **common
        ),
        None,
    )


def predict_model(model, dataset, config, label_mean, label_std):
    device = torch.device(config.device)
    loader, _ = _loader(
        dataset,
        config.evaluation_batch_size,
        num_workers=config.num_workers,
    )
    amp_dtype = _autocast_dtype(config.amp_dtype)
    model.eval()
    keys = []
    labels = []
    predictions = []
    latents = []
    with torch.no_grad():
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
    if not np.isfinite(predictions).all() or not np.isfinite(latents).all():
        raise RuntimeError("MSA pocket prediction contains non-finite values")
    return {
        "keys": tuple(keys),
        "labels": labels,
        "predictions": predictions,
        "latents": latents,
        "metrics": affinity_metrics(labels, predictions),
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


def fit_msa_pocket_expert(train_dataset, valid_dataset, output, config):
    _seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    label_mean = float(np.mean(train_dataset.labels))
    label_std = float(np.std(train_dataset.labels))
    if not math.isfinite(label_std) or label_std <= 0.0:
        raise ValueError("training labels have zero or invalid variance")
    model = MSAPocketExpertV9(**config.model_kwargs()).to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader, spread_sampler = _loader(
        train_dataset,
        config.batch_size,
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
    global_step = 0
    for epoch in range(config.maximum_epochs):
        spread_sampler.set_epoch(epoch)
        model.train()
        sums = {"loss": 0.0, "smooth_l1": 0.0, "mse": 0.0, "rank": 0.0, "contact": 0.0}
        batches = 0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            normalized = (moved["labels"] - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = msa_pocket_loss(output_value, normalized, moved, config)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MSA pocket training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite MSA pocket gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, config.ema_decay)
            sums["loss"] += float(loss.detach())
            for name, value in components.items():
                sums[name] += value
            batches += 1
            global_step += 1
        raw = predict_model(model, valid_dataset, config, label_mean, label_std)
        ema = predict_model(ema_model, valid_dataset, config, label_mean, label_std)
        variants = (("raw", raw), ("ema", ema))
        variant_name, selected = max(
            variants,
            key=lambda item: (
                float(item[1]["metrics"]["CI"]),
                -float(item[1]["metrics"]["RMSE"]),
            ),
        )
        score = (
            float(selected["metrics"]["CI"]),
            -float(selected["metrics"]["RMSE"]),
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": dict(raw["metrics"]),
            "ema": dict(ema["metrics"]),
            "selected_variant": variant_name,
        }
        history.append(row)
        print(json.dumps({"event": "msa_v9_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > best["score"]:
            chosen_model = model if variant_name == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "config": asdict(config),
                "epoch": epoch,
                "label_mean": label_mean,
                "label_std": label_std,
                "metrics": dict(selected["metrics"]),
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
        raise RuntimeError("MSA pocket training did not produce a checkpoint")
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
        handle.write("\n")
    selected_model = MSAPocketExpertV9(**config.model_kwargs()).to(device)
    selected_model.load_state_dict(best["checkpoint"]["model_state"])
    return selected_model, best["checkpoint"], history


__all__ = [
    "TrainingConfig",
    "fit_msa_pocket_expert",
    "msa_pocket_loss",
    "predict_model",
]
