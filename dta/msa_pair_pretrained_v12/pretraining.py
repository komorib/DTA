"""Masked-MSA and residue-distogram pretraining for the V12 pair encoder."""

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
from torch import nn
from torch.nn import functional as F

from .dataset import MSAPairCollator, move_batch_to_device
from .model import MSA_MASK_INDEX, PocketMSAToPairV12, residue_distogram_targets


@dataclass(frozen=True)
class PretrainingConfig:
    device: str = "cuda:0"
    seed: int = 42
    batch_size: int = 4
    evaluation_batch_size: int = 6
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 12
    patience: int = 4
    gradient_clip: float = 1.0
    msa_width: int = 192
    msa_layers: int = 4
    msa_heads: int = 6
    pair_factor: int = 8
    pair_width: int = 96
    dropout: float = 0.10
    row_depth: int = 64
    mask_probability: float = 0.15
    distogram_weight: float = 1.0
    contact_weight: float = 0.20
    ema_decay: float = 0.995

    def encoder_kwargs(self):
        return {
            "width": self.msa_width,
            "layers": self.msa_layers,
            "heads": self.msa_heads,
            "pair_factor": self.pair_factor,
            "pair_width": self.pair_width,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class PretrainingOutput:
    evolutionary: object
    token_logits: torch.Tensor
    token_targets: torch.Tensor
    token_mask: torch.Tensor


class MSAPairPretrainerV12(nn.Module):
    def __init__(self, encoder_kwargs, mask_probability=0.15):
        super().__init__()
        self.encoder = PocketMSAToPairV12(**encoder_kwargs)
        self.mask_probability = float(mask_probability)
        self.token_head = nn.Linear(int(encoder_kwargs["width"]), 22)

    def _corrupt_tokens(self, batch):
        original = batch["pocket_msa_tokens"]
        valid = batch["pocket_msa_row_mask"] & (original <= 21)
        selected = valid & (
            torch.rand_like(original, dtype=torch.float32) < self.mask_probability
        )
        if not selected.any():
            first = torch.nonzero(valid, as_tuple=False)
            if first.numel() == 0:
                raise ValueError("pretraining batch contains no valid MSA tokens")
            index = first[0]
            selected[index[0], index[1], index[2]] = True
        corrupted = original.clone()
        policy = torch.rand_like(original, dtype=torch.float32)
        corrupted[selected & (policy < 0.80)] = MSA_MASK_INDEX
        random_mask = selected & (policy >= 0.80) & (policy < 0.90)
        random_tokens = torch.randint(0, 22, original.shape, device=original.device)
        corrupted[random_mask] = random_tokens[random_mask]
        return corrupted, original, selected

    def forward(self, batch):
        corrupted, targets, selected = self._corrupt_tokens(batch)
        encoder_batch = dict(batch)
        encoder_batch["pocket_msa_tokens"] = corrupted
        evolutionary = self.encoder(encoder_batch)
        return PretrainingOutput(
            evolutionary=evolutionary,
            token_logits=self.token_head(evolutionary.row_states),
            token_targets=targets,
            token_mask=selected,
        )


def _balanced_contact_loss(logits, target):
    target = target.to(logits.dtype)
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight
    )


def pretraining_loss(output, batch, config):
    token_logits = output.token_logits[output.token_mask]
    token_targets = output.token_targets[output.token_mask]
    masked_msa = F.cross_entropy(token_logits, token_targets)
    token_accuracy = (token_logits.argmax(dim=-1) == token_targets).float().mean()

    distogram_targets, pair_mask, distances = residue_distogram_targets(batch)
    if not pair_mask.any():
        raise ValueError("pretraining batch contains no valid residue pairs")
    distogram_logits = output.evolutionary.distogram_logits[pair_mask]
    distogram = F.cross_entropy(distogram_logits, distogram_targets[pair_mask])
    distogram_accuracy = (
        distogram_logits.argmax(dim=-1) == distogram_targets[pair_mask]
    ).float().mean()
    contact = _balanced_contact_loss(
        output.evolutionary.residue_contact_logits[pair_mask],
        (distances <= 8.0)[pair_mask],
    )
    total = (
        masked_msa
        + config.distogram_weight * distogram
        + config.contact_weight * contact
    )
    return total, {
        "masked_msa": float(masked_msa.detach()),
        "token_accuracy": float(token_accuracy.detach()),
        "distogram": float(distogram.detach()),
        "distogram_accuracy": float(distogram_accuracy.detach()),
        "contact": float(contact.detach()),
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


def _loader(dataset, config, training):
    workers = int(config.num_workers)
    arguments = {
        "dataset": dataset,
        "batch_size": config.batch_size if training else config.evaluation_batch_size,
        "shuffle": training,
        "drop_last": training,
        "collate_fn": MSAPairCollator(row_depth=config.row_depth, training=training),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        arguments["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**arguments)


def _update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _evaluate_with_fixed_masks(model, dataset, config, seed):
    torch.manual_seed(int(seed))
    device = torch.device(config.device)
    dtype = _autocast_dtype(config.amp_dtype)
    loader = _loader(dataset, config, training=False)
    model.eval()
    names = (
        "loss",
        "masked_msa",
        "token_accuracy",
        "distogram",
        "distogram_accuracy",
        "contact",
    )
    sums = {name: 0.0 for name in names}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            moved = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=dtype):
                output = model(moved)
                loss, components = pretraining_loss(output, moved, config)
            sums["loss"] += float(loss.detach())
            for name, value in components.items():
                sums[name] += value
            batches += 1
    return {name: value / max(1, batches) for name, value in sums.items()}


def _evaluate(model, dataset, config, seed):
    device = torch.device(config.device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        return _evaluate_with_fixed_masks(model, dataset, config, seed)


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


def fit_msa_pair_pretrainer(train_dataset, valid_dataset, output, config):
    _seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = MSAPairPretrainerV12(
        config.encoder_kwargs(), mask_probability=config.mask_probability
    ).to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = _loader(train_dataset, config, training=True)
    total_steps = max(1, config.maximum_epochs * len(train_loader))
    warmup_steps = max(1, int(0.05 * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    dtype = _autocast_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and dtype == torch.float16
    )
    history = []
    best = None
    stale = 0
    names = (
        "loss",
        "masked_msa",
        "token_accuracy",
        "distogram",
        "distogram_accuracy",
        "contact",
    )
    for epoch in range(config.maximum_epochs):
        model.train()
        sums = {name: 0.0 for name in names}
        batches = 0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=dtype):
                output_value = model(moved)
                loss, components = pretraining_loss(output_value, moved, config)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V12 pretraining loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite V12 pretraining gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, config.ema_decay)
            sums["loss"] += float(loss.detach())
            for name, value in components.items():
                sums[name] += value
            batches += 1
        train_metrics = {
            name: value / max(1, batches) for name, value in sums.items()
        }
        raw = _evaluate(model, valid_dataset, config, config.seed + 1000)
        ema = _evaluate(ema_model, valid_dataset, config, config.seed + 1000)
        variant, validation = min(
            (("raw", raw), ("ema", ema)), key=lambda item: item[1]["loss"]
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "raw": raw,
            "ema": ema,
            "selected_variant": variant,
        }
        history.append(row)
        print(
            json.dumps({"event": "msa_v12_pretrain_epoch", **row}, sort_keys=True),
            flush=True,
        )
        if best is None or validation["loss"] < best["loss"]:
            selected_model = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "config": asdict(config),
                "epoch": epoch,
                "metrics": validation,
                "variant": variant,
                "encoder_state": {
                    key: value.detach().cpu()
                    for key, value in selected_model.encoder.state_dict().items()
                },
                "pretrainer_state": {
                    key: value.detach().cpu()
                    for key, value in selected_model.state_dict().items()
                },
            }
            _atomic_torch_save(checkpoint, output / "best_pretrain.pt")
            best = {"loss": validation["loss"], "checkpoint": checkpoint}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    with (output / "pretrain_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if best is None:
        raise RuntimeError("V12 pretraining did not produce a checkpoint")
    return best["checkpoint"], history


__all__ = [
    "MSAPairPretrainerV12",
    "PretrainingConfig",
    "fit_msa_pair_pretrainer",
    "pretraining_loss",
]
