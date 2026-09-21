"""Joint masked-MSA, residue-pair, and protein-ligand interface pretraining."""

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
from .model import (
    MSA_MASK_INDEX,
    PocketMSAInterfaceV13,
    interface_distance_targets,
    residue_distogram_targets,
)
from utils.experiment_integrity import sha256_file


@dataclass(frozen=True)
class PretrainingConfig:
    initial_msa_checkpoint: str = None
    device: str = "cuda:0"
    seed: int = 42
    batch_size: int = 4
    evaluation_batch_size: int = 6
    num_workers: int = 4
    amp_dtype: str = "bfloat16"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 16
    patience: int = 5
    gradient_clip: float = 1.0
    msa_width: int = 192
    msa_layers: int = 4
    msa_heads: int = 6
    pair_factor: int = 8
    pair_width: int = 96
    interface_blocks: int = 2
    dropout: float = 0.10
    row_depth: int = 64
    mask_probability: float = 0.15
    interface_mask_probability: float = 0.35
    distogram_weight: float = 0.35
    contact_weight: float = 0.10
    interface_distogram_weight: float = 1.0
    interface_contact_weight: float = 0.30
    match_weight: float = 0.10
    ema_decay: float = 0.995

    def encoder_kwargs(self):
        return {
            "width": self.msa_width,
            "layers": self.msa_layers,
            "heads": self.msa_heads,
            "pair_factor": self.pair_factor,
            "pair_width": self.pair_width,
            "interface_blocks": self.interface_blocks,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class PretrainingOutput:
    evolutionary: object
    token_logits: torch.Tensor
    token_targets: torch.Tensor
    token_mask: torch.Tensor
    interface_mask: torch.Tensor


class MSAInterfacePretrainerV13(nn.Module):
    def __init__(
        self, encoder_kwargs, mask_probability=0.15, interface_mask_probability=0.35
    ):
        super().__init__()
        self.encoder = PocketMSAInterfaceV13(**encoder_kwargs)
        self.mask_probability = float(mask_probability)
        self.interface_mask_probability = float(interface_mask_probability)
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

    def _mask_interface(self, batch):
        valid = (
            batch["contact_mask"]
            & batch["pocket_mask"][:, :, None]
            & batch["ligand_mask"][:, None, :]
        )
        selected = valid & (
            torch.rand_like(valid, dtype=torch.float32)
            < self.interface_mask_probability
        )
        for batch_index in range(valid.shape[0]):
            if valid[batch_index].any() and not selected[batch_index].any():
                first = torch.nonzero(valid[batch_index], as_tuple=False)[0]
                selected[batch_index, first[0], first[1]] = True
        if not selected.any():
            raise ValueError("pretraining batch contains no valid interface pairs")
        return selected, valid & ~selected

    def forward(self, batch):
        corrupted, targets, selected = self._corrupt_tokens(batch)
        interface_mask, interface_visible = self._mask_interface(batch)
        encoder_batch = dict(batch)
        encoder_batch["pocket_msa_tokens"] = corrupted
        encoder_batch["interface_distance_visible"] = interface_visible
        evolutionary = self.encoder(encoder_batch)
        return PretrainingOutput(
            evolutionary=evolutionary,
            token_logits=self.token_head(evolutionary.row_states),
            token_targets=targets,
            token_mask=selected,
            interface_mask=interface_mask,
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
    residue_contact = _balanced_contact_loss(
        output.evolutionary.residue_contact_logits[pair_mask],
        (distances <= 8.0)[pair_mask],
    )

    interface_targets, valid_interface, interface_distances = (
        interface_distance_targets(batch)
    )
    interface_mask = output.interface_mask & valid_interface
    if not interface_mask.any():
        raise ValueError("pretraining batch contains no masked interface pairs")
    interface_logits = output.evolutionary.interface_distance_logits[interface_mask]
    interface_distogram = F.cross_entropy(
        interface_logits, interface_targets[interface_mask]
    )
    interface_distogram_accuracy = (
        interface_logits.argmax(dim=-1) == interface_targets[interface_mask]
    ).float().mean()
    interface_contact = _balanced_contact_loss(
        output.evolutionary.interface_contact_logits[interface_mask],
        (interface_distances <= 4.5)[interface_mask],
    )

    if output.evolutionary.match_logits.shape[0] > 1:
        match_target = torch.arange(
            output.evolutionary.match_logits.shape[0],
            device=output.evolutionary.match_logits.device,
        )
        match = 0.5 * (
            F.cross_entropy(output.evolutionary.match_logits, match_target)
            + F.cross_entropy(output.evolutionary.match_logits.T, match_target)
        )
    else:
        match = output.evolutionary.match_logits.sum() * 0.0

    total = (
        masked_msa
        + config.distogram_weight * distogram
        + config.contact_weight * residue_contact
        + config.interface_distogram_weight * interface_distogram
        + config.interface_contact_weight * interface_contact
        + config.match_weight * match
    )
    return total, {
        "masked_msa": float(masked_msa.detach()),
        "token_accuracy": float(token_accuracy.detach()),
        "distogram": float(distogram.detach()),
        "distogram_accuracy": float(distogram_accuracy.detach()),
        "residue_contact": float(residue_contact.detach()),
        "interface_distogram": float(interface_distogram.detach()),
        "interface_distogram_accuracy": float(interface_distogram_accuracy.detach()),
        "interface_contact": float(interface_contact.detach()),
        "match": float(match.detach()),
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
        "residue_contact",
        "interface_distogram",
        "interface_distogram_accuracy",
        "interface_contact",
        "match",
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


def _load_initial_msa_encoder(encoder, checkpoint_path):
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "V13 initial MSA checkpoint does not exist: %s" % checkpoint_path
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError("V13 initial MSA checkpoint schema is invalid")
    encoder_state = checkpoint.get("encoder_state")
    if not isinstance(encoder_state, dict) or not encoder_state:
        raise ValueError("V13 initial MSA checkpoint has no encoder_state")
    current = encoder.state_dict()
    incompatible = [
        name
        for name, value in encoder_state.items()
        if name not in current or tuple(current[name].shape) != tuple(value.shape)
    ]
    if incompatible:
        raise ValueError(
            "V13 initial MSA architecture mismatch: %s" % ", ".join(incompatible[:5])
        )
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.unexpected_keys:
        raise ValueError("V13 initial MSA checkpoint has unexpected parameters")
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "variant": str(checkpoint.get("variant", "unknown")),
        "metrics": checkpoint.get("metrics"),
        "loaded_parameter_count": len(encoder_state),
    }


def fit_msa_interface_pretrainer(train_dataset, valid_dataset, output, config):
    _seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = MSAInterfacePretrainerV13(
        config.encoder_kwargs(),
        mask_probability=config.mask_probability,
        interface_mask_probability=config.interface_mask_probability,
    )
    initial_msa_lineage = _load_initial_msa_encoder(
        model.encoder, config.initial_msa_checkpoint
    )
    model = model.to(device)
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
        "residue_contact",
        "interface_distogram",
        "interface_distogram_accuracy",
        "interface_contact",
        "match",
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
                raise RuntimeError("non-finite V13 pretraining loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite V13 pretraining gradient")
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
            json.dumps({"event": "msa_interface_v13_pretrain_epoch", **row}, sort_keys=True),
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
                "initial_msa_lineage": initial_msa_lineage,
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
        raise RuntimeError("V13 pretraining did not produce a checkpoint")
    return best["checkpoint"], history


__all__ = [
    "MSAInterfacePretrainerV13",
    "PretrainingConfig",
    "fit_msa_interface_pretrainer",
    "pretraining_loss",
]
