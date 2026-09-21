"""Fit one strictly outer-held-out V18-like privileged-MSA teacher for V31.

The command receives a prebuilt V31 fold root and a fold index.  It has no
official validation/test arguments.  Inner selection uses only a partition of
the outer-fit rows; the outer holdout is opened only after the selected (or
optionally refit) checkpoint has been sealed, then is exported as a feature
artifact for the CPU OOF scalar teacher builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from dta.msa_interface_pair_pooling_v18.dataset import MSAPairComplexDataset, move_batch_to_device
from dta.msa_interface_pair_pooling_v18.model import MSAInterfacePairPoolExpertV18
from dta.msa_interface_pair_pooling_v18.training import (
    TrainingConfig,
    _autocast_dtype,
    _load_pretrained_encoder,
    _loader,
    _mask_interface_for_training,
    _seed_everything,
    _update_ema,
    fit_msa_interface_expert,
    msa_interface_loss,
    predict_model,
)
from dta.optional_structure_staged_v22.data import sha256_file

from .folds import load_fold_metadata


VERSION = "optional-structure-privileged-msa-pair-outer-teacher-v31"
SCHEMA_VERSION = 1


def _atomic_npz(path, **arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(value, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_config(args, fold):
    # These are the V18 pair-pooling dimensions that yielded the historical
    # contact-aware latent.  No V18 affinity checkpoint is loaded here.
    return TrainingConfig(
        pretrained_checkpoint=str(Path(args.pretrained_checkpoint).expanduser().resolve()),
        affinity_initial_checkpoint=None,
        device=str(args.device),
        seed=int(args.seed) + 1009 * int(fold),
        batch_size=int(args.batch_size),
        evaluation_batch_size=int(args.evaluation_batch_size),
        num_workers=int(args.num_workers),
        amp_dtype=str(args.amp_dtype),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        maximum_epochs=int(args.maximum_epochs),
        patience=int(args.patience),
        model_width=512,
        layers=8,
        heads=8,
        dropout=float(args.dropout),
        latent_width=1024,
        msa_width=192,
        msa_layers=4,
        msa_heads=6,
        pair_factor=8,
        pair_width=96,
        interface_blocks=2,
        pair_token_width=256,
        pair_token_heads=8,
        pair_token_layers=2,
        pair_token_count=128,
        row_depth=64,
        inference_subsets=2,
        rank_weight=float(args.rank_weight),
        mse_weight=float(args.mse_weight),
        contact_weight=float(args.contact_weight),
        residue_contact_weight=float(args.residue_contact_weight),
        affinity_class_weight=float(args.affinity_class_weight),
        distogram_weight=float(args.distogram_weight),
        interface_distogram_weight=float(args.interface_distogram_weight),
        interface_contact_weight=float(args.interface_contact_weight),
        match_weight=float(args.match_weight),
        interface_mask_probability=float(args.interface_mask_probability),
        encoder_learning_rate_scale=float(args.encoder_learning_rate_scale),
    )


def _validate_pretraining_checkpoint(path, policy, fold_dir):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("V31 teacher pretraining checkpoint is missing")
    checkpoint = torch.load(str(path), map_location="cpu")
    if int(checkpoint.get("schema_version", -1)) != 1 or not checkpoint.get("encoder_state"):
        raise ValueError("V31 teacher needs a V13-style self-supervised encoder checkpoint")
    if checkpoint.get("model_state"):
        raise ValueError("V31 refuses an affinity-model checkpoint as teacher initialization")
    forbidden = ("MSAInterfacePairPooling_v18", "MSAPairExpert_v12")
    if any(token in str(path) for token in forbidden):
        raise ValueError("V31 refuses historical supervised V18/V12 initialization")
    if policy == "fold_local_pretraining":
        try:
            path.relative_to(Path(fold_dir).expanduser().resolve())
        except ValueError as error:
            raise ValueError("fold_local_pretraining checkpoint must live under this fold directory") from error
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "policy": str(policy),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_variant": str(checkpoint.get("variant", "unknown")),
    }


def _parameter_groups(model, config):
    encoder_parameters = list(model.msa_pair_encoder.parameters())
    ids = {id(value) for value in encoder_parameters}
    task_parameters = [value for value in model.parameters() if id(value) not in ids]
    return [
        {"params": task_parameters, "lr": float(config.learning_rate)},
        {
            "params": encoder_parameters,
            "lr": float(config.learning_rate) * float(config.encoder_learning_rate_scale),
        },
    ]


def _refit_outer_teacher(dataset, config, selected_epoch, selected_variant, output):
    """Refit on all outer-fit rows for the inner-selected epoch/EMA variant.

    There is intentionally no prediction/evaluation call in this function.
    The outer holdout cannot influence refit time, LR schedule, EMA choice, or
    checkpoint selection.
    """

    _seed_everything(config.seed + 7919)
    device = torch.device(config.device)
    model = MSAInterfacePairPoolExpertV18(**config.model_kwargs())
    pretraining_lineage = _load_pretrained_encoder(model, config.pretrained_checkpoint)
    model = model.to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    label_mean = float(np.mean(dataset.labels))
    label_std = float(np.std(dataset.labels))
    if not math.isfinite(label_std) or label_std <= 0.0:
        raise ValueError("V31 refit has invalid label standard deviation")
    optimizer = torch.optim.AdamW(_parameter_groups(model, config), weight_decay=config.weight_decay)
    loader, spread_sampler = _loader(
        dataset,
        config.batch_size,
        config.row_depth,
        training=True,
        labels=dataset.labels,
        seed=config.seed + 7919,
        minimum_gap=config.minimum_rank_gap * label_std,
        num_workers=config.num_workers,
    )
    total_steps = max(1, int(config.maximum_epochs) * len(loader))
    warmup_steps = max(1, int(0.08 * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_dtype = _autocast_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    epochs = int(selected_epoch) + 1
    if epochs <= 0 or epochs > int(config.maximum_epochs):
        raise ValueError("V31 inner-selected epoch is outside refit configuration")
    history = []
    for epoch in range(epochs):
        spread_sampler.set_epoch(epoch)
        model.train()
        sums, batches = {}, 0
        for batch in loader:
            moved = _mask_interface_for_training(
                move_batch_to_device(batch, device), config.interface_mask_probability
            )
            normalized = (moved["labels"] - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                value = model(moved)
                loss, components = msa_interface_loss(value, normalized, moved, config)
            if not torch.isfinite(loss):
                raise RuntimeError("V31 outer refit produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("V31 outer refit produced a non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, config.ema_decay)
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1
        row = {
            "epoch": epoch,
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "encoder_learning_rate": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(row)
        print(json.dumps({"event": "v31_outer_refit_epoch", **row}, sort_keys=True), flush=True)
    selected_model = model if str(selected_variant) == "raw" else ema_model
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "phase": "outer_refit_fixed_inner_selection",
        "config": asdict(config),
        "epoch": int(selected_epoch),
        "variant": str(selected_variant),
        "label_mean": label_mean,
        "label_std": label_std,
        "pretraining_lineage": pretraining_lineage,
        "affinity_initialization_lineage": None,
        "model_state": {name: value.detach().cpu() for name, value in selected_model.state_dict().items()},
        "selection_free": True,
    }
    _atomic_torch_save(checkpoint, output)
    return selected_model, checkpoint, history


def _export_features(model, dataset, config, label_mean, label_std, path):
    result = predict_model(model, dataset, config, label_mean, label_std)
    latent = np.asarray(result["latents"], dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != 2048:
        raise ValueError("V31 teacher did not emit V18 base+pair latent width 2048")
    pair_latent = latent[:, 1024:]
    if not np.isfinite(pair_latent).all():
        raise RuntimeError("V31 teacher pair latent contains non-finite values")
    _atomic_npz(
        path,
        keys=np.asarray(result["keys"]),
        labels=np.asarray(result["labels"], dtype=np.float64),
        predictions=np.asarray(result["predictions"], dtype=np.float64),
        pair_latents=pair_latent,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "count": int(pair_latent.shape[0]),
        "pair_latent_width": int(pair_latent.shape[1]),
    }


def _load_checkpoint_model(path, device):
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("version") not in (VERSION, None):
        raise ValueError("V31 outer teacher checkpoint version mismatch")
    config = TrainingConfig(**checkpoint["config"])
    config = replace(config, device=str(device))
    model = MSAInterfacePairPoolExpertV18(**config.model_kwargs()).to(torch.device(config.device))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, config, checkpoint


def run(args):
    fold_root = Path(args.fold_root).expanduser().resolve()
    source_train = Path(args.source_train_manifest).expanduser().resolve()
    metadata = load_fold_metadata(fold_root, source_train, require_full=bool(args.require_full_folds))
    fold = int(args.fold)
    if fold < 0 or fold >= int(metadata["folds"]):
        raise ValueError("V31 requested outer fold is invalid")
    fold_info = metadata["fold_artifacts"][fold]
    fold_dir = fold_root / ("fold_%d" % fold)
    for name, expected in fold_info["manifest_sha256"].items():
        path = fold_dir / Path(fold_info["manifests"][name]).name
        if sha256_file(path) != expected:
            raise ValueError("V31 fold manifest checksum mismatch: %s" % name)
    initialization = _validate_pretraining_checkpoint(
        args.pretrained_checkpoint, args.initialization_policy, fold_dir
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("V31 outer teacher output exists; choose a new TEACHER_ID")
    output.mkdir(parents=True, exist_ok=False)
    manifests = {name: fold_dir / Path(value).name for name, value in fold_info["manifests"].items()}
    inner_train = MSAPairComplexDataset(manifests["inner_train"], args.train_cache, args.row_cache)
    inner_valid = MSAPairComplexDataset(manifests["inner_valid"], args.train_cache, args.row_cache)
    config = _model_config(args, fold)
    # The historical V18 affinity checkpoint is intentionally not an argument
    # to this command.  Only the separately audited self-supervised encoder
    # may initialize the outer teacher.
    selected_model, selected_checkpoint, history = fit_msa_interface_expert(
        inner_train, inner_valid, output / "inner_selection", config
    )
    selected_path = output / "inner_selection" / "best.pt"
    selection = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "phase": "inner_selection",
        "fold": fold,
        "fold_root": str(fold_root),
        "folds_sha256": sha256_file(fold_root / "folds.json"),
        "source_train_manifest": str(source_train),
        "source_train_manifest_sha256": metadata["train_manifest_sha256"],
        "manifests": {name: str(path) for name, path in manifests.items()},
        "manifest_sha256": fold_info["manifest_sha256"],
        "counts": fold_info["counts"],
        "initialization": initialization,
        "affinity_initial_checkpoint": None,
        "outer_holdout_opened_before_selection": False,
        "official_valid_manifest_read": False,
        "test_manifest_read": False,
        "selected_epoch": int(selected_checkpoint["epoch"]),
        "selected_variant": str(selected_checkpoint["variant"]),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "inner_history_length": len(history),
    }
    _atomic_json(output / "selection.json", selection)
    if args.refit_outer:
        # Constructing outer_fit is delayed until now.  The outer holdout is
        # still unopened at this point and remains absent from refit.
        outer_fit = MSAPairComplexDataset(manifests["outer_fit"], args.train_cache, args.row_cache)
        model, checkpoint, refit_history = _refit_outer_teacher(
            outer_fit,
            config,
            selected_checkpoint["epoch"],
            selected_checkpoint["variant"],
            output / "outer_refit.pt",
        )
        effective_config = config
        effective_checkpoint = checkpoint
        effective_path = output / "outer_refit.pt"
        selection["refit"] = {
            "enabled": True,
            "checkpoint": str(effective_path),
            "checkpoint_sha256": sha256_file(effective_path),
            "history_length": len(refit_history),
            "selection_free": True,
        }
    else:
        outer_fit = MSAPairComplexDataset(manifests["outer_fit"], args.train_cache, args.row_cache)
        model = selected_model
        effective_config = config
        effective_checkpoint = selected_checkpoint
        effective_path = selected_path
        selection["refit"] = {"enabled": False, "selection_free": False}
    # Inference on outer holdout is deliberately after every checkpoint choice.
    outer_holdout = MSAPairComplexDataset(manifests["outer_holdout"], args.train_cache, args.row_cache)
    fit_features = _export_features(
        model,
        outer_fit,
        effective_config,
        effective_checkpoint["label_mean"],
        effective_checkpoint["label_std"],
        output / "outer_fit_features.npz",
    )
    holdout_features = _export_features(
        model,
        outer_holdout,
        effective_config,
        effective_checkpoint["label_mean"],
        effective_checkpoint["label_std"],
        output / "outer_holdout_features.npz",
    )
    selection.update(
        {
            "status": "complete",
            "effective_checkpoint": str(effective_path),
            "effective_checkpoint_sha256": sha256_file(effective_path),
            "outer_holdout_opened_before_selection": False,
            "outer_holdout_opened_after_selection": True,
            "outer_holdout_labels_used_for_checkpoint_selection": False,
            "outer_fit_features": fit_features,
            "outer_holdout_features": holdout_features,
            "outer_holdout_metrics": None,
        }
    )
    _atomic_json(output / "result.json", selection)
    print(json.dumps(selection, sort_keys=True), flush=True)
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--source-train-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--row-cache", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initialization-policy",
        choices=("historical_self_supervised", "fold_local_pretraining"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--maximum-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=0.40)
    parser.add_argument("--mse-weight", type=float, default=0.15)
    parser.add_argument("--contact-weight", type=float, default=0.05)
    parser.add_argument("--residue-contact-weight", type=float, default=0.03)
    parser.add_argument("--affinity-class-weight", type=float, default=0.05)
    parser.add_argument("--distogram-weight", type=float, default=0.02)
    parser.add_argument("--interface-distogram-weight", type=float, default=0.08)
    parser.add_argument("--interface-contact-weight", type=float, default=0.05)
    parser.add_argument("--match-weight", type=float, default=0.02)
    parser.add_argument("--interface-mask-probability", type=float, default=0.15)
    parser.add_argument("--encoder-learning-rate-scale", type=float, default=0.20)
    parser.add_argument("--refit-outer", action="store_true")
    parser.add_argument("--require-full-folds", action="store_true")
    return parser


def main(argv=None):
    return 0 if run(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
