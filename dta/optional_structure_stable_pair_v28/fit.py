"""Optimization loop for V28 with a fixed base teacher and bounded pair transfer."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from dta.optional_structure_staged_v22.training import (
    _amp_dtype,
    _binary_training_statistics,
    _compatible_state,
    _normalizers,
    cache_lineage,
    label_statistics,
    seed_everything,
    update_ema,
)

from .data import DATASET_INDEX, TargetAwareBalancedBatchSampler, manifest_fingerprint, move_batch_to_device
from .losses import stable_pair_loss
from .model import OptionalStructureStablePairDTA
from .training import (
    VERSION,
    _parameter_groups,
    _scheduler,
    _selection_score,
    _set_pretrained_trainable,
    apply_training_masks,
    atomic_torch_save,
    build_loader,
    pair_distillation_scale,
    predict,
    save_predictions,
)


def _save_last(
    path,
    config,
    epoch,
    model,
    ema_model,
    optimizer,
    scheduler,
    scaler,
    history,
    best,
    stale,
    sources,
):
    atomic_torch_save(
        {
            "config": asdict(config),
            "epoch": int(epoch),
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "ema_state": {key: value.detach().cpu() for key, value in ema_model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "best": best,
            "stale": int(stale),
            "sources": sources,
        },
        path,
    )


def fit(
    train_dataset,
    valid_dataset,
    output,
    config,
    protein_cache,
    ligand_cache,
    warm_checkpoint=None,
    resume=False,
):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V28 output exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    if warm_checkpoint is None:
        raise ValueError("V28 requires the sealed V22 structure checkpoint as a fixed base teacher")
    device = torch.device(config.device)
    label_stats = label_statistics(train_dataset)
    means, standard_deviations = _normalizers(label_stats, device)
    thresholds, positive_weights = _binary_training_statistics(train_dataset, label_stats, device)
    model = OptionalStructureStablePairDTA(**config.model_kwargs()).to(device)
    warm_source = _compatible_state(model, warm_checkpoint)
    sources = {
        "warm_v22": warm_source,
        "fixed_base_teacher": {
            "kind": "frozen V28 snapshot immediately after V22 compatible loading",
            "path": warm_source["path"],
            "sha256": warm_source["sha256"],
        },
    }

    # This immutable snapshot always predicts through force_no_3d=True. Its zero-initialized
    # V28 adapters preserve the loaded V22 sequence+2D prediction at initialization.
    base_teacher = deepcopy(model).to(device).eval()
    for parameter in base_teacher.parameters():
        parameter.requires_grad_(False)

    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(_parameter_groups(model, config), weight_decay=config.weight_decay)
    sampler = TargetAwareBalancedBatchSampler(
        train_dataset,
        config.batch_size,
        config.steps_per_epoch,
        {
            "LP_PDBBind": config.lp_sampling,
            "DAVIS": config.davis_sampling,
            "KIBA": config.kiba_sampling,
        },
        pair_batch_fraction=config.pair_batch_fraction,
        seed=config.seed,
    )
    loader = build_loader(train_dataset, config, training=True, sampler=sampler)
    scheduler = _scheduler(optimizer, config, len(loader))
    amp_dtype = _amp_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    history, best, stale, start_epoch = [], None, 0, 0
    last_path = output / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume requires %s" % last_path)
        state = torch.load(str(last_path), map_location="cpu")
        if state["config"] != asdict(config) or state["sources"] != sources:
            raise ValueError("V28 resume lineage/config differs")
        model.load_state_dict(state["model_state"], strict=True)
        ema_model.load_state_dict(state["ema_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history, best, stale = state["history"], state["best"], int(state["stale"])
        start_epoch = int(state["epoch"]) + 1

    loss_config = config.loss_config()
    for epoch in range(start_epoch, config.maximum_epochs):
        sampler.set_epoch(epoch)
        pretrained_frozen = int(epoch) < int(config.pretrained_freeze_epochs)
        pretrained_parameter_count = _set_pretrained_trainable(model, not pretrained_frozen)
        distillation_scale = pair_distillation_scale(epoch, config)
        model.train()
        base_teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        sums, batches = {}, 0
        for batch_index, batch in enumerate(loader):
            moved = apply_training_masks(move_batch_to_device(batch, device), config)
            normalized = (
                moved["labels"] - means[moved["dataset_index"]]
            ) / standard_deviations[moved["dataset_index"]]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                    anchor_output = base_teacher(moved, force_no_3d=True)
                    base_anchor_prediction = anchor_output.base_prediction.detach()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = stable_pair_loss(
                    model,
                    output_value,
                    normalized,
                    moved,
                    thresholds,
                    positive_weights,
                    loss_config,
                    base_anchor_prediction=base_anchor_prediction,
                    pair_teacher_latent=output_value.full_pair_latent.detach(),
                    pair_distillation_scale=distillation_scale,
                )
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V28 training loss")
            if float(components["pair_distillation"]) > float(config.maximum_pair_distillation):
                raise RuntimeError(
                    "V28 normalized pair distillation exceeded fail-fast bound: %.6f > %.6f"
                    % (components["pair_distillation"], config.maximum_pair_distillation)
                )
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % config.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V28 gradient")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, config.ema_decay)
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1

        raw = predict(model, valid_dataset, config, label_stats)
        ema = predict(ema_model, valid_dataset, config, label_stats)
        variant, selected = max(
            (("raw", raw), ("ema", ema)), key=lambda item: _selection_score(item[1])
        )
        score = _selection_score(selected)
        row = {
            "epoch": int(epoch),
            "learning_rates": {
                group["name"]: float(group["lr"]) for group in optimizer.param_groups
            },
            "pretrained_frozen": bool(pretrained_frozen),
            "pretrained_parameter_count": int(pretrained_parameter_count),
            "pair_distillation_scale": float(distillation_scale),
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["metrics"],
            "ema": ema["metrics"],
            "selected_variant": variant,
            "selection_score": list(score),
        }
        history.append(row)
        print(json.dumps({"event": "v28_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "version": VERSION,
                "phase": "stable_pair",
                "config": asdict(config),
                "model_kwargs": config.model_kwargs(),
                "model_state": {
                    key: value.detach().cpu() for key, value in chosen.state_dict().items()
                },
                "label_statistics": label_stats,
                "binary_training_statistics": {
                    "normalized_thresholds": thresholds.detach().cpu().tolist(),
                    "positive_weights": positive_weights.detach().cpu().tolist(),
                },
                "epoch": int(epoch),
                "variant": variant,
                "validation_metrics": selected["metrics"],
                "validation_base_metrics": selected["base_metrics"],
                "selection_rule": "validation weighted CI 0.50 LP + 0.25 Davis + 0.25 KIBA; then LP CI, macro Spearman, LP RMSE",
                "stability_strategy": {
                    "fixed_base_teacher": True,
                    "normalized_stop_gradient_pair_teacher": True,
                    "pretrained_freeze_epochs": int(config.pretrained_freeze_epochs),
                    "pair_distillation_warmup_epochs": int(
                        config.pair_distillation_warmup_epochs
                    ),
                    "maximum_pair_distillation": float(config.maximum_pair_distillation),
                },
                "train_manifest_sha256": manifest_fingerprint(train_dataset.manifest_path),
                "valid_manifest_sha256": manifest_fingerprint(valid_dataset.manifest_path),
                "protein_cache": cache_lineage(protein_cache),
                "ligand_cache": cache_lineage(ligand_cache),
                "sources": sources,
            }
            atomic_torch_save(checkpoint, output / "best.pt")
            best = {"score": list(score), "epoch": int(epoch), "variant": variant}
            stale = 0
        else:
            stale += 1
        _save_last(
            last_path,
            config,
            epoch,
            model,
            ema_model,
            optimizer,
            scheduler,
            scaler,
            history,
            best,
            stale,
            sources,
        )
        if stale >= config.patience:
            break
    if best is None:
        raise RuntimeError("V28 training did not produce a checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    selected_model = OptionalStructureStablePairDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    valid_result = predict(selected_model, valid_dataset, config, label_stats)
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


__all__ = ["fit"]

