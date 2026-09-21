"""Optimization loop for V27 (kept separate from reusable training utilities)."""

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
from .losses import privileged_pair_loss
from .model import OptionalStructurePrivilegedPairDTA
from .training import (
    VERSION,
    _parameter_groups,
    _scheduler,
    _selection_score,
    apply_training_masks,
    atomic_torch_save,
    build_loader,
    predict,
    save_predictions,
)


def _save_last(path, config, epoch, model, ema_model, optimizer, scheduler, scaler, history, best, stale, sources):
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


def fit(train_dataset, valid_dataset, output, config, protein_cache, ligand_cache, warm_checkpoint=None, resume=False):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V27 output exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    label_stats = label_statistics(train_dataset)
    means, standard_deviations = _normalizers(label_stats, device)
    thresholds, positive_weights = _binary_training_statistics(train_dataset, label_stats, device)
    model = OptionalStructurePrivilegedPairDTA(**config.model_kwargs()).to(device)
    sources = {}
    if warm_checkpoint is not None:
        sources["warm_v22"] = _compatible_state(model, warm_checkpoint)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(_parameter_groups(model, config), weight_decay=config.weight_decay)
    sampler = TargetAwareBalancedBatchSampler(
        train_dataset,
        config.batch_size,
        config.steps_per_epoch,
        {"LP_PDBBind": config.lp_sampling, "DAVIS": config.davis_sampling, "KIBA": config.kiba_sampling},
        pair_batch_fraction=config.pair_batch_fraction,
        seed=config.seed,
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
            raise ValueError("V27 resume lineage/config differs")
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
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums, batches = {}, 0
        for batch_index, batch in enumerate(loader):
            moved = apply_training_masks(move_batch_to_device(batch, device), config)
            normalized = (moved["labels"] - means[moved["dataset_index"]]) / standard_deviations[moved["dataset_index"]]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = privileged_pair_loss(
                    model, output_value, normalized, moved, thresholds, positive_weights, loss_config
                )
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V27 training loss")
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % config.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V27 gradient")
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
        variant, selected = max((("raw", raw), ("ema", ema)), key=lambda item: _selection_score(item[1]))
        score = _selection_score(selected)
        row = {
            "epoch": int(epoch),
            "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["metrics"],
            "ema": ema["metrics"],
            "selected_variant": variant,
            "selection_score": list(score),
        }
        history.append(row)
        print(json.dumps({"event": "v27_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            checkpoint = {
                "schema_version": 1,
                "version": VERSION,
                "phase": "privileged_pair",
                "config": asdict(config),
                "model_kwargs": config.model_kwargs(),
                "model_state": {key: value.detach().cpu() for key, value in chosen.state_dict().items()},
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
        _save_last(last_path, config, epoch, model, ema_model, optimizer, scheduler, scaler, history, best, stale, sources)
        if stale >= config.patience:
            break
    if best is None:
        raise RuntimeError("V27 training did not produce a checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    selected_model = OptionalStructurePrivilegedPairDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    valid_result = predict(selected_model, valid_dataset, config, label_stats)
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


__all__ = ["fit"]
