"""Optimization loop for V29 sealed V22 bounded 3D residual learning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from dta.optional_structure_staged_v22.training import (
    _amp_dtype,
    _normalizers,
    cache_lineage,
)

from .data import (
    TargetAwareBalancedBatchSampler,
    manifest_fingerprint,
    move_batch_to_device,
    sha256_file,
)
from .losses import sealed_residual_loss
from .model import OptionalStructureSealedResidualDTA, load_sealed_v22_backbone
from .training import (
    VERSION,
    _parameter_groups,
    _scheduler,
    _selection_score,
    atomic_torch_save,
    build_loader,
    predict,
    save_predictions,
    seed_everything,
)


def _update_residual_ema(ema_model, model, decay):
    """Update only V29's trainable residual branch.

    The V22 backbone is a sealed numerical reference, not merely a module with
    ``requires_grad=False``.  Applying an ordinary floating-point EMA to its
    unchanged parameters still introduces rounding drift, which is amplified
    by BF16 inference.  Therefore backbone parameters and buffers are left
    bitwise untouched in the EMA copy.
    """
    raw_parameters = dict(model.named_parameters())
    ema_parameters = dict(ema_model.named_parameters())
    if raw_parameters.keys() != ema_parameters.keys():
        raise RuntimeError("V29 raw/EMA parameter names differ")
    raw_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    if raw_buffers.keys() != ema_buffers.keys():
        raise RuntimeError("V29 raw/EMA buffer names differ")
    with torch.no_grad():
        for name, parameter in raw_parameters.items():
            if name.startswith("backbone."):
                continue
            ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for name, buffer in raw_buffers.items():
            if name.startswith("backbone."):
                continue
            ema_buffers[name].copy_(buffer)


def _reseal_backbone(model, sealed_v22):
    """Restore the exact V22 state after loading any V29 artifact."""
    model.backbone.load_state_dict(sealed_v22["model_state"], strict=True)
    model.freeze_backbone()


def _reseal_checkpoint_state(checkpoint, sealed_v22):
    """Repair a legacy V29 checkpoint whose EMA had rounded the V22 state."""
    state = checkpoint["model_state"]
    changed = False
    for name, value in sealed_v22["model_state"].items():
        key = "backbone." + name
        if key not in state:
            raise ValueError("V29 checkpoint lacks sealed backbone tensor %s" % key)
        if not torch.equal(state[key], value):
            state[key] = value.detach().cpu().clone()
            changed = True
    return changed


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


def _validate_v22_lineage(checkpoint, train_dataset, valid_dataset, protein_cache, ligand_cache):
    if checkpoint.get("train_manifest_sha256") != manifest_fingerprint(train_dataset.manifest_path):
        raise ValueError("V29 training manifest differs from sealed V22 lineage")
    if checkpoint.get("valid_manifest_sha256") != manifest_fingerprint(valid_dataset.manifest_path):
        raise ValueError("V29 validation manifest differs from sealed V22 lineage")
    if checkpoint.get("protein_cache") != cache_lineage(protein_cache):
        raise ValueError("V29 protein cache differs from sealed V22 lineage")
    if checkpoint.get("ligand_cache") != cache_lineage(ligand_cache):
        raise ValueError("V29 ligand cache differs from sealed V22 lineage")


def fit(
    train_dataset,
    valid_dataset,
    output,
    config,
    protein_cache,
    ligand_cache,
    sealed_v22_checkpoint=None,
    resume=False,
):
    seed_everything(config.seed)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("V29 output exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    if sealed_v22_checkpoint is None:
        raise ValueError("V29 requires the sealed V22 structure checkpoint")
    device = torch.device(config.device)
    model = OptionalStructureSealedResidualDTA(**config.model_kwargs()).to(device)
    sealed_v22 = load_sealed_v22_backbone(model, sealed_v22_checkpoint)
    _validate_v22_lineage(sealed_v22, train_dataset, valid_dataset, protein_cache, ligand_cache)
    label_stats = sealed_v22["label_statistics"]
    means, standard_deviations = _normalizers(label_stats, device)
    sources = {
        "sealed_v22": {
            "path": str(Path(sealed_v22_checkpoint).expanduser().resolve()),
            "sha256": sha256_file(sealed_v22_checkpoint),
            "version": sealed_v22["version"],
            "phase": sealed_v22["phase"],
            "train_manifest_sha256": sealed_v22["train_manifest_sha256"],
            "valid_manifest_sha256": sealed_v22["valid_manifest_sha256"],
        },
        "backbone_policy": "V22 structure model is strict-loaded, eval-only, requires_grad=false, and forward runs under no_grad; EMA updates only V29 residual tensors and never the V22 backbone",
    }
    ema_model = deepcopy(model).to(device).eval()
    ema_model.freeze_backbone()
    optimizer = torch.optim.AdamW(_parameter_groups(model, config), weight_decay=config.weight_decay)
    sampler = TargetAwareBalancedBatchSampler(
        train_dataset,
        config.batch_size,
        config.steps_per_epoch,
        {"LP_PDBBind": 1.0, "DAVIS": 0.0, "KIBA": 0.0},
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
            raise ValueError("V29 resume lineage/config differs")
        model.load_state_dict(state["model_state"], strict=True)
        _reseal_backbone(model, sealed_v22)
        ema_model.load_state_dict(state["ema_state"], strict=True)
        _reseal_backbone(ema_model, sealed_v22)
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
            moved = move_batch_to_device(batch, device)
            normalized = (
                moved["labels"] - means[moved["dataset_index"]]
            ) / standard_deviations[moved["dataset_index"]]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                output_value = model(moved)
                loss, components = sealed_residual_loss(
                    output_value, normalized, moved, loss_config
                )
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V29 training loss")
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % config.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.gradient_clip,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite V29 gradient")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                _update_residual_ema(ema_model, model, config.ema_decay)
                ema_model.freeze_backbone()
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
            "backbone_trainable_parameter_count": int(
                sum(int(parameter.numel()) for parameter in model.backbone.parameters() if parameter.requires_grad)
            ),
            "residual_trainable_parameter_count": int(model.trainable_parameter_count()),
            "train": {name: value / max(1, batches) for name, value in sums.items()},
            "raw": raw["metrics"],
            "ema": ema["metrics"],
            "raw_backbone": raw["backbone_metrics"],
            "ema_backbone": ema["backbone_metrics"],
            "selected_variant": variant,
            "selection_score": list(score),
        }
        history.append(row)
        print(__import__("json").dumps({"event": "v29_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > tuple(best["score"]):
            chosen = model if variant == "raw" else ema_model
            _reseal_backbone(chosen, sealed_v22)
            checkpoint = {
                "schema_version": 1,
                "version": VERSION,
                "phase": "sealed_residual",
                "config": asdict(config),
                "model_kwargs": config.model_kwargs(),
                "model_state": {
                    key: value.detach().cpu() for key, value in chosen.state_dict().items()
                },
                "label_statistics": label_stats,
                "epoch": int(epoch),
                "variant": variant,
                "validation_metrics": selected["metrics"],
                "validation_base_metrics": selected["base_metrics"],
                "validation_backbone_metrics": selected["backbone_metrics"],
                "selection_rule": "validation LP CI, then LP Spearman, negative LP RMSE, negative LP MAE",
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
        raise RuntimeError("V29 training did not produce a checkpoint")
    checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
    if _reseal_checkpoint_state(checkpoint, sealed_v22):
        atomic_torch_save(checkpoint, output / "best.pt")
    selected_model = OptionalStructureSealedResidualDTA(**checkpoint["model_kwargs"]).to(device)
    selected_model.load_state_dict(checkpoint["model_state"], strict=True)
    _reseal_backbone(selected_model, sealed_v22)
    valid_result = predict(selected_model, valid_dataset, config, checkpoint["label_statistics"])
    save_predictions(output / "valid_predictions.npz", valid_result)
    return selected_model, checkpoint, valid_result, history


__all__ = ["fit"]
