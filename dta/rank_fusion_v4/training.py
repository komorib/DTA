"""One-seed Rank Fusion v4 training and full-validation evaluation."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.experiment_integrity import atomic_write_json, sha256_file

from .data import (
    FusionCacheDataset,
    LabelSpreadBatchSampler,
    collate_fusion_cache,
)
from .losses import rank_fusion_loss
from .metrics import affinity_metrics


class NonfiniteTrainingError(RuntimeError):
    pass


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _move_batch(batch, device):
    return {
        "keys": batch["keys"],
        "labels": batch["labels"].to(device, non_blocking=True),
        "raw_predictions": batch["raw_predictions"].to(
            device, non_blocking=True
        ),
        "standardized_predictions": batch["standardized_predictions"].to(
            device, non_blocking=True
        ),
        "available": batch["available"].to(device, non_blocking=True),
        "latents": [
            value.to(device, non_blocking=True) for value in batch["latents"]
        ],
    }


def _tensor_statistics(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    selected = value[finite]
    return {
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "finite_count": int(finite.sum()),
        "minimum": float(selected.min()) if selected.numel() else None,
        "maximum": float(selected.max()) if selected.numel() else None,
    }


def _batch_tensors(batch):
    yield "labels", batch["labels"]
    yield "raw_predictions", batch["raw_predictions"]
    yield "standardized_predictions", batch["standardized_predictions"]
    for index, latent in enumerate(batch["latents"]):
        yield f"latent_{index}", latent


def _raise_nonfinite(stage, epoch, batch, tensors):
    diagnostics = {
        name: _tensor_statistics(value) for name, value in tensors
    }
    raise NonfiniteTrainingError(
        f"nonfinite {stage}; epoch={epoch}; keys={','.join(batch['keys'])}; "
        + json.dumps(diagnostics, sort_keys=True)
    )


def _validate_finite_batch(batch, epoch):
    bad = [
        (name, value)
        for name, value in _batch_tensors(batch)
        if not bool(torch.isfinite(value).all())
    ]
    if bad:
        _raise_nonfinite("features", epoch, batch, bad)


def _update_digest(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def cache_identity(cache):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(cache.keys), separators=(",", ":")).encode("utf-8"))
    for value in (
        cache.labels,
        cache.predictions,
        cache.standardized_predictions,
        cache.available,
        cache.upper_bound_latents,
        cache.boltz_latents,
        cache.three_dprot_latents,
    ):
        _update_digest(digest, value)
    return digest.hexdigest()


def _model_config(model):
    return {
        "class": type(model).__name__,
        "latent_dims": list(model.latent_dims),
        "model_dim": int(model.model_dim),
        "layers": len(model.blocks),
        "heads": int(model.blocks[0].attention.num_heads),
        "dropout": float(model.blocks[0].attention.dropout),
        "drop_path_rate": float(model.blocks[-1].drop_path.probability),
    }


def _run_config(
    model,
    train_cache,
    valid_cache,
    seed,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
):
    train_normalization = train_cache.metadata.get("normalization_sha256")
    valid_normalization = valid_cache.metadata.get("normalization_sha256")
    if not train_normalization or train_normalization != valid_normalization:
        raise ValueError("train and validation normalization hashes do not match")
    return {
        "seed": int(seed),
        "train_cache_sha256": cache_identity(train_cache),
        "valid_cache_sha256": cache_identity(valid_cache),
        "train_manifest_sha256": train_cache.metadata.get("manifest_sha256"),
        "valid_manifest_sha256": valid_cache.metadata.get("manifest_sha256"),
        "normalization_sha256": train_normalization,
        "architecture": _model_config(model),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_epochs": 5,
        "schedule": "linear_warmup_cosine",
    }


def _atomic_torch_save(path, payload):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=".predictions-", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _epoch_learning_rate(base, epoch, total_epochs, warmup=5):
    if epoch < warmup:
        return float(base) * float(epoch + 1) / float(warmup)
    remaining = max(1, int(total_epochs) - int(warmup))
    progress = min(1.0, float(epoch - warmup) / float(remaining))
    return float(base) * 0.5 * (1.0 + math.cos(math.pi * progress))


def is_better_validation(candidate, current, tolerance=1e-12):
    if current is None:
        return True
    for name, direction in (("CI", 1.0), ("Spearman", 1.0), ("RMSE", -1.0)):
        delta = direction * (float(candidate[name]) - float(current[name]))
        if delta > tolerance:
            return True
        if delta < -tolerance:
            return False
    return False


@torch.inference_mode()
def evaluate_rank_fusion(model, cache, device, batch_size=64):
    device = torch.device(device)
    loader = DataLoader(
        FusionCacheDataset(cache),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fusion_cache,
    )
    model = model.to(device).eval()
    keys, predictions, mixture_weights, residuals = [], [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        _validate_finite_batch(batch, epoch=-1)
        output = model(
            batch["latents"],
            batch["raw_predictions"],
            batch["standardized_predictions"],
            batch["available"],
        )
        if not all(
            bool(torch.isfinite(output[name]).all())
            for name in ("prediction", "mixture_weights", "rank_residual")
        ):
            _raise_nonfinite(
                "validation output",
                -1,
                batch,
                [(name, output[name]) for name in output],
            )
        keys.extend(batch["keys"])
        predictions.append(output["prediction"].float().cpu().numpy())
        mixture_weights.append(output["mixture_weights"].float().cpu().numpy())
        residuals.append(output["rank_residual"].float().cpu().numpy())
    prediction_array = np.concatenate(predictions).astype(np.float32, copy=False)
    return {
        "keys": tuple(keys),
        "labels": cache.labels.copy(),
        "predictions": prediction_array,
        "expert_predictions": cache.predictions.copy(),
        "mixture_weights": np.concatenate(mixture_weights).astype(
            np.float32, copy=False
        ),
        "rank_residuals": np.concatenate(residuals).astype(np.float32, copy=False),
        "metrics": affinity_metrics(cache.labels, prediction_array),
    }


def _validate_resume_config(actual, expected):
    if actual == expected:
        return
    if actual.get("normalization_sha256") != expected.get("normalization_sha256"):
        raise ValueError("resume normalization hash does not match")
    differing = sorted(
        key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
    )
    raise ValueError("resume configuration does not match: " + ", ".join(differing))


def fit_one_seed(
    model,
    train_cache,
    valid_cache,
    output_dir,
    *,
    seed,
    device,
    epochs=100,
    patience=12,
    batch_size=64,
    learning_rate=2e-4,
    weight_decay=1e-3,
    resume=False,
):
    """Train one seed, early-stop on full validation, and return metadata."""
    for value, name in ((epochs, "epochs"), (patience, "patience"), (batch_size, "batch_size")):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    config = _run_config(
        model,
        train_cache,
        valid_cache,
        seed,
        epochs,
        patience,
        batch_size,
        learning_rate,
        weight_decay,
    )
    output_path = Path(output_dir).expanduser().resolve()
    summary_path = output_path / "summary.json"
    checkpoint_last = output_path / "checkpoint_last.pt"
    if output_path.exists() and any(output_path.iterdir()) and not resume:
        raise FileExistsError("output is nonempty; pass resume=True")
    output_path.mkdir(parents=True, exist_ok=True)
    if resume and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") != "complete":
            raise ValueError("existing summary is incomplete")
        _validate_resume_config(summary.get("run_config", {}), config)
        checkpoint_best = output_path / "checkpoint_best.pt"
        if summary.get("best_checkpoint_sha256") != sha256_file(checkpoint_best):
            raise ValueError("completed best checkpoint hash does not match")
        return summary

    _seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    sampler = LabelSpreadBatchSampler(train_cache.labels, int(batch_size), int(seed))
    loader = DataLoader(
        FusionCacheDataset(train_cache),
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fusion_cache,
    )
    start_epoch, stale, best, history = 0, 0, None, []
    if resume and checkpoint_last.is_file():
        state = torch.load(str(checkpoint_last), map_location="cpu")
        _validate_resume_config(state.get("run_config", {}), config)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        stale = int(state["stale"])
        best = state["best"]
        history = list(state["history"])

    baseline = evaluate_rank_fusion(
        model, valid_cache, device, batch_size=int(batch_size)
    )
    for epoch in range(start_epoch, int(epochs)):
        learning_rate_epoch = _epoch_learning_rate(
            learning_rate, epoch, int(epochs)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_epoch
        sampler.set_epoch(epoch)
        model.train()
        totals = {}
        batches = 0
        for batch in loader:
            batch = _move_batch(batch, device)
            _validate_finite_batch(batch, epoch)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                output = model(
                    batch["latents"],
                    batch["raw_predictions"],
                    batch["standardized_predictions"],
                    batch["available"],
                )
                components = rank_fusion_loss(
                    output["prediction"],
                    batch["labels"],
                    batch["raw_predictions"],
                    output["mixture_weights"],
                    batch["available"],
                )
            if not all(bool(torch.isfinite(value)) for value in components.values()):
                _raise_nonfinite("loss", epoch, batch, list(components.items()))
            scaler.scale(components["loss"]).backward()
            scaler.unscale_(optimizer)
            bad_gradients = [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradients:
                _raise_nonfinite("gradient", epoch, batch, bad_gradients)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                _raise_nonfinite(
                    "gradient norm", epoch, batch, [("gradient_norm", gradient_norm)]
                )
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

        validation = evaluate_rank_fusion(
            model, valid_cache, device, batch_size=int(batch_size)
        )
        losses = {name: value / batches for name, value in totals.items()}
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate_epoch,
            "losses": losses,
            "validation_metrics": validation["metrics"],
        }
        history.append(row)
        if is_better_validation(validation["metrics"], None if best is None else best["metrics"]):
            stale = 0
            _atomic_torch_save(
                output_path / "checkpoint_best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config["architecture"],
                    "epoch": epoch,
                    "metrics": validation["metrics"],
                    "run_config": config,
                },
            )
            _atomic_npz(
                output_path / "valid_predictions_best.npz",
                keys=np.asarray(validation["keys"], dtype=np.str_),
                labels=validation["labels"],
                predictions=validation["predictions"],
                expert_predictions=validation["expert_predictions"],
                mixture_weights=validation["mixture_weights"],
                rank_residuals=validation["rank_residuals"],
            )
            atomic_write_json(output_path / "best_metrics.json", validation["metrics"])
            best = {"epoch": epoch, "metrics": validation["metrics"]}
        else:
            stale += 1
        _atomic_torch_save(
            checkpoint_last,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "stale": stale,
                "best": best,
                "history": history,
                "run_config": config,
            },
        )
        if stale >= int(patience):
            break

    if best is None:
        raise RuntimeError("training completed without a validation checkpoint")
    summary = {
        "status": "complete",
        "seed": int(seed),
        "run_config": config,
        "baseline_validation_metrics": baseline["metrics"],
        "validation_count": len(valid_cache.keys),
        "history": history,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "best_checkpoint_sha256": sha256_file(output_path / "checkpoint_best.pt"),
        "stop_reason": "patience" if stale >= int(patience) else "epochs",
    }
    atomic_write_json(summary_path, summary)
    return summary
