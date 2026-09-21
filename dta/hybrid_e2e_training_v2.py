"""Packed end-to-end fine-tuning helpers for the upper-bound fusion model."""

import torch


BASE_LEARNING_RATES = {
    "fusion": 3e-5,
    "interface": 3e-6,
    "gerna": 1e-6,
}
LAYERWISE_DECAY = 0.85


def _module_parameter_groups(component, component_name, base_learning_rate, layer_decay):
    named_parameters = list(component.named_parameters())
    if not named_parameters:
        raise ValueError(f"{component_name} has no parameters")
    modules = {}
    for name, parameter in named_parameters:
        parameter.requires_grad = True
        owner = name.rsplit(".", 1)[0] if "." in name else "root"
        modules.setdefault(owner, []).append(parameter)
    owners = list(modules)
    groups = []
    for index, owner in enumerate(owners):
        learning_rate = float(base_learning_rate) * float(layer_decay) ** (
            len(owners) - index - 1
        )
        groups.append(
            {
                "name": f"{component_name}.{owner}",
                "component": component_name,
                "params": modules[owner],
                "lr": learning_rate,
            }
        )
    return groups


def build_e2e_optimizer(
    model,
    *,
    fusion_learning_rate=BASE_LEARNING_RATES["fusion"],
    interface_learning_rate=BASE_LEARNING_RATES["interface"],
    gerna_learning_rate=BASE_LEARNING_RATES["gerna"],
    layer_decay=LAYERWISE_DECAY,
    weight_decay=1e-4,
):
    """Create the all-trainable optimizer used by joint upper-bound tuning."""
    if not 0.0 < float(layer_decay) <= 1.0:
        raise ValueError("layer_decay must be in (0, 1]")
    for name, value in (
        ("fusion_learning_rate", fusion_learning_rate),
        ("interface_learning_rate", interface_learning_rate),
        ("gerna_learning_rate", gerna_learning_rate),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")
    required = ("fusion", "interface", "gerna")
    if any(not hasattr(model, name) for name in required):
        raise ValueError("model must own fusion, interface, and gerna modules")

    for parameter in model.parameters():
        parameter.requires_grad = True
    groups = [
        {
            "name": "fusion",
            "component": "fusion",
            "params": list(model.fusion.parameters()),
            "lr": float(fusion_learning_rate),
        },
        *_module_parameter_groups(
            model.interface,
            "interface",
            interface_learning_rate,
            layer_decay,
        ),
        *_module_parameter_groups(
            model.gerna,
            "gerna",
            gerna_learning_rate,
            layer_decay,
        ),
    ]
    assigned = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in assigned}) != len(assigned):
        raise RuntimeError("end-to-end optimizer parameter groups overlap")
    if {id(parameter) for parameter in assigned} != {
        id(parameter) for parameter in model.parameters()
    }:
        raise RuntimeError("end-to-end optimizer does not cover all parameters")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))



class E2EValidationTracker:
    """Track raw-validation CI improvements and the joint-training patience."""

    def __init__(self, patience):
        self.patience = int(patience)
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        self.raw_best = None
        self.stale_epochs = 0

    @property
    def should_stop(self):
        return self.stale_epochs >= self.patience

    def record_raw(self, metrics):
        from dta.hybrid_cached_training_v2 import is_better_validation

        if is_better_validation(metrics, self.raw_best):
            self.raw_best = dict(metrics)
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    def state_dict(self):
        return {
            "patience": self.patience,
            "raw_best": self.raw_best,
            "stale_epochs": self.stale_epochs,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {
            "patience", "raw_best", "stale_epochs"
        }:
            raise RuntimeError("invalid end-to-end validation tracker state")
        if int(state["patience"]) != self.patience:
            raise RuntimeError("end-to-end validation patience differs")
        self.raw_best = (
            None if state["raw_best"] is None else dict(state["raw_best"])
        )
        self.stale_epochs = int(state["stale_epochs"])



def load_cached_fusion_checkpoint(fusion, checkpoint_path):
    """Strictly transfer a cached-fusion checkpoint before joint tuning."""
    from pathlib import Path

    from utils.experiment_integrity import sha256_file

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    state_dict = (
        payload.get("model_state_dict")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(state_dict, dict):
        raise RuntimeError("cached checkpoint is missing model_state_dict")
    fusion.load_state_dict(state_dict, strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}



def train_e2e_group(
    model,
    optimizer,
    group,
    device,
    *,
    prepare_fn=None,
    scaler=None,
    amp_enabled=False,
    max_grad_norm=1.0,
    ema=None,
):
    """Run one packed joint-training update and return finite diagnostics."""
    from contextlib import nullcontext

    from data_utils.packed_hybrid_batch import prepare_packed_hybrid_group
    from dta.hybrid_affinity_training import NonfiniteTrainingError
    from dta.hybrid_cached_training_v2 import upper_bound_affinity_loss

    samples = list(group)
    if not samples:
        raise ValueError("joint training group must be nonempty")
    prepare_fn = prepare_packed_hybrid_group if prepare_fn is None else prepare_fn
    keys, gerna_inputs, packed_interface, targets = prepare_fn(samples, device)
    if len(keys) != len(samples) or targets.shape != (len(samples),):
        raise ValueError("packed group keys and labels do not align")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )
    try:
        with autocast:
            output = model(gerna_inputs, packed_interface)
            if not isinstance(output, dict) or {
                "prediction", "expert_weights"
            }.difference(output):
                raise RuntimeError("joint model output is missing fusion tensors")
            components = upper_bound_affinity_loss(
                output["prediction"], targets, output["expert_weights"]
            )
            loss = components["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise NonfiniteTrainingError("loss", keys, {"loss": loss})
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        nonfinite = {
            name: gradient
            for name, gradient in gradients.items()
            if not bool(torch.isfinite(gradient).all())
        }
        if nonfinite:
            raise NonfiniteTrainingError("gradient", keys, nonfinite)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                float(max_grad_norm),
            )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if ema is not None:
            ema.update(model)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    result = {
        "keys": list(keys),
        "labels": targets.detach().float().cpu().numpy(),
        "predictions": output["prediction"].detach().float().cpu().numpy(),
        "optimizer_step": True,
    }
    result.update(
        {
            name: float(value.detach().float().cpu())
            for name, value in components.items()
        }
    )
    return result



def evaluate_e2e(model, data_loader, device, *, prepare_fn=None, amp_enabled=False):
    """Evaluate raw or EMA joint weights over packed hybrid groups."""
    from contextlib import nullcontext

    from data_utils.packed_hybrid_batch import prepare_packed_hybrid_group
    from dta.hybrid_cached_training_v2 import _metrics

    prepare_fn = prepare_packed_hybrid_group if prepare_fn is None else prepare_fn
    was_training = model.training
    model.eval()
    keys, labels, predictions = [], [], []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )
    try:
        with torch.inference_mode(), autocast:
            for group in data_loader:
                group_keys, gerna_inputs, packed_interface, targets = prepare_fn(
                    group, device
                )
                output = model(gerna_inputs, packed_interface)
                if not isinstance(output, dict) or "prediction" not in output:
                    raise RuntimeError("joint model output is missing prediction")
                prediction = output["prediction"].reshape(-1).float()
                if prediction.shape != targets.reshape(-1).shape:
                    raise RuntimeError("joint validation prediction and label sizes differ")
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("joint validation prediction is nonfinite")
                keys.extend(str(key) for key in group_keys)
                labels.extend(targets.reshape(-1).float().cpu().tolist())
                predictions.extend(prediction.cpu().tolist())
    finally:
        model.train(was_training)
    if len(keys) != len(set(keys)):
        raise ValueError("joint validation keys must be unique")
    if not keys:
        raise ValueError("joint validation is empty")
    labels = __import__("numpy").asarray(labels, dtype="float64")
    predictions = __import__("numpy").asarray(predictions, dtype="float64")
    return {
        "keys": keys,
        "labels": labels,
        "predictions": predictions,
        "metrics": _metrics(labels, predictions),
    }



def fit_e2e(
    model,
    optimizer,
    train_loader,
    valid_loader,
    device,
    *,
    epochs=10,
    patience=2,
    prepare_fn=None,
    amp_enabled=False,
    scheduler=None,
    ema_decay=0.999,
):
    """Fit joint weights and evaluate EMA only after raw CI improves."""
    import numpy as np

    from dta.hybrid_affinity_training import ModelEMA

    epochs = int(epochs)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    tracker = E2EValidationTracker(patience)
    ema = ModelEMA(model, decay=ema_decay)
    history = []
    ema_best = None
    best_raw_result = None
    best_ema_result = None
    for epoch in range(epochs):
        losses = []
        for group in train_loader:
            result = train_e2e_group(
                model,
                optimizer,
                group,
                device,
                prepare_fn=prepare_fn,
                amp_enabled=amp_enabled,
                ema=ema,
            )
            losses.append(result["loss"])
        if scheduler is not None:
            scheduler.step()
        raw = evaluate_e2e(
            model,
            valid_loader,
            device,
            prepare_fn=prepare_fn,
            amp_enabled=amp_enabled,
        )
        raw_improved = tracker.record_raw(raw["metrics"])
        ema_result = None
        if raw_improved:
            with ema.average_parameters(model):
                ema_result = evaluate_e2e(
                    model,
                    valid_loader,
                    device,
                    prepare_fn=prepare_fn,
                    amp_enabled=amp_enabled,
                )
            if ema_best is None:
                ema_best = dict(ema_result["metrics"])
                best_ema_result = ema_result
            else:
                from dta.hybrid_cached_training_v2 import is_better_validation

                if is_better_validation(ema_result["metrics"], ema_best):
                    ema_best = dict(ema_result["metrics"])
                    best_ema_result = ema_result
        if raw_improved:
            best_raw_result = raw
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "raw": dict(raw["metrics"]),
            "ema_evaluated": bool(raw_improved),
            "ema": None if ema_result is None else dict(ema_result["metrics"]),
            "stale_epochs": tracker.stale_epochs,
        }
        history.append(row)
        if tracker.should_stop:
            break
    return {
        "history": history,
        "best": {
            "raw": None if best_raw_result is None else best_raw_result,
            "ema": best_ema_result,
        },
        "ema": ema,
        "validation": tracker.state_dict(),
    }



def _atomic_torch_save(payload, path):
    import os
    import tempfile
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_e2e_predictions(path, evaluation):
    import numpy as np

    np.savez_compressed(
        path,
        keys=np.asarray(evaluation["keys"]),
        labels=evaluation["labels"],
        predictions=evaluation["predictions"],
    )


def run_e2e_training(args):
    """Train the approved packed joint upper-bound model on CUDA only."""
    import csv
    import json
    import random
    from pathlib import Path

    import numpy as np
    from torch.utils.data import DataLoader, Subset

    from data_utils.hybrid_affinity_dataset import (
        LabelSpreadBatchSampler,
        collate_hybrid_group,
    )
    from dta.hybrid_affinity_training import (
        ModelEMA,
        capture_rng_state,
        restore_rng_state,
    )
    from dta.hybrid_cached_training_v2 import is_better_validation
    from dta.train_hybrid_affinity import (
        build_gerna_backbone,
        build_hybrid_dataset,
        build_interface_backbone,
        load_backbone_checkpoint,
        validate_train_valid_manifests,
    )
    from net.hybrid_upper_bound import (
        HybridUpperBoundEndToEndModel,
        HybridUpperBoundFusion,
    )
    from utils.experiment_integrity import atomic_write_json, manifest_fingerprint, sha256_file

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal end-to-end training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    epochs = int(args.epochs)
    if epochs <= 0 or epochs > 10:
        raise ValueError("epochs must be in [1, 10]")
    group_size = int(args.group_size)
    limit = getattr(args, "limit", None)
    if limit is not None:
        limit = int(limit)
        if limit < 2:
            raise ValueError("limit must be at least two")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError("output is nonempty; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_train_valid_manifests(args.train_manifest, args.valid_manifest)
    torch.cuda.set_device(device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    gerna = build_gerna_backbone()
    interface = build_interface_backbone()
    backbone_artifacts = {
        "gerna": load_backbone_checkpoint(gerna, args.gerna_checkpoint),
        "interface": load_backbone_checkpoint(
            interface, args.interface_checkpoint
        ),
    }
    fusion = HybridUpperBoundFusion(dropout=float(args.dropout))
    cached_artifact = load_cached_fusion_checkpoint(
        fusion, args.cached_checkpoint
    )
    model = HybridUpperBoundEndToEndModel(gerna, interface, fusion).to(device)
    optimizer = build_e2e_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    amp_enabled = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    ema = ModelEMA(model, decay=float(args.ema_decay))
    tracker = E2EValidationTracker(args.patience)

    train_dataset = build_hybrid_dataset(
        args.train_manifest, args.train_native_cache, expected_split="train"
    )
    valid_dataset = build_hybrid_dataset(
        args.valid_manifest, args.valid_native_cache, expected_split="valid"
    )
    train_labels = train_dataset.labels
    if limit is not None:
        if limit > len(train_dataset) or limit > len(valid_dataset):
            raise ValueError("limit exceeds an available split")
        train_dataset = Subset(train_dataset, range(limit))
        valid_dataset = Subset(valid_dataset, range(limit))
        train_labels = train_labels[:limit]
    train_sampler = LabelSpreadBatchSampler(
        train_labels, group_size, int(args.seed), min_gap=0.05
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_hybrid_group,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=group_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_hybrid_group,
        pin_memory=True,
    )
    provenance = {
        "train_manifest": manifest_fingerprint(args.train_manifest),
        "valid_manifest": manifest_fingerprint(args.valid_manifest),
        "backbones": backbone_artifacts,
        "cached_fusion": cached_artifact,
        "sample_limit": limit,
    }
    checkpoint_path = output_dir / "training_checkpoint.pth"
    history = []
    start_epoch = 0
    best_raw = None
    best_ema = None
    if args.resume and checkpoint_path.is_file():
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if state.get("provenance") != provenance:
            raise RuntimeError("resume checkpoint provenance does not match")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(state["scaler"])
        ema.load_state_dict(state["ema"])
        tracker.load_state_dict(state["validation"])
        restore_rng_state(state["rng"])
        start_epoch = int(state["epoch"]) + 1
        history = list(state["history"])
        best_raw = state.get("best_raw")
        best_ema = state.get("best_ema")

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        losses = []
        for group in train_loader:
            result = train_e2e_group(
                model,
                optimizer,
                group,
                device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                ema=ema,
            )
            losses.append(result["loss"])
        scheduler.step()
        raw = evaluate_e2e(
            model, valid_loader, device, amp_enabled=amp_enabled
        )
        raw_improved = tracker.record_raw(raw["metrics"])
        if raw_improved:
            best_raw = dict(raw["metrics"])
            _write_e2e_predictions(output_dir / "valid_predictions_raw.npz", raw)
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": best_raw,
                    "epoch": epoch,
                    "provenance": provenance,
                    "variant": "raw",
                },
                output_dir / "Model_e2e_raw_best.pth",
            )
        ema_metrics = None
        if raw_improved:
            with ema.average_parameters(model):
                ema_result = evaluate_e2e(
                    model, valid_loader, device, amp_enabled=amp_enabled
                )
                ema_metrics = dict(ema_result["metrics"])
                if best_ema is None or is_better_validation(
                    ema_metrics, best_ema
                ):
                    best_ema = ema_metrics
                    _write_e2e_predictions(
                        output_dir / "valid_predictions_ema.npz", ema_result
                    )
                    _atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "metrics": best_ema,
                            "epoch": epoch,
                            "provenance": provenance,
                            "variant": "ema",
                        },
                        output_dir / "Model_e2e_ema_best.pth",
                    )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "raw": dict(raw["metrics"]),
            "ema": ema_metrics,
            "ema_evaluated": bool(raw_improved),
            "stale_epochs": tracker.stale_epochs,
        }
        history.append(row)
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "ema": ema.state_dict(),
                "validation": tracker.state_dict(),
                "rng": capture_rng_state(),
                "epoch": epoch,
                "history": history,
                "best_raw": best_raw,
                "best_ema": best_ema,
                "provenance": provenance,
            },
            checkpoint_path,
        )
        if tracker.should_stop:
            break

    _atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": tracker.raw_best,
            "provenance": provenance,
            "variant": "last",
        },
        output_dir / "Model_e2e_last.pth",
    )
    atomic_write_json(output_dir / "best_metrics.json", {
        "raw": best_raw,
        "ema": best_ema,
        "raw_validation": tracker.state_dict(),
    })
    atomic_write_json(output_dir / "run_config.json", {
        "seed": int(args.seed),
        "device": str(device),
        "epochs": epochs,
        "group_size": group_size,
        "limit": limit,
        "amp": amp_enabled,
        "ema_decay": float(args.ema_decay),
    })
    atomic_write_json(output_dir / "provenance.json", provenance)
    for variant, predictions, checkpoint in (
        ("raw", "valid_predictions_raw.npz", "Model_e2e_raw_best.pth"),
        ("ema", "valid_predictions_ema.npz", "Model_e2e_ema_best.pth"),
    ):
        if (output_dir / predictions).is_file() and (output_dir / checkpoint).is_file():
            atomic_write_json(output_dir / f"candidate_{variant}.json", {
                "split": "valid",
                "model_sha256": sha256_file(output_dir / checkpoint),
                "valid_manifest_sha256": provenance["valid_manifest"]["sha256"],
                "variant": variant,
            })
    with (output_dir / "metrics_history.json").open("w") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
    with (output_dir / "metrics_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "raw_CI", "raw_Spearman", "raw_RMSE", "ema_CI", "ema_evaluated", "stale_epochs"))
        for row in history:
            writer.writerow((
                row["epoch"], row["train_loss"], row["raw"]["CI"],
                row["raw"]["Spearman"], row["raw"]["RMSE"],
                None if row["ema"] is None else row["ema"]["CI"],
                row["ema_evaluated"], row["stale_epochs"],
            ))
    return {"raw": best_raw, "ema": best_ema, "history": history}
