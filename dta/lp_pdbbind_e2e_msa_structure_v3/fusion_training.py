"""Joint Stage 2 training plus explicitly historical head-only helpers."""

from __future__ import annotations

from copy import deepcopy
import math
import json

import numpy as np
import torch
from torch.nn import functional as F

from dta.msa_interface_pair_pooling_v18 import training as v18_training
from dta.rank_fusion_v4.metrics import affinity_metrics


PROMOTION_METRICS = ("CI", "RMSE", "MAE", "Pearson", "Spearman")
LOWER_IS_BETTER = frozenset(("RMSE", "MAE"))
INFERENCE_AGGREGATION = "per_subset_full_prediction_then_mean"
FUSION_HEAD_MODES = ("train", "frozen_zero")


def sealed_inference_config(config):
    """Return every setting that defines the v3 evaluation/deployment path."""

    return {
        "aggregation": INFERENCE_AGGREGATION,
        "subset_policy": "fixed_eval_subset_index_per_full_forward_v1",
        "inference_subsets": int(config.inference_subsets),
        "row_depth": int(config.row_depth),
        "evaluation_batch_size": int(config.evaluation_batch_size),
        "amp_dtype": str(config.amp_dtype),
        "output_scale": "train_label_mean_plus_std_times_normalized_prediction",
    }


def validate_feature_arrays(features, labels, label):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("%s feature matrix and labels do not align" % label)
    if features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError("%s feature matrix is too small" % label)
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("%s contains non-finite values" % label)
    return features, labels


def validation_gate(candidate, baseline, tolerance=1e-6):
    missing = [name for name in PROMOTION_METRICS if name not in baseline]
    if missing:
        raise ValueError("baseline metrics are missing: %s" % ", ".join(missing))
    comparisons = {}
    for name in PROMOTION_METRICS:
        if name not in candidate or not np.isfinite(float(candidate[name])):
            comparisons[name] = False
            continue
        if name in LOWER_IS_BETTER:
            comparisons[name] = float(candidate[name]) <= float(baseline[name]) + float(tolerance)
        else:
            comparisons[name] = float(candidate[name]) + float(tolerance) >= float(baseline[name])
    return {
        "passed": bool(all(comparisons.values())),
        "criteria": {
            "require_no_regression_on": list(PROMOTION_METRICS),
            "metric_tolerance": float(tolerance),
        },
        "comparisons": comparisons,
        "delta": {
            name: float(candidate[name]) - float(baseline[name])
            for name in PROMOTION_METRICS
            if name in candidate
        },
    }


def fit_fusion_head(
    model,
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    label_mean,
    label_std,
    *,
    baseline_metrics=None,
    device="cpu",
    learning_rate=3e-4,
    weight_decay=1e-4,
    batch_size=128,
    maximum_epochs=120,
    patience=16,
    delta_l2=1e-4,
    metric_tolerance=1e-6,
    seed=20260925,
):
    """Fit only model.fusion_head; training features must be group-OOF outputs.

    Feature column zero is the normalized core affinity prediction. Both
    training and validation labels are in raw affinity units; the supplied
    normalization must be the full training-manifest label mean/std.
    """

    train_features, train_labels = validate_feature_arrays(
        train_features, train_labels, "training"
    )
    valid_features, valid_labels = validate_feature_arrays(
        valid_features, valid_labels, "validation"
    )
    if train_features.shape[1] != valid_features.shape[1]:
        raise ValueError("training/validation feature widths differ")
    if train_features.shape[1] != model.fusion_head.network[0].normalized_shape[0]:
        raise ValueError("feature width does not match the model fusion head")
    label_mean = float(label_mean)
    label_std = float(label_std)
    if not np.isfinite(label_mean) or not np.isfinite(label_std) or label_std <= 0.0:
        raise ValueError("training label normalization is invalid")
    if maximum_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("training schedule values must be positive")
    if baseline_metrics is not None:
        missing = [name for name in PROMOTION_METRICS if name not in baseline_metrics]
        if missing:
            raise ValueError("baseline metrics are missing: %s" % ", ".join(missing))

    device = torch.device(device)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.unfreeze_fusion_head()
    model.eval()
    model.fusion_head.train()

    train_x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(
        (train_labels - label_mean) / label_std,
        dtype=torch.float32,
        device=device,
    )
    valid_x = torch.as_tensor(valid_features, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.fusion_head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_state = None
    best_metrics = None
    best_score = None
    best_epoch = None
    stale = 0
    history = []

    for epoch in range(int(maximum_epochs)):
        model.fusion_head.train()
        order = torch.randperm(train_x.shape[0], generator=generator).to(device)
        total_loss = 0.0
        count = 0
        for start in range(0, train_x.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch_x = train_x.index_select(0, indices)
            batch_y = train_y.index_select(0, indices)
            residual = model.fusion_head(batch_x)
            prediction = batch_x[:, 0] + residual
            loss = F.smooth_l1_loss(prediction, batch_y, beta=0.5)
            loss = loss + float(delta_l2) * residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite fusion-head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fusion_head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * int(indices.numel())
            count += int(indices.numel())

        model.fusion_head.eval()
        with torch.no_grad():
            valid_residual = model.fusion_head(valid_x).cpu().numpy().astype(np.float64)
        valid_prediction = (
            valid_features[:, 0].astype(np.float64) + valid_residual
        ) * label_std + label_mean
        metrics = affinity_metrics(valid_labels, valid_prediction)
        passed = False
        if baseline_metrics is not None:
            passed = validation_gate(
                metrics, baseline_metrics, tolerance=metric_tolerance
            )["passed"]
        score = (
            int(passed),
            float(metrics["CI"]),
            -float(metrics["RMSE"]),
            -float(metrics["MAE"]),
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(total_loss / max(1, count)),
                "validation_metrics": dict(metrics),
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.fusion_head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break

    if best_state is None:
        raise RuntimeError("fusion-head fitting did not produce a checkpoint")
    model.fusion_head.load_state_dict(best_state, strict=True)
    gate = None
    if baseline_metrics is not None:
        gate = validation_gate(
            best_metrics, baseline_metrics, tolerance=metric_tolerance
        )
    return {
        "model": model,
        "fusion_head_state": deepcopy(best_state),
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
        "validation_gate": gate,
        "validation_selected": bool(gate is not None and gate["passed"]),
        "history": history,
    }


def build_joint_optimizer(
    model,
    *,
    backbone_learning_rate,
    msa_learning_rate,
    fusion_learning_rate,
    weight_decay,
    fusion_head_mode="train",
):
    """Make disjoint groups; the frozen-zero control excludes the fusion head."""

    if fusion_head_mode not in FUSION_HEAD_MODES:
        raise ValueError("invalid fusion head mode: %s" % fusion_head_mode)
    if fusion_head_mode == "frozen_zero":
        final_layer = model.fusion_head.network[-1]
        if torch.count_nonzero(final_layer.weight).item() or torch.count_nonzero(final_layer.bias).item():
            raise ValueError("frozen-zero control requires a zero residual head")
        model.freeze_fusion_head()
    else:
        model.unfreeze_fusion_head()
    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not all_parameters:
        raise ValueError("Stage 2 has no trainable model parameters")
    encoder_parameters = [p for p in model.msa_pair_encoder.parameters() if p.requires_grad]
    fusion_parameters = [p for p in model.fusion_head.parameters() if p.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    if encoder_ids & fusion_ids:
        raise RuntimeError("MSA encoder and fusion head parameter groups overlap")
    backbone_parameters = [
        parameter
        for parameter in all_parameters
        if id(parameter) not in encoder_ids and id(parameter) not in fusion_ids
    ]
    groups = [
        {
            "params": backbone_parameters,
            "lr": float(backbone_learning_rate),
            "group_name": "backbone",
        },
        {
            "params": encoder_parameters,
            "lr": float(msa_learning_rate),
            "group_name": "msa_encoder",
        },
    ]
    if fusion_head_mode == "train":
        groups.append({
            "params": fusion_parameters,
            "lr": float(fusion_learning_rate),
            "group_name": "fusion_head",
        })
    if any(not group["params"] for group in groups):
        raise ValueError("Stage 2 expected non-empty trainable parameter groups")
    group_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("Stage 2 optimizer contains duplicate parameters")
    if set(group_ids) != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError("Stage 2 optimizer does not cover every trainable parameter")
    if any(not math.isfinite(group["lr"]) or group["lr"] <= 0.0 for group in groups):
        raise ValueError("Stage 2 learning rates must be finite and positive")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    summary = [
        {
            "name": group["group_name"],
            "learning_rate": float(group["lr"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "tensor_count": len(group["params"]),
        }
        for group in groups
    ]
    return optimizer, summary


def predict_complete_model(
    model, dataset, config, label_mean, label_std, *, inference_subsets=None
):
    """Use the deployable per-subset full-forward, then mean aggregation."""

    expected_subsets = int(config.inference_subsets)
    if inference_subsets is not None and int(inference_subsets) != expected_subsets:
        raise ValueError("inference subset override differs from sealed checkpoint config")
    result = v18_training.predict_model(
        model,
        dataset,
        config,
        label_mean,
        label_std,
        subsets=expected_subsets,
    )
    result["prediction_aggregation"] = INFERENCE_AGGREGATION
    result["inference_subsets"] = expected_subsets
    return result


def fit_joint_stage2(
    model,
    train_dataset,
    valid_dataset,
    config,
    baseline_metrics,
    label_mean,
    label_std,
    *,
    backbone_learning_rate=1.5e-5,
    fusion_learning_rate=5e-5,
    weight_decay=1e-4,
    batch_size=4,
    maximum_epochs=12,
    patience=5,
    metric_tolerance=1e-6,
    seed=20260925,
    train_loader_factory=None,
    predictor=None,
    fusion_head_mode="train",
):
    """Warm-start Stage 1 for joint training or the matched frozen-head control."""

    if maximum_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("Stage 2 schedule values must be positive")
    label_mean = float(label_mean)
    label_std = float(label_std)
    train_labels = np.asarray(train_dataset.labels, dtype=np.float64)
    if train_labels.size != len(train_dataset) or not np.isfinite(train_labels).all():
        raise ValueError("Stage 2 training labels are invalid")
    observed_mean = float(train_labels.mean())
    observed_std = float(train_labels.std())
    if not np.isclose(observed_mean, label_mean, rtol=0.0, atol=1e-8):
        raise ValueError("Stage 1 label mean differs from the current training manifest")
    if not np.isclose(observed_std, label_std, rtol=0.0, atol=1e-8) or label_std <= 0.0:
        raise ValueError("Stage 1 label std differs from the current training manifest")
    if any(name not in baseline_metrics for name in PROMOTION_METRICS):
        raise ValueError("V20 baseline is missing one or more promotion metrics")

    # Seed before constructing the sampler, scheduler, EMA copy, or taking any
    # stochastic training step. The supplied checkpoint already fixes model
    # initialization, but the interface mask and minibatch order are random.
    v18_training._seed_everything(seed)
    device = torch.device(config.device)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    starting_head_state = {
        name: value.detach().cpu().clone()
        for name, value in model.fusion_head.state_dict().items()
    }
    msa_learning_rate = float(backbone_learning_rate) * float(
        config.encoder_learning_rate_scale
    )
    optimizer, optimizer_summary = build_joint_optimizer(
        model,
        backbone_learning_rate=backbone_learning_rate,
        msa_learning_rate=msa_learning_rate,
        fusion_learning_rate=fusion_learning_rate,
        weight_decay=weight_decay,
        fusion_head_mode=fusion_head_mode,
    )
    loader_factory = train_loader_factory or v18_training._loader
    predictor = predictor or predict_complete_model
    train_loader, spread_sampler = loader_factory(
        train_dataset,
        batch_size,
        config.row_depth,
        training=True,
        labels=train_labels,
        seed=seed,
        minimum_gap=config.minimum_rank_gap * label_std,
        num_workers=config.num_workers,
    )
    total_steps = max(1, int(maximum_epochs) * len(train_loader))
    warmup_steps = max(1, int(0.08 * total_steps))

    def schedule(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_dtype = v18_training._autocast_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    best = None
    history = []
    stale = 0
    optimizer_steps = 0
    component_names = (
        "loss",
        "smooth_l1",
        "mse",
        "rank",
        "contact",
        "residue_contact",
        "affinity_class",
        "distogram",
        "interface_distogram",
        "interface_contact",
        "match",
    )
    for epoch in range(int(maximum_epochs)):
        if spread_sampler is not None:
            spread_sampler.set_epoch(epoch)
        model.train()
        totals = {name: 0.0 for name in component_names}
        batch_count = 0
        for batch in train_loader:
            moved = v18_training._mask_interface_for_training(
                v18_training.move_batch_to_device(batch, device),
                config.interface_mask_probability,
            )
            normalized = (moved["labels"] - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda", dtype=amp_dtype
            ):
                output_value = model(moved)
                # output_value.prediction is anchor + the live residual head.
                loss, components = v18_training.msa_interface_loss(
                    output_value, normalized, moved, config
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite joint Stage 2 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite joint Stage 2 gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            v18_training._update_ema(ema_model, model, config.ema_decay)
            if fusion_head_mode == "frozen_zero":
                # The generic EMA arithmetic can round even unchanged frozen
                # weights. Keep both candidate heads bit-identical to Stage 1.
                ema_model.fusion_head.load_state_dict(model.fusion_head.state_dict())
            optimizer_steps += 1
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value)
            batch_count += 1

        candidates = []
        for variant, candidate_model in (("raw", model), ("ema", ema_model)):
            prediction_result = predictor(
                candidate_model, valid_dataset, config, label_mean, label_std
            )
            metrics = dict(prediction_result["metrics"])
            gate = validation_gate(metrics, baseline_metrics, tolerance=metric_tolerance)
            score = (
                int(gate["passed"]),
                float(metrics["CI"]),
                -float(metrics["RMSE"]),
                -float(metrics["MAE"]),
                float(metrics["Pearson"]),
                float(metrics["Spearman"]),
            )
            candidates.append(
                (score, variant, candidate_model, metrics, gate, prediction_result)
            )
        score, variant_name, selected_model, metrics, gate, prediction_result = max(
            candidates, key=lambda item: item[0]
        )
        row = {
            "epoch": int(epoch),
            "selected_variant": variant_name,
            "train": {
                name: totals[name] / float(max(1, batch_count))
                for name in component_names
            },
            "validation_metrics": metrics,
            "validation_gate": gate,
            "learning_rates": {
                group["group_name"]: float(group["lr"])
                for group in optimizer.param_groups
            },
        }
        history.append(row)
        print(json.dumps({"event": "v3_stage2_epoch", "fusion_head_mode": fusion_head_mode, **row}, sort_keys=True), flush=True)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in selected_model.state_dict().items()
                },
                "epoch": int(epoch),
                "variant": variant_name,
                "metrics": metrics,
                "gate": gate,
                "prediction_result": {
                    "keys": tuple(prediction_result["keys"]),
                    "labels": np.asarray(prediction_result["labels"]).copy(),
                    "predictions": np.asarray(prediction_result["predictions"]).copy(),
                },
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break

    if best is None:
        raise RuntimeError("joint Stage 2 did not produce a selectable checkpoint")
    model.load_state_dict(best["state"], strict=True)
    changed_head_tensors = sum(
        not torch.equal(value.detach().cpu(), starting_head_state[name])
        for name, value in model.fusion_head.state_dict().items()
    )
    if fusion_head_mode == "frozen_zero" and changed_head_tensors:
        raise RuntimeError("frozen-zero control changed the fusion head")
    return {
        "model_state": best["state"],
        "best_epoch": best["epoch"],
        "selected_variant": best["variant"],
        "validation_metrics": best["metrics"],
        "validation_gate": best["gate"],
        "validation_prediction_result": best["prediction_result"],
        "validation_selected": bool(best["gate"]["passed"]),
        "history": history,
        "optimizer_groups": optimizer_summary,
        "fusion_head_mode": fusion_head_mode,
        "fusion_head_changed_tensor_count": int(changed_head_tensors),
        "optimizer_steps": int(optimizer_steps),
        "epochs_completed": len(history),
        "scheduler": "new_warmup_cosine_stage2",
        "optimizer_restored": False,
        "scheduler_restored": False,
        "learning_rate_config": {
            "backbone": float(backbone_learning_rate),
            "msa_encoder": float(msa_learning_rate),
            "fusion_head": float(fusion_learning_rate),
            "weight_decay": float(weight_decay),
        },
    }


__all__ = [
    "PROMOTION_METRICS",
    "FUSION_HEAD_MODES",
    "build_joint_optimizer",
    "fit_joint_stage2",
    "fit_fusion_head",
    "INFERENCE_AGGREGATION",
    "predict_complete_model",
    "sealed_inference_config",
    "validate_feature_arrays",
    "validation_gate",
]
