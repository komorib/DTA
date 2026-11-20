"""Run online, same-trunk joint Stage 2 and the V20 validation gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from dta.msa_interface_pair_pooling_v18.dataset import MSAPairComplexDataset
from dta.msa_interface_pair_pooling_v18.training import TrainingConfig
from utils.experiment_integrity import sha256_file

from . import VERSION
from .fusion_training import (
    FUSION_HEAD_MODES,
    PROMOTION_METRICS,
    fit_joint_stage2,
    predict_complete_model,
    sealed_inference_config,
    validation_gate,
)
from .model import E2EMSAInterfacePairFusionV3, architecture_summary


DEFAULT_V20_SELECTION = (
    "Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/"
    "msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000_"
    "validation_selection/result.json"
)


def _atomic_torch_save(value, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".pt.tmp")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_archive(path, required, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing %s archive: %s" % (label, path))
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("%s archive is missing: %s" % (label, ", ".join(missing)))
        return {name: np.asarray(archive[name]) for name in required}


def _manifest_rows(path, expected_split):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("split") != expected_split:
        raise ValueError("manifest split mismatch: expected %s" % expected_split)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest has no samples: %s" % path)
    rows = {}
    for sample in samples:
        key = str(sample.get("lmdb_key", ""))
        if not key or key in rows:
            raise ValueError("manifest keys are empty or duplicated")
        rows[key] = float(sample["label"])
    if not np.isfinite(np.asarray(list(rows.values()), dtype=np.float64)).all():
        raise ValueError("manifest labels contain non-finite values")
    return path, rows


def _validate_archive_identity(archive, expected_rows, label):
    keys = np.asarray(archive["keys"]).astype(str).reshape(-1)
    labels = np.asarray(archive["labels"], dtype=np.float64).reshape(-1)
    if keys.shape != labels.shape or np.unique(keys).size != keys.size:
        raise ValueError("%s artifact keys/labels are malformed" % label)
    if set(keys.tolist()) != set(expected_rows):
        raise ValueError("%s artifact keys differ from the validation manifest" % label)
    by_key = dict(zip(keys.tolist(), labels.tolist()))
    if any(
        not np.isfinite(by_key[key])
        or not np.isclose(by_key[key], value, rtol=0.0, atol=1e-6)
        for key, value in expected_rows.items()
    ):
        raise ValueError("%s artifact labels differ from the validation manifest" % label)
    return {"count": len(keys), "key_count": len(keys)}


def _load_v20_baseline(path, valid_rows):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing V20 validation selection: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    if baseline.get("version") != "msa-interface-pair-latent-stability-fusion-v20":
        raise ValueError("baseline is not the selected V20 result")
    if baseline.get("status") != "validation_passed" or not baseline.get(
        "validation_selected"
    ):
        raise ValueError("V20 validation selection did not pass")
    metrics = baseline.get("selected_validation_metrics")
    if not isinstance(metrics, dict) or any(
        name not in metrics or not np.isfinite(float(metrics[name]))
        for name in PROMOTION_METRICS
    ):
        raise ValueError("V20 validation selection has invalid metrics")
    checked = _validate_v20_validation_sources(path, valid_rows)
    return path, metrics, checked, baseline


def _validate_v20_validation_sources(baseline_path, valid_rows, valid_labels=None):
    if valid_labels is not None:
        keys = np.asarray(valid_rows).astype(str).reshape(-1)
        labels = np.asarray(valid_labels, dtype=np.float64).reshape(-1)
        if keys.shape != labels.shape or np.unique(keys).size != keys.size:
            raise ValueError("validation keys/labels are malformed")
        valid_rows = dict(zip(keys.tolist(), labels.tolist()))
    with Path(baseline_path).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    inputs = baseline.get("inputs", {})
    checked = {}
    for name in ("v16_valid", "v19_benchmark_valid", "v19_features_valid"):
        source = inputs.get(name)
        source_hash = inputs.get(name + "_sha256")
        if not source or not source_hash:
            raise ValueError("V20 baseline lacks hashed %s provenance" % name)
        source_path = Path(source).expanduser().resolve()
        if sha256_file(source_path) != source_hash:
            raise ValueError("V20 baseline %s artifact hash mismatch" % name)
        archive = _load_archive(source_path, ("keys", "labels"), name)
        checked[name] = {
            "path": str(source_path),
            "sha256": source_hash,
            **_validate_archive_identity(archive, valid_rows, name),
        }
    return checked


def _load_stage1(stage1_checkpoint_path):
    checkpoint_path = Path(stage1_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("missing v3 Stage 1 checkpoint: %s" % checkpoint_path)
    selection_path = checkpoint_path.parent / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("missing Stage 1 selection record: %s" % selection_path)
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if (
        selection.get("version") != VERSION
        or selection.get("training_protocol") != "same_trunk_two_stage_v1"
        or selection.get("training_stage") != "stage1_backbone_warmup"
        or selection.get("status") != "stage1_complete"
    ):
        raise ValueError("checkpoint is not a completed v3 Stage 1 warm-up")
    checkpoint_hash = sha256_file(checkpoint_path)
    if selection.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Stage 1 checkpoint hash differs from its selection record")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if (
        checkpoint.get("model_version") != VERSION
        or checkpoint.get("training_protocol") != "same_trunk_two_stage_v1"
        or checkpoint.get("training_stage") != "stage1_backbone_warmup"
    ):
        raise ValueError("Stage 1 checkpoint metadata is invalid")
    config = TrainingConfig(**checkpoint["config"])
    model_kwargs = dict(checkpoint.get("model_kwargs") or config.model_kwargs())
    if model_kwargs != config.model_kwargs():
        raise ValueError("Stage 1 checkpoint model_kwargs differ from its saved config")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Stage 1 checkpoint has no complete model_state")
    stage1_training = checkpoint.get("stage1_training", {})
    if (
        stage1_training.get("method")
        != "supervised_backbone_warmup_frozen_zero_residual_head"
        or stage1_training.get("optimizer_state_saved") is not False
    ):
        raise ValueError("Stage 1 checkpoint lacks the expected warm-up provenance")
    model = E2EMSAInterfacePairFusionV3(**model_kwargs).cpu()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    final_fusion_layer = model.fusion_head.network[-1]
    if torch.count_nonzero(final_fusion_layer.weight).item() or torch.count_nonzero(
        final_fusion_layer.bias
    ).item():
        raise ValueError("Stage 1 checkpoint residual head is not zero initialized")
    inference_config = checkpoint.get("inference_config", {})
    expected_inference = sealed_inference_config(config)
    if inference_config != expected_inference:
        raise ValueError("Stage 1 prediction aggregation config is invalid")
    if selection.get("inference_config") != expected_inference:
        raise ValueError("Stage 1 selection prediction config is invalid")
    if selection.get("selected_variant") != checkpoint.get("variant"):
        raise ValueError("Stage 1 checkpoint variant differs from its selection record")
    return checkpoint_path, checkpoint_hash, checkpoint, selection, config, model


def _verify_stage1_validation_forward(
    model, dataset, config, checkpoint, selection, valid_rows
):
    """Check that the full checkpoint reproduces its saved Stage 1 validation."""

    prediction_path = selection.get("valid_predictions")
    prediction_sha = selection.get("valid_predictions_sha256")
    if not prediction_path or not prediction_sha:
        raise ValueError("Stage 1 selection lacks hashed validation predictions")
    prediction_path = Path(prediction_path).expanduser().resolve()
    if sha256_file(prediction_path) != prediction_sha:
        raise ValueError("Stage 1 validation prediction hash mismatch")
    archive = _load_archive(
        prediction_path, ("keys", "labels", "predictions"), "Stage 1 validation"
    )
    _validate_archive_identity(archive, valid_rows, "Stage 1 validation")
    saved_keys = np.asarray(archive["keys"]).astype(str).reshape(-1)
    saved_labels = np.asarray(archive["labels"], dtype=np.float64).reshape(-1)
    saved_predictions = np.asarray(archive["predictions"], dtype=np.float64).reshape(-1)
    if saved_keys.shape != saved_labels.shape or saved_labels.shape != saved_predictions.shape:
        raise ValueError("Stage 1 validation prediction arrays are malformed")

    live = predict_complete_model(
        model, dataset, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    live_keys = np.asarray(live["keys"]).astype(str).reshape(-1)
    live_labels = np.asarray(live["labels"], dtype=np.float64).reshape(-1)
    live_predictions = np.asarray(live["predictions"], dtype=np.float64).reshape(-1)
    if tuple(live_keys.tolist()) != tuple(saved_keys.tolist()):
        raise ValueError("Stage 1 live validation key order differs from its saved artifact")
    if not np.allclose(live_labels, saved_labels, rtol=0.0, atol=1e-6):
        raise ValueError("Stage 1 live validation labels differ from its saved artifact")
    maximum_prediction_delta = float(np.max(np.abs(live_predictions - saved_predictions)))
    if maximum_prediction_delta > 1e-6:
        raise ValueError(
            "Stage 1 full checkpoint prediction differs from its saved validation output: %.9g"
            % maximum_prediction_delta
        )
    recorded_metrics = selection.get("validation_metrics", {})
    metric_deltas = {
        name: float(live["metrics"][name]) - float(recorded_metrics[name])
        for name in PROMOTION_METRICS
        if name in recorded_metrics
    }
    if len(metric_deltas) != len(PROMOTION_METRICS) or any(
        not np.isfinite(value) or abs(value) > 1e-6
        for value in metric_deltas.values()
    ):
        raise ValueError("Stage 1 checkpoint metrics differ from the saved selection")
    return {
        "valid_prediction_path": str(prediction_path),
        "valid_prediction_sha256": prediction_sha,
        "maximum_absolute_prediction_delta": maximum_prediction_delta,
        "metric_deltas": metric_deltas,
        "prediction_aggregation": live["prediction_aggregation"],
        "inference_subsets": live["inference_subsets"],
    }


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("output exists; choose a new Stage 2 directory: %s" % output)
    fusion_head_mode = getattr(args, "fusion_head_mode", "train")
    if fusion_head_mode not in FUSION_HEAD_MODES:
        raise ValueError("invalid fusion head mode: %s" % fusion_head_mode)
    training_stage = (
        "stage2_backbone_control" if fusion_head_mode == "frozen_zero"
        else "stage2_joint_finetune"
    )

    (
        stage1_checkpoint_path,
        stage1_checkpoint_sha,
        stage1_checkpoint,
        stage1_selection,
        saved_config,
        model,
    ) = _load_stage1(args.stage1_checkpoint)

    train_manifest_path, train_rows = _manifest_rows(args.train_manifest, "train")
    valid_manifest_path, valid_rows = _manifest_rows(args.valid_manifest, "valid")
    if set(train_rows) & set(valid_rows):
        raise ValueError("official train and validation manifests overlap")
    provenance = stage1_checkpoint.get("manifest_provenance", {})
    if sha256_file(train_manifest_path) != provenance.get("train_manifest_sha256"):
        raise ValueError("Stage 1 checkpoint used a different training manifest")
    if sha256_file(valid_manifest_path) != provenance.get("valid_manifest_sha256"):
        raise ValueError("Stage 1 checkpoint used a different validation manifest")
    train_labels = np.asarray(list(train_rows.values()), dtype=np.float64)
    if not np.isclose(float(train_labels.mean()), float(stage1_checkpoint["label_mean"]), rtol=0.0, atol=1e-8):
        raise ValueError("Stage 1 label mean differs from the official train manifest")
    if not np.isclose(float(train_labels.std()), float(stage1_checkpoint["label_std"]), rtol=0.0, atol=1e-8):
        raise ValueError("Stage 1 label std differs from the official train manifest")

    baseline_path, baseline_metrics, baseline_sources, _ = _load_v20_baseline(
        args.v20_baseline, valid_rows
    )

    if (
        args.evaluation_batch_size is not None
        and int(args.evaluation_batch_size) != int(saved_config.evaluation_batch_size)
    ):
        raise ValueError(
            "evaluation batch size differs from the sealed Stage 1 validation config"
        )
    if args.amp_dtype is not None and str(args.amp_dtype) != str(saved_config.amp_dtype):
        raise ValueError("AMP dtype differs from the sealed Stage 1 validation config")

    # All checkpoint/config, official split identity, normalization, and V20
    # validation-source checks occur before any LMDB-backed dataset is opened.
    config = replace(
        saved_config,
        device=args.device,
        batch_size=args.batch_size,
        evaluation_batch_size=saved_config.evaluation_batch_size,
        num_workers=args.num_workers,
        amp_dtype=args.amp_dtype or saved_config.amp_dtype,
        learning_rate=args.backbone_learning_rate,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    if sealed_inference_config(config) != stage1_checkpoint.get("inference_config"):
        raise ValueError("Stage 2 evaluation config differs from the sealed Stage 1 config")
    model = model.to(torch.device(args.device))
    train_dataset = MSAPairComplexDataset(
        str(train_manifest_path), args.train_cache, args.row_cache
    )
    valid_dataset = MSAPairComplexDataset(
        str(valid_manifest_path), args.valid_cache, args.row_cache
    )
    if set(train_dataset.keys) != set(train_rows) or set(valid_dataset.keys) != set(valid_rows):
        raise ValueError("dataset keys differ from the checked manifests")
    if any(
        not np.isclose(float(label), train_rows[key], rtol=0.0, atol=1e-6)
        for key, label in zip(train_dataset.keys, train_dataset.labels)
    ):
        raise ValueError("training dataset labels differ from the checked manifest")
    if any(
        not np.isclose(float(label), valid_rows[key], rtol=0.0, atol=1e-6)
        for key, label in zip(valid_dataset.keys, valid_dataset.labels)
    ):
        raise ValueError("validation dataset labels differ from the checked manifest")

    stage1_validation_check = _verify_stage1_validation_forward(
        model,
        valid_dataset,
        config,
        stage1_checkpoint,
        stage1_selection,
        valid_rows,
    )

    fitted = fit_joint_stage2(
        model,
        train_dataset,
        valid_dataset,
        config,
        baseline_metrics,
        stage1_checkpoint["label_mean"],
        stage1_checkpoint["label_std"],
        backbone_learning_rate=args.backbone_learning_rate,
        fusion_learning_rate=args.fusion_learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        metric_tolerance=args.metric_tolerance,
        seed=args.seed,
        fusion_head_mode=fusion_head_mode,
    )

    output.mkdir(parents=True, exist_ok=False)
    model_kwargs = dict(stage1_checkpoint["model_kwargs"])
    inference_config = dict(stage1_checkpoint["inference_config"])
    stage2_run_config = {
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "maximum_epochs": int(args.maximum_epochs),
        "patience": int(args.patience),
        "weight_decay": float(args.weight_decay),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "msa_encoder_learning_rate": float(args.backbone_learning_rate) * float(config.encoder_learning_rate_scale),
        "fusion_learning_rate": float(args.fusion_learning_rate),
        "row_depth": int(config.row_depth),
        "inference_subsets": int(config.inference_subsets),
        "interface_mask_probability": float(config.interface_mask_probability),
        "minimum_rank_gap": float(config.minimum_rank_gap),
        "ema_decay": float(config.ema_decay),
        "amp_dtype": str(config.amp_dtype),
    }
    checkpoint = dict(stage1_checkpoint)
    checkpoint.update(
        {
            "config": asdict(config),
            "model_kwargs": model_kwargs,
            "model_state": fitted["model_state"],
            "epoch": int(fitted["best_epoch"]),
            "variant": str(fitted["selected_variant"]),
            "training_protocol": "same_trunk_two_stage_v1",
            "training_stage": training_stage,
            "parent_stage1_checkpoint": str(stage1_checkpoint_path),
            "parent_stage1_checkpoint_sha256": stage1_checkpoint_sha,
            "optimizer_restored": False,
            "scheduler_restored": False,
            "stage2_training": {
                "method": "online_same_trunk_full_model_forward",
                "fusion_head_mode": fusion_head_mode,
                "fusion_head_changed_tensor_count": fitted["fusion_head_changed_tensor_count"],
                "optimizer_steps": fitted["optimizer_steps"],
                "epochs_completed": fitted["epochs_completed"],
                "run_config": stage2_run_config,
                "best_epoch": fitted["best_epoch"],
                "selected_variant": fitted["selected_variant"],
                "optimizer_groups": fitted["optimizer_groups"],
                "learning_rate_config": fitted["learning_rate_config"],
                "scheduler": fitted["scheduler"],
                "optimizer_restored": False,
                "scheduler_restored": False,
                "validation_metrics_before_reload": fitted["validation_metrics"],
                "validation_gate_before_reload": fitted["validation_gate"],
                "test_manifest_read": False,
                "stage1_validation_reproduction": stage1_validation_check,
            },
            "inference_config": inference_config,
            "manifest_provenance": {
                "train_manifest": str(train_manifest_path),
                "train_manifest_sha256": sha256_file(train_manifest_path),
                "valid_manifest": str(valid_manifest_path),
                "valid_manifest_sha256": sha256_file(valid_manifest_path),
            },
        }
    )
    checkpoint_path = output / "best.pt"
    _atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    # The promotion gate is evaluated from the exact full checkpoint after
    # saving and reloading, with the same model config and per-subset forward.
    reloaded = torch.load(str(checkpoint_path), map_location="cpu")
    reloaded_model = E2EMSAInterfacePairFusionV3(**model_kwargs).to(
        torch.device(args.device)
    )
    reloaded_model.load_state_dict(reloaded["model_state"], strict=True)
    reloaded_config = TrainingConfig(**reloaded["config"])
    reloaded_metrics_result = predict_complete_model(
        reloaded_model,
        valid_dataset,
        reloaded_config,
        reloaded["label_mean"],
        reloaded["label_std"],
    )
    reloaded_metrics = dict(reloaded_metrics_result["metrics"])
    metric_deltas = {
        name: float(reloaded_metrics[name]) - float(fitted["validation_metrics"][name])
        for name in PROMOTION_METRICS
    }
    before_reload = fitted["validation_prediction_result"]
    after_keys = tuple(reloaded_metrics_result["keys"])
    before_keys = tuple(before_reload["keys"])
    prediction_deltas = None
    if after_keys == before_keys:
        before_labels = np.asarray(before_reload["labels"], dtype=np.float64)
        after_labels = np.asarray(reloaded_metrics_result["labels"], dtype=np.float64)
        before_predictions = np.asarray(before_reload["predictions"], dtype=np.float64)
        after_predictions = np.asarray(reloaded_metrics_result["predictions"], dtype=np.float64)
        if before_labels.shape == after_labels.shape and before_predictions.shape == after_predictions.shape:
            prediction_deltas = {
                "maximum_absolute_prediction": float(
                    np.max(np.abs(after_predictions - before_predictions))
                ),
                "maximum_absolute_label": float(np.max(np.abs(after_labels - before_labels))),
            }
    reload_consistent = (
        prediction_deltas is not None
        and prediction_deltas["maximum_absolute_prediction"] <= 1e-6
        and prediction_deltas["maximum_absolute_label"] <= 1e-6
        and all(abs(value) <= 1e-6 for value in metric_deltas.values())
    )
    gate = validation_gate(
        reloaded_metrics, baseline_metrics, tolerance=args.metric_tolerance
    )
    metrics_reload_match = all(
        np.isfinite(value) and abs(value) <= 1e-6
        for value in metric_deltas.values()
    )
    predictions_reload_match = (
        prediction_deltas is not None
        and prediction_deltas["maximum_absolute_prediction"] <= 1e-6
        and prediction_deltas["maximum_absolute_label"] <= 1e-6
    )
    gate["checkpoint_reload_metrics_match"] = bool(metrics_reload_match)
    gate["checkpoint_reload_predictions_match"] = bool(predictions_reload_match)
    gate["checkpoint_reload_consistent"] = bool(reload_consistent)
    if not reload_consistent:
        gate["passed"] = False
    status = "validation_passed" if gate["passed"] else "validation_failed"
    valid_prediction_path = output / "valid_predictions.npz"
    np.savez_compressed(
        valid_prediction_path,
        keys=np.asarray(reloaded_metrics_result["keys"]),
        labels=np.asarray(reloaded_metrics_result["labels"], dtype=np.float64),
        predictions=np.asarray(reloaded_metrics_result["predictions"], dtype=np.float64),
        latents=np.asarray(reloaded_metrics_result["latents"], dtype=np.float32),
    )
    result = {
        "status": status,
        "version": VERSION,
        "training_protocol": "same_trunk_two_stage_v1",
        "training_stage": training_stage,
        "fusion_head_mode": fusion_head_mode,
        "fusion_head_changed_tensor_count": fitted["fusion_head_changed_tensor_count"],
        "optimizer_steps": fitted["optimizer_steps"],
        "epochs_completed": fitted["epochs_completed"],
        "stage2_run_config": stage2_run_config,
        "validation_selected": bool(gate["passed"]),
        "test_manifest_read": False,
        "test_metrics": None,
        "architecture": architecture_summary(),
        "inference_config": inference_config,
        "manifests": {
            "train": {
                "path": str(train_manifest_path),
                "sha256": sha256_file(train_manifest_path),
                "count": len(train_rows),
            },
            "valid": {
                "path": str(valid_manifest_path),
                "sha256": sha256_file(valid_manifest_path),
                "count": len(valid_rows),
            },
        },
        "label_normalization": {
            "mean": float(reloaded["label_mean"]),
            "std": float(reloaded["label_std"]),
        },
        "stage1_reference": {
            "checkpoint": str(stage1_checkpoint_path),
            "checkpoint_sha256": stage1_checkpoint_sha,
            "validation_metrics": stage1_selection.get("validation_metrics"),
        },
        "best_epoch": int(fitted["best_epoch"]),
        "selected_variant": fitted["selected_variant"],
        "validation_metrics_before_reload": fitted["validation_metrics"],
        "validation_metrics": reloaded_metrics,
        "checkpoint_reload_metric_deltas": metric_deltas,
        "checkpoint_reload_prediction_deltas": prediction_deltas,
        "checkpoint_reload_consistent": reload_consistent,
        "stage1_validation_reproduction": stage1_validation_check,
        "v20_baseline_validation_metrics": dict(baseline_metrics),
        "v20_validation_sources": baseline_sources,
        "v20_baseline_selection": str(baseline_path),
        "v20_baseline_selection_sha256": sha256_file(baseline_path),
        "validation_gate": gate,
        "optimizer_groups": fitted["optimizer_groups"],
        "learning_rate_config": fitted["learning_rate_config"],
        "optimizer_restored": False,
        "scheduler_restored": False,
        "training_history": fitted["history"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "valid_predictions": str(valid_prediction_path),
        "valid_predictions_sha256": sha256_file(valid_prediction_path),
        "output": str(output),
    }
    (output / "history.json").write_text(
        json.dumps(fitted["history"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("result.json", "selection.json"):
        (output / name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--v20-baseline", default=DEFAULT_V20_SELECTION)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--backbone-learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--fusion-learning-rate", type=float, default=5e-5)
    parser.add_argument("--fusion-head-mode", choices=FUSION_HEAD_MODES, default="train")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
