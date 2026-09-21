"""Train OOF rank-unlocked fusion and evaluate the sealed LP-PDBbind test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.training import static_predictions
from dta.rank_fusion_v4.cache import load_fusion_cache
from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.rank_locked_calibration_v6.data import (
    RankLockedCalibrationDataset,
    collate_rank_locked_calibration,
    load_validation_caches,
    reconstruct_v5_anchor,
)
from dta.rank_locked_calibration_v6.folds import assign_grouped_folds, load_fold_rows
from utils.experiment_integrity import sha256_file

from . import META_TRAIN_COUNT, TEST_COUNT, VERSION
from .selection import apply_selection, build_oof_selection
from .training import TrainingConfig, cross_fit, predict_cross_fitted_ensemble


V5_TEST_CI = 0.7086996133362983


def _json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def test_improves_v5(metrics, baseline_metrics):
    return float(metrics["CI"]) > float(baseline_metrics["CI"])


def _subset(dataset, count):
    if count is None:
        return dataset, np.arange(len(dataset), dtype=np.int64)
    if count <= 0 or count > len(dataset):
        raise ValueError("limit is outside the dataset")
    indices = np.arange(count, dtype=np.int64)
    return torch.utils.data.Subset(dataset, indices.tolist()), indices


def _validate_test_cache(cache):
    if len(cache.keys) != TEST_COUNT:
        raise ValueError("test cache count does not match")
    if not cache.available[:, 0].all():
        raise ValueError("test Upper-Bound coverage is incomplete")
    if cache.upper_bound_latents.shape[1] != 1024:
        raise ValueError("Upper-Bound latent width does not match")
    if cache.boltz_latents.shape[1] != 640:
        raise ValueError("Boltz latent width does not match")
    if cache.three_dprot_latents.shape[1] != 1024:
        raise ValueError("3DProt latent width does not match")


def _checkpoint_hashes(checkpoints):
    return {
        str(seed): [sha256_file(Path(path)) for path in paths]
        for seed, paths in checkpoints.items()
    }


def run_pipeline(args):
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and not args.resume and any(output.iterdir()):
        raise ValueError("V7 output already exists; use --resume or a new output")
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "event": "v7_preflight_start",
        "device": args.device,
        "mode": "smoke" if args.smoke else "formal",
        "version": VERSION,
    }, sort_keys=True), flush=True)

    references = load_validation_caches(
        args.cache_root,
        {"train": 11037, "valid": META_TRAIN_COUNT},
    )
    anchor = reconstruct_v5_anchor(
        args.v5_package, args.v5_validation_root, references
    )
    valid_cache = references["valid"].cache
    full_meta_dataset = RankLockedCalibrationDataset(
        valid_cache, anchor.valid_predictions
    )
    meta_dataset, meta_indices = _subset(full_meta_dataset, args.limit)
    meta_keys = tuple(valid_cache.keys[index] for index in meta_indices)
    meta_labels = np.asarray(valid_cache.labels[meta_indices], dtype=np.float64)
    meta_anchor = np.asarray(anchor.valid_predictions[meta_indices], dtype=np.float64)

    rows = load_fold_rows(
        args.metadata_csv,
        meta_keys,
        expected_split="val",
        allow_invalid_scaffold_fallback=True,
    )
    fold_assignment = assign_grouped_folds(rows, fold_count=5, seed=42)
    fold_ids = np.asarray(
        [fold_assignment.by_key[key] for key in meta_keys], dtype=np.int64
    )
    config = TrainingConfig(
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        maximum_epochs=args.maximum_epochs,
        patience=min(args.patience, args.maximum_epochs),
        gradient_clip=1.0,
        model_width=args.model_width,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )
    fit = cross_fit(
        meta_dataset,
        fold_ids,
        output / "models",
        config,
        collate_rank_locked_calibration,
        resume=args.resume,
    )
    selection = build_oof_selection(
        meta_anchor, fit["oof_predictions"], meta_labels
    )
    selection.update({
        "schema_version": 1,
        "version": VERSION,
        "mode": "smoke" if args.smoke else "formal",
        "meta_count": len(meta_dataset),
        "fold_map_sha256": fold_assignment.sha256,
        "checkpoint_sha256": _checkpoint_hashes(fit["checkpoints"]),
        "source_revision": args.source_revision,
        "v5_package": str(Path(args.v5_package).resolve()),
    })
    _atomic_npz(
        output / "oof_predictions.npz",
        keys=np.asarray(meta_keys),
        labels=meta_labels,
        anchor=meta_anchor,
        seed_1=fit["oof_predictions"][1],
        seed_7=fit["oof_predictions"][7],
        seed_21=fit["oof_predictions"][21],
    )
    _atomic_json(output / "training_history.json", fit["histories"])
    _atomic_json(output / "selection.json", selection)
    selection_hash = hashlib.sha256(_json_bytes(selection)).hexdigest()
    print(json.dumps({
        "event": "v7_oof_complete",
        "anchor_metrics": selection["anchor_metrics"],
        "oof_metrics": selection["oof_metrics"],
        "status": selection["status"],
    }, sort_keys=True), flush=True)

    if selection["status"] != "passed" and not args.smoke:
        result = {
            "status": "validation_failed",
            "selection": selection,
            "official_test_allowed": False,
        }
        _atomic_json(output / "result.json", result)
        return result

    test_cache = load_fusion_cache(
        Path(args.cache_root) / "joined" / "test", expected_split="test"
    )
    _validate_test_cache(test_cache)
    test_anchor_full = np.asarray(
        static_predictions(test_cache, anchor.static_spec), dtype=np.float64
    )
    full_test_dataset = RankLockedCalibrationDataset(test_cache, test_anchor_full)
    test_dataset, test_indices = _subset(full_test_dataset, args.limit)
    test_anchor = test_anchor_full[test_indices]
    test_labels = np.asarray(test_cache.labels[test_indices], dtype=np.float64)
    test_keys = tuple(test_cache.keys[index] for index in test_indices)
    seed_test_predictions = predict_cross_fitted_ensemble(
        fit["checkpoints"], test_dataset, config, collate_rank_locked_calibration
    )
    selected_test = apply_selection(
        test_anchor, seed_test_predictions, selection
    )
    test_metrics = affinity_metrics(test_labels, selected_test)
    try:
        with Path(args.v5_fallback_result).open("r", encoding="utf-8") as handle:
            fallback = json.load(handle)
        baseline_metrics = dict(fallback["metrics"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("V5 fallback result is malformed") from error
    if args.smoke:
        status = "smoke_complete"
    else:
        status = "passed" if test_improves_v5(test_metrics, baseline_metrics) else "failed_gate"
    _atomic_npz(
        output / "test_predictions.npz",
        keys=np.asarray(test_keys),
        labels=test_labels,
        anchor=test_anchor,
        predictions=selected_test,
    )
    result = {
        "status": status,
        "count": len(test_dataset),
        "metrics": dict(test_metrics),
        "v5_test_metrics": baseline_metrics,
        "delta_ci_vs_v5": float(test_metrics["CI"] - baseline_metrics["CI"]),
        "selection_sha256": selection_hash,
        "selection": selection,
        "test_manifest_sha256": test_cache.metadata.get("manifest_sha256"),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--v5-validation-root", required=True)
    parser.add_argument("--v5-package", required=True)
    parser.add_argument("--v5-fallback-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-width", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "run_pipeline", "test_improves_v5"]
