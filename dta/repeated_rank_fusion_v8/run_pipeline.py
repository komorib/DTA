"""Train repeated V8 OOF predictors, stack them, and evaluate once on test."""

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
from dta.rank_unlocked_fusion_v7.selection import apply_selection
from dta.rank_unlocked_fusion_v7.training import (
    TrainingConfig,
    predict_cross_fitted_ensemble,
)
from utils.experiment_integrity import sha256_file

from . import FOLD_REPEATS, STACKER_REGULARIZATIONS, VERSION
from .stacker import (
    cross_fit_rank_stacker,
    fit_positive_rank_stacker,
    predict_rank_stacker,
)
from .training import fit_repeat, predict_repeat_ensemble


META_COUNT = 2340
TEST_COUNT = 4751
FEATURE_NAMES = (
    "v5_anchor",
    "v7_seed_1",
    "v7_seed_7",
    "v7_seed_21",
    "repeat_137",
    "repeat_271",
    "repeat_389",
)


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


def _subset(dataset, count):
    if count is None:
        return dataset, np.arange(len(dataset), dtype=np.int64)
    if count <= 0 or count > len(dataset):
        raise ValueError("V8 limit is outside the dataset")
    indices = np.arange(count, dtype=np.int64)
    return torch.utils.data.Subset(dataset, indices.tolist()), indices


def test_improves_v7(metrics, baseline_metrics):
    return float(metrics["CI"]) > float(baseline_metrics["CI"])


def _load_json(path, name):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError(name + " is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(name + " must be a dictionary")
    return value


def _v7_checkpoint_inventory(v7_root, v7_result):
    expected = v7_result.get("selection", {}).get("checkpoint_sha256")
    if not isinstance(expected, dict) or set(expected) != {"1", "7", "21"}:
        raise ValueError("V7 checkpoint hashes are missing")
    inventory = {}
    for seed in (1, 7, 21):
        paths = [
            Path(v7_root) / "models" / ("seed_%d" % seed) / ("fold_%d.pt" % fold)
            for fold in range(5)
        ]
        hashes = expected[str(seed)]
        if len(hashes) != len(paths):
            raise ValueError("V7 checkpoint hash count does not match")
        for path, expected_hash in zip(paths, hashes):
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError("V7 checkpoint identity does not match: " + str(path))
        inventory[seed] = [str(path.resolve()) for path in paths]
    return inventory


def _load_v7_oof(v7_root, keys, labels, anchor, indices):
    path = Path(v7_root) / "oof_predictions.npz"
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            required = {"keys", "labels", "anchor", "seed_1", "seed_7", "seed_21"}
            if set(archive.files) != required:
                raise ValueError("V7 OOF fields do not match")
            stored_keys = tuple(str(value) for value in archive["keys"].tolist())
            stored_labels = np.asarray(archive["labels"], dtype=np.float64)
            stored_anchor = np.asarray(archive["anchor"], dtype=np.float64)
            seeds = {
                seed: np.asarray(archive["seed_%d" % seed], dtype=np.float64)
                for seed in (1, 7, 21)
            }
    except (OSError, ValueError) as error:
        raise ValueError("V7 OOF predictions are malformed") from error
    if stored_keys != tuple(keys) or not np.allclose(stored_labels, labels, atol=1e-6):
        raise ValueError("V7 OOF keys or labels do not match")
    if not np.allclose(stored_anchor, anchor, rtol=0.0, atol=1e-10):
        raise ValueError("V7 OOF anchor does not match")
    return {seed: values[indices] for seed, values in seeds.items()}


def _select_stacker(features, labels, fold_ids, maximum_pairs):
    candidates = {}
    best = None
    for regularization in STACKER_REGULARIZATIONS:
        cross_fit = cross_fit_rank_stacker(
            features,
            labels,
            fold_ids,
            regularization=regularization,
            maximum_pairs=maximum_pairs,
            seed=20260820,
        )
        metrics = affinity_metrics(labels, cross_fit["predictions"])
        name = "positive_pairwise_" + str(regularization)
        candidates[name] = {
            "regularization": regularization,
            "metrics": dict(metrics),
        }
        score = (
            float(metrics["CI"]),
            -float(metrics["RMSE"]),
            -float(metrics["MAE"]),
        )
        if best is None or score > best[0]:
            best = (score, name, regularization, cross_fit)
    _, name, regularization, cross_fit = best
    final_model = fit_positive_rank_stacker(
        features,
        labels,
        regularization=regularization,
        maximum_pairs=maximum_pairs,
        seed=20260820,
    )
    return {
        "selected_name": name,
        "selected_metrics": candidates[name]["metrics"],
        "selected_oof_predictions": cross_fit["predictions"],
        "candidate_metrics": candidates,
        "model": final_model,
    }


def run_pipeline(args):
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and not args.resume and any(output.iterdir()):
        raise ValueError("V8 output already exists; use --resume or a new output")
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "event": "v8_preflight_start",
        "device": args.device,
        "mode": "smoke" if args.smoke else "formal",
        "version": VERSION,
    }, sort_keys=True), flush=True)

    references = load_validation_caches(
        args.cache_root, {"train": 11037, "valid": META_COUNT}
    )
    anchor = reconstruct_v5_anchor(
        args.v5_package, args.v5_validation_root, references
    )
    valid_cache = references["valid"].cache
    full_dataset = RankLockedCalibrationDataset(valid_cache, anchor.valid_predictions)
    dataset, indices = _subset(full_dataset, args.limit)
    keys = tuple(valid_cache.keys[index] for index in indices)
    labels = np.asarray(valid_cache.labels[indices], dtype=np.float64)
    anchor_values = np.asarray(anchor.valid_predictions[indices], dtype=np.float64)
    rows = load_fold_rows(
        args.metadata_csv,
        keys,
        expected_split="val",
        allow_invalid_scaffold_fallback=True,
    )
    meta_assignment = assign_grouped_folds(rows, fold_count=5, seed=42)
    meta_fold_ids = np.asarray(
        [meta_assignment.by_key[key] for key in keys], dtype=np.int64
    )

    v7_root = Path(args.v7_root).expanduser().resolve()
    v7_result = _load_json(v7_root / "result.json", "V7 result")
    if v7_result.get("status") != "passed":
        raise ValueError("V7 result is not a passing formal result")
    v7_inventory = _v7_checkpoint_inventory(v7_root, v7_result)
    v7_oof = _load_v7_oof(
        v7_root,
        valid_cache.keys,
        np.asarray(valid_cache.labels, dtype=np.float64),
        np.asarray(anchor.valid_predictions, dtype=np.float64),
        indices,
    )
    v7_selection = v7_result["selection"]
    v7_oof_selected = apply_selection(anchor_values, v7_oof, v7_selection)
    v7_oof_metrics = affinity_metrics(labels, v7_oof_selected)

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
    repeat_results = []
    for repeat in FOLD_REPEATS:
        assignment = assign_grouped_folds(
            rows, fold_count=5, seed=repeat["fold_seed"]
        )
        fold_ids = np.asarray(
            [assignment.by_key[key] for key in keys], dtype=np.int64
        )
        data_lineage_sha256 = hashlib.sha256(_json_bytes({
            "cache_metadata": valid_cache.metadata,
            "keys": list(keys),
            "labels": labels.tolist(),
            "anchor": anchor_values.tolist(),
            "fold_ids": fold_ids.tolist(),
        })).hexdigest()
        repeat_results.append(
            fit_repeat(
                dataset,
                fold_ids,
                repeat_name=repeat["name"],
                fold_seed=repeat["fold_seed"],
                model_seed=repeat["model_seed"],
                data_lineage_sha256=data_lineage_sha256,
                output=output / "models",
                config=config,
                collate_fn=collate_rank_locked_calibration,
                resume=args.resume,
            )
        )

    feature_columns = [
        anchor_values,
        v7_oof[1],
        v7_oof[7],
        v7_oof[21],
    ] + [result["predictions"] for result in repeat_results]
    features = np.stack(feature_columns, axis=1)
    stacker = _select_stacker(
        features, labels, meta_fold_ids, args.maximum_pairs
    )
    selected_metrics = stacker["selected_metrics"]
    improved_oof = float(selected_metrics["CI"]) > float(v7_oof_metrics["CI"])
    selection = {
        "schema_version": 1,
        "version": VERSION,
        "mode": "smoke" if args.smoke else "formal",
        "status": "passed" if improved_oof else "failed",
        "feature_names": list(FEATURE_NAMES),
        "selected_name": stacker["selected_name"],
        "selected_oof_metrics": selected_metrics,
        "candidate_metrics": stacker["candidate_metrics"],
        "v7_oof_metrics": dict(v7_oof_metrics),
        "model": stacker["model"],
        "meta_count": len(dataset),
        "meta_fold_map_sha256": meta_assignment.sha256,
        "repeat_inventory": [
            {
                "name": result["repeat_name"],
                "fold_seed": result["fold_seed"],
                "model_seed": result["model_seed"],
                "data_lineage_sha256": result["data_lineage_sha256"],
                "checkpoints": result["checkpoints"],
                "checkpoint_sha256": [
                    sha256_file(Path(path)) for path in result["checkpoints"]
                ],
            }
            for result in repeat_results
        ],
        "v7_root": str(v7_root),
        "source_revision": args.source_revision,
    }
    _atomic_npz(
        output / "oof_predictions.npz",
        keys=np.asarray(keys),
        labels=labels,
        features=features,
        v7_selected=v7_oof_selected,
        v8_selected=stacker["selected_oof_predictions"],
    )
    _atomic_json(
        output / "training_history.json",
        [item for result in repeat_results for item in result["histories"]],
    )
    _atomic_json(output / "selection.json", selection)
    selection_hash = hashlib.sha256(_json_bytes(selection)).hexdigest()
    print(json.dumps({
        "event": "v8_oof_complete",
        "v7_oof_metrics": v7_oof_metrics,
        "v8_oof_metrics": selected_metrics,
        "status": selection["status"],
    }, sort_keys=True), flush=True)
    if selection["status"] != "passed" and not args.smoke:
        result = {
            "status": "validation_failed",
            "official_test_allowed": False,
            "selection": selection,
        }
        _atomic_json(output / "result.json", result)
        return result

    test_cache = load_fusion_cache(
        Path(args.cache_root) / "joined" / "test", expected_split="test"
    )
    if len(test_cache.keys) != TEST_COUNT:
        raise ValueError("V8 test cache count does not match")
    test_anchor_full = np.asarray(
        static_predictions(test_cache, anchor.static_spec), dtype=np.float64
    )
    full_test_dataset = RankLockedCalibrationDataset(test_cache, test_anchor_full)
    test_dataset, test_indices = _subset(full_test_dataset, args.limit)
    test_keys = tuple(test_cache.keys[index] for index in test_indices)
    test_labels = np.asarray(test_cache.labels[test_indices], dtype=np.float64)
    test_anchor = test_anchor_full[test_indices]

    v7_test_seeds = predict_cross_fitted_ensemble(
        v7_inventory, test_dataset, config, collate_rank_locked_calibration
    )
    repeat_test = [
        predict_repeat_ensemble(
            result["checkpoints"],
            test_dataset,
            config,
            collate_rank_locked_calibration,
        )
        for result in repeat_results
    ]
    test_features = np.stack(
        [
            test_anchor,
            v7_test_seeds[1],
            v7_test_seeds[7],
            v7_test_seeds[21],
        ]
        + repeat_test,
        axis=1,
    )
    predictions = predict_rank_stacker(test_features, stacker["model"])
    test_metrics = affinity_metrics(test_labels, predictions)
    v7_predictions_path = v7_root / "test_predictions.npz"
    try:
        with np.load(str(v7_predictions_path), allow_pickle=False) as archive:
            v7_test_keys = tuple(str(value) for value in archive["keys"].tolist())
            v7_test_predictions = np.asarray(
                archive["predictions"], dtype=np.float64
            )[test_indices]
    except (OSError, ValueError) as error:
        raise ValueError("V7 test predictions are malformed") from error
    if v7_test_keys != tuple(test_cache.keys):
        raise ValueError("V7 test prediction keys do not match")
    v7_test_metrics = affinity_metrics(test_labels, v7_test_predictions)
    status = (
        "smoke_complete"
        if args.smoke
        else "passed" if test_improves_v7(test_metrics, v7_test_metrics) else "failed_gate"
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=np.asarray(test_keys),
        labels=test_labels,
        features=test_features,
        predictions=predictions,
        v7_predictions=v7_test_predictions,
    )
    result = {
        "status": status,
        "count": len(test_dataset),
        "metrics": dict(test_metrics),
        "v7_test_metrics": dict(v7_test_metrics),
        "delta_ci_vs_v7": float(test_metrics["CI"] - v7_test_metrics["CI"]),
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
    parser.add_argument("--v7-root", required=True)
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
    parser.add_argument("--maximum-pairs", type=int, default=200000)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "run_pipeline", "test_improves_v7"]
