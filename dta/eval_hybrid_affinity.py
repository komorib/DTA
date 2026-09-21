#!/usr/bin/env python
"""Guarded one-time test evaluation for frozen hybrid affinity models."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
import torch

from dta.hybrid_affinity_training import (
    atomic_write_predictions,
    evaluate_hybrid_affinity,
)
from dta.train_hybrid_affinity import build_preflight_binding
from scripts.select_hybrid_affinity_runs import (
    REQUIRED_METRICS,
    load_prediction_archive,
)
from utils.experiment_integrity import (
    atomic_write_json,
    hash_ordered_keys,
    sha256_file,
)


FIXED_TEST_MANIFEST = (
    _REPOSITORY_ROOT
    / "data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/test_manifest.json"
)
FIXED_TEST_NATIVE_CACHE = (
    _REPOSITORY_ROOT
    / "data/LP_PDBBind_InterfaceFirstGerNA_v1/native_test"
)
EXPECTED_SEEDS = (1, 7, 21, 42, 87)
MINIMUM_VALIDATION_CI = 0.697287
FIXED_FUSION_TEST_METRICS = {
    "RMSE": 1.447771,
    "MAE": 1.137539,
    "Pearson": 0.550403,
    "Spearman": 0.529191,
    "CI": 0.685273,
}


def _load_json(path, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_metrics(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    result = {}
    for metric_name in REQUIRED_METRICS:
        if metric_name not in value:
            raise ValueError(f"{name} is missing {metric_name}")
        result[metric_name] = _finite_float(
            value[metric_name], f"{name}.{metric_name}"
        )
    return result


def _require_sha256(path, expected, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise RuntimeError(f"{name} hash differs: {path}")
    return path, actual


def _validated_run_config(
    candidate,
    *,
    fusion_type,
    rank_weight,
    seed,
    model_path,
    model_sha256,
    prediction_path,
    prediction_sha256,
):
    if not isinstance(candidate, dict):
        raise ValueError("selected candidate must be a mapping")
    run_config = candidate.get("run_config")
    expected_top_level = {
        "schema_version": 1,
        "fusion_type": fusion_type,
        "rank_weight": rank_weight,
        "seed": seed,
    }
    for name, expected in expected_top_level.items():
        actual = candidate.get(name)
        if name == "rank_weight":
            try:
                matches = float(actual) == float(expected)
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f"selected candidate config differs for seed {seed}: {name}"
            )
    if not isinstance(run_config, dict):
        raise RuntimeError(
            f"selected candidate config is missing for seed {seed}"
        )
    for name, expected in expected_top_level.items():
        actual = run_config.get(name)
        if name == "rank_weight":
            try:
                matches = float(actual) == float(expected)
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f"selected candidate run config differs for seed {seed}: "
                + name
            )
    if (
        run_config.get("amp") is not True
        or int(run_config.get("num_workers", -1)) != 0
        or int(run_config.get("group_size", 0)) not in {2, 4}
        or str(run_config.get("device")) != "cuda:0"
    ):
        raise RuntimeError(
            f"selected candidate run config is not formal for seed {seed}"
        )
    if Path(candidate.get("model_path", "")).expanduser().resolve() != model_path:
        raise RuntimeError(
            f"selected candidate model path differs for seed {seed}"
        )
    if candidate.get("model_sha256") != model_sha256:
        raise RuntimeError(
            f"selected candidate model hash differs for seed {seed}"
        )
    if (
        Path(candidate.get("prediction_path", ""))
        .expanduser()
        .resolve()
        != prediction_path
    ):
        raise RuntimeError(
            f"selected candidate prediction path differs for seed {seed}"
        )
    if candidate.get("prediction_sha256") != prediction_sha256:
        raise RuntimeError(
            f"selected candidate prediction hash differs for seed {seed}"
        )
    provenance_hashes = candidate.get("provenance_hashes")
    required_hashes = {
        "gerna_checkpoint_sha256",
        "interface_checkpoint_sha256",
        "train_manifest_sha256",
        "train_native_metadata_sha256",
        "train_ordered_key_sha256",
        "train_sample_count",
        "valid_manifest_sha256",
        "valid_native_metadata_sha256",
        "valid_ordered_key_sha256",
        "valid_sample_count",
    }
    if (
        not isinstance(provenance_hashes, dict)
        or set(provenance_hashes) != required_hashes
    ):
        raise RuntimeError(
            f"selected candidate provenance differs for seed {seed}"
        )
    training_path, training_sha = _require_sha256(
        candidate.get("training_checkpoint", ""),
        candidate.get("training_checkpoint_sha256"),
        f"training checkpoint for seed {seed}",
    )
    return {
        "run_config": dict(run_config),
        "provenance_hashes": dict(provenance_hashes),
        "training_checkpoint": str(training_path),
        "training_checkpoint_sha256": training_sha,
    }


def load_and_validate_selection(
    selection_manifest_path,
    *,
    minimum_validation_ci=MINIMUM_VALIDATION_CI,
):
    selection_path = Path(selection_manifest_path).expanduser().resolve()
    payload = _load_json(selection_path, "selection manifest")
    if (
        payload.get("schema_version") != 1
        or payload.get("dataset") != "LP-PDBBind"
        or payload.get("selection_split") != "valid"
    ):
        raise ValueError("selection manifest identity differs")
    fusion_type = str(payload.get("fusion_type"))
    if fusion_type not in {"mlp", "transformer"}:
        raise ValueError("selection fusion type is invalid")
    rank_weight = _finite_float(
        payload.get("rank_weight"), "selection rank_weight"
    )
    if rank_weight not in {0.1, 0.2, 0.4}:
        raise ValueError("selection rank_weight is invalid")
    metrics = _validated_metrics(
        payload.get("validation_metrics"), "validation metrics"
    )
    minimum_validation_ci = _finite_float(
        minimum_validation_ci, "minimum validation CI"
    )
    if metrics["CI"] <= minimum_validation_ci:
        raise ValueError(
            "validation CI must exceed "
            + f"{minimum_validation_ci:.6f}; got {metrics['CI']:.6f}"
        )

    try:
        seeds = tuple(int(seed) for seed in payload["seeds"])
        model_paths = list(payload["model_paths"])
        model_hashes = list(payload["model_sha256"])
        prediction_paths = list(payload["prediction_paths"])
        prediction_hashes = list(payload["prediction_sha256"])
        weights = np.asarray(payload["weights"], dtype=np.float64)
        intercept = _finite_float(
            payload["intercept"], "selection intercept"
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("selection manifest arrays are malformed") from error
    if seeds != EXPECTED_SEEDS:
        raise ValueError("selection seeds differ from the fixed five seeds")
    lengths = {
        len(model_paths),
        len(model_hashes),
        len(prediction_paths),
        len(prediction_hashes),
        int(weights.size),
        len(seeds),
    }
    if lengths != {len(EXPECTED_SEEDS)} or weights.ndim != 1:
        raise ValueError("selection model arrays are misaligned")
    if not np.isfinite(weights).all() or np.any(np.abs(weights) > 2.0):
        raise ValueError("selection weights are invalid")

    key_hash = payload.get("prediction_key_sha256")
    if not isinstance(key_hash, str) or len(key_hash) != 64:
        raise ValueError("selection prediction key hash is invalid")
    validation_prediction_path, _ = _require_sha256(
        payload.get("validation_prediction_path", ""),
        payload.get("validation_prediction_sha256"),
        "ensemble validation prediction",
    )
    validation_archive = load_prediction_archive(
        validation_prediction_path
    )
    if hash_ordered_keys(validation_archive["keys"]) != key_hash:
        raise RuntimeError("ensemble validation prediction key hash differs")

    records = []
    common_provenance = None
    for index, seed in enumerate(seeds):
        model_path, model_sha = _require_sha256(
            model_paths[index],
            model_hashes[index],
            f"model hash for seed {seed}",
        )
        prediction_path, prediction_sha = _require_sha256(
            prediction_paths[index],
            prediction_hashes[index],
            f"validation prediction for seed {seed}",
        )
        archive = load_prediction_archive(prediction_path)
        if hash_ordered_keys(archive["keys"]) != key_hash:
            raise RuntimeError(
                f"validation prediction key hash differs for seed {seed}"
            )
        candidate_path = model_path.parent / "selected_candidate.json"
        candidate = _load_json(
            candidate_path, f"selected candidate for seed {seed}"
        )
        config = _validated_run_config(
            candidate,
            fusion_type=fusion_type,
            rank_weight=rank_weight,
            seed=seed,
            model_path=model_path,
            model_sha256=model_sha,
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
        )
        if common_provenance is None:
            common_provenance = config["provenance_hashes"]
        elif config["provenance_hashes"] != common_provenance:
            raise RuntimeError(
                f"selected candidate provenance differs for seed {seed}"
            )
        records.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "model_sha256": model_sha,
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction_sha,
                "selected_candidate_path": str(candidate_path.resolve()),
                "selected_candidate_sha256": sha256_file(candidate_path),
                **config,
            }
        )
    return {
        "path": str(selection_path),
        "sha256": sha256_file(selection_path),
        "payload": payload,
        "fusion_type": fusion_type,
        "rank_weight": rank_weight,
        "metrics": metrics,
        "weights": weights,
        "intercept": intercept,
        "prediction_key_sha256": key_hash,
        "records": records,
    }


def validate_final_request(
    selection_manifest_path,
    test_manifest_path,
    test_native_cache_path,
    output_dir,
    *,
    confirm_final_test,
    expected_test_manifest=FIXED_TEST_MANIFEST,
    expected_test_native_cache=FIXED_TEST_NATIVE_CACHE,
):
    if confirm_final_test is not True:
        raise ValueError("explicit final test confirmation is required")
    selection_path = Path(selection_manifest_path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"selection manifest: {selection_path}")

    manifest_path = Path(test_manifest_path).expanduser().resolve()
    expected_manifest = (
        Path(expected_test_manifest).expanduser().resolve()
    )
    if manifest_path != expected_manifest:
        raise ValueError(
            f"test manifest is not the fixed path: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixed test manifest: {manifest_path}")

    native_cache = Path(test_native_cache_path).expanduser().resolve()
    expected_cache = (
        Path(expected_test_native_cache).expanduser().resolve()
    )
    if native_cache != expected_cache:
        raise ValueError(
            f"test native cache is not the fixed path: {native_cache}"
        )
    if not native_cache.is_dir():
        raise FileNotFoundError(f"fixed test native cache: {native_cache}")

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"final output is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"final output directory is not empty: {output}"
            )
    return {
        "selection_manifest": selection_path,
        "test_manifest": manifest_path,
        "test_native_cache": native_cache,
        "output_dir": output,
    }


def _load_test_manifest(path):
    payload = _load_json(path, "test manifest")
    if payload.get("split") != "test":
        raise ValueError("fixed manifest split must be test")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("fixed test manifest samples must be nonempty")
    keys = []
    labels = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or "lmdb_key" not in sample:
            raise ValueError(
                f"fixed test manifest sample {index} has no lmdb_key"
            )
        key = str(sample["lmdb_key"])
        if not key:
            raise ValueError(
                f"fixed test manifest sample {index} has an empty key"
            )
        keys.append(key)
        labels.append(
            _finite_float(
                sample.get("label"),
                f"fixed test manifest label for {key}",
            )
        )
    if len(keys) != len(set(keys)):
        raise ValueError("fixed test manifest contains duplicate keys")
    return payload, keys, np.asarray(labels, dtype=np.float64)


def _build_test_dataset(manifest_path, native_cache_path):
    from data_utils.hybrid_affinity_dataset import HybridAffinityDataset
    from data_utils.lmdb_dataset import PDBbindLmdbDataset
    from data_utils.local_interface_dataset import NativeInterfaceDataset

    manifest, _, _ = _load_test_manifest(manifest_path)
    gerna_dataset = PDBbindLmdbDataset(
        os.fspath(manifest_path),
        protein_3d_mode="ligand_pocket_residue_esm",
        protein_embedding_mode="esm",
        ligand_pocket_cutoff=8.0,
        disable_protein_2d=True,
    )
    interface_dataset = NativeInterfaceDataset(
        os.fspath(manifest_path), os.fspath(native_cache_path)
    )
    return HybridAffinityDataset(
        manifest["samples"], gerna_dataset, interface_dataset
    )


def _close_dataset(dataset):
    for name in ("gerna_dataset", "interface_dataset"):
        owner = getattr(dataset, name, None)
        close = getattr(owner, "close", None)
        if callable(close):
            close()


def _evaluate_selected_model(record, selection, dataset, device):
    from torch.utils.data import DataLoader

    from data_utils.hybrid_affinity_dataset import collate_hybrid_group
    from dta.train_hybrid_affinity import (
        build_gerna_backbone,
        build_interface_backbone,
    )
    from net.hybrid_affinity import (
        HybridAffinityFusion,
        HybridAffinityModel,
    )

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("final hybrid evaluation requires CUDA")
    torch.cuda.set_device(device)
    payload = torch.load(record["model_path"], map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or not isinstance(payload.get("model_state_dict"), dict)
    ):
        raise RuntimeError(
            f"selected model checkpoint schema differs: "
            + record["model_path"]
        )
    model = None
    try:
        gerna = build_gerna_backbone()
        interface = build_interface_backbone()
        fusion = HybridAffinityFusion(
            fusion_type=selection["fusion_type"],
            token_dim=128,
            dropout=0.1,
        )
        model = HybridAffinityModel(gerna, interface, fusion)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_hybrid_group,
            pin_memory=False,
            persistent_workers=False,
        )
        return evaluate_hybrid_affinity(
            model,
            loader,
            device,
            amp_enabled=True,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _validated_evaluations(
    evaluations, expected_keys, expected_labels
):
    evaluations = list(evaluations)
    if len(evaluations) != len(EXPECTED_SEEDS):
        raise ValueError("one evaluation per fixed seed is required")
    expected_keys = list(expected_keys)
    expected_labels = np.asarray(expected_labels, dtype=np.float64)
    columns = []
    model_metrics = []
    for seed, evaluation in zip(EXPECTED_SEEDS, evaluations):
        if not isinstance(evaluation, dict):
            raise ValueError(f"evaluation is malformed for seed {seed}")
        try:
            keys = [str(key) for key in evaluation["keys"]]
            labels = np.asarray(
                evaluation["labels"], dtype=np.float64
            )
            predictions = np.asarray(
                evaluation["predictions"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"evaluation is malformed for seed {seed}"
            ) from error
        if keys != expected_keys:
            raise ValueError(
                f"test key order differs for seed {seed}"
            )
        if (
            labels.shape != expected_labels.shape
            or not np.allclose(
                labels,
                expected_labels,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"test labels differ for seed {seed}")
        if (
            predictions.shape != expected_labels.shape
            or not np.isfinite(predictions).all()
        ):
            raise ValueError(
                f"test predictions are invalid for seed {seed}"
            )
        columns.append(predictions)
        model_metrics.append(
            {
                "seed": seed,
                "metrics": _compute_metrics(labels, predictions),
            }
        )
    return np.stack(columns, axis=1), model_metrics


def _compute_metrics(labels, predictions):
    from utils.metrics import get_score_metrics

    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if (
        labels.ndim != 1
        or predictions.shape != labels.shape
        or not np.isfinite(labels).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("test metric inputs are invalid")
    values = get_score_metrics(predictions, labels)
    metrics = {
        name: float(value)
        for name, value in zip(REQUIRED_METRICS, values)
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("final test metrics are nonfinite")
    return metrics


def run_final_evaluation(
    *,
    selection_manifest_path,
    test_manifest_path,
    test_native_cache_path,
    output_dir,
    device,
    confirm_final_test,
    expected_test_manifest=FIXED_TEST_MANIFEST,
    expected_test_native_cache=FIXED_TEST_NATIVE_CACHE,
    dataset_builder=None,
    model_evaluator=None,
):
    request = validate_final_request(
        selection_manifest_path,
        test_manifest_path,
        test_native_cache_path,
        output_dir,
        confirm_final_test=confirm_final_test,
        expected_test_manifest=expected_test_manifest,
        expected_test_native_cache=expected_test_native_cache,
    )
    selection = load_and_validate_selection(
        request["selection_manifest"]
    )
    _, expected_keys, expected_labels = _load_test_manifest(
        request["test_manifest"]
    )
    test_binding = build_preflight_binding(
        request["test_manifest"], request["test_native_cache"]
    )
    if test_binding["ordered_key_sha256"] != hash_ordered_keys(
        expected_keys
    ):
        raise RuntimeError("fixed test binding key hash differs")

    if dataset_builder is None:
        dataset_builder = _build_test_dataset
    if model_evaluator is None:
        model_evaluator = _evaluate_selected_model
    dataset = dataset_builder(
        request["test_manifest"], request["test_native_cache"]
    )
    try:
        evaluations = [
            model_evaluator(
                record,
                selection,
                dataset,
                torch.device(device),
            )
            for record in selection["records"]
        ]
    finally:
        _close_dataset(dataset)

    matrix, model_metrics = _validated_evaluations(
        evaluations, expected_keys, expected_labels
    )
    predictions = (
        matrix @ selection["weights"] + selection["intercept"]
    )
    if not np.isfinite(predictions).all():
        raise FloatingPointError(
            "final ensemble predictions are nonfinite"
        )
    test_metrics = _compute_metrics(expected_labels, predictions)
    deltas = {
        name: float(
            test_metrics[name] - FIXED_FUSION_TEST_METRICS[name]
        )
        for name in REQUIRED_METRICS
    }

    output = request["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "test_predictions.npz"
    metrics_path = output / "final_test_metrics.json"
    snapshot_path = output / "final_selection_snapshot.json"
    atomic_write_predictions(
        prediction_path,
        {
            "keys": expected_keys,
            "labels": expected_labels,
            "predictions": predictions,
        },
    )
    metrics_payload = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "split": "test",
        "selection_manifest": selection["path"],
        "selection_manifest_sha256": selection["sha256"],
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_key_sha256": hash_ordered_keys(expected_keys),
        "test_metrics": test_metrics,
        "per_seed_test_metrics": model_metrics,
        "fixed_fusion_test_metrics": dict(
            FIXED_FUSION_TEST_METRICS
        ),
        "metric_deltas_vs_fixed_fusion": deltas,
        "improved_test_ci": bool(deltas["CI"] > 0.0),
    }
    snapshot = {
        "schema_version": 1,
        "selection_manifest": selection["path"],
        "selection_manifest_sha256": selection["sha256"],
        "selection": selection["payload"],
        "verified_selected_candidates": [
            {
                "seed": record["seed"],
                "path": record["selected_candidate_path"],
                "sha256": record["selected_candidate_sha256"],
                "training_checkpoint": record[
                    "training_checkpoint"
                ],
                "training_checkpoint_sha256": record[
                    "training_checkpoint_sha256"
                ],
                "provenance_hashes": record["provenance_hashes"],
            }
            for record in selection["records"]
        ],
        "fixed_test_binding": {
            "manifest_path": str(request["test_manifest"]),
            "native_cache_path": str(request["test_native_cache"]),
            **test_binding,
        },
        "device": str(torch.device(device)),
    }
    atomic_write_json(metrics_path, metrics_payload)
    atomic_write_json(snapshot_path, snapshot)
    return metrics_payload


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen hybrid selection on the fixed test split."
        )
    )
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--test-native-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        required=True,
        help="Acknowledge that this is the one final test evaluation.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_final_evaluation(
        selection_manifest_path=args.selection_manifest,
        test_manifest_path=args.test_manifest,
        test_native_cache_path=args.test_native_cache,
        output_dir=args.output_dir,
        device=args.device,
        confirm_final_test=args.confirm_final_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
