"""Independent sealed-package test evaluator for Rank Fusion v4."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.experiment_integrity import atomic_write_json, sha256_file

from .cache import EXPERT_ORDER, load_fusion_cache
from .data import FusionCacheDataset, collate_fusion_cache
from .metrics import affinity_metrics
from .model import RankFusionV4


SELECTION_TYPES = frozenset(
    {"single_expert", "v4_seed", "raw_ensemble", "rank_ensemble"}
)


def _package_hashes(package):
    hashes = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "seal.json":
            hashes[str(path.relative_to(package))] = sha256_file(path)
    return hashes


def _load_sealed_selection(package_dir):
    package = Path(package_dir).expanduser().resolve()
    seal_path = package / "seal.json"
    selection_path = package / "selection.json"
    if not seal_path.is_file() or not selection_path.is_file():
        raise ValueError("selection seal or selection.json is missing")
    with seal_path.open("r", encoding="utf-8") as handle:
        seal = json.load(handle)
    if seal.get("status") != "sealed" or not isinstance(seal.get("files"), dict):
        raise ValueError("selection seal is invalid")
    if _package_hashes(package) != seal["files"]:
        raise ValueError("selection seal does not match package files")
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "complete":
        raise ValueError("sealed selection is incomplete")
    if selection.get("selected_type") not in SELECTION_TYPES:
        raise ValueError("sealed selection type is invalid")
    members = selection.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("sealed selection members are invalid")
    return package, selection, seal


def _validate_lineage(selection, cache, expected_count):
    if cache.metadata.get("split") != "test":
        raise ValueError("test evaluator requires a test cache")
    if len(cache.keys) != int(expected_count):
        raise ValueError("test cache count does not match expected_count")
    if len(cache.keys) != len(set(cache.keys)):
        raise ValueError("test cache contains duplicate keys")
    if len(cache.keys) != len(cache.labels):
        raise ValueError("test cache key and label counts do not match")
    if selection.get("normalization_sha256") != cache.metadata.get(
        "normalization_sha256"
    ):
        raise ValueError("selection and test normalization hashes do not match")
    if selection.get("expert_checkpoint_sha256") != cache.metadata.get(
        "expert_checkpoint_sha256"
    ):
        raise ValueError("selection and test expert checkpoint hashes do not match")


def _model_from_checkpoint(path, expected_sha256, device):
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("packaged checkpoint hash does not match selection")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("packaged checkpoint is invalid")
    config = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("packaged checkpoint model metadata is incomplete")
    if config.get("class") != "RankFusionV4":
        raise ValueError("packaged checkpoint model class does not match")
    model = RankFusionV4(
        tuple(config["latent_dims"]),
        model_dim=int(config["model_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        drop_path_rate=float(config["drop_path_rate"]),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _move_batch(batch, device):
    return {
        "raw_predictions": batch["raw_predictions"].to(device),
        "standardized_predictions": batch["standardized_predictions"].to(device),
        "available": batch["available"].to(device),
        "latents": [value.to(device) for value in batch["latents"]],
    }


@torch.inference_mode()
def _predict_model(model, cache, device, batch_size):
    loader = DataLoader(
        FusionCacheDataset(cache),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fusion_cache,
    )
    predictions = []
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(
            moved["latents"],
            moved["raw_predictions"],
            moved["standardized_predictions"],
            moved["available"],
        )
        value = output["prediction"].float()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("packaged model produced nonfinite predictions")
        predictions.append(value.cpu().numpy())
    result = np.concatenate(predictions).astype(np.float64, copy=False)
    if result.shape != cache.labels.shape:
        raise RuntimeError("packaged model prediction count does not match test cache")
    return result


def _rank_percentiles(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return (ranks + 0.5) / float(len(values))


def _single_expert_predictions(selection, cache):
    expert = selection.get("selected_name")
    if expert not in EXPERT_ORDER:
        members = selection.get("members", [])
        expert = members[0] if len(members) == 1 else None
    if expert not in EXPERT_ORDER:
        raise ValueError("selected single expert is invalid")
    index = EXPERT_ORDER.index(expert)
    return np.where(
        cache.available[:, index],
        cache.predictions[:, index],
        cache.predictions[:, 0],
    ).astype(np.float64, copy=False)


def _v4_member_predictions(package, selection, cache, device, batch_size):
    members = selection["members"]
    checkpoint_hashes = selection.get("packaged_checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != set(members):
        raise ValueError("packaged checkpoint inventory does not match members")
    predictions = []
    for member in members:
        checkpoint_path = package / "checkpoints" / f"{member}.pt"
        model = _model_from_checkpoint(
            checkpoint_path, checkpoint_hashes[member], device
        )
        predictions.append(
            _predict_model(model, cache, device, batch_size)
        )
    return predictions


def _selected_predictions(package, selection, cache, device, batch_size):
    selected_type = selection["selected_type"]
    if selected_type == "single_expert":
        return _single_expert_predictions(selection, cache)
    member_predictions = _v4_member_predictions(
        package, selection, cache, device, batch_size
    )
    if selected_type == "v4_seed":
        if len(member_predictions) != 1:
            raise ValueError("v4_seed selection requires exactly one member")
        return member_predictions[0]
    if len(member_predictions) < 2:
        raise ValueError("ensemble selection requires multiple members")
    if selected_type == "raw_ensemble":
        if selection.get("ensemble_rule") != "arithmetic_mean":
            raise ValueError("raw ensemble rule does not match")
        return np.stack(member_predictions, axis=0).mean(axis=0)
    if selection.get("ensemble_rule") != "stable_average_percentile_mean":
        raise ValueError("rank ensemble rule does not match")
    percentiles = np.stack(
        [_rank_percentiles(values) for values in member_predictions], axis=0
    ).mean(axis=0)
    calibration = selection.get("affine_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("rank ensemble affine calibration is missing")
    try:
        slope = float(calibration["slope"])
        intercept = float(calibration["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("rank ensemble affine calibration is invalid") from error
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("rank ensemble affine calibration is nonfinite")
    return intercept + slope * percentiles


def _atomic_npz(path, **arrays):
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=".test-predictions-", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_sealed_selection(
    selection_dir,
    test_cache,
    output_dir,
    *,
    device,
    expected_count=4751,
    batch_size=256,
):
    """Evaluate one sealed validation choice exactly once on the test cache."""
    package, selection, seal = _load_sealed_selection(selection_dir)
    _validate_lineage(selection, test_cache, expected_count)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("test output directory must be empty")
    output_was_empty = not output.exists() or not any(output.iterdir())
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA test evaluation requested but CUDA is unavailable")
    predictions = _selected_predictions(
        package,
        selection,
        test_cache,
        device,
        int(batch_size),
    )
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if predictions.shape != test_cache.labels.shape or not np.isfinite(predictions).all():
        raise ValueError("selected test predictions are invalid")
    metrics = affinity_metrics(test_cache.labels, predictions)
    coverage = {
        "status": "complete",
        "test_count": len(test_cache.keys),
        "experts": {
            expert: {
                "available_count": int(test_cache.available[:, index].sum()),
                "missing_count": int((~test_cache.available[:, index]).sum()),
                "fallback_count": int((~test_cache.available[:, index]).sum()),
            }
            for index, expert in enumerate(EXPERT_ORDER)
        },
    }
    _atomic_npz(
        output / "test_predictions_selected.npz",
        keys=np.asarray(test_cache.keys, dtype=np.str_),
        labels=np.asarray(test_cache.labels, dtype=np.float32),
        predictions=np.asarray(predictions, dtype=np.float32),
    )
    atomic_write_json(output / "test_metrics_selected.json", metrics)
    atomic_write_json(output / "test_expert_coverage.json", coverage)
    summary = {
        "status": "complete",
        "test_count": len(test_cache.keys),
        "test_manifest_sha256": test_cache.metadata.get("manifest_sha256"),
        "test_ordered_key_sha256": test_cache.metadata.get("ordered_key_sha256"),
        "normalization_sha256": test_cache.metadata.get("normalization_sha256"),
        "expert_checkpoint_sha256": test_cache.metadata.get(
            "expert_checkpoint_sha256"
        ),
        "selection_seal_sha256": sha256_file(package / "seal.json"),
        "selection_files": seal["files"],
        "selected_type": selection["selected_type"],
        "selected_name": selection.get("selected_name"),
        "members": selection["members"],
        "selected_validation_metrics": selection.get(
            "selected_validation_metrics"
        ),
        "test_metrics": metrics,
        "output_was_empty": output_was_empty,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a sealed Rank Fusion v4 selection on test"
    )
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-count", type=int, default=4751)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cache = load_fusion_cache(args.test_cache, expected_split="test")
    summary = evaluate_sealed_selection(
        args.selection,
        cache,
        args.output,
        device=args.device,
        expected_count=args.expected_count,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
