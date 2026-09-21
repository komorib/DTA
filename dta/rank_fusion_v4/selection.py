"""Validation-only candidate selection and sealing for Rank Fusion v4."""

import json
import math
from pathlib import Path
import shutil

import numpy as np

from utils.experiment_integrity import atomic_write_json, sha256_file

from .metrics import affinity_metrics


SEEDS = ("1", "7", "21", "42", "87")
EXPERTS = ("upper_bound", "boltz_pairwise", "three_dprot")


def rank_percentiles(values):
    """Return stable zero-based average ranks shifted by 0.5 and divided by N."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("rank values must be finite and nonempty")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return (ranks + 0.5) / float(len(values))


def _metric_key(candidate):
    metrics = candidate["metrics"]
    return (
        float(metrics["CI"]),
        float(metrics["Spearman"]),
        -float(metrics["RMSE"]),
    )


def _load_run(run_dir, seed, cache):
    path = Path(run_dir).expanduser().resolve()
    summary_path = path / "summary.json"
    predictions_path = path / "valid_predictions_best.npz"
    checkpoint_path = path / "checkpoint_best.pt"
    for required in (summary_path, predictions_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "complete" or str(summary.get("seed")) != seed:
        raise ValueError(f"seed {seed} run summary is incomplete or mismatched")
    run_config = summary.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"seed {seed} run config is missing")
    if run_config.get("valid_manifest_sha256") != cache.metadata.get(
        "manifest_sha256"
    ):
        raise ValueError(f"seed {seed} validation manifest lineage does not match")
    if run_config.get("normalization_sha256") != cache.metadata.get(
        "normalization_sha256"
    ):
        raise ValueError(f"seed {seed} normalization lineage does not match")
    try:
        with np.load(str(predictions_path), allow_pickle=False) as archive:
            required_names = {"keys", "labels", "predictions"}
            if not required_names.issubset(archive.files):
                raise ValueError(f"seed {seed} prediction artifact is incomplete")
            keys = tuple(str(value) for value in archive["keys"].tolist())
            labels = np.asarray(archive["labels"], dtype=np.float64).reshape(-1)
            predictions = np.asarray(
                archive["predictions"], dtype=np.float64
            ).reshape(-1)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "seed" in str(error):
            raise
        raise ValueError(f"seed {seed} prediction artifact is invalid") from error
    if keys != cache.keys:
        raise ValueError(f"seed {seed} validation keys do not match cache")
    if labels.shape != cache.labels.shape or not np.allclose(
        labels, cache.labels, rtol=0.0, atol=1e-5
    ):
        raise ValueError(f"seed {seed} validation labels do not match cache")
    if predictions.shape != cache.labels.shape or not np.isfinite(predictions).all():
        raise ValueError(f"seed {seed} validation predictions are invalid")
    metrics = affinity_metrics(cache.labels, predictions)
    return {
        "name": "seed" + seed,
        "candidate_type": "v4_seed",
        "members": [seed],
        "metrics": metrics,
        "predictions": predictions,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_summary_sha256": sha256_file(summary_path),
    }


def _expert_candidates(cache):
    candidates = []
    coverage = {}
    upper_bound = np.asarray(cache.predictions[:, 0], dtype=np.float64)
    for index, expert in enumerate(EXPERTS):
        available = cache.available[:, index]
        predictions = np.where(
            available,
            cache.predictions[:, index],
            cache.predictions[:, 0],
        ).astype(np.float64, copy=False)
        fallback_count = int((~available).sum())
        coverage[expert] = {
            "available_count": int(available.sum()),
            "missing_count": fallback_count,
            "fallback_count": fallback_count,
            "fallback_expert": None if index == 0 else "upper_bound",
        }
        candidates.append(
            {
                "name": expert,
                "candidate_type": "single_expert",
                "members": [expert],
                "metrics": affinity_metrics(cache.labels, predictions),
                "predictions": predictions,
                "fallback_count": fallback_count,
            }
        )
    if not np.isfinite(upper_bound).all():
        raise ValueError("Upper-Bound validation predictions are nonfinite")
    return candidates, coverage


def _fit_affine(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centered = source - source.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        slope = 0.0
    else:
        slope = float(np.dot(centered, target - target.mean()) / denominator)
    intercept = float(target.mean() - slope * source.mean())
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise FloatingPointError("validation affine calibration is nonfinite")
    return slope, intercept


def _ensemble_candidates(v4_candidates, labels):
    ordered = sorted(v4_candidates, key=_metric_key, reverse=True)
    top_three = ordered[:3]
    if len(top_three) != 3:
        raise ValueError("exactly three top v4 seeds are required for ensembles")
    members = [candidate["members"][0] for candidate in top_three]
    stack = np.stack([candidate["predictions"] for candidate in top_three], axis=0)
    raw_prediction = stack.mean(axis=0)
    raw = {
        "name": "raw_ensemble",
        "candidate_type": "raw_ensemble",
        "members": members,
        "metrics": affinity_metrics(labels, raw_prediction),
        "predictions": raw_prediction,
        "ensemble_rule": "arithmetic_mean",
        "affine_calibration": {"slope": 1.0, "intercept": 0.0},
    }
    percentile = np.stack(
        [rank_percentiles(candidate["predictions"]) for candidate in top_three],
        axis=0,
    ).mean(axis=0)
    slope, intercept = _fit_affine(percentile, labels)
    rank_prediction = intercept + slope * percentile
    rank = {
        "name": "rank_ensemble",
        "candidate_type": "rank_ensemble",
        "members": members,
        "metrics": affinity_metrics(labels, rank_prediction),
        "predictions": rank_prediction,
        "ensemble_rule": "stable_average_percentile_mean",
        "affine_calibration": {"slope": slope, "intercept": intercept},
    }
    return [raw, rank]


def _public_candidate(candidate):
    names = (
        "name",
        "candidate_type",
        "members",
        "metrics",
        "fallback_count",
        "checkpoint_sha256",
        "run_summary_sha256",
        "ensemble_rule",
        "affine_calibration",
    )
    return {name: candidate[name] for name in names if name in candidate}


def _package_hashes(package):
    hashes = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "seal.json":
            hashes[str(path.relative_to(package))] = sha256_file(path)
    return hashes


def verify_sealed_package(package_dir):
    """Verify exact package inventory and every sealed SHA256."""
    package = Path(package_dir).expanduser().resolve()
    seal_path = package / "seal.json"
    if not seal_path.is_file():
        raise ValueError("selection seal is missing")
    with seal_path.open("r", encoding="utf-8") as handle:
        seal = json.load(handle)
    if seal.get("status") != "sealed" or not isinstance(seal.get("files"), dict):
        raise ValueError("selection seal is invalid")
    actual = _package_hashes(package)
    if actual != seal["files"]:
        raise ValueError("selection seal does not match package files")
    selection_path = package / "selection.json"
    if "selection.json" not in actual or not selection_path.is_file():
        raise ValueError("sealed selection.json is missing")
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "complete":
        raise ValueError("sealed selection is incomplete")
    return selection


def select_validation_package(
    valid_cache,
    run_dirs,
    output_dir,
    *,
    expected_count=2340,
    expected_seeds=SEEDS,
):
    """Select on validation only and atomically finish with a package seal."""
    if len(valid_cache.keys) != int(expected_count):
        raise ValueError("validation cache count does not match expected_count")
    if valid_cache.metadata.get("split") != "valid":
        raise ValueError("selection requires a validation cache")
    expected_seeds = tuple(str(seed) for seed in expected_seeds)
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("expected_seeds must be unique and nonempty")
    if set(str(seed) for seed in run_dirs) != set(expected_seeds):
        raise ValueError("selection requires seeds " + ", ".join(expected_seeds))
    v4_candidates = [
        _load_run(run_dirs[seed], seed, valid_cache) for seed in expected_seeds
    ]
    expert_candidates, coverage = _expert_candidates(valid_cache)
    ensemble_candidates = (
        _ensemble_candidates(v4_candidates, valid_cache.labels)
        if len(v4_candidates) >= 3
        else []
    )
    learned_candidates = v4_candidates + ensemble_candidates
    best_learned = max(learned_candidates, key=_metric_key)
    best_expert = max(expert_candidates, key=_metric_key)
    if float(best_learned["metrics"]["CI"]) > float(best_expert["metrics"]["CI"]):
        selected = best_learned
    else:
        selected = best_expert

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("selection output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    packaged_checkpoint_hashes = {}
    if selected["candidate_type"] != "single_expert":
        checkpoints = output / "checkpoints"
        checkpoints.mkdir()
        by_seed = {candidate["members"][0]: candidate for candidate in v4_candidates}
        for member in selected["members"]:
            destination = checkpoints / f"{member}.pt"
            shutil.copy2(by_seed[member]["checkpoint_path"], destination)
            packaged_checkpoint_hashes[member] = sha256_file(destination)

    selected_type = selected["candidate_type"]
    selection = {
        "status": "complete",
        "selected_type": selected_type,
        "selected_name": selected["name"],
        "members": list(selected["members"]),
        "selected_validation_metrics": selected["metrics"],
        "ensemble_rule": selected.get("ensemble_rule"),
        "affine_calibration": selected.get(
            "affine_calibration", {"slope": 1.0, "intercept": 0.0}
        ),
        "target_manifest_sha256": valid_cache.metadata.get("manifest_sha256"),
        "target_ordered_key_sha256": valid_cache.metadata.get("ordered_key_sha256"),
        "target_count": len(valid_cache.keys),
        "normalization_sha256": valid_cache.metadata.get("normalization_sha256"),
        "expert_checkpoint_sha256": valid_cache.metadata.get(
            "expert_checkpoint_sha256"
        ),
        "packaged_checkpoint_sha256": packaged_checkpoint_hashes,
        "expert_coverage": coverage,
        "all_candidates": [
            _public_candidate(candidate)
            for candidate in expert_candidates + v4_candidates + ensemble_candidates
        ],
        "selection_order": ["CI", "Spearman", "-RMSE"],
        "baseline_guard": "learned CI must strictly exceed best single-expert CI",
    }
    atomic_write_json(output / "selection.json", selection)
    seal = {
        "status": "sealed",
        "schema_version": 1,
        "files": _package_hashes(output),
    }
    atomic_write_json(output / "seal.json", seal)
    verify_sealed_package(output)
    return selection
