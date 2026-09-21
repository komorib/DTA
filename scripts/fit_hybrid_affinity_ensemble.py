#!/usr/bin/env python
"""Fit a bounded hybrid-affinity ensemble using validation predictions."""

import argparse
import json
import math
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np

from dta.hybrid_affinity_training import atomic_write_predictions
from scripts.select_hybrid_affinity_runs import (
    REQUIRED_METRICS,
    align_prediction_archives,
    candidate_is_better,
    load_prediction_archive,
    select_run,
)
from utils.experiment_integrity import (
    atomic_write_json,
    sha256_file,
)


EXPECTED_SEEDS = (1, 7, 21, 42, 87)


def fit_bounded_linear_ensemble(
    prediction_matrix,
    labels,
    *,
    l2_penalty=1e-3,
    maximum_iterations=10000,
    tolerance=1e-12,
):
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("prediction matrix must be nonempty and 2D")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("labels do not align with prediction matrix")
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError("ensemble inputs contain nonfinite values")
    l2_penalty = float(l2_penalty)
    if not math.isfinite(l2_penalty) or l2_penalty <= 0.0:
        raise ValueError("l2_penalty must be finite and positive")
    maximum_iterations = int(maximum_iterations)
    tolerance = float(tolerance)
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    sample_count, model_count = matrix.shape
    augmented = np.column_stack(
        [matrix, np.ones(sample_count, dtype=np.float64)]
    )
    penalty = np.zeros(
        (model_count + 1, model_count + 1), dtype=np.float64
    )
    penalty[:model_count, :model_count] = (
        np.eye(model_count, dtype=np.float64) * l2_penalty
    )
    normal = augmented.T @ augmented / sample_count + penalty
    target = augmented.T @ labels / sample_count
    parameters = np.linalg.pinv(normal) @ target
    parameters[:model_count] = np.clip(
        parameters[:model_count], -2.0, 2.0
    )

    hessian = 2.0 * normal
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        raise RuntimeError("ensemble objective has invalid curvature")
    step_size = 1.0 / lipschitz
    converged = False
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        residual = augmented @ parameters - labels
        gradient = (
            2.0 * augmented.T @ residual / sample_count
            + 2.0 * penalty @ parameters
        )
        updated = parameters - step_size * gradient
        updated[:model_count] = np.clip(
            updated[:model_count], -2.0, 2.0
        )
        difference = float(np.max(np.abs(updated - parameters)))
        parameters = updated
        if difference <= tolerance:
            converged = True
            break

    weights = parameters[:model_count]
    intercept = float(parameters[-1])
    residual = matrix @ weights + intercept - labels
    objective = float(
        np.mean(residual * residual)
        + l2_penalty * np.dot(weights, weights)
    )
    if (
        not np.isfinite(weights).all()
        or not math.isfinite(intercept)
        or not math.isfinite(objective)
    ):
        raise FloatingPointError("bounded ensemble fit is nonfinite")
    return {
        "weights": weights.copy(),
        "intercept": intercept,
        "objective": objective,
        "iterations": int(iterations),
        "converged": bool(converged),
        "l2_penalty": l2_penalty,
    }


def _compute_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if (
        labels.ndim != 1
        or predictions.shape != labels.shape
        or not np.isfinite(labels).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("metric inputs are invalid")
    from utils.metrics import get_score_metrics

    values = get_score_metrics(predictions, labels)
    result = {
        name: float(value)
        for name, value in zip(REQUIRED_METRICS, values)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(
            "validation ensemble metrics are nonfinite"
        )
    return result


def _calibrated_intercept(matrix, labels, weights):
    predictions = matrix @ weights
    return float(np.mean(labels - predictions))


def _ensemble_candidates(matrix, labels, optimized):
    model_count = matrix.shape[1]
    definitions = []
    for index in range(model_count):
        weights = np.zeros(model_count, dtype=np.float64)
        weights[index] = 1.0
        definitions.append((f"single_seed_index_{index}", weights))
    definitions.append(
        (
            "equal_weight",
            np.full(
                model_count, 1.0 / model_count, dtype=np.float64
            ),
        )
    )
    definitions.append(
        (
            "bounded_ridge",
            np.asarray(optimized["weights"], dtype=np.float64),
        )
    )

    candidates = []
    for name, weights in definitions:
        if name == "bounded_ridge":
            intercept = float(optimized["intercept"])
        else:
            intercept = _calibrated_intercept(
                matrix, labels, weights
            )
        predictions = matrix @ weights + intercept
        candidates.append(
            {
                "name": name,
                "epoch": 0,
                "weights": weights.copy(),
                "intercept": intercept,
                "predictions": predictions,
                "metrics": _compute_metrics(labels, predictions),
            }
        )
    return candidates


def _validated_record(record):
    if not isinstance(record, dict):
        raise ValueError("selected model record must be a mapping")
    try:
        seed = int(record["seed"])
        fusion_type = str(record["fusion_type"])
        rank_weight = float(record["rank_weight"])
        model_path = Path(record["model_path"]).expanduser().resolve()
        prediction_path = (
            Path(record["prediction_path"]).expanduser().resolve()
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("selected model record is malformed") from error
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected final seed: {seed}")
    if fusion_type not in {"mlp", "transformer"}:
        raise ValueError(f"unexpected fusion type: {fusion_type}")
    if rank_weight not in {0.1, 0.2, 0.4}:
        raise ValueError(f"unexpected rank weight: {rank_weight}")
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))
    if not prediction_path.is_file():
        raise FileNotFoundError(str(prediction_path))
    return {
        "seed": seed,
        "fusion_type": fusion_type,
        "rank_weight": rank_weight,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
    }


def fit_ensemble_selection(
    selected_records,
    output_path,
    *,
    l2_penalty=1e-3,
):
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(str(output_path))
    records = sorted(
        [_validated_record(record) for record in selected_records],
        key=lambda record: record["seed"],
    )
    seeds = tuple(record["seed"] for record in records)
    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            "final ensemble requires seeds "
            + ", ".join(str(seed) for seed in EXPECTED_SEEDS)
        )
    configurations = {
        (record["fusion_type"], record["rank_weight"])
        for record in records
    }
    if len(configurations) != 1:
        raise ValueError(
            "final ensemble model configurations differ"
        )
    fusion_type, rank_weight = next(iter(configurations))

    archives = [
        load_prediction_archive(record["prediction_path"])
        for record in records
    ]
    aligned = align_prediction_archives(archives)
    matrix = aligned["matrix"]
    labels = aligned["labels"]
    optimized = fit_bounded_linear_ensemble(
        matrix, labels, l2_penalty=l2_penalty
    )
    candidates = _ensemble_candidates(matrix, labels, optimized)
    selected = candidates[0]
    for challenger in candidates[1:]:
        if candidate_is_better(challenger, selected):
            selected = challenger

    weights = np.asarray(selected["weights"], dtype=np.float64)
    if np.any(weights < -2.0) or np.any(weights > 2.0):
        raise RuntimeError("selected ensemble weights exceed bounds")
    validation_prediction_path = output_path.with_name(
        output_path.stem + "_valid_predictions.npz"
    )
    atomic_write_predictions(
        validation_prediction_path,
        {
            "keys": aligned["keys"],
            "labels": labels,
            "predictions": selected["predictions"],
        },
    )
    result = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "selection_split": "valid",
        "fusion_type": fusion_type,
        "rank_weight": rank_weight,
        "seeds": list(seeds),
        "model_paths": [record["model_path"] for record in records],
        "model_sha256": [
            record["model_sha256"] for record in records
        ],
        "prediction_paths": [
            record["prediction_path"] for record in records
        ],
        "prediction_sha256": [
            record["prediction_sha256"] for record in records
        ],
        "weights": [float(weight) for weight in weights],
        "intercept": float(selected["intercept"]),
        "validation_metrics": {
            name: float(selected["metrics"][name])
            for name in REQUIRED_METRICS
        },
        "prediction_key_sha256": aligned[
            "prediction_key_sha256"
        ],
        "validation_prediction_path": str(
            validation_prediction_path.resolve()
        ),
        "validation_prediction_sha256": sha256_file(
            validation_prediction_path
        ),
        "ensemble_candidate": str(selected["name"]),
        "l2_penalty": float(l2_penalty),
        "bounded_fit": {
            "objective": float(optimized["objective"]),
            "iterations": int(optimized["iterations"]),
            "converged": bool(optimized["converged"]),
        },
    }
    atomic_write_json(output_path, result)
    return result


def build_from_run_root(
    run_root,
    output_path,
    *,
    device="cuda:0",
    l2_penalty=1e-3,
):
    run_root = Path(run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(str(run_root))
    run_dirs = sorted(
        path.parent
        for path in run_root.glob("*/run_config.json")
        if path.is_file()
    )
    records = [
        select_run(
            run_dir,
            device=device,
            require_full_stage=True,
        )
        for run_dir in run_dirs
    ]
    return fit_ensemble_selection(
        records, output_path, l2_penalty=l2_penalty
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a bounded hybrid affinity ensemble on validation data."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = build_from_run_root(
        args.run_root,
        args.output,
        device=args.device,
        l2_penalty=args.l2_penalty,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
