"""Validation-only selection for the LP-PDBBind upper-bound fusion."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np

from dta.hybrid_cached_training_v2 import _metrics


@dataclass(frozen=True)
class ValidationCandidate:
    name: str
    root: Path
    keys: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    provenance: dict


def load_validation_candidate(name, root):
    """Load one complete validation candidate and reject test-like artifacts."""
    root = Path(root).expanduser().resolve()
    prediction_path = root / "valid_predictions.npz"
    provenance_path = root / "provenance.json"
    if not prediction_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "candidate requires valid_predictions.npz and provenance.json"
        )
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("candidate provenance is unreadable") from error
    if not isinstance(provenance, dict) or provenance.get("split") != "valid":
        raise ValueError("candidate provenance must explicitly declare split=valid")
    if not isinstance(provenance.get("model_sha256"), str) or not provenance["model_sha256"]:
        raise ValueError("candidate provenance is missing model_sha256")
    if not isinstance(provenance.get("valid_manifest_sha256"), str) or not provenance["valid_manifest_sha256"]:
        raise ValueError("candidate provenance is missing valid_manifest_sha256")
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {"keys", "labels", "predictions"}
        if set(archive.files) != required:
            raise ValueError("candidate prediction archive has an invalid schema")
        keys = np.asarray(archive["keys"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.float64).reshape(-1)
        predictions = np.asarray(
            archive["predictions"], dtype=np.float64
        ).reshape(-1)
    if not len(keys) or len(keys) != len(labels) or len(keys) != len(predictions):
        raise ValueError("candidate key, label, and prediction lengths differ")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("candidate validation keys are not unique")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("candidate validation values must be finite")
    return ValidationCandidate(
        str(name), root, keys, labels, predictions, provenance
    )


def align_validation_candidates(candidates):
    """Require identical ordered validation observations across candidates."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("at least one validation candidate is required")
    reference = candidates[0]
    manifest_hash = reference.provenance["valid_manifest_sha256"]
    for candidate in candidates[1:]:
        if not np.array_equal(candidate.keys, reference.keys):
            raise ValueError("candidate validation keys differ")
        if not np.array_equal(candidate.labels, reference.labels):
            raise ValueError("candidate validation labels differ")
        if candidate.provenance["valid_manifest_sha256"] != manifest_hash:
            raise ValueError("candidate validation manifest hashes differ")
    return reference.keys.copy(), reference.labels.copy(), np.stack(
        [candidate.predictions for candidate in candidates], axis=1
    )


def _ensemble_objective(weights, labels, prediction_matrix):
    prediction = prediction_matrix.dot(weights)
    return -_metrics(labels, prediction)["CI"] + 1e-4 * float(
        np.dot(weights, weights)
    )


def _positive_affine_calibration(labels, raw_score):
    """Calibrate a ranking score to label units without reversing its order."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    raw_score = np.asarray(raw_score, dtype=np.float64).reshape(-1)
    if len(labels) != len(raw_score) or not len(labels):
        raise ValueError("labels and raw_score must have equal nonzero length")
    if not np.isfinite(labels).all() or not np.isfinite(raw_score).all():
        raise ValueError("calibration inputs must be finite")
    centered_score = raw_score - raw_score.mean()
    denominator = float(np.dot(centered_score, centered_score))
    if denominator <= np.finfo(np.float64).eps:
        slope = 1.0
    else:
        centered_labels = labels - labels.mean()
        slope = float(np.dot(centered_score, centered_labels) / denominator)
        slope = max(slope, np.finfo(np.float64).eps)
    intercept = float(labels.mean() - slope * raw_score.mean())
    predictions = slope * raw_score + intercept
    return slope, intercept, predictions


def fit_real_weight_ensemble(labels, prediction_matrix, *, seed=42, random_starts=16):
    """Fit unconstrained real-valued ensemble weights with deterministic starts."""
    from scipy.optimize import minimize

    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    prediction_matrix = np.asarray(prediction_matrix, dtype=np.float64)
    if prediction_matrix.ndim != 2 or prediction_matrix.shape[0] != len(labels):
        raise ValueError("prediction_matrix must have shape [N, members]")
    if prediction_matrix.shape[1] < 2:
        raise ValueError("an ensemble needs at least two members")
    if not np.isfinite(labels).all() or not np.isfinite(prediction_matrix).all():
        raise ValueError("ensemble inputs must be finite")
    random_starts = int(random_starts)
    if random_starts < 0:
        raise ValueError("random_starts must be nonnegative")
    members = prediction_matrix.shape[1]
    starts = []
    for index in range(members):
        start = np.zeros(members, dtype=np.float64)
        start[index] = 1.0
        starts.append(start)
    starts.append(np.full(members, 1.0 / members))
    least_squares, *_ = np.linalg.lstsq(
        np.column_stack([prediction_matrix, np.ones(len(labels))]),
        labels,
        rcond=None,
    )
    starts.append(least_squares[:-1])
    rng = np.random.RandomState(int(seed))
    for _ in range(random_starts):
        starts.append(rng.normal(size=members))
    results = [
        minimize(
            _ensemble_objective,
            start,
            args=(labels, prediction_matrix),
            method="Powell",
            options={"maxiter": 500, "xtol": 1e-5, "ftol": 1e-7},
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    ranking_weights = np.asarray(best.x, dtype=np.float64)
    raw_score = prediction_matrix.dot(ranking_weights)
    slope, intercept, prediction = _positive_affine_calibration(labels, raw_score)
    weights = slope * ranking_weights
    return {
        "weights": weights,
        "intercept": intercept,
        "predictions": prediction,
        "metrics": _metrics(labels, prediction),
        "member_metrics": [
            _metrics(labels, prediction_matrix[:, index])
            for index in range(members)
        ],
        "objective": float(best.fun),
        "optimizer_success": bool(best.success),
    }



def write_selection_package(output_dir, candidates, ensemble, checkpoint_paths):
    """Write a self-contained ensemble and verify its validation prediction."""
    import os
    import tempfile

    import torch

    from utils.experiment_integrity import sha256_file

    candidates = list(candidates)
    checkpoint_paths = [Path(path).expanduser().resolve() for path in checkpoint_paths]
    if not candidates or len(candidates) != len(checkpoint_paths):
        raise ValueError("package members and checkpoint paths must align")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keys, labels, prediction_matrix = align_validation_candidates(candidates)
    weights = np.asarray(ensemble["weights"], dtype=np.float64).reshape(-1)
    intercept = float(ensemble["intercept"])
    if weights.shape != (len(candidates),):
        raise ValueError("ensemble weight count does not match members")
    reconstructed = prediction_matrix.dot(weights) + intercept
    if not np.allclose(reconstructed, ensemble["predictions"], rtol=1e-10, atol=1e-10):
        raise RuntimeError("ensemble predictions cannot be reconstructed")
    members = []
    for candidate, checkpoint_path in zip(candidates, checkpoint_paths):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_sha256 != candidate.provenance["model_sha256"]:
            raise RuntimeError("candidate provenance model_sha256 does not match checkpoint")
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError("candidate checkpoint is missing model_state_dict")
        members.append({
            "name": candidate.name,
            "checkpoint_sha256": checkpoint_sha256,
            "state_dict": state_dict,
            "provenance": candidate.provenance,
        })
    package = {
        "format_version": 1,
        "members": members,
        "weights": torch.as_tensor(weights, dtype=torch.float64),
        "intercept": intercept,
        "ordered_keys": keys.tolist(),
        "labels": torch.as_tensor(labels, dtype=torch.float64),
        "validation_predictions": torch.as_tensor(reconstructed, dtype=torch.float64),
        "metrics": dict(ensemble["metrics"]),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix="Model_hybrid_upper_bound.", suffix=".tmp", dir=str(output_dir)
    )
    os.close(descriptor)
    try:
        torch.save(package, temporary)
        os.replace(temporary, output_dir / "Model_hybrid_upper_bound.pth")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    np.savez_compressed(
        output_dir / "valid_predictions.npz",
        keys=keys,
        labels=labels,
        predictions=reconstructed,
    )
    selection = {
        "members": [candidate.name for candidate in candidates],
        "weights": weights.tolist(),
        "intercept": intercept,
        "metrics": dict(ensemble["metrics"]),
        "member_metrics": list(ensemble["member_metrics"]),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True)
    )
    (output_dir / "ensemble.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True)
    )
    reloaded = torch.load(
        str(output_dir / "Model_hybrid_upper_bound.pth"), map_location="cpu"
    )
    replay = prediction_matrix.dot(
        reloaded["weights"].cpu().numpy()
    ) + float(reloaded["intercept"])
    if not np.allclose(replay, reconstructed, rtol=1e-10, atol=1e-10):
        raise RuntimeError("saved ensemble package does not reproduce validation predictions")
    return package



def load_validation_candidate_artifacts(
    name,
    prediction_path,
    provenance_path,
):
    """Load a candidate when predictions and provenance have distinct names."""
    prediction_path = Path(prediction_path).expanduser().resolve()
    provenance_path = Path(provenance_path).expanduser().resolve()
    if not prediction_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "candidate requires validation predictions and provenance"
        )
    staging = prediction_path.parent
    if (
        prediction_path.name == "valid_predictions.npz"
        and provenance_path.name == "provenance.json"
        and provenance_path.parent == staging
    ):
        return load_validation_candidate(name, staging)
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("candidate provenance is unreadable") from error
    if not isinstance(provenance, dict) or provenance.get("split") != "valid":
        raise ValueError("candidate provenance must explicitly declare split=valid")
    if not isinstance(provenance.get("model_sha256"), str) or not provenance["model_sha256"]:
        raise ValueError("candidate provenance is missing model_sha256")
    if not isinstance(provenance.get("valid_manifest_sha256"), str) or not provenance["valid_manifest_sha256"]:
        raise ValueError("candidate provenance is missing valid_manifest_sha256")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "labels", "predictions"}:
            raise ValueError("candidate prediction archive has an invalid schema")
        keys = np.asarray(archive["keys"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.float64).reshape(-1)
        predictions = np.asarray(archive["predictions"], dtype=np.float64).reshape(-1)
    if not len(keys) or len(keys) != len(labels) or len(keys) != len(predictions):
        raise ValueError("candidate key, label, and prediction lengths differ")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("candidate validation keys are not unique")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("candidate validation values must be finite")
    return ValidationCandidate(
        str(name), prediction_path.parent, keys, labels, predictions, provenance
    )


def _parse_candidate_spec(specification):
    fields = str(specification).split("=", 3)
    if len(fields) != 4 or not all(fields):
        raise ValueError(
            "candidate must be NAME=PREDICTIONS=PROVENANCE=CHECKPOINT"
        )
    return fields


def run_selection(args):
    """Select the highest validation-CI single model or real-weight ensemble."""
    from dta.hybrid_cached_training_v2 import is_better_validation

    parsed = [_parse_candidate_spec(value) for value in args.candidate]
    names = [value[0] for value in parsed]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    candidates = [
        load_validation_candidate_artifacts(name, prediction, provenance)
        for name, prediction, provenance, _ in parsed
    ]
    checkpoints = [checkpoint for _, _, _, checkpoint in parsed]
    _, labels, prediction_matrix = align_validation_candidates(candidates)
    fitted = fit_real_weight_ensemble(
        labels,
        prediction_matrix,
        seed=args.seed,
        random_starts=args.random_starts,
    )
    member_metrics = fitted["member_metrics"]
    best_index = 0
    for index in range(1, len(member_metrics)):
        if is_better_validation(member_metrics[index], member_metrics[best_index]):
            best_index = index
    if is_better_validation(fitted["metrics"], member_metrics[best_index]):
        package_candidates = candidates
        package_checkpoints = checkpoints
        package_ensemble = fitted
        selected_type = "ensemble"
    else:
        package_candidates = [candidates[best_index]]
        package_checkpoints = [checkpoints[best_index]]
        single_prediction = candidates[best_index].predictions
        package_ensemble = {
            "weights": np.asarray([1.0]),
            "intercept": 0.0,
            "predictions": single_prediction,
            "metrics": member_metrics[best_index],
            "member_metrics": [member_metrics[best_index]],
        }
        selected_type = "single"
    package = write_selection_package(
        args.output, package_candidates, package_ensemble, package_checkpoints
    )
    selection = {
        "status": "complete",
        "selected_type": selected_type,
        "selected_metrics": package_ensemble["metrics"],
        "ensemble_metrics": fitted["metrics"],
        "members": [candidate.name for candidate in package_candidates],
        "weights": package_ensemble["weights"].tolist(),
        "intercept": float(package_ensemble["intercept"]),
        "package": str(Path(args.output).expanduser().resolve()),
    }
    (Path(args.output).expanduser().resolve() / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True)
    )
    return selection, package


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Select a validation-only LP-PDBBind upper-bound ensemble."
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-starts", type=int, default=16)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    selection, _ = run_selection(args)
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
