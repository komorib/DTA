"""One-shot sealed official evaluation for calibration v6."""

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
import tempfile

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.selection import verify_validation_seal
from dta.anchored_expert_fusion_v5.training import StaticSpec, static_predictions
from dta.rank_fusion_v4.cache import load_fusion_cache
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import TEST_GATE
from .calibrators import MonotonicSplineCalibrator, PositiveAffineCalibrator
from .data import RankLockedCalibrationDataset
from .projection import rank_locked_project
from .selection import verify_validation_package
from .training import (
    ConditionalTrainingConfig,
    predict_conditional_checkpoint,
)


OFFICIAL_TEST_IDENTITY = {
    "split": "test",
    "count": 4751,
    "manifest_sha256": "f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8",
    "ordered_key_sha256": "432db77b96cc1c5ff8eff4de655b64b4a487c4b448913ee7db33cdaf886b303f",
    "files": {
        "features.npz": "b8ded10b2499f1d4d81b9321a914d5d8146f1296528c89d095369843c928abd3",
        "metadata.json": "9a539a21e71d2b3f093a43d43762a78bd5df1c92937d8559c66155a68e173fbe",
        "coverage.json": "26672db86031ad4d6dfedf3be873867ece9537d45909c556737b2dd776c6e280",
        "normalization.json": "65c5b00a0cc5952c8da3ce3be0e8a96b299ad07fb1acebda15c16b1e7d3ea340",
    },
}


def _canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation payload must be canonical JSON compatible") from error


def _atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_bytes(path, _canonical_json_bytes(value))


def _npz_bytes(**values):
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def _load_json(path, name):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(name + " is malformed") from error


def _finite_metric(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(name + " must be finite")
    return float(value)


def test_status(metrics):
    if not isinstance(metrics, Mapping):
        raise ValueError("test metrics must be a mapping")
    values = {
        name: _finite_metric(metrics.get(name), name)
        for name in ("CI", "RMSE", "MAE")
    }
    passed = (
        values["CI"] >= TEST_GATE["CI"]
        and values["RMSE"] < TEST_GATE["RMSE"]
        and values["MAE"] < TEST_GATE["MAE"]
    )
    return "passed" if passed else "failed_gate"


def reserve_evaluation(output, payload):
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation reservation payload must be a mapping")
    source = Path(output).expanduser()
    if source.is_symlink():
        raise ValueError("evaluation output must not be a symlink")
    output = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("evaluation is already reserved or complete")
    start = output / "evaluation.start"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(str(start), flags, 0o400)
    except FileExistsError as error:
        raise ValueError("evaluation is already reserved") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return start


def _finite_prediction(value, count):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("selected model predictions must be finite") from error
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("selected model predictions must be finite and complete")
    result = result.copy()
    result.setflags(write=False)
    return result


def predict_selected_model(
    selected_model,
    *,
    dataset,
    base_predictions,
    device,
):
    if not isinstance(selected_model, Mapping):
        raise ValueError("selected model payload must be a mapping")
    if selected_model.get("status") != "complete":
        raise ValueError("selected model is not complete")
    name = selected_model.get("name")
    candidate_type = selected_model.get("candidate_type")
    model = selected_model.get("model")
    base = np.asarray(base_predictions, dtype=np.float64)
    if base.ndim != 1 or base.size == 0 or not np.isfinite(base).all():
        raise ValueError("base predictions must be finite")
    if candidate_type == "identity":
        if name != "identity":
            raise ValueError("identity selected model name does not match")
        proposed = base
    elif candidate_type == "analytic":
        if not isinstance(model, Mapping):
            raise ValueError("analytic selected model is malformed")
        if model.get("type") == "positive_affine":
            expected_objective = {
                "affine_mse": "mse",
                "affine_huber": "huber",
            }.get(name)
            if model.get("objective") != expected_objective:
                raise ValueError(
                    "analytic selected model name does not match descriptor"
                )
            calibrator = PositiveAffineCalibrator.from_dict(dict(model))
        elif model.get("type") == "monotonic_spline":
            expected_knots = {
                "spline_8_knots": 8,
                "spline_16_knots": 16,
            }.get(name)
            if model.get("knot_count") != expected_knots:
                raise ValueError(
                    "analytic selected model name does not match descriptor"
                )
            calibrator = MonotonicSplineCalibrator.from_dict(dict(model))
        else:
            raise ValueError("analytic selected model type does not match")
        proposed = calibrator.predict(base)
    elif candidate_type == "conditional":
        if name not in {
            "conditional_seed_1",
            "conditional_seed_7",
            "conditional_seed_21",
        }:
            raise ValueError("conditional selected model name does not match")
        if dataset is None:
            raise ValueError("conditional selected model requires a dataset")
        config = ConditionalTrainingConfig(
            maximum_epochs=1,
            patience=1,
            batch_size=256,
            device=device,
        )
        proposed = predict_conditional_checkpoint(model, dataset, config=config)
    elif candidate_type == "ensemble":
        if (
            name != "conditional_ensemble"
            or dataset is None
            or not isinstance(model, Mapping)
            or set(model) != {"type", "rule", "members"}
            or model.get("type") != "conditional_ensemble"
            or model.get("rule")
            != "arithmetic_mean_before_one_rank_projection"
        ):
            raise ValueError("ensemble selected model is malformed")
        members = model.get("members")
        if not isinstance(members, Mapping) or set(members) != {"1", "7", "21"}:
            raise ValueError("ensemble selected model members do not match")
        config = ConditionalTrainingConfig(
            maximum_epochs=1,
            patience=1,
            batch_size=256,
            device=device,
        )
        values = [
            predict_conditional_checkpoint(
                members[str(seed)], dataset, config=config
            )
            for seed in (1, 7, 21)
        ]
        proposed = np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64)
    else:
        raise ValueError("selected model candidate type does not match")
    if name is None:
        raise ValueError("selected model name is missing")
    return _finite_prediction(proposed, base.shape[0])


def _validate_official_cache(directory, cache):
    directory = Path(directory).expanduser().resolve()
    metadata = cache.metadata
    expected = OFFICIAL_TEST_IDENTITY
    if (
        len(cache.keys) != expected["count"]
        or metadata.get("split") != expected["split"]
        or metadata.get("count") != expected["count"]
        or metadata.get("manifest_sha256") != expected["manifest_sha256"]
        or metadata.get("ordered_key_sha256") != expected["ordered_key_sha256"]
    ):
        raise ValueError("official test cache identity does not match")
    for name, expected_hash in expected["files"].items():
        path = directory / name
        mode = os.lstat(path).st_mode if path.exists() or path.is_symlink() else None
        if mode is None or stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError("official test cache file is invalid")
        if sha256_file(path) != expected_hash:
            raise ValueError("official test cache file hash does not match")


def _load_selected_model(package, selection):
    try:
        value = torch.load(
            str(package / "selected_model.pt"),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        value = torch.load(str(package / "selected_model.pt"), map_location="cpu")
    if (
        not isinstance(value, Mapping)
        or value.get("name") != selection["selected_name"]
        or value.get("candidate_type") != selection["selected_type"]
        or value.get("status") != "complete"
    ):
        raise ValueError("sealed selected model does not match selection")
    return value


def _load_v5_base(package, cache):
    rollback = _load_json(package / "rollback.json", "rollback")
    v5_package = Path(rollback["v5_validation_package"]).resolve()
    verify_validation_seal(v5_package, artifact_root=v5_package.parent)
    static_specs = _load_json(v5_package / "static_specs.json", "v5 static specs")
    if not isinstance(static_specs, dict) or "rank" not in static_specs:
        raise ValueError("v5 rank static spec is missing")
    spec = StaticSpec.from_dict(static_specs["rank"])
    if spec.family != "rank":
        raise ValueError("v5 static spec must be rank")
    base = np.asarray(static_predictions(cache, spec), dtype=np.float64)
    fallback_result_path = Path(rollback["v5_official_test_result"]).resolve()
    fallback_result = _load_json(fallback_result_path, "v5 official result")
    fallback_metrics = fallback_result.get("metrics")
    if not isinstance(fallback_metrics, Mapping):
        raise ValueError("v5 fallback metrics are missing")
    actual_metrics = affinity_metrics(cache.labels, base)
    if any(
        actual_metrics[name] != float(fallback_metrics[name])
        for name in ("CI", "MAE", "RMSE", "Pearson", "Spearman")
    ):
        raise ValueError("v5 rank StaticSpec does not reproduce fallback metrics")
    return base, dict(fallback_metrics), rollback


def evaluate_official_test(package, test_cache, output, *, device="cuda:0"):
    package = Path(package).expanduser().resolve()
    selection = verify_validation_package(package)
    seal_hash = sha256_file(package / "selection_seal.json")
    cache_directory = Path(test_cache).expanduser().resolve()
    cache = load_fusion_cache(
        cache_directory,
        expected_split="test",
        expected_manifest_sha256=OFFICIAL_TEST_IDENTITY["manifest_sha256"],
    )
    _validate_official_cache(cache_directory, cache)
    package_normalization = _load_json(
        package / "normalization.json", "sealed normalization"
    )
    cache_normalization = _load_json(
        cache_directory / "normalization.json", "test normalization"
    )
    if package_normalization != cache_normalization:
        raise ValueError("sealed normalization does not match official test cache")
    selected_model = _load_selected_model(package, selection)
    base, fallback_metrics, rollback = _load_v5_base(package, cache)
    reservation = {
        "schema_version": 1,
        "status": "started",
        "selection_seal_sha256": seal_hash,
        "selected_name": selection["selected_name"],
        "test_manifest_sha256": OFFICIAL_TEST_IDENTITY["manifest_sha256"],
        "test_ordered_key_sha256": OFFICIAL_TEST_IDENTITY["ordered_key_sha256"],
    }
    output = Path(output).expanduser().resolve()
    reserve_evaluation(output, reservation)
    dataset = RankLockedCalibrationDataset(cache, base)
    proposed = predict_selected_model(
        selected_model,
        dataset=dataset,
        base_predictions=base,
        device=device,
    )
    final, projection = rank_locked_project(base, proposed)
    metrics = affinity_metrics(cache.labels, final)
    status = test_status(metrics)
    gates = {
        "CI": metrics["CI"] >= TEST_GATE["CI"],
        "RMSE": metrics["RMSE"] < TEST_GATE["RMSE"],
        "MAE": metrics["MAE"] < TEST_GATE["MAE"],
    }
    prediction_bytes = _npz_bytes(
        keys=np.asarray(cache.keys, dtype=np.str_),
        labels=np.asarray(cache.labels, dtype=np.float64),
        base_predictions=np.asarray(base, dtype=np.float64),
        proposed_predictions=np.asarray(proposed, dtype=np.float64),
        predictions=np.asarray(final, dtype=np.float64),
        selected_name=np.asarray(selection["selected_name"], dtype=np.str_),
    )
    prediction_path = output / "predictions.npz"
    _atomic_bytes(prediction_path, prediction_bytes)
    projection_payload = asdict(projection)
    _atomic_json(output / "projection_report.json", projection_payload)
    result = {
        "schema_version": 1,
        "status": status,
        "count": len(cache.keys),
        "coverage": 1.0,
        "selected_name": selection["selected_name"],
        "selected_type": selection["selected_type"],
        "selected_validation_metrics": selection["selected_validation_metrics"],
        "metrics": metrics,
        "gates": gates,
        "test_gate": dict(TEST_GATE),
        "delta_vs_v5": {
            name: metrics[name] - float(fallback_metrics[name])
            for name in ("CI", "MAE", "RMSE", "Pearson", "Spearman")
        },
        "v5_metrics": fallback_metrics,
        "v5_rollback": rollback,
        "selection_seal_sha256": seal_hash,
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "test_manifest_sha256": OFFICIAL_TEST_IDENTITY["manifest_sha256"],
        "test_ordered_key_sha256": OFFICIAL_TEST_IDENTITY["ordered_key_sha256"],
        "test_file_sha256": dict(OFFICIAL_TEST_IDENTITY["files"]),
    }
    _atomic_json(output / "result.json", result)
    complete = {
        "schema_version": 1,
        "status": "complete",
        "result_sha256": sha256_file(output / "result.json"),
        "prediction_sha256": result["prediction_sha256"],
        "projection_report_sha256": sha256_file(
            output / "projection_report.json"
        ),
        "selection_seal_sha256": seal_hash,
    }
    _atomic_json(output / "evaluation.complete", complete)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = evaluate_official_test(
        args.package,
        args.test_cache,
        args.output,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OFFICIAL_TEST_IDENTITY",
    "build_arg_parser",
    "evaluate_official_test",
    "predict_selected_model",
    "reserve_evaluation",
    "test_status",
]
