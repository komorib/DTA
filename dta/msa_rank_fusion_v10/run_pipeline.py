"""Fit validation-selected V8/V9 latent fusion and evaluate the frozen test set."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.rank_locked_calibration_v6.folds import assign_grouped_folds, load_fold_rows
from utils.experiment_integrity import sha256_file

from . import VERSION


BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))
MINIMUM_LEAVES = (10, 20, 40)
MAX_FEATURE_FRACTIONS = (0.3, 0.7, 1.0)
RESIDUAL_BUDGETS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.30)


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_joblib(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".joblib")
    os.close(descriptor)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_npz(path, required):
    path = Path(path).expanduser().resolve()
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            if set(archive.files) != set(required):
                raise ValueError("prediction fields do not match")
            values = {name: np.asarray(archive[name]) for name in required}
    except (OSError, ValueError) as error:
        raise ValueError("prediction archive is malformed: " + str(path)) from error
    return path, values


def _fit_affine(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        raise ValueError("cannot calibrate a constant prediction")
    slope = float(np.dot(centered, labels - labels.mean()) / denominator)
    slope = max(slope, 1e-8)
    intercept = float(labels.mean() - slope * values.mean())
    return intercept, slope


def select_standardized_blend(v8, v9, labels):
    v8 = np.asarray(v8, dtype=np.float64)
    v9 = np.asarray(v9, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if v8.shape != labels.shape or v9.shape != labels.shape:
        raise ValueError("blend predictions do not match labels")
    means = (float(v8.mean()), float(v9.mean()))
    scales = (float(v8.std()), float(v9.std()))
    if min(scales) <= 1e-12:
        raise ValueError("blend prediction variance is zero")
    standardized_v8 = (v8 - means[0]) / scales[0]
    standardized_v9 = (v9 - means[1]) / scales[1]
    candidates = {}
    best = None
    for weight in BLEND_WEIGHTS:
        raw = weight * standardized_v8 + (1.0 - weight) * standardized_v9
        intercept, slope = _fit_affine(raw, labels)
        prediction = intercept + slope * raw
        metrics = affinity_metrics(labels, prediction)
        name = "v8_%0.2f" % weight
        candidates[name] = dict(metrics)
        score = (float(metrics["CI"]), -float(metrics["RMSE"]), weight)
        if best is None or score > best[0]:
            best = (score, weight, intercept, slope, prediction, metrics)
    _, weight, intercept, slope, prediction, metrics = best
    return {
        "weight": float(weight),
        "intercept": float(intercept),
        "slope": float(slope),
        "means": means,
        "scales": scales,
        "predictions": prediction,
        "metrics": dict(metrics),
        "candidates": candidates,
    }


def apply_standardized_blend(v8, v9, selection):
    v8 = np.asarray(v8, dtype=np.float64)
    v9 = np.asarray(v9, dtype=np.float64)
    weight = float(selection["weight"])
    means = tuple(float(value) for value in selection["means"])
    scales = tuple(float(value) for value in selection["scales"])
    raw = weight * (v8 - means[0]) / scales[0]
    raw += (1.0 - weight) * (v9 - means[1]) / scales[1]
    return float(selection["intercept"]) + float(selection["slope"]) * raw


def _new_tree(minimum_leaf, max_features, estimators, n_jobs, seed):
    return ExtraTreesRegressor(
        n_estimators=int(estimators),
        min_samples_leaf=int(minimum_leaf),
        max_features=float(max_features),
        n_jobs=int(n_jobs),
        random_state=int(seed),
    )


def select_residual_model(
    features,
    labels,
    anchor,
    fold_ids,
    *,
    estimators,
    n_jobs,
    seed,
):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if features.shape[0] != labels.size or anchor.shape != labels.shape:
        raise ValueError("residual feature rows do not match")
    unique_folds = tuple(sorted(int(value) for value in np.unique(fold_ids)))
    if unique_folds != tuple(range(len(unique_folds))) or len(unique_folds) < 2:
        raise ValueError("residual folds are invalid")
    candidates = {}
    best = None
    target = labels - anchor
    for minimum_leaf in MINIMUM_LEAVES:
        for max_features in MAX_FEATURE_FRACTIONS:
            residual = np.full(labels.shape, np.nan, dtype=np.float64)
            for fold in unique_folds:
                train = fold_ids != fold
                holdout = fold_ids == fold
                model = _new_tree(
                    minimum_leaf,
                    max_features,
                    estimators,
                    n_jobs,
                    seed + fold,
                )
                model.fit(features[train], target[train])
                residual[holdout] = model.predict(features[holdout])
            if not np.isfinite(residual).all():
                raise RuntimeError("residual OOF predictions are incomplete")
            for budget in RESIDUAL_BUDGETS:
                prediction = anchor + float(budget) * residual
                metrics = affinity_metrics(labels, prediction)
                name = "leaf_%d_features_%0.1f_budget_%0.3f" % (
                    minimum_leaf,
                    max_features,
                    budget,
                )
                candidates[name] = dict(metrics)
                score = (
                    float(metrics["CI"]),
                    -float(metrics["RMSE"]),
                    -float(metrics["MAE"]),
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        name,
                        minimum_leaf,
                        max_features,
                        budget,
                        prediction,
                        metrics,
                    )
    _, name, minimum_leaf, max_features, budget, prediction, metrics = best
    return {
        "name": name,
        "minimum_leaf": int(minimum_leaf),
        "max_features": float(max_features),
        "budget": float(budget),
        "predictions": prediction,
        "metrics": dict(metrics),
        "candidates": candidates,
    }


def _validate_alignment(v8, v9, split):
    keys8 = np.asarray(v8["keys"]).astype(str)
    keys9 = np.asarray(v9["keys"]).astype(str)
    labels8 = np.asarray(v8["labels"], dtype=np.float64)
    labels9 = np.asarray(v9["labels"], dtype=np.float64)
    if not np.array_equal(keys8, keys9) or not np.allclose(labels8, labels9, atol=1e-6):
        raise ValueError("V8/V9 %s keys or labels do not align" % split)
    return keys8, labels8


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V10 output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    v8_root = Path(args.v8_root).expanduser().resolve()
    v9_root = Path(args.v9_root).expanduser().resolve()
    v8_valid_path, v8_valid = _load_npz(
        v8_root / "oof_predictions.npz",
        ("keys", "labels", "features", "v7_selected", "v8_selected"),
    )
    v9_valid_path, v9_valid = _load_npz(
        v9_root / "valid_predictions.npz",
        ("keys", "labels", "predictions", "latents"),
    )
    valid_keys, valid_labels = _validate_alignment(v8_valid, v9_valid, "valid")
    if v9_valid["latents"].ndim != 2 or v9_valid["latents"].shape[0] != valid_labels.size:
        raise ValueError("V9 validation latents have the wrong shape")
    blend = select_standardized_blend(
        v8_valid["v8_selected"], v9_valid["predictions"], valid_labels
    )
    rows = load_fold_rows(
        args.metadata_csv,
        tuple(valid_keys.tolist()),
        expected_split="val",
        allow_invalid_scaffold_fallback=True,
    )
    assignment = assign_grouped_folds(rows, fold_count=5, seed=args.fold_seed)
    fold_ids = np.asarray([assignment.by_key[key] for key in valid_keys], dtype=np.int64)
    components = min(int(args.pca_components), valid_labels.size - 1, v9_valid["latents"].shape[1])
    if components <= 0:
        raise ValueError("PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=args.seed,
    )
    latent_valid = pca.fit_transform(np.asarray(v9_valid["latents"], dtype=np.float64))
    scalar_scaler = StandardScaler()
    scalar_valid = scalar_scaler.fit_transform(
        np.column_stack((v8_valid["features"], v9_valid["predictions"]))
    )
    features_valid = np.column_stack((latent_valid, scalar_valid))
    residual = select_residual_model(
        features_valid,
        valid_labels,
        blend["predictions"],
        fold_ids,
        estimators=args.oof_estimators,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    validation_selected = float(residual["metrics"]["CI"]) > float(
        blend["metrics"]["CI"]
    )
    if not validation_selected:
        residual["minimum_leaf"] = 10
        residual["max_features"] = 1.0
        residual["budget"] = 0.0
        residual["name"] = "blend_fallback"
        residual["predictions"] = blend["predictions"]
        residual["metrics"] = blend["metrics"]
    final_tree = _new_tree(
        residual["minimum_leaf"],
        residual["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(features_valid, valid_labels - blend["predictions"])

    v8_test_path, v8_test = _load_npz(
        v8_root / "test_predictions.npz",
        ("keys", "labels", "features", "predictions", "v7_predictions"),
    )
    v9_test_path, v9_test = _load_npz(
        v9_root / "test_predictions.npz",
        ("keys", "labels", "predictions", "latents"),
    )
    test_keys, test_labels = _validate_alignment(v8_test, v9_test, "test")
    latent_test = pca.transform(np.asarray(v9_test["latents"], dtype=np.float64))
    scalar_test = scalar_scaler.transform(
        np.column_stack((v8_test["features"], v9_test["predictions"]))
    )
    features_test = np.column_stack((latent_test, scalar_test))
    blend_test = apply_standardized_blend(
        v8_test["predictions"], v9_test["predictions"], blend
    )
    test_predictions = blend_test + float(residual["budget"]) * final_tree.predict(
        features_test
    )
    test_metrics = affinity_metrics(test_labels, test_predictions)
    v8_valid_metrics = affinity_metrics(valid_labels, v8_valid["v8_selected"])
    v8_test_metrics = affinity_metrics(test_labels, v8_test["predictions"])
    passed = validation_selected and float(test_metrics["CI"]) > float(v8_test_metrics["CI"])
    model_package = {
        "version": VERSION,
        "pca": pca,
        "scalar_scaler": scalar_scaler,
        "tree": final_tree,
        "blend": {key: value for key, value in blend.items() if key not in ("predictions", "candidates")},
        "residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
        },
    }
    _atomic_joblib(output / "model.joblib", model_package)
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=valid_keys,
        labels=valid_labels,
        v8_predictions=np.asarray(v8_valid["v8_selected"], dtype=np.float64),
        v9_predictions=np.asarray(v9_valid["predictions"], dtype=np.float64),
        blend_predictions=np.asarray(blend["predictions"], dtype=np.float64),
        predictions=np.asarray(residual["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v8_predictions=np.asarray(v8_test["predictions"], dtype=np.float64),
        v9_predictions=np.asarray(v9_test["predictions"], dtype=np.float64),
        blend_predictions=np.asarray(blend_test, dtype=np.float64),
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": "passed" if passed else "not_improved",
        "version": VERSION,
        "count": int(test_labels.size),
        "validation_selected_residual": bool(validation_selected),
        "selected_blend": {
            "v8_weight": blend["weight"],
            "v9_weight": 1.0 - blend["weight"],
            "metrics": blend["metrics"],
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "v8_validation_metrics": dict(v8_valid_metrics),
        "v8_test_metrics": dict(v8_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v8": float(test_metrics["CI"] - v8_test_metrics["CI"]),
        "inputs": {
            "v8_valid": str(v8_valid_path),
            "v8_valid_sha256": sha256_file(v8_valid_path),
            "v8_test": str(v8_test_path),
            "v8_test_sha256": sha256_file(v8_test_path),
            "v9_valid": str(v9_valid_path),
            "v9_valid_sha256": sha256_file(v9_valid_path),
            "v9_test": str(v9_test_path),
            "v9_test_sha256": sha256_file(v9_test_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-root", required=True)
    parser.add_argument("--v9-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--oof-estimators", type=int, default=200)
    parser.add_argument("--final-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv=None):
    result = run(build_arg_parser().parse_args(argv))
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "apply_standardized_blend",
    "select_residual_model",
    "select_standardized_blend",
]
