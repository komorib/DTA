"""Validation-gated stability fusion over frozen V15 spatial interface features."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dta.msa_rank_fusion_v10.run_pipeline import (
    _atomic_joblib,
    _atomic_json,
    _atomic_npz,
    _load_npz,
    _new_tree,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import VERSION


BASE_VALID_FIELDS = (
    "keys",
    "labels",
    "fold_ids",
    "v11_fusion_predictions",
    "v12_predictions",
    "predictions",
)
BASE_TEST_FIELDS = tuple(name for name in BASE_VALID_FIELDS if name != "fold_ids")
BENCHMARK_VALID_FIELDS = (
    "keys",
    "labels",
    "v12_base_predictions",
    "v14_anchor_predictions",
    "predictions",
    "fold_ids",
)
BENCHMARK_TEST_FIELDS = tuple(
    name for name in BENCHMARK_VALID_FIELDS if name != "fold_ids"
)
FEATURE_FIELDS = (
    "keys",
    "labels",
    "pretrain_features",
    "affinity_features",
)
FAMILIES = ("affinity", "pretrain", "delta", "joint")
MINIMUM_LEAVES = (2, 3, 4, 5, 7)
MAX_FEATURE_FRACTIONS = (0.7, 0.85, 1.0)
RESIDUAL_BUDGETS = tuple(round(index * 0.025, 3) for index in range(41))


def _limit_archive(values, limit):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 1:
        raise ValueError("V16 fusion limit must exceed one")
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _normalize_fold_ids(values, shape):
    fold_ids = np.asarray(values, dtype=np.int64)
    if fold_ids.shape != shape:
        raise ValueError("V16 grouped fold IDs have the wrong shape")
    unique = np.unique(fold_ids)
    if unique.size < 2:
        raise ValueError("V16 grouped fold IDs contain fewer than two folds")
    return np.searchsorted(unique, fold_ids).astype(np.int64, copy=False)


def _alignment(reference, other, label):
    reference_keys = np.asarray(reference["keys"]).astype(str)
    other_keys = np.asarray(other["keys"]).astype(str)
    reference_labels = np.asarray(reference["labels"], dtype=np.float64)
    other_labels = np.asarray(other["labels"], dtype=np.float64)
    if not np.array_equal(reference_keys, other_keys):
        raise ValueError(f"V16 {label} keys do not align")
    if not np.allclose(reference_labels, other_labels, atol=1e-6):
        raise ValueError(f"V16 {label} labels do not align")
    return reference_keys, reference_labels


def _validate_inputs(base, benchmark, features, split):
    keys, labels = _alignment(base, benchmark, f"V12/V15 {split}")
    _alignment(base, features, f"V12/features {split}")
    return keys, labels


def _family_raw(features, family):
    pretrain = np.asarray(features["pretrain_features"], dtype=np.float64)
    affinity = np.asarray(features["affinity_features"], dtype=np.float64)
    if pretrain.ndim != 2 or affinity.shape != pretrain.shape:
        raise ValueError("V16 spatial feature matrices have the wrong shape")
    if family == "affinity":
        result = affinity
    elif family == "pretrain":
        result = pretrain
    elif family == "delta":
        result = affinity - pretrain
    elif family == "joint":
        result = np.column_stack((affinity, affinity - pretrain))
    else:
        raise ValueError(f"unknown V16 spatial feature family: {family}")
    if not np.isfinite(result).all():
        raise ValueError("V16 spatial feature family contains non-finite values")
    return result


def _fit_transform(raw, requested_components, seed):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    components = min(int(requested_components), raw.shape[0] - 1, raw.shape[1])
    if components <= 0:
        raise ValueError("V16 PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    )
    transformed = pca.fit_transform(scaled)
    if not np.isfinite(transformed).all():
        raise RuntimeError("V16 transformed validation features are non-finite")
    return {"scaler": scaler, "pca": pca}, transformed


def _apply_transform(raw, transform):
    transformed = transform["pca"].transform(transform["scaler"].transform(raw))
    if not np.isfinite(transformed).all():
        raise RuntimeError("V16 transformed test features are non-finite")
    return transformed


def _fold_diagnostics(labels, benchmark, candidate, fold_ids):
    diagnostics = []
    positive = 0
    for fold in np.unique(fold_ids):
        selected = fold_ids == fold
        benchmark_metrics = affinity_metrics(labels[selected], benchmark[selected])
        candidate_metrics = affinity_metrics(labels[selected], candidate[selected])
        gain = float(candidate_metrics["CI"] - benchmark_metrics["CI"])
        positive += int(gain > 0.0)
        diagnostics.append(
            {
                "fold": int(fold),
                "count": int(selected.sum()),
                "v15_ci": float(benchmark_metrics["CI"]),
                "candidate_ci": float(candidate_metrics["CI"]),
                "ci_gain": gain,
            }
        )
    return diagnostics, positive


def _select_tree_configuration(
    features,
    labels,
    base,
    fold_ids,
    benchmark,
    args,
    family_index,
):
    unique_folds = tuple(sorted(int(value) for value in np.unique(fold_ids)))
    target = labels - base
    configurations = {}
    best = None
    configuration_index = 0
    for minimum_leaf in MINIMUM_LEAVES:
        for max_features in MAX_FEATURE_FRACTIONS:
            raw_residual = np.full(labels.shape, np.nan, dtype=np.float64)
            for fold in unique_folds:
                train = fold_ids != fold
                holdout = fold_ids == fold
                tree = _new_tree(
                    minimum_leaf,
                    max_features,
                    args.oof_estimators,
                    args.n_jobs,
                    args.seed + 10000 * family_index + 100 * configuration_index + fold,
                )
                tree.fit(features[train], target[train])
                raw_residual[holdout] = tree.predict(features[holdout])
            if not np.isfinite(raw_residual).all():
                raise RuntimeError("V16 residual OOF predictions are incomplete")
            configuration_best = None
            for budget in RESIDUAL_BUDGETS:
                predictions = base + float(budget) * raw_residual
                metrics = affinity_metrics(labels, predictions)
                diagnostics, positive_folds = _fold_diagnostics(
                    labels, benchmark, predictions, fold_ids
                )
                score = (
                    float(metrics["CI"]),
                    -float(metrics["RMSE"]),
                    -float(metrics["MAE"]),
                    -abs(float(budget) - 0.30),
                )
                value = {
                    "minimum_leaf": int(minimum_leaf),
                    "max_features": float(max_features),
                    "budget": float(budget),
                    "metrics": dict(metrics),
                    "positive_folds": int(positive_folds),
                    "fold_diagnostics": diagnostics,
                }
                if configuration_best is None or score > configuration_best[0]:
                    configuration_best = (score, value)
                if best is None or score > best[0]:
                    best = (
                        score,
                        value,
                        np.asarray(predictions, dtype=np.float64),
                        np.asarray(raw_residual, dtype=np.float64),
                    )
            name = "leaf_%d_features_%0.2f" % (minimum_leaf, max_features)
            configurations[name] = configuration_best[1]
            configuration_index += 1
    _, selected, predictions, raw_residual = best
    selected = dict(selected)
    selected["name"] = "leaf_%d_features_%0.2f_budget_%0.3f" % (
        selected["minimum_leaf"],
        selected["max_features"],
        selected["budget"],
    )
    return {
        "selected": selected,
        "predictions": predictions,
        "raw_residual": raw_residual,
        "configuration_candidates": configurations,
    }


def _select_validation(features, labels, base, benchmark, fold_ids, args):
    family_candidates = {}
    best = None
    for family_index, family in enumerate(FAMILIES):
        transform, transformed = _fit_transform(
            _family_raw(features, family),
            args.pca_components,
            args.seed + family_index,
        )
        residual = _select_tree_configuration(
            transformed,
            labels,
            base,
            fold_ids,
            benchmark,
            args,
            family_index,
        )
        summary = {
            "family": family,
            "pca_components": int(transformed.shape[1]),
            "selected_residual": residual["selected"],
            "configuration_candidates": residual["configuration_candidates"],
        }
        family_candidates[family] = summary
        metrics = residual["selected"]["metrics"]
        score = (
            float(metrics["CI"]),
            -float(metrics["RMSE"]),
            -float(metrics["MAE"]),
            -family_index,
        )
        if best is None or score > best[0]:
            best = (
                score,
                family,
                transform,
                transformed,
                residual,
                summary,
            )
    _, family, transform, transformed, residual, summary = best
    return {
        "family": family,
        "transform": transform,
        "features": transformed,
        "residual": residual,
        "summary": summary,
        "family_candidates": family_candidates,
    }


def _gate(selection, benchmark_metrics, args):
    metrics = selection["residual"]["selected"]["metrics"]
    ci_gain = float(metrics["CI"] - benchmark_metrics["CI"])
    spearman_gain = float(metrics["Spearman"] - benchmark_metrics["Spearman"])
    rmse_increase = float(metrics["RMSE"] - benchmark_metrics["RMSE"])
    positive_folds = int(selection["residual"]["selected"]["positive_folds"])
    passed = (
        (bool(args.allow_zero_budget) or float(selection["residual"]["selected"]["budget"]) > 0.0)
        and ci_gain >= float(args.minimum_validation_ci_gain)
        and spearman_gain >= float(args.minimum_spearman_gain)
        and rmse_increase <= float(args.maximum_rmse_increase)
        and positive_folds >= int(args.minimum_positive_folds)
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "spearman_gain": spearman_gain,
        "rmse_increase": rmse_increase,
        "positive_folds": positive_folds,
        "criteria": {
            "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
            "minimum_spearman_gain": float(args.minimum_spearman_gain),
            "maximum_rmse_increase": float(args.maximum_rmse_increase),
            "minimum_positive_folds": int(args.minimum_positive_folds),
            "allow_zero_budget": bool(args.allow_zero_budget),
        },
    }


def _validation_result(
    output,
    status,
    args,
    keys,
    labels,
    fold_ids,
    base_predictions,
    benchmark_predictions,
    base_metrics,
    benchmark_metrics,
    selection,
    gate,
    base_valid_path,
    benchmark_valid_path,
    feature_valid_path,
):
    selected = selection["residual"]["selected"]
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=keys,
        labels=labels,
        v12_base_predictions=np.asarray(base_predictions, dtype=np.float64),
        v15_anchor_predictions=np.asarray(benchmark_predictions, dtype=np.float64),
        predictions=np.asarray(selection["residual"]["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    result = {
        "status": status,
        "version": VERSION,
        "validation_count": int(labels.size),
        "limit": args.limit,
        "validation_selected": status == "validation_passed",
        "selected_family": selection["family"],
        "selected_pca_components": int(selection["features"].shape[1]),
        "selected_residual": selected,
        "selected_validation_metrics": dict(selected["metrics"]),
        "v12_base_validation_metrics": dict(base_metrics),
        "v15_anchor_validation_metrics": dict(benchmark_metrics),
        "validation_gate": gate,
        "family_candidates": selection["family_candidates"],
        "search_space": {
            "families": list(FAMILIES),
            "pca_components": int(args.pca_components),
            "minimum_leaves": list(MINIMUM_LEAVES),
            "max_feature_fractions": list(MAX_FEATURE_FRACTIONS),
            "residual_budgets": list(RESIDUAL_BUDGETS),
            "oof_estimators": int(args.oof_estimators),
            "final_estimators": int(args.final_estimators),
            "seed": int(args.seed),
        },
        "inputs": {
            "v12_valid": str(base_valid_path),
            "v12_valid_sha256": sha256_file(base_valid_path),
            "v15_anchor_valid": str(benchmark_valid_path),
            "v15_anchor_valid_sha256": sha256_file(benchmark_valid_path),
            "v15_features_valid": str(feature_valid_path),
            "v15_features_valid_sha256": sha256_file(feature_valid_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _verify_selection_result(
    args,
    base_valid_path,
    benchmark_valid_path,
    feature_valid_path,
    selection,
    gate,
):
    selection_path = Path(args.selection_result).expanduser().resolve()
    try:
        with selection_path.open("r", encoding="utf-8") as handle:
            sealed = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("V16 selection result is malformed") from error
    if sealed.get("version") != VERSION or sealed.get("status") != "validation_passed":
        raise ValueError("V16 selection result is not a passed V16 selection")
    expected_hashes = {
        "v12_valid_sha256": sha256_file(base_valid_path),
        "v15_anchor_valid_sha256": sha256_file(benchmark_valid_path),
        "v15_features_valid_sha256": sha256_file(feature_valid_path),
    }
    sealed_inputs = sealed.get("inputs", {})
    for name, expected in expected_hashes.items():
        if sealed_inputs.get(name) != expected:
            raise ValueError("V16 selection input SHA256 mismatch: " + name)
    current_space = {
        "families": list(FAMILIES),
        "pca_components": int(args.pca_components),
        "minimum_leaves": list(MINIMUM_LEAVES),
        "max_feature_fractions": list(MAX_FEATURE_FRACTIONS),
        "residual_budgets": list(RESIDUAL_BUDGETS),
        "oof_estimators": int(args.oof_estimators),
        "final_estimators": int(args.final_estimators),
        "seed": int(args.seed),
    }
    if sealed.get("search_space") != current_space:
        raise ValueError("V16 selection search space does not match current run")
    selected = selection["residual"]["selected"]
    sealed_selected = sealed.get("selected_residual", {})
    for name in ("name", "minimum_leaf", "max_features", "budget"):
        if sealed_selected.get(name) != selected.get(name):
            raise ValueError("V16 recomputed selection does not match seal")
    if sealed.get("selected_family") != selection["family"]:
        raise ValueError("V16 recomputed family does not match seal")
    if int(sealed.get("selected_pca_components", -1)) != int(selection["features"].shape[1]):
        raise ValueError("V16 recomputed PCA width does not match seal")
    sealed_metrics = sealed.get("selected_validation_metrics", {})
    for name, value in selected["metrics"].items():
        if name not in sealed_metrics or not np.isclose(
            float(sealed_metrics[name]), float(value), rtol=0.0, atol=1e-12
        ):
            raise ValueError("V16 recomputed validation metrics do not match seal")
    if not gate.get("passed") or sealed.get("validation_gate", {}).get("criteria") != gate.get("criteria"):
        raise ValueError("V16 recomputed validation gate does not match seal")
    return selection_path, sealed


def run(args):
    if bool(args.validation_only) == bool(args.selection_result):
        raise ValueError(
            "V16 requires exactly one of --validation-only or --selection-result"
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V16 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.base_root).expanduser().resolve()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()

    base_valid_path, base_valid = _load_npz(
        base_root / "valid_predictions.npz", BASE_VALID_FIELDS
    )
    benchmark_valid_path, benchmark_valid = _load_npz(
        benchmark_root / "valid_predictions.npz", BENCHMARK_VALID_FIELDS
    )
    feature_valid_path, feature_valid = _load_npz(
        feature_root / "valid_features.npz", FEATURE_FIELDS
    )
    base_valid = _limit_archive(base_valid, args.limit)
    benchmark_valid = _limit_archive(benchmark_valid, args.limit)
    feature_valid = _limit_archive(feature_valid, args.limit)
    keys, labels = _validate_inputs(
        base_valid, benchmark_valid, feature_valid, "validation"
    )
    fold_ids = _normalize_fold_ids(base_valid["fold_ids"], labels.shape)
    benchmark_fold_ids = _normalize_fold_ids(
        benchmark_valid["fold_ids"], labels.shape
    )
    if not np.array_equal(fold_ids, benchmark_fold_ids):
        raise ValueError("V16 V12/V15 grouped folds do not align")
    base_predictions = np.asarray(base_valid["predictions"], dtype=np.float64)
    benchmark_predictions = np.asarray(
        benchmark_valid["predictions"], dtype=np.float64
    )
    base_metrics = affinity_metrics(labels, base_predictions)
    benchmark_metrics = affinity_metrics(labels, benchmark_predictions)
    selection = _select_validation(
        feature_valid,
        labels,
        base_predictions,
        benchmark_predictions,
        fold_ids,
        args,
    )
    gate = _gate(selection, benchmark_metrics, args)
    validation_arguments = (
        output,
        "validation_passed" if gate["passed"] else "validation_failed",
        args,
        keys,
        labels,
        fold_ids,
        base_predictions,
        benchmark_predictions,
        base_metrics,
        benchmark_metrics,
        selection,
        gate,
        base_valid_path,
        benchmark_valid_path,
        feature_valid_path,
    )
    if args.validation_only:
        return _validation_result(*validation_arguments)
    if not gate["passed"]:
        raise ValueError("V16 recomputed validation selection does not pass")
    selection_path, sealed_selection = _verify_selection_result(
        args,
        base_valid_path,
        benchmark_valid_path,
        feature_valid_path,
        selection,
        gate,
    )

    feature_test_file = feature_root / "test_features.npz"
    if not feature_test_file.is_file():
        raise ValueError("V16 frozen test feature archive is missing")

    selected = selection["residual"]["selected"]
    final_tree = _new_tree(
        selected["minimum_leaf"],
        selected["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(selection["features"], labels - base_predictions)

    base_test_path, base_test = _load_npz(
        base_root / "test_predictions.npz", BASE_TEST_FIELDS
    )
    benchmark_test_path, benchmark_test = _load_npz(
        benchmark_root / "test_predictions.npz", BENCHMARK_TEST_FIELDS
    )
    feature_test_path, feature_test = _load_npz(feature_test_file, FEATURE_FIELDS)
    base_test = _limit_archive(base_test, args.limit)
    benchmark_test = _limit_archive(benchmark_test, args.limit)
    feature_test = _limit_archive(feature_test, args.limit)
    test_keys, test_labels = _validate_inputs(
        base_test, benchmark_test, feature_test, "test"
    )
    base_test_predictions = np.asarray(base_test["predictions"], dtype=np.float64)
    benchmark_test_predictions = np.asarray(
        benchmark_test["predictions"], dtype=np.float64
    )
    test_features = _apply_transform(
        _family_raw(feature_test, selection["family"]),
        selection["transform"],
    )
    test_predictions = base_test_predictions + float(selected["budget"]) * final_tree.predict(
        test_features
    )
    test_metrics = affinity_metrics(test_labels, test_predictions)
    base_test_metrics = affinity_metrics(test_labels, base_test_predictions)
    benchmark_test_metrics = affinity_metrics(
        test_labels, benchmark_test_predictions
    )
    passed = float(test_metrics["CI"]) > float(benchmark_test_metrics["CI"])

    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "family": selection["family"],
            "transform": selection["transform"],
            "tree": final_tree,
            "residual": {
                "name": selected["name"],
                "minimum_leaf": selected["minimum_leaf"],
                "max_features": selected["max_features"],
                "budget": selected["budget"],
            },
            "validation_gate": gate,
        },
    )
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=keys,
        labels=labels,
        v12_base_predictions=base_predictions,
        v15_anchor_predictions=benchmark_predictions,
        predictions=np.asarray(selection["residual"]["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v12_base_predictions=base_test_predictions,
        v15_anchor_predictions=benchmark_test_predictions,
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": "passed" if passed else "not_improved",
        "version": VERSION,
        "count": int(test_labels.size),
        "limit": args.limit,
        "validation_selected": True,
        "selected_family": selection["family"],
        "selected_pca_components": int(selection["features"].shape[1]),
        "selected_residual": selected,
        "selected_validation_metrics": dict(selected["metrics"]),
        "v12_base_validation_metrics": dict(base_metrics),
        "v15_anchor_validation_metrics": dict(benchmark_metrics),
        "validation_gate": gate,
        "v12_base_test_metrics": dict(base_test_metrics),
        "v15_anchor_test_metrics": dict(benchmark_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v15_anchor": float(
            test_metrics["CI"] - benchmark_test_metrics["CI"]
        ),
        "family_candidates": selection["family_candidates"],
        "search_space": sealed_selection.get("search_space"),
        "selection": {
            "result": str(selection_path),
            "result_sha256": sha256_file(selection_path),
        },
        "inputs": {
            "v12_valid": str(base_valid_path),
            "v12_valid_sha256": sha256_file(base_valid_path),
            "v12_test": str(base_test_path),
            "v12_test_sha256": sha256_file(base_test_path),
            "v15_anchor_valid": str(benchmark_valid_path),
            "v15_anchor_valid_sha256": sha256_file(benchmark_valid_path),
            "v15_anchor_test": str(benchmark_test_path),
            "v15_anchor_test_sha256": sha256_file(benchmark_test_path),
            "v15_features_valid": str(feature_valid_path),
            "v15_features_valid_sha256": sha256_file(feature_valid_path),
            "v15_features_test": str(feature_test_path),
            "v15_features_test_sha256": sha256_file(feature_test_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--selection-result")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.0005)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--allow-zero-budget", action="store_true")
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
