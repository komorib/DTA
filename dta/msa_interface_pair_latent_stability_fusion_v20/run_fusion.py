"""Validation-gated raw-PCA stability search over frozen V19 pair latents.

V16 remains the residual-training base.  The V19 out-of-fold prediction is
used only as the promotion benchmark, never as a second-level training anchor.
The validation-only invocation cannot read any test archive.
"""

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
    _new_tree,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import VERSION


BASE_SCALAR_NAMES = (
    "v12_base_predictions",
    "v15_anchor_predictions",
    "predictions",
)
BASE_VALID_FIELDS = ("keys", "labels", "fold_ids") + BASE_SCALAR_NAMES
BASE_TEST_FIELDS = tuple(name for name in BASE_VALID_FIELDS if name != "fold_ids")
BENCHMARK_VALID_FIELDS = (
    "keys",
    "labels",
    "v16_anchor_predictions",
    "predictions",
    "fold_ids",
)
BENCHMARK_TEST_FIELDS = tuple(
    name for name in BENCHMARK_VALID_FIELDS if name != "fold_ids"
)
FEATURE_FIELDS = ("keys", "labels", "v18_predictions", "pair_latents")

# This is the validation-audited V20 search space.  The order is part of the
# deterministic seed schedule and is therefore sealed in result.json.
MINIMUM_LEAVES = (2, 3, 5, 7, 10, 15, 20, 30, 40)
MAX_FEATURE_FRACTIONS = (0.5, 0.7, 0.85, 1.0)
RESIDUAL_BUDGETS = tuple(round(index * 0.025, 3) for index in range(41))
TREE_SEED_STRIDE = 100


def _load_npz(path, required):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing V20 input archive: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("V20 input archive is missing: %s" % ", ".join(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    return path, values


def _limit(values, limit):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 1:
        raise ValueError("V20 fusion limit must exceed one")
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _align(reference, other, label):
    reference_keys = np.asarray(reference["keys"]).astype(str)
    other_keys = np.asarray(other["keys"]).astype(str)
    reference_labels = np.asarray(reference["labels"], dtype=np.float64)
    other_labels = np.asarray(other["labels"], dtype=np.float64)
    if not np.array_equal(reference_keys, other_keys):
        raise ValueError("V20 %s keys do not align" % label)
    if not np.allclose(reference_labels, other_labels, atol=1e-6):
        raise ValueError("V20 %s labels do not align" % label)
    return reference_keys, reference_labels


def _fold_ids(values, shape):
    raw = np.asarray(values, dtype=np.int64)
    if raw.shape != shape:
        raise ValueError("V20 grouped fold IDs have the wrong shape")
    unique = np.unique(raw)
    if unique.size < 2:
        raise ValueError("V20 grouped fold IDs contain fewer than two folds")
    return np.searchsorted(unique, raw).astype(np.int64, copy=False)


def _scalar_matrix(base, v18_predictions):
    matrix = np.column_stack(
        [np.asarray(base[name], dtype=np.float64) for name in BASE_SCALAR_NAMES]
        + [np.asarray(v18_predictions, dtype=np.float64)]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("V20 scalar features contain non-finite values")
    return matrix


def _fit_transform(base, features, requested_components, seed):
    pair_latents = np.asarray(features["pair_latents"], dtype=np.float64)
    if pair_latents.ndim != 2 or pair_latents.shape[0] != len(features["labels"]):
        raise ValueError("V20 pair latents have the wrong shape")
    if not np.isfinite(pair_latents).all():
        raise ValueError("V20 pair latents contain non-finite values")
    components = min(
        int(requested_components), pair_latents.shape[0] - 1, pair_latents.shape[1]
    )
    if components <= 0:
        raise ValueError("V20 PCA component count is invalid")

    # Deliberately fit PCA on the raw pair latent.  A validation-only audit
    # found that pre-scaling the 1,024 latent axes weakened the signal.
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    )
    compressed = pca.fit_transform(pair_latents)
    scalar_scaler = StandardScaler()
    scalars = scalar_scaler.fit_transform(
        _scalar_matrix(base, features["v18_predictions"])
    )
    matrix = np.column_stack((compressed, scalars))
    if not np.isfinite(matrix).all():
        raise RuntimeError("V20 transformed validation features are non-finite")
    return {
        "pca": pca,
        "scalar_scaler": scalar_scaler,
        "components": int(components),
    }, matrix


def _apply_transform(base, features, transform):
    pair_latents = np.asarray(features["pair_latents"], dtype=np.float64)
    compressed = transform["pca"].transform(pair_latents)
    scalars = transform["scalar_scaler"].transform(
        _scalar_matrix(base, features["v18_predictions"])
    )
    matrix = np.column_stack((compressed, scalars))
    if not np.isfinite(matrix).all():
        raise RuntimeError("V20 transformed test features are non-finite")
    return matrix


def _fold_diagnostics(labels, benchmark, candidate, folds):
    diagnostics = []
    positive = 0
    for fold in np.unique(folds):
        selected = folds == fold
        benchmark_metrics = affinity_metrics(labels[selected], benchmark[selected])
        candidate_metrics = affinity_metrics(labels[selected], candidate[selected])
        gain = float(candidate_metrics["CI"] - benchmark_metrics["CI"])
        positive += int(gain > 0.0)
        diagnostics.append(
            {
                "fold": int(fold),
                "count": int(selected.sum()),
                "v19_ci": float(benchmark_metrics["CI"]),
                "candidate_ci": float(candidate_metrics["CI"]),
                "ci_gain": gain,
            }
        )
    return diagnostics, int(positive)


def _select_residual(matrix, labels, base, benchmark, folds, args):
    unique_folds = tuple(sorted(int(value) for value in np.unique(folds)))
    target = labels - base
    configuration_candidates = {}
    best = None
    configuration_index = 0
    for minimum_leaf in MINIMUM_LEAVES:
        for max_features in MAX_FEATURE_FRACTIONS:
            raw_residual = np.full(labels.shape, np.nan, dtype=np.float64)
            configuration_seed = (
                int(args.seed) + TREE_SEED_STRIDE * configuration_index
            )
            for fold in unique_folds:
                train = folds != fold
                holdout = folds == fold
                tree = _new_tree(
                    minimum_leaf,
                    max_features,
                    args.oof_estimators,
                    args.n_jobs,
                    configuration_seed + fold,
                )
                # V16 is intentionally the base used to form this target.
                tree.fit(matrix[train], target[train])
                raw_residual[holdout] = tree.predict(matrix[holdout])
            if not np.isfinite(raw_residual).all():
                raise RuntimeError("V20 residual OOF predictions are incomplete")

            configuration_best = None
            for budget in RESIDUAL_BUDGETS:
                predictions = base + float(budget) * raw_residual
                metrics = affinity_metrics(labels, predictions)
                diagnostics, positive_folds = _fold_diagnostics(
                    labels, benchmark, predictions, folds
                )
                score = (
                    float(metrics["CI"]),
                    -float(metrics["RMSE"]),
                    -float(metrics["MAE"]),
                )
                value = {
                    "configuration_index": int(configuration_index),
                    "configuration_seed": int(configuration_seed),
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
            configuration_candidates[name] = configuration_best[1]
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
        "configuration_candidates": configuration_candidates,
    }


def _gate(labels, benchmark, predictions, folds, budget, args):
    benchmark_metrics = affinity_metrics(labels, benchmark)
    candidate_metrics = affinity_metrics(labels, predictions)
    diagnostics, positive_folds = _fold_diagnostics(
        labels, benchmark, predictions, folds
    )
    ci_gain = float(candidate_metrics["CI"] - benchmark_metrics["CI"])
    spearman_gain = float(
        candidate_metrics["Spearman"] - benchmark_metrics["Spearman"]
    )
    rmse_increase = float(
        candidate_metrics["RMSE"] - benchmark_metrics["RMSE"]
    )
    passed = (
        (bool(args.allow_zero_budget) or float(budget) > 0.0)
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
        "positive_folds": int(positive_folds),
        "fold_diagnostics": diagnostics,
        "criteria": {
            "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
            "minimum_spearman_gain": float(args.minimum_spearman_gain),
            "maximum_rmse_increase": float(args.maximum_rmse_increase),
            "minimum_positive_folds": int(args.minimum_positive_folds),
            "allow_zero_budget": bool(args.allow_zero_budget),
        },
    }


def _search_space(args):
    return {
        "latent_preprocessing": "raw_pca_whiten",
        "pca_components": int(args.pca_components),
        "scalar_names": list(BASE_SCALAR_NAMES) + ["v18_predictions"],
        "minimum_leaves": list(MINIMUM_LEAVES),
        "max_feature_fractions": list(MAX_FEATURE_FRACTIONS),
        "residual_budgets": list(RESIDUAL_BUDGETS),
        "tree_seed_stride": int(TREE_SEED_STRIDE),
        "oof_estimators": int(args.oof_estimators),
        "final_estimators": int(args.final_estimators),
        "seed": int(args.seed),
        "residual_training_base": "v16",
        "promotion_benchmark": "v19",
    }


def _validation_payload(args, fitted, output, status):
    selected = fitted["residual"]["selected"]
    return {
        "status": status,
        "version": VERSION,
        "validation_selected": status == "validation_passed",
        "validation_count": int(fitted["labels"].size),
        "limit": args.limit,
        "selected_pca_components": int(fitted["transform"]["components"]),
        "selected_residual": selected,
        "selected_validation_metrics": dict(selected["metrics"]),
        "v16_base_validation_metrics": dict(fitted["base_metrics"]),
        "v19_benchmark_validation_metrics": dict(fitted["benchmark_metrics"]),
        "validation_gate": fitted["gate"],
        "configuration_candidates": fitted["residual"][
            "configuration_candidates"
        ],
        "search_space": _search_space(args),
        "inputs": {
            "v16_valid": str(fitted["base_path"]),
            "v16_valid_sha256": sha256_file(fitted["base_path"]),
            "v19_benchmark_valid": str(fitted["benchmark_path"]),
            "v19_benchmark_valid_sha256": sha256_file(
                fitted["benchmark_path"]
            ),
            "v19_features_valid": str(fitted["feature_path"]),
            "v19_features_valid_sha256": sha256_file(fitted["feature_path"]),
        },
        "output": str(output),
    }


def _fit_validation(args):
    base_root = Path(args.base_root).expanduser().resolve()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    base_path, base = _load_npz(
        base_root / "valid_predictions.npz", BASE_VALID_FIELDS
    )
    benchmark_path, benchmark = _load_npz(
        benchmark_root / "valid_predictions.npz", BENCHMARK_VALID_FIELDS
    )
    feature_path, features = _load_npz(
        feature_root / "valid_features.npz", FEATURE_FIELDS
    )
    base = _limit(base, args.limit)
    benchmark = _limit(benchmark, args.limit)
    features = _limit(features, args.limit)
    keys, labels = _align(base, benchmark, "V16/V19 validation")
    _align(base, features, "V16/features validation")
    folds = _fold_ids(base["fold_ids"], labels.shape)
    benchmark_folds = _fold_ids(benchmark["fold_ids"], labels.shape)
    if not np.array_equal(folds, benchmark_folds):
        raise ValueError("V20 V16/V19 validation folds do not align")
    base_predictions = np.asarray(base["predictions"], dtype=np.float64)
    benchmark_predictions = np.asarray(
        benchmark["predictions"], dtype=np.float64
    )
    transform, matrix = _fit_transform(
        base, features, args.pca_components, args.seed
    )
    residual = _select_residual(
        matrix,
        labels,
        base_predictions,
        benchmark_predictions,
        folds,
        args,
    )
    gate = _gate(
        labels,
        benchmark_predictions,
        residual["predictions"],
        folds,
        residual["selected"]["budget"],
        args,
    )
    return {
        "base_root": base_root,
        "benchmark_root": benchmark_root,
        "feature_root": feature_root,
        "base_path": base_path,
        "benchmark_path": benchmark_path,
        "feature_path": feature_path,
        "base": base,
        "benchmark": benchmark,
        "features": features,
        "keys": keys,
        "labels": labels,
        "folds": folds,
        "base_predictions": base_predictions,
        "benchmark_predictions": benchmark_predictions,
        "base_metrics": affinity_metrics(labels, base_predictions),
        "benchmark_metrics": affinity_metrics(labels, benchmark_predictions),
        "transform": transform,
        "matrix": matrix,
        "residual": residual,
        "gate": gate,
    }


def _verify_selection(args, fitted, current_payload):
    selection_path = Path(args.selection_result).expanduser().resolve()
    try:
        with selection_path.open("r", encoding="utf-8") as handle:
            sealed = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("V20 selection result is malformed") from error
    if sealed.get("version") != VERSION:
        raise ValueError("V20 selection version is invalid")
    if sealed.get("status") != "validation_passed":
        raise ValueError("V20 selection did not pass validation")
    if sealed.get("search_space") != current_payload["search_space"]:
        raise ValueError("V20 selection search space changed before test")
    for name, value in current_payload["inputs"].items():
        if name.endswith("_sha256") and sealed.get("inputs", {}).get(name) != value:
            raise ValueError("V20 selection input SHA256 changed: %s" % name)
    expected = current_payload["selected_residual"]
    observed = sealed.get("selected_residual", {})
    for name in (
        "name",
        "configuration_index",
        "configuration_seed",
        "minimum_leaf",
        "max_features",
        "budget",
    ):
        if observed.get(name) != expected.get(name):
            raise ValueError("V20 residual selection changed before test: %s" % name)
    for name, value in expected["metrics"].items():
        if not np.isclose(
            observed.get("metrics", {}).get(name), value, rtol=0.0, atol=1e-12
        ):
            raise ValueError("V20 validation metric changed before test: %s" % name)
    if not fitted["gate"]["passed"]:
        raise ValueError("V20 recomputed validation gate failed before test")
    if sealed.get("validation_gate", {}).get("criteria") != fitted["gate"].get(
        "criteria"
    ):
        raise ValueError("V20 validation gate criteria changed before test")
    return selection_path, sealed


def _save_valid_predictions(output, fitted):
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=fitted["keys"],
        labels=fitted["labels"],
        v16_base_predictions=fitted["base_predictions"],
        v19_benchmark_predictions=fitted["benchmark_predictions"],
        predictions=np.asarray(
            fitted["residual"]["predictions"], dtype=np.float64
        ),
        fold_ids=fitted["folds"],
    )


def run(args):
    if bool(args.validation_only) == bool(args.selection_result):
        raise ValueError(
            "V20 requires exactly one of --validation-only or --selection-result"
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V20 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    fitted = _fit_validation(args)
    status = (
        "validation_passed" if fitted["gate"]["passed"] else "validation_failed"
    )
    payload = _validation_payload(args, fitted, output, status)
    if args.validation_only:
        _save_valid_predictions(output, fitted)
        _atomic_json(output / "result.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    if not fitted["gate"]["passed"]:
        raise ValueError("V20 recomputed validation selection does not pass")
    selection_path, sealed = _verify_selection(args, fitted, payload)

    # Test paths are constructed and opened only after the validation seal has
    # passed every check above.
    base_test_path, base_test = _load_npz(
        fitted["base_root"] / "test_predictions.npz", BASE_TEST_FIELDS
    )
    benchmark_test_path, benchmark_test = _load_npz(
        fitted["benchmark_root"] / "test_predictions.npz", BENCHMARK_TEST_FIELDS
    )
    feature_test_path, feature_test = _load_npz(
        fitted["feature_root"] / "test_features.npz", FEATURE_FIELDS
    )
    base_test = _limit(base_test, args.limit)
    benchmark_test = _limit(benchmark_test, args.limit)
    feature_test = _limit(feature_test, args.limit)
    test_keys, test_labels = _align(base_test, benchmark_test, "V16/V19 test")
    _align(base_test, feature_test, "V16/features test")

    selected = fitted["residual"]["selected"]
    final_tree = _new_tree(
        selected["minimum_leaf"],
        selected["max_features"],
        args.final_estimators,
        args.n_jobs,
        selected["configuration_seed"],
    )
    final_tree.fit(
        fitted["matrix"], fitted["labels"] - fitted["base_predictions"]
    )
    test_matrix = _apply_transform(base_test, feature_test, fitted["transform"])
    base_test_predictions = np.asarray(
        base_test["predictions"], dtype=np.float64
    )
    benchmark_test_predictions = np.asarray(
        benchmark_test["predictions"], dtype=np.float64
    )
    predictions = base_test_predictions + float(selected["budget"]) * final_tree.predict(
        test_matrix
    )
    test_metrics = affinity_metrics(test_labels, predictions)
    base_test_metrics = affinity_metrics(test_labels, base_test_predictions)
    benchmark_test_metrics = affinity_metrics(
        test_labels, benchmark_test_predictions
    )
    final_status = (
        "passed"
        if float(test_metrics["CI"]) > float(benchmark_test_metrics["CI"])
        else "not_improved"
    )

    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "pca": fitted["transform"]["pca"],
            "scalar_scaler": fitted["transform"]["scalar_scaler"],
            "tree": final_tree,
            "residual": {
                name: selected[name]
                for name in (
                    "name",
                    "configuration_index",
                    "configuration_seed",
                    "minimum_leaf",
                    "max_features",
                    "budget",
                )
            },
            "validation_gate": fitted["gate"],
        },
    )
    _save_valid_predictions(output, fitted)
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v16_base_predictions=base_test_predictions,
        v19_benchmark_predictions=benchmark_test_predictions,
        predictions=np.asarray(predictions, dtype=np.float64),
    )
    result = dict(payload)
    result.update(
        {
            "status": final_status,
            "validation_selected": True,
            "count": int(test_labels.size),
            "v16_base_test_metrics": dict(base_test_metrics),
            "v19_benchmark_test_metrics": dict(benchmark_test_metrics),
            "test_metrics": dict(test_metrics),
            "delta_ci_vs_v19_benchmark": float(
                test_metrics["CI"] - benchmark_test_metrics["CI"]
            ),
            "selection": {
                "result": str(selection_path),
                "result_sha256": sha256_file(selection_path),
            },
            "search_space": sealed["search_space"],
        }
    )
    result["inputs"].update(
        {
            "v16_test": str(base_test_path),
            "v16_test_sha256": sha256_file(base_test_path),
            "v19_benchmark_test": str(benchmark_test_path),
            "v19_benchmark_test_sha256": sha256_file(benchmark_test_path),
            "v19_features_test": str(feature_test_path),
            "v19_features_test_sha256": sha256_file(feature_test_path),
        }
    )
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
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.0005)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--allow-zero-budget", action="store_true")
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
