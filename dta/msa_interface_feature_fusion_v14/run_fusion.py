"""Validation-only fusion of frozen V12 with V14 pure interface features."""

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
    select_residual_model,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import VERSION


ANCHOR_FIELDS = (
    "keys",
    "labels",
    "fold_ids",
    "v11_fusion_predictions",
    "v12_predictions",
    "predictions",
)
FEATURE_FIELDS = (
    "keys",
    "labels",
    "pretrain_features",
    "affinity_features",
)
FAMILIES = ("pretrain", "affinity", "delta", "combined")


def _limit_archive(values, limit):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 1:
        raise ValueError("V14 fusion limit must exceed one")
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _normalize_fold_ids(values, shape):
    fold_ids = np.asarray(values, dtype=np.int64)
    if fold_ids.shape != shape:
        raise ValueError("V14 grouped fold IDs have the wrong shape")
    unique = np.unique(fold_ids)
    if unique.size < 2:
        raise ValueError("V14 grouped fold IDs contain fewer than two folds")
    return np.searchsorted(unique, fold_ids).astype(np.int64, copy=False)


def _validate_alignment(anchor, features, split):
    anchor_keys = np.asarray(anchor["keys"]).astype(str)
    feature_keys = np.asarray(features["keys"]).astype(str)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.float64)
    feature_labels = np.asarray(features["labels"], dtype=np.float64)
    if not np.array_equal(anchor_keys, feature_keys):
        raise ValueError(f"V12 anchor/V14 {split} keys do not align")
    if not np.allclose(anchor_labels, feature_labels, atol=1e-6):
        raise ValueError(f"V12 anchor/V14 {split} labels do not align")
    return anchor_keys, anchor_labels


def _family_raw(features, family):
    pretrain = np.asarray(features["pretrain_features"], dtype=np.float64)
    affinity = np.asarray(features["affinity_features"], dtype=np.float64)
    if pretrain.ndim != 2 or affinity.shape != pretrain.shape:
        raise ValueError("V14 interface feature matrices have the wrong shape")
    if family == "pretrain":
        result = pretrain
    elif family == "affinity":
        result = affinity
    elif family == "delta":
        result = affinity - pretrain
    elif family == "combined":
        result = np.column_stack((pretrain, affinity, affinity - pretrain))
    else:
        raise ValueError(f"unknown V14 interface feature family: {family}")
    if not np.isfinite(result).all():
        raise ValueError("V14 interface feature family contains non-finite values")
    return result


def _fit_transform(raw, requested_components, seed):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    components = min(int(requested_components), raw.shape[0] - 1, raw.shape[1])
    if components <= 0:
        raise ValueError("V14 PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    )
    transformed = pca.fit_transform(scaled)
    if not np.isfinite(transformed).all():
        raise RuntimeError("V14 transformed validation features are non-finite")
    return {"scaler": scaler, "pca": pca}, transformed


def _apply_transform(raw, transform):
    transformed = transform["pca"].transform(transform["scaler"].transform(raw))
    if not np.isfinite(transformed).all():
        raise RuntimeError("V14 transformed test features are non-finite")
    return transformed


def _fold_diagnostics(labels, anchor, candidate, fold_ids):
    diagnostics = []
    positive = 0
    for fold in np.unique(fold_ids):
        selected = fold_ids == fold
        anchor_metrics = affinity_metrics(labels[selected], anchor[selected])
        candidate_metrics = affinity_metrics(labels[selected], candidate[selected])
        gain = float(candidate_metrics["CI"] - anchor_metrics["CI"])
        positive += int(gain > 0.0)
        diagnostics.append(
            {
                "fold": int(fold),
                "count": int(selected.sum()),
                "anchor_ci": float(anchor_metrics["CI"]),
                "candidate_ci": float(candidate_metrics["CI"]),
                "ci_gain": gain,
            }
        )
    return diagnostics, positive


def _select_validation(features, labels, anchor, fold_ids, args):
    candidates = {}
    best = None
    for family_index, family in enumerate(FAMILIES):
        transform, transformed = _fit_transform(
            _family_raw(features, family), args.pca_components, args.seed + family_index
        )
        residual = select_residual_model(
            transformed,
            labels,
            anchor,
            fold_ids,
            estimators=args.oof_estimators,
            n_jobs=args.n_jobs,
            seed=args.seed + 1000 * family_index,
        )
        diagnostics, positive_folds = _fold_diagnostics(
            labels, anchor, residual["predictions"], fold_ids
        )
        summary = {
            "family": family,
            "pca_components": int(transformed.shape[1]),
            "selected_residual": {
                "name": residual["name"],
                "minimum_leaf": residual["minimum_leaf"],
                "max_features": residual["max_features"],
                "budget": residual["budget"],
                "metrics": residual["metrics"],
            },
            "positive_folds": int(positive_folds),
            "fold_diagnostics": diagnostics,
        }
        candidates[family] = summary
        score = (
            float(residual["metrics"]["CI"]),
            -float(residual["metrics"]["RMSE"]),
            -float(residual["metrics"]["MAE"]),
            -family_index,
        )
        if best is None or score > best[0]:
            best = (score, family, transform, transformed, residual, summary)
    _, family, transform, transformed, residual, summary = best
    return {
        "family": family,
        "transform": transform,
        "features": transformed,
        "residual": residual,
        "summary": summary,
        "family_candidates": candidates,
    }


def _gate(selection, anchor_metrics, args):
    metrics = selection["residual"]["metrics"]
    ci_gain = float(metrics["CI"] - anchor_metrics["CI"])
    spearman_gain = float(metrics["Spearman"] - anchor_metrics["Spearman"])
    rmse_increase = float(metrics["RMSE"] - anchor_metrics["RMSE"])
    positive_folds = int(selection["summary"]["positive_folds"])
    passed = (
        (bool(args.allow_zero_budget) or float(selection["residual"]["budget"]) > 0.0)
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
    anchor_predictions,
    anchor_metrics,
    selection,
    gate,
    anchor_valid_path,
    feature_valid_path,
):
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=keys,
        labels=labels,
        anchor_predictions=np.asarray(anchor_predictions, dtype=np.float64),
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
        "selected_residual": selection["summary"]["selected_residual"],
        "selected_validation_metrics": dict(selection["residual"]["metrics"]),
        "v12_fusion_validation_metrics": dict(anchor_metrics),
        "validation_gate": gate,
        "family_candidates": selection["family_candidates"],
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "v14_valid": str(feature_valid_path),
            "v14_valid_sha256": sha256_file(feature_valid_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V14 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()

    anchor_valid_path, anchor_valid = _load_npz(
        anchor_root / "valid_predictions.npz", ANCHOR_FIELDS
    )
    feature_valid_path, feature_valid = _load_npz(
        feature_root / "valid_features.npz", FEATURE_FIELDS
    )
    anchor_valid = _limit_archive(anchor_valid, args.limit)
    feature_valid = _limit_archive(feature_valid, args.limit)
    keys, labels = _validate_alignment(anchor_valid, feature_valid, "validation")
    fold_ids = _normalize_fold_ids(anchor_valid["fold_ids"], labels.shape)
    anchor_predictions = np.asarray(anchor_valid["predictions"], dtype=np.float64)
    anchor_metrics = affinity_metrics(labels, anchor_predictions)
    selection = _select_validation(
        feature_valid, labels, anchor_predictions, fold_ids, args
    )
    gate = _gate(selection, anchor_metrics, args)
    if not gate["passed"]:
        return _validation_result(
            output,
            "validation_failed",
            args,
            keys,
            labels,
            fold_ids,
            anchor_predictions,
            anchor_metrics,
            selection,
            gate,
            anchor_valid_path,
            feature_valid_path,
        )

    feature_test_file = feature_root / "test_features.npz"
    if not feature_test_file.is_file():
        return _validation_result(
            output,
            "validation_passed",
            args,
            keys,
            labels,
            fold_ids,
            anchor_predictions,
            anchor_metrics,
            selection,
            gate,
            anchor_valid_path,
            feature_valid_path,
        )

    final_tree = _new_tree(
        selection["residual"]["minimum_leaf"],
        selection["residual"]["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(selection["features"], labels - anchor_predictions)

    anchor_test_path, anchor_test = _load_npz(
        anchor_root / "test_predictions.npz",
        tuple(name for name in ANCHOR_FIELDS if name != "fold_ids"),
    )
    feature_test_path, feature_test = _load_npz(feature_test_file, FEATURE_FIELDS)
    anchor_test = _limit_archive(anchor_test, args.limit)
    feature_test = _limit_archive(feature_test, args.limit)
    test_keys, test_labels = _validate_alignment(anchor_test, feature_test, "test")
    anchor_test_predictions = np.asarray(anchor_test["predictions"], dtype=np.float64)
    test_raw = _family_raw(feature_test, selection["family"])
    test_features = _apply_transform(test_raw, selection["transform"])
    test_predictions = anchor_test_predictions + float(
        selection["residual"]["budget"]
    ) * final_tree.predict(test_features)
    test_metrics = affinity_metrics(test_labels, test_predictions)
    anchor_test_metrics = affinity_metrics(test_labels, anchor_test_predictions)
    passed = float(test_metrics["CI"]) > float(anchor_test_metrics["CI"])

    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "family": selection["family"],
            "transform": selection["transform"],
            "tree": final_tree,
            "residual": {
                "name": selection["residual"]["name"],
                "minimum_leaf": selection["residual"]["minimum_leaf"],
                "max_features": selection["residual"]["max_features"],
                "budget": selection["residual"]["budget"],
            },
            "validation_gate": gate,
        },
    )
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=keys,
        labels=labels,
        anchor_predictions=anchor_predictions,
        predictions=np.asarray(selection["residual"]["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        anchor_predictions=anchor_test_predictions,
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": "passed" if passed else "not_improved",
        "version": VERSION,
        "count": int(test_labels.size),
        "limit": args.limit,
        "validation_selected": True,
        "selected_family": selection["family"],
        "selected_residual": selection["summary"]["selected_residual"],
        "selected_validation_metrics": dict(selection["residual"]["metrics"]),
        "v12_fusion_validation_metrics": dict(anchor_metrics),
        "validation_gate": gate,
        "v12_fusion_test_metrics": dict(anchor_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v12_fusion": float(
            test_metrics["CI"] - anchor_test_metrics["CI"]
        ),
        "family_candidates": selection["family_candidates"],
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "anchor_test": str(anchor_test_path),
            "anchor_test_sha256": sha256_file(anchor_test_path),
            "v14_valid": str(feature_valid_path),
            "v14_valid_sha256": sha256_file(feature_valid_path),
            "v14_test": str(feature_test_path),
            "v14_test_sha256": sha256_file(feature_test_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.001)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.005)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--allow-zero-budget", action="store_true")
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
