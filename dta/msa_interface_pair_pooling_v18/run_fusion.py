"""Fuse frozen V16 anchor predictions with pretrained V18 predictions and latents."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dta.msa_rank_fusion_v10.run_pipeline import (
    _atomic_joblib,
    _atomic_npz,
    _load_npz,
    _new_tree,
    apply_standardized_blend,
    select_residual_model,
    select_standardized_blend,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file


VERSION = "msa-interface-pair-pooling-fusion-v18"
ANCHOR_SCALAR_NAMES = ("v12_base_predictions", "v15_anchor_predictions", "predictions")


def _promotion_gate(labels, anchor, candidate, fold_ids, budget, args):
    fold_gains = []
    for fold in np.unique(fold_ids):
        selected = fold_ids == fold
        fold_gains.append(
            float(affinity_metrics(labels[selected], candidate[selected])["CI"]
            - affinity_metrics(labels[selected], anchor[selected])["CI"])
        )
    anchor_metrics = affinity_metrics(labels, anchor)
    candidate_metrics = affinity_metrics(labels, candidate)
    ci_gain = float(candidate_metrics["CI"] - anchor_metrics["CI"])
    spearman_gain = float(candidate_metrics["Spearman"] - anchor_metrics["Spearman"])
    rmse_increase = float(candidate_metrics["RMSE"] - anchor_metrics["RMSE"])
    positive_folds = int(sum(gain > 0.0 for gain in fold_gains))
    return {
        "passed": bool(
            (bool(args.allow_zero_contribution) or float(budget) > 0.0)
            and ci_gain >= float(args.minimum_validation_ci_gain)
            and spearman_gain >= float(args.minimum_spearman_gain)
            and rmse_increase <= float(args.maximum_rmse_increase)
            and positive_folds >= int(args.minimum_positive_folds)
        ),
        "ci_gain": ci_gain,
        "spearman_gain": spearman_gain,
        "rmse_increase": rmse_increase,
        "positive_folds": positive_folds,
        "fold_ci_gains": fold_gains,
    }


def _validate_alignment(anchor, expert, split):
    anchor_keys = np.asarray(anchor["keys"]).astype(str)
    expert_keys = np.asarray(expert["keys"]).astype(str)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.float64)
    expert_labels = np.asarray(expert["labels"], dtype=np.float64)
    if not np.array_equal(anchor_keys, expert_keys):
        raise ValueError("V16 anchor/V18 %s keys do not align" % split)
    if not np.allclose(anchor_labels, expert_labels, atol=1e-6):
        raise ValueError("V16 anchor/V18 %s labels do not align" % split)
    return anchor_keys, anchor_labels


def _scalar_matrix(anchor, expert_predictions):
    return np.column_stack(
        [np.asarray(anchor[name], dtype=np.float64) for name in ANCHOR_SCALAR_NAMES]
        + [np.asarray(expert_predictions, dtype=np.float64)]
    )


def _limit_archive(values, limit):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 1:
        raise ValueError("fusion limit must exceed one")
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _normalize_fold_ids(fold_ids, expected_shape):
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if fold_ids.shape != expected_shape:
        raise ValueError("V16 anchor grouped fold IDs have the wrong shape")
    unique_folds = np.unique(fold_ids)
    if unique_folds.size < 2:
        raise ValueError("V16 anchor grouped fold IDs contain fewer than two folds")
    return np.searchsorted(unique_folds, fold_ids).astype(np.int64, copy=False)


def _write_validation_result(
    output,
    status,
    args,
    valid_keys,
    valid_labels,
    fold_ids,
    anchor_predictions,
    expert_predictions,
    anchor_metrics,
    blend,
    residual,
    validation_gain,
    anchor_valid_path,
    expert_valid_path,
):
    validation_gate = _promotion_gate(
        valid_labels, anchor_predictions, residual["predictions"],
        fold_ids, max(float(residual["budget"]), 1.0 - float(blend["weight"])), args
    )
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=valid_keys,
        labels=valid_labels,
        v16_anchor_predictions=anchor_predictions,
        v18_predictions=np.asarray(expert_predictions, dtype=np.float64),
        predictions=np.asarray(residual["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    result = {
        "status": status,
        "version": VERSION,
        "validation_count": int(valid_labels.size),
        "limit": args.limit,
        "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
        "validation_ci_gain": float(validation_gain),
        "validation_selected": status == "validation_passed",
        "validation_gate": validation_gate,
        "selected_blend": {
            "v16_anchor_weight": blend["weight"],
            "v18_weight": 1.0 - blend["weight"],
            "metrics": blend["metrics"],
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "v16_anchor_validation_metrics": dict(anchor_metrics),
        "selected_validation_metrics": dict(residual["metrics"]),
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "v18_valid": str(expert_valid_path),
            "v18_valid_sha256": sha256_file(expert_valid_path),
        },
        "output": str(output),
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V18 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    v18_root = Path(args.v18_root).expanduser().resolve()
    anchor_valid_path, anchor_valid = _load_npz(
        anchor_root / "valid_predictions.npz",
        ("keys", "labels", "fold_ids") + ANCHOR_SCALAR_NAMES,
    )
    v18_valid_path, v18_valid = _load_npz(
        v18_root / "valid_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    anchor_valid = _limit_archive(anchor_valid, args.limit)
    v18_valid = _limit_archive(v18_valid, args.limit)
    valid_keys, valid_labels = _validate_alignment(anchor_valid, v18_valid, "validation")
    fold_ids = _normalize_fold_ids(
        anchor_valid["fold_ids"], valid_labels.shape
    )

    anchor_valid_predictions = np.asarray(
        anchor_valid["predictions"], dtype=np.float64
    )
    blend = select_standardized_blend(
        anchor_valid_predictions,
        np.asarray(v18_valid["predictions"], dtype=np.float64),
        valid_labels,
    )
    latent_valid = np.asarray(v18_valid["latents"], dtype=np.float64)
    if latent_valid.ndim != 2 or latent_valid.shape[0] != valid_labels.size:
        raise ValueError("V18 validation latents have the wrong shape")
    components = min(int(args.pca_components), valid_labels.size - 1, latent_valid.shape[1])
    if components <= 0:
        raise ValueError("PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=args.seed,
    )
    compressed_valid = pca.fit_transform(latent_valid)
    scalar_scaler = StandardScaler()
    scalar_valid = scalar_scaler.fit_transform(
        _scalar_matrix(anchor_valid, v18_valid["predictions"])
    )
    features_valid = np.column_stack((compressed_valid, scalar_valid))
    residual = select_residual_model(
        features_valid,
        valid_labels,
        blend["predictions"],
        fold_ids,
        estimators=args.oof_estimators,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    anchor_valid_metrics = affinity_metrics(valid_labels, anchor_valid_predictions)
    validation_gain = float(residual["metrics"]["CI"]) - float(
        anchor_valid_metrics["CI"]
    )
    promotion_gate = _promotion_gate(
        valid_labels, anchor_valid_predictions, residual["predictions"],
        fold_ids, max(float(residual["budget"]), 1.0 - float(blend["weight"])), args
    )
    validation_selected = bool(promotion_gate["passed"])
    if not validation_selected:
        return _write_validation_result(
            output,
            "validation_failed",
            args,
            valid_keys,
            valid_labels,
            fold_ids,
            anchor_valid_predictions,
            v18_valid["predictions"],
            anchor_valid_metrics,
            blend,
            residual,
            validation_gain,
            anchor_valid_path,
            v18_valid_path,
        )
    if not (v18_root / "test_predictions.npz").is_file():
        return _write_validation_result(
            output,
            "validation_passed",
            args,
            valid_keys,
            valid_labels,
            fold_ids,
            anchor_valid_predictions,
            v18_valid["predictions"],
            anchor_valid_metrics,
            blend,
            residual,
            validation_gain,
            anchor_valid_path,
            v18_valid_path,
        )
    final_tree = _new_tree(
        residual["minimum_leaf"],
        residual["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(features_valid, valid_labels - blend["predictions"])

    anchor_test_path, anchor_test = _load_npz(
        anchor_root / "test_predictions.npz", ("keys", "labels") + ANCHOR_SCALAR_NAMES
    )
    v18_test_path, v18_test = _load_npz(
        v18_root / "test_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    anchor_test = _limit_archive(anchor_test, args.limit)
    v18_test = _limit_archive(v18_test, args.limit)
    test_keys, test_labels = _validate_alignment(anchor_test, v18_test, "test")
    compressed_test = pca.transform(np.asarray(v18_test["latents"], dtype=np.float64))
    scalar_test = scalar_scaler.transform(_scalar_matrix(anchor_test, v18_test["predictions"]))
    features_test = np.column_stack((compressed_test, scalar_test))
    blend_test = apply_standardized_blend(
        anchor_test["predictions"], v18_test["predictions"], blend
    )
    if validation_selected:
        candidate_test_predictions = blend_test + float(residual["budget"]) * final_tree.predict(
            features_test
        )
    else:
        candidate_test_predictions = np.asarray(anchor_test["predictions"], dtype=np.float64)
    candidate_test_metrics = affinity_metrics(test_labels, candidate_test_predictions)
    anchor_test_metrics = affinity_metrics(test_labels, anchor_test["predictions"])
    passed = validation_selected and float(candidate_test_metrics["CI"]) > float(
        anchor_test_metrics["CI"]
    )
    test_predictions = candidate_test_predictions
    test_metrics = candidate_test_metrics

    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "pca": pca,
            "scalar_scaler": scalar_scaler,
            "tree": final_tree,
            "blend": {
                key: value
                for key, value in blend.items()
                if key not in ("predictions", "candidates")
            },
            "residual": {
                "name": residual["name"],
                "minimum_leaf": residual["minimum_leaf"],
                "max_features": residual["max_features"],
                "budget": residual["budget"],
            },
        },
    )
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=valid_keys,
        labels=valid_labels,
        v16_anchor_predictions=anchor_valid_predictions,
        v18_predictions=np.asarray(v18_valid["predictions"], dtype=np.float64),
        predictions=np.asarray(residual["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v16_anchor_predictions=np.asarray(
            anchor_test["predictions"], dtype=np.float64
        ),
        v18_predictions=np.asarray(v18_test["predictions"], dtype=np.float64),
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": "passed" if passed else "not_improved",
        "version": VERSION,
        "count": int(test_labels.size),
        "limit": args.limit,
        "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
        "validation_ci_gain": float(validation_gain),
        "validation_selected": bool(validation_selected),
        "validation_gate": promotion_gate,
        "selected_blend": {
            "v16_anchor_weight": blend["weight"],
            "v18_weight": 1.0 - blend["weight"],
            "metrics": blend["metrics"],
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "v16_anchor_validation_metrics": dict(anchor_valid_metrics),
        "selected_validation_metrics": dict(residual["metrics"]),
        "v16_anchor_test_metrics": dict(anchor_test_metrics),
        "candidate_test_metrics": dict(candidate_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v16_anchor": float(test_metrics["CI"] - anchor_test_metrics["CI"]),
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "anchor_test": str(anchor_test_path),
            "anchor_test_sha256": sha256_file(anchor_test_path),
            "v18_valid": str(v18_valid_path),
            "v18_valid_sha256": sha256_file(v18_valid_path),
            "v18_test": str(v18_test_path),
            "v18_test_sha256": sha256_file(v18_test_path),
        },
        "output": str(output),
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--v18-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.001)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--allow-zero-contribution", action="store_true")
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
