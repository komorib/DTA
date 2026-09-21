"""Fuse frozen V12 fusion predictions with pretrained V13 predictions and latents."""

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


VERSION = "msa-interface-pretrained-fusion-v13"
ANCHOR_SCALAR_NAMES = ("v11_fusion_predictions", "v12_predictions", "predictions")


def _validate_alignment(anchor, expert, split):
    anchor_keys = np.asarray(anchor["keys"]).astype(str)
    expert_keys = np.asarray(expert["keys"]).astype(str)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.float64)
    expert_labels = np.asarray(expert["labels"], dtype=np.float64)
    if not np.array_equal(anchor_keys, expert_keys):
        raise ValueError("V12 fusion/V13 %s keys do not align" % split)
    if not np.allclose(anchor_labels, expert_labels, atol=1e-6):
        raise ValueError("V12 fusion/V13 %s labels do not align" % split)
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
        raise ValueError("V12 fusion grouped fold IDs have the wrong shape")
    unique_folds = np.unique(fold_ids)
    if unique_folds.size < 2:
        raise ValueError("V12 fusion grouped fold IDs contain fewer than two folds")
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
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=valid_keys,
        labels=valid_labels,
        v12_fusion_predictions=anchor_predictions,
        v13_predictions=np.asarray(expert_predictions, dtype=np.float64),
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
        "selected_blend": {
            "v12_fusion_weight": blend["weight"],
            "v13_weight": 1.0 - blend["weight"],
            "metrics": blend["metrics"],
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "v12_fusion_validation_metrics": dict(anchor_metrics),
        "selected_validation_metrics": dict(residual["metrics"]),
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "v13_valid": str(expert_valid_path),
            "v13_valid_sha256": sha256_file(expert_valid_path),
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
        raise ValueError("V13 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    v13_root = Path(args.v13_root).expanduser().resolve()
    anchor_valid_path, anchor_valid = _load_npz(
        anchor_root / "valid_predictions.npz",
        ("keys", "labels", "fold_ids") + ANCHOR_SCALAR_NAMES,
    )
    v13_valid_path, v13_valid = _load_npz(
        v13_root / "valid_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    anchor_valid = _limit_archive(anchor_valid, args.limit)
    v13_valid = _limit_archive(v13_valid, args.limit)
    valid_keys, valid_labels = _validate_alignment(anchor_valid, v13_valid, "validation")
    fold_ids = _normalize_fold_ids(
        anchor_valid["fold_ids"], valid_labels.shape
    )

    anchor_valid_predictions = np.asarray(
        anchor_valid["predictions"], dtype=np.float64
    )
    blend = select_standardized_blend(
        anchor_valid_predictions,
        np.asarray(v13_valid["predictions"], dtype=np.float64),
        valid_labels,
    )
    latent_valid = np.asarray(v13_valid["latents"], dtype=np.float64)
    if latent_valid.ndim != 2 or latent_valid.shape[0] != valid_labels.size:
        raise ValueError("V13 validation latents have the wrong shape")
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
        _scalar_matrix(anchor_valid, v13_valid["predictions"])
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
    validation_selected = validation_gain >= float(args.minimum_validation_ci_gain)
    if not validation_selected:
        return _write_validation_result(
            output,
            "validation_failed",
            args,
            valid_keys,
            valid_labels,
            fold_ids,
            anchor_valid_predictions,
            v13_valid["predictions"],
            anchor_valid_metrics,
            blend,
            residual,
            validation_gain,
            anchor_valid_path,
            v13_valid_path,
        )
    if not (v13_root / "test_predictions.npz").is_file():
        return _write_validation_result(
            output,
            "validation_passed",
            args,
            valid_keys,
            valid_labels,
            fold_ids,
            anchor_valid_predictions,
            v13_valid["predictions"],
            anchor_valid_metrics,
            blend,
            residual,
            validation_gain,
            anchor_valid_path,
            v13_valid_path,
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
    v13_test_path, v13_test = _load_npz(
        v13_root / "test_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    anchor_test = _limit_archive(anchor_test, args.limit)
    v13_test = _limit_archive(v13_test, args.limit)
    test_keys, test_labels = _validate_alignment(anchor_test, v13_test, "test")
    compressed_test = pca.transform(np.asarray(v13_test["latents"], dtype=np.float64))
    scalar_test = scalar_scaler.transform(_scalar_matrix(anchor_test, v13_test["predictions"]))
    features_test = np.column_stack((compressed_test, scalar_test))
    blend_test = apply_standardized_blend(
        anchor_test["predictions"], v13_test["predictions"], blend
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
        v12_fusion_predictions=anchor_valid_predictions,
        v13_predictions=np.asarray(v13_valid["predictions"], dtype=np.float64),
        predictions=np.asarray(residual["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v12_fusion_predictions=np.asarray(
            anchor_test["predictions"], dtype=np.float64
        ),
        v13_predictions=np.asarray(v13_test["predictions"], dtype=np.float64),
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
        "selected_blend": {
            "v12_fusion_weight": blend["weight"],
            "v13_weight": 1.0 - blend["weight"],
            "metrics": blend["metrics"],
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "v12_fusion_validation_metrics": dict(anchor_valid_metrics),
        "selected_validation_metrics": dict(residual["metrics"]),
        "v12_fusion_test_metrics": dict(anchor_test_metrics),
        "candidate_test_metrics": dict(candidate_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v12_fusion": float(test_metrics["CI"] - anchor_test_metrics["CI"]),
        "inputs": {
            "anchor_valid": str(anchor_valid_path),
            "anchor_valid_sha256": sha256_file(anchor_valid_path),
            "anchor_test": str(anchor_test_path),
            "anchor_test_sha256": sha256_file(anchor_test_path),
            "v13_valid": str(v13_valid_path),
            "v13_valid_sha256": sha256_file(v13_valid_path),
            "v13_test": str(v13_test_path),
            "v13_test_sha256": sha256_file(v13_test_path),
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
    parser.add_argument("--v13-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.003)
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
