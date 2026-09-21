"""Fuse the frozen V10 predictor with V11 MSA pair predictions and latents."""

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


VERSION = "msa-pair-fusion-v11"
V10_SCALAR_NAMES = ("v8_predictions", "v9_predictions", "blend_predictions", "predictions")


def _validate_alignment(anchor, expert, split):
    anchor_keys = np.asarray(anchor["keys"]).astype(str)
    expert_keys = np.asarray(expert["keys"]).astype(str)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.float64)
    expert_labels = np.asarray(expert["labels"], dtype=np.float64)
    if not np.array_equal(anchor_keys, expert_keys):
        raise ValueError("V10/V11 %s keys do not align" % split)
    if not np.allclose(anchor_labels, expert_labels, atol=1e-6):
        raise ValueError("V10/V11 %s labels do not align" % split)
    return anchor_keys, anchor_labels


def _scalar_matrix(anchor, expert_predictions):
    return np.column_stack(
        [np.asarray(anchor[name], dtype=np.float64) for name in V10_SCALAR_NAMES]
        + [np.asarray(expert_predictions, dtype=np.float64)]
    )


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V11 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    v10_root = Path(args.v10_root).expanduser().resolve()
    v11_root = Path(args.v11_root).expanduser().resolve()
    v10_valid_path, v10_valid = _load_npz(
        v10_root / "valid_predictions.npz",
        ("keys", "labels", "fold_ids") + V10_SCALAR_NAMES,
    )
    v11_valid_path, v11_valid = _load_npz(
        v11_root / "valid_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    valid_keys, valid_labels = _validate_alignment(v10_valid, v11_valid, "validation")
    fold_ids = np.asarray(v10_valid["fold_ids"], dtype=np.int64)
    if fold_ids.shape != valid_labels.shape or len(np.unique(fold_ids)) < 2:
        raise ValueError("V10 grouped fold IDs are invalid")

    anchor_valid = np.asarray(v10_valid["predictions"], dtype=np.float64)
    blend = select_standardized_blend(
        anchor_valid, np.asarray(v11_valid["predictions"], dtype=np.float64), valid_labels
    )
    latent_valid = np.asarray(v11_valid["latents"], dtype=np.float64)
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
        _scalar_matrix(v10_valid, v11_valid["predictions"])
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
    anchor_valid_metrics = affinity_metrics(valid_labels, anchor_valid)
    validation_selected = float(residual["metrics"]["CI"]) > float(
        anchor_valid_metrics["CI"]
    )
    if not validation_selected:
        residual.update(
            {
                "minimum_leaf": 10,
                "max_features": 1.0,
                "budget": 0.0,
                "name": "v10_fallback",
                "predictions": anchor_valid,
                "metrics": anchor_valid_metrics,
            }
        )
    final_tree = _new_tree(
        residual["minimum_leaf"],
        residual["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(features_valid, valid_labels - blend["predictions"])

    v10_test_path, v10_test = _load_npz(
        v10_root / "test_predictions.npz", ("keys", "labels") + V10_SCALAR_NAMES
    )
    v11_test_path, v11_test = _load_npz(
        v11_root / "test_predictions.npz", ("keys", "labels", "predictions", "latents")
    )
    test_keys, test_labels = _validate_alignment(v10_test, v11_test, "test")
    compressed_test = pca.transform(np.asarray(v11_test["latents"], dtype=np.float64))
    scalar_test = scalar_scaler.transform(_scalar_matrix(v10_test, v11_test["predictions"]))
    features_test = np.column_stack((compressed_test, scalar_test))
    blend_test = apply_standardized_blend(
        v10_test["predictions"], v11_test["predictions"], blend
    )
    if validation_selected:
        candidate_test_predictions = blend_test + float(residual["budget"]) * final_tree.predict(
            features_test
        )
    else:
        candidate_test_predictions = np.asarray(v10_test["predictions"], dtype=np.float64)
    candidate_test_metrics = affinity_metrics(test_labels, candidate_test_predictions)
    anchor_test_metrics = affinity_metrics(test_labels, v10_test["predictions"])
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
        v10_predictions=anchor_valid,
        v11_predictions=np.asarray(v11_valid["predictions"], dtype=np.float64),
        predictions=np.asarray(residual["predictions"], dtype=np.float64),
        fold_ids=fold_ids,
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v10_predictions=np.asarray(v10_test["predictions"], dtype=np.float64),
        v11_predictions=np.asarray(v11_test["predictions"], dtype=np.float64),
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": "passed" if passed else "not_improved",
        "version": VERSION,
        "count": int(test_labels.size),
        "validation_selected": bool(validation_selected),
        "v10_validation_metrics": dict(anchor_valid_metrics),
        "selected_validation_metrics": dict(residual["metrics"]),
        "v10_test_metrics": dict(anchor_test_metrics),
        "candidate_test_metrics": dict(candidate_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v10": float(test_metrics["CI"] - anchor_test_metrics["CI"]),
        "inputs": {
            "v10_valid": str(v10_valid_path),
            "v10_valid_sha256": sha256_file(v10_valid_path),
            "v10_test": str(v10_test_path),
            "v10_test_sha256": sha256_file(v10_test_path),
            "v11_valid": str(v11_valid_path),
            "v11_valid_sha256": sha256_file(v11_valid_path),
            "v11_test": str(v11_test_path),
            "v11_test_sha256": sha256_file(v11_test_path),
        },
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-root", required=True)
    parser.add_argument("--v11-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--oof-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
