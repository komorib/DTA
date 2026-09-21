"""Validation-selected V18 pair-latent residual against the frozen V16 anchor."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dta.msa_rank_fusion_v10.run_pipeline import (
    _atomic_joblib,
    _atomic_npz,
    _new_tree,
    select_residual_model,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import VERSION


ANCHOR_SCALAR_NAMES = (
    "v12_base_predictions",
    "v15_anchor_predictions",
    "predictions",
)


def _load_npz(path, required):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing V19 input archive: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("V19 input archive is missing: %s" % ", ".join(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    return path, values


def _limit(values, limit):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 1:
        raise ValueError("V19 fusion limit must exceed one")
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _align(anchor, features, split):
    anchor_keys = np.asarray(anchor["keys"]).astype(str)
    feature_keys = np.asarray(features["keys"]).astype(str)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.float64)
    feature_labels = np.asarray(features["labels"], dtype=np.float64)
    if not np.array_equal(anchor_keys, feature_keys):
        raise ValueError("V16 anchor/V19 %s keys do not align" % split)
    if not np.allclose(anchor_labels, feature_labels, atol=1e-6):
        raise ValueError("V16 anchor/V19 %s labels do not align" % split)
    return anchor_keys, anchor_labels


def _fold_ids(values, shape):
    raw = np.asarray(values, dtype=np.int64)
    if raw.shape != shape:
        raise ValueError("V16 anchor fold IDs have the wrong shape")
    unique = np.unique(raw)
    if unique.size < 2:
        raise ValueError("V16 anchor contains fewer than two folds")
    return np.searchsorted(unique, raw).astype(np.int64, copy=False)


def _scalar_matrix(anchor, v18_predictions):
    return np.column_stack(
        [np.asarray(anchor[name], dtype=np.float64) for name in ANCHOR_SCALAR_NAMES]
        + [np.asarray(v18_predictions, dtype=np.float64)]
    )


def _promotion_gate(labels, anchor, candidate, folds, budget, args):
    fold_gains = []
    for fold in np.unique(folds):
        selected = folds == fold
        fold_gains.append(
            float(
                affinity_metrics(labels[selected], candidate[selected])["CI"]
                - affinity_metrics(labels[selected], anchor[selected])["CI"]
            )
        )
    anchor_metrics = affinity_metrics(labels, anchor)
    candidate_metrics = affinity_metrics(labels, candidate)
    ci_gain = float(candidate_metrics["CI"] - anchor_metrics["CI"])
    spearman_gain = float(
        candidate_metrics["Spearman"] - anchor_metrics["Spearman"]
    )
    rmse_increase = float(candidate_metrics["RMSE"] - anchor_metrics["RMSE"])
    positive_folds = int(sum(value > 0.0 for value in fold_gains))
    return {
        "passed": bool(
            float(budget) > 0.0
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


def _fit_validation(args):
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    feature_root = Path(args.v19_root).expanduser().resolve()
    anchor_path, anchor = _load_npz(
        anchor_root / "valid_predictions.npz",
        ("keys", "labels", "fold_ids") + ANCHOR_SCALAR_NAMES,
    )
    feature_path, features = _load_npz(
        feature_root / "valid_features.npz",
        ("keys", "labels", "v18_predictions", "pair_latents"),
    )
    anchor = _limit(anchor, args.limit)
    features = _limit(features, args.limit)
    keys, labels = _align(anchor, features, "validation")
    folds = _fold_ids(anchor["fold_ids"], labels.shape)
    anchor_predictions = np.asarray(anchor["predictions"], dtype=np.float64)
    pair_latents = np.asarray(features["pair_latents"], dtype=np.float64)
    if pair_latents.ndim != 2 or pair_latents.shape[0] != labels.size:
        raise ValueError("V19 pair latents have the wrong shape")
    components = min(
        int(args.pca_components), labels.size - 1, pair_latents.shape[1]
    )
    if components <= 0:
        raise ValueError("V19 PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=args.seed,
    )
    compressed = pca.fit_transform(pair_latents)
    scaler = StandardScaler()
    scalars = scaler.fit_transform(
        _scalar_matrix(anchor, features["v18_predictions"])
    )
    matrix = np.column_stack((compressed, scalars))
    residual = select_residual_model(
        matrix,
        labels,
        anchor_predictions,
        folds,
        estimators=args.oof_estimators,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    anchor_metrics = affinity_metrics(labels, anchor_predictions)
    gate = _promotion_gate(
        labels,
        anchor_predictions,
        residual["predictions"],
        folds,
        residual["budget"],
        args,
    )
    return {
        "anchor_root": anchor_root,
        "feature_root": feature_root,
        "anchor_path": anchor_path,
        "feature_path": feature_path,
        "anchor": anchor,
        "features": features,
        "keys": keys,
        "labels": labels,
        "folds": folds,
        "anchor_predictions": anchor_predictions,
        "anchor_metrics": anchor_metrics,
        "pca": pca,
        "scaler": scaler,
        "matrix": matrix,
        "components": components,
        "residual": residual,
        "gate": gate,
    }


def _validation_payload(args, fitted, status, output):
    residual = fitted["residual"]
    return {
        "status": status,
        "version": VERSION,
        "validation_selected": status == "validation_passed",
        "validation_count": int(fitted["labels"].size),
        "limit": args.limit,
        "pca_components": int(fitted["components"]),
        "selection_criteria": {
            "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
            "minimum_spearman_gain": float(args.minimum_spearman_gain),
            "maximum_rmse_increase": float(args.maximum_rmse_increase),
            "minimum_positive_folds": int(args.minimum_positive_folds),
            "require_positive_budget": True,
        },
        "selected_residual": {
            "name": residual["name"],
            "minimum_leaf": residual["minimum_leaf"],
            "max_features": residual["max_features"],
            "budget": residual["budget"],
            "metrics": residual["metrics"],
        },
        "validation_gate": fitted["gate"],
        "v16_anchor_validation_metrics": dict(fitted["anchor_metrics"]),
        "selected_validation_metrics": dict(residual["metrics"]),
        "inputs": {
            "anchor_valid": str(fitted["anchor_path"]),
            "anchor_valid_sha256": sha256_file(fitted["anchor_path"]),
            "v19_valid": str(fitted["feature_path"]),
            "v19_valid_sha256": sha256_file(fitted["feature_path"]),
        },
        "output": str(output),
    }


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_selection(selection, payload):
    if selection.get("version") != VERSION:
        raise ValueError("V19 selection version is invalid")
    if selection.get("status") != "validation_passed":
        raise ValueError("V19 selection did not pass validation")
    for name in ("anchor_valid_sha256", "v19_valid_sha256"):
        if selection.get("inputs", {}).get(name) != payload["inputs"][name]:
            raise ValueError("V19 selection input SHA256 changed: %s" % name)
    if selection.get("selection_criteria") != payload["selection_criteria"]:
        raise ValueError("V19 selection criteria changed before test")
    if int(selection.get("pca_components", -1)) != int(payload["pca_components"]):
        raise ValueError("V19 PCA selection changed before test")
    expected = payload["selected_residual"]
    observed = selection.get("selected_residual", {})
    for name in ("name", "minimum_leaf", "max_features", "budget"):
        if observed.get(name) != expected[name]:
            raise ValueError("V19 residual selection changed before test: %s" % name)
    for name, value in expected["metrics"].items():
        if not np.isclose(observed.get("metrics", {}).get(name), value, atol=1e-12):
            raise ValueError("V19 validation metric changed before test: %s" % name)


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V19 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    fitted = _fit_validation(args)
    status = "validation_passed" if fitted["gate"]["passed"] else "validation_failed"
    payload = _validation_payload(args, fitted, status, output)
    if args.selection_result is None:
        _atomic_npz(
            output / "valid_predictions.npz",
            keys=fitted["keys"],
            labels=fitted["labels"],
            v16_anchor_predictions=fitted["anchor_predictions"],
            predictions=np.asarray(
                fitted["residual"]["predictions"], dtype=np.float64
            ),
            fold_ids=fitted["folds"],
        )
        _write_json(output / "result.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload
    if not fitted["gate"]["passed"]:
        raise ValueError("V19 recomputed validation gate failed before test")
    selection_path = Path(args.selection_result).expanduser().resolve()
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    _validate_selection(selection, payload)
    _atomic_npz(
        output / "valid_predictions.npz",
        keys=fitted["keys"],
        labels=fitted["labels"],
        v16_anchor_predictions=fitted["anchor_predictions"],
        predictions=np.asarray(
            fitted["residual"]["predictions"], dtype=np.float64
        ),
        fold_ids=fitted["folds"],
    )

    residual = fitted["residual"]
    final_tree = _new_tree(
        residual["minimum_leaf"],
        residual["max_features"],
        args.final_estimators,
        args.n_jobs,
        args.seed,
    )
    final_tree.fit(
        fitted["matrix"], fitted["labels"] - fitted["anchor_predictions"]
    )
    anchor_test_path, anchor_test = _load_npz(
        fitted["anchor_root"] / "test_predictions.npz",
        ("keys", "labels") + ANCHOR_SCALAR_NAMES,
    )
    feature_test_path, feature_test = _load_npz(
        fitted["feature_root"] / "test_features.npz",
        ("keys", "labels", "v18_predictions", "pair_latents"),
    )
    anchor_test = _limit(anchor_test, args.limit)
    feature_test = _limit(feature_test, args.limit)
    test_keys, test_labels = _align(anchor_test, feature_test, "test")
    compressed_test = fitted["pca"].transform(
        np.asarray(feature_test["pair_latents"], dtype=np.float64)
    )
    scalar_test = fitted["scaler"].transform(
        _scalar_matrix(anchor_test, feature_test["v18_predictions"])
    )
    test_matrix = np.column_stack((compressed_test, scalar_test))
    anchor_test_predictions = np.asarray(
        anchor_test["predictions"], dtype=np.float64
    )
    predictions = anchor_test_predictions + float(residual["budget"]) * final_tree.predict(
        test_matrix
    )
    test_metrics = affinity_metrics(test_labels, predictions)
    anchor_test_metrics = affinity_metrics(test_labels, anchor_test_predictions)
    final_status = (
        "passed"
        if float(test_metrics["CI"]) > float(anchor_test_metrics["CI"])
        else "not_improved"
    )
    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "pca": fitted["pca"],
            "scalar_scaler": fitted["scaler"],
            "tree": final_tree,
            "residual": {
                "name": residual["name"],
                "minimum_leaf": residual["minimum_leaf"],
                "max_features": residual["max_features"],
                "budget": residual["budget"],
            },
        },
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        v16_anchor_predictions=anchor_test_predictions,
        predictions=np.asarray(predictions, dtype=np.float64),
    )
    result = dict(payload)
    result.update(
        {
            "status": final_status,
            "validation_selected": True,
            "count": int(test_labels.size),
            "v16_anchor_test_metrics": dict(anchor_test_metrics),
            "test_metrics": dict(test_metrics),
            "delta_ci_vs_v16_anchor": float(
                test_metrics["CI"] - anchor_test_metrics["CI"]
            ),
            "selection_result": str(selection_path),
            "selection_result_sha256": sha256_file(selection_path),
        }
    )
    result["inputs"].update(
        {
            "anchor_test": str(anchor_test_path),
            "anchor_test_sha256": sha256_file(anchor_test_path),
            "v19_test": str(feature_test_path),
            "v19_test_sha256": sha256_file(feature_test_path),
        }
    )
    _write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--v19-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-result")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.0005)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--oof-estimators", type=int, default=400)
    parser.add_argument("--final-estimators", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
