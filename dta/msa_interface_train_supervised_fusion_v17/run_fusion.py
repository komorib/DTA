"""Train-supervised V17 affinity heads with validation-gated V16 blending."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
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

from . import FEATURE_VERSION, VERSION


ANCHOR_VALID_FIELDS = (
    "keys",
    "labels",
    "v12_base_predictions",
    "v15_anchor_predictions",
    "predictions",
    "fold_ids",
)
ANCHOR_TEST_FIELDS = tuple(
    name for name in ANCHOR_VALID_FIELDS if name != "fold_ids"
)
FEATURE_FIELDS = (
    "keys",
    "labels",
    "pretrain_features",
    "affinity_features",
)
VALID_PREDICTION_FIELDS = (
    "keys",
    "labels",
    "anchor_predictions",
    "direct_head_predictions",
    "predictions",
    "fold_ids",
)
FAMILIES = ("pretrain", "affinity", "delta", "joint")
DEFAULT_PCA_COMPONENTS = (96, 192)
BLEND_WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0)
HEAD_SPECS = (
    {
        "name": "extra_trees_leaf3_features085",
        "kind": "extra_trees",
        "minimum_leaf": 3,
        "max_features": 0.85,
    },
    {
        "name": "extra_trees_leaf7_features100",
        "kind": "extra_trees",
        "minimum_leaf": 7,
        "max_features": 1.0,
    },
    {
        "name": "hist_gradient_leaf31",
        "kind": "hist_gradient",
        "max_leaf_nodes": 31,
        "minimum_leaf": 20,
        "learning_rate": 0.04,
        "l2_regularization": 1.0,
    },
    {
        "name": "pls_32",
        "kind": "pls",
        "components": 32,
    },
)


def _limit_archive(values, limit, label):
    if limit is None:
        return values
    limit = int(limit)
    if limit <= 2:
        raise ValueError("V17 %s limit must exceed two" % label)
    return {name: np.asarray(value)[:limit] for name, value in values.items()}


def _normalize_fold_ids(values, shape):
    fold_ids = np.asarray(values, dtype=np.int64)
    if fold_ids.shape != shape:
        raise ValueError("V17 grouped fold IDs have the wrong shape")
    unique = np.unique(fold_ids)
    if unique.size < 2:
        raise ValueError("V17 grouped fold IDs contain fewer than two folds")
    return np.searchsorted(unique, fold_ids).astype(np.int64, copy=False)


def _keys_labels(values, label):
    keys = np.asarray(values["keys"]).astype(str)
    labels = np.asarray(values["labels"], dtype=np.float64)
    if keys.shape != labels.shape or np.unique(keys).size != keys.size:
        raise ValueError("V17 %s keys are malformed or duplicated" % label)
    if not np.isfinite(labels).all():
        raise ValueError("V17 %s labels contain non-finite values" % label)
    return keys, labels


def _align(reference, other, label):
    reference_keys, reference_labels = _keys_labels(reference, "reference " + label)
    other_keys, other_labels = _keys_labels(other, label)
    if not np.array_equal(reference_keys, other_keys):
        raise ValueError("V17 %s keys do not align" % label)
    if not np.allclose(reference_labels, other_labels, atol=1e-6):
        raise ValueError("V17 %s labels do not align" % label)
    return reference_keys, reference_labels


def _family_raw(features, family):
    pretrain = np.asarray(features["pretrain_features"], dtype=np.float32)
    affinity = np.asarray(features["affinity_features"], dtype=np.float32)
    if pretrain.ndim != 2 or affinity.shape != pretrain.shape:
        raise ValueError("V17 spatial feature matrices have the wrong shape")
    if family == "pretrain":
        result = pretrain
    elif family == "affinity":
        result = affinity
    elif family == "delta":
        result = affinity - pretrain
    elif family == "joint":
        result = np.column_stack((affinity, affinity - pretrain)).astype(
            np.float32, copy=False
        )
    else:
        raise ValueError("unknown V17 feature family: " + str(family))
    if not np.isfinite(result).all():
        raise ValueError("V17 feature family contains non-finite values")
    return result


def _fit_transform(train_raw, other_raw, requested_components, seed):
    if train_raw.ndim != 2 or other_raw.ndim != 2:
        raise ValueError("V17 transform inputs must be matrices")
    if train_raw.shape[1] != other_raw.shape[1]:
        raise ValueError("V17 transform feature widths differ")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    other_scaled = scaler.transform(other_raw)
    components = min(
        int(requested_components), train_raw.shape[0] - 1, train_raw.shape[1]
    )
    if components <= 0:
        raise ValueError("V17 PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    )
    train_transformed = pca.fit_transform(train_scaled)
    other_transformed = pca.transform(other_scaled)
    if not np.isfinite(train_transformed).all() or not np.isfinite(
        other_transformed
    ).all():
        raise RuntimeError("V17 transformed features contain non-finite values")
    return (
        {"scaler": scaler, "pca": pca},
        np.asarray(train_transformed, dtype=np.float32),
        np.asarray(other_transformed, dtype=np.float32),
    )


def _apply_transform(raw, transform):
    values = transform["pca"].transform(transform["scaler"].transform(raw))
    if not np.isfinite(values).all():
        raise RuntimeError("V17 applied transform contains non-finite values")
    return np.asarray(values, dtype=np.float32)


def _head_spec(name):
    for spec in HEAD_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise ValueError("unknown V17 head: " + str(name))


def _new_head(spec, args, seed, feature_width, final=False):
    if spec["kind"] == "extra_trees":
        estimators = args.final_estimators if final else args.selection_estimators
        return _new_tree(
            spec["minimum_leaf"],
            spec["max_features"],
            estimators,
            args.n_jobs,
            seed,
        )
    if spec["kind"] == "hist_gradient":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(args.hist_max_iter),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["minimum_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            random_state=int(seed),
        )
    if spec["kind"] == "pls":
        components = max(1, min(int(spec["components"]), int(feature_width)))
        return PLSRegression(
            n_components=components,
            scale=False,
            max_iter=500,
            tol=1e-6,
        )
    raise ValueError("unknown V17 head kind: " + str(spec["kind"]))


def _fit_head(spec, arguments, train_features, train_labels, seed, final=False):
    model = _new_head(
        spec, arguments, seed, train_features.shape[1], final=final
    )
    model.fit(train_features, train_labels)
    return model


def _predict_head(model, features):
    predictions = np.asarray(model.predict(features), dtype=np.float64).reshape(-1)
    if predictions.shape != (features.shape[0],) or not np.isfinite(
        predictions
    ).all():
        raise RuntimeError("V17 head predictions are malformed")
    return predictions


def _candidate_seed(args, family_index, component_index, head_index):
    return int(args.seed) + 10000 * family_index + 100 * component_index + head_index


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
                "v16_ci": float(anchor_metrics["CI"]),
                "candidate_ci": float(candidate_metrics["CI"]),
                "ci_gain": gain,
            }
        )
    return diagnostics, positive


def _score(metrics, weight, family_index, component_index, head_index):
    return (
        float(metrics["CI"]),
        float(metrics["Spearman"]),
        -float(metrics["RMSE"]),
        -float(metrics["MAE"]),
        -abs(float(weight) - 0.30),
        -family_index,
        -component_index,
        -head_index,
    )


def _select_validation(train, valid, anchor, fold_ids, args):
    _, train_labels = _keys_labels(train, "train features")
    _, valid_labels = _keys_labels(valid, "validation features")
    requested_components = tuple(sorted(set(int(v) for v in args.pca_components)))
    if not requested_components or requested_components[0] <= 0:
        raise ValueError("V17 PCA component inventory is invalid")
    candidate_summaries = {}
    best = None
    for family_index, family in enumerate(FAMILIES):
        train_raw = _family_raw(train, family)
        valid_raw = _family_raw(valid, family)
        seen_components = set()
        for component_index, requested in enumerate(requested_components):
            transform, train_features, valid_features = _fit_transform(
                train_raw,
                valid_raw,
                requested,
                int(args.seed) + 1000 * family_index + component_index,
            )
            actual_components = int(train_features.shape[1])
            if actual_components in seen_components:
                continue
            seen_components.add(actual_components)
            for head_index, original_spec in enumerate(HEAD_SPECS):
                spec = dict(original_spec)
                seed = _candidate_seed(
                    args, family_index, component_index, head_index
                )
                model = _fit_head(
                    spec, args, train_features, train_labels, seed, final=False
                )
                direct_predictions = _predict_head(model, valid_features)
                direct_metrics = affinity_metrics(valid_labels, direct_predictions)
                head_best = None
                for weight in BLEND_WEIGHTS:
                    predictions = anchor + float(weight) * (
                        direct_predictions - anchor
                    )
                    metrics = affinity_metrics(valid_labels, predictions)
                    diagnostics, positive_folds = _fold_diagnostics(
                        valid_labels, anchor, predictions, fold_ids
                    )
                    value = {
                        "family": family,
                        "requested_pca_components": int(requested),
                        "pca_components": actual_components,
                        "transform_seed": int(args.seed)
                        + 1000 * family_index
                        + component_index,
                        "head": spec,
                        "seed": int(seed),
                        "head_weight": float(weight),
                        "direct_metrics": dict(direct_metrics),
                        "metrics": dict(metrics),
                        "positive_folds": int(positive_folds),
                        "fold_diagnostics": diagnostics,
                    }
                    score = _score(
                        metrics,
                        weight,
                        family_index,
                        component_index,
                        head_index,
                    )
                    if head_best is None or score > head_best[0]:
                        head_best = (score, value)
                    if best is None or score > best[0]:
                        best = (
                            score,
                            value,
                            np.asarray(direct_predictions, dtype=np.float64),
                            np.asarray(predictions, dtype=np.float64),
                        )
                name = "%s_pca%d_%s" % (
                    family,
                    actual_components,
                    spec["name"],
                )
                candidate_summaries[name] = head_best[1]
    if best is None:
        raise RuntimeError("V17 candidate inventory is empty")
    _, selected, direct_predictions, predictions = best
    return {
        "selected": selected,
        "direct_predictions": direct_predictions,
        "predictions": predictions,
        "candidate_summaries": candidate_summaries,
    }


def _gate(selected, anchor_metrics, args):
    metrics = selected["metrics"]
    ci_gain = float(metrics["CI"] - anchor_metrics["CI"])
    spearman_gain = float(metrics["Spearman"] - anchor_metrics["Spearman"])
    rmse_increase = float(metrics["RMSE"] - anchor_metrics["RMSE"])
    passed = (
        (
            bool(getattr(args, "allow_zero_head_weight", False))
            or float(selected["head_weight"]) > 0.0
        )
        and ci_gain >= float(args.minimum_validation_ci_gain)
        and spearman_gain >= float(args.minimum_spearman_gain)
        and rmse_increase <= float(args.maximum_rmse_increase)
        and int(selected["positive_folds"]) >= int(args.minimum_positive_folds)
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "spearman_gain": spearman_gain,
        "rmse_increase": rmse_increase,
        "positive_folds": int(selected["positive_folds"]),
        "criteria": {
            "minimum_validation_ci_gain": float(args.minimum_validation_ci_gain),
            "minimum_spearman_gain": float(args.minimum_spearman_gain),
            "maximum_rmse_increase": float(args.maximum_rmse_increase),
            "minimum_positive_folds": int(args.minimum_positive_folds),
            "require_positive_head_weight": not bool(
                getattr(args, "allow_zero_head_weight", False)
            ),
        },
    }


def _search_space(args):
    return {
        "families": list(FAMILIES),
        "pca_components": sorted(set(int(v) for v in args.pca_components)),
        "heads": [dict(value) for value in HEAD_SPECS],
        "blend_weights": list(BLEND_WEIGHTS),
        "selection_estimators": int(args.selection_estimators),
        "final_estimators": int(args.final_estimators),
        "hist_max_iter": int(args.hist_max_iter),
        "seed": int(args.seed),
    }


def _validation_result(
    output,
    args,
    train_count,
    train_path,
    valid_path,
    anchor_path,
    keys,
    labels,
    fold_ids,
    anchor_predictions,
    anchor_metrics,
    selection,
    gate,
):
    predictions_path = output / "valid_predictions.npz"
    _atomic_npz(
        predictions_path,
        keys=keys,
        labels=labels,
        anchor_predictions=np.asarray(anchor_predictions, dtype=np.float64),
        direct_head_predictions=np.asarray(
            selection["direct_predictions"], dtype=np.float64
        ),
        predictions=np.asarray(selection["predictions"], dtype=np.float64),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
    )
    status = "validation_passed" if gate["passed"] else "validation_failed"
    result = {
        "status": status,
        "version": VERSION,
        "validation_selected": bool(gate["passed"]),
        "counts": {
            "train": int(train_count),
            "valid": int(labels.size),
            "test": None,
        },
        "limits": {
            "train": args.limit_train,
            "valid": args.limit_valid,
            "test": args.limit_test,
        },
        "selected": selection["selected"],
        "selected_validation_metrics": dict(selection["selected"]["metrics"]),
        "v16_anchor_validation_metrics": dict(anchor_metrics),
        "validation_gate": gate,
        "candidate_summaries": selection["candidate_summaries"],
        "search_space": _search_space(args),
        "valid_predictions_sha256": sha256_file(predictions_path),
        "inputs": {
            "v17_train_features": str(train_path),
            "v17_train_features_sha256": sha256_file(train_path),
            "v17_valid_features": str(valid_path),
            "v17_valid_features_sha256": sha256_file(valid_path),
            "v16_anchor_valid": str(anchor_path),
            "v16_anchor_valid_sha256": sha256_file(anchor_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("V17 %s is malformed" % label) from error


def _verify_selection(args, train_path, valid_path, anchor_path):
    selection_path = Path(args.selection_result).expanduser().resolve()
    sealed = _load_json(selection_path, "selection result")
    if sealed.get("version") != VERSION or sealed.get("status") != "validation_passed":
        raise ValueError("V17 selection is not a passed V17 result")
    expected = {
        "v17_train_features_sha256": sha256_file(train_path),
        "v17_valid_features_sha256": sha256_file(valid_path),
        "v16_anchor_valid_sha256": sha256_file(anchor_path),
    }
    for name, value in expected.items():
        if sealed.get("inputs", {}).get(name) != value:
            raise ValueError("V17 selection input SHA256 mismatch: " + name)
    if sealed.get("search_space") != _search_space(args):
        raise ValueError("V17 selection search space changed after validation")
    selected = sealed.get("selected", {})
    if selected.get("family") not in FAMILIES:
        raise ValueError("V17 selected family is invalid")
    if selected.get("head", {}).get("name") not in {
        spec["name"] for spec in HEAD_SPECS
    }:
        raise ValueError("V17 selected head is invalid")
    if float(selected.get("head_weight", -1.0)) not in BLEND_WEIGHTS:
        raise ValueError("V17 selected head weight is invalid")
    if sealed.get("validation_gate", {}).get("criteria") != _gate(
        selected,
        sealed["v16_anchor_validation_metrics"],
        args,
    )["criteria"]:
        raise ValueError("V17 validation gate criteria changed")
    prediction_path = selection_path.parent / "valid_predictions.npz"
    if not prediction_path.is_file() or sha256_file(prediction_path) != sealed.get(
        "valid_predictions_sha256"
    ):
        raise ValueError("V17 sealed validation predictions changed")
    return selection_path, sealed, prediction_path


def _fit_selected(train, other, selected, args, final=False):
    family = selected["family"]
    transform, train_features, other_features = _fit_transform(
        _family_raw(train, family),
        _family_raw(other, family),
        int(selected["pca_components"]),
        int(selected["transform_seed"]),
    )
    spec = _head_spec(selected["head"]["name"])
    seed = int(selected["seed"])
    model = _fit_head(
        spec,
        args,
        train_features,
        np.asarray(train["labels"], dtype=np.float64),
        seed,
        final=final,
    )
    return transform, model, _predict_head(model, other_features)


def run(args):
    if bool(args.validation_only) == bool(args.selection_result):
        raise ValueError(
            "V17 requires exactly one of --validation-only or --selection-result"
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V17 fusion output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root).expanduser().resolve()
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    train_path, train = _load_npz(feature_root / "train_features.npz", FEATURE_FIELDS)
    valid_path, valid = _load_npz(feature_root / "valid_features.npz", FEATURE_FIELDS)
    anchor_valid_path, anchor_valid = _load_npz(
        anchor_root / "valid_predictions.npz", ANCHOR_VALID_FIELDS
    )
    train = _limit_archive(train, args.limit_train, "train")
    valid = _limit_archive(valid, args.limit_valid, "validation")
    anchor_valid = _limit_archive(anchor_valid, args.limit_valid, "validation")
    train_keys, _ = _keys_labels(train, "train features")
    valid_keys, valid_labels = _align(anchor_valid, valid, "validation features")
    if np.intersect1d(train_keys, valid_keys).size:
        raise ValueError("V17 train and validation keys overlap")
    fold_ids = _normalize_fold_ids(anchor_valid["fold_ids"], valid_labels.shape)
    anchor_predictions = np.asarray(anchor_valid["predictions"], dtype=np.float64)
    anchor_metrics = affinity_metrics(valid_labels, anchor_predictions)

    if args.validation_only:
        selection = _select_validation(
            train, valid, anchor_predictions, fold_ids, args
        )
        gate = _gate(selection["selected"], anchor_metrics, args)
        return _validation_result(
            output,
            args,
            int(np.asarray(train["labels"]).size),
            train_path,
            valid_path,
            anchor_valid_path,
            valid_keys,
            valid_labels,
            fold_ids,
            anchor_predictions,
            anchor_metrics,
            selection,
            gate,
        )

    selection_path, sealed, sealed_prediction_path = _verify_selection(
        args, train_path, valid_path, anchor_valid_path
    )
    feature_result = _load_json(feature_root / "result.json", "feature result")
    if feature_result.get("version") != FEATURE_VERSION or feature_result.get("status") != "complete":
        raise ValueError("V17 test features were not gated and completed")
    if feature_result.get("selection_result_sha256") != sha256_file(selection_path):
        raise ValueError("V17 feature extraction used a different selection seal")
    test_path = feature_root / "test_features.npz"
    if feature_result.get("test_feature_sha256") != sha256_file(test_path):
        raise ValueError("V17 test feature SHA256 mismatch")

    # Reproduce the selected validation model only; the full candidate grid is not rerun.
    selected = sealed["selected"]
    _, _, reproduced_direct = _fit_selected(train, valid, selected, args, final=False)
    reproduced = anchor_predictions + float(selected["head_weight"]) * (
        reproduced_direct - anchor_predictions
    )
    _, sealed_predictions = _load_npz(
        sealed_prediction_path, VALID_PREDICTION_FIELDS
    )
    if not np.allclose(
        reproduced,
        np.asarray(sealed_predictions["predictions"], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("V17 selected validation model is not reproducible")

    anchor_test_path, anchor_test = _load_npz(
        anchor_root / "test_predictions.npz", ANCHOR_TEST_FIELDS
    )
    test_feature_path, test = _load_npz(test_path, FEATURE_FIELDS)
    anchor_test = _limit_archive(anchor_test, args.limit_test, "test")
    test = _limit_archive(test, args.limit_test, "test")
    test_keys, test_labels = _align(anchor_test, test, "test features")
    if np.intersect1d(np.concatenate((train_keys, valid_keys)), test_keys).size:
        raise ValueError("V17 train/validation and test keys overlap")

    combined = {
        name: np.concatenate((np.asarray(train[name]), np.asarray(valid[name])), axis=0)
        for name in FEATURE_FIELDS
    }
    transform, model, direct_test_predictions = _fit_selected(
        combined, test, selected, args, final=True
    )
    anchor_test_predictions = np.asarray(anchor_test["predictions"], dtype=np.float64)
    test_predictions = anchor_test_predictions + float(selected["head_weight"]) * (
        direct_test_predictions - anchor_test_predictions
    )
    test_metrics = affinity_metrics(test_labels, test_predictions)
    anchor_test_metrics = affinity_metrics(test_labels, anchor_test_predictions)
    status = (
        "passed"
        if float(test_metrics["CI"]) > float(anchor_test_metrics["CI"])
        else "not_improved"
    )
    _atomic_joblib(
        output / "model.joblib",
        {
            "version": VERSION,
            "family": selected["family"],
            "pca_components": int(selected["pca_components"]),
            "head": dict(selected["head"]),
            "head_weight": float(selected["head_weight"]),
            "transform": transform,
            "model": model,
            "refit": "train_plus_validation",
            "selection_result_sha256": sha256_file(selection_path),
        },
    )
    _atomic_npz(
        output / "valid_predictions.npz",
        **{name: sealed_predictions[name] for name in VALID_PREDICTION_FIELDS},
    )
    _atomic_npz(
        output / "test_predictions.npz",
        keys=test_keys,
        labels=test_labels,
        anchor_predictions=anchor_test_predictions,
        direct_head_predictions=np.asarray(direct_test_predictions, dtype=np.float64),
        predictions=np.asarray(test_predictions, dtype=np.float64),
    )
    result = {
        "status": status,
        "version": VERSION,
        "validation_selected": True,
        "counts": {
            "train": int(np.asarray(train["labels"]).size),
            "valid": int(valid_labels.size),
            "test": int(test_labels.size),
            "final_refit": int(np.asarray(combined["labels"]).size),
        },
        "limits": sealed.get("limits"),
        "selected": selected,
        "selected_validation_metrics": sealed["selected_validation_metrics"],
        "v16_anchor_validation_metrics": sealed["v16_anchor_validation_metrics"],
        "validation_gate": sealed["validation_gate"],
        "direct_head_test_metrics": affinity_metrics(
            test_labels, direct_test_predictions
        ),
        "v16_anchor_test_metrics": dict(anchor_test_metrics),
        "test_metrics": dict(test_metrics),
        "delta_ci_vs_v16_anchor": float(
            test_metrics["CI"] - anchor_test_metrics["CI"]
        ),
        "search_space": sealed["search_space"],
        "selection": {
            "result": str(selection_path),
            "result_sha256": sha256_file(selection_path),
        },
        "inputs": {
            "v17_train_features": str(train_path),
            "v17_train_features_sha256": sha256_file(train_path),
            "v17_valid_features": str(valid_path),
            "v17_valid_features_sha256": sha256_file(valid_path),
            "v17_test_features": str(test_feature_path),
            "v17_test_features_sha256": sha256_file(test_feature_path),
            "v16_anchor_valid": str(anchor_valid_path),
            "v16_anchor_valid_sha256": sha256_file(anchor_valid_path),
            "v16_anchor_test": str(anchor_test_path),
            "v16_anchor_test_sha256": sha256_file(anchor_test_path),
        },
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--selection-result")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=list(DEFAULT_PCA_COMPONENTS),
    )
    parser.add_argument("--minimum-validation-ci-gain", type=float, default=0.001)
    parser.add_argument("--minimum-spearman-gain", type=float, default=0.0)
    parser.add_argument("--maximum-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--allow-zero-head-weight", action="store_true")
    parser.add_argument("--selection-estimators", type=int, default=400)
    parser.add_argument("--final-estimators", type=int, default=1200)
    parser.add_argument("--hist-max-iter", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
