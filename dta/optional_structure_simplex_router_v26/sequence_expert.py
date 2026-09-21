"""Train-only sequence/SMILES and calibrated binary experts for V26."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.sequence_smiles_binary_probe_v1.core import (
    BiasSVDModel,
    RepresentationEncoder,
    TargetStatisticsEncoder,
    affinity_to_score,
    best_decision_threshold,
    binary_metrics,
    fit_models,
)

from .core import RANKS, pair_key, safe_average_precision, safe_roc_auc


SIMPLEX_UNITS = 20
AFFINITY_COMPONENTS = {
    "DAVIS": ("v22", "reg_all", "bias_svd_rank8", "bias_svd_rank16"),
    "KIBA": ("v22", "reg_all", "bias_svd_rank8", "bias_svd_rank16"),
}
BINARY_COMPONENTS = {
    "DAVIS": ("v22", "balanced_extra_trees", "reg_all", "bias_svd_rank8"),
    "KIBA": ("v22", "bias_svd_rank16", "cls_all", "reg_all"),
}


def _decontaminated_rows(train_rows, evaluation_rows):
    evaluation_pairs = {pair_key(row) for row in evaluation_rows}
    filtered = [row for row in train_rows if pair_key(row) not in evaluation_pairs]
    if len(filtered) < 4:
        raise ValueError("too few decontaminated rows for V26 sequence expert")
    return filtered, len(train_rows) - len(filtered)


def fit_sequence_expert(train_rows, evaluation_rows, seed=20260910, max_iter=300):
    if not train_rows or not evaluation_rows:
        raise ValueError("V26 sequence expert needs train and evaluation rows")
    dataset = evaluation_rows[0]["dataset"]
    if any(row["dataset"] != dataset for row in train_rows + evaluation_rows):
        raise ValueError("V26 sequence expert rows must belong to one dataset")
    filtered, excluded = _decontaminated_rows(train_rows, evaluation_rows)
    for row in filtered + evaluation_rows:
        if "sequence" not in row or "smiles" not in row:
            raise ValueError("V26 sequence expert requires sequence and SMILES")

    train_smiles = [str(row["smiles"]) for row in filtered]
    train_proteins = [str(row["sequence"]) for row in filtered]
    train_affinity = np.asarray([row["label"] for row in filtered], dtype=np.float64)
    affinity_threshold = 7.522878745280337 if dataset == "DAVIS" else 12.1
    train_binary = (train_affinity >= affinity_threshold).astype(np.int8)

    representation = RepresentationEncoder(seed=int(seed)).fit(
        train_smiles, train_proteins
    )
    statistics = TargetStatisticsEncoder().fit(
        train_smiles, train_proteins, train_affinity, train_binary
    )
    train_representation = representation.transform(train_smiles, train_proteins)
    train_statistics = statistics.transform(
        train_smiles,
        train_proteins,
        affinity=train_affinity,
        binary=train_binary,
        leave_one_out=True,
    )
    train_features = np.concatenate(
        [train_representation, train_statistics], axis=1
    )
    models = fit_models(
        train_features,
        statistics_width=train_statistics.shape[1],
        affinity=train_affinity,
        binary=train_binary,
        max_iter=int(max_iter),
        seed=int(seed),
    )
    if dataset == "DAVIS":
        balanced_tree = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(seed) + 505,
        )
        balanced_tree.fit(train_features, train_binary)
        models["balanced_extra_trees"] = balanced_tree
    bias_svd = BiasSVDModel(rank=max(RANKS), seed=int(seed)).fit(
        [str(row["ligand_id"]) for row in filtered],
        [str(row["sequence_id"]) for row in filtered],
        train_affinity,
    )
    fitted = {
        "dataset": dataset,
        "representation": representation,
        "statistics": statistics,
        "statistics_width": int(train_statistics.shape[1]),
        "models": models,
        "bias_svd": bias_svd,
        "affinity_threshold": float(affinity_threshold),
        "affinity_scale": max(0.25, float(np.std(train_affinity)) * 0.35),
    }
    audit = {
        "dataset": dataset,
        "original_train_count": int(len(train_rows)),
        "fitted_train_count": int(len(filtered)),
        "excluded_exact_evaluation_pairs": int(excluded),
        "evaluation_count": int(len(evaluation_rows)),
        "fit_uses_evaluation_labels": False,
        "representation_uses_evaluation_inputs": False,
        "balanced_extra_trees": dataset == "DAVIS",
    }
    return fitted, audit


def predict_sequence_expert(fitted, rows, v22_fallback):
    smiles = [str(row["smiles"]) for row in rows]
    proteins = [str(row["sequence"]) for row in rows]
    representation = fitted["representation"].transform(smiles, proteins)
    statistics = fitted["statistics"].transform(smiles, proteins)
    features = np.concatenate([representation, statistics], axis=1)
    stats = features[:, -int(fitted["statistics_width"]):]
    affinity = {
        "v22": np.asarray(v22_fallback, dtype=np.float64),
        "reg_all": np.asarray(fitted["models"]["reg_all"].predict(features), dtype=np.float64),
        "reg_stats": np.asarray(fitted["models"]["reg_stats"].predict(stats), dtype=np.float64),
    }
    ligand_ids = [str(row["ligand_id"]) for row in rows]
    sequence_ids = [str(row["sequence_id"]) for row in rows]
    eligible = np.asarray(
        [
            ligand_id in fitted["bias_svd"].drug_index
            and sequence_id in fitted["bias_svd"].protein_index
            for ligand_id, sequence_id in zip(ligand_ids, sequence_ids)
        ],
        dtype=bool,
    )
    for rank in RANKS:
        prediction = np.asarray(
            fitted["bias_svd"].predict(ligand_ids, sequence_ids, rank=rank),
            dtype=np.float64,
        )
        prediction[~eligible] = affinity["v22"][~eligible]
        affinity["bias_svd_rank%d" % rank] = prediction

    threshold = float(fitted["affinity_threshold"])
    scale = float(fitted["affinity_scale"])
    scores = {
        name: affinity_to_score(prediction, threshold, scale)
        for name, prediction in affinity.items()
    }
    scores["cls_all"] = np.asarray(
        fitted["models"]["cls_all"].predict_proba(features)[:, 1], dtype=np.float64
    )
    scores["cls_stats"] = np.asarray(
        fitted["models"]["cls_stats"].predict_proba(stats)[:, 1], dtype=np.float64
    )
    if "balanced_extra_trees" in fitted["models"]:
        scores["balanced_extra_trees"] = np.asarray(
            fitted["models"]["balanced_extra_trees"].predict_proba(features)[:, 1],
            dtype=np.float64,
        )
    return affinity, scores, eligible


def _expression_score(expression, predictions):
    if expression["kind"] == "base":
        return np.asarray(predictions[expression["name"]], dtype=np.float64)
    if expression["kind"] == "blend":
        weight = float(expression["left_weight"])
        return (
            weight * np.asarray(predictions[expression["left"]], dtype=np.float64)
            + (1.0 - weight)
            * np.asarray(predictions[expression["right"]], dtype=np.float64)
        )
    if expression["kind"] == "mixture":
        terms = expression.get("terms", [])
        if not terms:
            raise ValueError("V26 mixture has no terms")
        result = np.zeros_like(
            np.asarray(predictions[terms[0]["name"]], dtype=np.float64)
        )
        total_weight = 0.0
        for term in terms:
            weight = float(term["weight"])
            if weight <= 0.0:
                raise ValueError("V26 mixture weights must be positive")
            result += weight * np.asarray(
                predictions[term["name"]], dtype=np.float64
            )
            total_weight += weight
        if not np.isclose(total_weight, 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V26 mixture weights must sum to one")
        return result
    raise ValueError("invalid V26 expression")


def _simplex_expressions(names):
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("V26 simplex requires four unique components")
    expressions = []
    for first in range(SIMPLEX_UNITS + 1):
        for second in range(SIMPLEX_UNITS - first + 1):
            for third in range(SIMPLEX_UNITS - first - second + 1):
                weights = (
                    first,
                    second,
                    third,
                    SIMPLEX_UNITS - first - second - third,
                )
                terms = [
                    {"name": name, "weight": float(weight) / SIMPLEX_UNITS}
                    for name, weight in zip(names, weights)
                    if weight > 0
                ]
                expressions.append({"kind": "mixture", "terms": terms})
    return expressions


def select_affinity_expression(labels, predictions, dataset):
    names = AFFINITY_COMPONENTS[dataset]
    missing = [name for name in names if name not in predictions]
    if missing:
        raise ValueError("missing V26 affinity components: %s" % ", ".join(missing))
    expressions = _simplex_expressions(names)
    candidates = []
    best = None
    for expression in expressions:
        prediction = _expression_score(expression, predictions)
        metrics = affinity_metrics(labels, prediction)
        candidate = {"expression": expression, "metrics": metrics}
        candidates.append(candidate)
        key = (metrics["CI"], -metrics["RMSE"], metrics["Spearman"])
        if best is None or key > best[0]:
            best = (key, candidate, prediction)
    return dict(best[1]), np.asarray(best[2], dtype=np.float64), candidates


def apply_affinity_expression(selection, predictions):
    return _expression_score(selection["expression"], predictions)


def select_binary_expression(affinity_labels, scores, affinity_threshold, dataset):
    binary_labels = (
        np.asarray(affinity_labels, dtype=np.float64) >= float(affinity_threshold)
    ).astype(np.int8)
    names = BINARY_COMPONENTS[dataset]
    missing = [name for name in names if name not in scores]
    if missing:
        raise ValueError("missing V26 binary components: %s" % ", ".join(missing))
    expressions = _simplex_expressions(names)
    candidates = []
    best = None
    for expression in expressions:
        score = _expression_score(expression, scores)
        candidate = {
            "expression": expression,
            "AUPRC": safe_average_precision(binary_labels, score),
            "AUROC": safe_roc_auc(binary_labels, score),
        }
        candidates.append(candidate)
        key = (candidate["AUPRC"], candidate["AUROC"])
        if best is None or key > best[0]:
            best = (key, expression, score)
    threshold, objective = best_decision_threshold(
        binary_labels, best[2], objective="mcc"
    )
    selected = {
        "expression": dict(best[1]),
        "PR-AUC": float(best[0][0]),
        "ROC-AUC": float(best[0][1]),
        "decision_threshold": float(threshold),
        "threshold_validation_objective": objective,
        "validation_metrics": binary_metrics(
            binary_labels, best[2], threshold
        ),
    }
    return selected, np.asarray(best[2], dtype=np.float64), candidates


def apply_binary_expression(selection, scores):
    return _expression_score(selection["expression"], scores)
