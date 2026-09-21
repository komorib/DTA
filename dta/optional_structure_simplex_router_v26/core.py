"""Core utilities for the validation-only V26 sparse simplex router.

The neural predictions remain those of V22 (1D protein + 2D ligand + optional
3D) and V20 (structured LP-PDBBind).  The only new learned component is a
train-only Bias-SVD branch for Davis/KIBA.  Exact evaluation pairs are removed
before fitting and samples containing an unseen entity hard-fallback to V22.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.sequence_smiles_binary_probe_v1.core import (
    BiasSVDModel,
    affinity_to_score,
    best_decision_threshold,
    binary_metrics,
)

from . import VERSION


DATASET_INDEX = {"LP_PDBBind": 0, "DAVIS": 1, "KIBA": 2}
BINARY_THRESHOLDS = {
    "DAVIS": 7.522878745280337,
    "KIBA": 12.1,
}
COLLABORATIVE_DATASETS = ("DAVIS", "KIBA")
RANKS = (8, 16, 32)
BLEND_WEIGHTS = tuple(round(index * 0.05, 2) for index in range(21))


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    rows = []
    sample_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("sample_id", "dataset", "label", "sequence_id", "ligand_id")
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(
                    "manifest row %d is missing %s" % (line_number, ", ".join(missing))
                )
            dataset = str(row["dataset"])
            if dataset not in DATASET_INDEX:
                raise ValueError("unknown dataset in manifest: %s" % dataset)
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError("duplicate sample_id in manifest: %s" % sample_id)
            label = float(row["label"])
            if not np.isfinite(label):
                raise ValueError("non-finite manifest label: %s" % sample_id)
            copied = dict(row)
            copied["sample_id"] = sample_id
            copied["dataset"] = dataset
            copied["label"] = label
            rows.append(copied)
            sample_ids.add(sample_id)
    if not rows:
        raise ValueError("manifest is empty: %s" % path)
    return path, rows


def rows_for_dataset(rows, dataset):
    return [row for row in rows if row["dataset"] == dataset]


def pair_key(row):
    return str(row["sequence_id"]), str(row["ligand_id"])


def _load_npz(path, required):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing V26 input archive: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("input archive is missing: %s" % ", ".join(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    return path, values


def load_v22_predictions(path, rows):
    """Align a V22 archive to manifest sample_ids and verify labels/datasets."""

    path, raw = _load_npz(
        path,
        ("sample_ids", "labels", "predictions", "base_predictions", "dataset_index"),
    )
    sample_ids = np.asarray(raw["sample_ids"]).astype(str)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("V22 prediction sample_ids are not unique")
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}
    order = []
    for row in rows:
        if row["sample_id"] not in lookup:
            raise ValueError("V22 predictions are missing sample_id %s" % row["sample_id"])
        order.append(lookup[row["sample_id"]])
    order = np.asarray(order, dtype=np.int64)
    labels = np.asarray(raw["labels"], dtype=np.float64)[order]
    expected_labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    if not np.allclose(labels, expected_labels, atol=1e-6, rtol=0.0):
        raise ValueError("V22 labels do not align with the manifest")
    dataset_index = np.asarray(raw["dataset_index"], dtype=np.int64)[order]
    expected_index = np.asarray([DATASET_INDEX[row["dataset"]] for row in rows], dtype=np.int64)
    if not np.array_equal(dataset_index, expected_index):
        raise ValueError("V22 dataset indices do not align with the manifest")
    predictions = np.asarray(raw["predictions"], dtype=np.float64)[order]
    base_predictions = np.asarray(raw["base_predictions"], dtype=np.float64)[order]
    if not np.isfinite(predictions).all() or not np.isfinite(base_predictions).all():
        raise ValueError("V22 predictions contain non-finite values")
    return path, {
        "labels": labels,
        "predictions": predictions,
        "base_predictions": base_predictions,
        "dataset_index": dataset_index,
    }


def load_v25_anchor_predictions(path, rows):
    """Align the sealed V25 validation archive used as the V26 gate anchor."""

    path, raw = _load_npz(
        path,
        ("sample_ids", "labels", "predictions", "binary_scores", "dataset_index"),
    )
    sample_ids = np.asarray(raw["sample_ids"]).astype(str)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("V25 anchor sample_ids are not unique")
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}
    try:
        order = np.asarray(
            [lookup[str(row["sample_id"])] for row in rows], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("V25 anchor is missing sample_id %s" % error.args[0])
    labels = np.asarray(raw["labels"], dtype=np.float64)[order]
    expected_labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    if not np.allclose(labels, expected_labels, atol=1e-6, rtol=0.0):
        raise ValueError("V25 anchor labels do not align with the manifest")
    dataset_index = np.asarray(raw["dataset_index"], dtype=np.int64)[order]
    expected_index = np.asarray(
        [DATASET_INDEX[row["dataset"]] for row in rows], dtype=np.int64
    )
    if not np.array_equal(dataset_index, expected_index):
        raise ValueError("V25 anchor dataset indices do not align")
    predictions = np.asarray(raw["predictions"], dtype=np.float64)[order]
    binary_scores = np.asarray(raw["binary_scores"], dtype=np.float64)[order]
    collaborative = dataset_index > 0
    if not np.isfinite(predictions).all():
        raise ValueError("V25 anchor affinity predictions are non-finite")
    if not np.isfinite(binary_scores[collaborative]).all():
        raise ValueError("V25 anchor binary scores are non-finite")
    return path, {
        "predictions": predictions,
        "binary_scores": binary_scores,
        "dataset_index": dataset_index,
    }


def load_v20_lp_predictions(path, lp_rows):
    """Align V20 structure_key predictions to LP rows; unavailable rows are masked."""

    path, raw = _load_npz(path, ("keys", "labels", "predictions"))
    keys = np.asarray(raw["keys"]).astype(str)
    if np.unique(keys).size != keys.size:
        raise ValueError("V20 prediction keys are not unique")
    lookup = {key: index for index, key in enumerate(keys.tolist())}
    predictions = np.full(len(lp_rows), np.nan, dtype=np.float64)
    available = np.zeros(len(lp_rows), dtype=bool)
    raw_labels = np.asarray(raw["labels"], dtype=np.float64)
    raw_predictions = np.asarray(raw["predictions"], dtype=np.float64)
    for index, row in enumerate(lp_rows):
        key = row.get("structure_key")
        if not bool(row.get("has_3d", False)) or key is None or str(key) not in lookup:
            continue
        source = lookup[str(key)]
        if not np.isclose(raw_labels[source], row["label"], atol=1e-6, rtol=0.0):
            raise ValueError("V20 label does not align for structure_key %s" % key)
        predictions[index] = raw_predictions[source]
        available[index] = True
    if available.any() and not np.isfinite(predictions[available]).all():
        raise ValueError("V20 LP predictions contain non-finite values")
    return path, predictions, available


def fit_decontaminated_bias_svd(train_rows, evaluation_rows, seed=20260909):
    """Fit on train labels after removing exact evaluation entity pairs."""

    if not train_rows or not evaluation_rows:
        raise ValueError("Bias-SVD requires non-empty train and evaluation rows")
    dataset = evaluation_rows[0]["dataset"]
    if dataset not in COLLABORATIVE_DATASETS:
        raise ValueError("Bias-SVD is restricted to Davis/KIBA")
    if any(row["dataset"] != dataset for row in train_rows + evaluation_rows):
        raise ValueError("Bias-SVD rows must belong to one dataset")
    evaluation_pairs = {pair_key(row) for row in evaluation_rows}
    filtered = [row for row in train_rows if pair_key(row) not in evaluation_pairs]
    excluded = len(train_rows) - len(filtered)
    if len(filtered) < 4:
        raise ValueError("too few decontaminated training rows for %s" % dataset)
    model = BiasSVDModel(rank=max(RANKS), seed=int(seed)).fit(
        [str(row["ligand_id"]) for row in filtered],
        [str(row["sequence_id"]) for row in filtered],
        np.asarray([row["label"] for row in filtered], dtype=np.float64),
    )
    remaining_pairs = {pair_key(row) for row in filtered}
    if remaining_pairs.intersection(evaluation_pairs):
        raise RuntimeError("exact evaluation pairs remained in Bias-SVD training")
    audit = {
        "dataset": dataset,
        "original_train_count": int(len(train_rows)),
        "fitted_train_count": int(len(filtered)),
        "excluded_exact_evaluation_pairs": int(excluded),
        "evaluation_count": int(len(evaluation_rows)),
        "evaluation_unique_pairs": int(len(evaluation_pairs)),
        "fit_uses_evaluation_labels": False,
    }
    return model, audit


def collaborative_predictions(model, evaluation_rows, fallback, rank):
    fallback = np.asarray(fallback, dtype=np.float64)
    if fallback.shape != (len(evaluation_rows),):
        raise ValueError("collaborative fallback has the wrong shape")
    ligand_ids = [str(row["ligand_id"]) for row in evaluation_rows]
    sequence_ids = [str(row["sequence_id"]) for row in evaluation_rows]
    eligible = np.asarray(
        [
            ligand_id in model.drug_index and sequence_id in model.protein_index
            for ligand_id, sequence_id in zip(ligand_ids, sequence_ids)
        ],
        dtype=bool,
    )
    raw = model.predict(ligand_ids, sequence_ids, rank=int(rank))
    predictions = fallback.copy()
    predictions[eligible] = raw[eligible]
    return predictions, eligible


def safe_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    if labels.sum() == labels.size:
        return 1.0
    return float(average_precision_score(labels, scores))


def safe_roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def hash_fold_ids(sample_ids, count=5):
    if int(count) < 2:
        raise ValueError("hash fold count must be at least two")
    folds = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:8], byteorder="big") % int(count))
    folds = np.asarray(folds, dtype=np.int64)
    if np.unique(folds).size != int(count):
        raise ValueError("one or more validation hash folds are empty")
    return folds


def _affinity_candidate(labels, predictions, rank, weight):
    return {
        "rank": None if rank is None else int(rank),
        "collaborative_weight": float(weight),
        "metrics": affinity_metrics(labels, predictions),
    }


def select_affinity_blend(labels, base_predictions, collaborative_by_rank):
    labels = np.asarray(labels, dtype=np.float64)
    base = np.asarray(base_predictions, dtype=np.float64)
    candidates = []
    best = None
    anchor = _affinity_candidate(labels, base, None, 0.0)
    candidates.append(anchor)
    best = ((anchor["metrics"]["CI"], -anchor["metrics"]["RMSE"], anchor["metrics"]["Spearman"]), anchor, base)
    for rank in RANKS:
        collaborative = np.asarray(collaborative_by_rank[rank], dtype=np.float64)
        for weight in BLEND_WEIGHTS[1:]:
            predictions = (1.0 - weight) * base + weight * collaborative
            candidate = _affinity_candidate(labels, predictions, rank, weight)
            candidates.append(candidate)
            score = (
                candidate["metrics"]["CI"],
                -candidate["metrics"]["RMSE"],
                candidate["metrics"]["Spearman"],
            )
            if score > best[0]:
                best = (score, candidate, predictions)
    return dict(best[1]), np.asarray(best[2], dtype=np.float64), candidates


def apply_affinity_selection(selection, base_predictions, collaborative_by_rank):
    weight = float(selection["collaborative_weight"])
    if weight == 0.0 or selection.get("rank") is None:
        return np.asarray(base_predictions, dtype=np.float64).copy()
    rank = int(selection["rank"])
    return (1.0 - weight) * np.asarray(base_predictions, dtype=np.float64) + weight * np.asarray(
        collaborative_by_rank[rank], dtype=np.float64
    )


def select_binary_blend(labels, base_predictions, collaborative_by_rank, affinity_threshold, affinity_scale):
    binary_labels = (np.asarray(labels, dtype=np.float64) >= float(affinity_threshold)).astype(np.int8)
    base_score = affinity_to_score(base_predictions, affinity_threshold, affinity_scale)
    candidates = []
    expressions = [(None, 0.0, base_score)]
    for rank in RANKS:
        collaborative_score = affinity_to_score(
            collaborative_by_rank[rank], affinity_threshold, affinity_scale
        )
        for weight in BLEND_WEIGHTS[1:]:
            score = (1.0 - weight) * base_score + weight * collaborative_score
            expressions.append((rank, weight, score))
    best = None
    for rank, weight, score in expressions:
        candidate = {
            "rank": None if rank is None else int(rank),
            "collaborative_weight": float(weight),
            "PR-AUC": safe_average_precision(binary_labels, score),
            "ROC-AUC": safe_roc_auc(binary_labels, score),
        }
        candidates.append(candidate)
        key = (candidate["PR-AUC"], candidate["ROC-AUC"])
        if best is None or key > best[0]:
            best = (key, candidate, score)
    threshold, objective = best_decision_threshold(binary_labels, best[2], objective="mcc")
    selected = dict(best[1])
    selected["decision_threshold"] = float(threshold)
    selected["threshold_validation_objective"] = objective
    selected["validation_metrics"] = binary_metrics(binary_labels, best[2], threshold)
    base_threshold, _ = best_decision_threshold(binary_labels, base_score, objective="mcc")
    base_metrics = binary_metrics(binary_labels, base_score, base_threshold)
    return selected, np.asarray(best[2], dtype=np.float64), candidates, base_metrics


def apply_binary_selection(selection, base_predictions, collaborative_by_rank, affinity_threshold, affinity_scale):
    base_score = affinity_to_score(base_predictions, affinity_threshold, affinity_scale)
    weight = float(selection["collaborative_weight"])
    if weight == 0.0 or selection.get("rank") is None:
        return base_score
    rank = int(selection["rank"])
    collaborative_score = affinity_to_score(
        collaborative_by_rank[rank], affinity_threshold, affinity_scale
    )
    return (1.0 - weight) * base_score + weight * collaborative_score


def validation_fold_diagnostics(
    sample_ids,
    affinity_labels,
    binary_labels,
    base_predictions,
    selected_predictions,
    base_scores,
    selected_scores,
):
    folds = hash_fold_ids(sample_ids, count=5)
    affinity_labels = np.asarray(affinity_labels, dtype=np.float64)
    binary_labels = np.asarray(binary_labels, dtype=np.int8)
    diagnostics = []
    positive_ci = 0
    positive_pr = 0
    for fold in range(5):
        mask = folds == fold
        base_affinity = affinity_metrics(affinity_labels[mask], base_predictions[mask])
        selected_affinity = affinity_metrics(affinity_labels[mask], selected_predictions[mask])
        base_pr = safe_average_precision(binary_labels[mask], base_scores[mask])
        selected_pr = safe_average_precision(binary_labels[mask], selected_scores[mask])
        ci_gain = float(selected_affinity["CI"] - base_affinity["CI"])
        pr_gain = float(selected_pr - base_pr)
        positive_ci += int(ci_gain > 0.0)
        positive_pr += int(pr_gain > 0.0)
        diagnostics.append(
            {
                "fold": int(fold),
                "count": int(mask.sum()),
                "ci_gain": ci_gain,
                "pr_auc_gain": pr_gain,
                "base_ci": float(base_affinity["CI"]),
                "selected_ci": float(selected_affinity["CI"]),
                "base_pr_auc": base_pr,
                "selected_pr_auc": selected_pr,
            }
        )
    return diagnostics, int(positive_ci), int(positive_pr)


def collaborative_validation_gate(
    sample_ids,
    affinity_labels,
    binary_labels,
    base_predictions,
    selected_predictions,
    base_scores,
    selected_scores,
    minimum_ci_gain=0.01,
    minimum_pr_auc_gain=0.02,
    minimum_positive_folds=4,
):
    affinity_labels = np.asarray(affinity_labels, dtype=np.float64)
    binary_labels = np.asarray(binary_labels, dtype=np.int8)
    base_affinity = affinity_metrics(affinity_labels, base_predictions)
    selected_affinity = affinity_metrics(affinity_labels, selected_predictions)
    base_pr = safe_average_precision(binary_labels, base_scores)
    selected_pr = safe_average_precision(binary_labels, selected_scores)
    diagnostics, positive_ci, positive_pr = validation_fold_diagnostics(
        sample_ids,
        affinity_labels,
        binary_labels,
        base_predictions,
        selected_predictions,
        base_scores,
        selected_scores,
    )
    ci_gain = float(selected_affinity["CI"] - base_affinity["CI"])
    pr_gain = float(selected_pr - base_pr)
    passed = (
        ci_gain >= float(minimum_ci_gain)
        and pr_gain >= float(minimum_pr_auc_gain)
        and positive_ci >= int(minimum_positive_folds)
        and positive_pr >= int(minimum_positive_folds)
    )
    return {
        "passed": bool(passed),
        "ci_gain": ci_gain,
        "pr_auc_gain": pr_gain,
        "positive_ci_folds": positive_ci,
        "positive_pr_auc_folds": positive_pr,
        "fold_diagnostics": diagnostics,
        "criteria": {
            "minimum_ci_gain": float(minimum_ci_gain),
            "minimum_pr_auc_gain": float(minimum_pr_auc_gain),
            "minimum_positive_folds": int(minimum_positive_folds),
        },
    }


def select_lp_blend(labels, v22_full, v22_base, v20_predictions, v20_available):
    labels = np.asarray(labels, dtype=np.float64)
    v22_full = np.asarray(v22_full, dtype=np.float64)
    v22_base = np.asarray(v22_base, dtype=np.float64)
    v20_predictions = np.asarray(v20_predictions, dtype=np.float64)
    available = np.asarray(v20_available, dtype=bool)
    anchor = v22_base.copy()
    anchor[available] = v20_predictions[available]
    candidates = []
    best = None
    for v20_weight in BLEND_WEIGHTS:
        prediction = v22_base.copy()
        prediction[available] = (
            v20_weight * v20_predictions[available]
            + (1.0 - v20_weight) * v22_full[available]
        )
        metrics = affinity_metrics(labels, prediction)
        candidate = {"v20_weight": float(v20_weight), "metrics": metrics}
        candidates.append(candidate)
        key = (metrics["CI"], -metrics["RMSE"], metrics["Spearman"])
        if best is None or key > best[0]:
            best = (key, candidate, prediction)
    anchor_metrics = affinity_metrics(labels, anchor)
    selected_metrics = best[1]["metrics"]
    return (
        dict(best[1]),
        np.asarray(best[2], dtype=np.float64),
        candidates,
        anchor_metrics,
    )


def apply_lp_selection(selection, v22_full, v22_base, v20_predictions, v20_available):
    prediction = np.asarray(v22_base, dtype=np.float64).copy()
    available = np.asarray(v20_available, dtype=bool)
    weight = float(selection["v20_weight"])
    prediction[available] = (
        weight * np.asarray(v20_predictions, dtype=np.float64)[available]
        + (1.0 - weight) * np.asarray(v22_full, dtype=np.float64)[available]
    )
    return prediction


def lp_validation_gate(anchor_metrics, selected_metrics, maximum_rmse_increase=0.003):
    ci_gain = float(selected_metrics["CI"] - anchor_metrics["CI"])
    rmse_increase = float(selected_metrics["RMSE"] - anchor_metrics["RMSE"])
    passed = ci_gain >= -1e-12 and rmse_increase <= float(maximum_rmse_increase)
    return {
        "passed": bool(passed),
        "ci_gain_vs_v20_route": ci_gain,
        "rmse_increase_vs_v20_route": rmse_increase,
        "criteria": {
            "minimum_ci_gain_vs_v20_route": 0.0,
            "maximum_rmse_increase": float(maximum_rmse_increase),
        },
    }


def dataset_indices(rows, dataset):
    return np.asarray(
        [index for index, row in enumerate(rows) if row["dataset"] == dataset],
        dtype=np.int64,
    )


def metrics_by_dataset(rows, predictions):
    predictions = np.asarray(predictions, dtype=np.float64)
    result = {}
    for dataset in DATASET_INDEX:
        indices = dataset_indices(rows, dataset)
        if indices.size == 0:
            continue
        labels = np.asarray([rows[index]["label"] for index in indices], dtype=np.float64)
        result[dataset] = affinity_metrics(labels, predictions[indices])
    return result


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_joblib(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".joblib.tmp")
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def group_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    return {name: int(counts[name]) for name in DATASET_INDEX}


__all__ = [
    "BINARY_THRESHOLDS",
    "BLEND_WEIGHTS",
    "COLLABORATIVE_DATASETS",
    "DATASET_INDEX",
    "RANKS",
    "apply_affinity_selection",
    "apply_binary_selection",
    "apply_lp_selection",
    "atomic_joblib",
    "atomic_json",
    "atomic_npz",
    "collaborative_predictions",
    "collaborative_validation_gate",
    "dataset_indices",
    "fit_decontaminated_bias_svd",
    "group_counts",
    "load_v20_lp_predictions",
    "load_v25_anchor_predictions",
    "load_v22_predictions",
    "lp_validation_gate",
    "metrics_by_dataset",
    "pair_key",
    "read_manifest",
    "rows_for_dataset",
    "safe_average_precision",
    "safe_roc_auc",
    "select_affinity_blend",
    "select_binary_blend",
    "select_lp_blend",
    "sha256_file",
    "validation_fold_diagnostics",
]
