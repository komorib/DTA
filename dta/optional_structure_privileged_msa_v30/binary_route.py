"""Frozen V26 binary-route alignment and reporting for V30."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import DATASET_INDEX
from .selection import load_aligned_predictions


AFFINITY_THRESHOLDS = {"DAVIS": 7.522878745280337, "KIBA": 12.1}


def load_aligned_binary_route(path, sample_ids, labels, dataset_index):
    """Load frozen V26 classifications under the affinity lineage checks."""
    load_aligned_predictions(path, sample_ids, labels, dataset_index)
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "binary_scores", "binary_predictions"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("binary route archive is missing: %s" % sorted(missing))
        source_ids = archive["sample_ids"].astype(str)
        positions = {value: index for index, value in enumerate(source_ids.tolist())}
        order = np.asarray([positions[str(value)] for value in sample_ids], dtype=np.int64)
        scores = np.asarray(archive["binary_scores"], dtype=np.float64)[order]
        predictions = np.asarray(archive["binary_predictions"], dtype=np.int8)[order]
    routed = np.asarray(dataset_index, dtype=np.int64) != DATASET_INDEX["LP_PDBBind"]
    if not np.isfinite(scores[routed]).all():
        raise ValueError("binary route scores are non-finite for Davis/KIBA")
    if not np.isin(predictions[routed], (0, 1)).all():
        raise ValueError("binary route predictions are invalid for Davis/KIBA")
    return {"scores": scores, "predictions": predictions}


def frozen_binary_metrics(labels, dataset_index, scores, predictions):
    """Measure frozen V26 classifications without thresholding V30 affinity."""
    labels = np.asarray(labels, dtype=np.float64)
    dataset_index = np.asarray(dataset_index, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int8)
    result = {}
    for name, threshold in AFFINITY_THRESHOLDS.items():
        selected = dataset_index == DATASET_INDEX[name]
        target = (labels[selected] >= float(threshold)).astype(np.int8)
        prediction = predictions[selected]
        score = scores[selected]
        true_positive = int(((target == 1) & (prediction == 1)).sum())
        true_negative = int(((target == 0) & (prediction == 0)).sum())
        false_positive = int(((target == 0) & (prediction == 1)).sum())
        false_negative = int(((target == 1) & (prediction == 0)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        denominator = float(
            np.sqrt(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            )
        )
        mcc = (
            (true_positive * true_negative - false_positive * false_negative) / denominator
            if denominator > 0.0
            else 0.0
        )
        both_classes = np.unique(target).size == 2
        result[name] = {
            "ACC": float((prediction == target).mean()),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "MCC": float(mcc),
            "ROC-AUC": float(roc_auc_score(target, score)) if both_classes else float("nan"),
            "PR-AUC": float(average_precision_score(target, score)) if both_classes else float("nan"),
            "affinity_threshold": float(threshold),
        }
    return result


def affinity_only_metrics(metrics):
    keys = ("CI", "RMSE", "MAE", "Pearson", "Spearman")
    return {
        name: {key: float(values[key]) for key in keys}
        for name, values in metrics.items()
    }


__all__ = [
    "AFFINITY_THRESHOLDS",
    "affinity_only_metrics",
    "frozen_binary_metrics",
    "load_aligned_binary_route",
]
