from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


def binary_metrics(labels, logits) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "ACC": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "PR-AUC": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "classification_threshold": 0.5,
    }

