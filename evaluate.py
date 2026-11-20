"""Evaluate saved validation predictions without reading test data."""

import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr


def ci(labels, predictions):
    total = concordant = tied = 0.0
    for i in range(len(labels)):
        delta = labels[i + 1:] - labels[i]
        pred = predictions[i + 1:] - predictions[i]
        valid = delta != 0
        total += float(valid.sum())
        concordant += float(((pred[valid] * delta[valid]) > 0).sum())
        tied += float((pred[valid] == 0).sum())
    return (concordant + 0.5 * tied) / total


def metrics(path):
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.float64)
        predictions = np.asarray(data["predictions"], dtype=np.float64)
    return {
        "CI": float(ci(labels, predictions)),
        "RMSE": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "MAE": float(np.mean(np.abs(predictions - labels))),
        "Pearson": float(pearsonr(labels, predictions).statistic),
        "Spearman": float(spearmanr(labels, predictions).statistic),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_npz")
    print(metrics(parser.parse_args().predictions_npz))
