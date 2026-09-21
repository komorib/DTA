#!/usr/bin/env python
"""Rank/raw ensemble for Boltz-DTA prediction files.

Each input run directory must contain:
  valid_pred.npy  shape [N,2], columns [prediction, target]
  test_pred.npy   shape [N,2], columns [prediction, target]

Weights are selected only on validation metrics, then applied once to test.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn import metrics as skm


FULL_METRICS = ["mse", "rmse", "mae", "pearson", "spearman", "ci", "r2"]


def eval_ci(y_true, y_pred, chunk_size=4096):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n_pairs = n_correct = n_tied = 0
    for start in range(0, len(y_true), chunk_size):
        end = start + chunk_size
        td = y_true[start:end, None] - y_true[None, :]
        pd = y_pred[start:end, None] - y_pred[None, :]
        mask = td > 0
        if not np.any(mask):
            continue
        d = pd[mask]
        n_pairs += int(d.size)
        n_correct += int(np.sum(d > 0))
        n_tied += int(np.sum(d == 0))
    return (n_correct + 0.5 * n_tied) / n_pairs if n_pairs else float("nan")


def compute_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    yp = np.nan_to_num(yp, nan=0.0, posinf=0.0, neginf=0.0)
    out = {
        "mse": float(skm.mean_squared_error(yt, yp)),
        "rmse": float(math.sqrt(skm.mean_squared_error(yt, yp))),
        "mae": float(skm.mean_absolute_error(yt, yp)),
        "pearson": float(stats.pearsonr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
        "spearman": float(stats.spearmanr(yt, yp)[0]) if np.std(yp) > 0 else float("nan"),
        "ci": float(eval_ci(yt, yp)),
        "r2": float(skm.r2_score(yt, yp)),
        "n": int(len(yt)),
    }
    return out


def rank_normalize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = stats.rankdata(values, method="average").astype(np.float64)
    ranks = ranks - ranks.mean()
    std = ranks.std()
    if std <= 0:
        return np.zeros_like(ranks)
    return ranks / std


def standardize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    std = values.std()
    if std <= 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def simplex_weights(n, step=0.1):
    if n < 1:
        raise ValueError("n must be positive")
    units = int(round(1.0 / float(step)))
    if units <= 0 or abs(units * float(step) - 1.0) > 1e-8:
        raise ValueError("step must divide 1.0 exactly, e.g. 0.5, 0.25, 0.1, 0.05")
    weights = []
    current = []

    def rec(pos, remaining):
        if pos == n - 1:
            weights.append(tuple((current + [remaining])[i] / units for i in range(n)))
            return
        for value in range(remaining + 1):
            current.append(value)
            rec(pos + 1, remaining - value)
            current.pop()

    rec(0, units)
    return weights


def load_pred_file(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Prediction file must have shape [N,2+]: {path}, got {arr.shape}")
    return np.asarray(arr[:, 0], dtype=np.float64), np.asarray(arr[:, 1], dtype=np.float64)


def load_runs(run_dirs):
    runs = []
    valid_true = None
    test_true = None
    for run_dir in run_dirs:
        path = Path(run_dir)
        valid_pred, vtrue = load_pred_file(path / "valid_pred.npy")
        test_pred, ttrue = load_pred_file(path / "test_pred.npy")
        if valid_true is None:
            valid_true = vtrue
            test_true = ttrue
        else:
            if len(vtrue) != len(valid_true) or not np.allclose(vtrue, valid_true):
                raise ValueError(f"Validation targets/order differ for run: {path}")
            if len(ttrue) != len(test_true) or not np.allclose(ttrue, test_true):
                raise ValueError(f"Test targets/order differ for run: {path}")
        runs.append({"name": path.name, "dir": str(path), "valid_pred": valid_pred, "test_pred": test_pred})
    if not runs:
        raise ValueError("At least one run directory is required")
    return {"runs": runs, "valid_true": valid_true, "test_true": test_true}


def prediction_matrix(runs, split, mode):
    cols = []
    for run in runs:
        values = run[f"{split}_pred"]
        if mode == "rank":
            values = rank_normalize(values)
        elif mode == "standardized":
            values = standardize(values)
        elif mode == "raw":
            values = np.asarray(values, dtype=np.float64)
        else:
            raise ValueError(f"unknown prediction mode: {mode}")
        cols.append(values)
    return np.stack(cols, axis=1)


def combine(matrix, weights):
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return matrix @ w


def metric_value(metrics, metric):
    value = metrics[metric]
    if metric in {"mse", "rmse", "mae"}:
        return -value
    return value


def optimize_weights(bundle, step=0.1, prediction_mode="rank", metric="ci"):
    runs = bundle["runs"]
    valid_matrix = prediction_matrix(runs, "valid", prediction_mode)
    test_matrix = prediction_matrix(runs, "test", prediction_mode)
    valid_true = bundle["valid_true"]
    test_true = bundle["test_true"]

    individual = []
    for i, run in enumerate(runs):
        individual.append({
            "name": run["name"],
            "dir": run["dir"],
            "valid": compute_metrics(valid_true, valid_matrix[:, i]),
            "test": compute_metrics(test_true, test_matrix[:, i]),
        })

    best = None
    for weights in simplex_weights(len(runs), step=step):
        valid_pred = combine(valid_matrix, weights)
        valid_metrics = compute_metrics(valid_true, valid_pred)
        score = metric_value(valid_metrics, metric)
        rmse = valid_metrics["rmse"]
        if best is None or score > best["score"] or (abs(score - best["score"]) <= 1e-12 and rmse < best["valid"]["rmse"]):
            test_pred = combine(test_matrix, weights)
            best = {
                "weights": [float(x) for x in weights],
                "score": float(score),
                "valid": valid_metrics,
                "test": compute_metrics(test_true, test_pred),
            }

    return {
        "selection_metric": metric,
        "prediction_mode": prediction_mode,
        "grid_step": float(step),
        "run_names": [run["name"] for run in runs],
        "individual": individual,
        **best,
    }


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Boltz-DTA valid/test prediction files")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing valid_pred.npy and test_pred.npy; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--metric", default="ci", choices=FULL_METRICS)
    parser.add_argument("--prediction-mode", default="rank", choices=["rank", "standardized", "raw"])
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_runs(args.run_dir)
    result = optimize_weights(bundle, step=args.grid_step, prediction_mode=args.prediction_mode, metric=args.metric)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(result), handle, indent=2, ensure_ascii=False)
    print(json.dumps(finite_json(result), indent=2, ensure_ascii=False))
    print(f"metrics: {out_dir / 'ensemble_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
