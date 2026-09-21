#!/usr/bin/env python
"""Merge LP benchmark prediction shards and compute metrics."""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LP benchmark prediction shards")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="external_benchmark")
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    permissible = 0.0
    n = len(y_true)
    for i in range(n):
        diff_true = y_true[i] - y_true[i + 1 :]
        diff_pred = y_pred[i] - y_pred[i + 1 :]
        mask = diff_true != 0
        if not np.any(mask):
            continue
        prod = diff_true[mask] * diff_pred[mask]
        concordant += float(np.sum(prod > 0))
        concordant += 0.5 * float(np.sum(prod == 0))
        permissible += float(np.sum(mask))
    return concordant / permissible if permissible else float("nan")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    out = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    err = y_pred - y_true
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        try:
            from scipy.stats import spearmanr

            out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            out["spearman"] = float("nan")
        out["ci"] = float(concordance_index(y_true, y_pred))
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
        out["ci"] = float("nan")
    return out


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["source_file"] = file
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True)
    pred = pred.drop_duplicates(subset=["pdb_id"], keep="first").sort_values("pdb_id").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "method": args.method,
        "split": args.split,
        "shard_files": len(files),
        "predicted_rows": int(len(pred)),
        "overall": compute_metrics(pred["y_true"], pred["y_pred"]),
    }
    novelty_col = "valid_novelty_code_vs_train" if args.split == "val" else "test_novelty_code_vs_train"
    if novelty_col in pred.columns:
        metrics["by_novelty"] = {}
        for code, group in pred.groupby(novelty_col, dropna=True):
            metrics["by_novelty"][str(int(code))] = compute_metrics(group["y_true"], group["y_pred"])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(metrics), handle, indent=2, ensure_ascii=False)

    print(json.dumps(finite_json(metrics), indent=2, ensure_ascii=False))
    print(f"predictions: {out_dir / 'predictions.csv'}")
    print(f"metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
