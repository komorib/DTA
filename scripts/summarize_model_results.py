#!/usr/bin/env python3
"""Summarize GerNA experiment best_metrics.json files."""

import argparse
import json
import os


def metric(metrics, split, name):
    value = metrics.get(split, {}).get(name)
    return float(value) if value is not None else None


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Model")
    parser.add_argument("--sort", choices=["test_CI", "valid_CI", "test_Pearson", "valid_Pearson"], default="test_CI")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = []
    for root, _, files in os.walk(args.root):
        if "best_metrics.json" not in files:
            continue
        path = os.path.join(root, "best_metrics.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        row = {
            "path": path,
            "epoch": metrics.get("epoch"),
            "dataset": metrics.get("dataset"),
            "split_method": metrics.get("split_method"),
            "test_CI": metric(metrics, "test", "CI"),
            "valid_CI": metric(metrics, "valid", "CI"),
            "test_Pearson": metric(metrics, "test", "Pearson"),
            "valid_Pearson": metric(metrics, "valid", "Pearson"),
            "test_RMSE": metric(metrics, "test", "RMSE"),
            "valid_RMSE": metric(metrics, "valid", "RMSE"),
        }
        rows.append(row)

    rows = [row for row in rows if row[args.sort] is not None]
    rows.sort(key=lambda row: row[args.sort], reverse=True)
    for row in rows[: args.limit]:
        print(
            f"{args.sort}={fmt(row[args.sort])} "
            f"test_CI={fmt(row['test_CI'])} valid_CI={fmt(row['valid_CI'])} "
            f"test_P={fmt(row['test_Pearson'])} valid_P={fmt(row['valid_Pearson'])} "
            f"test_RMSE={fmt(row['test_RMSE'])} valid_RMSE={fmt(row['valid_RMSE'])} "
            f"epoch={row['epoch']} path={row['path']}"
        )


if __name__ == "__main__":
    main()
