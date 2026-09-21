#!/usr/bin/env python
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

TEST_RE = re.compile(
    r"Ensemble model \| Test Loss: (?P<loss>[0-9.]+) \| "
    r"mse: (?P<mse>[0-9.]+) \| spearman: (?P<spearman>[-0-9.]+) \| pearson: (?P<pearson>[-0-9.]+)"
)


def parse_exp_log(run_dir):
    log_path = Path(run_dir) / "exp.log"
    if not log_path.exists():
        raise FileNotFoundError(f"missing exp.log: {log_path}")
    match = None
    for line in log_path.read_text(errors="replace").splitlines():
        found = TEST_RE.search(line)
        if found:
            match = found
    if match is None:
        raise ValueError(f"no Ensemble model test metrics found in {log_path}")
    row = {"run_dir": str(Path(run_dir))}
    row.update({k: float(v) for k, v in match.groupdict().items()})
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize KDBNet exp.log test metrics.")
    parser.add_argument("--glob", required=True, help="Glob matching run directories.")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(args.glob))
    if not run_dirs:
        raise SystemExit(f"no run directories matched: {args.glob}")

    rows = [parse_exp_log(run_dir) for run_dir in run_dirs]
    df = pd.DataFrame(rows)
    metric_cols = ["loss", "mse", "spearman", "pearson"]
    summary_df = pd.DataFrame([
        {"run_dir": "mean", **{metric: df[metric].mean() for metric in metric_cols}},
        {"run_dir": "std", **{metric: df[metric].std(ddof=1) for metric in metric_cols}},
    ])
    out_df = pd.concat([df, summary_df], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, sep="\t", index=False, float_format="%.6f")
    print(out_df.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
