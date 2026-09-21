#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kdbnet.dta import create_fold
from kdbnet.pdbbind import parse_pdbbind_pocket_pdb


def overlap_report(split, col):
    tr = set(split["train"][col].astype(str))
    va = set(split["valid"][col].astype(str))
    te = set(split["test"][col].astype(str))
    return {
        "column": col,
        "train_test_overlap": len(tr & te),
        "valid_test_overlap": len(va & te),
        "train_valid_overlap": len(tr & va),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit PDBbind split overlaps.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_pocket_seq", action="store_true")
    args = parser.parse_args()

    df = pd.read_table(args.manifest)
    split = create_fold(df, args.seed, [0.7, 0.1, 0.2])
    print("rows", {name: len(frame) for name, frame in split.items()})
    print("total", len(df))
    print("unique_complex_id", df["complex_id"].nunique())

    for col in ["complex_id", "drug", "protein", "ligand_sdf", "pocket_pdb"]:
        rep = overlap_report(split, col)
        print(
            f"{col} train_test_overlap={rep['train_test_overlap']} "
            f"valid_test_overlap={rep['valid_test_overlap']} "
            f"train_valid_overlap={rep['train_valid_overlap']}"
        )

    if not args.check_pocket_seq:
        return

    seq_cache = {}
    for _, row in df.iterrows():
        seq_cache[str(row["complex_id"])] = parse_pdbbind_pocket_pdb(row["pocket_pdb"])["seq"]
    for name in split:
        split[name] = split[name].copy()
        split[name]["pocket_seq"] = split[name]["complex_id"].astype(str).map(seq_cache)

    tr_seq = set(split["train"]["pocket_seq"])
    va_seq = set(split["valid"]["pocket_seq"])
    te_seq = set(split["test"]["pocket_seq"])
    print("unique_pocket_seq", {name: frame["pocket_seq"].nunique() for name, frame in split.items()})
    print("pocket_seq_train_test_overlap", len(tr_seq & te_seq))
    print("pocket_seq_valid_test_overlap", len(va_seq & te_seq))
    print("pocket_seq_train_valid_overlap", len(tr_seq & va_seq))
    print(
        "test_rows_with_exact_train_pocket_seq",
        f"{int(split['test']['pocket_seq'].isin(tr_seq).sum())}/{len(split['test'])}",
    )
    print(
        "test_rows_with_exact_train_or_valid_pocket_seq",
        f"{int(split['test']['pocket_seq'].isin(tr_seq | va_seq).sum())}/{len(split['test'])}",
    )


if __name__ == "__main__":
    main()
