#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def parse_frac(value):
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split fraction must have three comma-separated values")
    if any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("split fraction values must be non-negative")
    total = sum(parts)
    if abs(total - 1.0) > 1e-8:
        raise argparse.ArgumentTypeError("split fraction values must sum to 1")
    return parts


def create_fixed_split(df, seed, valid_seed, frac):
    train_frac, valid_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=seed)
    train_valid = df[~df.index.isin(test.index)]
    valid = train_valid.sample(
        frac=valid_frac / (train_frac + valid_frac),
        replace=False,
        random_state=valid_seed,
    )
    train = train_valid[~train_valid.index.isin(valid.index)]

    split = pd.concat(
        [
            pd.DataFrame({"complex_id": train["complex_id"].astype(str), "split": "train"}),
            pd.DataFrame({"complex_id": valid["complex_id"].astype(str), "split": "valid"}),
            pd.DataFrame({"complex_id": test["complex_id"].astype(str), "split": "test"}),
        ],
        ignore_index=True,
    )
    return split


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed train/valid/test split TSV for PDBbind PL."
    )
    parser.add_argument(
        "--manifest",
        default="data/PDBbind_full/processed/pdbbind_pl_manifest.tsv",
        help="PDBbind PL manifest TSV.",
    )
    parser.add_argument(
        "--output",
        default="data/PDBbind_full/processed/pdbbind_pl_fixed_split_seed42.tsv",
        help="Output TSV with columns: complex_id, split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the fixed split.")
    parser.add_argument(
        "--valid_seed",
        type=int,
        default=1,
        help="Random seed for validation sampling after test rows are removed.",
    )
    parser.add_argument(
        "--frac",
        type=parse_frac,
        default=parse_frac("0.7,0.1,0.2"),
        help="Train,valid,test fractions as comma-separated values.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    df = pd.read_table(manifest_path)
    if "complex_id" not in df.columns:
        raise ValueError("manifest must contain a complex_id column")
    if df["complex_id"].duplicated().any():
        duplicated = df.loc[df["complex_id"].duplicated(), "complex_id"].astype(str).head().tolist()
        raise ValueError("manifest contains duplicated complex_id values: {}".format(duplicated))

    split = create_fixed_split(df, args.seed, args.valid_seed, args.frac)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, sep="\t", index=False)

    counts = split["split"].value_counts().reindex(["train", "valid", "test"]).to_dict()
    print("wrote {}".format(output_path))
    print("rows {}".format(len(split)))
    print("counts {}".format(counts))


if __name__ == "__main__":
    main()
