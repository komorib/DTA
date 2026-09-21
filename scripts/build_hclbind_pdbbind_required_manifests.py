#!/usr/bin/env python3
"""Build HCLBind PDBbind manifests requiring keys in feature and fusion LMDBs."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import lmdb


def lmdb_keys(path):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    with env.begin(write=False) as txn:
        return {
            key.decode("utf-8")
            for key in txn.cursor().iternext(keys=True, values=False)
            if key != b"__metadata__"
        }


def make_sample(row):
    pdb_id = row["Assembly_ID"].strip().lower()
    return {
        "lmdb_key": f"{pdb_id}_protein",
        "label": float(row["affinity"]),
        "pdb_id": pdb_id,
        "hclbind_type": row.get("type", ""),
        "hclbind_contacting_domains": int(row.get("contacting_domains") or 0),
        "hclbind_total_domains": int(row.get("total_domains") or 0),
        "hclbind_split": row["split"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hclbind-csv", default="external/HCLBind/data/pdbbind/pdbbind_ft.csv")
    parser.add_argument("--lmdb-path", required=True)
    parser.add_argument("--required-lmdb", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-method", required=True)
    parser.add_argument("--protein-3d-mode", default="residue")
    args = parser.parse_args()

    feature_keys = lmdb_keys(args.lmdb_path)
    required_keys = set(feature_keys)
    for path in args.required_lmdb:
        required_keys &= lmdb_keys(path)

    split_map = {"train": "train", "val": "valid", "test": "test"}
    samples_by_split = defaultdict(list)
    original = Counter()
    missing_feature = Counter()
    missing_required = Counter()
    ignored_unknown = 0
    missing_feature_examples = []
    missing_required_examples = []

    with open(args.hclbind_csv, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split == "unknown":
                ignored_unknown += 1
                continue
            if split not in split_map:
                continue
            original[split] += 1
            sample = make_sample(row)
            key = sample["lmdb_key"]
            if key not in feature_keys:
                missing_feature[split] += 1
                if len(missing_feature_examples) < 30:
                    missing_feature_examples.append(key)
                continue
            if key not in required_keys:
                missing_required[split] += 1
                if len(missing_required_examples) < 30:
                    missing_required_examples.append(key)
                continue
            samples_by_split[split_map[split]].append(sample)

    os.makedirs(args.output_dir, exist_ok=True)
    rel_lmdb = os.path.relpath(os.path.abspath(args.lmdb_path), os.path.abspath(args.output_dir))
    for split in ("train", "valid", "test"):
        manifest = {
            "dataset_type": "pdbbind_lmdb",
            "task": "dta",
            "split": split,
            "split_method": args.split_method,
            "lmdb_path": rel_lmdb,
            "structure_lmdb_path": rel_lmdb,
            "protein_3d_mode": args.protein_3d_mode,
            "source": "HCLBind data/pdbbind/pdbbind_ft.csv",
            "source_csv": os.path.abspath(args.hclbind_csv),
            "required_lmdb": [os.path.abspath(path) for path in args.required_lmdb],
            "samples": samples_by_split[split],
        }
        with open(os.path.join(args.output_dir, f"{split}_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    summary = {
        "hclbind_csv": os.path.abspath(args.hclbind_csv),
        "lmdb_path": os.path.abspath(args.lmdb_path),
        "required_lmdb": [os.path.abspath(path) for path in args.required_lmdb],
        "output_dir": os.path.abspath(args.output_dir),
        "split_method": args.split_method,
        "ignored_unknown": ignored_unknown,
        "original_counts": dict(original),
        "missing_feature_counts": dict(missing_feature),
        "missing_required_counts": dict(missing_required),
        "kept_counts": {split: len(samples_by_split[split]) for split in ("train", "valid", "test")},
        "missing_feature_examples": missing_feature_examples,
        "missing_required_examples": missing_required_examples,
    }
    with open(os.path.join(args.output_dir, "build_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
