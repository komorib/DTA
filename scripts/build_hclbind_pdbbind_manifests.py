#!/usr/bin/env python3
"""Build GerNA LMDB manifests from the HCLBind PDBbind finetune CSV."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import lmdb


def lmdb_keys(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
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


def build_manifests(args):
    keys = lmdb_keys(args.lmdb_path)
    split_map = {"train": "train", "val": "valid", "test": "test"}
    samples_by_split = defaultdict(list)
    missing_by_split = Counter()
    original_by_split = Counter()
    unknown_count = 0
    missing_examples = []

    with open(args.hclbind_csv, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split == "unknown":
                unknown_count += 1
                continue
            if split not in split_map:
                continue
            original_by_split[split] += 1
            sample = make_sample(row)
            if sample["lmdb_key"] not in keys:
                missing_by_split[split] += 1
                if len(missing_examples) < args.max_missing_examples:
                    missing_examples.append(sample["lmdb_key"])
                continue
            samples_by_split[split_map[split]].append(sample)

    os.makedirs(args.output_dir, exist_ok=True)
    rel_lmdb_path = os.path.relpath(
        os.path.abspath(args.lmdb_path),
        os.path.abspath(args.output_dir),
    )
    summary = {
        "hclbind_csv": os.path.abspath(args.hclbind_csv),
        "lmdb_path": os.path.abspath(args.lmdb_path),
        "output_dir": os.path.abspath(args.output_dir),
        "split_method": args.split_method,
        "protein_3d_mode": args.protein_3d_mode,
        "unknown_rows_ignored": unknown_count,
        "original_counts": dict(original_by_split),
        "missing_counts": dict(missing_by_split),
        "kept_counts": {split: len(samples) for split, samples in samples_by_split.items()},
        "missing_examples": missing_examples,
    }

    for split in ("train", "valid", "test"):
        manifest = {
            "dataset_type": "pdbbind_lmdb",
            "task": "dta",
            "split": split,
            "split_method": args.split_method,
            "lmdb_path": rel_lmdb_path,
            "structure_lmdb_path": rel_lmdb_path,
            "protein_3d_mode": args.protein_3d_mode,
            "source": "HCLBind data/pdbbind/pdbbind_ft.csv",
            "source_csv": os.path.abspath(args.hclbind_csv),
            "samples": samples_by_split[split],
        }
        path = os.path.join(args.output_dir, f"{split}_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    with open(os.path.join(args.output_dir, "build_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hclbind-csv", default="external/HCLBind/data/pdbbind/pdbbind_ft.csv")
    parser.add_argument("--lmdb-path", default="data/PDBbind_chain_lmdb_esm_gernabind_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-method", default="hclbind_pdbbind_ft")
    parser.add_argument("--protein-3d-mode", default="residue")
    parser.add_argument("--max-missing-examples", type=int, default=30)
    args = parser.parse_args()
    build_manifests(args)


if __name__ == "__main__":
    main()
