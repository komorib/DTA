#!/usr/bin/env python3
"""Merge two LMDB feature stores and their index.pkl files without modifying inputs."""

import argparse
import json
import os
import pickle

import lmdb


def copy_value(value):
    return None if value is None else bytes(value)


def load_index(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def key_from_row(row):
    if isinstance(row, dict) and "lmdb_key" in row:
        return row["lmdb_key"]
    raise ValueError(f"cannot read lmdb_key from index row: {row!r}")


def merge(args):
    if os.path.exists(args.output_lmdb) and os.listdir(args.output_lmdb):
        raise RuntimeError(f"output LMDB already exists and is non-empty: {args.output_lmdb}")
    os.makedirs(args.output_lmdb, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_index) or ".", exist_ok=True)

    primary_env = lmdb.open(args.primary_lmdb, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    supplement_env = lmdb.open(args.supplement_lmdb, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    output_env = lmdb.open(args.output_lmdb, map_size=args.map_size_gb * 1024**3, subdir=True)

    written = 0
    duplicate_supplement = 0
    with output_env.begin(write=True) as out_txn:
        with primary_env.begin(write=False) as in_txn:
            for key, value in in_txn.cursor():
                out_txn.put(key, copy_value(value))
                written += 1
        with supplement_env.begin(write=False) as in_txn:
            for key, value in in_txn.cursor():
                if out_txn.get(key) is not None:
                    duplicate_supplement += 1
                    continue
                out_txn.put(key, copy_value(value))
                written += 1

    primary_index = load_index(args.primary_index)
    supplement_index = load_index(args.supplement_index)
    seen_keys = {key_from_row(row) for row in primary_index}
    merged_index = list(primary_index)
    added_index = 0
    for row in supplement_index:
        key = key_from_row(row)
        if key in seen_keys:
            continue
        merged_index.append(row)
        seen_keys.add(key)
        added_index += 1

    with open(args.output_index, "wb") as handle:
        pickle.dump(merged_index, handle)

    output_env.close()
    supplement_env.close()
    primary_env.close()

    summary = {
        "primary_lmdb": os.path.abspath(args.primary_lmdb),
        "supplement_lmdb": os.path.abspath(args.supplement_lmdb),
        "output_lmdb": os.path.abspath(args.output_lmdb),
        "written_lmdb_entries": written,
        "duplicate_supplement_lmdb_entries": duplicate_supplement,
        "primary_index_entries": len(primary_index),
        "supplement_index_entries": len(supplement_index),
        "merged_index_entries": len(merged_index),
        "added_index_entries": added_index,
        "output_index": os.path.abspath(args.output_index),
    }
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-lmdb", required=True)
    parser.add_argument("--primary-index", required=True)
    parser.add_argument("--supplement-lmdb", required=True)
    parser.add_argument("--supplement-index", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--map-size-gb", type=int, default=220)
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
