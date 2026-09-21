#!/usr/bin/env python3
"""Prepare input CSVs for supplementing HCLBind PDBbind samples missing in GerNA LMDB."""

import argparse
import csv
import json
import os
from collections import Counter

import lmdb


def load_lmdb_keys(lmdb_path):
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


def load_path_index(path_csv, pdbbind_root):
    by_id = {}
    with open(path_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_path = row.get("pdb_path") or row.get("pdb")
            pocket_path = row.get("pocket_path")
            ligand_path = row.get("ligand_path") or row.get("ligand")
            if not pdb_path or not ligand_path:
                continue
            pdb_id = os.path.basename(pdb_path).split("_", 1)[0].lower()
            by_id[pdb_id] = {
                "pdb_path": os.path.abspath(os.path.join(pdbbind_root, pdb_path)),
                "pocket_path": os.path.abspath(os.path.join(pdbbind_root, pocket_path)) if pocket_path else "",
                "ligand_path": os.path.abspath(os.path.join(pdbbind_root, ligand_path)),
            }
    return by_id


def fallback_paths(pdb_id, pdbbind_pl_root):
    candidates = []
    for era in sorted(os.listdir(pdbbind_pl_root)):
        candidate_dir = os.path.join(pdbbind_pl_root, era, pdb_id)
        if os.path.isdir(candidate_dir):
            candidates.append(candidate_dir)
    if not candidates:
        return None
    sample_dir = candidates[0]
    return {
        "pdb_path": os.path.abspath(os.path.join(sample_dir, f"{pdb_id}_protein.pdb")),
        "pocket_path": os.path.abspath(os.path.join(sample_dir, f"{pdb_id}_pocket.pdb")),
        "ligand_path": os.path.abspath(os.path.join(sample_dir, f"{pdb_id}_ligand.sdf")),
    }


def files_exist(paths):
    required = ("pdb_path", "pocket_path", "ligand_path")
    return all(paths.get(name) and os.path.exists(paths[name]) for name in required)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hclbind-csv", default="external/HCLBind/data/pdbbind/pdbbind_ft.csv")
    parser.add_argument("--existing-lmdb", default="data/PDBbind_chain_lmdb_esm_gernabind_v2")
    parser.add_argument("--path-csv", default="data/pdbbind_protein_pocket_ligand_affinity.csv")
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--pdbbind-pl-root", default="PDBbind/P-L")
    parser.add_argument("--output-dir", default="data/HCLBindPDBbindFT_missing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing_keys = load_lmdb_keys(args.existing_lmdb)
    path_index = load_path_index(args.path_csv, args.pdbbind_root)

    missing_rows = []
    input_rows = []
    unavailable_rows = []

    with open(args.hclbind_csv, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val", "test"}:
                continue
            pdb_id = row["Assembly_ID"].strip().lower()
            lmdb_key = f"{pdb_id}_protein"
            if lmdb_key in existing_keys:
                continue

            row_with_key = dict(row)
            row_with_key["lmdb_key"] = lmdb_key
            missing_rows.append(row_with_key)

            paths = path_index.get(pdb_id) or fallback_paths(pdb_id, args.pdbbind_pl_root)
            if not paths or not files_exist(paths):
                unavailable = dict(row_with_key)
                if paths:
                    unavailable.update(paths)
                unavailable_rows.append(unavailable)
                continue

            input_rows.append(
                {
                    "pdb_path": paths["pdb_path"],
                    "pocket_path": paths["pocket_path"],
                    "ligand_path": paths["ligand_path"],
                    "affinity": row["affinity"],
                    "lmdb_key": lmdb_key,
                    "split": row["split"],
                    "Assembly_ID": row["Assembly_ID"],
                }
            )

    missing_rows_path = os.path.join(args.output_dir, "missing_rows.csv")
    missing_keys_path = os.path.join(args.output_dir, "missing_keys.txt")
    input_csv_path = os.path.join(args.output_dir, "missing_lmdb_input.csv")
    unavailable_path = os.path.join(args.output_dir, "missing_unavailable.csv")
    summary_path = os.path.join(args.output_dir, "prepare_summary.json")

    with open(missing_rows_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(missing_rows[0].keys()) if missing_rows else ["lmdb_key"])
        writer.writeheader()
        writer.writerows(missing_rows)

    with open(missing_keys_path, "w", encoding="utf-8") as handle:
        for row in missing_rows:
            handle.write(row["lmdb_key"] + "\n")

    with open(input_csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pdb_path", "pocket_path", "ligand_path", "affinity", "lmdb_key", "split", "Assembly_ID"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(input_rows)

    with open(unavailable_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in unavailable_rows for key in row}) if unavailable_rows else ["lmdb_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unavailable_rows)

    summary = {
        "hclbind_csv": os.path.abspath(args.hclbind_csv),
        "existing_lmdb": os.path.abspath(args.existing_lmdb),
        "path_csv": os.path.abspath(args.path_csv),
        "pdbbind_root": os.path.abspath(args.pdbbind_root),
        "missing_total": len(missing_rows),
        "missing_by_split": dict(Counter(row["split"] for row in missing_rows)),
        "available_for_lmdb": len(input_rows),
        "unavailable": len(unavailable_rows),
        "outputs": {
            "missing_rows": os.path.abspath(missing_rows_path),
            "missing_keys": os.path.abspath(missing_keys_path),
            "missing_lmdb_input": os.path.abspath(input_csv_path),
            "missing_unavailable": os.path.abspath(unavailable_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
