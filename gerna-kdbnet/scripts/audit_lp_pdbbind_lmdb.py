#!/usr/bin/env python
"""Audit a KDBNet LP-PDBbind LMDB before training."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("Install py-lmdb: pip install lmdb") from exc
    return lmdb


def main():
    parser = argparse.ArgumentParser(description="Audit LP-PDBbind LMDB schema and split integrity")
    parser.add_argument("--lmdb_dir", required=True)
    parser.add_argument("--index_path")
    parser.add_argument("--embedding_key", default="esm2_embedding")
    parser.add_argument("--embedding_dim", type=int, default=1280)
    parser.add_argument("--check_paths", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    index_path = Path(args.index_path) if args.index_path else lmdb_dir / "index.pkl"
    with open(index_path, "rb") as handle:
        rows = pickle.load(handle)
    df = pd.DataFrame(rows)
    errors = []
    warnings = []
    required = {"lmdb_key", "complex_id", "affinity", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Index missing columns: {}".format(sorted(missing)))

    duplicated = df[df["complex_id"].duplicated()]["complex_id"].tolist()
    if duplicated:
        errors.append({"reason": "duplicate_complex_id", "examples": duplicated[:20]})
    overlap = df.groupby("complex_id")["split"].nunique()
    if (overlap > 1).any():
        errors.append({"reason": "split_overlap", "examples": overlap[overlap > 1].index[:20].tolist()})
    unknown = sorted(set(df["split"]).difference({"train", "valid", "test"}))
    if unknown:
        errors.append({"reason": "unknown_split", "examples": unknown})

    lmdb = import_lmdb()
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=128)
    with env.begin(write=False) as txn:
        for row in tqdm(rows, desc="audit LMDB"):
            cid = str(row["complex_id"])
            raw = txn.get(str(row["lmdb_key"]).encode())
            if raw is None:
                errors.append({"complex_id": cid, "reason": "missing_lmdb_key"})
                continue
            try:
                entry = pickle.loads(raw)
                protein = entry["protein_features"]
                if not isinstance(protein, dict) or not protein:
                    raise ValueError("empty protein_features")
                for chain, chain_data in protein.items():
                    seq = chain_data["sequence"]
                    coords = np.asarray(chain_data["backbone_coords"])
                    emb = np.asarray(chain_data[args.embedding_key])
                    if coords.shape != (len(seq), 4, 3):
                        raise ValueError("coords shape {} != ({},4,3)".format(coords.shape, len(seq)))
                    if emb.shape != (len(seq), args.embedding_dim):
                        raise ValueError("embedding shape {} != ({},{})".format(emb.shape, len(seq), args.embedding_dim))
                ligand_features = entry.get("ligand_features", {})
                if "mol" in ligand_features:
                    mol = ligand_features["mol"]
                    if mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
                        raise ValueError("invalid stored ligand mol")
                metadata = entry.get("metadata", {})
                if metadata.get("complex_id", cid) != cid:
                    raise ValueError("metadata complex_id mismatch")
                if args.check_paths:
                    for key in ("protein_path", "pocket_path", "ligand_path"):
                        value = metadata.get(key)
                        if value and not Path(value).exists():
                            warnings.append({"complex_id": cid, "reason": "missing_path", "field": key, "path": value})
            except Exception as exc:
                errors.append({"complex_id": cid, "reason": type(exc).__name__, "message": str(exc)})
    env.close()

    report = {
        "lmdb_dir": str(lmdb_dir.resolve()),
        "index_path": str(index_path.resolve()),
        "rows": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_reasons": dict(Counter(item["reason"] for item in errors)),
        "warning_reasons": dict(Counter(item["reason"] for item in warnings)),
        "error_examples": errors[:100],
        "warning_examples": warnings[:100],
        "passed": len(errors) == 0,
    }
    report_path = Path(args.report) if args.report else lmdb_dir / "audit_report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({key: report[key] for key in ("rows", "split_counts", "errors", "warnings", "passed")}, indent=2))
    print("report={}".format(report_path))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
