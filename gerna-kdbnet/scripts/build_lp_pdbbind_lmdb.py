#!/usr/bin/env python
"""Build the KDBNet-compatible LMDB directly from LP-PDBbind.

The script accepts flexible LP split artifacts while enforcing strict base
complex disjointness. It writes:

  <output_lmdb>/data.mdb, lock.mdb
  <output_lmdb>/index.pkl
  <output_lmdb>/index.tsv
  <output_lmdb>/split.tsv
  <output_lmdb>/schema.json
  <output_lmdb>/build_report.json

Use ``--dry_run`` to validate paths, splits, labels, structures and embeddings
without requiring py-lmdb or writing the database.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.lp_pdbbind_lmdb import (
    ESMEmbedder,
    build_complex_record,
    discover_structure_files,
    load_affinity_table,
    load_lp_split,
    merge_split_and_affinity,
    serialisable_report,
)


def import_lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "py-lmdb is required to write the database. Install it with "
            "`pip install lmdb`, or run first with --dry_run."
        ) from exc
    return lmdb


def put_with_resize(env, txn, key: bytes, value: bytes, max_map_size: int):
    """Put one item, doubling map size on MapFullError."""
    lmdb = import_lmdb()
    while True:
        try:
            txn.put(key, value, overwrite=True)
            return txn
        except lmdb.MapFullError:
            txn.abort()
            current = env.info()["map_size"]
            new_size = min(current * 2, max_map_size)
            if new_size <= current:
                raise RuntimeError(
                    "LMDB map is full at {:.1f} GiB; increase --max_map_size_gb".format(
                        current / 2**30
                    )
                )
            env.set_mapsize(new_size)
            txn = env.begin(write=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LP-PDBbind structures, labels and splits into KDBNet LMDB"
    )
    parser.add_argument("--structure_root", required=True,
                        help="Root recursively containing <PDBID>_protein.pdb, _pocket.pdb and ligand files.")
    parser.add_argument("--split_source", required=True,
                        help="LP split table/pickle/json, or directory with train/valid/test ID files.")
    parser.add_argument("--affinity_source", required=True,
                        help="PDBbind INDEX_general_PL*.lst or LP table containing affinity/pK labels.")
    parser.add_argument("--output_lmdb", required=True,
                        help="Output LMDB directory.")
    parser.add_argument("--numeric_affinity_is", default="pk", choices=["pk", "molar", "auto"],
                        help="Interpret numeric labels in non-standard affinity tables.")
    parser.add_argument("--protein_source", default="auto", choices=["auto", "protein", "pocket"],
                        help="Use full protein PDB when available (recommended), or pocket-only PDB.")
    parser.add_argument("--path_mode", default="absolute", choices=["absolute", "relative"],
                        help="How structure paths are recorded in LMDB metadata.")

    parser.add_argument("--esm_mode", default="zero", choices=["zero", "precomputed", "generate"],
                        help="Embedding source. Use generate for final experiments; zero is for pipeline checks.")
    parser.add_argument("--embedding_dir",
                        help="Directory with <complex_id>.pt/.npy embeddings for --esm_mode precomputed.")
    parser.add_argument("--embedding_key", default="esm2_embedding")
    parser.add_argument("--embedding_dim", type=int, default=1280)
    parser.add_argument("--embedding_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--esm_model_name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm_model_path", help="Local fair-esm checkpoint, avoiding network download.")
    parser.add_argument("--esm_repr_layer", type=int, default=33)
    parser.add_argument("--esm_device", default="cuda")
    parser.add_argument("--esm_max_tokens", type=int, default=1022)
    parser.add_argument("--esm_overlap", type=int, default=128)
    parser.add_argument("--esm_cache_dir", help="Sequence-hash cache for generated embeddings.")

    parser.add_argument("--normalise_ligands", action="store_true",
                        help="Write RDKit-normalised SDFs under output_lmdb/structures and reference them.")
    parser.add_argument("--no_store_ligand_mol", action="store_true",
                        help="Do not store the RDKit Mol; runtime must use --ligand_source sdf.")
    parser.add_argument("--commit_every", type=int, default=128)
    parser.add_argument("--map_size_gb", type=float, default=32.0)
    parser.add_argument("--max_map_size_gb", type=float, default=512.0)
    parser.add_argument("--limit", type=int, help="Process first N accepted split/affinity rows for testing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate and build records in memory without writing LMDB.")
    parser.add_argument("--fail_fast", action="store_true",
                        help="Abort on first bad complex instead of reporting/skipping it.")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    start_time = time.time()
    structure_root = Path(args.structure_root).resolve()
    output_lmdb = Path(args.output_lmdb).resolve()

    if output_lmdb.exists() and args.overwrite:
        shutil.rmtree(output_lmdb)
    if output_lmdb.exists() and any(output_lmdb.iterdir()):
        raise FileExistsError(
            "Output directory is not empty: {}. Use --overwrite.".format(output_lmdb)
        )
    output_lmdb.mkdir(parents=True, exist_ok=True)

    split_df = load_lp_split(args.split_source)
    affinity_df = load_affinity_table(args.affinity_source, args.numeric_affinity_is)
    merged = merge_split_and_affinity(split_df, affinity_df)
    split_without_affinity = merged.attrs.get("split_without_affinity", [])
    affinity_without_split = merged.attrs.get("affinity_without_split", [])
    if args.limit is not None:
        merged = merged.head(int(args.limit)).copy()
    structures = discover_structure_files(structure_root)

    embedder = None
    if args.esm_mode == "generate":
        embedder = ESMEmbedder(
            model_name=args.esm_model_name,
            model_path=args.esm_model_path,
            repr_layer=args.esm_repr_layer,
            device=args.esm_device,
            max_tokens=args.esm_max_tokens,
            overlap=args.esm_overlap,
            cache_dir=args.esm_cache_dir or (output_lmdb / "esm_cache"),
        )
    embedding_dir = Path(args.embedding_dir).resolve() if args.embedding_dir else None
    if args.esm_mode == "precomputed" and embedding_dir is None:
        raise ValueError("--esm_mode precomputed requires --embedding_dir")

    normalised_ligand_dir = output_lmdb / "structures" if args.normalise_ligands else None
    index_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    split_counts = Counter()
    protein_source_counts = Counter()

    env = None
    txn = None
    lmdb = None
    max_map_size = int(args.max_map_size_gb * 2**30)
    if not args.dry_run:
        lmdb = import_lmdb()
        env = lmdb.open(
            str(output_lmdb),
            map_size=int(args.map_size_gb * 2**30),
            subdir=True,
            meminit=False,
            map_async=True,
            max_dbs=1,
        )
        txn = env.begin(write=True)

    for item_index, row in enumerate(tqdm(merged.to_dict("records"), desc="LP-PDBbind -> LMDB"), start=1):
        cid = str(row["complex_id"])
        files = structures.get(cid)
        if files is None:
            skipped.append({"complex_id": cid, "reason": "structure_directory_not_found"})
            if args.fail_fast:
                raise FileNotFoundError("No structures found for {}".format(cid))
            continue
        try:
            entry, index_row = build_complex_record(
                row=row,
                files=files,
                structure_root=structure_root,
                protein_source=args.protein_source,
                esm_mode=args.esm_mode,
                embedding_key=args.embedding_key,
                embedding_dim=args.embedding_dim,
                embedding_dtype=args.embedding_dtype,
                embedder=embedder,
                embedding_dir=embedding_dir,
                path_mode=args.path_mode,
                normalised_ligand_dir=normalised_ligand_dir,
                store_ligand_mol=not args.no_store_ligand_mol,
            )
            if not args.dry_run:
                raw = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
                txn = put_with_resize(
                    env, txn, index_row["lmdb_key"].encode("utf-8"), raw, max_map_size
                )
                if item_index % args.commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            index_rows.append(index_row)
            split_counts[index_row["split"]] += 1
            protein_source_counts[index_row["protein_source"]] += 1
        except Exception as exc:
            skipped.append({
                "complex_id": cid,
                "reason": type(exc).__name__,
                "message": str(exc),
            })
            if args.fail_fast:
                raise

    if not args.dry_run:
        # Add schema metadata keys; these keys are not listed in index.pkl and
        # therefore never appear as training samples.
        schema = {
            "schema_version": 2,
            "dataset": "LP-PDBbind",
            "entry_key": "<complex_id>_protein",
            "protein_features": "chain -> sequence/backbone_coords/embedding",
            "backbone_atom_order": ["N", "CA", "C", "O"],
            "embedding_key": args.embedding_key,
            "embedding_dim": args.embedding_dim,
            "embedding_dtype": args.embedding_dtype,
            "target": "pK = -log10(K[M])",
        }
        txn.put(b"__schema__", pickle.dumps(schema, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(b"__keys__", pickle.dumps([row["lmdb_key"] for row in index_rows]))
        txn.put(b"__len__", str(len(index_rows)).encode("ascii"))
        txn.commit()
        env.sync()
        env.close()
    else:
        schema = {
            "schema_version": 2,
            "dataset": "LP-PDBbind",
            "embedding_key": args.embedding_key,
            "embedding_dim": args.embedding_dim,
        }

    index_df = pd.DataFrame(index_rows)
    if not index_df.empty:
        # Re-audit split disjointness after all skips.
        split_nunique = index_df.groupby("complex_id")["split"].nunique()
        if (split_nunique > 1).any():
            raise RuntimeError("Internal error: accepted complex crosses split boundary")

    with open(output_lmdb / "index.pkl", "wb") as handle:
        pickle.dump(index_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    index_df.to_csv(output_lmdb / "index.tsv", sep="\t", index=False)
    split_columns = [col for col in ("complex_id", "split", "split_detail") if col in index_df.columns]
    index_df[split_columns].to_csv(output_lmdb / "split.tsv", sep="\t", index=False)
    with open(output_lmdb / "schema.json", "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)

    report = {
        "dry_run": bool(args.dry_run),
        "structure_root": str(structure_root),
        "split_source": str(Path(args.split_source).resolve()),
        "affinity_source": str(Path(args.affinity_source).resolve()),
        "output_lmdb": str(output_lmdb),
        "split_rows": int(len(split_df)),
        "affinity_rows": int(len(affinity_df)),
        "merged_rows": int(len(merged)),
        "accepted": int(len(index_rows)),
        "skipped": int(len(skipped)),
        "split_counts": dict(split_counts),
        "protein_source_counts": dict(protein_source_counts),
        "split_without_affinity": len(split_without_affinity),
        "split_without_affinity_examples": split_without_affinity[:50],
        "affinity_without_split": len(affinity_without_split),
        "affinity_without_split_examples": affinity_without_split[:50],
        "discovered_structure_ids": len(structures),
        "skip_reasons": dict(Counter(item["reason"] for item in skipped)),
        "skipped_examples": skipped[:100],
        "esm_mode": args.esm_mode,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "recommended_runtime": {
            "lmdb_dir": str(output_lmdb),
            "index_path": str(output_lmdb / "index.pkl"),
            "split_file": str(output_lmdb / "split.tsv"),
            "structure_root": str(structure_root),
            "ligand_source": "lmdb_mol" if not args.no_store_ligand_mol else "sdf",
            "esm_key": args.embedding_key,
            "prot_emb_dim": args.embedding_dim,
        },
    }
    with open(output_lmdb / "build_report.json", "w", encoding="utf-8") as handle:
        json.dump(serialisable_report(report), handle, indent=2, sort_keys=True)

    print("accepted={} skipped={}".format(len(index_rows), len(skipped)))
    print("split_counts={}".format(dict(split_counts)))
    print("index={}".format(output_lmdb / "index.pkl"))
    print("split={}".format(output_lmdb / "split.tsv"))
    print("report={}".format(output_lmdb / "build_report.json"))
    if args.dry_run:
        print("dry_run=True: LMDB data.mdb was not written")


if __name__ == "__main__":
    main()
