#!/usr/bin/env python
"""Create a ligand Uni-Mol2 embedding LMDB for PDBbind fixed-split samples.

This script is intended to run in a Uni-Mol2 environment, not necessarily in
the KDBNet training environment. It reads the same SDF files used by KDBNet's
ligand graph featurizer and stores atom-level Uni-Mol2 representations keyed by
complex_id.
"""

import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def _import_py_lmdb():
    """Import py-lmdb even when this repo has a local lmdb/ directory."""
    module = sys.modules.get("lmdb")
    if module is not None and hasattr(module, "open"):
        return module

    repo_root = Path(__file__).resolve().parents[1]
    old_path = list(sys.path)
    old_module = sys.modules.pop("lmdb", None)
    module = None
    try:
        filtered_path = []
        for entry in old_path:
            try:
                resolved = Path(entry or ".").resolve()
            except OSError:
                filtered_path.append(entry)
                continue
            if resolved == repo_root or resolved == repo_root / "lmdb":
                continue
            filtered_path.append(entry)
        sys.path = filtered_path
        module = importlib.import_module("lmdb")
    finally:
        sys.path = old_path
        if old_module is not None and (module is None or not hasattr(module, "open")):
            sys.modules["lmdb"] = old_module

    if not hasattr(module, "open"):
        raise ImportError(
            "Could not import py-lmdb. Install the external package with "
            "`pip install lmdb` in the Uni-Mol2 environment."
        )
    return module


def _infer_complex_id(index_row, key_suffix="_protein"):
    if "complex_id" in index_row and pd.notna(index_row["complex_id"]):
        return str(index_row["complex_id"]).lower()
    lmdb_key = str(index_row["lmdb_key"]).lower()
    for suffix in (key_suffix, "_protein", "_pocket"):
        if suffix and lmdb_key.endswith(suffix):
            return lmdb_key[: -len(suffix)]
    return lmdb_key


def _read_split_ids(split_file):
    split_df = pd.read_table(split_file)
    required = {"complex_id", "split"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError("Split file missing columns: {}".format(sorted(missing)))

    split_df = split_df[["complex_id", "split"]].copy()
    split_df["complex_id"] = split_df["complex_id"].astype(str).str.lower()
    split_df["split"] = split_df["split"].astype(str).str.lower()
    duplicated = split_df[split_df["complex_id"].duplicated()]["complex_id"]
    if len(duplicated) > 0:
        raise ValueError(
            "Split file has duplicated complex_id values: {}".format(
                duplicated.head().tolist()
            )
        )
    valid_splits = {"train", "valid", "test"}
    unknown = sorted(set(split_df["split"]).difference(valid_splits))
    if unknown:
        raise ValueError("Split file has unknown split labels: {}".format(unknown))
    return split_df


def _read_manifest(manifest_path):
    manifest = pd.read_table(manifest_path)
    required = {"complex_id", "ligand_sdf"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError("Manifest missing columns: {}".format(sorted(missing)))
    manifest = manifest[["complex_id", "ligand_sdf"]].copy()
    manifest["complex_id"] = manifest["complex_id"].astype(str).str.lower()
    manifest["ligand_sdf"] = manifest["ligand_sdf"].astype(str)
    duplicated = manifest[manifest["complex_id"].duplicated()]["complex_id"]
    if len(duplicated) > 0:
        raise ValueError(
            "Manifest has duplicated complex_id values: {}".format(
                duplicated.head().tolist()
            )
        )
    return manifest


def _read_protein_lmdb_index(index_path, key_suffix="_protein"):
    with open(index_path, "rb") as handle:
        rows = pickle.load(handle)
    df = pd.DataFrame(rows)
    if "lmdb_key" not in df.columns:
        raise ValueError("Protein LMDB index missing column: lmdb_key")
    df["complex_id"] = [
        _infer_complex_id(row, key_suffix=key_suffix)
        for row in df.to_dict("records")
    ]
    df["complex_id"] = df["complex_id"].astype(str).str.lower()
    keep_cols = ["complex_id"]
    if "ligand_path" in df.columns:
        keep_cols.append("ligand_path")
    df = df[keep_cols].copy()
    if "ligand_path" in df.columns:
        df = df.rename(columns={"ligand_path": "protein_lmdb_ligand_path"})
    duplicated = df[df["complex_id"].duplicated()]["complex_id"]
    if len(duplicated) > 0:
        raise ValueError(
            "Protein LMDB index has duplicated complex_id values: {}".format(
                duplicated.head().tolist()
            )
        )
    return df


def _resolve_ligand_sdf(complex_id, path_value=None, structure_root=None):
    candidates = []
    if path_value and str(path_value).lower() != "nan":
        path = Path(str(path_value))
        candidates.append(path)
        if structure_root is not None:
            root = Path(structure_root)
            candidates.append(root / path)
            if len(path.parts) >= 2:
                complex_dir = path.parts[-2]
                filename = path.name
                candidates.extend([
                    root / complex_dir / filename,
                    root / "P-L" / complex_dir / filename,
                ])

    if structure_root is not None:
        root = Path(structure_root)
        filename = "{}_ligand.sdf".format(complex_id)
        candidates.extend([
            root / complex_id / filename,
            root / "P-L" / complex_id / filename,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve ligand SDF for {} from {}".format(complex_id, path_value)
    )


def _load_sdf_for_unimol(sdf_path):
    # Match KDBNet mol_graph.featurize_drug(), which calls MolFromMolFile
    # without overriding RDKit's default hydrogen handling.
    mol = Chem.MolFromMolFile(str(sdf_path))
    if mol is None:
        raise ValueError("RDKit failed to parse SDF")
    if mol.GetNumConformers() == 0:
        raise ValueError("SDF has no conformer")

    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.asarray(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    if len(atoms) != coords.shape[0]:
        raise ValueError("Atom/coordinate count mismatch")
    return atoms, coords


def _prepare_rows(split_df, manifest, structure_root, protein_lmdb_index=None, limit=None):
    df = split_df.copy()
    if protein_lmdb_index is not None:
        protein_df = _read_protein_lmdb_index(protein_lmdb_index)
        before = len(df)
        df = df.merge(protein_df, on="complex_id", how="inner", validate="one_to_one")
        print("filtered_by_protein_lmdb: {} -> {}".format(before, len(df)))

    df = df.merge(manifest, on="complex_id", how="left", validate="one_to_one")
    if limit is not None:
        df = df.head(int(limit)).copy()

    rows, skipped = [], []
    for row in tqdm(df.to_dict("records"), desc="resolve_sdf"):
        complex_id = row["complex_id"]
        ligand_sdf = row.get("protein_lmdb_ligand_path") or row.get("ligand_sdf")
        try:
            sdf_path = _resolve_ligand_sdf(
                complex_id,
                path_value=ligand_sdf,
                structure_root=structure_root,
            )
            atoms, coords = _load_sdf_for_unimol(sdf_path)
            rows.append({
                "complex_id": complex_id,
                "split": row["split"],
                "ligand_sdf": str(sdf_path),
                "atoms": atoms,
                "coordinates": coords,
            })
        except Exception as exc:
            skipped.append({
                "complex_id": complex_id,
                "ligand_sdf": ligand_sdf,
                "reason": type(exc).__name__,
                "message": str(exc),
            })
    return rows, skipped


def _existing_record(txn, key):
    raw = txn.get(key.encode())
    if raw is None:
        return None
    return pickle.loads(raw)


def _index_from_record(record):
    return {
        "complex_id": record["complex_id"],
        "lmdb_key": record["complex_id"],
        "split": record["split"],
        "ligand_sdf": record["ligand_sdf"],
        "num_atoms": int(record["num_atoms"]),
        "embedding_dim": int(record["embedding_dim"]),
        "model_name": record["model_name"],
        "model_size": record["model_size"],
    }


def _embed_batch(repr_model, batch_rows):
    data = {
        "atoms": [row["atoms"] for row in batch_rows],
        "coordinates": [row["coordinates"].tolist() for row in batch_rows],
    }
    return repr_model.get_repr(data, return_atomic_reprs=True)


def _write_embedding_batch(txn, rows, repr_out, args):
    index_rows = []
    for i, row in enumerate(rows):
        atom_emb = np.asarray(repr_out["atomic_reprs"][i], dtype=np.float32)
        cls_emb = np.asarray(repr_out["cls_repr"][i], dtype=np.float32)
        atoms = list(row["atoms"])
        output_symbols = repr_out.get("atomic_symbol")
        if output_symbols is not None:
            output_atoms = [str(symbol) for symbol in output_symbols[i]]
        else:
            output_atoms = atoms
        coords = np.asarray(row["coordinates"], dtype=np.float32)
        if atom_emb.ndim != 2:
            raise ValueError("atomic_reprs for {} is not 2D".format(row["complex_id"]))
        if atom_emb.shape[0] != len(atoms):
            raise ValueError(
                "Atom embedding length mismatch for {}: atoms {}, embedding {}".format(
                    row["complex_id"], len(atoms), tuple(atom_emb.shape)
                )
            )
        if cls_emb.ndim != 1:
            raise ValueError("cls_repr for {} is not 1D".format(row["complex_id"]))
        if output_atoms != atoms:
            raise ValueError(
                "Uni-Mol2 atom symbols/order mismatch for {}: input {} vs output {}".format(
                    row["complex_id"], atoms[:10], output_atoms[:10]
                )
            )

        record = {
            "complex_id": row["complex_id"],
            "split": row["split"],
            "ligand_sdf": row["ligand_sdf"],
            "atoms": atoms,
            "unimol_atomic_symbol": output_atoms,
            "coordinates": coords,
            "atom_emb": atom_emb,
            "cls_emb": cls_emb,
            "num_atoms": int(len(atoms)),
            "embedding_dim": int(atom_emb.shape[1]),
            "model_name": args.model_name,
            "model_size": args.model_size,
            "remove_hs": bool(args.remove_hs),
            "source": "unimol_tools.UniMolRepr",
        }
        txn.put(row["complex_id"].encode(), pickle.dumps(record), overwrite=True)
        index_rows.append(_index_from_record(record))
    return index_rows


def _generate_embeddings(rows, args):
    if args.weight_dir:
        os.environ["UNIMOL_WEIGHT_DIR"] = str(Path(args.weight_dir))

    from unimol_tools import UniMolRepr

    repr_model = UniMolRepr(
        data_type="molecule",
        model_name=args.model_name,
        model_size=args.model_size,
        remove_hs=args.remove_hs,
        batch_size=args.batch_size,
    )

    lmdb = _import_py_lmdb()
    args.output_lmdb.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(args.output_lmdb),
        map_size=int(args.map_size_gb * 1024 ** 3),
        subdir=True,
        lock=True,
        readahead=False,
        max_readers=128,
    )

    index_rows, skipped, existing = [], [], 0
    with env.begin(write=True) as txn:
        pending = []
        for row in rows:
            key = row["complex_id"]
            old_record = _existing_record(txn, key)
            if old_record is not None and not args.overwrite:
                existing += 1
                index_rows.append(_index_from_record(old_record))
                continue
            pending.append(row)

        for start in tqdm(range(0, len(pending), args.batch_size), desc="unimol2"):
            batch_rows = pending[start:start + args.batch_size]
            try:
                repr_out = _embed_batch(repr_model, batch_rows)
                index_rows.extend(_write_embedding_batch(txn, batch_rows, repr_out, args))
            except Exception as batch_exc:
                if len(batch_rows) == 1:
                    row = batch_rows[0]
                    skipped.append({
                        "complex_id": row["complex_id"],
                        "ligand_sdf": row["ligand_sdf"],
                        "reason": type(batch_exc).__name__,
                        "message": str(batch_exc),
                    })
                    continue
                for row in batch_rows:
                    try:
                        repr_out = _embed_batch(repr_model, [row])
                        index_rows.extend(_write_embedding_batch(txn, [row], repr_out, args))
                    except Exception as exc:
                        skipped.append({
                            "complex_id": row["complex_id"],
                            "ligand_sdf": row["ligand_sdf"],
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        })

    env.sync()
    env.close()
    return index_rows, skipped, existing


def main():
    parser = argparse.ArgumentParser(
        description="Generate a ligand Uni-Mol2 embedding LMDB for PDBbind."
    )
    parser.add_argument(
        "--split_file",
        default="data/PDBbind_full/processed/pdbbind_pl_fixed_split_seed42.tsv",
        help="Fixed split TSV with complex_id and split columns.",
    )
    parser.add_argument(
        "--manifest",
        default="data/PDBbind_full/processed/pdbbind_pl_manifest.tsv",
        help="PDBbind manifest TSV with ligand_sdf paths.",
    )
    parser.add_argument(
        "--structure_root",
        default="data/PDBbind_full/extracted",
        help="Fallback root for resolving ligand SDF paths.",
    )
    parser.add_argument(
        "--protein_lmdb_index",
        help="Optional protein LMDB index.pkl; if set, only process split IDs in this LMDB.",
    )
    parser.add_argument(
        "--output_lmdb",
        type=Path,
        default=Path("data/PDBbind_ligand_unimol2_84m_lmdb"),
        help="Output LMDB directory.",
    )
    parser.add_argument(
        "--weight_dir",
        help="Path assigned to UNIMOL_WEIGHT_DIR before loading Uni-Mol2.",
    )
    parser.add_argument("--model_name", default="unimolv2")
    parser.add_argument("--model_size", default="84m")
    parser.add_argument(
        "--remove_hs",
        action="store_true",
        help="Ask Uni-Mol tools to remove hydrogens. Default keeps the RDKit-loaded atom list unchanged.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Process only the first N resolved rows.")
    parser.add_argument("--map_size_gb", type=float, default=64.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LMDB records.")
    args = parser.parse_args()

    split_df = _read_split_ids(args.split_file)
    manifest = _read_manifest(args.manifest)
    rows, pre_skipped = _prepare_rows(
        split_df,
        manifest,
        structure_root=args.structure_root,
        protein_lmdb_index=args.protein_lmdb_index,
        limit=args.limit,
    )
    print("resolved_rows={}".format(len(rows)))
    print("preprocess_skipped={}".format(len(pre_skipped)))

    index_rows, embed_skipped, existing = _generate_embeddings(rows, args)
    index_rows = sorted(index_rows, key=lambda item: item["complex_id"])

    index_path = args.output_lmdb / "index.pkl"
    with open(index_path, "wb") as handle:
        pickle.dump(index_rows, handle)

    report = {
        "split_file": str(args.split_file),
        "manifest": str(args.manifest),
        "structure_root": str(args.structure_root),
        "protein_lmdb_index": str(args.protein_lmdb_index) if args.protein_lmdb_index else None,
        "output_lmdb": str(args.output_lmdb),
        "weight_dir": str(args.weight_dir) if args.weight_dir else None,
        "model_name": args.model_name,
        "model_size": args.model_size,
        "remove_hs": bool(args.remove_hs),
        "batch_size": args.batch_size,
        "resolved_rows": len(rows),
        "existing_records": existing,
        "written_or_existing": len(index_rows),
        "preprocess_skipped": len(pre_skipped),
        "embedding_skipped": len(embed_skipped),
        "skipped_examples": (pre_skipped + embed_skipped)[:100],
    }
    report_path = args.output_lmdb / "unimol2_state.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("wrote_lmdb={}".format(args.output_lmdb))
    print("wrote_index={}".format(index_path))
    print("wrote_report={}".format(report_path))
    print("written_or_existing={}/{}".format(len(index_rows), len(rows)))
    print("skipped={}".format(len(pre_skipped) + len(embed_skipped)))


if __name__ == "__main__":
    main()
