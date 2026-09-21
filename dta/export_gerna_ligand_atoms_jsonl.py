#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.lmdb_dataset import resolve_manifest_path  # noqa: E402


def load_manifest_paths(manifest_dir, splits):
    paths = [Path(manifest_dir) / f"{split}_manifest.json" for split in splits]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manifest files not found: {missing}")
    return paths


def collect_samples(manifest_paths):
    samples = []
    seen = set()
    lmdb_paths = set()
    split_counts = {}
    for path in manifest_paths:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset_type") != "pdbbind_lmdb":
            raise ValueError(f"not a PDBbind LMDB manifest: {path}")
        manifest_dir = os.path.dirname(os.path.abspath(path))
        lmdb_paths.add(os.path.abspath(resolve_manifest_path(manifest_dir, manifest["lmdb_path"])))
        split_name = manifest.get("split") or path.name.replace("_manifest.json", "")
        split_counts[split_name] = len(manifest.get("samples", []))
        for sample in manifest.get("samples", []):
            key = sample["lmdb_key"]
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
    if len(lmdb_paths) != 1:
        raise ValueError(f"all manifests must use the same feature LMDB, got {sorted(lmdb_paths)}")
    return samples, sorted(lmdb_paths)[0], split_counts


def ligand_atoms_and_coords(ligand_features, remove_hs=True):
    if "mol" not in ligand_features:
        raise ValueError("ligand_features is missing RDKit mol")
    mol = ligand_features["mol"]
    if mol.GetNumConformers() == 0:
        raise ValueError("ligand mol has no conformer")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(atoms) != len(coords):
        raise ValueError(f"atom/coord length mismatch: atoms={len(atoms)} coords={len(coords)}")
    if remove_hs:
        keep = [idx for idx, atom in enumerate(atoms) if atom != "H"]
        atoms = [atoms[idx] for idx in keep]
        coords = coords[keep]
    if not atoms:
        raise ValueError("ligand has no usable atoms after hydrogen filtering")
    return atoms, coords


def export(args):
    manifest_paths = load_manifest_paths(args.manifest_dir, args.splits)
    samples, feature_lmdb_path, split_counts = collect_samples(manifest_paths)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    env = lmdb.open(feature_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True, max_readers=256)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "manifest_dir": os.path.abspath(args.manifest_dir),
        "feature_lmdb": os.path.abspath(feature_lmdb_path),
        "output_jsonl": os.path.abspath(args.output_jsonl),
        "split_counts": split_counts,
        "candidate_samples": len(samples),
        "written": 0,
        "feature_missing": 0,
        "parse_errors": 0,
        "remove_hs": bool(args.remove_hs),
        "errors_preview": [],
    }

    try:
        with env.begin(write=False) as txn, open(args.output_jsonl, "w", encoding="utf-8") as out:
            for sample in samples:
                key = sample["lmdb_key"]
                raw = txn.get(key.encode("utf-8"))
                if raw is None:
                    stats["feature_missing"] += 1
                    if len(stats["errors_preview"]) < 20:
                        stats["errors_preview"].append({"key": key, "error": "feature LMDB key not found"})
                    continue
                try:
                    obj = pickle.loads(raw)
                    atoms, coords = ligand_atoms_and_coords(obj["ligand_features"], remove_hs=bool(args.remove_hs))
                    record = {
                        "lmdb_key": key,
                        "pdb_id": sample.get("pdb_id"),
                        "atoms": atoms,
                        "coordinates": coords.astype(float).tolist(),
                    }
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    stats["written"] += 1
                except Exception as exc:
                    stats["parse_errors"] += 1
                    if len(stats["errors_preview"]) < 20:
                        stats["errors_preview"].append({"key": key, "error": str(exc)})
    finally:
        env.close()

    if args.metadata_output:
        Path(args.metadata_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata_output).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Export GerNA LMDB ligand atoms/coordinates in GerNA atom order.")
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--metadata_output", default="")
    parser.set_defaults(remove_hs=True)
    parser.add_argument("--remove_hs", dest="remove_hs", action="store_true")
    parser.add_argument("--no-remove_hs", dest="remove_hs", action="store_false")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main():
    export(parse_args())


if __name__ == "__main__":
    main()
