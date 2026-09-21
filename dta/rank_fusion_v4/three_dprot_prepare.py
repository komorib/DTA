"""Parallel, resumable preparation of official LP-PDBbind data for 3DProtDTA."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import pandas as pd


BUILDER_VERSION = "rank-fusion-v4-three-dprot-prepare-v1"
SPLIT_ORDER = ("train", "valid", "test")
RAW_SPLITS = {"train": "train", "valid": "val", "test": "test"}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _manifest_rows(path, expected_split):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("split") != expected_split:
        raise ValueError("manifest split mismatch for %s" % path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    rows = []
    seen = set()
    for sample in samples:
        key = sample.get("lmdb_key") if isinstance(sample, dict) else None
        if not isinstance(key, str) or not key.lower().endswith("_protein"):
            raise ValueError("invalid manifest lmdb_key: %r" % key)
        pdb_id = key[:-8].lower()
        if not pdb_id or pdb_id in seen:
            raise ValueError("duplicate or empty manifest PDB ID: %s" % pdb_id)
        try:
            label = float(sample["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid manifest label for %s" % key) from error
        if not math.isfinite(label):
            raise ValueError("nonfinite manifest label for %s" % key)
        sample_pdb_id = str(sample.get("pdb_id", pdb_id)).lower()
        if sample_pdb_id != pdb_id:
            raise ValueError("manifest key/PDB ID mismatch for %s" % key)
        seen.add(pdb_id)
        rows.append((pdb_id, label))
    return path, rows


def load_official_rows(raw_csv, manifest_paths, label_tolerance=1e-5):
    """Return official rows in train/valid/test manifest order."""
    raw_csv = Path(raw_csv).expanduser().resolve()
    raw = pd.read_csv(raw_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    required = {"pdb_id", "smiles", "value", "new_split", "date"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("raw LP CSV is missing columns: " + ", ".join(missing))
    raw = raw.dropna(subset=["pdb_id", "value", "new_split", "date"]).copy()
    raw["pdb_id"] = raw["pdb_id"].astype(str).str.lower()
    if raw["pdb_id"].duplicated().any():
        duplicates = raw.loc[raw["pdb_id"].duplicated(), "pdb_id"].head(5).tolist()
        raise ValueError("raw LP CSV has duplicate PDB IDs: %s" % duplicates)
    raw_by_pdb = raw.set_index("pdb_id", drop=False)

    paths = {name: Path(manifest_paths[name]) for name in SPLIT_ORDER}
    rows = []
    counts = {}
    all_pdb_ids = set()
    manifest_hashes = {}
    for split in SPLIT_ORDER:
        path, manifest_rows = _manifest_rows(paths[split], split)
        manifest_hashes[split] = _sha256_file(path)
        counts[split] = len(manifest_rows)
        for pdb_id, label in manifest_rows:
            if pdb_id in all_pdb_ids:
                raise ValueError("PDB ID appears in multiple manifests: %s" % pdb_id)
            if pdb_id not in raw_by_pdb.index:
                raise ValueError("official PDB ID is absent from raw CSV: %s" % pdb_id)
            raw_row = raw_by_pdb.loc[pdb_id]
            expected_raw_split = RAW_SPLITS[split]
            if str(raw_row["new_split"]) != expected_raw_split:
                raise ValueError("split mismatch for %s" % pdb_id)
            raw_label = float(raw_row["value"])
            if not math.isfinite(raw_label) or abs(raw_label - label) > label_tolerance:
                raise ValueError(
                    "label mismatch for %s: manifest=%r raw=%r"
                    % (pdb_id, label, raw_label)
                )
            raw_smiles = raw_row["smiles"]
            if pd.isna(raw_smiles) or not str(raw_smiles).strip():
                raw_smiles = None
            else:
                raw_smiles = str(raw_smiles)
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "smiles": raw_smiles,
                    "label": label,
                    "split": split,
                    "date": str(raw_row["date"]),
                }
            )
            all_pdb_ids.add(pdb_id)

    provenance = {
        "builder_version": BUILDER_VERSION,
        "raw_csv": str(raw_csv),
        "raw_csv_sha256": _sha256_file(raw_csv),
        "manifest_paths": {name: str(paths[name].resolve()) for name in SPLIT_ORDER},
        "manifest_sha256": manifest_hashes,
        "counts": counts,
        "rows": len(rows),
        "missing_raw_smiles": sum(row["smiles"] is None for row in rows),
    }
    return rows, provenance


def _row_fingerprint(rows):
    normalized = [
        [row["pdb_id"], row["smiles"], float(row["label"]), row["split"], row["date"]]
        for row in rows
    ]
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_reusable_shard(path, rows, signature):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (EOFError, OSError, pickle.PickleError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return None
    if payload.get("signature") != signature:
        return None
    if payload.get("row_fingerprint") != _row_fingerprint(rows):
        return None
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(rows):
        return None
    if [record.get("pdb_id") for record in records] != [row["pdb_id"] for row in rows]:
        return None
    return payload


def build_or_reuse_shard(shard_path, rows, signature, build_record):
    """Build one atomic shard, or reuse it when input provenance is identical."""
    shard_path = Path(shard_path)
    reusable = _load_reusable_shard(shard_path, rows, signature)
    if reusable is not None:
        return {
            "path": str(shard_path),
            "count": len(reusable["records"]),
            "reused": True,
        }
    records = [build_record(row) for row in rows]
    payload = {
        "status": "complete",
        "builder_version": BUILDER_VERSION,
        "signature": signature,
        "row_fingerprint": _row_fingerprint(rows),
        "records": records,
    }
    _atomic_pickle(shard_path, payload)
    return {"path": str(shard_path), "count": len(records), "reused": False}


def _validate_record(record, expected, label_tolerance=1e-5):
    if not isinstance(record, dict):
        raise ValueError("shard record must be a mapping")
    pdb_id = str(record.get("pdb_id", "")).lower()
    if pdb_id != expected["pdb_id"]:
        raise ValueError("shard order/key mismatch for %s" % expected["pdb_id"])
    if record.get("split") != expected["split"]:
        raise ValueError("shard split mismatch for %s" % pdb_id)
    try:
        label = float(record["label"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid shard label for %s" % pdb_id) from error
    if not math.isfinite(label) or abs(label - float(expected["label"])) > label_tolerance:
        raise ValueError("label mismatch for %s" % pdb_id)
    ligand = record.get("ligand")
    if not isinstance(ligand, str) or not ligand:
        raise ValueError("invalid canonical ligand for %s" % pdb_id)
    for name in ("ligand_graph", "ligand_ecfp", "protein_graph"):
        if record.get(name) is None:
            raise ValueError("missing %s for %s" % (name, pdb_id))


def merge_prepared_shards(shard_paths, rows, output_dir, provenance):
    """Validate all shards and atomically commit 3DProt-compatible files."""
    output_dir = Path(output_dir).expanduser().resolve()
    summary_path = output_dir / "prepare_summary.json"
    try:
        summary_path.unlink()
    except FileNotFoundError:
        pass

    records = []
    for path in shard_paths:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("status") != "complete":
            raise ValueError("incomplete shard: %s" % path)
        records.extend(payload.get("records", []))
    if len(records) != len(rows):
        raise ValueError("shard row count mismatch: %d != %d" % (len(records), len(rows)))

    ligand_to_graph = {}
    ligand_to_ecfp = {}
    protein_to_graph = {}
    full_rows = []
    for record, expected in zip(records, rows):
        _validate_record(record, expected)
        pdb_id = expected["pdb_id"]
        if pdb_id in protein_to_graph:
            raise ValueError("duplicate protein graph for %s" % pdb_id)
        ligand = record["ligand"]
        ligand_to_graph.setdefault(ligand, record["ligand_graph"])
        ligand_to_ecfp.setdefault(ligand, record["ligand_ecfp"])
        protein_to_graph[pdb_id] = record["protein_graph"]
        full_rows.append(
            {"ligand": ligand, "protein": pdb_id, "label": float(expected["label"])}
        )

    expected_pdb_ids = [row["pdb_id"] for row in rows]
    if set(protein_to_graph) != set(expected_pdb_ids):
        raise ValueError("protein graph coverage does not match official rows")
    output_dir = Path(output_dir).expanduser().resolve()
    folds_dir = output_dir / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "full.csv"
    full_tmp = Path(str(full_path) + ".tmp")
    pd.DataFrame(full_rows, columns=["ligand", "protein", "label"]).to_csv(
        full_tmp, index=False
    )
    os.replace(str(full_tmp), str(full_path))
    valid_indices = [index for index, row in enumerate(rows) if row["split"] == "valid"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    _atomic_json(folds_dir / "train_fold_setting1.txt", [valid_indices])
    _atomic_json(folds_dir / "test_fold_setting1.txt", test_indices)
    _atomic_pickle(output_dir / "ligand_to_graph.pkl", ligand_to_graph)
    _atomic_pickle(output_dir / "ligand_to_ecfp.pkl", ligand_to_ecfp)
    _atomic_pickle(output_dir / "protein_to_graph.pkl", protein_to_graph)

    counts = {split: sum(row["split"] == split for row in rows) for split in SPLIT_ORDER}
    summary = {
        "status": "complete",
        "builder_version": BUILDER_VERSION,
        "output": str(output_dir),
        "rows": len(rows),
        "counts": counts,
        "ligands": len(ligand_to_graph),
        "proteins": len(protein_to_graph),
        "provenance": provenance,
    }
    _atomic_json(output_dir / "prepare_summary.json", summary)
    return summary


def resolve_ligand_smiles(row, protein_path):
    """Use raw SMILES when valid, otherwise recover it from ligand SDF."""
    from rdkit import Chem

    raw_smiles = row.get("smiles")
    if raw_smiles is not None:
        molecule = Chem.MolFromSmiles(str(raw_smiles))
        if molecule is not None:
            return Chem.MolToSmiles(molecule, isomericSmiles=True), "raw_csv"
    protein_path = Path(protein_path)
    suffix = "_protein.pdb"
    if not protein_path.name.endswith(suffix):
        raise ValueError("cannot derive ligand SDF from %s" % protein_path)
    ligand_path = protein_path.with_name(
        protein_path.name[: -len(suffix)] + "_ligand.sdf"
    )
    if not ligand_path.is_file():
        raise FileNotFoundError(str(ligand_path))
    molecule = Chem.MolFromMolFile(str(ligand_path), sanitize=True, removeHs=True)
    if molecule is None:
        raise ValueError("ligand SDF parsing failed for %s" % row["pdb_id"])
    return Chem.MolToSmiles(molecule, isomericSmiles=True), "ligand_sdf"


def _build_record(row, pdbbind_root, protein_cutoff):
    from scripts.train_3dprot_dta_lp import (
        build_ligand_graph,
        build_morgan_ecfp,
        build_protein_graph_from_pdb,
        pdbbind_protein_path,
    )

    protein_path = pdbbind_protein_path(
        pdbbind_root, row["pdb_id"], row["date"]
    )
    if not protein_path.is_file():
        raise FileNotFoundError(str(protein_path))
    ligand, ligand_source = resolve_ligand_smiles(row, protein_path)
    return {
        "pdb_id": row["pdb_id"],
        "ligand": ligand,
        "label": float(row["label"]),
        "split": row["split"],
        "ligand_source": ligand_source,
        "ligand_graph": build_ligand_graph(ligand),
        "ligand_ecfp": build_morgan_ecfp(ligand),
        "protein_graph": build_protein_graph_from_pdb(
            protein_path, cutoff=protein_cutoff
        ),
    }


def _build_shard_worker(arguments):
    shard_path, rows, signature, pdbbind_root, protein_cutoff = arguments

    def build_record(row):
        return _build_record(row, pdbbind_root, protein_cutoff)

    return build_or_reuse_shard(shard_path, rows, signature, build_record)


def _signature(provenance, protein_cutoff, shard_size):
    payload = {
        "builder_version": BUILDER_VERSION,
        "raw_csv_sha256": provenance["raw_csv_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "protein_cutoff": float(protein_cutoff),
        "shard_size": int(shard_size),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_three_dprot_data(
    raw_csv,
    manifest_paths,
    pdbbind_root,
    output_dir,
    shard_root,
    *,
    workers=8,
    shard_size=64,
    protein_cutoff=10.0,
):
    rows, provenance = load_official_rows(raw_csv, manifest_paths)
    signature = _signature(provenance, protein_cutoff, shard_size)
    shard_root = Path(shard_root).expanduser().resolve()
    shard_root.mkdir(parents=True, exist_ok=True)
    shards = []
    tasks = []
    for start in range(0, len(rows), shard_size):
        index = start // shard_size
        path = shard_root / ("shard_%05d.pkl" % index)
        subset = rows[start : start + shard_size]
        shards.append(path)
        tasks.append((str(path), subset, signature, str(pdbbind_root), protein_cutoff))

    completed = reused = 0
    errors = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_task = {
            executor.submit(_build_shard_worker, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                completed += result["count"]
                reused += result["count"] if result["reused"] else 0
                print(
                    json.dumps(
                        {
                            "event": "three_dprot_shard_complete",
                            "completed": completed,
                            "total": len(rows),
                            "reused": reused,
                            "shard": Path(result["path"]).name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                errors.append({"shard": str(task[0]), "error": repr(error)})
    if errors:
        failure = {
            "status": "failed",
            "signature": signature,
            "completed": completed,
            "total": len(rows),
            "errors": errors,
        }
        _atomic_json(shard_root / "prepare_failures.json", failure)
        raise RuntimeError("%d 3DProt preparation shards failed" % len(errors))

    provenance = dict(provenance)
    provenance.update(
        {
            "signature": signature,
            "protein_cutoff": float(protein_cutoff),
            "shard_size": int(shard_size),
            "workers": int(workers),
            "shard_root": str(shard_root),
        }
    )
    return merge_prepared_shards(shards, rows, output_dir, provenance)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build resumable official LP-PDBbind data for 3DProtDTA"
    )
    parser.add_argument("--lp-csv", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--pdbbind-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--protein-cutoff", type=float, default=10.0)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    manifest_root = Path(args.manifest_root)
    summary = prepare_three_dprot_data(
        args.lp_csv,
        {
            "train": manifest_root / "train_manifest.json",
            "valid": manifest_root / "valid_manifest.json",
            "test": manifest_root / "test_manifest.json",
        },
        args.pdbbind_root,
        args.output,
        args.shard_root,
        workers=args.workers,
        shard_size=args.shard_size,
        protein_cutoff=args.protein_cutoff,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
