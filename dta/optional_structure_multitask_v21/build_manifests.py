"""Build leak-audited LP-PDBBind/Davis/KIBA manifests for V21."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger

from dta.sequence_smiles_binary_probe_v1.core import load_split

from .data import canonical_sequence, sha256_file, stable_id, write_jsonl


VERSION = "optional-structure-multitask-manifest-v21"


@lru_cache(maxsize=None)
def canonical_smiles(smiles):
    smiles = str(smiles).strip()
    molecule = Chem.MolFromSmiles(smiles)
    sanitized = molecule is not None
    if molecule is None:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            raise ValueError("RDKit could not parse SMILES connectivity: %s" % smiles)
        molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(molecule, catchErrors=True)
    try:
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        canonical = smiles
    return canonical, sanitized


def _row(dataset, split, sample_id, sequence, smiles, label, **extra):
    sequence = _cached_sequence(str(sequence))
    smiles, smiles_sanitized = canonical_smiles(str(smiles))
    row = {
        "dataset": dataset,
        "split": split,
        "sample_id": str(sample_id),
        "sequence": sequence,
        "sequence_id": stable_id(sequence),
        "smiles": smiles,
        "smiles_sanitized": smiles_sanitized,
        "ligand_id": stable_id(smiles),
        "pair_id": stable_id(sequence + "\0" + smiles),
        "label": float(label),
        "has_3d": False,
    }
    row.update(extra)
    if not smiles:
        structure_key = row.get("structure_key")
        if not structure_key:
            raise ValueError("empty SMILES requires a structure-backed 2D graph")
        ligand_token = "structure-cache-2d:" + str(structure_key)
        row["ligand_id"] = stable_id(ligand_token)
        row["pair_id"] = stable_id(row["sequence_id"] + "\0" + ligand_token)
        row["ligand_source"] = "structure_cache_2d"
    else:
        row["ligand_source"] = "smiles"
    return row


_cached_sequence = lru_cache(maxsize=None)(canonical_sequence)


def _load_lp_rows(csv_path, manifest_root, structure_root, limit):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_pdb = {str(row["Unnamed: 0"]).lower(): row for row in rows}
    result = {}
    for split in ("train", "valid", "test"):
        manifest_path = manifest_root / (split + "_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = manifest["samples"][: int(limit) if limit is not None else None]
        converted = []
        for sample in samples:
            pdb_id = str(sample["pdb_id"]).lower()
            source = by_pdb.get(pdb_id)
            if source is None:
                raise KeyError("LP-PDBBind CSV is missing %s" % pdb_id)
            if abs(float(source["value"]) - float(sample["label"])) > 1.0e-4:
                raise ValueError("LP-PDBBind label mismatch for %s" % pdb_id)
            converted.append(
                _row(
                    "LP_PDBBind",
                    split,
                    "LP_PDBBind:%s" % pdb_id,
                    source["seq"],
                    source["smiles"],
                    sample["label"],
                    has_3d=True,
                    structure_key=str(sample["lmdb_key"]),
                    structure_lmdb=str((structure_root / split / "complexes.lmdb").resolve()),
                    pdb_id=pdb_id,
                )
            )
        result[split] = converted
    return result


def _load_sequence_dataset(name, root, limit):
    result = {}
    for split in ("train", "valid", "test"):
        source = load_split(root, split)
        count = min(len(source), int(limit)) if limit is not None else len(source)
        result[split] = [
            _row(
                name,
                split,
                "%s:%s:%d" % (name, split, index),
                source.proteins[index],
                source.smiles[index],
                source.affinity[index],
            )
            for index in range(count)
        ]
    return result


def _source_hashes(paths):
    hashes = {}
    for path in paths:
        path = Path(path).resolve()
        hashes[str(path)] = sha256_file(path)
    return hashes


def build(args):
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    lp_csv = args.lp_csv.expanduser().resolve()
    lp_manifest_root = args.lp_manifest_root.expanduser().resolve()
    structure_root = args.structure_root.expanduser().resolve()
    davis_root = args.davis_root.expanduser().resolve()
    kiba_root = args.kiba_root.expanduser().resolve()
    RDLogger.DisableLog("rdApp.error")
    datasets = {
        "LP_PDBBind": _load_lp_rows(
            lp_csv, lp_manifest_root, structure_root, args.limit_per_split
        ),
        "DAVIS": _load_sequence_dataset("DAVIS", davis_root, args.limit_per_split),
        "KIBA": _load_sequence_dataset("KIBA", kiba_root, args.limit_per_split),
    }

    protected_pairs = set()
    for splits in datasets.values():
        for split in ("valid", "test"):
            protected_pairs.update(row["pair_id"] for row in splits[split])
    overlap = Counter()
    removed = []
    for name, splits in datasets.items():
        kept = []
        for row in splits["train"]:
            if row["pair_id"] in protected_pairs:
                overlap[name] += 1
                removed.append({"dataset": name, "sample_id": row["sample_id"], "pair_id": row["pair_id"]})
            else:
                kept.append(row)
        splits["train"] = kept

    combined = defaultdict(list)
    for splits in datasets.values():
        for split in ("train", "valid", "test"):
            combined[split].extend(splits[split])
    for split in combined:
        combined[split].sort(key=lambda row: (row["dataset"], row["sample_id"]))
        write_jsonl(output / (split + ".jsonl"), combined[split])
    write_jsonl(output / "removed_training_pair_overlaps.jsonl", removed or [{"status": "none"}])

    source_paths = [lp_csv]
    source_paths.extend(lp_manifest_root / (split + "_manifest.json") for split in ("train", "valid", "test"))
    for root in (davis_root, kiba_root):
        source_paths.extend([root / "input0" / "dict.txt", root / "input1" / "dict.txt"])
        for split in ("train", "valid", "test"):
            source_paths.extend(
                [
                    root / "input0" / (split + ".idx"),
                    root / "input0" / (split + ".bin"),
                    root / "input1" / (split + ".idx"),
                    root / "input1" / (split + ".bin"),
                    root / "label" / (split + ".label"),
                ]
            )
    counts = {
        split: dict(Counter(row["dataset"] for row in rows)) for split, rows in combined.items()
    }
    metadata = {
        "version": VERSION,
        "output": str(output),
        "counts": counts,
        "training_exact_pair_overlaps_removed": dict(overlap),
        "unsanitized_smiles_rows": int(
            sum(not row["smiles_sanitized"] for rows in combined.values() for row in rows)
        ),
        "protected_pair_definition": "union of all validation and test sequence/SMILES pairs; labels are not used for filtering",
        "source_sha256": _source_hashes(source_paths),
        "manifest_sha256": {
            split: sha256_file(output / (split + ".jsonl")) for split in ("train", "valid", "test")
        },
        "test_labels_used_for_model_selection": False,
        "status": "built",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lp-csv", type=Path, default=Path("data/LP_PDBBind.csv"))
    parser.add_argument(
        "--lp-manifest-root",
        type=Path,
        default=Path("data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"),
    )
    parser.add_argument(
        "--structure-root",
        type=Path,
        default=Path("data/LP_PDBBind_MSA_v9/complex_cache_full"),
    )
    parser.add_argument("--davis-root", type=Path, default=Path("data/DAVIS"))
    parser.add_argument("--kiba-root", type=Path, default=Path("data/KIBA"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/OptionalStructureMultitask_v21/manifests")
    )
    parser.add_argument("--limit-per-split", type=int)
    return parser


def main():
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
