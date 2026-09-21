"""Build a coordinate-free RDKit atom/bond cache for all V21 manifests."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .data import read_jsonl, sha256_file


VERSION = "optional-structure-ligand-cache-v21"


def molecule_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    sanitized = molecule is not None
    if molecule is None:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is not None:
            molecule.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(molecule, catchErrors=True)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("RDKit could not build a non-empty graph for %s" % smiles)
    try:
        AllChem.ComputeGasteigerCharges(molecule)
    except Exception:
        pass
    categorical = []
    continuous = []
    for atom in molecule.GetAtoms():
        categorical.append(
            [
                min(atom.GetAtomicNum(), 127),
                min(atom.GetDegree(), 8),
                min(max(atom.GetFormalCharge() + 8, 0), 16),
                min(int(atom.GetHybridization()), 9),
                int(atom.GetIsAromatic()),
                min(atom.GetTotalNumHs(), 9),
                int(atom.IsInRing()),
                min(int(atom.GetChiralTag()), 8),
            ]
        )
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = 0.0
        if not np.isfinite(charge):
            charge = 0.0
        continuous.append(
            [
                atom.GetMass() / 100.0,
                float(atom.GetImplicitValence()) / 8.0,
                float(atom.GetTotalValence()) / 8.0,
                float(np.clip(charge, -3.0, 3.0)),
            ]
        )
    edges = []
    features = []
    bond_types = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    for bond in molecule.GetBonds():
        feature = [0.0] * 6
        feature[bond_types.get(bond.GetBondType(), 0)] = 1.0
        feature[4] = float(bond.GetIsConjugated())
        feature[5] = float(bond.IsInRing())
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((source, target), (target, source)))
        features.extend((feature, feature))
    edge_array = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    return {
        "smiles": smiles,
        "sanitized": sanitized,
        "categorical": np.asarray(categorical, dtype=np.int16),
        "continuous": np.asarray(continuous, dtype=np.float32),
        "bond_edges": edge_array,
        "bond_features": np.asarray(features, dtype=np.float32).reshape(-1, 6),
    }


def structure_cache_graph(record):
    source = np.asarray(record["ligand_categorical"], dtype=np.int64)
    source_continuous = np.asarray(record["ligand_continuous"], dtype=np.float32)
    periodic_table = Chem.GetPeriodicTable()
    atomic_number = source[:, 0].clip(0, 127)
    masses = np.asarray(
        [periodic_table.GetAtomicWeight(int(value)) if value > 0 else 0.0 for value in atomic_number],
        dtype=np.float32,
    )
    categorical = np.stack(
        (
            atomic_number,
            source[:, 8].clip(0, 8),
            (source[:, 3] + 8).clip(0, 16),
            source[:, 4].clip(0, 9),
            source[:, 5].clip(0, 1),
            source[:, 10].clip(0, 9),
            source[:, 11].clip(0, 1),
            source[:, 6].clip(0, 8),
        ),
        axis=1,
    ).astype(np.int16)
    continuous = np.stack(
        (
            masses / 100.0,
            np.maximum(source[:, 9] - source[:, 8], 0) / 8.0,
            source[:, 9] / 8.0,
            np.clip(source_continuous[:, 1], -3.0, 3.0),
        ),
        axis=1,
    ).astype(np.float32)
    edges = np.asarray(record["bond_edges"], dtype=np.int64)
    raw_bond = np.asarray(record["bond_features"], dtype=np.int64)
    features = np.zeros((raw_bond.shape[0], 6), dtype=np.float32)
    features[:, 0] = raw_bond[:, 0] == 1
    features[:, 1] = raw_bond[:, 0] == 2
    features[:, 2] = raw_bond[:, 0] == 3
    features[:, 3] = (raw_bond[:, 2] > 0) | (raw_bond[:, 0] == 12)
    features[:, 4] = raw_bond[:, 4] > 0
    features[:, 5] = raw_bond[:, 3] > 0
    return {
        "smiles": "",
        "sanitized": False,
        "source": "structure_cache_2d",
        "categorical": categorical,
        "continuous": continuous,
        "bond_edges": edges,
        "bond_features": features,
    }


def build(args):
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty cache: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = [path.expanduser().resolve() for path in args.manifests]
    ligands = {}
    for manifest in manifests:
        for row in read_jsonl(manifest):
            source = str(row.get("ligand_source", "smiles"))
            specification = {
                "smiles": str(row["smiles"]),
                "source": source,
            }
            if source == "structure_cache_2d":
                specification["structure_lmdb"] = row.get("structure_lmdb")
                specification["structure_key"] = row.get("structure_key")
            previous = ligands.setdefault(str(row["ligand_id"]), specification)
            if previous != specification:
                raise ValueError("ligand hash collision for %s" % row["ligand_id"])
    RDLogger.DisableLog("rdApp.warning")
    environment = lmdb.open(str(output), map_size=int(args.map_size_gb * (1 << 30)), subdir=True)
    atom_counts = []
    structure_environments = {}
    source_counts = {"smiles": 0, "structure_cache_2d": 0}
    with environment.begin(write=True) as transaction:
        for index, (ligand_id, specification) in enumerate(sorted(ligands.items()), 1):
            if specification["source"] == "structure_cache_2d":
                path = str(Path(specification["structure_lmdb"]).resolve())
                if path not in structure_environments:
                    structure_environments[path] = lmdb.open(
                        path, readonly=True, lock=False, readahead=False, meminit=False
                    )
                raw = structure_environments[path].begin(write=False).get(
                    str(specification["structure_key"]).encode("utf-8")
                )
                if raw is None:
                    raise KeyError("structure cache is missing %s" % specification["structure_key"])
                graph = structure_cache_graph(pickle.loads(bytes(raw)))
            else:
                graph = molecule_graph(specification["smiles"])
            source_counts[specification["source"]] += 1
            atom_counts.append(int(graph["categorical"].shape[0]))
            transaction.put(ligand_id.encode("ascii"), pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))
            if index % 1000 == 0:
                print("cached ligands: %d/%d" % (index, len(ligands)), flush=True)
    environment.sync()
    environment.close()
    for structure_environment in structure_environments.values():
        structure_environment.close()
    metadata = {
        "version": VERSION,
        "count": len(ligands),
        "source_counts": source_counts,
        "atom_count": {
            "minimum": min(atom_counts),
            "maximum": max(atom_counts),
            "mean": float(np.mean(atom_counts)),
        },
        "manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        "coordinate_free": True,
        "status": "built",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/OptionalStructureMultitask_v21/ligands.lmdb")
    )
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    return parser


def main():
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
