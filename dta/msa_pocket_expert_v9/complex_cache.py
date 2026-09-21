"""Build compact LP-PDBbind complex caches with pocket-aligned MSA profiles."""

import argparse
import json
import math
import os
from pathlib import Path
import pickle

import lmdb
import numpy as np

from data_utils.lmdb_dataset import AA_TO_ID, ordered_chain_items
from utils.experiment_integrity import sha256_file

from . import VERSION
from .msa_profile import PROFILE_FEATURE_NAMES, _atomic_json
from .sequence_manifest import normalize_query_sequence


RESIDUE_SCALAR_FIELDS = (
    "hydrophobicity",
    "charge",
    "polarity",
    "aromatic",
    "hbd",
    "hba",
    "phi_sin",
    "phi_cos",
    "psi_sin",
    "psi_cos",
    "omega_sin",
    "omega_cos",
    "backbone_complete_mask",
    "sidechain_complete_mask",
    "cb_valid_mask",
)
LIGAND_CATEGORICAL_FIELDS = (
    "atom_atomic_numbers",
    "atom_is_donor",
    "atom_is_acceptor",
    "atom_formal_charge",
    "atom_hybridization",
    "atom_is_aromatic",
    "atom_chiral_tag",
    "atom_num_rings",
    "atom_degree",
    "atom_total_valence",
    "atom_num_hydrogens",
    "atom_is_in_ring",
    "atom_min_ring_size",
    "atom_is_hydrophobe",
    "atom_is_polar",
    "atom_is_positive",
    "atom_is_negative",
    "atom_is_halogen",
    "atom_is_metal_binding_candidate",
)
LIGAND_CONTINUOUS_FIELDS = (
    "atom_logp_contribution",
    "atom_partial_charge",
)
BOND_FIELDS = (
    "bond_type",
    "bond_is_rotatable",
    "bond_is_aromatic",
    "bond_is_in_ring",
    "bond_is_conjugated",
    "bond_stereo",
)
CACHE_SCHEMA_VERSION = 1


def _load_json(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return path, value


def _resolve_manifest_lmdb(manifest_path, manifest):
    value = manifest.get("lmdb_path")
    if not value:
        raise ValueError("LP manifest does not define lmdb_path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _occurrence_index(sequence_manifest):
    result = {}
    for record in sequence_manifest.get("sequences", []):
        sequence_id = str(record["sequence_id"])
        for occurrence in record.get("occurrences", []):
            key = (
                str(occurrence["split"]),
                str(occurrence["lmdb_key"]),
                str(occurrence["chain_id"]),
            )
            if key in result:
                raise ValueError(f"duplicate chain occurrence: {key}")
            result[key] = sequence_id
    return result


def _stack_fields(payload, fields, length, dtype):
    columns = []
    for field in fields:
        value = np.asarray(payload[field])
        if value.shape != (length,):
            raise ValueError(f"field {field} has shape {value.shape}, expected {(length,)}")
        columns.append(value.astype(dtype, copy=False))
    return np.stack(columns, axis=1)


def _residue_ligand_distances(chain, ligand_coords):
    residue_count = len(chain["sequence"])
    atom_coords = np.asarray(chain["atom_coords"], dtype=np.float32)
    residue_indices = np.asarray(chain["atom_residue_indices"], dtype=np.int64)
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
        raise ValueError("protein atom coordinates have the wrong shape")
    if residue_indices.shape != (atom_coords.shape[0],):
        raise ValueError("protein atom residue indices have the wrong shape")
    if np.any(residue_indices < 0) or np.any(residue_indices >= residue_count):
        raise ValueError("protein atom residue index is outside the chain")
    delta = atom_coords[:, None, :] - ligand_coords[None, :, :]
    atom_distances = np.sqrt(np.sum(delta * delta, axis=-1))
    result = np.full((residue_count, ligand_coords.shape[0]), np.inf, dtype=np.float32)
    for ligand_index in range(ligand_coords.shape[0]):
        np.minimum.at(result[:, ligand_index], residue_indices, atom_distances[:, ligand_index])
    return result


def _select_pocket(distance_matrix, radius, minimum, maximum):
    minimum_distance = np.min(distance_matrix, axis=1)
    finite = np.flatnonzero(np.isfinite(minimum_distance))
    if finite.size == 0:
        raise ValueError("complex contains no finite residue-ligand distance")
    order = finite[np.argsort(minimum_distance[finite], kind="mergesort")]
    within = order[minimum_distance[order] <= float(radius)]
    target_count = min(max(int(minimum), len(within)), int(maximum), len(order))
    selected = order[:target_count]
    return np.sort(selected), minimum_distance[selected]


def _build_complex(
    item,
    sample,
    split,
    occurrence_by_chain,
    profile_transaction,
    pocket_radius,
    minimum_pocket_residues,
    maximum_pocket_residues,
    maximum_ligand_atoms,
):
    key = str(sample["lmdb_key"])
    protein_features = item.get("protein_features")
    ligand = item.get("ligand_features")
    if not isinstance(protein_features, dict) or not isinstance(ligand, dict):
        raise ValueError(f"complex features are missing for {key}")
    ligand_coords = np.asarray(ligand["atom_coords"], dtype=np.float32)
    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3 or ligand_coords.shape[0] == 0:
        raise ValueError(f"ligand coordinates have the wrong shape for {key}")
    if ligand_coords.shape[0] > int(maximum_ligand_atoms):
        raise ValueError(
            f"ligand {key} has {ligand_coords.shape[0]} atoms, limit is {maximum_ligand_atoms}"
        )

    msa_rows = []
    esm_rows = []
    scalar_rows = []
    aa_rows = []
    coord_rows = []
    distance_rows = []
    chain_rows = []
    position_rows = []
    profile_ids = []
    for chain_index, (chain_id, chain) in enumerate(ordered_chain_items(protein_features)):
        sequence = normalize_query_sequence(chain["sequence"])
        length = len(sequence)
        sequence_id = occurrence_by_chain.get((split, key, str(chain_id)))
        if sequence_id is None:
            raise KeyError(f"MSA sequence occurrence missing for {(split, key, chain_id)}")
        raw_profile = profile_transaction.get(sequence_id.encode("ascii"))
        if raw_profile is None:
            raise KeyError(f"MSA profile missing for {sequence_id}")
        profile = pickle.loads(bytes(raw_profile))
        if profile["query_sequence"] != sequence:
            raise ValueError(f"MSA query sequence mismatch for {(key, chain_id)}")
        msa = np.asarray(profile["residue_features"], dtype=np.float32)
        esm = np.asarray(chain["esm2_embedding"], dtype=np.float32)
        coords = np.asarray(chain["ca_coords"], dtype=np.float32)
        if msa.shape != (length, len(PROFILE_FEATURE_NAMES)):
            raise ValueError(f"MSA profile shape mismatch for {(key, chain_id)}")
        if esm.shape != (length, 640) or coords.shape != (length, 3):
            raise ValueError(f"residue feature shape mismatch for {(key, chain_id)}")
        msa_rows.append(msa)
        esm_rows.append(esm)
        scalar_rows.append(_stack_fields(chain, RESIDUE_SCALAR_FIELDS, length, np.float32))
        aa_rows.append(np.asarray([AA_TO_ID.get(residue, 0) for residue in sequence], dtype=np.int16))
        coord_rows.append(coords)
        distance_rows.append(_residue_ligand_distances(chain, ligand_coords))
        chain_rows.append(np.full(length, chain_index, dtype=np.int16))
        position_rows.append(np.arange(length, dtype=np.int32))
        profile_ids.append(sequence_id)

    msa = np.concatenate(msa_rows, axis=0)
    esm = np.concatenate(esm_rows, axis=0)
    residue_scalar = np.concatenate(scalar_rows, axis=0)
    residue_aa = np.concatenate(aa_rows, axis=0)
    residue_coords = np.concatenate(coord_rows, axis=0)
    residue_ligand_distance = np.concatenate(distance_rows, axis=0)
    chain_index = np.concatenate(chain_rows, axis=0)
    sequence_position = np.concatenate(position_rows, axis=0)
    selected, _minimum_distance = _select_pocket(
        residue_ligand_distance,
        pocket_radius,
        minimum_pocket_residues,
        maximum_pocket_residues,
    )
    selected_pair_distance = residue_ligand_distance[selected]

    ligand_categorical = _stack_fields(
        ligand, LIGAND_CATEGORICAL_FIELDS, ligand_coords.shape[0], np.int16
    )
    ligand_continuous = _stack_fields(
        ligand, LIGAND_CONTINUOUS_FIELDS, ligand_coords.shape[0], np.float32
    )
    bond_edges = np.asarray(ligand["bond_edge_idx"], dtype=np.int64)
    if bond_edges.ndim != 2 or bond_edges.shape[0] != 2:
        raise ValueError(f"ligand bond edges have the wrong shape for {key}")
    bond_features = _stack_fields(ligand, BOND_FIELDS, bond_edges.shape[1], np.int16)
    if np.any(bond_edges < 0) or np.any(bond_edges >= ligand_coords.shape[0]):
        raise ValueError(f"ligand bond edge is outside atom range for {key}")

    selected_coords = residue_coords[selected]
    if not (
        np.isfinite(selected_coords).all()
        and np.isfinite(selected_pair_distance).all()
        and np.isfinite(ligand_coords).all()
    ):
        raise ValueError(f"nonfinite geometry in selected pocket for {key}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "version": VERSION,
        "key": key,
        "label": float(sample["label"]),
        "pdb_id": str(sample.get("pdb_id", key.split("_", 1)[0])),
        "profile_ids": tuple(profile_ids),
        "pocket_msa": msa[selected].astype(np.float16),
        "pocket_esm": esm[selected].astype(np.float16),
        "pocket_scalar": residue_scalar[selected].astype(np.float16),
        "pocket_aa": residue_aa[selected],
        "pocket_coords": selected_coords.astype(np.float32),
        "pocket_chain_index": chain_index[selected],
        "pocket_sequence_position": sequence_position[selected],
        "residue_ligand_distance": np.clip(selected_pair_distance, 0.0, 32.0).astype(np.float16),
        "contact_label": (selected_pair_distance <= 4.5),
        "ligand_categorical": ligand_categorical,
        "ligand_continuous": ligand_continuous.astype(np.float16),
        "ligand_coords": ligand_coords.astype(np.float32),
        "bond_edges": bond_edges.astype(np.int32),
        "bond_features": bond_features,
    }


def build_complex_cache(
    manifest_path,
    sequence_manifest_path,
    profile_lmdb,
    output,
    *,
    pocket_radius=10.0,
    minimum_pocket_residues=32,
    maximum_pocket_residues=160,
    maximum_ligand_atoms=160,
    map_size_gb=64.0,
    limit=None,
):
    manifest_path, manifest = _load_json(manifest_path)
    sequence_manifest_path, sequence_manifest = _load_json(sequence_manifest_path)
    split = str(manifest["split"])
    samples = list(manifest["samples"])
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        samples = samples[: int(limit)]
    source_lmdb = _resolve_manifest_lmdb(manifest_path, manifest)
    profile_lmdb = Path(profile_lmdb).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    occurrence_by_chain = _occurrence_index(sequence_manifest)
    lineage = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "version": VERSION,
        "split": split,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "sequence_manifest": str(sequence_manifest_path),
        "sequence_manifest_sha256": sha256_file(sequence_manifest_path),
        "source_lmdb": str(source_lmdb),
        "profile_lmdb": str(profile_lmdb),
        "pocket_radius": float(pocket_radius),
        "minimum_pocket_residues": int(minimum_pocket_residues),
        "maximum_pocket_residues": int(maximum_pocket_residues),
        "maximum_ligand_atoms": int(maximum_ligand_atoms),
        "limit": None if limit is None else int(limit),
    }
    lineage_path = output / "lineage.json"
    if lineage_path.is_file():
        existing = json.loads(lineage_path.read_text(encoding="utf-8"))
        if existing != lineage:
            raise ValueError("complex cache lineage does not match")
    else:
        _atomic_json(lineage_path, lineage)

    source_env = lmdb.open(
        str(source_lmdb), readonly=True, lock=False, readahead=False, meminit=False
    )
    profile_env = lmdb.open(
        str(profile_lmdb), readonly=True, lock=False, readahead=False, meminit=False
    )
    output_env = lmdb.open(
        str(output / "complexes.lmdb"),
        map_size=int(float(map_size_gb) * (1 << 30)),
        subdir=True,
    )
    pocket_sizes = []
    ligand_sizes = []
    reused = 0
    written = 0
    try:
        with source_env.begin(write=False) as source_txn, profile_env.begin(write=False) as profile_txn:
            for index, sample in enumerate(samples, start=1):
                key = str(sample["lmdb_key"])
                with output_env.begin(write=False) as output_txn:
                    cached = output_txn.get(key.encode("utf-8"))
                if cached is not None:
                    complex_item = pickle.loads(bytes(cached))
                    reused += 1
                else:
                    raw = source_txn.get(key.encode("utf-8"))
                    if raw is None:
                        raise KeyError(f"source LMDB key not found: {key}")
                    item = pickle.loads(bytes(raw))
                    complex_item = _build_complex(
                        item,
                        sample,
                        split,
                        occurrence_by_chain,
                        profile_txn,
                        pocket_radius,
                        minimum_pocket_residues,
                        maximum_pocket_residues,
                        maximum_ligand_atoms,
                    )
                    with output_env.begin(write=True) as output_txn:
                        output_txn.put(
                            key.encode("utf-8"),
                            pickle.dumps(complex_item, protocol=4),
                            overwrite=False,
                        )
                    written += 1
                pocket_sizes.append(int(complex_item["pocket_msa"].shape[0]))
                ligand_sizes.append(int(complex_item["ligand_coords"].shape[0]))
                if index % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "msa_complex_cache_progress",
                                "processed": index,
                                "reused": reused,
                                "written": written,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        output_env.sync()
        output_env.close()
        profile_env.close()
        source_env.close()
    report = {
        "status": "complete",
        "split": split,
        "processed": len(samples),
        "written": written,
        "reused": reused,
        "pocket_size": {
            "minimum": min(pocket_sizes),
            "mean": sum(pocket_sizes) / len(pocket_sizes),
            "maximum": max(pocket_sizes),
        },
        "ligand_size": {
            "minimum": min(ligand_sizes),
            "mean": sum(ligand_sizes) / len(ligand_sizes),
            "maximum": max(ligand_sizes),
        },
        "output": str(output),
    }
    _atomic_json(output / "cache_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--profile-lmdb", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--minimum-pocket-residues", type=int, default=32)
    parser.add_argument("--maximum-pocket-residues", type=int, default=160)
    parser.add_argument("--maximum-ligand-atoms", type=int, default=160)
    parser.add_argument("--map-size-gb", type=float, default=64.0)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    build_complex_cache(
        args.manifest,
        args.sequence_manifest,
        args.profile_lmdb,
        args.output,
        pocket_radius=args.pocket_radius,
        minimum_pocket_residues=args.minimum_pocket_residues,
        maximum_pocket_residues=args.maximum_pocket_residues,
        maximum_ligand_atoms=args.maximum_ligand_atoms,
        map_size_gb=args.map_size_gb,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
