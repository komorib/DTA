from dataclasses import dataclass
import os

import numpy as np
from rdkit import Chem


LOCAL_INTERFACE_SCHEMA_VERSION = 1
PROTEIN_ELEMENT_SYMBOLS = {0: "C", 1: "N", 2: "O", 3: "S"}
PROTEIN_ATOM_NAME_VOCAB = (
    "UNK",
    "N",
    "CA",
    "C",
    "O",
    "OXT",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CZ",
    "CZ2",
    "CZ3",
    "CH2",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
)
PROTEIN_ATOM_NAME_TO_CLASS = {
    name: index for index, name in enumerate(PROTEIN_ATOM_NAME_VOCAB)
}
RESIDUE_TYPE_SYMBOLS = tuple("ACDEFGHIKLMNPQRSTVWY")
RESIDUE_TYPE_TO_CLASS = {
    symbol: index for index, symbol in enumerate(RESIDUE_TYPE_SYMBOLS)
}


@dataclass(frozen=True)
class LocalInterfaceConfig:
    core_cutoff: float = 6.0
    outer_cutoff: float = 10.0
    shell_budget: int = 128

    def __post_init__(self):
        if not np.isfinite(self.core_cutoff) or self.core_cutoff <= 0:
            raise ValueError("core_cutoff must be positive and finite")
        if not np.isfinite(self.outer_cutoff) or self.outer_cutoff <= self.core_cutoff:
            raise ValueError("outer_cutoff must be finite and greater than core_cutoff")
        if int(self.shell_budget) != self.shell_budget or self.shell_budget < 0:
            raise ValueError("shell_budget must be a non-negative integer")


def resolve_source_lmdb(manifest, manifest_dir):
    path_value = manifest.get("structure_lmdb_path") or manifest.get("lmdb_path")
    if not path_value:
        raise ValueError("manifest must define lmdb_path or structure_lmdb_path")
    path = os.fspath(path_value)
    if not os.path.isabs(path):
        path = os.path.join(os.fspath(manifest_dir), path)
    return os.path.abspath(path)


def _sequence_to_string(sequence):
    if isinstance(sequence, np.ndarray):
        return "".join(str(item) for item in sequence.reshape(-1).tolist())
    if isinstance(sequence, (list, tuple)):
        return "".join(str(item) for item in sequence)
    if isinstance(sequence, bytes):
        return sequence.decode("utf-8")
    return str(sequence)


def _ordered_chain_items(protein_features):
    if isinstance(protein_features, dict):
        return [(str(key), protein_features[key]) for key in sorted(protein_features)]
    return [(str(index), chain) for index, chain in enumerate(protein_features)]


def canonical_chain_signature(protein_features):
    sequences = sorted(
        _sequence_to_string(chain.get("sequence", ""))
        for _, chain in _ordered_chain_items(protein_features)
    )
    return "|".join(f"{len(sequence)}:{sequence}" for sequence in sequences)


def continuous_contact_weights(distances):
    distances = np.asarray(distances, dtype=np.float32)
    weights = np.zeros_like(distances, dtype=np.float32)
    weights[distances <= 5.0] = 1.0
    shell = (distances > 5.0) & (distances < 6.0)
    weights[shell] = 0.5 * (
        np.cos(np.pi * (distances[shell] - 5.0)) + 1.0
    )
    return weights


def ligand_automorphisms(mol, max_matches=4096):
    if int(max_matches) != max_matches or max_matches <= 0:
        raise ValueError("max_matches must be a positive integer")
    max_matches = int(max_matches)
    atom_count = mol.GetNumAtoms()
    identity = tuple(range(atom_count))
    matches = mol.GetSubstructMatches(
        mol,
        uniquify=False,
        useChirality=True,
        maxMatches=max_matches + 1,
    )
    if len(matches) > max_matches:
        raise ValueError(
            f"ligand automorphism count exceeds max_matches={max_matches}"
        )
    mappings = {tuple(int(index) for index in match) for match in matches}
    mappings.add(identity)
    return np.asarray(sorted(mappings), dtype=np.int64).reshape(-1, atom_count)


def _aligned_ligand_array(ligand_features, field, atom_count, dtype, fallback):
    value = ligand_features.get(field)
    if value is None:
        value = fallback
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if len(array) != atom_count:
        raise ValueError(
            f"ligand field {field} has length {len(array)}, expected {atom_count}"
        )
    return array


def ligand_heavy_atom_view(ligand_features):
    mol = ligand_features.get("mol")
    if mol is None:
        raise ValueError("ligand_features must contain an RDKit mol")
    mol = Chem.Mol(mol)
    atom_count = mol.GetNumAtoms()
    if atom_count == 0 or mol.GetNumConformers() == 0:
        raise ValueError("ligand mol must contain atoms and a conformer")

    original_atomic_numbers = _aligned_ligand_array(
        ligand_features,
        "atom_atomic_numbers",
        atom_count,
        np.int64,
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
    coordinates = np.asarray(
        ligand_features.get(
            "atom_coords", mol.GetConformer().GetPositions()
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    if len(coordinates) != atom_count:
        raise ValueError(
            f"ligand atom_coords has length {len(coordinates)}, expected {atom_count}"
        )

    heavy_indices = np.asarray(
        [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1
        ],
        dtype=np.int64,
    )
    if len(heavy_indices) == 0:
        raise ValueError("ligand has no heavy atoms")
    heavy_mol = Chem.RemoveAllHs(mol)
    if heavy_mol.GetNumAtoms() != len(heavy_indices):
        raise ValueError("RDKit hydrogen removal changed the heavy-atom count")
    heavy_atomic_numbers = original_atomic_numbers[heavy_indices]
    mol_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in heavy_mol.GetAtoms()], dtype=np.int64
    )
    if not np.array_equal(heavy_atomic_numbers, mol_atomic_numbers):
        raise ValueError("ligand heavy-atom order changed during hydrogen removal")
    heavy_chiral_tags = np.asarray(
        [int(atom.GetChiralTag()) for atom in heavy_mol.GetAtoms()],
        dtype=np.int64,
    )

    from utils.features import (
        get_bond_onehot_feature_dim,
        simplify_atom_to_graph_data_obj,
        simplify_atom_to_onehot_feature_vector,
    )

    if heavy_mol.GetNumBonds() == 0:
        atom_features = np.asarray(
            [
                simplify_atom_to_onehot_feature_vector(atom)
                for atom in heavy_mol.GetAtoms()
            ],
            dtype=np.int64,
        )
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty(
            (0, get_bond_onehot_feature_dim()), dtype=np.int64
        )
    else:
        graph = simplify_atom_to_graph_data_obj(heavy_mol)
        atom_features = graph.x.detach().cpu().numpy().astype(np.int64)
        edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
        edge_attr = graph.edge_attr.detach().cpu().numpy().astype(np.int64)
    if atom_features.ndim != 2 or atom_features.shape[1] != 55:
        raise ValueError(
            f"expected 55-D ligand atom features, got {atom_features.shape}"
        )
    canonical_ranks = np.asarray(
        Chem.CanonicalRankAtoms(
            heavy_mol, breakTies=False, includeChirality=True
        ),
        dtype=np.int64,
    )
    return {
        "mol": heavy_mol,
        "original_heavy_indices": heavy_indices,
        "coords": coordinates[heavy_indices].astype(np.float32),
        "atomic_numbers": heavy_atomic_numbers.astype(np.int64),
        "chiral_tags": heavy_chiral_tags,
        "canonical_ranks": canonical_ranks,
        "atom_features": atom_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "automorphisms": ligand_automorphisms(heavy_mol),
    }


def _required_chain_array(chain, field, atom_count, dtype):
    if field not in chain:
        raise ValueError(f"protein chain is missing {field}")
    array = np.asarray(chain[field], dtype=dtype).reshape(-1)
    if len(array) != atom_count:
        raise ValueError(
            f"protein field {field} has length {len(array)}, expected {atom_count}"
        )
    return array


def _flatten_protein(protein_features):
    atom_fields = {
        "coords": [],
        "element": [],
        "element_symbol": [],
        "atom_name": [],
        "atom_name_class": [],
        "residue_type": [],
        "is_backbone": [],
        "parent_residue": [],
        "chain_index": [],
        "source_atom_index": [],
    }
    residue_ca_coords = []
    residue_chain_index = []
    residue_chain_id = []
    residue_sequence = []
    residue_offset = 0

    for chain_index, (chain_id, chain) in enumerate(
        _ordered_chain_items(protein_features)
    ):
        sequence = _sequence_to_string(chain.get("sequence", ""))
        if not sequence:
            raise ValueError(f"protein chain {chain_id} has an empty sequence")
        ca_coords = np.asarray(chain.get("ca_coords", []), dtype=np.float32).reshape(-1, 3)
        if len(ca_coords) != len(sequence):
            raise ValueError(
                f"protein chain {chain_id} has {len(ca_coords)} CA coordinates "
                f"for {len(sequence)} residues"
            )
        coords = np.asarray(chain.get("atom_coords", []), dtype=np.float32).reshape(-1, 3)
        atom_count = len(coords)
        if atom_count == 0:
            raise ValueError(f"protein chain {chain_id} has no atoms")
        elements = _required_chain_array(
            chain, "atom_element", atom_count, np.int64
        )
        unknown_elements = sorted(
            set(elements.tolist()) - set(PROTEIN_ELEMENT_SYMBOLS)
        )
        if unknown_elements:
            raise ValueError(
                f"protein chain {chain_id} has unknown element tokens {unknown_elements}"
            )
        atom_names = _required_chain_array(
            chain, "atom_names", atom_count, str
        )
        local_parents = _required_chain_array(
            chain, "atom_residue_indices", atom_count, np.int64
        )
        if np.any(local_parents < 0) or np.any(local_parents >= len(sequence)):
            raise ValueError(
                f"protein chain {chain_id} has out-of-range residue indices"
            )
        residue_types = np.asarray(
            chain.get("atom_residue_type", np.full(atom_count, -1)),
            dtype=np.int64,
        ).reshape(-1)
        if len(residue_types) != atom_count:
            raise ValueError(
                f"protein field atom_residue_type has length {len(residue_types)}, "
                f"expected {atom_count}"
            )
        is_backbone = np.asarray(
            chain.get(
                "atom_is_backbone",
                np.isin(atom_names, ["N", "CA", "C", "O"]),
            ),
            dtype=bool,
        ).reshape(-1)
        if len(is_backbone) != atom_count:
            raise ValueError(
                f"protein field atom_is_backbone has length {len(is_backbone)}, "
                f"expected {atom_count}"
            )

        atom_fields["coords"].append(coords)
        atom_fields["element"].append(elements)
        atom_fields["element_symbol"].append(
            np.asarray([PROTEIN_ELEMENT_SYMBOLS[int(token)] for token in elements])
        )
        atom_fields["atom_name"].append(atom_names)
        atom_fields["atom_name_class"].append(
            np.asarray(
                [
                    PROTEIN_ATOM_NAME_TO_CLASS.get(str(name).strip().upper(), 0)
                    for name in atom_names
                ],
                dtype=np.int64,
            )
        )
        atom_fields["residue_type"].append(residue_types)
        atom_fields["is_backbone"].append(is_backbone)
        atom_fields["parent_residue"].append(local_parents + residue_offset)
        atom_fields["chain_index"].append(
            np.full(atom_count, chain_index, dtype=np.int64)
        )
        atom_fields["source_atom_index"].append(
            np.arange(atom_count, dtype=np.int64)
        )
        residue_ca_coords.append(ca_coords)
        residue_chain_index.append(
            np.full(len(sequence), chain_index, dtype=np.int64)
        )
        residue_chain_id.extend([chain_id] * len(sequence))
        residue_sequence.extend(sequence)
        residue_offset += len(sequence)

    if not residue_ca_coords:
        raise ValueError("protein_features contains no chains")
    flattened = {
        name: np.concatenate(values, axis=0)
        for name, values in atom_fields.items()
    }
    flattened.update(
        {
            "residue_ca_coords": np.concatenate(
                residue_ca_coords, axis=0
            ).astype(np.float32),
            "residue_chain_index": np.concatenate(
                residue_chain_index, axis=0
            ).astype(np.int64),
            "residue_chain_id": np.asarray(residue_chain_id),
            "residue_sequence": np.asarray(residue_sequence),
            "residue_count": residue_offset,
        }
    )
    return flattened


def _stable_atom_order(indices, minimum_distances, chain_indices, source_indices):
    return sorted(
        (int(index) for index in indices),
        key=lambda index: (
            float(minimum_distances[index]),
            int(chain_indices[index]),
            int(source_indices[index]),
        ),
    )


def _select_shell_atoms(distances, core_mask, shell_mask, protein, config):
    if config.shell_budget == 0:
        return []
    shell_indices = np.flatnonzero(shell_mask)
    selected = set()
    core_indices = np.flatnonzero(core_mask)
    if len(core_indices):
        covered_by_core = np.any(
            distances[core_indices] <= config.core_cutoff, axis=0
        )
    else:
        covered_by_core = np.zeros(distances.shape[1], dtype=bool)

    for ligand_index in np.flatnonzero(~covered_by_core):
        candidates = shell_indices[
            distances[shell_indices, ligand_index] <= config.outer_cutoff
        ]
        if len(candidates):
            nearest = min(
                (int(index) for index in candidates),
                key=lambda index: (
                    float(distances[index, ligand_index]),
                    int(protein["chain_index"][index]),
                    int(protein["source_atom_index"][index]),
                ),
            )
            selected.add(nearest)

    minimum_distances = distances.min(axis=1)
    ordered_shell = _stable_atom_order(
        shell_indices,
        minimum_distances,
        protein["chain_index"],
        protein["source_atom_index"],
    )
    target_count = max(int(config.shell_budget), len(selected))
    for index in ordered_shell:
        if len(selected) >= target_count:
            break
        selected.add(index)
    return _stable_atom_order(
        selected,
        minimum_distances,
        protein["chain_index"],
        protein["source_atom_index"],
    )


def extract_local_interface_record(payload, key, split, config=None):
    config = config or LocalInterfaceConfig()
    if "protein_features" not in payload or "ligand_features" not in payload:
        raise ValueError("payload must contain protein_features and ligand_features")
    ligand = ligand_heavy_atom_view(payload["ligand_features"])
    protein = _flatten_protein(payload["protein_features"])
    ligand_coords = ligand["coords"]
    protein_coords = protein["coords"].astype(np.float32)
    if not np.all(np.isfinite(ligand_coords)) or not np.all(np.isfinite(protein_coords)):
        raise ValueError("protein and ligand coordinates must be finite")

    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :], axis=-1
    )
    minimum_distances = distances.min(axis=1)
    core_mask = minimum_distances <= config.core_cutoff
    shell_mask = (
        (minimum_distances > config.core_cutoff)
        & (minimum_distances <= config.outer_cutoff)
    )
    core_indices = np.flatnonzero(core_mask).tolist()
    shell_indices = _select_shell_atoms(
        distances, core_mask, shell_mask, protein, config
    )
    selected_indices = np.asarray(core_indices + shell_indices, dtype=np.int64)
    if len(selected_indices) == 0:
        raise ValueError(
            f"sample {key} has no protein atoms within {config.outer_cutoff:g} A"
        )
    selected_distances = distances[selected_indices]
    ligand_has_neighbor = np.any(
        selected_distances <= config.outer_cutoff, axis=0
    )
    shell_candidate_count = int(shell_mask.sum())

    record = {
        "schema_version": LOCAL_INTERFACE_SCHEMA_VERSION,
        "key": str(key),
        "split": str(split),
        "target_signature": canonical_chain_signature(
            payload["protein_features"]
        ),
        "core_cutoff": float(config.core_cutoff),
        "outer_cutoff": float(config.outer_cutoff),
        "shell_budget": int(config.shell_budget),
        "ligand_original_heavy_indices": ligand[
            "original_heavy_indices"
        ].astype(np.int64),
        "ligand_coords": ligand_coords.astype(np.float32),
        "ligand_atomic_numbers": ligand["atomic_numbers"].astype(np.int64),
        "ligand_chiral_tag": ligand["chiral_tags"].astype(np.int64),
        "ligand_canonical_rank": ligand["canonical_ranks"].astype(np.int64),
        "ligand_atom_features": ligand["atom_features"].astype(np.int64),
        "ligand_edge_index": ligand["edge_index"].astype(np.int64),
        "ligand_edge_attr": ligand["edge_attr"].astype(np.int64),
        "ligand_automorphisms": ligand["automorphisms"].astype(np.int64),
        "ligand_heavy_atom_count": int(len(ligand_coords)),
        "protein_coords": protein_coords[selected_indices],
        "protein_element": protein["element"][selected_indices].astype(np.int64),
        "protein_element_symbol": protein["element_symbol"][selected_indices],
        "protein_atom_name": protein["atom_name"][selected_indices],
        "protein_atom_name_class": protein["atom_name_class"][
            selected_indices
        ].astype(np.int64),
        "protein_residue_type": protein["residue_type"][
            selected_indices
        ].astype(np.int64),
        "protein_is_backbone": protein["is_backbone"][
            selected_indices
        ].astype(bool),
        "protein_parent_residue": protein["parent_residue"][
            selected_indices
        ].astype(np.int64),
        "protein_chain_index": protein["chain_index"][
            selected_indices
        ].astype(np.int64),
        "protein_source_atom_index": protein["source_atom_index"][
            selected_indices
        ].astype(np.int64),
        "protein_min_ligand_distance": minimum_distances[
            selected_indices
        ].astype(np.float32),
        "protein_is_core": core_mask[selected_indices].astype(bool),
        "protein_atom_count": int(len(selected_indices)),
        "protein_core_atom_count": int(core_mask.sum()),
        "protein_shell_atom_count": int(len(shell_indices)),
        "protein_atoms_within_outer_cutoff": int(
            (core_mask | shell_mask).sum()
        ),
        "protein_shell_candidate_count": shell_candidate_count,
        "protein_shell_retention_fraction": (
            float(len(shell_indices)) / shell_candidate_count
            if shell_candidate_count
            else 1.0
        ),
        "protein_shell_budget_exceeded": bool(
            len(shell_indices) > config.shell_budget
        ),
        "ligand_has_selected_neighbor": ligand_has_neighbor.astype(bool),
        "ligand_neighbor_coverage": float(ligand_has_neighbor.mean()),
        "residue_ca_coords": protein["residue_ca_coords"].astype(np.float32),
        "residue_chain_index": protein["residue_chain_index"].astype(np.int64),
        "residue_chain_id": protein["residue_chain_id"],
        "residue_sequence": protein["residue_sequence"],
        "residue_count": int(protein["residue_count"]),
    }
    return validate_local_interface_record(record)


def validate_local_interface_record(record):
    if record.get("schema_version") != LOCAL_INTERFACE_SCHEMA_VERSION:
        raise ValueError("unsupported local-interface schema version")
    if "affinity" in record or "esm2_embedding" in record:
        raise ValueError("local-interface records must not contain labels or ESM")

    ligand_coords = np.asarray(record["ligand_coords"])
    protein_coords = np.asarray(record["protein_coords"])
    residue_coords = np.asarray(record["residue_ca_coords"])
    if ligand_coords.shape != (record["ligand_heavy_atom_count"], 3):
        raise ValueError("ligand coordinate shape does not match heavy-atom count")
    if protein_coords.shape != (record["protein_atom_count"], 3):
        raise ValueError("protein coordinate shape does not match atom count")
    if residue_coords.shape != (record["residue_count"], 3):
        raise ValueError("residue CA shape does not match residue count")
    if not all(
        np.all(np.isfinite(array))
        for array in (ligand_coords, protein_coords, residue_coords)
    ):
        raise ValueError("local-interface coordinates must be finite")

    ligand_count = record["ligand_heavy_atom_count"]
    for field in (
        "ligand_atomic_numbers",
        "ligand_chiral_tag",
        "ligand_canonical_rank",
    ):
        if np.asarray(record[field]).shape != (ligand_count,):
            raise ValueError(f"{field} is not aligned to ligand atoms")
    if np.asarray(record["ligand_atom_features"]).shape != (ligand_count, 55):
        raise ValueError("ligand_atom_features must have shape [L,55]")
    automorphisms = np.asarray(record["ligand_automorphisms"])
    if automorphisms.ndim != 2 or automorphisms.shape[1] != ligand_count:
        raise ValueError("ligand automorphisms must have shape [M,L]")
    expected_atoms = np.arange(ligand_count)
    if any(
        not np.array_equal(np.sort(mapping), expected_atoms)
        for mapping in automorphisms
    ):
        raise ValueError("ligand automorphisms must be atom permutations")
    edge_index = np.asarray(record["ligand_edge_index"])
    edge_attr = np.asarray(record["ligand_edge_attr"])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("ligand_edge_index must have shape [2,E]")
    if edge_attr.ndim != 2 or len(edge_attr) != edge_index.shape[1]:
        raise ValueError("ligand edge attributes are not aligned")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= ligand_count
    ):
        raise ValueError("ligand edge index is out of range")

    protein_count = record["protein_atom_count"]
    for field in (
        "protein_element",
        "protein_element_symbol",
        "protein_atom_name",
        "protein_atom_name_class",
        "protein_residue_type",
        "protein_is_backbone",
        "protein_parent_residue",
        "protein_chain_index",
        "protein_source_atom_index",
        "protein_min_ligand_distance",
        "protein_is_core",
    ):
        if np.asarray(record[field]).shape != (protein_count,):
            raise ValueError(f"{field} is not aligned to protein atoms")
    parents = np.asarray(record["protein_parent_residue"])
    if np.any(parents < 0) or np.any(parents >= record["residue_count"]):
        raise ValueError("protein parent-residue index is out of range")
    residue_chain_index = np.asarray(record["residue_chain_index"])
    if residue_chain_index.shape != (record["residue_count"],):
        raise ValueError("residue_chain_index is not aligned to residues")
    protein_chain_index = np.asarray(record["protein_chain_index"])
    if not np.array_equal(residue_chain_index[parents], protein_chain_index):
        raise ValueError("protein parent-residue chain does not match atom chain")
    atom_names = np.asarray(record["protein_atom_name"])
    expected_atom_name_classes = np.asarray(
        [
            PROTEIN_ATOM_NAME_TO_CLASS.get(str(name).strip().upper(), 0)
            for name in atom_names
        ],
        dtype=np.int64,
    )
    if not np.array_equal(
        expected_atom_name_classes,
        np.asarray(record["protein_atom_name_class"]),
    ):
        raise ValueError("protein atom-name classes do not match atom names")
    residue_sequence = np.asarray(record["residue_sequence"])
    if residue_sequence.shape != (record["residue_count"],):
        raise ValueError("residue_sequence is not aligned to residues")
    expected_residue_classes = np.asarray(
        [
            RESIDUE_TYPE_TO_CLASS.get(str(residue_sequence[parent]), -1)
            for parent in parents
        ],
        dtype=np.int64,
    )
    if not np.array_equal(
        expected_residue_classes,
        np.asarray(record["protein_residue_type"]),
    ):
        raise ValueError("protein residue classes do not match parent residues")
    expected_symbols = np.asarray(
        [
            PROTEIN_ELEMENT_SYMBOLS[int(token)]
            for token in np.asarray(record["protein_element"])
        ]
    )
    if not np.array_equal(
        expected_symbols, np.asarray(record["protein_element_symbol"])
    ):
        raise ValueError("protein element symbols do not match element tokens")
    core_flags = np.asarray(record["protein_is_core"], dtype=bool)
    minimum_distances = np.asarray(record["protein_min_ligand_distance"])
    if np.any(minimum_distances[core_flags] > record["core_cutoff"]):
        raise ValueError("core atom lies outside core cutoff")
    if np.any(minimum_distances[~core_flags] <= record["core_cutoff"]):
        raise ValueError("shell atom lies inside core cutoff")
    if np.any(minimum_distances > record["outer_cutoff"]):
        raise ValueError("selected protein atom lies outside outer cutoff")
    if int(core_flags.sum()) != record["protein_core_atom_count"]:
        raise ValueError("protein core count does not match core flags")
    if int((~core_flags).sum()) != record["protein_shell_atom_count"]:
        raise ValueError("protein shell count does not match core flags")
    return record
