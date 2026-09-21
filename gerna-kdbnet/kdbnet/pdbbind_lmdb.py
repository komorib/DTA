"""PDBbind LMDB dataset adapter for KDBNet.

This module keeps the LMDB path separate from the manifest/ESM-PT PDBbind
loader. It reads one LMDB entry per complex and converts the stored features
into the same PyG ``drug`` / ``protein`` objects consumed by ``DTAModel``.
"""

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch_geometric

from kdbnet import mol_graph, pdb_graph, pocket_atom_graph, protein_geometry
from kdbnet.dta import create_fold


_LMDB_ENV_CACHE = {}


def _import_py_lmdb():
    """Import the py-lmdb package even when this repo has a local lmdb/ folder."""
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
            "Could not import py-lmdb. The local repo folder named 'lmdb' "
            "is visible, but the external package providing lmdb.open() was "
            "not found in the Python environment."
        )
    return module


def _normalise_lmdb_dir(lmdb_dir):
    path = Path(lmdb_dir)
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _get_readonly_lmdb_env(lmdb_dir):
    lmdb_dir = _normalise_lmdb_dir(lmdb_dir)
    env = _LMDB_ENV_CACHE.get(lmdb_dir)
    if env is None:
        lmdb = _import_py_lmdb()
        env = lmdb.open(
            lmdb_dir,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=128,
        )
        _LMDB_ENV_CACHE[lmdb_dir] = env
    return env


def _infer_complex_id(index_row, key_suffix="_protein"):
    if "complex_id" in index_row and pd.notna(index_row["complex_id"]):
        return str(index_row["complex_id"]).lower()

    lmdb_key = str(index_row["lmdb_key"]).lower()
    suffixes = [key_suffix, "_protein", "_pocket"]
    for suffix in suffixes:
        if suffix and lmdb_key.endswith(suffix):
            return lmdb_key[: -len(suffix)]
    return lmdb_key


def _read_index(index_path, key_suffix="_protein", target_field="affinity"):
    with open(index_path, "rb") as handle:
        index_rows = pickle.load(handle)
    df = pd.DataFrame(index_rows)
    required = {"lmdb_key", target_field}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("LMDB index missing columns: {}".format(sorted(missing)))
    df["complex_id"] = [
        _infer_complex_id(row, key_suffix=key_suffix)
        for row in df.to_dict("records")
    ]
    df["lmdb_key"] = df["lmdb_key"].astype(str)
    df["complex_id"] = df["complex_id"].astype(str).str.lower()
    df["y"] = df[target_field].astype(float)
    duplicated = df[df["complex_id"].duplicated()]["complex_id"]
    if len(duplicated) > 0:
        raise ValueError(
            "LMDB index has duplicated complex_id values: {}".format(
                duplicated.head().tolist()
            )
        )
    return df


def _as_sequence(sequence):
    if isinstance(sequence, str):
        return sequence
    return "".join(str(aa) for aa in list(sequence))


def _select_esm_embedding(chain_data, esm_key):
    if esm_key in chain_data:
        return chain_data[esm_key]
    for fallback_key in ("esm2_embedding", "esm1b_embedding", "esm_embedding", "seq_emb"):
        if fallback_key in chain_data:
            return chain_data[fallback_key]
    return None


def protein_lmdb_features_to_graph(
    protein_features,
    name=None,
    esm_key="esm2_embedding",
    prot_emb_dim=1280,
    allow_zero_embedding=False,
    num_pos_emb=16,
    num_rbf=16,
    contact_cutoff=8.0,
):
    """Convert LMDB protein_features into the KDBNet protein PyG graph."""
    seq_parts = []
    coord_parts = []
    emb_parts = []

    if not isinstance(protein_features, dict):
        raise ValueError("protein_features must be a dict of chain_id -> features")

    for chain_id in sorted(protein_features.keys()):
        chain_data = protein_features[chain_id]
        if not isinstance(chain_data, dict):
            continue
        if "sequence" not in chain_data or "backbone_coords" not in chain_data:
            continue

        seq = _as_sequence(chain_data["sequence"])
        coords = np.asarray(chain_data["backbone_coords"], dtype=np.float32)
        if len(seq) == 0:
            continue
        if coords.shape != (len(seq), 4, 3):
            raise ValueError(
                "Backbone shape mismatch for {} chain {}: sequence {}, coords {}".format(
                    name, chain_id, len(seq), tuple(coords.shape)
                )
            )

        esm_embedding = _select_esm_embedding(chain_data, esm_key)
        if esm_embedding is None:
            if not allow_zero_embedding:
                raise ValueError(
                    "Missing ESM embedding for {} chain {}. Tried '{}' and fallbacks.".format(
                        name, chain_id, esm_key
                    )
                )
            emb = np.zeros((len(seq), int(prot_emb_dim)), dtype=np.float32)
        else:
            emb = np.asarray(esm_embedding, dtype=np.float32)
            if emb.shape[0] != len(seq):
                raise ValueError(
                    "ESM length mismatch for {} chain {}: sequence {}, embedding {}".format(
                        name, chain_id, len(seq), tuple(emb.shape)
                    )
                )
            if emb.ndim != 2 or emb.shape[1] != int(prot_emb_dim):
                raise ValueError(
                    "ESM dim mismatch for {} chain {}: expected (*, {}), got {}".format(
                        name, chain_id, prot_emb_dim, tuple(emb.shape)
                    )
                )

        seq_parts.append(seq)
        coord_parts.append(coords)
        emb_parts.append(emb)

    if not seq_parts:
        raise ValueError("No usable protein chains found for {}".format(name))

    protein_entry = {
        "name": name,
        "seq": "".join(seq_parts),
        "coords": np.concatenate(coord_parts, axis=0),
        "seq_emb": np.concatenate(emb_parts, axis=0),
    }
    return pdb_graph.featurize_protein_graph(
        protein_entry,
        name=name,
        num_pos_emb=num_pos_emb,
        num_rbf=num_rbf,
        contact_cutoff=contact_cutoff,
    )


def _resolve_structure_path(path_value, structure_root=None):
    path = Path(str(path_value))
    if path.exists():
        return path
    if structure_root is not None:
        root = Path(structure_root)
        candidates = [root / path]

        # LMDB paths may be like P-L/2001-2010/3jy9/3jy9_ligand.sdf,
        # while a local PDBbind extraction may be P-L/3jy9/3jy9_ligand.sdf.
        if len(path.parts) >= 2:
            complex_dir = path.parts[-2]
            filename = path.name
            candidates.extend([
                root / complex_dir / filename,
                root / "P-L" / complex_dir / filename,
            ])
            if path.parts[0] == root.name and len(path.parts) > 1:
                candidates.append(root / Path(*path.parts[1:]))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = list(root.rglob(path.name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            complex_dir = path.parts[-2] if len(path.parts) >= 2 else None
            for match in matches:
                if complex_dir is not None and match.parent.name == complex_dir:
                    return match
            raise FileNotFoundError(
                "Ambiguous structure path for {}. Matches: {}".format(
                    path_value, [str(match) for match in matches[:5]]
                )
            )
    raise FileNotFoundError("Could not resolve structure path: {}".format(path_value))


def drug_lmdb_features_to_graph(
        ligand_features, name=None, edge_cutoff=4.5, num_rbf=16):
    """Convert a stored RDKit mol in ligand_features into the KDBNet drug graph."""
    if "mol" not in ligand_features:
        raise ValueError("ligand_features does not contain a stored RDKit mol")
    mol = ligand_features["mol"]
    conf = mol.GetConformer()

    with torch.no_grad():
        coords = torch.tensor(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=torch.float32,
        )
        atom_feature = torch.as_tensor(
            mol_graph._build_atom_feature(mol),
            dtype=torch.float32,
        )
        dist_mat = torch.cdist(coords, coords)
        row, col = torch.where((dist_mat < edge_cutoff) & (dist_mat > 1e-5))
        edge_index = torch.stack([row, col], dim=0)

    edge_s, edge_v = mol_graph._build_edge_feature(
        coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf
    )
    return torch_geometric.data.Data(
        x=coords,
        edge_index=edge_index,
        name=name,
        node_v=coords.unsqueeze(1),
        node_s=atom_feature,
        edge_v=edge_v,
        edge_s=edge_s,
    )


class PDBbindLmdbGraphDataset(data.Dataset):
    """PyTorch Dataset backed by the PDBbind LMDB."""

    def __init__(
        self,
        df,
        lmdb_dir,
        structure_root=None,
        esm_key="esm2_embedding",
        prot_emb_dim=1280,
        allow_zero_embedding=False,
        ligand_source="sdf",
        use_pocket_atom=False,
        pocket_atom_cutoff=5.0,
        pocket_atom_edge_cutoff=4.5,
        pocket_atom_include_hydrogen=False,
        pocket_atom_include_hetatm=False,
        use_one_d_two_d=False,
        one_d_two_d_protein_edge_cutoff=8.0,
        one_d_two_d_max_protein_residues=128,
        use_interface_pairformer=False,
        interface_pairformer_pocket_radius=10.0,
        interface_pairformer_min_pocket_residues=16,
        interface_pairformer_max_pocket_residues=64,
        num_pos_emb=16,
        num_rbf=16,
        contact_cutoff=8.0,
        cache_graphs=False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lmdb_dir = str(lmdb_dir)
        self.structure_root = structure_root
        self.esm_key = esm_key
        self.prot_emb_dim = int(prot_emb_dim)
        self.allow_zero_embedding = allow_zero_embedding
        self.ligand_source = ligand_source
        self.use_pocket_atom = use_pocket_atom
        self.pocket_atom_cutoff = float(pocket_atom_cutoff)
        self.pocket_atom_edge_cutoff = float(pocket_atom_edge_cutoff)
        self.pocket_atom_include_hydrogen = pocket_atom_include_hydrogen
        self.pocket_atom_include_hetatm = pocket_atom_include_hetatm
        self.use_one_d_two_d = use_one_d_two_d
        self.one_d_two_d_protein_edge_cutoff = float(one_d_two_d_protein_edge_cutoff)
        self.one_d_two_d_max_protein_residues = int(one_d_two_d_max_protein_residues)
        self.use_interface_pairformer = bool(use_interface_pairformer)
        self.interface_pairformer_pocket_radius = float(interface_pairformer_pocket_radius)
        self.interface_pairformer_min_pocket_residues = int(
            interface_pairformer_min_pocket_residues
        )
        self.interface_pairformer_max_pocket_residues = int(
            interface_pairformer_max_pocket_residues
        )
        self.num_pos_emb = num_pos_emb
        self.num_rbf = num_rbf
        self.contact_cutoff = contact_cutoff
        self.cache_graphs = cache_graphs
        self._env = None
        self._cache = {}

    def __len__(self):
        return len(self.df)

    @property
    def env(self):
        if self._env is None:
            self._env = _get_readonly_lmdb_env(self.lmdb_dir)
        return self._env

    def _read_lmdb_entry(self, lmdb_key):
        with self.env.begin(write=False) as txn:
            raw = txn.get(str(lmdb_key).encode())
        if raw is None:
            raise KeyError("LMDB key not found: {}".format(lmdb_key))
        return pickle.loads(raw)


    def _entry_to_graphs(self, row, entry):
        complex_id = row["complex_id"]
        metadata = entry.get("metadata", {})
        protein = protein_lmdb_features_to_graph(
            entry["protein_features"],
            name=complex_id,
            esm_key=self.esm_key,
            prot_emb_dim=self.prot_emb_dim,
            allow_zero_embedding=self.allow_zero_embedding,
            num_pos_emb=self.num_pos_emb,
            num_rbf=self.num_rbf,
            contact_cutoff=self.contact_cutoff,
        )

        ligand_path = metadata.get("ligand_path") or row.get("ligand_path") or row.get("ligand_sdf")
        resolved_ligand_path = None
        if ligand_path:
            resolved_ligand_path = _resolve_structure_path(
                ligand_path,
                structure_root=self.structure_root,
            )

        pocket_path_value = (
            metadata.get("pocket_path")
            or metadata.get("pocket_pdb")
            or row.get("pocket_path")
            or row.get("pocket_pdb")
        )
        resolved_pocket_path = None
        if pocket_path_value:
            try:
                resolved_pocket_path = _resolve_structure_path(
                    pocket_path_value,
                    structure_root=self.structure_root,
                )
            except FileNotFoundError:
                # Interface Pairformer can fall back to backbone/pseudo-CB
                # geometry.  The dedicated pocket-atom branch remains strict.
                if self.use_pocket_atom:
                    raise

        if self.ligand_source == "lmdb_mol":
            drug = drug_lmdb_features_to_graph(
                entry["ligand_features"],
                name=complex_id,
                num_rbf=self.num_rbf,
            )
        else:
            if resolved_ligand_path is None:
                raise ValueError(
                    "LMDB entry for {} has no ligand_path; use --ligand_source "
                    "lmdb_mol or provide ligand paths in the index.".format(complex_id)
                )
            drug = mol_graph.featurize_drug(
                str(resolved_ligand_path),
                name=complex_id,
            )

        if self.use_interface_pairformer:
            # The base LMDB has backbone atoms only.  When the PDBbind pocket
            # file is available, attach real side-chain heavy atoms to the
            # corresponding residue nodes.  Failure to match falls back to the
            # invariant backbone/pseudo-CB geometry prepared in pdb_graph.py.
            geometry_stats = {"matched_residues": 0}
            if resolved_pocket_path is not None:
                try:
                    geometry_stats = protein_geometry.attach_pocket_residue_geometry(
                        protein,
                        resolved_pocket_path,
                        max_atoms_per_residue=protein.residue_atom_pos.size(1),
                        match_tolerance=1.5,
                    )
                except (OSError, ValueError):
                    # Keep the deterministic backbone/pseudo-CB fallback.
                    geometry_stats = {"matched_residues": 0}
            protein.interface_full_atom_residue_count = torch.tensor(
                [int(geometry_stats.get("matched_residues", 0))], dtype=torch.long
            )
            with torch.no_grad():
                nearest = protein_geometry.residue_min_ligand_distance(
                    protein, drug.x.float()
                )
                order = torch.argsort(nearest)
                selected = torch.nonzero(
                    nearest <= self.interface_pairformer_pocket_radius,
                    as_tuple=False,
                ).flatten()
                if selected.numel() < self.interface_pairformer_min_pocket_residues:
                    selected = order[: self.interface_pairformer_min_pocket_residues]
                else:
                    selected = selected[torch.argsort(nearest[selected])]
                selected = selected[: self.interface_pairformer_max_pocket_residues]
                interface_pocket_mask = torch.zeros(
                    protein.x.size(0), dtype=torch.bool
                )
                interface_pocket_mask[selected.cpu()] = True
                # PyG concatenates node-aligned boolean tensors during batching.
                # The mask is defined once from the reference crystal complex,
                # so future coordinate perturbations can keep residue identity fixed.
                protein.interface_pocket_mask = interface_pocket_mask

        pocket_atom = None
        if self.use_pocket_atom:
            if resolved_ligand_path is None:
                raise ValueError(
                    "Pocket atom branch for {} requires a ligand_path".format(
                        complex_id
                    )
                )
            if resolved_pocket_path is None:
                raise ValueError(
                    "Pocket atom branch for {} requires a pocket_path/pocket_pdb".format(
                        complex_id
                    )
                )
            pocket_atom = pocket_atom_graph.featurize_pocket_atom_graph(
                str(resolved_pocket_path),
                str(resolved_ligand_path),
                name=complex_id,
                contact_cutoff=self.pocket_atom_cutoff,
                edge_cutoff=self.pocket_atom_edge_cutoff,
                num_rbf=self.num_rbf,
                include_hydrogen=self.pocket_atom_include_hydrogen,
                include_hetatm=self.pocket_atom_include_hetatm,
            )

        one_d_two_d_graph = None
        if self.use_one_d_two_d:
            from kdbnet import one_d_two_d

            ligand_mol = None
            if resolved_ligand_path is None and "ligand_features" in entry:
                ligand_mol = entry["ligand_features"].get("mol")
            one_d_two_d_graph = one_d_two_d.featurize_one_d_two_d(
                entry["protein_features"],
                mol=ligand_mol,
                sdf_path=resolved_ligand_path,
                name=complex_id,
                esm_key=self.esm_key,
                prot_emb_dim=self.prot_emb_dim,
                allow_zero_embedding=self.allow_zero_embedding,
                protein_edge_cutoff=self.one_d_two_d_protein_edge_cutoff,
                max_protein_residues=self.one_d_two_d_max_protein_residues,
            )

        return drug, protein, pocket_atom, one_d_two_d_graph

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        lmdb_key = row["lmdb_key"]
        if self.cache_graphs and lmdb_key in self._cache:
            drug, protein, pocket_atom, one_d_two_d_graph = self._cache[lmdb_key]
        else:
            entry = self._read_lmdb_entry(lmdb_key)
            drug, protein, pocket_atom, one_d_two_d_graph = self._entry_to_graphs(row, entry)
            if self.cache_graphs:
                self._cache[lmdb_key] = (drug, protein, pocket_atom, one_d_two_d_graph)

        sample = {
            "drug": drug,
            "protein": protein,
            "y": float(row["y"]),
        }
        if self.use_pocket_atom:
            sample["pocket_atom"] = pocket_atom
        if self.use_one_d_two_d:
            sample["one_d_two_d"] = one_d_two_d_graph
        return sample


class PDBbindLmdb:
    """Task wrapper exposing get_split(), compatible with DTAExperiment."""

    def __init__(
        self,
        lmdb_dir,
        index_path=None,
        structure_root="data/PDBbind_full/extracted",
        split_file=None,
        key_suffix="_protein",
        target_field="affinity",
        esm_key="esm2_embedding",
        prot_emb_dim=1280,
        allow_zero_embedding=False,
        ligand_source="sdf",
        use_pocket_atom=False,
        pocket_atom_cutoff=5.0,
        pocket_atom_edge_cutoff=4.5,
        pocket_atom_include_hydrogen=False,
        pocket_atom_include_hetatm=False,
        use_one_d_two_d=False,
        one_d_two_d_protein_edge_cutoff=8.0,
        one_d_two_d_max_protein_residues=128,
        use_interface_pairformer=False,
        interface_pairformer_pocket_radius=10.0,
        interface_pairformer_min_pocket_residues=16,
        interface_pairformer_max_pocket_residues=64,
        limit=None,
        num_pos_emb=16,
        num_rbf=16,
        contact_cutoff=8.0,
        split_method="fixed",
        split_frac=(0.7, 0.1, 0.2),
        split_missing_policy="error",
        seed=42,
        onthefly=False,
        cache_graphs=False,
    ):
        del onthefly
        if lmdb_dir is None:
            raise ValueError("PDBbindLmdb requires lmdb_dir")
        self.lmdb_dir = Path(lmdb_dir)
        self.index_path = Path(index_path) if index_path else self.lmdb_dir / "index.pkl"
        self.structure_root = structure_root
        self.split_file = split_file
        self.key_suffix = key_suffix
        self.target_field = target_field
        self.esm_key = esm_key
        self.prot_emb_dim = int(prot_emb_dim)
        self.allow_zero_embedding = allow_zero_embedding
        self.ligand_source = ligand_source
        self.use_pocket_atom = use_pocket_atom
        self.pocket_atom_cutoff = float(pocket_atom_cutoff)
        self.pocket_atom_edge_cutoff = float(pocket_atom_edge_cutoff)
        self.pocket_atom_include_hydrogen = pocket_atom_include_hydrogen
        self.pocket_atom_include_hetatm = pocket_atom_include_hetatm
        self.use_one_d_two_d = use_one_d_two_d
        self.one_d_two_d_protein_edge_cutoff = float(one_d_two_d_protein_edge_cutoff)
        self.one_d_two_d_max_protein_residues = int(one_d_two_d_max_protein_residues)
        self.use_interface_pairformer = bool(use_interface_pairformer)
        self.interface_pairformer_pocket_radius = float(interface_pairformer_pocket_radius)
        self.interface_pairformer_min_pocket_residues = int(
            interface_pairformer_min_pocket_residues
        )
        self.interface_pairformer_max_pocket_residues = int(
            interface_pairformer_max_pocket_residues
        )
        self.limit = limit
        self.num_pos_emb = num_pos_emb
        self.num_rbf = num_rbf
        self.contact_cutoff = contact_cutoff
        self.split_method = split_method
        self.split_frac = split_frac
        self.split_missing_policy = split_missing_policy
        self.seed = seed
        self.cache_graphs = cache_graphs
        self.split_report = None

        if not self.index_path.exists():
            raise FileNotFoundError("Missing LMDB index: {}".format(self.index_path))
        _get_readonly_lmdb_env(self.lmdb_dir)
        self.df = _read_index(
            self.index_path,
            key_suffix=self.key_suffix,
            target_field=self.target_field,
        )
        if self.limit is not None:
            self.df = self.df.head(int(self.limit)).copy()
        if self.split_file is not None:
            self.df = self._merge_split(self.df, self.split_file)

    def _merge_split(self, df, split_file):
        split_df = pd.read_table(split_file)
        required = {"complex_id", "split"}
        missing = required.difference(split_df.columns)
        if missing:
            raise ValueError("Fixed split file missing columns: {}".format(sorted(missing)))

        split_df = split_df[["complex_id", "split"]].copy()
        split_df["complex_id"] = split_df["complex_id"].astype(str).str.lower()
        split_df["split"] = split_df["split"].astype(str).str.lower()
        duplicated_ids = sorted(
            split_df.loc[split_df["complex_id"].duplicated(), "complex_id"].unique()
        )
        if duplicated_ids:
            duplicate_examples = []
            for complex_id in duplicated_ids[:10]:
                duplicate_examples.append((
                    complex_id,
                    sorted(split_df.loc[
                        split_df["complex_id"] == complex_id, "split"
                    ].tolist()),
                ))
            raise ValueError(
                "Fixed split file must assign each base complex to exactly one "
                "partition. Duplicates: {}".format(duplicate_examples)
            )
        valid_splits = {"train", "valid", "test"}
        unknown_splits = sorted(set(split_df["split"]).difference(valid_splits))
        if unknown_splits:
            raise ValueError(
                "Fixed split file has unknown split labels: {}".format(unknown_splits)
            )

        lmdb_ids = set(df["complex_id"])
        split_ids = set(split_df["complex_id"])
        lmdb_without_split = sorted(lmdb_ids.difference(split_ids))
        split_without_lmdb = sorted(split_ids.difference(lmdb_ids))

        self.split_report = {
            "policy": self.split_missing_policy,
            "lmdb_rows_before_split": len(df),
            "split_rows": len(split_df),
            "lmdb_without_split": len(lmdb_without_split),
            "split_without_lmdb": len(split_without_lmdb),
            "lmdb_without_split_examples": lmdb_without_split[:10],
            "split_without_lmdb_examples": split_without_lmdb[:10],
        }

        if lmdb_without_split and self.split_missing_policy == "error":
            raise ValueError(
                "Fixed split file is missing LMDB complex_id values: {}".format(
                    lmdb_without_split[:10]
                )
            )
        if self.split_missing_policy == "intersection":
            merged = df.merge(
                split_df, on="complex_id", how="inner", validate="one_to_one"
            )
        elif self.split_missing_policy == "error":
            merged = df.merge(
                split_df, on="complex_id", how="left", validate="one_to_one"
            )
        else:
            raise ValueError(
                "Unknown split_missing_policy: {}".format(self.split_missing_policy)
            )

        self.split_report["rows_after_split_merge"] = len(merged)
        return merged

    def _build_dataset(self, df):
        return PDBbindLmdbGraphDataset(
            df=df,
            lmdb_dir=self.lmdb_dir,
            structure_root=self.structure_root,
            esm_key=self.esm_key,
            prot_emb_dim=self.prot_emb_dim,
            allow_zero_embedding=self.allow_zero_embedding,
            ligand_source=self.ligand_source,
            use_pocket_atom=self.use_pocket_atom,
            pocket_atom_cutoff=self.pocket_atom_cutoff,
            pocket_atom_edge_cutoff=self.pocket_atom_edge_cutoff,
            pocket_atom_include_hydrogen=self.pocket_atom_include_hydrogen,
            pocket_atom_include_hetatm=self.pocket_atom_include_hetatm,
            use_one_d_two_d=self.use_one_d_two_d,
            one_d_two_d_protein_edge_cutoff=self.one_d_two_d_protein_edge_cutoff,
            one_d_two_d_max_protein_residues=self.one_d_two_d_max_protein_residues,
            use_interface_pairformer=self.use_interface_pairformer,
            interface_pairformer_pocket_radius=self.interface_pairformer_pocket_radius,
            interface_pairformer_min_pocket_residues=self.interface_pairformer_min_pocket_residues,
            interface_pairformer_max_pocket_residues=self.interface_pairformer_max_pocket_residues,
            num_pos_emb=self.num_pos_emb,
            num_rbf=self.num_rbf,
            contact_cutoff=self.contact_cutoff,
            cache_graphs=self.cache_graphs,
        )

    def get_split(
        self,
        df=None,
        split_method=None,
        split_frac=None,
        seed=None,
        onthefly=None,
        return_df=False,
    ):
        del onthefly
        df = self.df if df is None else df
        split_method = split_method or self.split_method
        split_frac = split_frac or self.split_frac
        seed = seed or self.seed

        if split_method == "fixed":
            if "split" not in df.columns:
                raise ValueError("Fixed split requires --split_file")
            split_df = {
                split: df[df["split"] == split].reset_index(drop=True)
                for split in ["train", "valid", "test"]
            }
            empty_splits = [
                split for split, split_data in split_df.items()
                if len(split_data) == 0
            ]
            if empty_splits:
                raise ValueError("Fixed split has empty sets: {}".format(empty_splits))
        elif split_method == "random":
            split_df = create_fold(df, seed, split_frac)
        else:
            raise ValueError(
                "PDBbindLmdb supports split_method fixed or random, got {}".format(
                    split_method
                )
            )

        split_data = {
            split: self._build_dataset(split_frame)
            for split, split_frame in split_df.items()
        }
        if return_df:
            return split_data, split_df
        return split_data
