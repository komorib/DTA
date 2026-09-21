"""Unified manifests, LMDB-backed samples, and padding for staged V22."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch


DATASET_INDEX = {"LP_PDBBind": 0, "DAVIS": 1, "KIBA": 2}
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sequence(sequence):
    chains = []
    for chain in str(sequence).upper().split(":"):
        cleaned = "".join(character if character in AA_ALPHABET else "X" for character in chain if not character.isspace())
        if cleaned:
            chains.append(cleaned)
    if not chains:
        raise ValueError("protein sequence is empty after normalization")
    return "X".join(chains)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row or "dataset" not in row or "label" not in row:
                raise ValueError("invalid manifest row %d in %s" % (line_number, path))
            rows.append(row)
    if not rows:
        raise ValueError("manifest is empty: %s" % path)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def manifest_fingerprint(path):
    return sha256_file(path)


class _ReadOnlyLMDB:
    def __init__(self, path, subdir=True):
        self.path = str(Path(path).expanduser().resolve())
        self.subdir = bool(subdir)
        self.environment = None

    def get(self, key):
        if self.environment is None:
            self.environment = lmdb.open(
                self.path,
                subdir=self.subdir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=128,
            )
        with self.environment.begin(write=False) as transaction:
            raw = transaction.get(str(key).encode("utf-8"))
        if raw is None:
            raise KeyError("LMDB %s is missing key %s" % (self.path, key))
        return pickle.loads(bytes(raw))

    def __getstate__(self):
        state = dict(self.__dict__)
        state["environment"] = None
        return state


class OptionalStructureDataset(torch.utils.data.Dataset):
    """Loads only modalities declared by a manifest row."""

    def __init__(
        self,
        manifest_path,
        protein_cache,
        ligand_cache,
        with_structure=True,
        limit=None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            counts = defaultdict(int)
            selected = []
            for row in self.rows:
                dataset = str(row["dataset"])
                if counts[dataset] < int(limit):
                    selected.append(row)
                    counts[dataset] += 1
            self.rows = selected
        self.protein_store = _ReadOnlyLMDB(protein_cache)
        self.ligand_store = _ReadOnlyLMDB(ligand_cache)
        self.with_structure = bool(with_structure)
        self.structure_stores = {}
        self.dataset_to_indices = defaultdict(list)
        for index, row in enumerate(self.rows):
            dataset = str(row["dataset"])
            if dataset not in DATASET_INDEX:
                raise ValueError("unknown dataset in manifest: %s" % dataset)
            self.dataset_to_indices[dataset].append(index)

    def __len__(self):
        return len(self.rows)

    def _structure_store(self, path):
        path = str(Path(path).expanduser().resolve())
        if path not in self.structure_stores:
            self.structure_stores[path] = _ReadOnlyLMDB(path)
        return self.structure_stores[path]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["structure_stores"] = {}
        return state

    def __getitem__(self, index):
        row = self.rows[index]
        protein = self.protein_store.get(row["sequence_id"])
        ligand = self.ligand_store.get(row["ligand_id"])
        item = {
            "sample_id": str(row["sample_id"]),
            "dataset": str(row["dataset"]),
            "dataset_index": DATASET_INDEX[str(row["dataset"])],
            "label": float(row["label"]),
            "protein_embeddings": np.asarray(protein["embedding"], dtype=np.float32),
            "ligand_categorical": np.asarray(ligand["categorical"], dtype=np.int64),
            "ligand_continuous": np.asarray(ligand["continuous"], dtype=np.float32),
            "bond_edges": np.asarray(ligand["bond_edges"], dtype=np.int64),
            "bond_features": np.asarray(ligand["bond_features"], dtype=np.float32),
            "has_3d": False,
        }
        requested = bool(row.get("has_3d", False)) and self.with_structure
        if requested:
            structure = self._structure_store(row["structure_lmdb"]).get(row["structure_key"])
            if str(structure.get("key")) != str(row["structure_key"]):
                raise ValueError("structure cache lineage mismatch for %s" % row["sample_id"])
            item["has_3d"] = True
            item["structure"] = structure
        return item


def _select_structure(structure, maximum_pocket, maximum_ligand):
    distances = np.asarray(structure["residue_ligand_distance"], dtype=np.float32)
    ligand_count = min(int(distances.shape[1]), int(maximum_ligand))
    ligand_indices = np.arange(ligand_count, dtype=np.int64)
    if ligand_count == 0:
        raise ValueError("structured ligand contains no atoms")
    pocket_order = np.argsort(distances[:, ligand_indices].min(axis=1), kind="stable")
    pocket_indices = pocket_order[: int(maximum_pocket)]
    return pocket_indices, ligand_indices


class OptionalStructureCollator:
    def __init__(self, maximum_ligand_atoms=128, maximum_pocket_tokens=256):
        self.maximum_ligand_atoms = int(maximum_ligand_atoms)
        self.maximum_pocket_tokens = int(maximum_pocket_tokens)

    def __call__(self, items):
        if not items:
            raise ValueError("cannot collate an empty batch")
        batch_size = len(items)
        protein_width = int(items[0]["protein_embeddings"].shape[1])
        protein_count = max(int(item["protein_embeddings"].shape[0]) for item in items)
        ligand_count = max(
            min(int(item["ligand_categorical"].shape[0]), self.maximum_ligand_atoms)
            for item in items
        )
        protein_embeddings = torch.zeros(batch_size, protein_count, protein_width)
        protein_mask = torch.zeros(batch_size, protein_count, dtype=torch.bool)
        ligand_categorical = torch.zeros(batch_size, ligand_count, 8, dtype=torch.long)
        ligand_continuous = torch.zeros(batch_size, ligand_count, 4)
        ligand_mask = torch.zeros(batch_size, ligand_count, dtype=torch.bool)
        bond_features = torch.zeros(batch_size, ligand_count, ligand_count, 7)
        labels = torch.empty(batch_size)
        dataset_index = torch.empty(batch_size, dtype=torch.long)
        has_3d = torch.as_tensor([bool(item["has_3d"]) for item in items], dtype=torch.bool)
        sample_ids = []
        for batch_index, item in enumerate(items):
            pcount = int(item["protein_embeddings"].shape[0])
            lcount = min(int(item["ligand_categorical"].shape[0]), ligand_count)
            protein_embeddings[batch_index, :pcount] = torch.as_tensor(item["protein_embeddings"])
            protein_mask[batch_index, :pcount] = True
            ligand_categorical[batch_index, :lcount] = torch.as_tensor(
                item["ligand_categorical"][:lcount]
            )
            ligand_continuous[batch_index, :lcount] = torch.as_tensor(
                item["ligand_continuous"][:lcount]
            )
            ligand_mask[batch_index, :lcount] = True
            edges = item["bond_edges"]
            features = item["bond_features"]
            for edge_index in range(edges.shape[1]):
                source, target = int(edges[0, edge_index]), int(edges[1, edge_index])
                if source < lcount and target < lcount:
                    bond_features[batch_index, source, target, 0] = 1.0
                    bond_features[batch_index, source, target, 1:] = torch.as_tensor(
                        features[edge_index]
                    )
            labels[batch_index] = float(item["label"])
            dataset_index[batch_index] = int(item["dataset_index"])
            sample_ids.append(str(item["sample_id"]))
        batch = {
            "sample_ids": tuple(sample_ids),
            "labels": labels,
            "dataset_index": dataset_index,
            "has_3d": has_3d,
            "protein_embeddings": protein_embeddings,
            "protein_mask": protein_mask,
            "ligand_categorical": ligand_categorical,
            "ligand_continuous": ligand_continuous,
            "ligand_mask": ligand_mask,
            "bond_features": bond_features,
        }
        if not has_3d.any():
            return batch

        selections = {}
        for batch_index, item in enumerate(items):
            if item["has_3d"]:
                selections[batch_index] = _select_structure(
                    item["structure"], self.maximum_pocket_tokens, self.maximum_ligand_atoms
                )
        max_pocket = max(len(value[0]) for value in selections.values())
        max_structure_ligand = max(len(value[1]) for value in selections.values())
        batch.update(
            {
                "pocket_esm": torch.zeros(batch_size, max_pocket, 640),
                "pocket_scalar": torch.zeros(batch_size, max_pocket, 15),
                "pocket_aa": torch.zeros(batch_size, max_pocket, dtype=torch.long),
                "pocket_coords": torch.zeros(batch_size, max_pocket, 3),
                "pocket_mask": torch.zeros(batch_size, max_pocket, dtype=torch.bool),
                "structure_ligand_categorical": torch.zeros(
                    batch_size, max_structure_ligand, 19
                ),
                "structure_ligand_continuous": torch.zeros(
                    batch_size, max_structure_ligand, 2
                ),
                "structure_ligand_coords": torch.zeros(batch_size, max_structure_ligand, 3),
                "structure_ligand_mask": torch.zeros(
                    batch_size, max_structure_ligand, dtype=torch.bool
                ),
                "residue_ligand_distance": torch.full(
                    (batch_size, max_pocket, max_structure_ligand), 32.0
                ),
                "contact_label": torch.zeros(
                    batch_size, max_pocket, max_structure_ligand, dtype=torch.bool
                ),
            }
        )
        for batch_index, (pocket_indices, ligand_indices) in selections.items():
            structure = items[batch_index]["structure"]
            pcount, lcount = len(pocket_indices), len(ligand_indices)
            batch["pocket_esm"][batch_index, :pcount] = torch.as_tensor(
                np.asarray(structure["pocket_esm"])[pocket_indices], dtype=torch.float32
            )
            batch["pocket_scalar"][batch_index, :pcount] = torch.as_tensor(
                np.asarray(structure["pocket_scalar"])[pocket_indices], dtype=torch.float32
            )
            batch["pocket_aa"][batch_index, :pcount] = torch.as_tensor(
                np.asarray(structure["pocket_aa"])[pocket_indices], dtype=torch.long
            )
            batch["pocket_coords"][batch_index, :pcount] = torch.as_tensor(
                np.asarray(structure["pocket_coords"])[pocket_indices], dtype=torch.float32
            )
            batch["pocket_mask"][batch_index, :pcount] = True
            batch["structure_ligand_categorical"][batch_index, :lcount] = torch.as_tensor(
                np.asarray(structure["ligand_categorical"])[ligand_indices], dtype=torch.float32
            )
            batch["structure_ligand_continuous"][batch_index, :lcount] = torch.as_tensor(
                np.asarray(structure["ligand_continuous"])[ligand_indices], dtype=torch.float32
            )
            batch["structure_ligand_coords"][batch_index, :lcount] = torch.as_tensor(
                np.asarray(structure["ligand_coords"])[ligand_indices], dtype=torch.float32
            )
            batch["structure_ligand_mask"][batch_index, :lcount] = True
            selected_distance = np.asarray(structure["residue_ligand_distance"])[
                np.ix_(pocket_indices, ligand_indices)
            ]
            batch["residue_ligand_distance"][batch_index, :pcount, :lcount] = torch.as_tensor(
                selected_distance, dtype=torch.float32
            )
            contacts = structure.get("contact_label")
            if contacts is None:
                contacts = selected_distance <= 4.5
            else:
                contacts = np.asarray(contacts)[np.ix_(pocket_indices, ligand_indices)]
            batch["contact_label"][batch_index, :pcount, :lcount] = torch.as_tensor(
                contacts, dtype=torch.bool
            )
        return batch


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class DatasetBalancedBatchSampler(torch.utils.data.Sampler):
    """Homogeneous batches keep ranking losses dataset-local and balance large datasets."""

    def __init__(self, dataset, batch_size, steps_per_epoch, probabilities, seed=20260903):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.datasets = tuple(DATASET_INDEX)
        probability = np.asarray([float(probabilities[name]) for name in self.datasets])
        if np.any(probability < 0) or probability.sum() <= 0:
            raise ValueError("dataset sampling probabilities are invalid")
        self.probability = probability / probability.sum()
        self.indices = {
            name: np.asarray(dataset.dataset_to_indices[name], dtype=np.int64)
            for name in self.datasets
        }
        if any(values.size == 0 for values in self.indices.values()):
            raise ValueError("all configured datasets need at least one training row")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            name = self.datasets[int(generator.choice(len(self.datasets), p=self.probability))]
            yield generator.choice(self.indices[name], size=self.batch_size, replace=True).tolist()


__all__ = [
    "DATASET_INDEX",
    "DatasetBalancedBatchSampler",
    "OptionalStructureCollator",
    "OptionalStructureDataset",
    "canonical_sequence",
    "manifest_fingerprint",
    "move_batch_to_device",
    "read_jsonl",
    "sha256_file",
    "stable_id",
    "write_jsonl",
]
