"""Dataset and padding for compact MSA-pocket complex caches."""

import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch


class MSAPocketComplexDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, cache_dir, limit=None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.split = str(manifest["split"])
        self.samples = list(manifest["samples"])
        if limit is not None:
            self.samples = self.samples[: int(limit)]
        if not self.samples:
            raise ValueError("MSA pocket dataset is empty")
        self.keys = tuple(str(sample["lmdb_key"]) for sample in self.samples)
        self.labels = np.asarray([float(sample["label"]) for sample in self.samples], dtype=np.float64)
        if len(self.keys) != len(set(self.keys)) or not np.isfinite(self.labels).all():
            raise ValueError("MSA pocket manifest keys or labels are invalid")
        self._env = None

    def _environment(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.cache_dir / "complexes.lmdb"),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
        return self._env

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        raw = self._environment().begin(write=False).get(key.encode("utf-8"))
        if raw is None:
            raise KeyError(f"MSA pocket cache is missing {key}")
        item = pickle.loads(bytes(raw))
        if str(item["key"]) != key or abs(float(item["label"]) - self.labels[index]) > 1e-5:
            raise ValueError(f"MSA pocket cache lineage mismatch for {key}")
        return item


def _allocate(batch_size, maximum, tail, dtype, fill=0):
    return torch.full((batch_size, maximum) + tuple(tail), fill, dtype=dtype)


def collate_msa_pocket_complexes(items):
    if not items:
        raise ValueError("cannot collate an empty MSA pocket batch")
    batch_size = len(items)
    max_pocket = max(int(item["pocket_msa"].shape[0]) for item in items)
    max_ligand = max(int(item["ligand_coords"].shape[0]) for item in items)
    pocket_msa = _allocate(batch_size, max_pocket, (50,), torch.float32)
    pocket_esm = _allocate(batch_size, max_pocket, (640,), torch.float32)
    pocket_scalar = _allocate(batch_size, max_pocket, (15,), torch.float32)
    pocket_aa = _allocate(batch_size, max_pocket, (), torch.long)
    pocket_coords = _allocate(batch_size, max_pocket, (3,), torch.float32)
    pocket_chain = _allocate(batch_size, max_pocket, (), torch.long, fill=-1)
    pocket_position = _allocate(batch_size, max_pocket, (), torch.long, fill=-1)
    pocket_mask = _allocate(batch_size, max_pocket, (), torch.bool)
    ligand_categorical = _allocate(batch_size, max_ligand, (19,), torch.float32)
    ligand_continuous = _allocate(batch_size, max_ligand, (2,), torch.float32)
    ligand_coords = _allocate(batch_size, max_ligand, (3,), torch.float32)
    ligand_mask = _allocate(batch_size, max_ligand, (), torch.bool)
    residue_ligand_distance = _allocate(
        batch_size, max_pocket, (max_ligand,), torch.float32, fill=32.0
    )
    contact_label = _allocate(batch_size, max_pocket, (max_ligand,), torch.bool)
    bond_features = torch.zeros(
        (batch_size, max_ligand, max_ligand, 7), dtype=torch.float32
    )
    labels = torch.empty(batch_size, dtype=torch.float32)
    keys = []
    for batch_index, item in enumerate(items):
        pocket_count = int(item["pocket_msa"].shape[0])
        ligand_count = int(item["ligand_coords"].shape[0])
        pocket_msa[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_msa"], dtype=torch.float32
        )
        pocket_esm[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_esm"], dtype=torch.float32
        )
        pocket_scalar[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_scalar"], dtype=torch.float32
        )
        pocket_aa[batch_index, :pocket_count] = torch.as_tensor(item["pocket_aa"], dtype=torch.long)
        pocket_coords[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_coords"], dtype=torch.float32
        )
        pocket_chain[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_chain_index"], dtype=torch.long
        )
        pocket_position[batch_index, :pocket_count] = torch.as_tensor(
            item["pocket_sequence_position"], dtype=torch.long
        )
        pocket_mask[batch_index, :pocket_count] = True
        ligand_categorical[batch_index, :ligand_count] = torch.as_tensor(
            item["ligand_categorical"], dtype=torch.float32
        )
        ligand_continuous[batch_index, :ligand_count] = torch.as_tensor(
            item["ligand_continuous"], dtype=torch.float32
        )
        ligand_coords[batch_index, :ligand_count] = torch.as_tensor(
            item["ligand_coords"], dtype=torch.float32
        )
        ligand_mask[batch_index, :ligand_count] = True
        residue_ligand_distance[batch_index, :pocket_count, :ligand_count] = torch.as_tensor(
            item["residue_ligand_distance"], dtype=torch.float32
        )
        contact_label[batch_index, :pocket_count, :ligand_count] = torch.as_tensor(
            item["contact_label"], dtype=torch.bool
        )
        edges = np.asarray(item["bond_edges"], dtype=np.int64)
        edge_features = torch.as_tensor(item["bond_features"], dtype=torch.float32)
        for edge_index in range(edges.shape[1]):
            source, target = int(edges[0, edge_index]), int(edges[1, edge_index])
            bond_features[batch_index, source, target, 0] = 1.0
            bond_features[batch_index, source, target, 1:] = edge_features[edge_index]
        labels[batch_index] = float(item["label"])
        keys.append(str(item["key"]))
    return {
        "keys": tuple(keys),
        "labels": labels,
        "pocket_msa": pocket_msa,
        "pocket_esm": pocket_esm,
        "pocket_scalar": pocket_scalar,
        "pocket_aa": pocket_aa,
        "pocket_coords": pocket_coords,
        "pocket_chain": pocket_chain,
        "pocket_position": pocket_position,
        "pocket_mask": pocket_mask,
        "ligand_categorical": ligand_categorical,
        "ligand_continuous": ligand_continuous,
        "ligand_coords": ligand_coords,
        "ligand_mask": ligand_mask,
        "residue_ligand_distance": residue_ligand_distance,
        "contact_label": contact_label,
        "contact_mask": pocket_mask[:, :, None] & ligand_mask[:, None, :],
        "bond_features": bond_features,
    }


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


__all__ = [
    "MSAPocketComplexDataset",
    "collate_msa_pocket_complexes",
    "move_batch_to_device",
]
