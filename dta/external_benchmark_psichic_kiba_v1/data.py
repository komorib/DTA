from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import stable_id


class PSICHICKIBADataset(Dataset):
    def __init__(self, csv_path: str | Path, feature_root: str | Path, limit: int | None = None):
        frame = pd.read_csv(csv_path)
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        self.rows = frame.to_dict("records")
        self.feature_root = Path(feature_root).expanduser().resolve()
        self.protein_cache, self.ligand_cache = {}, {}

    def __len__(self):
        return len(self.rows)

    def _load(self, directory, key, cache):
        if key not in cache:
            cache[key] = torch.load(self.feature_root / directory / (stable_id(key) + ".pt"), map_location="cpu")
        return cache[key]

    def __getitem__(self, index):
        row = self.rows[index]
        protein = self._load("proteins", str(row["Protein"]), self.protein_cache)
        ligand = self._load("ligands", str(row["Ligand"]), self.ligand_cache)
        return {
            "protein_embedding": torch.as_tensor(protein["embedding"], dtype=torch.float32),
            "protein_edge_index": torch.as_tensor(protein["edge_index"], dtype=torch.long),
            "ligand_features": torch.as_tensor(ligand["atom_features"], dtype=torch.float32),
            "ligand_edge_index": torch.as_tensor(ligand["edge_index"], dtype=torch.long),
            "label": torch.tensor(float(row["classification_label"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
        }


def list_collate(items):
    return items

