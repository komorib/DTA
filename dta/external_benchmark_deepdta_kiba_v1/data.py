from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MAX_SMILES_LENGTH = 100
MAX_SEQUENCE_LENGTH = 1000
PROTEIN_ALPHABET = "ACBDEGF IHKMLONQPSRUTWVYXZ".replace(" ", "")
PROTEIN_TO_ID: Dict[str, int] = {char: idx + 1 for idx, char in enumerate(PROTEIN_ALPHABET)}
# DeepDTA's published isomeric-SMILES vocabulary (index zero is padding/unknown).
SMILES_ALPHABET = "#%)(+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\_abcdefghijklmnopqrstuvwxyz"
SMILES_TO_ID: Dict[str, int] = {char: idx + 1 for idx, char in enumerate(dict.fromkeys(SMILES_ALPHABET))}


def encode(text: str, mapping: Dict[str, int], length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.int64)
    for index, char in enumerate(text[:length]):
        output[index] = mapping.get(char, 0)
    return output


class KIBADataset(Dataset):
    def __init__(self, csv_path: str | Path, limit: int | None = None):
        frame = pd.read_csv(csv_path)
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        self.smiles = frame["compound_iso_smiles"].astype(str).tolist()
        self.sequences = frame["target_sequence"].astype(str).tolist()
        self.sample_ids = frame["sample_id"].astype(str).tolist()
        self.labels = frame["label"].astype(np.float32).to_numpy()
        self.drug = np.stack([encode(value, SMILES_TO_ID, MAX_SMILES_LENGTH) for value in self.smiles])
        self.protein = np.stack([encode(value, PROTEIN_TO_ID, MAX_SEQUENCE_LENGTH) for value in self.sequences])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {
            "drug": torch.from_numpy(self.drug[index]),
            "protein": torch.from_numpy(self.protein[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }
