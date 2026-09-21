from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data


ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
    "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu",
    "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
SEQ_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_TO_ID = {char: idx + 1 for idx, char in enumerate(SEQ_ALPHABET)}
MAX_SEQUENCE_LENGTH = 1000
NUM_ATOM_FEATURES = 78


def _one_hot(value, allowed, unknown_last: bool = False) -> List[bool]:
    if value not in allowed:
        if not unknown_last:
            raise ValueError("value %r is outside GraphDTA feature vocabulary" % (value,))
        value = allowed[-1]
    return [value == item for item in allowed]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    result = (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS, unknown_last=True)
        + _one_hot(atom.GetDegree(), list(range(11)))
        + _one_hot(atom.GetTotalNumHs(), list(range(11)), unknown_last=True)
        + _one_hot(atom.GetImplicitValence(), list(range(11)), unknown_last=True)
        + [atom.GetIsAromatic()]
    )
    vector = np.asarray(result, dtype=np.float32)
    total = float(vector.sum())
    return vector / total if total else vector


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("RDKit could not parse SMILES: %s" % smiles)
    features = np.stack([atom_features(atom) for atom in molecule.GetAtoms()])
    edges: List[Tuple[int, int]] = []
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        edges.extend([(left, right), (right, left)])
    if not edges:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T
    return features, edge_index


def encode_sequence(sequence: str) -> np.ndarray:
    encoded = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
    for index, residue in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        encoded[index] = SEQ_TO_ID.get(residue, 0)
    return encoded


@dataclass(frozen=True)
class GraphRecord:
    smiles: str
    sequence: str
    label: float
    sample_id: str


class KIBAGraphDataset(Dataset):
    """Lazy GraphDTA dataset; unique molecular graphs are cached in-process."""

    def __init__(self, csv_path: str | Path, limit: int | None = None):
        frame = pd.read_csv(csv_path)
        required = {"compound_iso_smiles", "target_sequence", "label", "sample_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError("missing columns in %s: %s" % (csv_path, sorted(missing)))
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        self.records = [
            GraphRecord(str(row.compound_iso_smiles), str(row.target_sequence), float(row.label), str(row.sample_id))
            for row in frame.itertuples(index=False)
        ]
        self.graph_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        record = self.records[index]
        graph = self.graph_cache.get(record.smiles)
        if graph is None:
            graph = smiles_to_graph(record.smiles)
            self.graph_cache[record.smiles] = graph
        features, edge_index = graph
        return Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(edge_index),
            # Keep a leading singleton dimension so PyG concatenates targets
            # along the batch dimension rather than flattening B x L to B*L.
            target=torch.from_numpy(encode_sequence(record.sequence)).unsqueeze(0),
            y=torch.tensor([record.label], dtype=torch.float32),
            sample_index=torch.tensor([index], dtype=torch.long),
        )
