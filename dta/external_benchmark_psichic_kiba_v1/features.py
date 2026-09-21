from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from dta.external_benchmark_graphdta_kiba_v1.data import atom_features as graphdta_atom_features


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_unique_inputs(data_root: str | Path) -> Tuple[List[str], List[str]]:
    root = Path(data_root)
    proteins, ligands = set(), set()
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(root / "psichic" / (split + ".csv"))
        proteins.update(frame["Protein"].astype(str).tolist())
        ligands.update(frame["Ligand"].astype(str).tolist())
    return sorted(proteins), sorted(ligands)


def _sequence_edges(length: int) -> np.ndarray:
    edges = []
    for distance in (1, 2):
        for index in range(max(0, length - distance)):
            edges.extend(((index, index + distance), (index + distance, index)))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def build_ligand_record(smiles: str) -> Dict[str, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("RDKit could not parse ligand: %s" % smiles)
    atom = np.stack([graphdta_atom_features(item) for item in molecule.GetAtoms()]).astype(np.float32)
    edges = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    return {"atom_features": atom, "edge_index": edge_index, "smiles": smiles}


def write_ligand_cache(data_root: str | Path, output: str | Path) -> Dict:
    _, ligands = read_unique_inputs(data_root)
    output = Path(output).expanduser().resolve()
    directory = output / "ligands"
    directory.mkdir(parents=True, exist_ok=True)
    for smiles in ligands:
        torch.save(build_ligand_record(smiles), directory / (stable_id(smiles) + ".pt"))
    return {"count": len(ligands), "directory": str(directory)}


def _windows(sequence: str, window_size: int = 1022, overlap: int = 128):
    starts = list(range(0, max(len(sequence) - overlap, 1), window_size - overlap))
    for start in starts:
        stop = min(start + window_size, len(sequence))
        yield start, sequence[start:stop]
        if stop == len(sequence):
            break


def build_protein_cache(data_root: str | Path, output: str | Path, model_name: str,
                        device: str, window_size: int = 1022, overlap: int = 128,
                        precision: str = "fp16", overwrite: bool = False) -> Dict:
    proteins, _ = read_unique_inputs(data_root)
    output = Path(output).expanduser().resolve()
    directory = output / "proteins"
    directory.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    autocast_enabled = str(device).startswith("cuda") and precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    for sequence in proteins:
        target_path = directory / (stable_id(sequence) + ".pt")
        if target_path.is_file() and not overwrite:
            print("protein_cached_existing=%s" % target_path.name, flush=True)
            continue
        # Overlap-stitch each window back to the original residue axis.
        # ``counts`` must exist before the first window is accumulated.
        counts = np.zeros((len(sequence),), dtype=np.float32)
        accum = None
        for start, window in _windows(sequence, window_size, overlap):
            encoded = tokenizer(window, return_tensors="pt", add_special_tokens=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                hidden = model(**encoded).last_hidden_state[0, 1 : 1 + len(window)].float().cpu().numpy()
            if accum is None:
                accum = np.zeros((len(sequence), hidden.shape[1]), dtype=np.float32)
            accum[start : start + len(window)] += hidden
            counts[start : start + len(window)] += 1.0
        embedding = (accum / counts[:, None].clip(1.0)).astype(np.float16)
        record = {
            "sequence": sequence,
            "embedding": embedding,
            "edge_index": _sequence_edges(len(sequence)),
            "edge_weight": np.ones((_sequence_edges(len(sequence)).shape[1],), dtype=np.float32),
            "embedding_model": model_name,
            "edge_policy": "sequence_offsets_plus_minus_1_and_2",
        }
        torch.save(record, target_path)
        print("protein_cached=%d/%d" % (len(list(directory.glob("*.pt"))), len(proteins)), flush=True)
    metadata = {
        "version": "psichic-sequence-edge-esm-cache-v1",
        "count": len(proteins),
        "embedding_model": model_name,
        "embedding_width": int(model.config.hidden_size),
        "edge_policy": "sequence_offsets_plus_minus_1_and_2",
        "contact_map_used": False,
        "data_root": str(Path(data_root).expanduser().resolve()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
