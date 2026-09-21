from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from rdkit import Chem

from dta.external_benchmark_graphdta_kiba_v1.data import atom_features
from .model import PSICHICSequenceEdgeClassifier


def main():
    parser = argparse.ArgumentParser(description="CPU PSICHIC sequence-edge synthetic forward/backward smoke test.")
    parser.add_argument("--protein-length", type=int, default=32)
    args = parser.parse_args()
    smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    molecule = Chem.MolFromSmiles(smiles)
    ligand = np.stack([atom_features(atom) for atom in molecule.GetAtoms()]).astype(np.float32)
    edges = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
    protein_edges = []
    for distance in (1, 2):
        for index in range(args.protein_length - distance):
            protein_edges.extend(((index, index + distance), (index + distance, index)))
    item = {
        "protein_embedding": torch.randn(args.protein_length, 1280),
        "protein_edge_index": torch.tensor(protein_edges, dtype=torch.long).t().contiguous(),
        "ligand_features": torch.from_numpy(ligand),
        "ligand_edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "label": torch.tensor(1.0), "sample_id": "synthetic",
    }
    model = PSICHICSequenceEdgeClassifier().cpu()
    logit = model([item])[0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[None], item["label"][None])
    loss.backward()
    print(json.dumps({"status": "passed", "protein_nodes": args.protein_length, "ligand_nodes": int(ligand.shape[0]), "loss": float(loss.detach()), "logit_finite": bool(torch.isfinite(logit)), "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())}, sort_keys=True))


if __name__ == "__main__":
    main()

