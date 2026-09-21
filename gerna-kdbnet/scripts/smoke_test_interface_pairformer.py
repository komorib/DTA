#!/usr/bin/env python
"""Dependency-light geometry tests for the interface Pairformer.

Verifies:
  1. variable-length batching and finite forward/backward values;
  2. joint rigid translation and rotation invariance;
  3. sensitivity to a 0.05 A ligand displacement; and
  4. sensitivity to a side-chain-only heavy-atom displacement.
"""

from pathlib import Path
from types import SimpleNamespace
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.interface_pairformer import PocketLigandInterfacePairformer


def _normalise(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)


def _make_frames(backbone):
    n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    e1 = _normalise(c - ca)
    n_vec = n - ca
    e2 = _normalise(n_vec - (n_vec * e1).sum(-1, keepdim=True) * e1)
    e3 = _normalise(torch.cross(e1, e2, dim=-1))
    e2 = _normalise(torch.cross(e3, e1, dim=-1))
    return torch.stack([e1, e2, e3], dim=-2)


def _fake_graphs(esm_dim=32, ligand_dim=66):
    protein_counts = [24, 17]
    ligand_counts = [11, 8]
    protein_batch = torch.cat([
        torch.full((count,), index, dtype=torch.long)
        for index, count in enumerate(protein_counts)
    ])
    ligand_batch = torch.cat([
        torch.full((count,), index, dtype=torch.long)
        for index, count in enumerate(ligand_counts)
    ])

    total_p = sum(protein_counts)
    ca = torch.randn(total_p, 3) * 2.0
    n = ca + torch.tensor([-1.20, 0.55, 0.10])
    c = ca + torch.tensor([1.25, 0.35, -0.10])
    o = c + torch.tensor([0.25, 0.80, 0.15])
    backbone = torch.stack([n, ca, c, o], dim=1)
    frame = _make_frames(backbone)
    cb = ca + 1.52 * frame[:, 1]
    side = cb + 0.80 * frame[:, 2]

    max_atoms = 14
    atom_pos = torch.zeros(total_p, max_atoms, 3)
    atom_mask = torch.zeros(total_p, max_atoms, dtype=torch.bool)
    atom_pos[:, :4] = backbone
    atom_pos[:, 4] = cb
    atom_pos[:, 5] = side
    atom_mask[:, :6] = True

    protein = SimpleNamespace(
        x=ca,
        batch=protein_batch,
        seq=torch.randint(0, 21, (total_p,)),
        node_s=torch.randn(total_p, 6),
        seq_emb=torch.randn(total_p, esm_dim),
        backbone_pos=backbone,
        backbone_atom_mask=torch.ones(total_p, 4, dtype=torch.bool),
        local_frame=frame,
        local_frame_mask=torch.ones(total_p, dtype=torch.bool),
        pseudo_cb=cb,
        residue_cb=cb,
        residue_cb_mask=torch.ones(total_p, dtype=torch.bool),
        sidechain_center=side,
        sidechain_center_mask=torch.ones(total_p, dtype=torch.bool),
        residue_atom_pos=atom_pos,
        residue_atom_mask=atom_mask,
        has_full_atom_geometry=torch.ones(total_p, dtype=torch.bool),
    )

    ligand_positions = []
    offset = 0
    for protein_count, ligand_count in zip(protein_counts, ligand_counts):
        pocket_center = ca[offset : offset + min(6, protein_count)].mean(0)
        ligand_positions.append(pocket_center + torch.randn(ligand_count, 3) * 1.1)
        offset += protein_count
    ligand = SimpleNamespace(
        x=torch.cat(ligand_positions, dim=0),
        batch=ligand_batch,
        node_s=torch.randn(sum(ligand_counts), ligand_dim),
    )
    return protein, ligand


def _transform_protein(protein, rotation, shift):
    values = dict(protein.__dict__)
    for name in [
        "x", "backbone_pos", "pseudo_cb", "residue_cb", "sidechain_center",
        "residue_atom_pos",
    ]:
        tensor = values[name]
        values[name] = tensor @ rotation.T + shift
    # Frame rows are direction vectors and receive rotation but no translation.
    values["local_frame"] = values["local_frame"] @ rotation.T
    return SimpleNamespace(**values)


def _transform_ligand(ligand, rotation, shift):
    return SimpleNamespace(**{**ligand.__dict__, "x": ligand.x @ rotation.T + shift})


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    protein, ligand = _fake_graphs()
    model = PocketLigandInterfacePairformer(
        protein_esm_dim=32,
        ligand_scalar_dim=66,
        single_dim=48,
        pair_dim=24,
        output_dim=32,
        num_blocks=2,
        num_heads=4,
        max_pocket_residues=20,
        max_ligand_atoms=16,
        max_residue_atoms=14,
        min_pocket_residues=8,
        dropout=0.0,
    )

    model.train()
    output = model(protein, ligand)
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.eval()
    angle = 0.83
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    shift = torch.tensor([5.0, -3.0, 2.0])
    protein_transformed = _transform_protein(protein, rotation, shift)
    ligand_transformed = _transform_ligand(ligand, rotation, shift)

    ligand_subtle = SimpleNamespace(
        **{**ligand.__dict__, "x": ligand.x + torch.tensor([0.05, 0.0, 0.0])}
    )
    sidechain_changed_values = dict(protein.__dict__)
    sidechain_changed_values["residue_atom_pos"] = protein.residue_atom_pos.clone()
    sidechain_changed_values["sidechain_center"] = protein.sidechain_center.clone()
    # Move only one interface side-chain atom/centroid by 0.10 A.
    sidechain_changed_values["residue_atom_pos"][0, 5, 1] += 0.10
    sidechain_changed_values["sidechain_center"][0, 1] += 0.10
    protein_sidechain_changed = SimpleNamespace(**sidechain_changed_values)

    with torch.no_grad():
        reference = model(protein, ligand)
        rigidly_transformed = model(protein_transformed, ligand_transformed)
        subtly_shifted = model(protein, ligand_subtle)
        sidechain_shifted = model(protein_sidechain_changed, ligand)

    rigid_difference = (reference - rigidly_transformed).abs().max().item()
    subtle_difference = (reference - subtly_shifted).abs().max().item()
    sidechain_difference = (reference - sidechain_shifted).abs().max().item()
    assert rigid_difference < 2e-5, rigid_difference
    assert subtle_difference > 1e-7, subtle_difference
    assert sidechain_difference > 1e-8, sidechain_difference

    print("PASS")
    print("output shape:", tuple(output.shape))
    print("joint rigid-transform max difference:", rigid_difference)
    print("0.05 A ligand-shift max difference:", subtle_difference)
    print("0.10 A side-chain-shift max difference:", sidechain_difference)


if __name__ == "__main__":
    main()
