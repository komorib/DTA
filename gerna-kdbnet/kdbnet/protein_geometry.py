"""Residue-level all-atom geometry helpers for the interface Pairformer.

The original KDBNet LMDB stores four backbone atoms (N, CA, C, O) per
residue.  PDBbind manifests normally also point to a ``*_pocket.pdb`` file.
This module attaches the pocket heavy atoms to the corresponding full-protein
residue nodes by matching C-alpha coordinates.  The resulting fixed-size node
attributes survive PyG batching and let the local interface branch use:

* real residue heavy atoms when a pocket PDB is available;
* real side-chain centroids and C-beta coordinates; and
* a backbone-only fallback otherwise.

No ligand-dependent target is introduced.  These are input geometry features.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from kdbnet.constants import NUM_TO_LETTER
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
PREFERRED_ATOM_ORDER = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "NZ", "CZ", "CZ2",
    "CZ3", "CH2", "NH1", "NH2", "OH", "OXT",
]


def _infer_element(atom_name: str, line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    for char in atom_name:
        if char.isalpha():
            return char.upper()
    return ""


def parse_pdb_residues(
    pdb_path,
    include_hydrogen: bool = False,
) -> List[Dict[str, object]]:
    """Parse standard protein residues and retain heavy-atom coordinates.

    Alternate locations other than blank/A are ignored.  HETATM records are
    intentionally excluded: metals/cofactors belong in a separate interface
    representation and should not be silently assigned to an amino acid.
    """
    residues: "OrderedDict[Tuple[str, int, str], Dict[str, object]]" = OrderedDict()
    with open(Path(pdb_path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip().upper()
            element = _infer_element(atom_name, line)
            if not include_hydrogen and element == "H":
                continue
            try:
                chain_id = line[21].strip() or "_"
                residue_number = int(line[22:26])
                insertion_code = line[26].strip()
                coord = torch.tensor(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=torch.float32,
                )
            except (ValueError, IndexError):
                continue
            residue_name = line[17:20].strip().upper()
            aa = THREE_TO_ONE.get(residue_name, "X")
            key = (chain_id, residue_number, insertion_code)
            record = residues.setdefault(
                key,
                {
                    "key": key,
                    "aa": aa,
                    "resname": residue_name,
                    "atoms": OrderedDict(),
                },
            )
            # Keep the first accepted conformer for duplicate atom names.
            record["atoms"].setdefault(atom_name, coord)
    return list(residues.values())


def _ordered_atom_names(atoms: Dict[str, torch.Tensor]) -> List[str]:
    rank = {name: index for index, name in enumerate(PREFERRED_ATOM_ORDER)}
    return sorted(atoms.keys(), key=lambda name: (rank.get(name, 10_000), name))


def attach_pocket_residue_geometry(
    protein_graph,
    pocket_pdb,
    max_atoms_per_residue: int = 14,
    match_tolerance: float = 1.5,
) -> Dict[str, int]:
    """Attach pocket heavy atoms to residue nodes of ``protein_graph``.

    Mapping is based on nearest C-alpha coordinates, with amino-acid identity
    used as a preference rather than a hard requirement.  PDBbind pocket files
    are in the same coordinate frame as the full protein, so this is robust to
    missing chain/residue metadata in the serialized LMDB.

    Existing backbone fallback tensors on ``protein_graph`` are preserved for
    residues that cannot be matched.
    """
    if max_atoms_per_residue < 5:
        raise ValueError("max_atoms_per_residue must be at least 5")
    if not hasattr(protein_graph, "residue_atom_pos"):
        raise ValueError("protein_graph must contain residue_atom_pos fallback geometry")

    residues = parse_pdb_residues(pocket_pdb)
    if not residues:
        return {"parsed_residues": 0, "matched_residues": 0, "unmatched_residues": 0}

    graph_ca = protein_graph.x.detach().cpu().float()
    graph_seq = protein_graph.seq.detach().cpu().long()
    used = torch.zeros(graph_ca.size(0), dtype=torch.bool)

    atom_pos = protein_graph.residue_atom_pos.detach().cpu().clone()
    atom_mask = protein_graph.residue_atom_mask.detach().cpu().clone()
    sidechain_center = protein_graph.sidechain_center.detach().cpu().clone()
    sidechain_mask = protein_graph.sidechain_center_mask.detach().cpu().clone()
    real_cb = protein_graph.residue_cb.detach().cpu().clone()
    real_cb_mask = protein_graph.residue_cb_mask.detach().cpu().clone()
    full_atom_mask = protein_graph.has_full_atom_geometry.detach().cpu().clone()
    matched = 0

    for residue in residues:
        atoms = residue["atoms"]
        if "CA" not in atoms:
            continue
        ca = atoms["CA"].float()
        distances = torch.linalg.norm(graph_ca - ca.unsqueeze(0), dim=-1)

        aa = str(residue.get("aa", "X"))
        aa_match = torch.tensor(
            [NUM_TO_LETTER.get(int(value), "X") == aa for value in graph_seq],
            dtype=torch.bool,
        )
        preferred = (~used) & aa_match
        candidate_mask = preferred if preferred.any() else ~used
        if not candidate_mask.any():
            break
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        candidate_distances = distances[candidate_indices]
        best_offset = int(torch.argmin(candidate_distances).item())
        graph_index = int(candidate_indices[best_offset].item())
        if float(distances[graph_index].item()) > float(match_tolerance):
            continue

        names = _ordered_atom_names(atoms)[: int(max_atoms_per_residue)]
        atom_pos[graph_index].zero_()
        atom_mask[graph_index].zero_()
        for atom_offset, atom_name in enumerate(names):
            atom_pos[graph_index, atom_offset] = atoms[atom_name]
            atom_mask[graph_index, atom_offset] = True

        side_names = [name for name in names if name not in BACKBONE_ATOMS]
        if side_names:
            side_coords = torch.stack([atoms[name] for name in side_names], dim=0)
            sidechain_center[graph_index] = side_coords.mean(dim=0)
            sidechain_mask[graph_index] = True
        if "CB" in atoms:
            real_cb[graph_index] = atoms["CB"]
            real_cb_mask[graph_index] = True

        full_atom_mask[graph_index] = True
        used[graph_index] = True
        matched += 1

    protein_graph.residue_atom_pos = atom_pos.to(protein_graph.x.device)
    protein_graph.residue_atom_mask = atom_mask.to(protein_graph.x.device)
    protein_graph.sidechain_center = sidechain_center.to(protein_graph.x.device)
    protein_graph.sidechain_center_mask = sidechain_mask.to(protein_graph.x.device)
    protein_graph.residue_cb = real_cb.to(protein_graph.x.device)
    protein_graph.residue_cb_mask = real_cb_mask.to(protein_graph.x.device)
    protein_graph.has_full_atom_geometry = full_atom_mask.to(protein_graph.x.device)
    return {
        "parsed_residues": len(residues),
        "matched_residues": matched,
        "unmatched_residues": len(residues) - matched,
    }


def residue_min_ligand_distance(protein_graph, ligand_positions: torch.Tensor) -> torch.Tensor:
    """Return one minimum heavy-atom-to-ligand distance per protein residue."""
    if hasattr(protein_graph, "residue_atom_pos") and hasattr(protein_graph, "residue_atom_mask"):
        atom_pos = protein_graph.residue_atom_pos.float()
        atom_mask = protein_graph.residue_atom_mask.bool()
        # [R,A,L]
        distance = torch.linalg.norm(
            atom_pos[:, :, None, :] - ligand_positions.float()[None, None, :, :],
            dim=-1,
        )
        distance = distance.masked_fill(~atom_mask[:, :, None], float("inf"))
        nearest = distance.amin(dim=(1, 2))
        fallback = torch.cdist(protein_graph.x.float(), ligand_positions.float()).amin(dim=1)
        return torch.where(torch.isfinite(nearest), nearest, fallback)
    return torch.cdist(protein_graph.x.float(), ligand_positions.float()).amin(dim=1)
