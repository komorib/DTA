"""Ligand-proximal pocket atom graph featurization."""

import numpy as np
import torch
import torch_geometric
from rdkit import Chem

from kdbnet.pdb_graph import _normalize, _rbf


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
ELEMENT_VOCAB = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "SE",
    "F",
    "CL",
    "BR",
    "I",
    "FE",
    "ZN",
    "MG",
    "MN",
    "CA",
    "NA",
    "K",
    "OTHER",
]
ATOM_GROUP_VOCAB = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "BACKBONE_OTHER",
    "SIDECHAIN",
]
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def pocket_atom_node_dim(num_rbf=16):
    return (
        len(ELEMENT_VOCAB)
        + len(AA_VOCAB)
        + len(ATOM_GROUP_VOCAB)
        + 1
        + int(num_rbf)
    )


def _one_hot(value, vocab, default):
    value = str(value).upper()
    if value not in vocab:
        value = default
    return [1.0 if value == item else 0.0 for item in vocab]


def _infer_pdb_element(atom_name, line):
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    for char in atom_name.strip():
        if char.isalpha():
            return char.upper()
    return "OTHER"


def _atom_group(atom_name):
    atom_name = str(atom_name).upper()
    if atom_name in {"N", "CA", "C", "O", "CB"}:
        return atom_name
    if atom_name in BACKBONE_ATOMS:
        return "BACKBONE_OTHER"
    return "SIDECHAIN"


def _parse_pocket_atoms(pocket_pdb, include_hydrogen=False, include_hetatm=False):
    atoms = []
    with open(pocket_pdb, "r") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ATOM":
                pass
            elif record == "HETATM" and include_hetatm:
                pass
            else:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue

            atom_name = line[12:16].strip().upper()
            element = _infer_pdb_element(atom_name, line)
            if not include_hydrogen and element == "H":
                continue

            try:
                coord = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            except ValueError:
                continue

            resname = line[17:20].strip().upper()
            aa = THREE_TO_ONE.get(resname, "X")
            atoms.append({
                "atom_name": atom_name,
                "element": element if element in ELEMENT_VOCAB else "OTHER",
                "aa": aa,
                "coord": coord,
                "is_backbone": atom_name in BACKBONE_ATOMS,
            })
    return atoms


def _load_ligand_coords(ligand_sdf, include_hydrogen=False):
    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=not include_hydrogen)
    if mol is None:
        raise ValueError("RDKit failed to parse ligand SDF: {}".format(ligand_sdf))
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if not include_hydrogen and atom.GetAtomicNum() == 1:
            continue
        point = conf.GetAtomPosition(atom.GetIdx())
        coords.append([point.x, point.y, point.z])
    if not coords:
        raise ValueError("Ligand has no usable atoms: {}".format(ligand_sdf))
    return np.asarray(coords, dtype=np.float32)


def _build_node_features(contact_atoms, min_dists, cutoff, num_rbf):
    features = []
    dist_tensor = torch.as_tensor(min_dists, dtype=torch.float32)
    dist_rbf = _rbf(dist_tensor, D_max=float(cutoff), D_count=int(num_rbf)).numpy()
    for atom, dist_feat in zip(contact_atoms, dist_rbf):
        item = []
        item.extend(_one_hot(atom["element"], ELEMENT_VOCAB, "OTHER"))
        item.extend(_one_hot(atom["aa"], AA_VOCAB, "X"))
        item.extend(_one_hot(_atom_group(atom["atom_name"]), ATOM_GROUP_VOCAB, "SIDECHAIN"))
        item.append(1.0 if atom["is_backbone"] else 0.0)
        item.extend(dist_feat.tolist())
        features.append(item)
    return torch.as_tensor(features, dtype=torch.float32)


def _build_edges(coords, edge_cutoff=4.5, num_rbf=16):
    dist_mat = torch.cdist(coords, coords)
    row, col = torch.where((dist_mat < float(edge_cutoff)) & (dist_mat > 1e-5))
    if row.numel() == 0 and coords.shape[0] > 1:
        masked = dist_mat + torch.eye(coords.shape[0], dtype=coords.dtype) * 1e6
        nearest = torch.argmin(masked, dim=1)
        row = torch.arange(coords.shape[0], dtype=torch.long)
        col = nearest.to(torch.long)
        row = torch.cat([row, col], dim=0)
        col = torch.cat([col, row[: coords.shape[0]]], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_vec = coords[edge_index[0]] - coords[edge_index[1]]
    edge_s = _rbf(edge_vec.norm(dim=-1), D_max=float(edge_cutoff), D_count=int(num_rbf))
    edge_v = _normalize(edge_vec).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_index, edge_s, edge_v


def featurize_pocket_atom_graph(
    pocket_pdb,
    ligand_sdf,
    name=None,
    contact_cutoff=5.0,
    edge_cutoff=4.5,
    num_rbf=16,
    include_hydrogen=False,
    include_hetatm=False,
):
    pocket_atoms = _parse_pocket_atoms(
        pocket_pdb,
        include_hydrogen=include_hydrogen,
        include_hetatm=include_hetatm,
    )
    if not pocket_atoms:
        raise ValueError("Pocket PDB has no usable atoms: {}".format(pocket_pdb))

    ligand_coords = _load_ligand_coords(ligand_sdf, include_hydrogen=include_hydrogen)
    protein_coords = np.asarray([atom["coord"] for atom in pocket_atoms], dtype=np.float32)
    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=-1,
    )
    min_distances = distances.min(axis=1)
    contact_mask = min_distances <= float(contact_cutoff)
    if not np.any(contact_mask):
        raise ValueError(
            "No pocket atoms within {} A for {}".format(contact_cutoff, name)
        )

    contact_atoms = [
        atom for atom, keep in zip(pocket_atoms, contact_mask) if keep
    ]
    contact_distances = min_distances[contact_mask]
    contact_coords = protein_coords[contact_mask]
    ligand_centroid = ligand_coords.mean(axis=0, keepdims=True)
    coords = torch.as_tensor(contact_coords - ligand_centroid, dtype=torch.float32)
    node_s = _build_node_features(
        contact_atoms,
        contact_distances,
        cutoff=contact_cutoff,
        num_rbf=num_rbf,
    )
    edge_index, edge_s, edge_v = _build_edges(
        coords,
        edge_cutoff=edge_cutoff,
        num_rbf=num_rbf,
    )

    return torch_geometric.data.Data(
        x=coords,
        edge_index=edge_index,
        name=name,
        node_v=coords.unsqueeze(1),
        node_s=node_s,
        edge_v=edge_v,
        edge_s=edge_s,
        contact_dist=torch.as_tensor(contact_distances, dtype=torch.float32),
        num_contact_atoms=torch.tensor([len(contact_atoms)], dtype=torch.long),
    )
