import argparse
import io
import json
import os
import pickle
import re
import sys
import tarfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

import utils.features as fea

RDLogger.DisableLog("rdApp.warning")


AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_LIST)}

PROTEIN_ATOM5_NAMES = ("N", "CA", "C", "O", "CB")
PROTEIN_ATOM5_TOKEN = {name: idx + 1 for idx, name in enumerate(PROTEIN_ATOM5_NAMES)}
PROTEIN_ATOM5_NUM_TOKENS = len(PROTEIN_ATOM5_TOKEN) + 1


ATOM_ID = {
    "C": 1,
    "N": 2,
    "Br": 3,
    "O": 4,
    "S": 5,
    "Cl": 6,
    "F": 7,
    "P": 8,
    "I": 9,
    "As": 9,
    "K": 9,
    "Na": 9,
    "Ge": 9,
    "Se": 9,
    "Au": 9,
    "Pt": 9,
    "Sr": 9,
    "Mg": 9,
    "B": 9,
    "Co": 9,
    "Ca": 9,
    "Rh": 9,
    "unknown": 9,
}

UNIT_TO_MOLAR = {
    "fM": 1e-15,
    "pM": 1e-12,
    "nM": 1e-9,
    "uM": 1e-6,
    "mM": 1e-3,
    "M": 1.0,
}


class BindingRecord:
    def __init__(self, pdb_id, binding_type, operator, value_molar, raw):
        self.pdb_id = pdb_id
        self.binding_type = binding_type
        self.operator = operator
        self.value_molar = value_molar
        self.raw = raw


def parse_binding_record(line):
    parts = line.split()
    if len(parts) < 4 or line.startswith("#"):
        return None

    pdb_id = parts[0].lower()
    raw = parts[3]
    match = re.match(
        r"^(?P<kind>Kd|Ki|IC50|EC50)(?P<op>[<>=~]*)"
        r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>fM|pM|nM|uM|mM|M)$",
        raw,
    )
    if not match:
        return None

    value_molar = float(match.group("value")) * UNIT_TO_MOLAR[match.group("unit")]
    operator = match.group("op") or "="
    if operator == "~":
        operator = "="

    return BindingRecord(
        pdb_id=pdb_id,
        binding_type=match.group("kind"),
        operator=operator,
        value_molar=value_molar,
        raw=raw,
    )


def classify_binding(record, positive_threshold_molar, negative_threshold_molar):
    # Ki/Kd/IC50 越小通常表示结合越强；带不等号的样本只在标签确定时保留。
    if record.operator in ("=", "=="):
        if record.value_molar <= positive_threshold_molar:
            return 1
        if record.value_molar >= negative_threshold_molar:
            return 0
        return None

    if record.operator in ("<", "<="):
        if record.value_molar <= positive_threshold_molar:
            return 1
        return None

    if record.operator in (">", ">="):
        if record.value_molar >= negative_threshold_molar:
            return 0
        return None

    return None


def load_index_records(index_tar_path, positive_threshold_molar, negative_threshold_molar):
    records = []
    with tarfile.open(index_tar_path, "r:gz") as tar:
        index_file = tar.extractfile("index/INDEX_general_PL.2020R1.lst")
        if index_file is None:
            raise FileNotFoundError("index/INDEX_general_PL.2020R1.lst not found")
        for raw_line in index_file:
            line = raw_line.decode("utf-8", errors="replace").strip()
            record = parse_binding_record(line)
            if record is None:
                continue
            label = classify_binding(
                record,
                positive_threshold_molar=positive_threshold_molar,
                negative_threshold_molar=negative_threshold_molar,
            )
            if label is None:
                continue
            records.append((record, label))
    return records


def build_tar_member_index(pl_tar_path):
    pocket_members = {}
    sdf_members = {}
    mol2_members = {}

    with tarfile.open(pl_tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            base = os.path.basename(name)
            if not member.isfile() or "_" not in base:
                continue
            pdb_id = base.split("_", 1)[0].lower()
            if base.endswith("_pocket.pdb"):
                pocket_members[pdb_id] = name
            elif base.endswith("_ligand.sdf"):
                sdf_members[pdb_id] = name
            elif base.endswith("_ligand.mol2"):
                mol2_members[pdb_id] = name

    return pocket_members, sdf_members, mol2_members


def residue_embedding(aa, position, length, dim=640):
    # 第一版不用 ESM，先做 640 维占位特征以兼容 model.py 的 input_dim_rna=640。
    vector = np.zeros(dim, dtype=np.float32)
    aa_index = AA_LIST.index(aa) if aa in AA_LIST else len(AA_LIST)
    vector[aa_index] = 1.0
    vector[32] = float(position) / max(length - 1, 1)
    vector[33] = float(length) / 512.0
    vector[34] = 1.0 if aa in {"D", "E"} else 0.0
    vector[35] = 1.0 if aa in {"K", "R", "H"} else 0.0
    vector[36] = 1.0 if aa in {"F", "W", "Y"} else 0.0
    vector[37] = 1.0 if aa in {"A", "V", "I", "L", "M"} else 0.0
    return vector


def protein_graph_edges(ca_coords, distance_cutoff):
    edges = set()
    n_res = len(ca_coords)

    for idx in range(n_res - 1):
        edges.add((idx, idx + 1))
        edges.add((idx + 1, idx))

    if n_res > 1:
        distances = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        contact_i, contact_j = np.where((distances <= distance_cutoff) & (distances > 0))
        for i, j in zip(contact_i.tolist(), contact_j.tolist()):
            edges.add((i, j))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.array(sorted(edges)).T, dtype=torch.long)


def read_pocket_from_pdb_text(pdb_text, distance_cutoff, protein_3d_atoms=PROTEIN_ATOM5_NAMES):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", io.StringIO(pdb_text))

    residues = []
    ca_coords = []
    protein_atom_tokens = []
    protein_atom_coords = []
    for residue in structure.get_residues():
        if residue.get_id()[0] != " ":
            continue
        aa = AA3_TO_1.get(residue.get_resname().upper())
        if aa is None or "CA" not in residue:
            continue
        residues.append(aa)
        ca_coords.append(residue["CA"].get_coord().astype(np.float32))

        # 1D/2D 分支仍以 residue 为节点；3D 分支改用 PDB pocket 里真实存在的 N/CA/C/O/CB 原子点。
        for atom_name in protein_3d_atoms:
            if atom_name not in residue:
                continue
            protein_atom_tokens.append(PROTEIN_ATOM5_TOKEN[atom_name])
            protein_atom_coords.append(residue[atom_name].get_coord().astype(np.float32))

    if not residues:
        raise ValueError("no standard amino-acid residues with CA atoms")
    if not protein_atom_coords:
        raise ValueError("no selected protein 3D atoms found")

    ca_coords = np.asarray(ca_coords, dtype=np.float32)
    protein_atom_tokens = np.asarray(protein_atom_tokens, dtype=np.int64)
    protein_atom_coords = np.asarray(protein_atom_coords, dtype=np.float32)
    repre = np.stack(
        [residue_embedding(aa, idx, len(residues)) for idx, aa in enumerate(residues)],
        axis=0,
    )
    residue_ids = np.asarray([AA_TO_ID.get(aa, 0) for aa in residues], dtype=np.int64)
    edge_index = protein_graph_edges(ca_coords, distance_cutoff=distance_cutoff)
    graph = Data(x=torch.tensor(repre, dtype=torch.float), edge_index=edge_index)
    residue_sequence = "".join(residues)

    return repre, graph, residue_ids, ca_coords, protein_atom_tokens, protein_atom_coords, residue_sequence


def sanitize_mol(mol):
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return None
    return mol


def read_ligand_from_blocks(sdf_block, mol2_block):
    errors = []
    if sdf_block:
        try:
            mol = Chem.MolFromMolBlock(sdf_block, sanitize=False, removeHs=False)
            mol = sanitize_mol(mol)
            if mol is not None:
                return mol
        except Exception as exc:
            errors.append(f"sdf: {exc}")

    if mol2_block:
        try:
            mol = Chem.MolFromMol2Block(mol2_block, sanitize=False, removeHs=False)
            mol = sanitize_mol(mol)
            if mol is not None:
                return mol
        except Exception as exc:
            errors.append(f"mol2: {exc}")

    raise ValueError("; ".join(errors) if errors else "ligand could not be parsed")


def ligand_features(mol):
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    atom_types = [
        ATOM_ID.get(Chem.GetPeriodicTable().GetElementSymbol(atom.GetAtomicNum()), ATOM_ID["unknown"])
        for atom in mol.GetAtoms()
    ]
    graph = fea.simplify_atom_to_graph_data_obj(mol)
    las = get_las_distance_constraint_mask(mol).numpy().astype(np.float32)
    return graph, np.asarray(atom_types, dtype=np.int64), coords, las


def ligand_smiles_and_scaffold(mol):
    smiles = Chem.MolToSmiles(mol, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = smiles
    return smiles, scaffold


def get_las_distance_constraint_mask(mol):
    # 与 utils.tankbind_feature_utils.get_LAS_distance_constraint_mask 等价；
    # 放在这里避免为了一个 mask 额外依赖 torchdrug。
    adj = torch.from_numpy(Chem.GetAdjacencyMatrix(mol))
    eye = torch.eye(adj.size(0), dtype=torch.long)
    one_hop = torch.where(adj + eye > 0, torch.ones_like(adj), torch.zeros_like(adj))
    two_hop = torch.where(one_hop @ one_hop > 0, torch.ones_like(adj), torch.zeros_like(adj))
    extend_adj = two_hop - eye

    for ring in Chem.GetSymmSSSR(mol):
        ring_atoms = list(ring)
        for i in ring_atoms:
            for j in ring_atoms:
                if i != j:
                    extend_adj[i][j] += 1

    return torch.where(extend_adj > 0, torch.ones_like(extend_adj), torch.zeros_like(extend_adj))



def empty_dataset_lists():
    return [[] for _ in range(10)]


def add_sample(dataset_lists, sample):
    for target, value in zip(dataset_lists, sample):
        target.append(value)


def extend_dataset_lists(target, source):
    for target_list, source_list in zip(target, source):
        target_list.extend(source_list)


def checkpoint_dir_for(args):
    return args.checkpoint_dir or (args.output + ".checkpoint")


def checkpoint_paths(args):
    ckpt_dir = checkpoint_dir_for(args)
    return ckpt_dir, os.path.join(ckpt_dir, "state.json")


def load_checkpoint(args):
    ckpt_dir, state_path = checkpoint_paths(args)
    state = {
        "completed_ids": [],
        "failed_ids": [],
        "stats": None,
        "next_chunk_index": 0,
    }
    if not os.path.isdir(ckpt_dir):
        return state

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state.update(json.load(handle))
        return state

    # 兼容没有 state.json 的中间目录：从 chunk 文件重建已完成 ID 和计数。
    completed_ids = []
    positive = 0
    negative = 0
    chunk_files = sorted(
        name for name in os.listdir(ckpt_dir)
        if name.startswith("chunk_") and name.endswith(".pkl")
    )
    for name in chunk_files:
        with open(os.path.join(ckpt_dir, name), "rb") as handle:
            payload = pickle.load(handle)
        for item in payload.get("metadata", []):
            completed_ids.append(item["pdb_id"])
            if int(item["label"]) == 1:
                positive += 1
            else:
                negative += 1
    state["completed_ids"] = completed_ids
    state["stats"] = {
        "candidate_records": 0,
        "kept": len(completed_ids),
        "missing_structure": 0,
        "parse_errors": 0,
        "positive": positive,
        "negative": negative,
    }
    state["next_chunk_index"] = len(chunk_files)
    return state


def write_checkpoint_state(args, state):
    ckpt_dir, state_path = checkpoint_paths(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp_path, state_path)


def write_checkpoint_chunk(args, chunk_data, chunk_metadata, state):
    if not chunk_metadata:
        return
    ckpt_dir, _ = checkpoint_paths(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    chunk_index = int(state.get("next_chunk_index", 0))
    chunk_path = os.path.join(ckpt_dir, f"chunk_{chunk_index:06d}.pkl")
    tmp_path = chunk_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump({"data": chunk_data, "metadata": chunk_metadata}, handle)
    os.replace(tmp_path, chunk_path)
    state["next_chunk_index"] = chunk_index + 1
    write_checkpoint_state(args, state)


def load_checkpoint_dataset(args):
    ckpt_dir, _ = checkpoint_paths(args)
    dataset = empty_dataset_lists()
    metadata = []
    if not os.path.isdir(ckpt_dir):
        return dataset, metadata

    chunk_files = sorted(
        name for name in os.listdir(ckpt_dir)
        if name.startswith("chunk_") and name.endswith(".pkl")
    )
    for name in chunk_files:
        with open(os.path.join(ckpt_dir, name), "rb") as handle:
            payload = pickle.load(handle)
        extend_dataset_lists(dataset, payload["data"])
        metadata.extend(payload.get("metadata", []))
    return dataset, metadata

def build_dataset(args):
    index_tar_path = os.path.join(args.pdbbind_dir, "index.tar.gz")
    pl_tar_path = os.path.join(args.pdbbind_dir, "P-L.tar.gz")
    if not os.path.exists(index_tar_path):
        raise FileNotFoundError(index_tar_path)
    if not os.path.exists(pl_tar_path):
        raise FileNotFoundError(pl_tar_path)

    ckpt_dir, _ = checkpoint_paths(args)
    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir) and not args.resume:
        raise RuntimeError(
            f"checkpoint dir exists: {ckpt_dir}; pass --resume to continue "
            "or choose a different --checkpoint_dir"
        )

    records = load_index_records(
        index_tar_path,
        positive_threshold_molar=args.positive_threshold_um * 1e-6,
        negative_threshold_molar=args.negative_threshold_um * 1e-6,
    )
    record_by_id = {record.pdb_id: (record, label) for record, label in records}

    state = load_checkpoint(args) if args.resume else {
        "completed_ids": [],
        "failed_ids": [],
        "stats": None,
        "next_chunk_index": 0,
    }
    completed_pdb_ids = set(state.get("completed_ids", []))
    failed_pdb_ids = set(state.get("failed_ids", []))
    stats = state.get("stats") or {
        "candidate_records": len(records),
        "kept": len(completed_pdb_ids),
        "missing_structure": 0,
        "parse_errors": len(failed_pdb_ids),
        "positive": 0,
        "negative": 0,
    }
    stats["candidate_records"] = len(records)

    pending = {}
    chunk_data = empty_dataset_lists()
    chunk_metadata = []

    def persist_chunk():
        nonlocal chunk_data, chunk_metadata
        if not chunk_metadata:
            return
        state["completed_ids"] = sorted(completed_pdb_ids)
        state["failed_ids"] = sorted(failed_pdb_ids)
        state["stats"] = stats
        write_checkpoint_chunk(args, chunk_data, chunk_metadata, state)
        chunk_data = empty_dataset_lists()
        chunk_metadata = []

    def process_pending_entry(pdb_id):
        if pdb_id in completed_pdb_ids or pdb_id in failed_pdb_ids:
            pending.pop(pdb_id, None)
            return

        entry = pending.get(pdb_id, {})
        if "pocket_text" not in entry or (
            "sdf_text" not in entry and "mol2_text" not in entry
        ):
            return

        record, label = record_by_id[pdb_id]
        pending.pop(pdb_id, None)

        try:
            p_repre, p_graph, p_residue_ids, p_ca, p_3d_feats, p_3d_coors, p_residue_sequence = read_pocket_from_pdb_text(
                entry["pocket_text"],
                distance_cutoff=args.protein_contact_cutoff,
            )
            ligand_mol = read_ligand_from_blocks(
                entry.get("sdf_text"),
                entry.get("mol2_text"),
            )
            l_graph, l_feats, l_coors, l_las = ligand_features(ligand_mol)
            ligand_smiles, ligand_scaffold = ligand_smiles_and_scaffold(ligand_mol)
        except Exception as exc:
            failed_pdb_ids.add(pdb_id)
            stats["parse_errors"] += 1
            state["completed_ids"] = sorted(completed_pdb_ids)
            state["failed_ids"] = sorted(failed_pdb_ids)
            state["stats"] = stats
            write_checkpoint_state(args, state)
            if args.verbose:
                print(f"skip {pdb_id}: {exc}", flush=True)
            return

        sample = (
            p_repre,
            l_graph,
            p_graph,
            p_3d_feats,
            p_ca,
            p_3d_coors,
            l_feats,
            l_coors,
            l_las,
            int(label),
        )
        item_metadata = {
            "pdb_id": pdb_id,
            "binding": record.raw,
            "binding_type": record.binding_type,
            "label": int(label),
            "n_residue": int(len(p_residue_ids)),
            "n_protein_3d_token": int(len(p_3d_feats)),
            "n_ligand_atom": int(len(l_feats)),
            "ligand_smiles": ligand_smiles,
            "ligand_scaffold_smiles": ligand_scaffold,
            "residue_sequence": p_residue_sequence,
            "residue_ids": p_residue_ids.tolist(),
            "protein_3d_mode": "atom5",
            "protein_3d_atoms": list(PROTEIN_ATOM5_NAMES),
            "protein_3d_num_tokens": PROTEIN_ATOM5_NUM_TOKENS,
        }
        add_sample(chunk_data, sample)
        chunk_metadata.append(item_metadata)
        completed_pdb_ids.add(pdb_id)

        stats["kept"] += 1
        if label == 1:
            stats["positive"] += 1
        else:
            stats["negative"] += 1

        if args.progress_every and stats["kept"] % args.progress_every == 0:
            print(json.dumps(stats), flush=True)
        if args.checkpoint_every and len(chunk_metadata) >= args.checkpoint_every:
            persist_chunk()

    with tarfile.open(pl_tar_path, "r|gz") as tar:
        for member in tar:
            if args.max_samples and stats["kept"] >= args.max_samples:
                break
            if not member.isfile():
                continue

            base = os.path.basename(member.name)
            if "_" not in base:
                continue
            pdb_id = base.split("_", 1)[0].lower()
            if (
                pdb_id not in record_by_id
                or pdb_id in completed_pdb_ids
                or pdb_id in failed_pdb_ids
            ):
                continue

            field = None
            if base.endswith("_pocket.pdb"):
                field = "pocket_text"
            elif base.endswith("_ligand.sdf"):
                field = "sdf_text"
            elif base.endswith("_ligand.mol2"):
                field = "mol2_text"
            if field is None:
                continue

            handle = tar.extractfile(member)
            if handle is None:
                continue
            pending.setdefault(pdb_id, {})[field] = handle.read().decode(
                "utf-8", errors="replace"
            )
            process_pending_entry(pdb_id)

    for pdb_id in list(pending):
        if args.max_samples and stats["kept"] >= args.max_samples:
            break
        process_pending_entry(pdb_id)

    persist_chunk()

    if not args.max_samples:
        stats["missing_structure"] = len(record_by_id) - len(completed_pdb_ids) - len(failed_pdb_ids)
    state["completed_ids"] = sorted(completed_pdb_ids)
    state["failed_ids"] = sorted(failed_pdb_ids)
    state["stats"] = stats
    write_checkpoint_state(args, state)

    dataset, metadata = load_checkpoint_dataset(args)
    return dataset, metadata, stats

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDBbind v2020.R1 protein-ligand data to GerNA-Bind-style pkl."
    )
    parser.add_argument("--pdbbind_dir", default="PDBbind")
    parser.add_argument("--output", default="data/pdbbind_protein_ligand.pkl")
    parser.add_argument("--metadata_output", default="data/pdbbind_protein_ligand_metadata.json")
    parser.add_argument("--positive_threshold_um", type=float, default=1.0)
    parser.add_argument("--negative_threshold_um", type=float, default=10.0)
    parser.add_argument("--protein_contact_cutoff", type=float, default=8.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data, metadata, stats = build_dataset(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(data, handle)

    os.makedirs(os.path.dirname(args.metadata_output) or ".", exist_ok=True)
    with open(args.metadata_output, "w", encoding="utf-8") as handle:
        json.dump({"stats": stats, "samples": metadata}, handle, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"wrote dataset: {args.output}")
    print(f"wrote metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
