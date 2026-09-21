#!/usr/bin/env python
"""Train official AttentionMGT-DTA model on LP-PDBBind.

The upstream model is kept unchanged. This adapter only builds LP-PDBBind
compound graphs, pocket residue graphs, and 320-d residue embeddings in the
format expected by AttentionMGT-DTA.
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import dgl
import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from scipy import sparse as sp
from scipy.spatial import cKDTree, distance_matrix
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_ROOT = PROJECT_ROOT / "external_benchmarks" / "AttentionMGT-DTA"

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Fe", "Zn", "Cu", "Mn", "Mo", "other"]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
RESIDUE_CODES = [
    "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR",
    "TRP", "SER", "THR", "CYS", "MET", "ASN", "GLN", "ASP",
    "GLU", "LYS", "ARG", "HIS", "MSE", "CSO", "PTR", "TPO",
    "KCX", "CSD", "SEP", "MLY", "PCA", "LLP", "M", "X",
]
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_AA3 = {v: k for k, v in AA3_TO_AA1.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train AttentionMGT-DTA on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--esm-lmdb", default="data/PDBbind_chain_lmdb_esm_gernabind_v2")
    parser.add_argument("--cache-dir", default="external_benchmarks/AttentionMGT-DTA/data/lp_pdbbind")
    parser.add_argument("--output-dir", default="external_benchmarks/results/attentionmgt_dta_lp_official")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pocket-cutoff", type=float, default=10.0)
    parser.add_argument("--match-tolerance", type=float, default=1.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def one_hot_unk(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == item else 0.0 for item in choices]


def laplacian_positional_encoding(graph, pos_enc_dim=8):
    num_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    adj = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_sqrt = np.power(np.clip(deg, 1.0, None), -0.5)
    norm = sp.diags(inv_sqrt, dtype=float)
    lap = sp.eye(num_nodes, dtype=float) - norm @ adj.astype(float) @ norm
    eig_val, eig_vec = np.linalg.eig(lap.toarray())
    order = eig_val.argsort()
    eig_vec = np.real(eig_vec[:, order])
    if eig_vec.shape[1] < pos_enc_dim + 1:
        pad = np.zeros((num_nodes, pos_enc_dim + 1 - eig_vec.shape[1]), dtype=eig_vec.dtype)
        eig_vec = np.concatenate([eig_vec, pad], axis=1)
    graph.ndata["lap_pos_enc"] = torch.from_numpy(eig_vec[:, 1:pos_enc_dim + 1]).float()
    return graph


def atom_features(atom):
    feats = (
        one_hot_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_hot_unk(atom.GetDegree(), ATOM_DEGREES)
        + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        + one_hot_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + one_hot_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    )
    try:
        feats += one_hot_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [1.0 if atom.HasProp("_ChiralityPossible") else 0.0]
    except Exception:
        feats += [0.0, 0.0, 1.0 if atom.HasProp("_ChiralityPossible") else 0.0]
    chiral_arr = [0.0, 0.0, 0.0]
    return feats + chiral_arr


def bond_features(bond):
    stereo = one_hot_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
    )
    return [
        bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
        bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
        bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ] + stereo


def build_compound_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        edge_feats.extend([feat, feat])
    if not src:
        src, dst, edge_feats = [0], [0], [[0.0] * 10]
    graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=mol.GetNumAtoms())
    graph.ndata["atom"] = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    graph.edata["bond"] = torch.tensor(edge_feats, dtype=torch.float32)
    return laplacian_positional_encoding(graph, pos_enc_dim=8)


def build_file_index(pdbbind_root, pdb_ids):
    root = Path(pdbbind_root)
    search_roots = [p for p in (root / "P-L").iterdir() if p.is_dir()] if (root / "P-L").exists() else [root]
    index = {}
    for pdb_id in sorted({str(x).lower() for x in pdb_ids}):
        for search_root in search_roots:
            base = search_root / pdb_id
            if base.exists():
                index[pdb_id] = {
                    "protein": base / f"{pdb_id}_protein.pdb",
                    "pocket": base / f"{pdb_id}_pocket.pdb",
                }
                break
    return index


def parse_pocket_residues(pocket_pdb):
    residues = {}
    with open(pocket_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq_text = line[22:26].strip()
            icode = line[26].strip()
            try:
                resseq = int(resseq_text)
                coord = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            key = (chain, resseq, icode)
            item = residues.setdefault(
                key,
                {"chain": chain, "resseq": resseq, "icode": icode, "resname": resname, "coords": [], "ca": None},
            )
            item["coords"].append(coord)
            if atom_name == "CA":
                item["ca"] = coord
    parsed = []
    for item in residues.values():
        if item["ca"] is None:
            if not item["coords"]:
                continue
            item["ca"] = np.mean(np.stack(item["coords"], axis=0), axis=0)
        parsed.append(item)
    parsed.sort(key=lambda x: (x["chain"], x["resseq"], x["icode"]))
    return parsed


def open_lmdb(path):
    return lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)


def load_esm_payload(txn, pdb_id):
    raw = txn.get(f"{pdb_id.lower()}_protein".encode("utf-8"))
    if raw is None:
        raise KeyError(f"ESM LMDB key missing: {pdb_id}_protein")
    return pickle.loads(raw)["protein_features"]


def resize_embedding(embedding, dim=320):
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.shape[-1] >= dim:
        return arr[..., :dim].astype(np.float32)
    out = np.zeros(arr.shape[:-1] + (dim,), dtype=np.float32)
    out[..., : arr.shape[-1]] = arr
    return out


def match_lmdb_residue(pocket_residue, chain_payloads, tolerance):
    candidates = []
    chain = pocket_residue["chain"]
    if chain in chain_payloads:
        candidates.append(chain_payloads[chain])
    candidates.extend(v for k, v in chain_payloads.items() if k != chain)
    ca = np.asarray(pocket_residue["ca"], dtype=np.float32)
    best = None
    best_dist = float("inf")
    for payload in candidates:
        coords = np.asarray(payload.get("ca_coords"), dtype=np.float32)
        if len(coords) == 0:
            continue
        idx = int(np.argmin(np.linalg.norm(coords - ca.reshape(1, 3), axis=1)))
        dist = float(np.linalg.norm(coords[idx] - ca))
        if dist < best_dist:
            best = (payload, idx)
            best_dist = dist
    if best is None or best_dist > tolerance:
        return None, None
    return best


def residue_type_features(resname):
    name = resname if resname in RESIDUE_CODES else "X"
    return one_hot_unk(name, RESIDUE_CODES)


def residue_geometry_features(residue):
    coords = np.asarray(residue["coords"], dtype=np.float32)
    if len(coords) == 0:
        return [0.0] * 5
    if len(coords) == 1:
        dmax = dmin = 0.0
    else:
        dist = distance_matrix(coords, coords)
        vals = dist[np.triu_indices(len(coords), k=1)]
        dmax = float(vals.max()) if len(vals) else 0.0
        dmin = float(vals.min()) if len(vals) else 0.0
    ca = np.asarray(residue["ca"], dtype=np.float32)
    return [dmax * 0.1, dmin * 0.1, 0.0, 0.0, 0.0 if ca is None else 0.0]


def residue_angles(payload, idx):
    if payload is None or idx is None:
        return [0.0, 0.0, 0.0, 0.0]
    values = []
    for key in ("phi", "psi", "omega"):
        arr = payload.get(key)
        if arr is None or idx >= len(arr):
            values.append(0.0)
        else:
            values.append(float(np.degrees(arr[idx])) * 0.01)
    chi = payload.get("chi_angles")
    values.append(float(np.degrees(chi[idx, 0])) * 0.01 if chi is not None and idx < len(chi) else 0.0)
    return values


def build_residue_records(pocket_residues, chain_payloads, tolerance):
    records = []
    for residue in pocket_residues:
        payload, idx = match_lmdb_residue(residue, chain_payloads, tolerance)
        if payload is not None and idx is not None:
            seq = payload.get("sequence")
            if seq is not None and idx < len(seq):
                resname = AA1_TO_AA3.get(str(seq[idx]), residue["resname"])
            else:
                resname = residue["resname"]
            esm = payload.get("esm2_embedding")
            embedding = resize_embedding(esm[idx], 320) if esm is not None and idx < len(esm) else np.zeros((320,), dtype=np.float32)
        else:
            resname = residue["resname"]
            embedding = np.zeros((320,), dtype=np.float32)
        records.append(
            {
                "resname": resname,
                "chain": residue["chain"],
                "resseq": residue["resseq"],
                "ca": tuple(float(x) for x in residue["ca"]),
                "coords": [tuple(float(x) for x in coord) for coord in residue["coords"]],
                "embedding": embedding.astype(np.float32),
                "angles": residue_angles(payload, idx),
            }
        )
    return records


def protein_node_features(record):
    return np.asarray(residue_type_features(record["resname"]) + residue_geometry_features(record) + list(record["angles"]), dtype=np.float32)


def edge_features(left, right):
    left_ca = np.asarray(left["ca"], dtype=np.float32)
    right_ca = np.asarray(right["ca"], dtype=np.float32)
    ca_dist = float(np.linalg.norm(left_ca - right_ca))
    left_coords = np.asarray(left["coords"], dtype=np.float32)
    right_coords = np.asarray(right["coords"], dtype=np.float32)
    atom_dist = distance_matrix(left_coords, right_coords) if len(left_coords) and len(right_coords) else np.asarray([[ca_dist]])
    center_dist = float(np.linalg.norm(left_coords.mean(axis=0) - right_coords.mean(axis=0))) if len(left_coords) and len(right_coords) else ca_dist
    connected = 1.0 if left["chain"] == right["chain"] and abs(int(left["resseq"]) - int(right["resseq"])) == 1 else 0.0
    return [connected, ca_dist * 0.1, center_dist * 0.1, float(atom_dist.min()) * 0.1, float(atom_dist.max()) * 0.1]


def build_protein_graph_from_residue_records(records, cutoff=10.0):
    if not records:
        raise ValueError("protein graph requires at least one residue")
    coords = np.asarray([record["ca"] for record in records], dtype=np.float32)
    pairs = set()
    if len(records) > 1:
        tree = cKDTree(coords)
        for i, j in tree.query_pairs(r=float(cutoff)):
            pairs.add((i, j))
            pairs.add((j, i))
        if not pairs:
            for i in range(len(records) - 1):
                pairs.add((i, i + 1))
                pairs.add((i + 1, i))
    else:
        pairs.add((0, 0))
    src, dst = zip(*sorted(pairs))
    graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=len(records))
    graph.ndata["feats"] = torch.from_numpy(np.stack([protein_node_features(record) for record in records], axis=0)).float()
    graph.edata["feats"] = torch.tensor([edge_features(records[i], records[j]) for i, j in sorted(pairs)], dtype=torch.float32)
    graph = laplacian_positional_encoding(graph, pos_enc_dim=8)
    embedding = np.stack([resize_embedding(record["embedding"], 320) for record in records], axis=0).astype(np.float32)
    return graph, embedding


def load_lp_splits(args):
    df = pd.read_csv(args.lp_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df = df.dropna(subset=["pdb_id", "smiles", "value", "new_split"]).copy()
    df = df[df["new_split"].isin(["train", "val", "test"])].copy()
    frames = {}
    for split, limit in (("train", args.max_train_samples), ("val", args.max_val_samples), ("test", args.max_test_samples)):
        frame = df[df["new_split"] == split].copy()
        if limit is not None:
            frame = frame.head(limit).copy()
        frames[split] = frame.reset_index(drop=True)
    return frames


def prepare_dataset(args):
    cache = Path(args.cache_dir)
    if args.rebuild_cache and cache.exists():
        import shutil
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    frames = load_lp_splits(args)
    all_rows = pd.concat(frames.values(), ignore_index=True)
    file_index = build_file_index(args.pdbbind_root, all_rows["pdb_id"].unique())
    env = open_lmdb(args.esm_lmdb)
    txn = env.begin()
    graphs = {split: {"compound": [], "protein": [], "embedding": [], "labels": [], "rows": []} for split in frames}
    errors = []

    for split, frame in frames.items():
        for idx, row in frame.iterrows():
            pdb_id = str(row["pdb_id"]).lower()
            files = file_index.get(pdb_id)
            try:
                if files is None:
                    raise FileNotFoundError(f"PDBbind files not found for {pdb_id}")
                if not files["pocket"].exists():
                    raise FileNotFoundError(str(files["pocket"]))
                compound_graph = build_compound_graph(row["smiles"])
                pocket_residues = parse_pocket_residues(files["pocket"])
                chain_payloads = load_esm_payload(txn, pdb_id)
                records = build_residue_records(pocket_residues, chain_payloads, args.match_tolerance)
                protein_graph, protein_embedding = build_protein_graph_from_residue_records(records, cutoff=args.pocket_cutoff)
                graphs[split]["compound"].append(compound_graph)
                graphs[split]["protein"].append(protein_graph)
                graphs[split]["embedding"].append(protein_embedding)
                graphs[split]["labels"].append(float(row["value"]))
                graphs[split]["rows"].append({"pdb_id": pdb_id, "smiles": str(row["smiles"]), "value": float(row["value"]), "split": split})
            except Exception as exc:
                errors.append({"pdb_id": pdb_id, "split": split, "error": repr(exc)})
            seen = sum(len(graphs[s]["labels"]) for s in graphs) + len(errors)
            if args.progress_every and seen % args.progress_every == 0:
                print(json.dumps({"processed": seen, "kept": sum(len(graphs[s]["labels"]) for s in graphs), "errors": len(errors), "last_pdb_id": pdb_id}, ensure_ascii=False), flush=True)

    for split, data in graphs.items():
        split_dir = cache / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dgl.save_graphs(str(split_dir / "compound_graph.bin"), data["compound"])
        dgl.save_graphs(str(split_dir / "protein_graph.bin"), data["protein"])
        np.save(split_dir / "protein_embedding.npy", np.asarray(data["embedding"], dtype=object), allow_pickle=True)
        np.save(split_dir / "label.npy", np.asarray(data["labels"], dtype=np.float32))
        pd.DataFrame(data["rows"]).to_csv(split_dir / "rows.csv", index=False)
    summary = {
        "cache_dir": str(cache),
        "train": len(graphs["train"]["labels"]),
        "val": len(graphs["val"]["labels"]),
        "test": len(graphs["test"]["labels"]),
        "errors": len(errors),
        "embedding_dim": 320,
    }
    with open(cache / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(cache / "prepare_errors.json", "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    env.close()


class AttentionMGTDataset(Dataset):
    def __init__(self, cache_dir, split):
        split_dir = Path(cache_dir) / split
        self.compound_graphs, _ = dgl.load_graphs(str(split_dir / "compound_graph.bin"))
        self.protein_graphs, _ = dgl.load_graphs(str(split_dir / "protein_graph.bin"))
        self.protein_embedding = np.load(split_dir / "protein_embedding.npy", allow_pickle=True)
        self.labels = np.load(split_dir / "label.npy", allow_pickle=True).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.compound_graphs[idx], self.protein_graphs[idx], self.protein_embedding[idx], self.labels[idx]


def collate_attentionmgt(samples):
    compounds, proteins, embeddings, labels = map(list, zip(*samples))
    max_len = max(np.asarray(item).shape[0] for item in embeddings)
    padded = []
    for item in embeddings:
        arr = np.asarray(item, dtype=np.float32)
        if arr.shape[0] < max_len:
            arr = np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)), mode="constant")
        padded.append(arr)
    return (
        dgl.batch(compounds),
        dgl.batch(proteins),
        torch.tensor(np.stack(padded, axis=0), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32).view(-1, 1),
    )


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    permissible = 0.0
    for i in range(len(y_true)):
        diff_true = y_true[i] - y_true[i + 1:]
        diff_pred = y_pred[i] - y_pred[i + 1:]
        mask = diff_true != 0
        if not np.any(mask):
            continue
        prod = diff_true[mask] * diff_pred[mask]
        concordant += float(np.sum(prod > 0))
        concordant += 0.5 * float(np.sum(prod == 0))
        permissible += float(np.sum(mask))
    return concordant / permissible if permissible else float("nan")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        try:
            from scipy.stats import spearmanr
            out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            out["spearman"] = float("nan")
        out["ci"] = float(concordance_index(y_true, y_pred))
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
        out["ci"] = float("nan")
    return out


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_checkpoints(out_dir, model, optimizer, epoch, best_record, is_best, args=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_record": best_record,
        "args": args or {},
    }
    torch.save(payload, out_dir / "last_model.pt")
    torch.save(payload, out_dir / "training_checkpoint.pt")
    if is_best:
        torch.save(payload, out_dir / "best_model.pt")


def evaluate(model, loader, device, criterion):
    model.eval()
    preds, labels = [], []
    losses = []
    with torch.no_grad():
        for compound_graph, protein_graph, protein_embedding, label in loader:
            compound_graph = compound_graph.to(device)
            protein_graph = protein_graph.to(device)
            protein_embedding = protein_embedding.to(device)
            label = label.to(device)
            pred = model(compound_graph, protein_graph, protein_embedding)
            loss = criterion(pred, label)
            losses.append(float(loss.item()))
            preds.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
            labels.extend(label.detach().cpu().numpy().reshape(-1).tolist())
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def format_train_progress(epoch, batch_idx, total_batches, loss):
    return f"epoch {epoch} batch {batch_idx + 1}/{total_batches} loss {loss:.6f}"


def train_one_epoch(model, loader, device, optimizer, criterion, epoch=None, progress_every=0, log_handle=None):
    model.train()
    losses = []
    total_batches = len(loader)
    for batch_idx, (compound_graph, protein_graph, protein_embedding, label) in enumerate(loader):
        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(compound_graph, protein_graph, protein_embedding)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        if epoch is not None and progress_every and (batch_idx == 0 or (batch_idx + 1) % progress_every == 0 or batch_idx + 1 == total_batches):
            message = format_train_progress(epoch, batch_idx, total_batches, loss_value)
            print(message, flush=True)
            if log_handle is not None:
                print(message, file=log_handle, flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_dgl_function_compatibility():
    import dgl.function as fn

    if not hasattr(fn, "src_mul_edge") and hasattr(fn, "u_mul_e"):
        fn.src_mul_edge = fn.u_mul_e
    if not hasattr(fn, "copy_edge") and hasattr(fn, "copy_e"):
        fn.copy_edge = fn.copy_e


def train_model(args):
    patch_dgl_function_compatibility()
    sys.path.insert(0, str(ATTENTION_ROOT))
    import models.net as attention_net
    from models.net import MDGTDTInet

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    attention_net.device = device

    train_set = AttentionMGTDataset(args.cache_dir, "train")
    val_set = AttentionMGTDataset(args.cache_dir, "val")
    test_set = AttentionMGTDataset(args.cache_dir, "test")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_attentionmgt, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_attentionmgt)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_attentionmgt)

    model = MDGTDTInet(compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.8, patience=80, verbose=True, min_lr=1e-5)
    criterion = nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    history_path = out_dir / "history.csv"
    log_path = out_dir / "train.log"
    best_ci = -float("inf")
    best_record = None
    start_epoch = 1
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        best_record = payload.get("best_record")
        best_ci = (best_record or {}).get("val", {}).get("ci", -float("inf"))
        start_epoch = int(payload["epoch"]) + 1

    write_header = not history_path.exists() or start_epoch == 1
    with open(history_path, "a", newline="", encoding="utf-8") as hist, open(log_path, "a", encoding="utf-8") as log:
        fieldnames = ["epoch", "train_loss", "val_mse", "val_rmse", "val_mae", "val_pearson", "val_spearman", "val_ci", "test_mse", "test_rmse", "test_mae", "test_pearson", "test_spearman", "test_ci"]
        writer = csv.DictWriter(hist, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            hist.flush()
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch=epoch, progress_every=args.progress_every, log_handle=log)
            should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
            if should_eval:
                val_metrics = evaluate(model, val_loader, device, criterion)
                test_metrics = evaluate(model, test_loader, device, criterion)
                scheduler.step(val_metrics["mse"])
                ci_value = val_metrics["ci"]
                improved = best_record is None or (not math.isnan(ci_value) and ci_value > best_ci)
                record = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "test": test_metrics}
                if improved:
                    if not math.isnan(ci_value):
                        best_ci = ci_value
                    best_record = record
                    with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
                        json.dump(finite_json(best_record), handle, indent=2, ensure_ascii=False)
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mse": val_metrics["mse"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_pearson": val_metrics["pearson"],
                    "val_spearman": val_metrics["spearman"],
                    "val_ci": val_metrics["ci"],
                    "test_mse": test_metrics["mse"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_pearson": test_metrics["pearson"],
                    "test_spearman": test_metrics["spearman"],
                    "test_ci": test_metrics["ci"],
                }
                writer.writerow(row)
                hist.flush()
                msg = f"epoch {epoch} train_loss {train_loss:.6f} val CI {val_metrics['ci']:.6f} RMSE {val_metrics['rmse']:.6f} Pearson {val_metrics['pearson']:.6f} test CI {test_metrics['ci']:.6f} best_ci {best_ci:.6f}"
                print(msg, flush=True)
                print(msg, file=log, flush=True)
                write_checkpoints(out_dir, model, optimizer, epoch, best_record, is_best=improved, args=vars(args))
            else:
                write_checkpoints(out_dir, model, optimizer, epoch, best_record, is_best=False, args=vars(args))


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not args.skip_prepare:
        prepare_dataset(args)
    if args.prepare_only:
        return
    train_model(args)


if __name__ == "__main__":
    main()
