#!/usr/bin/env python
"""LP-PDBBind data entry for the upstream 3DProtDTA model.

This adapter keeps the upstream 3DProtDTA model and training loop unchanged.
It only prepares LP-PDBBind rows into the data objects expected by
3DProtDTA: full.csv, folds, ligand graphs, protein graphs, and ECFP vectors.
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREED_PROT_ROOT = PROJECT_ROOT / "external_benchmarks" / "3d-prot-dta"

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Other"]
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    "Other",
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

RESIDUE_CODES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWYH")
SULFUR = set("CM")
TINY = set("AGCS")
SMALL = set("AGSTCDNPV")


def one_hot(value, choices, unknown=None):
    if value not in choices and unknown is not None:
        value = unknown
    return [1.0 if value == choice else 0.0 for choice in choices]


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def build_ligand_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles}")

    node_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_SYMBOLS:
            symbol = "Other"
        degree = atom.GetDegree()
        if degree not in DEGREES:
            degree = DEGREES[-1]
        hybridization = atom.GetHybridization()
        if hybridization not in HYBRIDIZATIONS:
            hybridization = "Other"
        feat = (
            one_hot(symbol, ATOM_SYMBOLS)
            + one_hot(degree, DEGREES)
            + [float(np.clip(atom.GetFormalCharge(), -3, 3)) / 3.0]
            + [1.0 if atom.GetIsAromatic() else 0.0]
            + one_hot(hybridization, HYBRIDIZATIONS, unknown="Other")
        )
        node_features.append(feat)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        base = (
            [1.0 if bt == choice else 0.0 for choice in BOND_TYPES]
            + [1.0 if bond.GetIsConjugated() else 0.0]
            + [1.0 if bond.IsInRing() else 0.0]
        )
        edges.extend([[u, v], [v, u]])
        edge_features.extend([base, base])

    if not edges:
        edges = [[0, 0]]
        edge_features = [[0.0] * 6]

    return (
        np.asarray(node_features, dtype=np.float32),
        np.asarray(edges, dtype=np.int64),
        np.asarray(edge_features, dtype=np.float32),
    )


def build_morgan_ecfp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def parse_pdb_residues(pdb_path):
    residues = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode, resname)
            try:
                coord = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            item = residues.setdefault(
                key,
                {"resname": resname, "chain": chain, "resseq": int(resseq) if resseq.lstrip("-").isdigit() else 0, "coords": [], "ca": None},
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
        parsed.append(
            {
                "resname": item["resname"],
                "chain": item["chain"],
                "resseq": item["resseq"],
                "ca": tuple(float(x) for x in item["ca"]),
            }
        )
    return parsed


def residue_node_features(residue, index, total):
    resname = residue["resname"] if residue["resname"] in RESIDUE_CODES else "UNK"
    aa1 = AA3_TO_AA1.get(resname, "X")
    t = 0.0 if total <= 1 else float(index) / float(total - 1)
    pos_features = []
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0):
        pos_features.append(math.sin(math.pi * freq * t))
        pos_features.append(math.cos(math.pi * freq * t))
    prop_features = [
        1.0 if aa1 in HYDROPHOBIC else 0.0,
        1.0 if aa1 in POLAR else 0.0,
        1.0 if aa1 in POSITIVE else 0.0,
        1.0 if aa1 in NEGATIVE else 0.0,
        1.0 if aa1 in AROMATIC else 0.0,
        1.0 if aa1 in SULFUR else 0.0,
        1.0 if aa1 in TINY else 0.0,
        1.0 if aa1 in SMALL else 0.0,
        1.0 if aa1 == "P" else 0.0,
        1.0 if aa1 == "G" else 0.0,
    ]
    features = one_hot(resname, RESIDUE_CODES, unknown="UNK") + prop_features + pos_features
    return np.asarray(features, dtype=np.float32)


def protein_edge_features(res_i, res_j, dist):
    seq_sep = abs(int(res_i["resseq"]) - int(res_j["resseq"]))
    same_chain = res_i["chain"] == res_j["chain"]
    bins = [dist <= 4.0, dist <= 8.0, dist <= 12.0, dist <= 16.0, dist > 16.0]
    return np.asarray(
        [
            min(dist / 20.0, 5.0),
            1.0 / (dist + 1.0),
            min(float(seq_sep) / 50.0, 5.0),
            1.0 if same_chain and seq_sep == 1 else 0.0,
            1.0 if same_chain else 0.0,
        ]
        + [1.0 if flag else 0.0 for flag in bins],
        dtype=np.float32,
    )


def build_protein_graph_from_residues(residues, cutoff=10.0):
    if len(residues) < 2:
        raise ValueError("protein graph needs at least two residues")

    coords = np.asarray([res["ca"] for res in residues], dtype=np.float32)
    node_features = np.stack(
        [residue_node_features(res, idx, len(residues)) for idx, res in enumerate(residues)],
        axis=0,
    )

    # Use a KD-tree instead of all-pairs residue scans. The resulting graph is
    # still the same cutoff graph, but full LP-PDBBind prepare is tractable.
    edge_pairs = set()
    tree = cKDTree(coords)
    for i, j in tree.query_pairs(r=float(cutoff)):
        edge_pairs.add((i, j))
        edge_pairs.add((j, i))

    for i in range(len(residues) - 1):
        if residues[i]["chain"] == residues[i + 1]["chain"]:
            edge_pairs.add((i, i + 1))
            edge_pairs.add((i + 1, i))

    edges = []
    edge_features = []
    for i, j in sorted(edge_pairs):
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        edges.append([i, j])
        edge_features.append(protein_edge_features(residues[i], residues[j], dist))

    if not edges:
        for i in range(len(residues) - 1):
            for u, v in ((i, i + 1), (i + 1, i)):
                dist = float(np.linalg.norm(coords[u] - coords[v]))
                edges.append([u, v])
                edge_features.append(protein_edge_features(residues[u], residues[v], dist))

    return (
        node_features.astype(np.float32),
        np.asarray(edges, dtype=np.int64),
        np.asarray(edge_features, dtype=np.float32),
    )


def build_protein_graph_from_pdb(pdb_path, cutoff=10.0):
    residues = parse_pdb_residues(pdb_path)
    if len(residues) < 2:
        raise ValueError(f"not enough residues in {pdb_path}")
    return build_protein_graph_from_residues(residues, cutoff=cutoff)


def pdbbind_year_group(date_value):
    year = int(str(date_value)[:4])
    if year <= 2000:
        return "1981-2000"
    if year <= 2010:
        return "2001-2010"
    return "2011-2019"


def pdbbind_protein_path(pdbbind_root, pdb_id, date_value):
    group = pdbbind_year_group(date_value)
    pdb_id = str(pdb_id).lower()
    return Path(pdbbind_root) / "P-L" / group / pdb_id / f"{pdb_id}_protein.pdb"


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def prepare_lp_dataset(args):
    out_dir = Path(args.data_dir)
    folds_dir = out_dir / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.lp_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    raw = raw.dropna(subset=["pdb_id", "smiles", "value", "new_split", "date"]).copy()
    raw = raw[raw["new_split"].isin(["train", "val", "test"])].copy()

    frames = []
    for split, limit in (("train", args.max_train_samples), ("val", args.max_val_samples), ("test", args.max_test_samples)):
        frame = raw[raw["new_split"] == split].copy()
        if limit is not None:
            frame = frame.head(limit).copy()
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    rows = []
    ligand_to_graph = {}
    ligand_to_ecfp = {}
    protein_to_graph = {}
    errors = []

    total_rows = len(raw)
    for processed, (_, row) in enumerate(raw.iterrows(), start=1):
        pdb_id = str(row["pdb_id"]).lower()
        try:
            smiles = canonicalize_smiles(row["smiles"])
            if smiles is None:
                raise ValueError("SMILES canonicalization failed")
            if smiles not in ligand_to_graph:
                ligand_to_graph[smiles] = build_ligand_graph(smiles)
                ligand_to_ecfp[smiles] = build_morgan_ecfp(smiles)
            if pdb_id not in protein_to_graph:
                protein_path = pdbbind_protein_path(args.pdbbind_root, pdb_id, row["date"])
                if not protein_path.exists():
                    raise FileNotFoundError(str(protein_path))
                protein_to_graph[pdb_id] = build_protein_graph_from_pdb(protein_path, cutoff=args.protein_cutoff)
            rows.append(
                {
                    "ligand": smiles,
                    "protein": pdb_id,
                    "label": float(row["value"]),
                    "split": row["new_split"],
                }
            )
        except Exception as exc:
            errors.append({"pdb_id": pdb_id, "split": row.get("new_split"), "error": repr(exc)})
        if args.progress_every and (processed % args.progress_every == 0 or processed == total_rows):
            status = {
                "prepare_processed": processed,
                "total": total_rows,
                "kept": len(rows),
                "ligands": len(ligand_to_graph),
                "proteins": len(protein_to_graph),
                "errors": len(errors),
                "last_pdb_id": pdb_id,
            }
            print(json.dumps(status, ensure_ascii=False), flush=True)
            with open(out_dir / "prepare_status.json", "w", encoding="utf-8") as handle:
                json.dump(status, handle, indent=2, ensure_ascii=False)

    full = pd.DataFrame(rows).reset_index(drop=True)
    full[["ligand", "protein", "label"]].to_csv(out_dir / "full.csv", index=False)
    val_idx = full.index[full["split"] == "val"].astype(int).tolist()
    test_idx = full.index[full["split"] == "test"].astype(int).tolist()
    with open(folds_dir / "train_fold_setting1.txt", "w", encoding="utf-8") as handle:
        json.dump([val_idx], handle)
    with open(folds_dir / "test_fold_setting1.txt", "w", encoding="utf-8") as handle:
        json.dump(test_idx, handle)
    save_pickle(ligand_to_graph, out_dir / "ligand_to_graph.pkl")
    save_pickle(ligand_to_ecfp, out_dir / "ligand_to_ecfp.pkl")
    save_pickle(protein_to_graph, out_dir / "protein_to_graph.pkl")
    with open(out_dir / "prepare_errors.json", "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, ensure_ascii=False)
    summary = {
        "data_dir": str(out_dir),
        "rows": int(len(full)),
        "train": int((full["split"] == "train").sum()) if len(full) else 0,
        "val": int((full["split"] == "val").sum()) if len(full) else 0,
        "test": int((full["split"] == "test").sum()) if len(full) else 0,
        "ligands": len(ligand_to_graph),
        "proteins": len(protein_to_graph),
        "errors": len(errors),
    }
    with open(out_dir / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def load_prepared_lp_data(data_dir):
    data_dir = Path(data_dir)
    full = pd.read_csv(data_dir / "full.csv")
    with open(data_dir / "folds" / "train_fold_setting1.txt", "r", encoding="utf-8") as handle:
        val_folds = json.load(handle)
    with open(data_dir / "folds" / "test_fold_setting1.txt", "r", encoding="utf-8") as handle:
        test_fold = json.load(handle)
    with open(data_dir / "ligand_to_graph.pkl", "rb") as handle:
        ligand_to_graph = pickle.load(handle)
    with open(data_dir / "protein_to_graph.pkl", "rb") as handle:
        protein_to_graph = pickle.load(handle)
    with open(data_dir / "ligand_to_ecfp.pkl", "rb") as handle:
        ligand_to_ecfp = pickle.load(handle)
    return full, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _flat_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def add_spearman_metric(metrics, y_true, y_pred):
    try:
        from scipy.stats import spearmanr

        true_values = _flat_numpy(y_true)
        pred_values = _flat_numpy(y_pred)
        if true_values.size < 2 or pred_values.size < 2:
            correlation = float("nan")
        else:
            correlation = spearmanr(true_values, pred_values).correlation
        metrics["spearman"] = float(correlation)
    except Exception:
        metrics["spearman"] = float("nan")
    return metrics


def write_epoch_outputs(out_dir, eval_split, epoch, epoch_to_metrics, best_epoch, best_record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = out_dir / f"official_3dprot_dta_lp_{eval_split}_metrics.json"
    metrics_pkl = out_dir / f"official_3dprot_dta_lp_{eval_split}_metrics.pkl"
    save_pickle(epoch_to_metrics, metrics_pkl)
    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(finite_json(epoch_to_metrics), handle, indent=2, ensure_ascii=False)

    history_path = out_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mse",
        "val_mae",
        "val_pearsonr",
        "val_ci",
        "val_rm2",
        "val_spearman",
        "train_mse",
        "train_mae",
        "train_pearsonr",
        "train_spearman",
    ]
    write_header = not history_path.exists()
    record = epoch_to_metrics[epoch]
    row = {
        "epoch": epoch,
        "train_loss": record.get("loss_train"),
        "val_loss": record.get("loss_val"),
        "train_mse": metric_value(record.get("metrics_train", {}), "mse"),
        "train_mae": metric_value(record.get("metrics_train", {}), "mae"),
        "train_pearsonr": metric_value(record.get("metrics_train", {}), "pearsonr"),
        "train_spearman": metric_value(record.get("metrics_train", {}), "spearman"),
        "val_mse": metric_value(record.get("metrics_val", {}), "mse"),
        "val_mae": metric_value(record.get("metrics_val", {}), "mae"),
        "val_pearsonr": metric_value(record.get("metrics_val", {}), "pearsonr"),
        "val_rm2": metric_value(record.get("metrics_val", {}), "rm2"),
        "val_ci": metric_value(record.get("metrics_val", {}), "ci"),
        "val_spearman": metric_value(record.get("metrics_val", {}), "spearman"),
    }
    with open(history_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    best_payload = {
        "epoch": best_epoch,
        "metrics": best_record,
        "selection": "max validation CI",
    }
    with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(best_payload), handle, indent=2, ensure_ascii=False)


def write_model_checkpoints(out_dir, model, optimizer, epoch, best_epoch, best_record, is_best, args=None):
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args or {},
        "best_epoch": best_epoch,
        "best_record": best_record,
    }
    torch.save(payload, out_dir / "last_model.pt")
    torch.save(payload, out_dir / "training_checkpoint.pt")
    if is_best:
        torch.save(payload, out_dir / "best_model.pt")


def write_eval_only_outputs(out_dir, eval_split, checkpoint_path, checkpoint_epoch, checkpoint_best_epoch, metrics, loss):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": eval_split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_epoch": checkpoint_best_epoch,
        "metrics": metrics,
        "loss": loss,
    }
    output_path = out_dir / f"official_3dprot_dta_lp_{eval_split}_eval_only_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(finite_json(payload), handle, indent=2, ensure_ascii=False)
    return output_path


def train_val_streaming(model, optimizer, criterion, train_dl, val_dl, epochs, score_fn, out_dir, eval_split, fold=None, verbose=False, with_rm2=False, with_ci=False, val_nth_epoch=10, run_args=None):
    import torch
    from utils import logger
    from utils.train import train, val

    torch.cuda.empty_cache()
    epoch_to_metrics = {}
    best_epoch = None
    best_record = None
    best_ci = -float("inf")
    best_loss = float("inf")
    log_path = Path(out_dir) / "train.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_handle:
        for epoch in range(1, epochs + 1):
            y_true_train, y_pred_train, loss_train = train(model, train_dl, optimizer, criterion)

            if epoch % val_nth_epoch != 0:
                write_model_checkpoints(out_dir, model, optimizer, epoch, best_epoch, best_record, is_best=False, args=run_args)
                continue

            y_true_val, y_pred_val, loss_val = val(model, val_dl, criterion)
            metrics_train = score_fn(y_true_train, y_pred_train)
            add_spearman_metric(metrics_train, y_true_train, y_pred_train)
            metrics_val = score_fn(y_true_val, y_pred_val, with_rm2=with_rm2, with_ci=with_ci)
            add_spearman_metric(metrics_val, y_true_val, y_pred_val)

            epoch_to_metrics[epoch] = {
                "metrics_train": metrics_train,
                "metrics_val": metrics_val,
                "loss_train": loss_train,
                "loss_val": loss_val,
            }

            ci = metric_value(metrics_val, "ci")
            if ci is not None:
                improved = ci > best_ci
            else:
                improved = loss_val < best_loss
            if improved:
                best_epoch = epoch
                best_record = epoch_to_metrics[epoch]
                if ci is not None:
                    best_ci = ci
                best_loss = loss_val

            write_epoch_outputs(out_dir, eval_split, epoch, epoch_to_metrics, best_epoch, best_record)
            write_model_checkpoints(out_dir, model, optimizer, epoch, best_epoch, best_record, is_best=improved, args=run_args)

            if verbose:
                lines = []
                if fold is not None:
                    lines.append(f"Fold       | {fold}")
                lines.append(f"Epoch      | {epoch}")
                lines.append("Train      | " + str({k: round(v, 3) for k, v in metrics_train.items()}) + " | loss: " + str(round(loss_train, 3)))
                lines.append("Validation | " + str({k: round(v, 3) for k, v in metrics_val.items()}) + " | loss: " + str(round(loss_val, 3)))
                best_ci_text = None if math.isinf(best_ci) else best_ci
                lines.append(f"Best       | epoch: {best_epoch}, ci: {best_ci_text}, loss: {best_loss:.6f}")
                for line in lines:
                    logger.info(line)
                    print(line, file=log_handle, flush=True)

    return epoch_to_metrics


def run_training(args):
    sys.path.insert(0, str(THREED_PROT_ROOT))
    import torch
    from helpers import CustomDataLoader, CustomTrial
    from models import DTIProtGraphChemGraphECFP
    from params import DEVICE, HP
    from utils import get_metrics_reg

    full, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_prepared_lp_data(args.data_dir)
    val_idx = set(val_folds[0])
    test_idx = set(test_fold)
    df_train_val = full[~full.index.isin(test_idx)]
    df_train = df_train_val[~df_train_val.index.isin(val_idx)]
    df_eval = full[full.index.isin(val_idx if args.eval_split == "val" else test_idx)]

    train_dl = CustomDataLoader(
        df=df_train[["ligand", "protein", "label"]].copy(),
        batch_size=args.batch_size,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=True,
    )
    eval_dl = CustomDataLoader(
        df=df_eval[["ligand", "protein", "label"]].copy(),
        batch_size=args.batch_size,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=False,
    )

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_to_metrics = train_val_streaming(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=eval_dl,
        epochs=args.epochs,
        score_fn=get_metrics_reg,
        out_dir=out_dir,
        eval_split=args.eval_split,
        fold=0,
        verbose=True,
        with_rm2=True,
        with_ci=True,
        val_nth_epoch=args.val_nth_epoch,
        run_args=vars(args),
    )
    save_pickle(epoch_to_metrics, out_dir / f"official_3dprot_dta_lp_{args.eval_split}_metrics.pkl")
    with open(out_dir / f"official_3dprot_dta_lp_{args.eval_split}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(epoch_to_metrics), handle, indent=2, ensure_ascii=False)
    print(f"metrics: {out_dir / f'official_3dprot_dta_lp_{args.eval_split}_metrics.json'}", flush=True)


def evaluate_checkpoint(args):
    sys.path.insert(0, str(THREED_PROT_ROOT))
    import torch
    from helpers import CustomDataLoader, CustomTrial
    from models import DTIProtGraphChemGraphECFP
    from params import DEVICE, HP
    from utils import get_metrics_reg
    from utils.train import val

    full, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_prepared_lp_data(args.data_dir)
    val_idx = set(val_folds[0])
    test_idx = set(test_fold)
    eval_indices = val_idx if args.eval_split == "val" else test_idx
    df_eval = full[full.index.isin(eval_indices)]

    eval_dl = CustomDataLoader(
        df=df_eval[["ligand", "protein", "label"]].copy(),
        batch_size=args.batch_size,
        device=DEVICE,
        e1_key_to_graph=ligand_to_graph,
        e2_key_to_graph=protein_to_graph,
        e1_key_to_fp=ligand_to_ecfp,
        shuffle=False,
    )

    checkpoint_path = Path(args.eval_only_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    model.load_state_dict(state_dict)
    criterion = torch.nn.MSELoss()

    y_true, y_pred, loss = val(model, eval_dl, criterion)
    metrics = get_metrics_reg(y_true, y_pred, with_rm2=True, with_ci=True)
    add_spearman_metric(metrics, y_true, y_pred)
    output_path = write_eval_only_outputs(
        out_dir=Path(args.output_dir),
        eval_split=args.eval_split,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        checkpoint_best_epoch=checkpoint.get("best_epoch") if isinstance(checkpoint, dict) else None,
        metrics=metrics,
        loss=loss,
    )
    print(f"eval_only_metrics: {output_path}", flush=True)
    print(json.dumps(finite_json({"split": args.eval_split, "metrics": metrics, "loss": loss}), ensure_ascii=False), flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train official 3DProtDTA model on LP-PDBBind data")
    parser.add_argument("--lp-csv", default=str(PROJECT_ROOT / "data" / "LP_PDBBind.csv"))
    parser.add_argument("--pdbbind-root", default=str(PROJECT_ROOT / "PDBbind"))
    parser.add_argument("--data-dir", default=str(THREED_PROT_ROOT / "data" / "lp_pdbbind"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "external_benchmarks" / "results" / "3dprot_dta_lp_official"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--protein-cutoff", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--val-nth-epoch", type=int, default=10)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--eval-only-checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_prepare:
        prepare_lp_dataset(args)
    if args.prepare_only:
        return
    if args.eval_only_checkpoint:
        evaluate_checkpoint(args)
        return
    run_training(args)


if __name__ == "__main__":
    main()
