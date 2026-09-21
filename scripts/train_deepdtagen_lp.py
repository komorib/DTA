#!/usr/bin/env python
"""LP-PDBBind training adapter for the official DeepDTAGen model.

The upstream DeepDTAGen model and FetterGrad optimizer are kept unchanged.
This script only converts LP-PDBBind rows into the PyG/TestbedDataset format
expected by DeepDTAGen, then trains with val/test evaluation and checkpoints.
"""

import argparse
import csv
import json
import math
import pickle
import random
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPDTAGEN_ROOT = PROJECT_ROOT / "external_benchmarks" / "DeepDTAGen"

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}
MAX_SEQ_LEN = 1000
DEEPDTAGEN_TARGET_MAX_LEN = 138

ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K",
    "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == item for item in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == item for item in allowable_set] + [x not in allowable_set]


def atom_features(atom):
    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
        + [atom.IsInRing()],
        dtype=np.float32,
    )


def bond_features(bond):
    bond_type = bond.GetBondType()
    feats = [0.0, 0.0, 0.0, 0.0, float(bond.GetBondTypeAsDouble())]
    if bond_type == Chem.rdchem.BondType.SINGLE:
        feats = [1.0, 0.0, 0.0, 0.0, float(bond.GetBondTypeAsDouble())]
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        feats = [0.0, 1.0, 0.0, 0.0, float(bond.GetBondTypeAsDouble())]
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        feats = [0.0, 0.0, 1.0, 0.0, float(bond.GetBondTypeAsDouble())]
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        feats = [0.0, 0.0, 0.0, 1.0, float(bond.GetBondTypeAsDouble())]
    return np.asarray(feats, dtype=np.float32)


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(str(smile))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smile}")

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        denom = float(feature.sum()) if float(feature.sum()) != 0.0 else 1.0
        features.append((feature / denom).astype(np.float32).tolist())

    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {"edge_feats": bond_features(bond)}))

    graph = nx.Graph()
    graph.add_nodes_from(range(c_size))
    graph.add_edges_from(edges)
    graph = graph.to_directed()
    edge_index = []
    edge_feats = []
    for e1, e2, feats in graph.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(np.asarray(feats["edge_feats"], dtype=np.float32).tolist())
    if not edge_index:
        edge_index = [[0, 0]]
        edge_feats = [[0.0] * 5]
    return c_size, features, edge_index, edge_feats


def sequence_to_array(sequence):
    clean = str(sequence).replace(":", "")
    arr = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for idx, ch in enumerate(clean[:MAX_SEQ_LEN]):
        arr[idx] = SEQ_DICT.get(ch, 0)
    return arr


def tokenize_smiles_for_model(tokenizer, smiles, max_len=DEEPDTAGEN_TARGET_MAX_LEN, policy="truncate"):
    tokens = list(tokenizer.parse(smiles))
    if len(tokens) > max_len:
        if policy == "drop":
            return None
        if policy != "truncate":
            raise ValueError(f"unknown long SMILES policy: {policy}")
        tokens = tokens[:max_len]
        tokens[-1] = tokenizer.s2i["<eos>"]
    return torch.LongTensor(tokens)


def install_deepdtagen_path():
    root = str(DEEPDTAGEN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_lp_frames(args):
    frame = pd.read_csv(args.lp_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    frame["pdb_id"] = frame["pdb_id"].astype(str).str.lower()
    frame = frame.dropna(subset=["pdb_id", "smiles", "seq", "value", "new_split"]).copy()
    frame = frame[frame["new_split"].isin(["train", "val", "test"])].copy()
    frames = {}
    for split, limit in (("train", args.max_train_samples), ("val", args.max_val_samples), ("test", args.max_test_samples)):
        split_frame = frame[frame["new_split"] == split].copy()
        if limit is not None:
            split_frame = split_frame.head(limit).copy()
        frames[split] = split_frame.reset_index(drop=True)
    return frames


def build_split_rows(frames):
    rows = {"train": [], "val": [], "test": []}
    errors = []
    for split, frame in frames.items():
        for _, row in frame.iterrows():
            pdb_id = str(row["pdb_id"]).lower()
            try:
                smiles = canonicalize_smiles(row["smiles"])
                if smiles is None:
                    raise ValueError("invalid SMILES")
                seq_arr = sequence_to_array(row["seq"])
                rows[split].append(
                    {
                        "pdb_id": pdb_id,
                        "compound_iso_smiles": smiles,
                        "target_smiles": smiles,
                        "target_sequence": str(row["seq"]),
                        "target_array": seq_arr,
                        "affinity": float(row["value"]),
                    }
                )
            except Exception as exc:
                errors.append({"pdb_id": pdb_id, "split": split, "error": repr(exc)})
    return rows, errors


def prepare_lp_dataset(args):
    install_deepdtagen_path()
    from utils import TestbedDataset, Tokenizer

    cache = Path(args.cache_dir)
    if args.rebuild_cache and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "csv").mkdir(parents=True, exist_ok=True)

    frames = load_lp_frames(args)
    rows_by_split, errors = build_split_rows(frames)
    all_smiles = [row["target_smiles"] for split_rows in rows_by_split.values() for row in split_rows]
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))
    with open(cache / "lp_pdbbind_tokenizer.pkl", "wb") as handle:
        pickle.dump(tokenizer, handle)

    unique_smiles = sorted(set(row["compound_iso_smiles"] for split_rows in rows_by_split.values() for row in split_rows))
    smile_graph = {}
    for idx, smiles in enumerate(unique_smiles, start=1):
        try:
            smile_graph[smiles] = smile_to_graph(smiles)
        except Exception as exc:
            errors.append({"pdb_id": None, "split": "graph", "smiles": smiles, "error": repr(exc)})
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"graph_processed": idx, "total_graphs": len(unique_smiles), "errors": len(errors)}), flush=True)

    summary = {"cache_dir": str(cache), "errors": len(errors), "tokenizer_size": len(tokenizer), "long_smiles_policy": args.long_smiles_policy}
    for split, split_rows in rows_by_split.items():
        filtered = []
        tokenized = []
        dropped_long = 0
        for row in split_rows:
            if row["compound_iso_smiles"] not in smile_graph:
                continue
            tokens = tokenize_smiles_for_model(tokenizer, row["target_smiles"], policy=args.long_smiles_policy)
            if tokens is None:
                dropped_long += 1
                errors.append({
                    "pdb_id": row["pdb_id"],
                    "split": split,
                    "smiles": row["target_smiles"],
                    "error": f"target_smiles_token_length>{DEEPDTAGEN_TARGET_MAX_LEN}",
                })
                continue
            filtered.append(row)
            tokenized.append(tokens)
        summary[split] = len(filtered)
        summary[f"{split}_dropped_long_smiles"] = dropped_long
        csv_rows = [
            {
                "pdb_id": row["pdb_id"],
                "compound_iso_smiles": row["compound_iso_smiles"],
                "target_smiles": row["target_smiles"],
                "target_sequence": row["target_sequence"],
                "affinity": row["affinity"],
            }
            for row in filtered
        ]
        pd.DataFrame(csv_rows).to_csv(cache / "csv" / f"lp_pdbbind_{split}.csv", index=False)
        xd = np.asarray([row["compound_iso_smiles"] for row in filtered])
        xdt = tokenized
        xt = np.asarray([row["target_array"] for row in filtered])
        y = np.asarray([row["affinity"] for row in filtered], dtype=np.float32)
        print(f"preparing lp_pdbbind_{split}.pt in pytorch format", flush=True)
        TestbedDataset(root=str(cache), dataset=f"lp_pdbbind_{split}", xd=xd, xdt=xdt, xt=xt, y=y, smile_graph=smile_graph)

    with open(cache / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(summary), handle, indent=2, ensure_ascii=False)
    with open(cache / "prepare_errors.json", "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, ensure_ascii=False)
    print(json.dumps(finite_json(summary), indent=2, ensure_ascii=False), flush=True)
    return summary


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    permissible = 0.0
    for i in range(len(y_true)):
        dy = y_true[i] - y_true[i + 1:]
        dp = y_pred[i] - y_pred[i + 1:]
        mask = dy != 0
        if not np.any(mask):
            continue
        prod = dy[mask] * dp[mask]
        concordant += float(np.sum(prod > 0))
        concordant += 0.5 * float(np.sum(prod == 0))
        permissible += float(np.sum(mask))
    return concordant / permissible if permissible else float("nan")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "ci": float(concordance_index(y_true, y_pred)),
    }
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        out["spearman"] = float(stats.spearmanr(y_true, y_pred).correlation)
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    return out


def is_finite_number(value):
    if value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(value) or math.isinf(value))


def is_better_validation(val_metrics, best_val_ci, best_val_loss):
    ci = val_metrics.get("ci")
    loss = val_metrics.get("loss")
    if is_finite_number(ci):
        ci = float(ci)
        return ci > best_val_ci, ci if ci > best_val_ci else best_val_ci, float(loss) if is_finite_number(loss) else best_val_loss
    if is_finite_number(loss) and float(loss) < best_val_loss:
        return True, best_val_ci, float(loss)
    return False, best_val_ci, best_val_loss


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, tuple):
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
        "optimizer_state_dict": optimizer.optimizer.state_dict() if hasattr(optimizer, "optimizer") else optimizer.state_dict(),
        "best_record": best_record,
        "args": args or {},
    }
    torch.save(payload, out_dir / "last_model.pt")
    torch.save(payload, out_dir / "training_checkpoint.pt")
    if is_best:
        torch.save(payload, out_dir / "best_model.pt")


def write_history(out_dir, row):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "history.csv"
    fieldnames = [
        "epoch", "train_loss", "train_mse", "train_lm_loss", "train_kl_loss",
        "val_loss", "val_mse", "val_rmse", "val_mae", "val_pearson", "val_spearman", "val_ci",
        "test_loss", "test_mse", "test_rmse", "test_mae", "test_pearson", "test_spearman", "test_ci",
    ]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def append_log(out_dir, text):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.log", "a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, device, optimizer, mse_loss_fn):
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "lm_loss": 0.0, "kl_loss": 0.0, "n": 0}
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        prediction, _, lm_loss, kl_loss = model(data)
        mse_loss = mse_loss_fn(prediction, data.y.view(-1, 1).float().to(device))
        loss = kl_loss * 0.001 + mse_loss + lm_loss
        optimizer.ft_backward([loss, mse_loss])
        optimizer.step()
        batch_n = int(data.num_graphs)
        totals["loss"] += float(loss.item()) * batch_n
        totals["mse"] += float(mse_loss.item()) * batch_n
        totals["lm_loss"] += float(lm_loss.item()) * batch_n
        totals["kl_loss"] += float(kl_loss.item()) * batch_n
        totals["n"] += batch_n
    n = max(totals.pop("n"), 1)
    return {key: value / n for key, value in totals.items()}


def evaluate(model, loader, device, mse_loss_fn):
    model.eval()
    y_true, y_pred = [], []
    totals = {"loss": 0.0, "mse": 0.0, "lm_loss": 0.0, "kl_loss": 0.0, "n": 0}
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            prediction, _, lm_loss, kl_loss = model(data)
            target = data.y.view(-1, 1).float().to(device)
            mse_loss = mse_loss_fn(prediction, target)
            loss = kl_loss * 0.001 + mse_loss + lm_loss
            batch_n = int(data.num_graphs)
            totals["loss"] += float(loss.item()) * batch_n
            totals["mse"] += float(mse_loss.item()) * batch_n
            totals["lm_loss"] += float(lm_loss.item()) * batch_n
            totals["kl_loss"] += float(kl_loss.item()) * batch_n
            totals["n"] += batch_n
            y_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
            y_pred.extend(prediction.detach().cpu().numpy().reshape(-1).tolist())
    n = max(totals.pop("n"), 1)
    losses = {key: value / n for key, value in totals.items()}
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(losses)
    return metrics


def load_dataset(cache_dir, split):
    install_deepdtagen_path()
    from utils import TestbedDataset
    return TestbedDataset(root=str(cache_dir), dataset=f"lp_pdbbind_{split}")


def train_model(args):
    install_deepdtagen_path()
    from FetterGrad import FetterGrad
    from model import DeepDTAGen
    from torch_geometric.loader import DataLoader

    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    cache = Path(args.cache_dir)
    with open(cache / "lp_pdbbind_tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    train_set = load_dataset(cache, "train")
    val_set = load_dataset(cache, "val")
    test_set = load_dataset(cache, "test")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DeepDTAGen(tokenizer).to(device)
    optimizer = FetterGrad(optim.Adam(model.parameters(), lr=args.lr))
    mse_loss_fn = nn.MSELoss()
    start_epoch = 1
    best_record = None
    best_val_ci = -float("inf")
    best_val_loss = float("inf")

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_record = checkpoint.get("best_record")
        if best_record:
            best_val_ci = float(best_record.get("val", {}).get("ci", best_val_ci)) if is_finite_number(best_record.get("val", {}).get("ci")) else best_val_ci
            best_val_loss = float(best_record.get("val", {}).get("loss", best_val_loss)) if is_finite_number(best_record.get("val", {}).get("loss")) else best_val_loss

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(vars(args)), handle, indent=2, ensure_ascii=False)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, mse_loss_fn)
        if epoch % args.eval_every != 0:
            write_checkpoints(out_dir, model, optimizer, epoch, best_record, is_best=False, args=vars(args))
            append_log(out_dir, f"epoch {epoch} train_loss {train_metrics['loss']:.6f} train_mse {train_metrics['mse']:.6f}")
            continue

        val_metrics = evaluate(model, val_loader, device, mse_loss_fn)
        test_metrics = evaluate(model, test_loader, device, mse_loss_fn)
        improved, best_val_ci, best_val_loss = is_better_validation(val_metrics, best_val_ci, best_val_loss)
        if improved:
            best_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "test": test_metrics}
        write_checkpoints(out_dir, model, optimizer, epoch, best_record, is_best=improved, args=vars(args))
        with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(finite_json(best_record), handle, indent=2, ensure_ascii=False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_lm_loss": train_metrics["lm_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "val_ci": val_metrics["ci"],
            "test_loss": test_metrics["loss"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_pearson": test_metrics["pearson"],
            "test_spearman": test_metrics["spearman"],
            "test_ci": test_metrics["ci"],
        }
        write_history(out_dir, row)
        append_log(
            out_dir,
            "epoch {epoch} train_loss {train_loss:.6f} train_mse {train_mse:.6f} "
            "valid RMSE {val_rmse:.6f} MAE {val_mae:.6f} Pearson {val_pearson:.6f} Spearman {val_spearman:.6f} CI {val_ci:.6f} "
            "test RMSE {test_rmse:.6f} MAE {test_mae:.6f} Pearson {test_pearson:.6f} Spearman {test_spearman:.6f} CI {test_ci:.6f} "
            "best_val_ci {best_val_ci:.6f}".format(best_val_ci=best_val_ci, **row),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train official DeepDTAGen on LP-PDBBind")
    parser.add_argument("--lp-csv", default=str(PROJECT_ROOT / "data" / "LP_PDBBind.csv"))
    parser.add_argument("--cache-dir", default=str(DEEPDTAGEN_ROOT / "data" / "lp_pdbbind"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "external_benchmarks" / "results" / "deepdtagen_lp_official"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4221)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--long-smiles-policy", choices=["truncate", "drop"], default="truncate")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_prepare:
        prepare_lp_dataset(args)
    if args.prepare_only:
        return
    train_model(args)


if __name__ == "__main__":
    main()
