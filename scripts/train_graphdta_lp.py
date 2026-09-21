#!/usr/bin/env python
"""Train GraphDTA on LP-PDBBind train/val/test splits.

This is a local adapter around the upstream GraphDTA models. It keeps the
original model definitions unchanged while using LP_PDBBind.csv as input.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

GRAPH_DTA_ROOT = Path(__file__).resolve().parents[1] / "external_benchmarks" / "GraphDTA"
sys.path.insert(0, str(GRAPH_DTA_ROOT))

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import ci, mse, pearson, rmse, spearman

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}
MAX_SEQ_LEN = 1000


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
                "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
                "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
                "Cr", "Pt", "Hg", "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def seq_cat(prot):
    x = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    j = 0
    for ch in str(prot).upper():
        if ch == ":":
            continue
        if j >= MAX_SEQ_LEN:
            break
        x[j] = SEQ_DICT.get(ch, 0)
        j += 1
    return x


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(str(smile))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid smiles: {smile}")
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        denom = feature.sum()
        features.append(feature / denom if denom else feature)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    graph = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in graph.edges]
    return c_size, features, edge_index


class LPDataset(InMemoryDataset):
    def __init__(self, root, dataset, frame=None, smile_graph=None):
        self.dataset = dataset
        self.frame = frame
        self.smile_graph = smile_graph
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def process(self):
        if self.frame is None or self.smile_graph is None:
            raise ValueError(f"Processed file missing and no frame provided for {self.dataset}")
        data_list = []
        for _, row in self.frame.iterrows():
            smiles = row["compound_iso_smiles"]
            c_size, features, edge_index = self.smile_graph[smiles]
            edge_tensor = torch.LongTensor(edge_index).transpose(1, 0) if edge_index else torch.empty((2, 0), dtype=torch.long)
            item = Data(
                x=torch.Tensor(features),
                edge_index=edge_tensor,
                y=torch.FloatTensor([float(row["affinity"])]),
            )
            item.target = torch.LongTensor([seq_cat(row["target_sequence"])])
            item.c_size = torch.LongTensor([c_size])
            item.pdb_id = str(row["pdb_id"])
            data_list.append(item)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphDTA on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--model", default="gin", choices=["gin", "gcn", "gat", "gat_gcn"])
    parser.add_argument("--data-root", default="external_benchmarks/GraphDTA/lp_pdbbind_data")
    parser.add_argument("--output-dir", default="external_benchmarks/results/graphdta_lp_gin")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_lp_frames(csv_path, max_train=None, max_val=None):
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "pdb_id"})
    df = df.dropna(subset=["smiles", "seq", "value", "new_split"]).copy()
    df["compound_iso_smiles"] = [canonicalize_smiles(x) for x in df["smiles"]]
    df = df.dropna(subset=["compound_iso_smiles"]).copy()
    df["target_sequence"] = df["seq"].astype(str)
    df["affinity"] = df["value"].astype(float)
    cols = ["pdb_id", "compound_iso_smiles", "target_sequence", "affinity", "new_split"]
    train = df[df["new_split"] == "train"][cols].copy()
    val = df[df["new_split"] == "val"][cols].copy()
    test = df[df["new_split"] == "test"][cols].copy()
    if max_train:
        train = train.head(max_train).copy()
    if max_val:
        val = val.head(max_val).copy()
        test = test.head(max_val).copy()
    return train, val, test


def build_smile_graph(frames):
    smiles = sorted(set().union(*[set(frame["compound_iso_smiles"]) for frame in frames]))
    graph = {}
    errors = []
    for smile in smiles:
        try:
            graph[smile] = smile_to_graph(smile)
        except Exception as exc:
            errors.append({"smiles": smile, "error": repr(exc)})
    return graph, errors


def filter_by_graph(frame, smile_graph):
    return frame[frame["compound_iso_smiles"].isin(smile_graph)].copy()


def make_model(name):
    if name == "gin":
        return GINConvNet()
    if name == "gcn":
        return GCNNet()
    if name == "gat":
        return GATNet()
    if name == "gat_gcn":
        return GAT_GCN()
    raise ValueError(name)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1).detach().cpu()
            preds.append(out)
            labels.append(batch.y.view(-1).detach().cpu())
    y = torch.cat(labels).numpy()
    p = torch.cat(preds).numpy()
    return {
        "n": int(len(y)),
        "rmse": float(rmse(y, p)),
        "mse": float(mse(y, p)),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson": float(pearson(y, p)) if len(y) > 1 else None,
        "spearman": float(spearman(y, p)) if len(y) > 1 else None,
        "ci": float(ci(y, p)) if len(y) > 1 else None,
    }


def finite_json(obj):
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if args.rebuild_cache and data_root.exists():
        import shutil
        shutil.rmtree(data_root / "processed", ignore_errors=True)

    train_df, val_df, test_df = load_lp_frames(args.lp_csv, args.max_train_samples, args.max_val_samples)
    smile_graph, graph_errors = build_smile_graph([train_df, val_df, test_df])
    train_df = filter_by_graph(train_df, smile_graph)
    val_df = filter_by_graph(val_df, smile_graph)
    test_df = filter_by_graph(test_df, smile_graph)

    with open(out_dir / "graph_errors.json", "w", encoding="utf-8") as handle:
        json.dump(graph_errors, handle, indent=2, ensure_ascii=False)
    train_df.to_csv(out_dir / "train_rows.csv", index=False)
    val_df.to_csv(out_dir / "val_rows.csv", index=False)
    test_df.to_csv(out_dir / "test_rows.csv", index=False)

    train_set = LPDataset(str(data_root), "lp_train", train_df, smile_graph)
    val_set = LPDataset(str(data_root), "lp_val", val_df, smile_graph)
    test_set = LPDataset(str(data_root), "lp_test", test_df, smile_graph)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    model = make_model(args.model).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = out_dir / "train.log"
    history_path = out_dir / "history.csv"
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    best_metrics_path = out_dir / "best_metrics.json"

    best_val_rmse = float("inf")
    best_record = None
    bad_epochs = 0

    with open(history_path, "w", newline="", encoding="utf-8") as hist_file, open(log_path, "w", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            hist_file,
            fieldnames=["epoch", "train_loss", "val_rmse", "val_mae", "val_pearson", "val_spearman", "val_ci", "test_rmse", "test_mae", "test_pearson", "test_spearman", "test_ci"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_n = 0
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch).view(-1)
                target = batch.y.view(-1).float().to(device)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(target.numel())
                total_n += int(target.numel())
                if batch_idx % 50 == 0:
                    msg = f"epoch {epoch} batch {batch_idx} loss {loss.item():.6f}"
                    print(msg, flush=True)
                    print(msg, file=log_file, flush=True)

            train_loss = total_loss / max(total_n, 1)
            val_metrics = evaluate(model, val_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_pearson": val_metrics["pearson"],
                "val_spearman": val_metrics["spearman"],
                "val_ci": val_metrics["ci"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_pearson": test_metrics["pearson"],
                "test_spearman": test_metrics["spearman"],
                "test_ci": test_metrics["ci"],
            }
            writer.writerow(row)
            hist_file.flush()
            msg = (
                f"epoch {epoch} train_loss {train_loss:.6f} "
                f"val RMSE {val_metrics['rmse']:.6f} MAE {val_metrics['mae']:.6f} Pearson {val_metrics['pearson']:.6f} Spearman {val_metrics['spearman']:.6f} CI {val_metrics['ci']:.6f} "
                f"test RMSE {test_metrics['rmse']:.6f} MAE {test_metrics['mae']:.6f} Pearson {test_metrics['pearson']:.6f} Spearman {test_metrics['spearman']:.6f} CI {test_metrics['ci']:.6f}"
            )
            print(msg, flush=True)
            print(msg, file=log_file, flush=True)

            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)}, last_path)
            if val_metrics["rmse"] < best_val_rmse - args.min_delta:
                best_val_rmse = val_metrics["rmse"]
                bad_epochs = 0
                best_record = {"epoch": epoch, "val": val_metrics, "test": test_metrics, "train_loss": train_loss, "args": vars(args)}
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args), "metrics": best_record}, best_path)
                with open(best_metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(finite_json(best_record), handle, indent=2, ensure_ascii=False)
                print(f"best updated epoch {epoch} val_rmse {best_val_rmse:.6f}", flush=True)
            else:
                bad_epochs += 1
                print(f"bad_epochs {bad_epochs}/{args.patience} best_val_rmse {best_val_rmse:.6f}", flush=True)
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(f"early stopping at epoch {epoch}", flush=True)
                    break

    print(f"best_model: {best_path}")
    print(f"best_metrics: {best_metrics_path}")
    print(f"history: {history_path}")
    print(f"train_log: {log_path}")


if __name__ == "__main__":
    main()
