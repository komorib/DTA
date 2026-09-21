#!/usr/bin/env python
"""Train upstream TankBind from scratch on LP-PDBBind native pockets.

This adapter keeps the upstream TankBind model, feature builders, and loss
formulation. It only replaces the dataset entry with LP_PDBBind rows and uses
the native ligand COM as the pocket center, so no official PDBbind checkpoint
or pretrained weights are loaded.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import RandomSampler

RDLogger.DisableLog("rdApp.*")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train TankBind from scratch on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--tankbind-root", default="external_benchmarks/TankBind")
    parser.add_argument("--deps-path", default="external_benchmarks/TankBind/tankbind_deps")
    parser.add_argument("--cache-dir", default="external_benchmarks/TankBind/lp_pdbbind_native_cache")
    parser.add_argument("--output-dir", default="external_benchmarks/results/tankbind_lp_native_from_scratch")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-n", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--relative-k", type=float, default=0.01)
    parser.add_argument("--relative-k-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warm-up-epochs", type=int, default=15)
    parser.add_argument("--add-noise", type=float, default=5.0)
    parser.add_argument("--pocket-radius", type=float, default=20.0)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--decoy-gap", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def install_tankbind_paths(tankbind_root, deps_path):
    deps_candidates = [resolve_path(deps_path), Path("/tmp/tankbind_deps")]
    tankbind_pkg = resolve_path(tankbind_root) / "tankbind"
    sys.path[:0] = [str(path) for path in deps_candidates if path.exists()] + [str(tankbind_pkg)]

    # torchdrug 0.1.2 references torch._six, which is removed in newer torch.
    if not hasattr(torch, "_six"):
        torch._six = types.SimpleNamespace(string_classes=(str,))

    try:
        import torchdrug.utils as td_utils
    except ModuleNotFoundError as exc:
        searched = ", ".join(str(path) for path in deps_candidates)
        raise ModuleNotFoundError(
            "torchdrug is required for TankBind. "
            f"Searched dependency paths: {searched}. "
            "Install/copy TankBind dependencies into external_benchmarks/TankBind/tankbind_deps "
            "or pass --deps-path to a directory containing torchdrug."
        ) from exc

    def safe_sparse_coo_tensor(indices, values, size):
        return torch.sparse_coo_tensor(indices, values, size)

    td_utils.sparse_coo_tensor = safe_sparse_coo_tensor


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_lp_frames_from_dataframe(frame):
    df = frame.rename(columns={"Unnamed: 0": "pdb_id"}).copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df = df.dropna(subset=["pdb_id", "smiles", "value", "new_split"]).copy()
    df["affinity"] = df["value"].astype(float)
    return {
        split: df[df["new_split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }


def load_lp_frames(csv_path, max_train=None, max_val=None, max_test=None):
    splits = load_lp_frames_from_dataframe(pd.read_csv(csv_path))
    if max_train is not None:
        splits["train"] = splits["train"].head(max_train).copy()
    if max_val is not None:
        splits["val"] = splits["val"].head(max_val).copy()
    if max_test is not None:
        splits["test"] = splits["test"].head(max_test).copy()
    return splits


def make_tankbind_frame(frame, group):
    rows = []
    for _, row in frame.iterrows():
        pdb_id = str(row["pdb_id"]).lower()
        rows.append(
            {
                "protein_name": pdb_id,
                "compound_name": f"{pdb_id}_ligand",
                "pdb": pdb_id,
                "smiles": str(row["smiles"]),
                "affinity": float(row["affinity"] if "affinity" in row else row["value"]),
                "uid": pdb_id,
                "pocket_com": None,
                "use_compound_com": True,
                "use_whole_protein": False,
                "group": group,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "protein_name",
            "compound_name",
            "pdb",
            "smiles",
            "affinity",
            "uid",
            "pocket_com",
            "use_compound_com",
            "use_whole_protein",
            "group",
        ],
    )


def build_file_index(pdbbind_root, pdb_ids):
    root = Path(pdbbind_root)
    p_l_root = root / "P-L"
    search_roots = [p for p in p_l_root.iterdir() if p.is_dir()] if p_l_root.exists() else [root]
    index = {}
    for pdb_id in sorted({str(x).lower() for x in pdb_ids}):
        for search_root in search_roots:
            base = search_root / pdb_id
            if base.exists():
                index[pdb_id] = {
                    "protein_file": base / f"{pdb_id}_protein.pdb",
                    "ligand_sdf": base / f"{pdb_id}_ligand.sdf",
                    "ligand_mol2": base / f"{pdb_id}_ligand.mol2",
                }
                break
    return index


def load_ligand_mol(files):
    for path, reader in (
        (files["ligand_sdf"], Chem.MolFromMolFile),
        (files["ligand_mol2"], Chem.MolFromMol2File),
    ):
        if not path.exists():
            continue
        mol = reader(str(path), sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
        except Exception:
            continue
        if mol.GetNumConformers() > 0:
            return mol
    raise ValueError("native ligand SDF/MOL2 could not be parsed")


def build_feature_cache(tankbind_rows, pdbbind_root, progress_every=100):
    from Bio.PDB import PDBParser
    from feature_utils import extract_torchdrug_feature_from_mol, get_clean_res_list, get_protein_feature

    pdb_ids = tankbind_rows["pdb"].astype(str).str.lower().unique()
    file_index = build_file_index(pdbbind_root, pdb_ids)
    parser = PDBParser(QUIET=True)
    protein_dict = {}
    compound_dict = {}
    kept_rows = []
    errors = []

    for i, row in tankbind_rows.iterrows():
        pdb_id = str(row["pdb"]).lower()
        files = file_index.get(pdb_id)
        if files is None:
            errors.append({"pdb": pdb_id, "error": "PDBbind structure directory not found"})
            continue
        try:
            if pdb_id not in protein_dict:
                if not files["protein_file"].exists():
                    raise FileNotFoundError(str(files["protein_file"]))
                structure = parser.get_structure(pdb_id, str(files["protein_file"]))
                residues = get_clean_res_list(structure.get_residues(), ensure_ca_exist=True)
                protein_dict[pdb_id] = get_protein_feature(residues)

            compound_name = str(row["compound_name"])
            if compound_name not in compound_dict:
                mol = load_ligand_mol(files)
                compound_dict[compound_name] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
            kept_rows.append(row)
        except Exception as exc:
            errors.append({"pdb": pdb_id, "error": repr(exc)})

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "prepare_processed": int(i + 1),
                        "candidate_rows": int(len(tankbind_rows)),
                        "kept": int(len(kept_rows)),
                        "proteins": int(len(protein_dict)),
                        "compounds": int(len(compound_dict)),
                        "errors": int(len(errors)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    kept = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else tankbind_rows.head(0).copy()
    return kept, protein_dict, compound_dict, errors


class TankBindNativeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        protein_dict,
        compound_dict,
        add_noise_to_com=None,
        pocket_radius=20.0,
        contact_cutoff=8.0,
    ):
        self.data = data.reset_index(drop=True)
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        self.add_noise_to_com = add_noise_to_com
        self.pocket_radius = pocket_radius
        self.contact_cutoff = contact_cutoff

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from utils import construct_data_from_graph_gvp

        line = self.data.iloc[idx]
        protein_name = str(line["protein_name"])
        compound_name = str(line["compound_name"])
        (
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
        ) = self.protein_dict[protein_name]
        (
            coords,
            compound_node_features,
            atom_edges,
            atom_edge_attr,
            pair_dis_distribution,
        ) = self.compound_dict[compound_name]

        data, _, _ = construct_data_from_graph_gvp(
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
            coords,
            compound_node_features,
            atom_edges,
            atom_edge_attr,
            includeDisMap=True,
            contactCutoff=self.contact_cutoff,
            pocket_radius=self.pocket_radius,
            add_noise_to_com=self.add_noise_to_com if line["group"] == "train" else None,
            use_whole_protein=False,
            use_compound_com_as_pocket=True,
            chosen_pocket_com=None,
            compoundMode=1,
        )
        data.affinity = torch.tensor([float(line["affinity"])], dtype=torch.float)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        data.pdb = str(line["pdb"])
        data.group = str(line["group"])
        data.real_affinity_mask = torch.tensor([True], dtype=torch.bool)
        data.real_y_mask = torch.ones(data.y.shape, dtype=torch.bool)
        data.is_equivalent_native_pocket = torch.tensor([True], dtype=torch.bool)
        data.equivalent_native_y_mask = torch.ones(data.y.shape, dtype=torch.bool)
        return data


def save_cache(cache_dir, splits, protein_dict, compound_dict, errors):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for split, frame in splits.items():
        frame.to_pickle(cache / f"{split}_data.pkl")
        frame.to_csv(cache / f"{split}_data.csv", index=False)
    torch.save(protein_dict, cache / "protein.pt")
    torch.save(compound_dict, cache / "compound.pt")
    with open(cache / "prepare_errors.json", "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, ensure_ascii=False)
    with open(cache / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "splits": {split: int(len(frame)) for split, frame in splits.items()},
                "proteins": int(len(protein_dict)),
                "compounds": int(len(compound_dict)),
                "errors": int(len(errors)),
                "representation": "TankBind_native_pocket_lp",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def load_cache(cache_dir):
    cache = Path(cache_dir)
    splits = {split: pd.read_pickle(cache / f"{split}_data.pkl") for split in ("train", "val", "test")}
    protein_dict = torch.load(cache / "protein.pt", map_location="cpu")
    compound_dict = torch.load(cache / "compound.pt", map_location="cpu")
    return splits, protein_dict, compound_dict


def prepare_or_load_data(args):
    cache = Path(args.cache_dir)
    if args.rebuild_cache and cache.exists():
        import shutil

        shutil.rmtree(cache)
    required = [cache / f"{split}_data.pkl" for split in ("train", "val", "test")] + [cache / "protein.pt", cache / "compound.pt"]
    if args.skip_prepare or all(path.exists() for path in required):
        return load_cache(cache)

    lp_splits = load_lp_frames(
        args.lp_csv,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        max_test=args.max_test_samples,
    )
    split_frames = {split: make_tankbind_frame(frame, group=split) for split, frame in lp_splits.items()}
    all_rows = pd.concat([split_frames["train"], split_frames["val"], split_frames["test"]], ignore_index=True)
    kept_all, protein_dict, compound_dict, errors = build_feature_cache(
        all_rows,
        args.pdbbind_root,
        progress_every=args.progress_every,
    )
    kept_splits = {
        split: kept_all[kept_all["group"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    save_cache(cache, kept_splits, protein_dict, compound_dict, errors)
    return kept_splits, protein_dict, compound_dict


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    permissible = 0.0
    for i in range(len(y_true)):
        diff_true = y_true[i] - y_true[i + 1 :]
        diff_pred = y_pred[i] - y_pred[i + 1 :]
        mask = diff_true != 0
        if not np.any(mask):
            continue
        prod = diff_true[mask] * diff_pred[mask]
        concordant += float(np.sum(prod > 0))
        concordant += 0.5 * float(np.sum(prod == 0))
        permissible += float(np.sum(mask))
    return concordant / permissible if permissible else float("nan")


def compute_affinity_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    out = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    err = y_pred - y_true
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))
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


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    return float(value)


def metric_float(metrics, key):
    value = metrics.get(key)
    if value is None:
        return float("nan")
    return scalar(value)


def evaluate_split(model, loader, device, criterion, affinity_criterion, relative_k):
    model.eval()
    total_contact_loss = 0.0
    total_contact_n = 0
    total_affinity_loss = 0.0
    total_affinity_n = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred_dis, affinity_pred = model(data)
            contact_loss = criterion(pred_dis, data.dis_map) if len(data.dis_map) else torch.tensor(0.0, device=device)
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)
            total_contact_loss += float(contact_loss.item()) * int(len(pred_dis))
            total_contact_n += int(len(pred_dis))
            total_affinity_loss += float(affinity_loss.item()) * int(len(affinity_pred))
            total_affinity_n += int(len(affinity_pred))
            y_true.extend(data.affinity.detach().cpu().numpy().reshape(-1).tolist())
            y_pred.extend(affinity_pred.detach().cpu().numpy().reshape(-1).tolist())

    metrics = compute_affinity_metrics(y_true, y_pred)
    metrics["loss"] = total_contact_loss / max(total_contact_n, 1) + total_affinity_loss / max(total_affinity_n, 1)
    metrics["y_loss"] = total_contact_loss / max(total_contact_n, 1)
    metrics["affinity_loss"] = total_affinity_loss / max(total_affinity_n, 1)
    return metrics


def relative_k_for_epoch(args, epoch_index):
    if args.relative_k_mode == 0:
        return min(args.relative_k * (2**epoch_index) / (2**args.warm_up_epochs), args.relative_k)
    return min(args.relative_k / max(args.warm_up_epochs, 1) * epoch_index, args.relative_k)


def train_one_epoch(model, loader, optimizer, device, criterion, affinity_criterion, args, epoch_index, log_file):
    from utils import my_affinity_criterion

    model.train()
    total_loss = 0.0
    total_contact_loss = 0.0
    total_affinity_loss = 0.0
    total_affinity_n = 0
    relative_k = relative_k_for_epoch(args, epoch_index)

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        pred_dis, affinity_pred = model(data)
        contact_loss = criterion(pred_dis, data.dis_map) if len(data.dis_map) else torch.tensor(0.0, device=device)
        affinity_loss = relative_k * my_affinity_criterion(
            affinity_pred,
            data.affinity,
            data.is_equivalent_native_pocket,
            decoy_gap=args.decoy_gap,
        )
        loss = contact_loss + affinity_loss
        loss.backward()
        optimizer.step()

        n_aff = int(len(affinity_pred))
        total_loss += float(loss.item()) * n_aff
        total_contact_loss += float(contact_loss.item()) * n_aff
        total_affinity_loss += float(affinity_loss.item()) * n_aff
        total_affinity_n += n_aff
        if args.progress_every > 0 and batch_idx % args.progress_every == 0:
            msg = (
                f"epoch {epoch_index + 1} batch {batch_idx} "
                f"loss {loss.item():.6f} contact_loss {contact_loss.item():.6f} "
                f"affinity_loss {affinity_loss.item():.6f} relative_k {relative_k:.6g}"
            )
            print(msg, flush=True)
            print(msg, file=log_file, flush=True)

    return {
        "loss": total_loss / max(total_affinity_n, 1),
        "contact_loss": total_contact_loss / max(total_affinity_n, 1),
        "affinity_loss": total_affinity_loss / max(total_affinity_n, 1),
        "relative_k": relative_k,
    }


def make_loaders(args, splits, protein_dict, compound_dict):
    from torch_geometric.loader import DataLoader

    train_set = TankBindNativeDataset(
        splits["train"],
        protein_dict,
        compound_dict,
        add_noise_to_com=args.add_noise,
        pocket_radius=args.pocket_radius,
        contact_cutoff=args.contact_cutoff,
    )
    val_set = TankBindNativeDataset(
        splits["val"],
        protein_dict,
        compound_dict,
        add_noise_to_com=None,
        pocket_radius=args.pocket_radius,
        contact_cutoff=args.contact_cutoff,
    )
    test_set = TankBindNativeDataset(
        splits["test"],
        protein_dict,
        compound_dict,
        add_noise_to_com=None,
        pocket_radius=args.pocket_radius,
        contact_cutoff=args.contact_cutoff,
    )
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=args.sample_n)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        follow_batch=["x", "compound_pair"],
        sampler=train_sampler,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        follow_batch=["x", "compound_pair"],
        shuffle=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.valid_batch_size,
        follow_batch=["x", "compound_pair"],
        shuffle=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    install_tankbind_paths(args.tankbind_root, args.deps_path)
    seed_everything(args.seed)
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    splits, protein_dict, compound_dict = prepare_or_load_data(args)
    for split, frame in splits.items():
        frame.to_csv(out_dir / f"{split}_rows.csv", index=False)
    print(
        json.dumps(
            {
                "train": int(len(splits["train"])),
                "val": int(len(splits["val"])),
                "test": int(len(splits["test"])),
                "proteins": int(len(protein_dict)),
                "compounds": int(len(compound_dict)),
                "cache_dir": args.cache_dir,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    import logging
    from model import get_model

    train_loader, val_loader, test_loader = make_loaders(args, splits, protein_dict, compound_dict)
    model = get_model(args.mode, logging, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    affinity_criterion = nn.MSELoss()

    start_epoch = 0
    best_val_ci = -float("inf")
    bad_epochs = 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_val_ci = float(checkpoint.get("best_val_ci", best_val_ci))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        print(f"resumed checkpoint {args.resume_checkpoint} at epoch {start_epoch}", flush=True)

    history_path = out_dir / "history.csv"
    log_path = out_dir / "train.log"
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    checkpoint_path = out_dir / "training_checkpoint.pt"
    best_metrics_path = out_dir / "best_metrics.json"

    history_exists = history_path.exists() and start_epoch > 0
    with open(history_path, "a" if history_exists else "w", newline="", encoding="utf-8") as hist_file, open(log_path, "a", encoding="utf-8") as log_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_contact_loss",
            "train_affinity_loss",
            "relative_k",
            "val_rmse",
            "val_mae",
            "val_pearson",
            "val_spearman",
            "val_ci",
            "test_rmse",
            "test_mae",
            "test_pearson",
            "test_spearman",
            "test_ci",
            "bad_epochs",
        ]
        writer = csv.DictWriter(hist_file, fieldnames=fieldnames)
        if not history_exists:
            writer.writeheader()

        for epoch_index in range(start_epoch, args.epochs):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion,
                affinity_criterion,
                args,
                epoch_index,
                log_file,
            )

            should_eval = ((epoch_index + 1) % args.eval_every == 0) or (epoch_index + 1 == args.epochs)
            if should_eval:
                val_metrics = evaluate_split(model, val_loader, device, criterion, affinity_criterion, args.relative_k)
                test_metrics = evaluate_split(model, test_loader, device, criterion, affinity_criterion, args.relative_k)
                val_ci = metric_float(val_metrics, "ci")
                improved = val_ci > best_val_ci + args.min_delta
                if improved:
                    best_val_ci = val_ci
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                row = {
                    "epoch": epoch_index + 1,
                    "train_loss": train_metrics["loss"],
                    "train_contact_loss": train_metrics["contact_loss"],
                    "train_affinity_loss": train_metrics["affinity_loss"],
                    "relative_k": train_metrics["relative_k"],
                    "val_rmse": metric_float(val_metrics, "rmse"),
                    "val_mae": metric_float(val_metrics, "mae"),
                    "val_pearson": metric_float(val_metrics, "pearson"),
                    "val_spearman": metric_float(val_metrics, "spearman"),
                    "val_ci": val_ci,
                    "test_rmse": metric_float(test_metrics, "rmse"),
                    "test_mae": metric_float(test_metrics, "mae"),
                    "test_pearson": metric_float(test_metrics, "pearson"),
                    "test_spearman": metric_float(test_metrics, "spearman"),
                    "test_ci": metric_float(test_metrics, "ci"),
                    "bad_epochs": bad_epochs,
                }
                writer.writerow(row)
                hist_file.flush()
                msg = (
                    f"epoch {epoch_index + 1} train_loss {train_metrics['loss']:.6f} "
                    f"valid RMSE {row['val_rmse']:.6f} MAE {row['val_mae']:.6f} "
                    f"Pearson {row['val_pearson']:.6f} Spearman {row['val_spearman']:.6f} CI {row['val_ci']:.6f} "
                    f"test RMSE {row['test_rmse']:.6f} MAE {row['test_mae']:.6f} "
                    f"Pearson {row['test_pearson']:.6f} Spearman {row['test_spearman']:.6f} CI {row['test_ci']:.6f} "
                    f"bad_epochs {bad_epochs}/{args.early_stop_patience if args.early_stop_patience > 0 else 'disabled'}"
                )
                print(msg, flush=True)
                print(msg, file=log_file, flush=True)

                record = {
                    "epoch": epoch_index + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                    "bad_epochs": bad_epochs,
                    "args": vars(args),
                }
                if improved:
                    torch.save({"epoch": epoch_index + 1, "model_state_dict": model.state_dict(), "args": vars(args), "metrics": record}, best_path)
                    with open(best_metrics_path, "w", encoding="utf-8") as handle:
                        json.dump(finite_json(record), handle, indent=2, ensure_ascii=False)
                    print(f"best updated epoch {epoch_index + 1} val_ci {best_val_ci:.6f}", flush=True)
                    print(f"best updated epoch {epoch_index + 1} val_ci {best_val_ci:.6f}", file=log_file, flush=True)
            else:
                val_metrics = None
                test_metrics = None

            checkpoint = {
                "epoch": epoch_index + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "best_val_ci": best_val_ci,
                "bad_epochs": bad_epochs,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save({"epoch": epoch_index + 1, "model_state_dict": model.state_dict(), "args": vars(args)}, last_path)

            if args.early_stop_patience > 0 and should_eval and bad_epochs >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch_index + 1} best_val_ci {best_val_ci:.6f}", flush=True)
                break

    print(f"best_model: {best_path}", flush=True)
    print(f"best_metrics: {best_metrics_path}", flush=True)
    print(f"last_model: {last_path}", flush=True)
    print(f"training_checkpoint: {checkpoint_path}", flush=True)
    print(f"history: {history_path}", flush=True)
    print(f"train_log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
