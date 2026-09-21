#!/usr/bin/env python
"""Run pretrained TankBind affinity prediction on LP-PDBBind rows.

The upstream TankBind prediction entrypoint is notebook-oriented and assumes
CUDA. This adapter keeps the upstream model and feature builders unchanged,
but drives them from LP_PDBBind.csv and writes compact benchmark outputs.
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Run TankBind on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--tankbind-root", default="external_benchmarks/TankBind")
    parser.add_argument("--deps-path", default="/tmp/tankbind_deps")
    parser.add_argument("--checkpoint", default="external_benchmarks/TankBind/saved_models/self_dock.pt")
    parser.add_argument("--output-dir", default="external_benchmarks/results/tankbind_lp_val_selfdock_cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--pocket-radius", type=float, default=20.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--balanced-val", action="store_true")
    parser.add_argument("--use-rdkit-fallback", action="store_true")
    return parser.parse_args()


def install_tankbind_paths(args):
    deps_path = Path(args.deps_path)
    tankbind_pkg = Path(args.tankbind_root) / "tankbind"
    sys.path[:0] = [str(deps_path), str(tankbind_pkg)]

    # torchdrug 0.1.2 still references torch._six, removed in newer torch.
    if not hasattr(torch, "_six"):
        torch._six = types.SimpleNamespace(string_classes=(str,))

    # Avoid torchdrug JIT-compiling its sparse COO extension on shared servers.
    import torchdrug.utils as td_utils

    def safe_sparse_coo_tensor(indices, values, size):
        return torch.sparse_coo_tensor(indices, values, size)

    td_utils.sparse_coo_tensor = safe_sparse_coo_tensor


def build_file_index(pdbbind_root, pdb_ids=None):
    root = Path(pdbbind_root)
    index = {}
    if pdb_ids is not None:
        p_l_root = root / "P-L"
        search_roots = [p for p in p_l_root.iterdir() if p.is_dir()] if p_l_root.exists() else [root]
        for pdb_id in sorted({str(x).lower() for x in pdb_ids}):
            for search_root in search_roots:
                base = search_root / pdb_id
                if not base.exists():
                    continue
                index[pdb_id] = {
                    "protein_file": base / f"{pdb_id}_protein.pdb",
                    "pocket_file": base / f"{pdb_id}_pocket.pdb",
                    "ligand_sdf": base / f"{pdb_id}_ligand.sdf",
                    "ligand_mol2": base / f"{pdb_id}_ligand.mol2",
                }
                break
        return index

    for protein_file in root.rglob("*_protein.pdb"):
        pdb_id = protein_file.name.split("_", 1)[0].lower()
        base = protein_file.parent
        index[pdb_id] = {
            "protein_file": protein_file,
            "pocket_file": base / f"{pdb_id}_pocket.pdb",
            "ligand_sdf": base / f"{pdb_id}_ligand.sdf",
            "ligand_mol2": base / f"{pdb_id}_ligand.mol2",
        }
    return index


def load_lp_rows(args):
    df = pd.read_csv(args.lp_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    rows = df[df["new_split"] == args.split].copy()
    if args.balanced_val:
        if args.split != "val":
            raise ValueError("--balanced-val only applies to split=val")
        rows = rows[~rows["remove_for_balancing_val"].astype(bool)].copy()
    rows = rows.dropna(subset=["pdb_id", "smiles", "value"]).reset_index(drop=True)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("--shard must satisfy 0 <= shard < num_shards")
    if args.num_shards > 1:
        rows = rows.iloc[args.shard :: args.num_shards].reset_index(drop=True)
    if args.max_samples is not None:
        rows = rows.head(args.max_samples).copy()
    return rows


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


def compute_metrics(y_true, y_pred):
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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_ligand_mol(row, files, use_rdkit_fallback):
    from feature_utils import generate_conformation

    ligand_sdf = files["ligand_sdf"]
    ligand_mol2 = files["ligand_mol2"]
    for source, path, reader in [
        ("native_sdf", ligand_sdf, Chem.MolFromMolFile),
        ("native_mol2", ligand_mol2, Chem.MolFromMol2File),
    ]:
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
            return mol, source

    if not use_rdkit_fallback:
        raise ValueError("native ligand SDF/MOL2 could not be parsed")

    mol = Chem.MolFromSmiles(str(row["smiles"]))
    if mol is None:
        raise ValueError("SMILES fallback failed")
    mol = generate_conformation(mol)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError("RDKit conformer fallback failed")
    return mol, "rdkit_fallback"


def build_one_data(row, files, protein_cache, args):
    from Bio.PDB import PDBParser
    from feature_utils import extract_torchdrug_feature_from_mol, get_clean_res_list, get_protein_feature
    from utils import construct_data_from_graph_gvp

    pdb_id = str(row["pdb_id"]).lower()
    if not files["protein_file"].exists():
        raise FileNotFoundError(f"protein file missing: {files['protein_file']}")

    if pdb_id not in protein_cache:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(files["protein_file"]))
        residues = get_clean_res_list(structure.get_residues(), ensure_ca_exist=True)
        protein_cache[pdb_id] = get_protein_feature(residues)

    (
        protein_node_xyz,
        protein_seq,
        protein_node_s,
        protein_node_v,
        protein_edge_index,
        protein_edge_s,
        protein_edge_v,
    ) = protein_cache[pdb_id]

    mol, ligand_source = load_ligand_mol(row, files, args.use_rdkit_fallback)
    coords, compound_node_features, atom_edges, atom_edge_attr, pair_dis_distribution = (
        extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
    )
    pocket_com = np.asarray(coords, dtype=float).mean(axis=0).reshape(1, 3)

    data, _, keep_node = construct_data_from_graph_gvp(
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
        contactCutoff=8.0,
        includeDisMap=True,
        pocket_radius=args.pocket_radius,
        use_whole_protein=False,
        use_compound_com_as_pocket=False,
        chosen_pocket_com=pocket_com,
        compoundMode=1,
    )
    data.affinity = torch.tensor([float(row["value"])], dtype=torch.float)
    data.compound_pair = pair_dis_distribution.reshape(-1, 16)
    data.pdb = pdb_id
    data.real_affinity_mask = torch.tensor([True], dtype=torch.bool)
    data.real_y_mask = torch.ones(data.y.shape).bool()
    data._meta = {
        "pdb_id": pdb_id,
        "ligand_source": ligand_source,
        "protein_nodes_kept": int(np.asarray(keep_node).sum()),
        "compound_atoms": int(coords.shape[0]),
    }
    return data


def write_outputs(output_dir, predictions, errors, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    metrics = {
        "method": "TankBind self_dock pretrained CPU",
        "split": args.split,
        "checkpoint": args.checkpoint,
        "pocket_radius": args.pocket_radius,
        "candidate_rows": int(len(predictions) + len(errors)),
        "predicted_rows": int(len(predictions)),
        "errors": int(len(errors)),
        "overall": compute_metrics(pred_df["y_true"], pred_df["y_pred"]) if len(pred_df) else {"n": 0},
    }
    novelty_col = "valid_novelty_code_vs_train" if args.split == "val" else "test_novelty_code_vs_train"
    if novelty_col in pred_df.columns and len(pred_df):
        metrics["by_novelty"] = {}
        for code, group in pred_df.groupby(novelty_col, dropna=True):
            metrics["by_novelty"][str(int(code))] = compute_metrics(group["y_true"], group["y_pred"])

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(metrics), handle, indent=2, ensure_ascii=False)
    with open(output_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print(json.dumps(finite_json(metrics), indent=2, ensure_ascii=False), flush=True)
    print(f"predictions: {output_dir / 'predictions.csv'}", flush=True)
    print(f"errors: {output_dir / 'errors.csv'}", flush=True)
    print(f"metrics: {output_dir / 'metrics.json'}", flush=True)


def main():
    args = parse_args()
    install_tankbind_paths(args)
    torch.set_num_threads(args.num_threads)

    import logging
    from model import get_model

    out_dir = Path(args.output_dir)
    rows = load_lp_rows(args)
    file_index = build_file_index(args.pdbbind_root, rows["pdb_id"].astype(str).str.lower().unique())

    device = torch.device("cpu")
    model = get_model(0, logging, device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    protein_cache = {}
    data_batch = []
    meta_batch = []
    predictions = []
    errors = []

    def flush_batch():
        if not data_batch:
            return
        loader = DataLoader(
            data_batch,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            follow_batch=["x", "compound_pair"],
        )
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                _, affinity_pred = model(batch.to(device))
                values = affinity_pred.detach().cpu().numpy().reshape(-1)
                for value in values:
                    meta = meta_batch[cursor]
                    meta["y_pred"] = float(value)
                    predictions.append(meta)
                    cursor += 1
        data_batch.clear()
        meta_batch.clear()

    for i, row in rows.iterrows():
        pdb_id = str(row["pdb_id"]).lower()
        files = file_index.get(pdb_id)
        if files is None:
            errors.append({"pdb_id": pdb_id, "error": "PDBbind structure directory not found"})
            continue
        try:
            data = build_one_data(row, files, protein_cache, args)
            meta = {
                "pdb_id": pdb_id,
                "y_true": float(row["value"]),
                "smiles": str(row["smiles"]),
                "valid_novelty_code_vs_train": row.get("valid_novelty_code_vs_train", np.nan),
                "test_novelty_code_vs_train": row.get("test_novelty_code_vs_train", np.nan),
                **data._meta,
            }
            data_batch.append(data)
            meta_batch.append(meta)
            if len(data_batch) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            errors.append({"pdb_id": pdb_id, "error": repr(exc)})

        seen = i + 1
        if args.progress_every > 0 and seen % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": int(seen),
                        "candidate_rows": int(len(rows)),
                        "written": int(len(predictions)),
                        "pending": int(len(data_batch)),
                        "errors": int(len(errors)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    flush_batch()
    write_outputs(out_dir, predictions, errors, args)


if __name__ == "__main__":
    main()
