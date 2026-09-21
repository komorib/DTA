#!/usr/bin/env python
"""Run the pretrained CASTER-DTA model on LP-PDBBind rows.

This script is intentionally kept outside the upstream CASTER-DTA checkout.
It builds the dataframe expected by CASTER-DTA, reuses its checkpoint metadata,
and writes a compact prediction CSV plus metric JSON for reporting.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run CASTER-DTA on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--caster-root", default="external_benchmarks/caster-dta")
    parser.add_argument("--model-folder", default="external_benchmarks/caster-dta/pretrained_model_downstream")
    parser.add_argument("--output-dir", default="external_benchmarks/results/caster_dta_lp_val")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--structure", default="protein", choices=["protein", "pocket"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu")
    parser.add_argument("--balanced-val", action="store_true", help="For val split, drop remove_for_balancing_val rows")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def build_structure_index(pdbbind_root, suffix):
    root = Path(pdbbind_root)
    index = {}
    for path in root.rglob(f"*_{suffix}.pdb"):
        pdb_id = path.name.split("_", 1)[0].lower()
        index.setdefault(pdb_id, str(path))
    return index


def load_lp_rows(args):
    df = pd.read_csv(args.lp_csv).rename(columns={"Unnamed: 0": "pdb_id"})
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    rows = df[df["new_split"] == args.split].copy()
    if args.balanced_val:
        if args.split != "val":
            raise ValueError("--balanced-val only applies to split=val")
        rows = rows[~rows["remove_for_balancing_val"].astype(bool)].copy()

    suffix = "protein" if args.structure == "protein" else "pocket"
    structure_index = build_structure_index(args.pdbbind_root, suffix)
    rows["protein_file"] = rows["pdb_id"].map(structure_index).fillna("")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("--shard must satisfy 0 <= shard < num_shards")
    if args.num_shards > 1:
        rows = rows.iloc[args.shard :: args.num_shards].copy()

    if args.max_samples is not None:
        rows = rows.head(args.max_samples).copy()

    return rows


def make_caster_dataframe(rows):
    usable = rows[rows["protein_file"] != ""].copy()
    usable = usable.dropna(subset=["seq", "smiles", "value"]).copy()
    caster_df = pd.DataFrame(
        {
            "protein_id": usable["pdb_id"].astype(str),
            "protein_sequence": usable["seq"].astype(str),
            "protein_file": usable["protein_file"].astype(str),
            "molecule_id": usable["pdb_id"].astype(str),
            "molecule_smiles": usable["smiles"].astype(str),
            "affinity_score": usable["value"].astype(float),
        }
    )
    return usable, caster_df


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    permissible = 0.0
    n = len(y_true)
    for i in range(n):
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


def install_serial_caster_loaders(dual_dataset):
    """Replace CASTER multiprocessing graph builders with serial builders.

    The upstream dataset uses multiprocessing.Pool for both protein and molecule
    preprocessing. On some shared servers the worker pool can hang while building
    or finalizing RDKit/PyG objects. Serial preprocessing is slower but reliable.
    """

    import torch_geometric as pyg
    from tqdm import tqdm

    dual_dataset._GERNA_SAFE_LOAD_ERRORS = {"protein": [], "molecule": []}

    def load_proteins_serial(
        protein_df,
        sparse_edges,
        dist_units,
        edge_thresh,
        thresh_type,
        keep_self_loops,
        protein_vector_features,
        protein_include_esm2,
        protein_include_residue_posenc,
        protein_include_aa_props,
    ):
        protein_dict = {}
        iterator = zip(protein_df["protein_id"], protein_df["protein_sequence"], protein_df["protein_file"])
        for protein_id, protein_seq, protein_file in tqdm(iterator, total=protein_df.shape[0], desc="protein_graphs"):
            try:
                protein_g = dual_dataset._protein_file_to_graph(
                    protein_file,
                    dist_units,
                    edge_thresh,
                    thresh_type,
                    keep_self_loops,
                    protein_vector_features,
                    protein_include_esm2,
                    protein_include_residue_posenc,
                    protein_include_aa_props,
                )
            except Exception as exc:
                dual_dataset._GERNA_SAFE_LOAD_ERRORS["protein"].append(
                    {"protein_id": protein_id, "protein_file": protein_file, "error": repr(exc)}
                )
                continue
            protein_g.protein_sequence = protein_seq
            protein_g.protein_id = protein_id
            protein_dict[protein_id] = protein_g

        protein_dict = {k: protein_dict[k] for k in sorted(protein_dict.keys())}
        if sparse_edges:
            to_sparse = pyg.transforms.ToSparseTensor(attr="edge_attr")
            protein_dict = {k: to_sparse(v) for k, v in protein_dict.items()}
        return protein_dict

    def load_molecules_serial(
        molecule_df,
        sparse_edges,
        molecule_full_atomtype,
        molecule_onehot_ordinal_feats,
        molecule_include_selfloops,
    ):
        molecule_dict = {}
        iterator = zip(molecule_df["molecule_id"], molecule_df["molecule_smiles"])
        for molecule_id, molecule_smiles in tqdm(iterator, total=molecule_df.shape[0], desc="molecule_graphs"):
            try:
                molecule_g = dual_dataset._molecule_smiles_to_graph(
                    molecule_smiles,
                    molecule_full_atomtype,
                    molecule_onehot_ordinal_feats,
                    molecule_include_selfloops,
                )
            except Exception as exc:
                dual_dataset._GERNA_SAFE_LOAD_ERRORS["molecule"].append(
                    {"molecule_id": molecule_id, "molecule_smiles": molecule_smiles, "error": repr(exc)}
                )
                continue
            molecule_g.molecule_smiles = molecule_smiles
            molecule_g.molecule_id = molecule_id
            molecule_dict[molecule_id] = molecule_g

        molecule_dict = {k: molecule_dict[k] for k in sorted(molecule_dict.keys())}
        if sparse_edges:
            to_sparse = pyg.transforms.ToSparseTensor(attr="edge_attr")
            molecule_dict = {k: to_sparse(v) for k, v in molecule_dict.items()}
        return molecule_dict

    dual_dataset.ProteinMoleculeDataset._load_proteins = staticmethod(load_proteins_serial)
    dual_dataset.ProteinMoleculeDataset._load_molecules = staticmethod(load_molecules_serial)


def filter_dataset_to_available_pairs(dataset):
    """Drop pairs whose protein or molecule graph failed to build."""

    new_pair_indices = {}
    new_affinity = []
    new_protein_strat = []
    new_molecule_strat = []
    kept_old_indices = []
    dropped_pairs = []

    for old_idx in sorted(dataset.pair_indices.keys()):
        protein_id, molecule_id = dataset.pair_indices[old_idx]
        if protein_id in dataset.protein_data and molecule_id in dataset.molecule_data:
            new_idx = len(new_affinity)
            new_pair_indices[new_idx] = (protein_id, molecule_id)
            new_affinity.append(dataset.affinity_data[old_idx])
            new_protein_strat.append(dataset.idx_protein_strat[old_idx])
            new_molecule_strat.append(dataset.idx_molecule_strat[old_idx])
            kept_old_indices.append(old_idx)
        else:
            dropped_pairs.append({"protein_id": protein_id, "molecule_id": molecule_id})

    if new_affinity:
        dataset.affinity_data = torch.stack(new_affinity).to(dtype=torch.float32)
        dataset.idx_protein_strat = torch.stack(new_protein_strat).to(dtype=torch.long)
        dataset.idx_molecule_strat = torch.stack(new_molecule_strat).to(dtype=torch.long)
    else:
        dataset.affinity_data = torch.empty(0, dtype=torch.float32)
        dataset.idx_protein_strat = torch.empty(0, dtype=torch.long)
        dataset.idx_molecule_strat = torch.empty(0, dtype=torch.long)
    dataset.pair_indices = new_pair_indices
    dataset._gerna_kept_old_indices = kept_old_indices
    dataset._gerna_dropped_pairs = dropped_pairs
    return dropped_pairs


def reindex_dataset_pairs(dataset, keep_old_indices):
    new_pair_indices = {}
    new_affinity = []
    new_protein_strat = []
    new_molecule_strat = []
    for old_idx in keep_old_indices:
        new_idx = len(new_affinity)
        new_pair_indices[new_idx] = dataset.pair_indices[old_idx]
        new_affinity.append(dataset.affinity_data[old_idx])
        new_protein_strat.append(dataset.idx_protein_strat[old_idx])
        new_molecule_strat.append(dataset.idx_molecule_strat[old_idx])

    if new_affinity:
        dataset.affinity_data = torch.stack(new_affinity).to(dtype=torch.float32)
        dataset.idx_protein_strat = torch.stack(new_protein_strat).to(dtype=torch.long)
        dataset.idx_molecule_strat = torch.stack(new_molecule_strat).to(dtype=torch.long)
    else:
        dataset.affinity_data = torch.empty(0, dtype=torch.float32)
        dataset.idx_protein_strat = torch.empty(0, dtype=torch.long)
        dataset.idx_molecule_strat = torch.empty(0, dtype=torch.long)
    dataset.pair_indices = new_pair_indices


def filter_dataset_to_model_type_ranges(dataset, model_folder):
    model_kwargs = json.load(open(Path(model_folder) / "model_kwargs.json", "r"))
    protein_limits = model_kwargs["protein_gnn_kwargs"]
    molecule_limits = model_kwargs["molecule_gnn_kwargs"]
    max_protein_ntype = int(protein_limits["num_ntypes"])
    max_protein_etype = int(protein_limits["num_etypes"])
    max_molecule_ntype = int(molecule_limits["num_ntypes"])
    max_molecule_etype = int(molecule_limits["num_etypes"])

    keep_old_indices = []
    incompatible = []
    for old_idx in sorted(dataset.pair_indices.keys()):
        protein_id, molecule_id = dataset.pair_indices[old_idx]
        protein_g = dataset.protein_data[protein_id]
        molecule_g = dataset.molecule_data[molecule_id]
        reasons = []
        if int(protein_g.node_type.max().item()) >= max_protein_ntype:
            reasons.append(f"protein_ntype_max={int(protein_g.node_type.max().item())}>=limit={max_protein_ntype}")
        if int(protein_g.edge_type.max().item()) >= max_protein_etype:
            reasons.append(f"protein_etype_max={int(protein_g.edge_type.max().item())}>=limit={max_protein_etype}")
        if int(molecule_g.node_type.max().item()) >= max_molecule_ntype:
            reasons.append(f"molecule_ntype_max={int(molecule_g.node_type.max().item())}>=limit={max_molecule_ntype}")
        if int(molecule_g.edge_type.max().item()) >= max_molecule_etype:
            reasons.append(f"molecule_etype_max={int(molecule_g.edge_type.max().item())}>=limit={max_molecule_etype}")

        if reasons:
            incompatible.append({
                "protein_id": protein_id,
                "molecule_id": molecule_id,
                "reason": ";".join(reasons),
            })
        else:
            keep_old_indices.append(old_idx)

    reindex_dataset_pairs(dataset, keep_old_indices)
    dataset._gerna_incompatible_type_pairs = incompatible
    return incompatible


def create_safe_caster_dataset(caster_df, model_folder, cache_dir, dual_dataset, rebuild_cache=False):
    """Build a CASTER dataset and cache only the graph-successful rows."""

    import hashlib

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(caster_df, index=True).values).hexdigest()
    ds_file = cache_dir / f"safe_dataset_{df_hash}.pt"

    if ds_file.exists() and not rebuild_cache:
        dataset = torch.load(ds_file, map_location="cpu")
        graph_errors = getattr(dataset, "_gerna_graph_errors", {"protein": [], "molecule": []})
        dropped_pairs = getattr(dataset, "_gerna_dropped_pairs", [])
        incompatible_type_pairs = getattr(dataset, "_gerna_incompatible_type_pairs", [])
        return dataset, graph_errors, dropped_pairs, incompatible_type_pairs, ds_file

    dataset_kwargs = json.load(open(Path(model_folder) / "dataset_kwargs.json", "r"))
    dataset = dual_dataset.ProteinMoleculeDataset(caster_df, **dataset_kwargs)
    graph_errors = getattr(dual_dataset, "_GERNA_SAFE_LOAD_ERRORS", {"protein": [], "molecule": []})
    dropped_pairs = filter_dataset_to_available_pairs(dataset)
    incompatible_type_pairs = filter_dataset_to_model_type_ranges(dataset, model_folder)

    rescale_params = json.load(open(Path(model_folder) / "dataset_rescale_params.json", "r"))
    dataset._load_scale_data_from_dict(rescale_params)
    dataset._gerna_graph_errors = graph_errors
    dataset._gerna_dropped_pairs = dropped_pairs
    dataset._gerna_incompatible_type_pairs = incompatible_type_pairs
    torch.save(dataset, ds_file)
    return dataset, graph_errors, dropped_pairs, incompatible_type_pairs, ds_file


def flatten_graph_errors(graph_errors):
    rows = []
    for source_name, entries in graph_errors.items():
        for entry in entries:
            row = {"source": source_name}
            row.update(entry)
            rows.append(row)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_lp_rows(args)
    usable, caster_df = make_caster_dataframe(rows)
    missing = rows[rows["protein_file"] == ""].copy()

    caster_df.to_csv(out_dir / "caster_input.csv", index=False)
    missing.to_csv(out_dir / "missing_structure_rows.csv", index=False)

    if caster_df.empty:
        raise ValueError("No usable rows after mapping structures and required fields")

    caster_root = os.path.abspath(args.caster_root)
    sys.path.insert(0, caster_root)

    from dataset import dual_dataset
    from inference.evaluation import run_model_on_dataset
    from inference.inference_utils import load_model_from_checkpoint

    # CASTER defaults to all CPU cores for graph construction; keep it explicit.
    # num_proc=1 uses true serial preprocessing to avoid Pool hangs.
    if args.num_proc <= 1:
        install_serial_caster_loaders(dual_dataset)
    else:
        dual_dataset.N_PROC = args.num_proc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.model_folder, best_model_type="val", device=device)
    dataset, graph_errors, dropped_pairs, incompatible_type_pairs, dataset_cache = create_safe_caster_dataset(
        caster_df, args.model_folder, cache_dir, dual_dataset, rebuild_cache=args.rebuild_cache
    )
    graph_error_rows = flatten_graph_errors(graph_errors)
    pd.DataFrame(graph_error_rows).to_csv(out_dir / "graph_error_rows.csv", index=False)
    pd.DataFrame(dropped_pairs).to_csv(out_dir / "dropped_graph_pairs.csv", index=False)
    pd.DataFrame(incompatible_type_pairs).to_csv(out_dir / "incompatible_type_pairs.csv", index=False)
    if len(dataset) == 0:
        raise ValueError("No graph-usable rows after CASTER preprocessing")

    parsed = run_model_on_dataset(
        model,
        dataset,
        device=device,
        max_batch_size=args.max_batch_size,
        do_gnn_explainer=False,
        output_dir=str(out_dir),
    )

    raw_path = out_dir / "caster_raw_results.pkl"
    parsed.to_pickle(raw_path)

    pred_df = parsed[["protein_id", "molecule_id", "affinity_score"]].copy()
    pred_df = pred_df.rename(columns={"protein_id": "pdb_id", "affinity_score": "y_pred"})
    pred_df["y_pred"] = np.asarray(pred_df["y_pred"].tolist(), dtype=float).reshape(-1)

    meta_cols = [
        "pdb_id",
        "value",
        "new_split",
        "category",
        "type",
        "remove_for_balancing_val",
        "valid_novelty_code_vs_train",
        "test_novelty_code_vs_train",
        "covalent",
    ]
    meta_cols = [c for c in meta_cols if c in usable.columns]
    result = usable[meta_cols].merge(pred_df[["pdb_id", "y_pred"]], on="pdb_id", how="inner")
    result = result.rename(columns={"value": "y_true"})
    result.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "method": "CASTER-DTA pretrained BindingDB",
        "split": args.split,
        "structure": args.structure,
        "input_rows": int(len(rows)),
        "num_shards": int(args.num_shards),
        "shard": int(args.shard),
        "usable_rows": int(len(usable)),
        "missing_structure_rows": int(len(missing)),
        "graph_error_rows": int(len(graph_error_rows)),
        "dropped_graph_pairs": int(len(dropped_pairs)),
        "incompatible_type_pairs": int(len(incompatible_type_pairs)),
        "dataset_cache": str(dataset_cache),
        "predicted_rows": int(len(result)),
        "overall": compute_metrics(result["y_true"], result["y_pred"]),
    }
    novelty_col = "valid_novelty_code_vs_train" if args.split == "val" else "test_novelty_code_vs_train"
    if novelty_col in result.columns:
        metrics["by_novelty"] = {}
        for code, group in result.groupby(novelty_col, dropna=True):
            metrics["by_novelty"][str(int(code))] = compute_metrics(group["y_true"], group["y_pred"])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(finite_json(metrics), handle, indent=2, ensure_ascii=False)

    print(json.dumps(finite_json(metrics), indent=2, ensure_ascii=False))
    print(f"predictions: {out_dir / 'predictions.csv'}")
    print(f"metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
