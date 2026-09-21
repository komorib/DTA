#!/usr/bin/env python
"""Train upstream CASTER-DTA from scratch on LP-PDBBind.

The upstream CASTER-DTA model and graph dataset are kept unchanged. This
adapter only maps LP_PDBBind rows into CASTER's protein/molecule dataframe,
uses the predefined LP train/val/test split, and writes checkpoints/metrics in
our external benchmark format.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scripts.run_caster_lp_val import (
        build_structure_index,
        compute_metrics,
        filter_dataset_to_available_pairs,
        finite_json,
        flatten_graph_errors,
        install_serial_caster_loaders,
    )
except ImportError:
    from run_caster_lp_val import (
        build_structure_index,
        compute_metrics,
        filter_dataset_to_available_pairs,
        finite_json,
        flatten_graph_errors,
        install_serial_caster_loaders,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASTER_ROOT = PROJECT_ROOT / "external_benchmarks" / "caster-dta"


def parse_args():
    parser = argparse.ArgumentParser(description="Train CASTER-DTA from scratch on LP-PDBBind")
    parser.add_argument("--lp-csv", default="data/LP_PDBBind.csv")
    parser.add_argument("--pdbbind-root", default="PDBbind")
    parser.add_argument("--caster-root", default="external_benchmarks/caster-dta")
    parser.add_argument("--cache-dir", default="external_benchmarks/caster-dta/data/lp_pdbbind_train_cache")
    parser.add_argument("--output-dir", default="external_benchmarks/results/caster_dta_lp_from_scratch")
    parser.add_argument("--structure", default="protein", choices=["protein", "pocket"])
    parser.add_argument("--batch-max-num", type=int, default=4_000_000)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=1, help="Graph preprocessing workers; 1 uses safe serial builders")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lp_rows(args):
    df = pd.read_csv(resolve_path(args.lp_csv)).rename(columns={"Unnamed: 0": "pdb_id"})
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df = df.dropna(subset=["pdb_id", "seq", "smiles", "value", "new_split"]).copy()

    frames = []
    max_by_split = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    for split in ("train", "val", "test"):
        part = df[df["new_split"] == split].copy()
        if max_by_split[split] is not None:
            part = part.head(max_by_split[split]).copy()
        frames.append(part)
    rows = pd.concat(frames, ignore_index=True)

    suffix = "protein" if args.structure == "protein" else "pocket"
    structure_index = build_structure_index(resolve_path(args.pdbbind_root), suffix)
    rows["protein_file"] = rows["pdb_id"].map(structure_index).fillna("")
    return rows


def make_caster_dataframe(rows):
    usable = rows[rows["protein_file"] != ""].copy()
    usable = usable.dropna(subset=["pdb_id", "seq", "smiles", "value", "new_split"]).copy()
    caster_df = pd.DataFrame(
        {
            "protein_id": usable["pdb_id"].astype(str),
            "protein_sequence": usable["seq"].astype(str),
            "protein_file": usable["protein_file"].astype(str),
            "molecule_id": usable["pdb_id"].astype(str),
            "molecule_smiles": usable["smiles"].astype(str),
            "affinity_score": usable["value"].astype(float),
            "split": usable["new_split"].astype(str),
        }
    )
    return usable, caster_df


def official_dataset_kwargs():
    return {
        "sparse_edges": False,
        "protein_dist_units": "angstroms",
        "protein_edge_thresh": 4,
        "protein_thresh_type": "dist",
        "protein_keep_selfloops": True,
        "protein_vector_features": True,
        "protein_include_esm2": False,
        "protein_include_residue_posenc": False,
        "protein_include_aa_props": True,
        "molecule_full_atomtype": False,
        "molecule_onehot_ordinal_feats": False,
        "molecule_include_selfloops": True,
        "scale_output": ["standardize"],
    }


def build_official_model_kwargs(metadata):
    protein_gnn_kwargs = dict(
        base_conv="lbamodel",
        in_channels=metadata["protein_node_features"],
        edge_dim=metadata["protein_edge_features"],
        num_ntypes=metadata["protein_node_types"],
        num_etypes=metadata["protein_edge_types"],
        ntype_emb_dim=None,
        etype_emb_dim=None,
        num_convs=2,
        hidden_channels=(16, 4),
        edge_hidden_channels=(32, 1),
        out_channels=64,
        dropout_rate=0.2,
        activation="leaky_relu",
        aggr="sum",
    )
    molecule_gnn_kwargs = dict(
        base_conv="gine",
        in_channels=metadata["molecule_node_features"],
        edge_dim=metadata["molecule_edge_features"],
        num_ntypes=metadata["molecule_node_types"],
        num_etypes=metadata["molecule_edge_types"],
        ntype_emb_dim=None,
        etype_emb_dim=None,
        num_convs=2,
        hidden_channels=16,
        out_channels=64,
        dropout_rate=0.2,
        activation="leaky_relu",
        aggr="sum",
        gin_trainable_eps=True,
    )
    joint_gnn_kwargs = dict(
        residue_lin_depth=1,
        atom_lin_depth=1,
        n_attention_heads=8,
        attention_dropout=0.0,
        protein_lin_depth=1,
        molecule_lin_depth=1,
        pairwise_embedding_dim=512,
        out_lin_depth=1,
        out_lin_factor=0.5,
        out_lin_norm_type=None,
        activation="leaky_relu",
        dropout=0.1,
        element_pooling="mean",
        include_residual_stream=True,
        residual_dim_ff_scale=2,
        num_cross_attn_layers=1,
        include_post_pool_layernorm=False,
    )
    return {
        "protein_gnn_kwargs": protein_gnn_kwargs,
        "molecule_gnn_kwargs": molecule_gnn_kwargs,
        "joint_gnn_kwargs": joint_gnn_kwargs,
    }


def install_caster_paths(caster_root):
    caster_root = resolve_path(caster_root)
    sys.path.insert(0, str(caster_root))
    return caster_root


def dataset_cache_path(cache_dir, caster_df, dataset_kwargs):
    import hashlib

    cache_dir = Path(cache_dir)
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(caster_df, index=True).values).hexdigest()
    kwargs_hash = hashlib.sha256(json.dumps(dataset_kwargs, sort_keys=True).encode()).hexdigest()
    return cache_dir / f"safe_dataset_{df_hash}_{kwargs_hash}.pt"


def build_or_load_dataset(caster_df, args, dual_dataset):
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = official_dataset_kwargs()
    ds_file = dataset_cache_path(cache_dir, caster_df, dataset_kwargs)

    if ds_file.exists() and not args.rebuild_cache:
        dataset = torch.load(ds_file, map_location="cpu")
        graph_errors = getattr(dataset, "_gerna_graph_errors", {"protein": [], "molecule": []})
        dropped_pairs = getattr(dataset, "_gerna_dropped_pairs", [])
        return dataset, graph_errors, dropped_pairs, ds_file, dataset_kwargs

    if args.skip_prepare:
        raise FileNotFoundError(f"Cached CASTER dataset not found: {ds_file}")

    if args.num_proc <= 1:
        install_serial_caster_loaders(dual_dataset)
    else:
        dual_dataset.N_PROC = args.num_proc

    dataset = dual_dataset.ProteinMoleculeDataset(caster_df, **dataset_kwargs)
    graph_errors = getattr(dual_dataset, "_GERNA_SAFE_LOAD_ERRORS", {"protein": [], "molecule": []})
    dropped_pairs = filter_dataset_to_available_pairs(dataset)
    dataset._gerna_graph_errors = graph_errors
    dataset._gerna_dropped_pairs = dropped_pairs
    torch.save(dataset, ds_file)
    return dataset, graph_errors, dropped_pairs, ds_file, dataset_kwargs


def split_indices_from_dataset(dataset, caster_df):
    pair_to_split = {
        (str(row.protein_id), str(row.molecule_id)): str(row.split)
        for row in caster_df[["protein_id", "molecule_id", "split"]].itertuples(index=False)
    }
    split_indices = {"train": [], "val": [], "test": []}
    for idx in sorted(dataset.pair_indices.keys()):
        split = pair_to_split.get(tuple(map(str, dataset.pair_indices[idx])))
        if split in split_indices:
            split_indices[split].append(idx)
    return split_indices


def make_subset(dataset, indices):
    return torch.utils.data.Subset(dataset, indices)


def make_loader(subset, args, shuffle, generator=None, worker_init_fn=None):
    from dataset.dual_dataset import PMD_DataLoader

    return PMD_DataLoader(
        subset,
        max_num=args.batch_max_num,
        max_batch_size=args.max_batch_size,
        shuffle=shuffle,
        pin_memory=(args.device == "cuda"),
        num_workers=args.num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )


def metric_float(metrics, key):
    value = metrics.get(key)
    if value is None:
        return float("nan")
    value = float(value)
    return value


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def unscale_tensor(dataset, values):
    values = values.detach()
    return dataset.unscale_target(values).detach().view(-1).cpu().numpy()


def train_one_epoch(model, loader, dataset, device, optimizer, scaler, loss_fn, report_loss_fn, args, epoch, log_handle):
    model.train()
    amp_enabled = args.amp and device.type == "cuda"
    total_loss = 0.0
    total_n = 0
    for batch_idx, (protein_g, molecule_g, target) in enumerate(loader):
        protein_g = protein_g.to(device, non_blocking=True)
        molecule_g = molecule_g.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp_enabled):
            pred, _ = model.forward_with_graphs(protein_g, molecule_g)
            pred = pred.view(-1)
            loss = loss_fn(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred_unscaled = dataset.unscale_target(pred.detach()).view(-1)
            target_unscaled = dataset.unscale_target(target.detach()).view(-1)
            report_loss = report_loss_fn(pred_unscaled, target_unscaled).item()
        batch_n = int(target.numel())
        total_loss += report_loss * batch_n
        total_n += batch_n
        if batch_idx % args.progress_every == 0:
            msg = f"epoch {epoch} batch {batch_idx} loss {report_loss:.6f} lr {current_lr(optimizer):.6g}"
            print(msg, flush=True)
            print(msg, file=log_handle, flush=True)
    return total_loss / max(total_n, 1)


def evaluate_split(model, loader, dataset, device, loss_fn, args):
    model.eval()
    amp_enabled = args.amp and device.type == "cuda"
    preds = []
    targets = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for protein_g, molecule_g, target in loader:
            protein_g = protein_g.to(device)
            molecule_g = molecule_g.to(device)
            target = target.to(device).view(-1)
            with torch.autocast(device.type, enabled=amp_enabled):
                pred, _ = model.forward_with_graphs(protein_g, molecule_g)
            pred = pred.view(-1)
            pred_unscaled = dataset.unscale_target(pred.detach()).view(-1)
            target_unscaled = dataset.unscale_target(target.detach()).view(-1)
            loss = loss_fn(pred_unscaled, target_unscaled).item()
            batch_n = int(target.numel())
            total_loss += loss * batch_n
            total_n += batch_n
            preds.append(pred_unscaled.cpu().numpy())
            targets.append(target_unscaled.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=float)
    y_true = np.concatenate(targets, axis=0) if targets else np.asarray([], dtype=float)
    metrics = compute_metrics(y_true, y_pred)
    metrics["mse"] = total_loss / max(total_n, 1)
    if "rmse" not in metrics and total_n > 0:
        metrics["rmse"] = float(math.sqrt(metrics["mse"]))
    return metrics


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(finite_json(payload), handle, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best_record, best_val_ci):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "best_record": best_record,
        "best_val_ci": best_val_ci,
    }
    torch.save(payload, path)


def load_state_dict_flex(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    if any(key.startswith("_orig_mod.") for key in state_dict):
        stripped = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(stripped)
        return
    prefixed = {"_orig_mod." + key: value for key, value in state_dict.items()}
    model.load_state_dict(prefixed)


def train_model(args, dataset, split_indices, model_kwargs, out_dir):
    from models.joint_gnn import JointGNN

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    train_set = make_subset(dataset, split_indices["train"])
    val_set = make_subset(dataset, split_indices["val"])
    test_set = make_subset(dataset, split_indices["test"])
    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        raise ValueError(f"Empty split after CASTER preprocessing: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = make_loader(train_set, args, shuffle=True, generator=generator, worker_init_fn=seed_worker)
    val_loader = make_loader(val_set, args, shuffle=False)
    test_loader = make_loader(test_set, args, shuffle=False)

    model = JointGNN(
        model_kwargs["protein_gnn_kwargs"],
        model_kwargs["molecule_gnn_kwargs"],
        **model_kwargs["joint_gnn_kwargs"],
    ).to(device)
    if args.compile_model:
        model = torch.compile(model, dynamic=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=50)
    loss_fn = torch.nn.MSELoss()
    report_loss_fn = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    history_path = out_dir / "history.csv"
    log_path = out_dir / "train.log"
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    checkpoint_path = out_dir / "training_checkpoint.pt"
    best_metrics_path = out_dir / "best_metrics.json"

    start_epoch = 1
    best_record = None
    best_val_ci = -float("inf")
    bad_epochs = 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        load_state_dict_flex(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_record = checkpoint.get("best_record")
        best_val_ci = float(checkpoint.get("best_val_ci", best_val_ci))
        bad_epochs = int((best_record or {}).get("bad_epochs", 0))
        print(f"resumed checkpoint {args.resume_checkpoint} at epoch {start_epoch - 1}", flush=True)

    write_header = not history_path.exists() or start_epoch == 1
    with open(history_path, "a", newline="", encoding="utf-8") as hist_file, open(log_path, "a", encoding="utf-8") as log_file:
        fieldnames = [
            "epoch",
            "lr",
            "train_loss",
            "val_mse",
            "val_rmse",
            "val_mae",
            "val_pearson",
            "val_spearman",
            "val_ci",
            "test_mse",
            "test_rmse",
            "test_mae",
            "test_pearson",
            "test_spearman",
            "test_ci",
            "bad_epochs",
        ]
        writer = csv.DictWriter(hist_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            hist_file.flush()

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, dataset, device, optimizer, scaler, loss_fn, report_loss_fn, args, epoch, log_file)
            should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            if should_eval:
                val_metrics = evaluate_split(model, val_loader, dataset, device, report_loss_fn, args)
                test_metrics = evaluate_split(model, test_loader, dataset, device, report_loss_fn, args)
                scheduler.step(metric_float(val_metrics, "mse"))
                val_ci = metric_float(val_metrics, "ci")
                improved = not math.isnan(val_ci) and val_ci > best_val_ci + args.min_delta
                if improved:
                    best_val_ci = val_ci
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val": val_metrics,
                    "test": test_metrics,
                    "bad_epochs": bad_epochs,
                    "selection": "max validation CI",
                    "args": vars(args),
                }
                row = {
                    "epoch": epoch,
                    "lr": current_lr(optimizer),
                    "train_loss": train_loss,
                    "val_mse": metric_float(val_metrics, "mse"),
                    "val_rmse": metric_float(val_metrics, "rmse"),
                    "val_mae": metric_float(val_metrics, "mae"),
                    "val_pearson": metric_float(val_metrics, "pearson"),
                    "val_spearman": metric_float(val_metrics, "spearman"),
                    "val_ci": val_ci,
                    "test_mse": metric_float(test_metrics, "mse"),
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
                    f"epoch {epoch} train_loss {train_loss:.6f} "
                    f"valid RMSE {row['val_rmse']:.6f} MAE {row['val_mae']:.6f} "
                    f"Pearson {row['val_pearson']:.6f} Spearman {row['val_spearman']:.6f} CI {row['val_ci']:.6f} "
                    f"test RMSE {row['test_rmse']:.6f} MAE {row['test_mae']:.6f} "
                    f"Pearson {row['test_pearson']:.6f} Spearman {row['test_spearman']:.6f} CI {row['test_ci']:.6f} "
                    f"bad_epochs {bad_epochs}/disabled"
                )
                print(msg, flush=True)
                print(msg, file=log_file, flush=True)
                if improved:
                    best_record = record
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch, args, best_record, best_val_ci)
                    write_json(best_metrics_path, best_record)
                    best_msg = f"best updated epoch {epoch} val_ci {best_val_ci:.6f}"
                    print(best_msg, flush=True)
                    print(best_msg, file=log_file, flush=True)

            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, args, best_record, best_val_ci)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args)}, last_path)

    print(f"best_model: {best_path}", flush=True)
    print(f"best_metrics: {best_metrics_path}", flush=True)
    print(f"last_model: {last_path}", flush=True)
    print(f"training_checkpoint: {checkpoint_path}", flush=True)
    print(f"history: {history_path}", flush=True)
    print(f"train_log: {log_path}", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_args.json", vars(args))

    caster_root = install_caster_paths(args.caster_root)
    from dataset import dual_dataset

    rows = load_lp_rows(args)
    usable, caster_df = make_caster_dataframe(rows)
    missing = rows[rows["protein_file"] == ""].copy()
    rows.to_csv(out_dir / "lp_rows_input.csv", index=False)
    usable.to_csv(out_dir / "lp_rows_usable_before_graph_filter.csv", index=False)
    caster_df.to_csv(out_dir / "caster_input.csv", index=False)
    missing.to_csv(out_dir / "missing_structure_rows.csv", index=False)
    if caster_df.empty:
        raise ValueError("No usable CASTER rows after structure mapping")

    dataset, graph_errors, dropped_pairs, dataset_cache, dataset_kwargs = build_or_load_dataset(caster_df, args, dual_dataset)
    graph_error_rows = flatten_graph_errors(graph_errors)
    pd.DataFrame(graph_error_rows).to_csv(out_dir / "graph_error_rows.csv", index=False)
    pd.DataFrame(dropped_pairs).to_csv(out_dir / "dropped_graph_pairs.csv", index=False)

    split_indices = split_indices_from_dataset(dataset, caster_df)
    model_kwargs = build_official_model_kwargs(dataset.metadata_dict)
    write_json(out_dir / "dataset_kwargs.json", dataset_kwargs)
    write_json(out_dir / "dataset_rescale_params.json", dataset._report_scale_data())
    write_json(out_dir / "model_kwargs.json", model_kwargs)
    write_json(
        out_dir / "dataset_summary.json",
        {
            "caster_root": str(caster_root),
            "dataset_cache": str(dataset_cache),
            "structure": args.structure,
            "input_rows": int(len(rows)),
            "usable_before_graph_filter": int(len(caster_df)),
            "graph_error_rows": int(len(graph_error_rows)),
            "dropped_graph_pairs": int(len(dropped_pairs)),
            "dataset_pairs_after_filter": int(len(dataset)),
            "split_counts": {key: int(len(value)) for key, value in split_indices.items()},
            "metadata": dataset.metadata_dict,
        },
    )
    print(
        json.dumps(
            finite_json(
                {
                    "dataset_cache": str(dataset_cache),
                    "pairs": int(len(dataset)),
                    "split_counts": {key: int(len(value)) for key, value in split_indices.items()},
                    "graph_errors": int(len(graph_error_rows)),
                    "dropped_pairs": int(len(dropped_pairs)),
                }
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    train_model(args, dataset, split_indices, model_kwargs, out_dir)


if __name__ == "__main__":
    main()
