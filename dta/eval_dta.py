import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch

from data_utils.dataset import custom_collate_fn
from dta.model_dta import GerNADTA
from dta.train_dta import (
    evaluate,
    infer_rna_3d_input_dim,
    infer_rna_num_tokens,
    load_dataset_for_path,
    print_metrics,
    resolve_dataset_paths,
    resolve_protein_3d_mode,
    write_json,
)


def parse_eval_splits(value):
    splits = [item.strip() for item in value.split(",") if item.strip()]
    valid_names = {"train", "valid", "test"}
    invalid = [item for item in splits if item not in valid_names]
    if invalid:
        raise ValueError("eval_splits must contain only train, valid, test")
    if not splits:
        raise ValueError("eval_splits cannot be empty")
    return splits


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.cuda}")
    return torch.device("cpu")


def build_eval_model(args, device):
    params = [args.GNN_depth, args.DMA_depth, args.hidden_size1, args.hidden_size2]
    rna_num_tokens = infer_rna_num_tokens(args.dataset, args.rna_num_tokens, args.protein_3d_mode)
    rna_3d_input_dim = infer_rna_3d_input_dim(args.protein_3d_mode)
    model = GerNADTA(
        params,
        trigonometry=True,
        rna_graph=not args.disable_protein_2d,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=rna_num_tokens,
        rna_3d_input_dim=rna_3d_input_dim,
        disable_protein_2d=args.disable_protein_2d,
        dti_front_fusion=args.dti_front_fusion,
        dti_front_input_dim=args.dti_front_input_dim if args.dti_front_input_dim > 0 else None,
        dti_late_fusion=args.dti_late_fusion,
        dti_late_input_dim=args.dti_late_input_dim if args.dti_late_input_dim > 0 else None,
        dti_late_hidden_dim=args.dti_late_hidden_dim if args.dti_late_hidden_dim > 0 else None,
        ligand_unimol_fusion=args.ligand_unimol_fusion,
        ligand_unimol_input_dim=args.ligand_unimol_input_dim if args.ligand_unimol_input_dim > 0 else None,
        ligand_unimol2_3d_fusion=args.ligand_unimol2_3d_fusion,
        ligand_unimol2_input_dim=args.ligand_unimol2_input_dim if args.ligand_unimol2_input_dim > 0 else None,
        ligand_unimol2_3d_fusion_stage=args.ligand_unimol2_3d_fusion_stage,
        pbcnet_late_fusion=args.pbcnet_late_fusion,
        pbcnet_input_dim=args.pbcnet_input_dim if args.pbcnet_input_dim > 0 else None,
        pbcnet_hidden_dim=args.pbcnet_hidden_dim if args.pbcnet_hidden_dim > 0 else None,
        boltz_interface_fusion=args.boltz_interface_fusion,
        boltz_interface_input_dim=args.boltz_interface_input_dim,
        boltz_interface_hidden_dim=args.boltz_interface_hidden_dim,
    )
    state = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_loader(dataset_path, args, device):
    dataset = load_dataset_for_path(dataset_path, args)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def eval_checkpoint(args):
    train_path, valid_path, test_path = resolve_dataset_paths(args)
    args.protein_3d_mode = resolve_protein_3d_mode(args, train_path)
    split_paths = {"train": train_path, "valid": valid_path, "test": test_path}
    eval_splits = parse_eval_splits(args.eval_splits)
    device = resolve_device(args)

    print("checkpoint_path:", args.checkpoint_path)
    print("protein_3d_mode:", args.protein_3d_mode)
    print("protein_embedding_mode:", args.protein_embedding_mode)
    print("ligand_pocket_cutoff:", float(args.ligand_pocket_cutoff))
    print("disable_protein_2d:", bool(args.disable_protein_2d))
    print("dti_front_fusion:", args.dti_front_fusion)
    print("dti_late_fusion:", args.dti_late_fusion)
    print("ligand_unimol_fusion:", args.ligand_unimol_fusion)
    print("ligand_unimol2_3d_fusion:", args.ligand_unimol2_3d_fusion)
    print("ligand_unimol2_3d_fusion_stage:", args.ligand_unimol2_3d_fusion_stage)
    print("pbcnet_late_fusion:", args.pbcnet_late_fusion)
    print("boltz_interface_fusion:", bool(args.boltz_interface_fusion))
    if args.ligand_unimol_lmdb_path:
        print("ligand_unimol_lmdb_path:", args.ligand_unimol_lmdb_path)
    if args.ligand_unimol2_lmdb_path:
        print("ligand_unimol2_lmdb_path:", args.ligand_unimol2_lmdb_path)
    if args.dti_fusion_lmdb_path:
        print("dti_fusion_lmdb_path:", args.dti_fusion_lmdb_path)
    if args.pbcnet_fusion_lmdb_path:
        print("pbcnet_fusion_lmdb_path:", args.pbcnet_fusion_lmdb_path)
    if args.boltz_interface_lmdb_path:
        print("boltz_interface_lmdb_path:", args.boltz_interface_lmdb_path)
    print("device:", device)

    model = build_eval_model(args, device)
    results = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "dataset": args.dataset,
        "split_method": args.split_method,
        "protein_3d_mode": args.protein_3d_mode,
        "protein_embedding_mode": args.protein_embedding_mode,
        "ligand_pocket_cutoff": float(args.ligand_pocket_cutoff),
        "disable_protein_2d": bool(args.disable_protein_2d),
        "dti_fusion_lmdb_path": args.dti_fusion_lmdb_path,
        "dti_front_fusion": args.dti_front_fusion,
        "dti_front_input_dim": int(args.dti_front_input_dim),
        "dti_late_fusion": args.dti_late_fusion,
        "dti_late_input_dim": int(args.dti_late_input_dim),
        "dti_late_hidden_dim": int(args.dti_late_hidden_dim),
        "ligand_unimol_lmdb_path": args.ligand_unimol_lmdb_path,
        "ligand_unimol_fusion": args.ligand_unimol_fusion,
        "ligand_unimol_input_dim": int(args.ligand_unimol_input_dim),
        "ligand_unimol2_lmdb_path": args.ligand_unimol2_lmdb_path,
        "ligand_unimol2_3d_fusion": args.ligand_unimol2_3d_fusion,
        "ligand_unimol2_3d_fusion_stage": args.ligand_unimol2_3d_fusion_stage,
        "ligand_unimol2_input_dim": int(args.ligand_unimol2_input_dim),
        "pbcnet_fusion_lmdb_path": args.pbcnet_fusion_lmdb_path,
        "pbcnet_late_fusion": args.pbcnet_late_fusion,
        "pbcnet_input_dim": int(args.pbcnet_input_dim),
        "pbcnet_hidden_dim": int(args.pbcnet_hidden_dim),
        "boltz_interface_lmdb_path": args.boltz_interface_lmdb_path,
        "boltz_interface_fusion": bool(args.boltz_interface_fusion),
        "boltz_interface_input_dim": int(args.boltz_interface_input_dim),
        "boltz_interface_hidden_dim": int(args.boltz_interface_hidden_dim),
        "boltz_pocket_radius": float(args.boltz_pocket_radius),
        "boltz_min_pocket_residues": int(args.boltz_min_pocket_residues),
        "boltz_max_pocket_residues": int(args.boltz_max_pocket_residues),
        "splits": {},
    }

    for split in eval_splits:
        loader = make_loader(split_paths[split], args, device)
        metrics, labels, preds = evaluate(model, loader, device)
        print_metrics(split, len(preds), metrics)
        results["splits"][split] = {
            "n_samples": int(len(preds)),
            "metrics": {name: float(value) for name, value in metrics.items()},
        }

    if args.output_json:
        write_json(args.output_json, results)
        print("wrote_metrics:", args.output_json)
    return results


def eval_interface_first_checkpoint(args):
    from data_utils.local_interface_dataset import (
        NativeInterfaceDataset,
        collate_single_interface,
    )
    from dta.interface_first_training import (
        _cache_provenance,
        evaluate_interface_first,
        load_interface_first_checkpoint,
        save_keyed_predictions,
    )

    train_path, valid_path, test_path = resolve_dataset_paths(args)
    split_paths = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }
    cache_paths = {
        "train": args.interface_train_native_lmdb,
        "valid": args.interface_valid_native_lmdb,
        "test": args.interface_test_native_lmdb,
    }
    eval_splits = parse_eval_splits(args.eval_splits)
    device = resolve_device(args)
    cache_provenance = {
        split: _cache_provenance(
            cache_paths[split],
            split_paths[split],
            split,
            args.interface_shell_budget,
        )
        for split in eval_splits
    }
    model, checkpoint = load_interface_first_checkpoint(
        args.checkpoint_path, device
    )
    results = {
        "model_arch": "interface_first",
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "checkpoint_valid_metrics": checkpoint.get("valid_metrics"),
        "native_caches": cache_provenance,
        "splits": {},
    }
    prediction_dir = (
        None
        if args.interface_predictions_dir is None
        else os.path.abspath(args.interface_predictions_dir)
    )
    for split in eval_splits:
        dataset = NativeInterfaceDataset(
            split_paths[split], cache_paths[split]
        )
        try:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_single_interface,
                num_workers=args.num_workers,
                pin_memory=False,
            )
            output = evaluate_interface_first(model, loader, device)
        finally:
            dataset.close()
        metrics = output["metrics"]
        print_metrics(split, len(output["predictions"]), metrics)
        results["splits"][split] = {
            "n_samples": int(len(output["predictions"])),
            "metrics": {
                name: float(value) for name, value in metrics.items()
            },
        }
        if prediction_dir:
            prediction_path = os.path.join(
                prediction_dir, f"{split}_predictions.npz"
            )
            save_keyed_predictions(prediction_path, output)
            results["splits"][split][
                "prediction_path"
            ] = prediction_path
    if args.output_json:
        write_json(args.output_json, results)
        print("wrote_metrics:", args.output_json)
    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a GerNA-Bind DTA checkpoint on train/valid/test splits.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_arch", choices=["gerna", "interface_first"], default="gerna")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--eval_splits", type=str, default="valid,test", help="Comma-separated list from: train,valid,test.")
    parser.add_argument("--dataset", type=str, default="PDBbind_atom5_dta")
    parser.add_argument("--split_method", type=str, default="random", help="Subdirectory under data/<dataset>/ used for split manifests, e.g. random, protein, mol, both, fixed_seed42.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--interface_train_native_lmdb", type=str, default=None)
    parser.add_argument("--interface_valid_native_lmdb", type=str, default=None)
    parser.add_argument("--interface_test_native_lmdb", type=str, default=None)
    parser.add_argument("--interface_predictions_dir", type=str, default=None)
    parser.add_argument("--interface_shell_budget", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--GNN_depth", type=int, default=4)
    parser.add_argument("--DMA_depth", type=int, default=2)
    parser.add_argument("--hidden_size1", type=int, default=128)
    parser.add_argument("--hidden_size2", type=int, default=128)
    parser.add_argument("--rna_num_tokens", type=int, default=None)
    parser.add_argument("--protein_3d_mode", choices=["residue", "all_atom", "atom5", "pocket_pharmacophore", "pocket_pharmacophore_esm", "pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"], default=None)
    parser.add_argument("--protein_embedding_mode", choices=["esm", "residue", "auto"], default="esm")
    parser.add_argument("--expected_protein_embedding_dim", type=int, default=640)
    parser.add_argument("--ligand_pocket_cutoff", type=float, default=8.0)
    parser.add_argument("--disable_protein_2d", action="store_true", help="Disable protein 2D graph and protein-side 2D triangle geometry while keeping chain ESM, ligand 2D, and 3D branches.")
    parser.add_argument("--dti_fusion_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key; values are pickled dicts with protein_tokens [L,D] and/or global [D].")
    parser.add_argument("--dti_front_fusion", choices=["none", "add", "concat"], default="none")
    parser.add_argument("--dti_front_input_dim", type=int, default=0)
    parser.add_argument("--dti_late_fusion", choices=["none", "concat"], default="none")
    parser.add_argument("--dti_late_input_dim", type=int, default=0)
    parser.add_argument("--dti_late_hidden_dim", type=int, default=128)
    parser.add_argument("--ligand_unimol_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with GerNA-order ligand Uni-Mol token embeddings.")
    parser.add_argument("--ligand_unimol_fusion", choices=["none", "add", "concat"], default="none")
    parser.add_argument("--ligand_unimol_input_dim", type=int, default=0)
    parser.add_argument("--ligand_unimol2_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with GerNA-order ligand Uni-Mol2 atom token embeddings.")
    parser.add_argument("--ligand_unimol2_3d_fusion", choices=["none", "add", "concat"], default="none")
    parser.add_argument("--ligand_unimol2_3d_fusion_stage", choices=["pre", "post"], default="post")
    parser.add_argument("--ligand_unimol2_input_dim", type=int, default=0)
    parser.add_argument("--pbcnet_fusion_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with PBCNet2.0 pocket-ligand global embeddings.")
    parser.add_argument("--boltz_interface_lmdb_path", type=str, default=None, help="Optional Boltz merged LMDB with frozen pocket residue and ligand atom embeddings.")
    parser.add_argument("--boltz_interface_fusion", action="store_true", help="Fuse frozen Boltz local pair and global interface features before the affinity head.")
    parser.add_argument("--boltz_interface_input_dim", type=int, default=384)
    parser.add_argument("--boltz_interface_hidden_dim", type=int, default=128)
    parser.add_argument("--boltz_pocket_radius", type=float, default=10.0)
    parser.add_argument("--boltz_min_pocket_residues", type=int, default=16)
    parser.add_argument("--boltz_max_pocket_residues", type=int, default=64)
    parser.add_argument("--pbcnet_late_fusion", choices=["none", "concat"], default="none")
    parser.add_argument("--pbcnet_input_dim", type=int, default=0)
    parser.add_argument("--pbcnet_hidden_dim", type=int, default=64)
    parser.add_argument("--protein_edge_mode", choices=["radius", "knn"], default="radius")
    parser.add_argument("--protein_contact_cutoff", type=float, default=6.0)
    parser.add_argument("--protein_knn_k", type=int, default=16)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0. Defaults to cuda:<cuda> when available, else cpu.")
    return parser


def validate_eval_args(parser, args):
    if args.boltz_interface_fusion and not args.boltz_interface_lmdb_path:
        parser.error("--boltz_interface_lmdb_path is required when --boltz_interface_fusion is enabled")
    if args.boltz_interface_input_dim <= 0 or args.boltz_interface_hidden_dim <= 0:
        parser.error("Boltz interface dimensions must be positive")
    if args.boltz_pocket_radius <= 0 or args.boltz_min_pocket_residues <= 0 or args.boltz_max_pocket_residues < args.boltz_min_pocket_residues:
        parser.error("Invalid Boltz interface pocket parameters")
    if args.dti_front_fusion != "none" and args.dti_front_input_dim <= 0:
        parser.error("--dti_front_input_dim must be positive when --dti_front_fusion is enabled")
    if args.dti_late_fusion != "none" and args.dti_late_input_dim <= 0:
        parser.error("--dti_late_input_dim must be positive when --dti_late_fusion is enabled")
    if (args.dti_front_fusion != "none" or args.dti_late_fusion != "none") and not args.dti_fusion_lmdb_path:
        parser.error("--dti_fusion_lmdb_path is required when any DTI fusion mode is enabled")
    if args.ligand_unimol_fusion != "none" and args.ligand_unimol_input_dim <= 0:
        parser.error("--ligand_unimol_input_dim must be positive when --ligand_unimol_fusion is enabled")
    if args.ligand_unimol_fusion != "none" and not args.ligand_unimol_lmdb_path:
        parser.error("--ligand_unimol_lmdb_path is required when --ligand_unimol_fusion is enabled")
    if args.ligand_unimol2_3d_fusion != "none" and args.ligand_unimol2_input_dim <= 0:
        parser.error("--ligand_unimol2_input_dim must be positive when --ligand_unimol2_3d_fusion is enabled")
    if args.ligand_unimol2_3d_fusion != "none" and not args.ligand_unimol2_lmdb_path:
        parser.error("--ligand_unimol2_lmdb_path is required when --ligand_unimol2_3d_fusion is enabled")
    if args.pbcnet_late_fusion != "none" and args.pbcnet_input_dim <= 0:
        parser.error("--pbcnet_input_dim must be positive when --pbcnet_late_fusion is enabled")
    if args.pbcnet_late_fusion != "none" and not args.pbcnet_fusion_lmdb_path:
        parser.error("--pbcnet_fusion_lmdb_path is required when --pbcnet_late_fusion is enabled")
    if args.model_arch == "interface_first":
        requested = parse_eval_splits(args.eval_splits)
        cache_by_split = {
            "train": args.interface_train_native_lmdb,
            "valid": args.interface_valid_native_lmdb,
            "test": args.interface_test_native_lmdb,
        }
        missing = [
            f"--interface_{split}_native_lmdb"
            for split in requested
            if not cache_by_split[split]
        ]
        if missing:
            parser.error(
                "interface_first evaluation requires "
                + ", ".join(missing)
            )
        if args.batch_size != 1:
            parser.error(
                "interface_first evaluation requires --batch_size 1"
            )
        if args.interface_shell_budget < 0:
            parser.error(
                "--interface_shell_budget must be non-negative"
            )
        external_fusion_enabled = any(
            (
                args.dti_fusion_lmdb_path,
                args.dti_front_fusion != "none",
                args.dti_late_fusion != "none",
                args.ligand_unimol_lmdb_path,
                args.ligand_unimol_fusion != "none",
                args.ligand_unimol2_lmdb_path,
                args.ligand_unimol2_3d_fusion != "none",
                args.pbcnet_fusion_lmdb_path,
                args.pbcnet_late_fusion != "none",
                args.boltz_interface_lmdb_path,
                args.boltz_interface_fusion,
            )
        )
        if external_fusion_enabled:
            parser.error(
                "interface_first does not permit external fusion inputs"
            )
        if args.disable_protein_2d:
            parser.error(
                "legacy GerNA branch flags do not apply to interface_first"
            )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_eval_args(parser, args)
    if args.model_arch == "interface_first":
        eval_interface_first_checkpoint(args)
        return
    eval_checkpoint(args)


if __name__ == "__main__":
    main()
