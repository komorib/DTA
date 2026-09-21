#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.experiment import DTAExperiment


def add_args(parser):
    parser.add_argument("--lmdb_dir", required=True, help="PDBbind LMDB directory.")
    parser.add_argument("--index_path", help="LMDB index.pkl path. Defaults to <lmdb_dir>/index.pkl.")
    parser.add_argument(
        "--structure_root",
        default="data/PDBbind_full/extracted",
        help="Root used to resolve relative ligand_path values from LMDB metadata.",
    )
    parser.add_argument("--split_file", required=True, help="Fixed split TSV with complex_id and split columns.")
    parser.add_argument("--split_method", default="fixed", choices=["fixed", "random"])
    parser.add_argument(
        "--split_missing_policy",
        default="intersection",
        choices=["error", "intersection"],
        help="How to handle LMDB/split sample-set mismatch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Limit index rows for smoke tests.")

    parser.add_argument("--lmdb_key_suffix", default="_protein")
    parser.add_argument("--target_field", default="affinity", help="Index field used as regression target y.")
    parser.add_argument("--esm_key", default="esm2_embedding", help="Chain feature key containing residue ESM embedding.")
    parser.add_argument("--prot_emb_dim", type=int, default=1280, help="Residue ESM embedding dimension.")
    parser.add_argument(
        "--ligand_source",
        default="sdf",
        choices=["sdf", "lmdb_mol"],
        help="Use ligand SDF path from metadata or stored RDKit mol from LMDB.",
    )
    parser.add_argument(
        "--drug_in_dim",
        type=int,
        help="Drug scalar input dimension after ligand scalar-add fusion. Defaults to drug_base_dim.",
    )
    parser.add_argument(
        "--drug_base_dim",
        type=int,
        default=66,
        help="RDKit atom feature dimension before appending ligand embeddings.",
    )
    parser.add_argument(
        "--allow_zero_embedding",
        action="store_true",
        help="Fill missing chain embeddings with zeros instead of raising an error.",
    )
    parser.add_argument(
        "--cache_graphs",
        action="store_true",
        help="Cache converted graphs in memory after first access.",
    )
    parser.add_argument(
        "--use_pocket_atom",
        action="store_true",
        help="Enable an additional ligand-proximal protein atom graph branch.",
    )
    parser.add_argument(
        "--pocket_atom_cutoff",
        type=float,
        default=5.0,
        help="Protein pocket atoms within this ligand distance are included.",
    )
    parser.add_argument(
        "--pocket_atom_edge_cutoff",
        type=float,
        default=4.5,
        help="Distance cutoff for edges inside the pocket atom graph.",
    )
    parser.add_argument(
        "--pocket_atom_include_hydrogen",
        action="store_true",
        help="Include hydrogen atoms in ligand/pocket atom graph construction.",
    )
    parser.add_argument(
        "--pocket_atom_include_hetatm",
        action="store_true",
        help="Include HETATM records from pocket PDB files.",
    )
    parser.add_argument(
        "--use_one_d_two_d",
        action="store_true",
        help="Enable the GerNA-style protein 1D/2D and ligand 2D interaction branch.",
    )
    parser.add_argument(
        "--one_d_two_d_protein_edge_cutoff",
        type=float,
        default=8.0,
        help="C-alpha distance cutoff for the protein 2D graph inside the 1D2D branch.",
    )
    parser.add_argument(
        "--one_d_two_d_max_protein_residues",
        type=int,
        default=128,
        help="Maximum nearest-to-ligand residues kept by the 1D2D branch; <=0 keeps all residues.",
    )
    parser.add_argument(
        "--one_d_two_d_hidden_dim",
        type=int,
        default=64,
        help="Hidden channel size for the 1D2D pair branch.",
    )
    parser.add_argument(
        "--one_d_two_d_output_dim",
        type=int,
        default=128,
        help="Raw embedding size emitted by the 1D2D branch before its FC projection.",
    )
    parser.add_argument(
        "--one_d_two_d_gnn_layers",
        type=int,
        default=3,
        help="Number of ligand/protein GNN layers used by the 1D2D branch.",
    )
    parser.add_argument(
        "--one_d_two_d_triangle_layers",
        type=int,
        default=2,
        help="Number of triangle pair-update layers in the 1D2D branch.",
    )
    parser.add_argument(
        "--one_d_two_d_dropout",
        type=float,
        default=0.25,
        help="Dropout used inside the 1D2D branch.",
    )
    parser.add_argument(
        "--no_one_d_two_d_protein_2d",
        action="store_true",
        help="Use only protein ESM 1D features in the 1D2D branch, disabling its protein CA graph GNN.",
    )
    parser.add_argument(
        "--one_d_two_d_fc_dims",
        type=int,
        nargs="+",
        default=[1024, 128],
        help="FC projection dimensions for the 1D2D branch before top-level concat.",
    )

    parser.add_argument(
        "--use_interface_pairformer",
        action="store_true",
        help=(
            "Enable the pocket-residue x ligand-atom interface Pairformer and "
            "gated local correction head."
        ),
    )
    parser.add_argument("--interface_pairformer_single_dim", type=int, default=128)
    parser.add_argument("--interface_pairformer_pair_dim", type=int, default=64)
    parser.add_argument("--interface_pairformer_output_dim", type=int, default=128)
    parser.add_argument("--interface_pairformer_blocks", type=int, default=4)
    parser.add_argument("--interface_pairformer_heads", type=int, default=4)
    parser.add_argument("--interface_pairformer_num_rbf", type=int, default=32)
    parser.add_argument("--interface_pairformer_fine_num_rbf", type=int, default=32)
    parser.add_argument("--interface_pairformer_max_pocket_residues", type=int, default=64)
    parser.add_argument("--interface_pairformer_max_ligand_atoms", type=int, default=64)
    parser.add_argument("--interface_pairformer_max_residue_atoms", type=int, default=14)
    parser.add_argument("--interface_pairformer_pocket_radius", type=float, default=10.0)
    parser.add_argument("--interface_pairformer_min_pocket_residues", type=int, default=16)
    parser.add_argument("--interface_pairformer_distance_max", type=float, default=20.0)
    parser.add_argument("--interface_pairformer_fine_distance_max", type=float, default=8.0)
    parser.add_argument("--interface_pairformer_dropout", type=float, default=0.15)
    parser.add_argument("--interface_pairformer_gate_hidden_dim", type=int, default=128)

    parser.add_argument("--contact_cutoff", type=float, default=8.0)
    parser.add_argument("--num_pos_emb", type=int, default=16)
    parser.add_argument("--num_rbf", type=int, default=16)

    parser.add_argument("--prot_gcn_dims", type=int, nargs="+", default=[128, 256, 256])
    parser.add_argument("--prot_fc_dims", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--drug_gcn_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--drug_fc_dims", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--pocket_atom_gcn_dims", type=int, nargs="+", default=[128, 32])
    parser.add_argument("--pocket_atom_fc_dims", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--mlp_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--mlp_dropout", type=float, default=0.25)

    parser.add_argument(
        "--use_molecular_dropout",
        action="store_true",
        help="Enable DockTDeep-style molecular dropout during training.",
    )
    parser.add_argument(
        "--molecular_dropout_p",
        type=float,
        default=0.06,
        help="Per-sample probability that molecular dropout fires.",
    )
    parser.add_argument(
        "--molecular_dropout_anchor",
        type=float,
        default=0.0,
        help="Regression target assigned to dropped samples.",
    )
    parser.add_argument(
        "--molecular_dropout_mode",
        default="complex",
        choices=["complex", "protein", "ligand"],
        help="Drop protein/ligand 50/50, or always drop one side.",
    )
    parser.add_argument("--n_ensembles", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--test_freq", type=int)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--monitor_metric", default="mse")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument(
        "--heteroscedastic",
        action="store_true",
        help="Enable heteroscedastic Gaussian NLL head (aleatoric uncertainty).",
    )
    parser.add_argument(
        "--nll_warmup_epochs",
        type=int,
        default=10,
        help="Number of MSE warm-up epochs before switching to Gaussian NLL.",
    )
    parser.add_argument(
        "--nll_logvar_min",
        type=float,
        default=-7.0,
        help="Lower clamp bound on predicted log-variance.",
    )
    parser.add_argument(
        "--nll_logvar_max",
        type=float,
        default=7.0,
        help="Upper clamp bound on predicted log-variance.",
    )

    parser.add_argument("--output_dir", default="../output")
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--save_prediction", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Run KDBNet on PDBbind LMDB data")
    add_args(parser)
    args = parser.parse_args()

    drug_in_dim = args.drug_in_dim
    if drug_in_dim is None:
        drug_in_dim = args.drug_base_dim
    args.drug_in_dim = drug_in_dim

    exp = DTAExperiment(
        task="pdbbind_lmdb",
        split_method=args.split_method,
        contact_cutoff=args.contact_cutoff,
        num_pos_emb=args.num_pos_emb,
        num_rbf=args.num_rbf,
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        drug_in_dim=drug_in_dim,
        drug_gcn_dims=args.drug_gcn_dims,
        drug_fc_dims=args.drug_fc_dims,
        use_pocket_atom=args.use_pocket_atom,
        pocket_atom_cutoff=args.pocket_atom_cutoff,
        pocket_atom_edge_cutoff=args.pocket_atom_edge_cutoff,
        pocket_atom_include_hydrogen=args.pocket_atom_include_hydrogen,
        pocket_atom_include_hetatm=args.pocket_atom_include_hetatm,
        pocket_atom_gcn_dims=args.pocket_atom_gcn_dims,
        pocket_atom_fc_dims=args.pocket_atom_fc_dims,
        use_one_d_two_d=args.use_one_d_two_d,
        one_d_two_d_protein_edge_cutoff=args.one_d_two_d_protein_edge_cutoff,
        one_d_two_d_max_protein_residues=args.one_d_two_d_max_protein_residues,
        one_d_two_d_hidden_dim=args.one_d_two_d_hidden_dim,
        one_d_two_d_output_dim=args.one_d_two_d_output_dim,
        one_d_two_d_gnn_layers=args.one_d_two_d_gnn_layers,
        one_d_two_d_triangle_layers=args.one_d_two_d_triangle_layers,
        one_d_two_d_dropout=args.one_d_two_d_dropout,
        one_d_two_d_use_protein_2d=not args.no_one_d_two_d_protein_2d,
        one_d_two_d_fc_dims=args.one_d_two_d_fc_dims,
        use_interface_pairformer=args.use_interface_pairformer,
        interface_pairformer_single_dim=args.interface_pairformer_single_dim,
        interface_pairformer_pair_dim=args.interface_pairformer_pair_dim,
        interface_pairformer_output_dim=args.interface_pairformer_output_dim,
        interface_pairformer_blocks=args.interface_pairformer_blocks,
        interface_pairformer_heads=args.interface_pairformer_heads,
        interface_pairformer_num_rbf=args.interface_pairformer_num_rbf,
        interface_pairformer_fine_num_rbf=args.interface_pairformer_fine_num_rbf,
        interface_pairformer_max_pocket_residues=args.interface_pairformer_max_pocket_residues,
        interface_pairformer_max_ligand_atoms=args.interface_pairformer_max_ligand_atoms,
        interface_pairformer_max_residue_atoms=args.interface_pairformer_max_residue_atoms,
        interface_pairformer_pocket_radius=args.interface_pairformer_pocket_radius,
        interface_pairformer_min_pocket_residues=args.interface_pairformer_min_pocket_residues,
        interface_pairformer_distance_max=args.interface_pairformer_distance_max,
        interface_pairformer_fine_distance_max=args.interface_pairformer_fine_distance_max,
        interface_pairformer_dropout=args.interface_pairformer_dropout,
        interface_pairformer_gate_hidden_dim=args.interface_pairformer_gate_hidden_dim,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        uncertainty=args.uncertainty,
        parallel=args.parallel,
        output_dir=args.output_dir,
        save_log=args.save_log,
        save_checkpoint=args.save_checkpoint,
        split_file=args.split_file,
        limit=args.limit,
        pdbbind_lmdb_dir=args.lmdb_dir,
        pdbbind_lmdb_index=args.index_path,
        pdbbind_lmdb_structure_root=args.structure_root,
        pdbbind_lmdb_esm_key=args.esm_key,
        pdbbind_lmdb_key_suffix=args.lmdb_key_suffix,
        pdbbind_lmdb_target_field=args.target_field,
        pdbbind_lmdb_ligand_source=args.ligand_source,
        pdbbind_lmdb_allow_zero_embedding=args.allow_zero_embedding,
        pdbbind_lmdb_cache_graphs=args.cache_graphs,
        pdbbind_lmdb_split_missing_policy=args.split_missing_policy,
        prot_emb_dim=args.prot_emb_dim,
        use_molecular_dropout=args.use_molecular_dropout,
        molecular_dropout_p=args.molecular_dropout_p,
        molecular_dropout_anchor=args.molecular_dropout_anchor,
        molecular_dropout_mode=args.molecular_dropout_mode,
        heteroscedastic=args.heteroscedastic,
        nll_warmup_epochs=args.nll_warmup_epochs,
        nll_logvar_min=args.nll_logvar_min,
        nll_logvar_max=args.nll_logvar_max,
    )

    if args.save_prediction or args.save_log or args.save_checkpoint:
        exp.saver.save_config(args.__dict__, "args.yaml")

    exp.train(
        n_epochs=args.n_epochs,
        patience=args.patience,
        eval_freq=args.eval_freq,
        test_freq=args.test_freq,
        monitoring_score=args.monitor_metric,
    )

    if args.recalibrate:
        val_results = exp.test(
            test_loader=exp.task_loader["valid"],
            test_df=exp.task_df["valid"],
            test_tag="valid set",
            print_log=True,
        )
    recalib_df = val_results["df"] if args.recalibrate else None

    exp.test(
        save_prediction=args.save_prediction,
        recalib_df=recalib_df,
        test_tag="Ensemble model",
        print_log=True,
    )


if __name__ == "__main__":
    main()
