"""Train V21 and seal a checkpoint using train/validation only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .training import TrainingConfig, fit


VERSION = "optional-structure-multitask-v21"


def _config(args, detected_width):
    names = set(TrainingConfig.__dataclass_fields__)
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    values["protein_input_width"] = int(args.protein_input_width or detected_width)
    return TrainingConfig(**values)


def run(args):
    output = args.output.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    protein_cache = args.protein_cache.expanduser().resolve()
    ligand_cache = args.ligand_cache.expanduser().resolve()
    train = OptionalStructureDataset(
        train_manifest, protein_cache, ligand_cache, with_structure=True, limit=args.limit_train
    )
    valid = OptionalStructureDataset(
        valid_manifest, protein_cache, ligand_cache, with_structure=True, limit=args.limit_valid
    )
    detected_width = int(train[0]["protein_embeddings"].shape[1])
    config = _config(args, detected_width)
    print(
        json.dumps(
            {
                "event": "v21_training_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "test_manifest_argument_exists": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _, checkpoint, valid_result, history = fit(
        train,
        valid,
        output,
        config,
        protein_cache,
        ligand_cache,
        resume=args.resume,
    )
    checkpoint_path = output / "best.pt"
    prediction_path = output / "valid_predictions.npz"
    selection = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "status": "sealed",
        "output": str(output),
        "best_epoch": int(checkpoint["epoch"]),
        "selected_variant": checkpoint["variant"],
        "selection_rule": checkpoint["selection_rule"],
        "validation_metrics": valid_result["metrics"],
        "validation_base_metrics": valid_result["base_metrics"],
        "counts": {"train": len(train), "valid": len(valid)},
        "history_length": len(history),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "valid_manifest": str(valid_manifest),
        "valid_manifest_sha256": manifest_fingerprint(valid_manifest),
        "protein_cache": str(protein_cache),
        "ligand_cache": str(ligand_cache),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "valid_predictions": str(prediction_path),
        "valid_predictions_sha256": sha256_file(prediction_path),
        "test_manifest_read": False,
        "test_metrics": None,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)
    return selection


def build_parser():
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--evaluation-batch-size", type=int, default=defaults.evaluation_batch_size)
    parser.add_argument("--gradient-accumulation", type=int, default=defaults.gradient_accumulation)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default=defaults.amp_dtype)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--structure-learning-rate-scale", type=float, default=defaults.structure_learning_rate_scale
    )
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--maximum-epochs", type=int, default=defaults.maximum_epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=defaults.steps_per_epoch)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--warmup-fraction", type=float, default=defaults.warmup_fraction)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--structure-dropout", type=float, default=defaults.structure_dropout)
    parser.add_argument("--lp-sampling", type=float, default=defaults.lp_sampling)
    parser.add_argument("--davis-sampling", type=float, default=defaults.davis_sampling)
    parser.add_argument("--kiba-sampling", type=float, default=defaults.kiba_sampling)
    parser.add_argument("--protein-input-width", type=int)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--heads", type=int, default=defaults.heads)
    parser.add_argument("--protein-layers", type=int, default=defaults.protein_layers)
    parser.add_argument("--ligand-layers", type=int, default=defaults.ligand_layers)
    parser.add_argument("--cross-layers", type=int, default=defaults.cross_layers)
    parser.add_argument("--latent-width", type=int, default=defaults.latent_width)
    parser.add_argument("--pair-width", type=int, default=defaults.pair_width)
    parser.add_argument("--structure-self-layers", type=int, default=defaults.structure_self_layers)
    parser.add_argument("--structure-pair-layers", type=int, default=defaults.structure_pair_layers)
    parser.add_argument("--structure-token-layers", type=int, default=defaults.structure_token_layers)
    parser.add_argument("--structure-top-pairs", type=int, default=defaults.structure_top_pairs)
    parser.add_argument("--maximum-protein-tokens", type=int, default=defaults.maximum_protein_tokens)
    parser.add_argument("--maximum-ligand-atoms", type=int, default=defaults.maximum_ligand_atoms)
    parser.add_argument("--maximum-pocket-tokens", type=int, default=defaults.maximum_pocket_tokens)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--base-affinity-weight", type=float, default=defaults.base_affinity_weight)
    parser.add_argument(
        "--structure-affinity-weight", type=float, default=defaults.structure_affinity_weight
    )
    parser.add_argument("--mse-weight", type=float, default=defaults.mse_weight)
    parser.add_argument("--base-rank-weight", type=float, default=defaults.base_rank_weight)
    parser.add_argument("--structure-rank-weight", type=float, default=defaults.structure_rank_weight)
    parser.add_argument("--contact-weight", type=float, default=defaults.contact_weight)
    parser.add_argument("--distance-weight", type=float, default=defaults.distance_weight)
    parser.add_argument("--distillation-weight", type=float, default=defaults.distillation_weight)
    parser.add_argument("--minimum-rank-gap", type=float, default=defaults.minimum_rank_gap)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
