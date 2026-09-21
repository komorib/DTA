"""Stage A: train and seal the V22 sequence+2D base without reading test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cli import add_config_arguments, config_from_args
from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .training import VERSION, fit_sequence


def run(args):
    output = args.output.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    protein_cache = args.protein_cache.expanduser().resolve()
    ligand_cache = args.ligand_cache.expanduser().resolve()
    initial_checkpoint = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None
        else None
    )
    train = OptionalStructureDataset(
        train_manifest,
        protein_cache,
        ligand_cache,
        with_structure=False,
        limit=args.limit_train,
    )
    valid = OptionalStructureDataset(
        valid_manifest,
        protein_cache,
        ligand_cache,
        with_structure=False,
        limit=args.limit_valid,
    )
    detected_width = int(train[0]["protein_embeddings"].shape[1])
    config = config_from_args(args, detected_width)
    print(
        json.dumps(
            {
                "event": "v22_sequence_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                "with_structure": False,
                "test_manifest_argument_exists": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _, checkpoint, valid_result, history = fit_sequence(
        train,
        valid,
        output,
        config,
        protein_cache,
        ligand_cache,
        initial_checkpoint=initial_checkpoint,
        resume=args.resume,
    )
    checkpoint_path = output / "best.pt"
    prediction_path = output / "valid_predictions.npz"
    selection = {
        "version": VERSION,
        "phase": "sequence_validation_sealed",
        "status": "sealed",
        "output": str(output),
        "best_epoch": int(checkpoint["epoch"]),
        "selected_variant": checkpoint["variant"],
        "selection_rule": checkpoint["selection_rule"],
        "validation_metrics": valid_result["base_metrics"],
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
        "sources": checkpoint["sources"],
        "test_manifest_read": False,
        "test_metrics": None,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    return add_config_arguments(parser)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
